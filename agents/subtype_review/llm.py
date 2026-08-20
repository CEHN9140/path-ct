from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.subtype_review.schemas import ReviserOutput, RouterAction, RouterLLMOutput, VerifierOutput
from agents.subtype_review.tools import build_validation_tools
from utils.llm_utils import (
    LocalLLMClient,
    extract_json_object,
    local_llm_server_available,
    resolve_api_key,
)


class LLMCallBudgetExceeded(RuntimeError):
    pass


class LLMUsageTracker:
    def __init__(self, max_llm_calls: int | None = None):
        self.max_llm_calls = max_llm_calls
        self.api_calls = 0
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None

    def before_request(self) -> None:
        if self.max_llm_calls is not None and self.api_calls >= self.max_llm_calls:
            raise LLMCallBudgetExceeded(
                f"LLM call budget exhausted: {self.max_llm_calls} calls"
            )
        self.api_calls += 1

    def record_response(self, response: Any) -> None:
        if isinstance(response, dict):
            usage = response.get("usage") or response.get("usage_metadata")
            metadata = response.get("response_metadata") or {}
            usage = usage or metadata.get("token_usage")
        else:
            usage = (
                getattr(response, "usage", None)
                or getattr(response, "usage_metadata", None)
                or (getattr(response, "response_metadata", {}) or {}).get("token_usage")
            )
        if not usage:
            return
        get_value = usage.get if isinstance(usage, dict) else lambda key: getattr(usage, key, None)
        values = {
            "prompt_tokens": get_value("prompt_tokens") or get_value("input_tokens"),
            "completion_tokens": get_value("completion_tokens") or get_value("output_tokens"),
            "total_tokens": get_value("total_tokens"),
        }
        for key, value in values.items():
            if value is not None:
                setattr(self, key, int(getattr(self, key) or 0) + int(value))

    def snapshot(self) -> dict[str, int | None]:
        return {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def load_prompt(prompt_dir: str | Path, name: str) -> str:
    return (Path(prompt_dir) / name).read_text(encoding="utf-8")


def load_prompt_with_protocol(prompt_dir: str | Path, name: str) -> str:
    directory = Path(prompt_dir)
    prompt = load_prompt(directory, name)
    protocol = directory / "protocol.md"
    return f"{prompt}\n\n{protocol.read_text(encoding='utf-8')}" if protocol.exists() else prompt


def prompt_dir(config: dict[str, Any], config_dir: str | Path) -> Path:
    path = Path(config.get("prompt_dir", "agents/subtype_review/prompts"))
    return path if path.is_absolute() else Path(config_dir).resolve().parent / path


def parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return dict(content)
    parsed = extract_json_object(str(content or ""))
    if not isinstance(parsed, dict) or parsed.get("_parse_error"):
        raise RuntimeError("LLM returned invalid JSON object")
    return parsed


def validate_verifier_payload(payload: dict[str, Any]) -> None:
    VerifierOutput.model_validate(payload)


class JsonStructuredModel:
    def __init__(self, llm_config: dict[str, Any], schema: type, system_prompt: str, usage_tracker: LLMUsageTracker | None = None):
        self.config = dict(llm_config)
        self.schema = schema
        self.usage_tracker = usage_tracker
        self.system_prompt = f"{system_prompt.rstrip()}\n\nReturn exactly one valid JSON object."

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(
            api_key=resolve_api_key(self.config),
            base_url=str(self.config["base_url"]),
            timeout=float(self.config.get("timeout", 120)),
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        attempts = int(self.config.get("json_retries", 2) or 2) + 1
        last_error = ""
        for _ in range(attempts):
            content = None
            if self.usage_tracker is not None:
                self.usage_tracker.before_request()
            try:
                response = client.chat.completions.create(
                    model=str(self.config["model_name"]),
                    messages=messages,
                    temperature=float(self.config.get("temperature", 0.1)),
                    max_tokens=int(self.config.get("max_new_tokens", 2048)),
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                if self.usage_tracker is not None:
                    self.usage_tracker.record_response(response)
                content = response.choices[0].message.content
                parsed = parse_json_content(content)
                return dict(self.schema.model_validate(parsed).model_dump())
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if content is not None:
                    messages.extend([
                        {"role": "assistant", "content": str(content)},
                        {
                            "role": "user",
                            "content": (
                                "The previous JSON failed schema validation: "
                                f"{last_error[:2000]}. Return one corrected JSON object "
                                "using only the requested schema values."
                            ),
                        },
                    ])
        raise RuntimeError(last_error or "structured LLM call failed")


class LocalStructuredModel:
    def __init__(self, config: dict[str, Any], schema: type, system_prompt: str, usage_tracker: LLMUsageTracker | None = None):
        client_config = {**config, "api_key": resolve_api_key(config)}
        self.client = LocalLLMClient(client_config)
        self.schema = schema
        self.system_prompt = system_prompt
        self.usage_tracker = usage_tracker

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not local_llm_server_available(self.client.base_url):
            raise RuntimeError(f"Local LLM server is unavailable: {self.client.base_url}")
        if self.usage_tracker is not None:
            self.usage_tracker.before_request()
        response = self.client.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        if self.usage_tracker is not None:
            self.usage_tracker.record_response(response)
        return dict(self.schema.model_validate(parse_json_content(response.get("content"))).model_dump())


class ProtocolSelfReviewModel:
    def __init__(self, draft_model: Any, review_model: Any):
        self.draft_model = draft_model
        self.review_model = review_model

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        proposal = self.draft_model.invoke(payload)
        return self.review_model.invoke({
            **payload,
            "mode": "protocol_self_review",
            "proposed_action": proposal,
        })


class VerifierChatModel:
    def __init__(self, model: Any, system_prompt: str, tools: list[Any], correction_attempts: int = 2, usage_tracker: LLMUsageTracker | None = None):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {str(item.name): item for item in tools}
        self.correction_attempts = correction_attempts
        self.usage_tracker = usage_tracker

    def invoke_model(self, model: Any, messages: Any) -> Any:
        if self.usage_tracker is not None:
            self.usage_tracker.before_request()
        response = model.invoke(messages)
        if self.usage_tracker is not None:
            self.usage_tracker.record_response(response)
        return response

    def invoke(self, payload: dict[str, Any]) -> Any:
        request = dict(payload)
        mode = str(payload.get("mode", "audit"))
        if mode == "acquire":
            dimension = str(dict(payload.get("request", {}) or {}).get("dimension", ""))
            tool = self.tools.get(dimension)
            if tool is None:
                raise ValueError(f"No validation tool is configured for {dimension}")
            model = self.model.bind_tools([tool], tool_choice="required")
        else:
            model = self.model.bind(response_format={"type": "json_object"})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        response = self.invoke_model(model, messages)
        if mode == "acquire":
            return response
        review_request = {
            **request,
            "mode": "protocol_self_review",
            "proposed_audit": parse_json_content(getattr(response, "content", response)),
        }
        reviewed = self.invoke_model(model, [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(review_request, ensure_ascii=False)},
        ])
        for _ in range(self.correction_attempts + 1):
            try:
                validate_verifier_payload(parse_json_content(getattr(reviewed, "content", reviewed)))
                return reviewed
            except Exception as exc:
                reviewed = self.invoke_model(model, [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps({
                        **request,
                        "mode": "audit_correction",
                        "proposed_audit": parse_json_content(getattr(reviewed, "content", reviewed)),
                        "validation_error": f"{type(exc).__name__}: {exc}",
                        "instruction": "Return the corrected complete VerifierOutput JSON.",
                    }, ensure_ascii=False)},
                ])
        raise RuntimeError("Verifier audit failed schema correction")


class LocalVerifierModel:
    def __init__(self, config: dict[str, Any], system_prompt: str, tools: list[Any], usage_tracker: LLMUsageTracker | None = None):
        client_config = {**config, "api_key": resolve_api_key(config)}
        self.client = LocalLLMClient(client_config)
        self.system_prompt = system_prompt
        self.tools = {str(item.name): item for item in tools}
        self.correction_attempts = int(config.get("json_retries", 2) or 2)
        self.usage_tracker = usage_tracker

    def chat(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.usage_tracker is not None:
            self.usage_tracker.before_request()
        response = self.client.chat(messages, tools=tools)
        if self.usage_tracker is not None:
            self.usage_tracker.record_response(response)
        return response

    def invoke(self, payload: dict[str, Any]) -> Any:
        if not local_llm_server_available(self.client.base_url):
            raise RuntimeError(f"Local LLM server is unavailable: {self.client.base_url}")
        mode = str(payload.get("mode", "audit"))
        request_tools = []
        if mode == "acquire":
            dimension = str(dict(payload.get("request", {}) or {}).get("dimension", ""))
            tool = self.tools.get(dimension)
            if tool is None:
                raise ValueError(f"No validation tool is configured for {dimension}")
            schema = tool.args_schema.model_json_schema() if getattr(tool, "args_schema", None) else {"type": "object"}
            request_tools = [{
                "type": "function",
                "function": {
                    "name": dimension,
                    "description": str(getattr(tool, "description", "") or ""),
                    "parameters": schema,
                },
            }]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = self.chat(
            messages,
            tools=request_tools or None,
        )
        if response.get("tool_calls"):
            return response
        audit = parse_json_content(response.get("content"))
        if mode == "acquire":
            return audit
        review_request = {
            **payload,
            "mode": "protocol_self_review",
            "proposed_audit": audit,
        }
        reviewed = self.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(review_request, ensure_ascii=False)},
        ])
        for _ in range(self.correction_attempts + 1):
            try:
                audit = parse_json_content(reviewed.get("content"))
                validate_verifier_payload(audit)
                return audit
            except Exception as exc:
                reviewed = self.chat([
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps({
                        **payload,
                        "mode": "audit_correction",
                        "proposed_audit": parse_json_content(reviewed.get("content")),
                        "validation_error": f"{type(exc).__name__}: {exc}",
                        "instruction": "Return the corrected complete VerifierOutput JSON.",
                    }, ensure_ascii=False)},
                ])
        raise RuntimeError("Verifier audit failed schema correction")


def build_structured_model(config: dict[str, Any], schema: type, prompt: str, usage_tracker: LLMUsageTracker | None = None) -> Any:
    if str(config.get("structured_output", "json_object")) == "json_prompt":
        return LocalStructuredModel(config, schema, prompt, usage_tracker)
    return JsonStructuredModel(config, schema, prompt, usage_tracker)


def build_default_verifier(config: dict[str, Any], config_dir: str | Path, *, usage_tracker: LLMUsageTracker | None = None) -> Any:
    cfg = dict(config["llm"])
    prompt = load_prompt_with_protocol(prompt_dir(config, config_dir), "verifier.md")
    tools = build_validation_tools()
    if str(cfg.get("structured_output", "json_object")) == "json_prompt":
        return LocalVerifierModel(cfg, prompt, tools, usage_tracker)
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=str(cfg["model_name"]),
        base_url=str(cfg["base_url"]),
        api_key=resolve_api_key(cfg),
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(cfg.get("max_new_tokens", 2048)),
        extra_body={"thinking": {"type": "disabled"}},
    )
    return VerifierChatModel(model, prompt, tools, int(cfg.get("json_retries", 2) or 2), usage_tracker)


def build_default_reviser(config: dict[str, Any], config_dir: str | Path, *, usage_tracker: LLMUsageTracker | None = None) -> Any:
    cfg = dict(config["llm"])
    return build_structured_model(
        cfg,
        ReviserOutput,
        load_prompt_with_protocol(prompt_dir(config, config_dir), "reviser.md"),
        usage_tracker,
    )


def build_default_router(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    self_review: bool = True,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    cfg = dict(config["llm"])
    model = build_structured_model(
        cfg,
        RouterLLMOutput,
        load_prompt_with_protocol(prompt_dir(config, config_dir), "router.md"),
        usage_tracker,
    )
    return ProtocolSelfReviewModel(model, model) if self_review else model


def parse_router_action(value: Any) -> RouterAction:
    if hasattr(value, "model_dump"):
        return RouterAction.model_validate(value.model_dump())
    return RouterAction.model_validate(parse_json_content(value))
