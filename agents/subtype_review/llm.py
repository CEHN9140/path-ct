from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from agents.subtype_review.schemas import (
    EvidenceReportBatch,
    RevisionPlan,
    RouterPlan,
)
from agents.subtype_review.tools import TOOL_REGISTRY, build_validation_tools
from utils.llm_utils import (
    LocalLLMClient,
    extract_json_object,
    local_llm_server_available,
    resolve_api_key,
)


class LLMOutputLengthError(RuntimeError):
    pass


class LLMUsageTracker:
    def __init__(self):
        self.api_calls = 0
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None

    def before_request(self) -> None:
        self.api_calls += 1

    def record_response(self, response: Any) -> None:
        if isinstance(response, dict):
            usage = response.get("usage") or response.get("usage_metadata")
            usage = usage or (response.get("response_metadata") or {}).get(
                "token_usage"
            )
        else:
            usage = (
                getattr(response, "usage", None)
                or getattr(response, "usage_metadata", None)
                or (getattr(response, "response_metadata", {}) or {}).get("token_usage")
            )
        if not usage:
            return
        get = (
            usage.get
            if isinstance(usage, dict)
            else lambda key: getattr(usage, key, None)
        )
        for key, names in {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
        }.items():
            value = next((get(name) for name in names if get(name) is not None), None)
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


def prompt_dir(config: Mapping[str, Any], config_dir: str | Path) -> Path:
    path = Path(str(config.get("prompt_dir", "agents/subtype_review/prompts")))
    return path if path.is_absolute() else Path(config_dir).resolve().parent / path


def review_signature_manifest(config: Mapping[str, Any], config_dir: str | Path) -> dict[str, Any]:
    directory = prompt_dir(config, config_dir)
    prompt_hashes = {
        f"{name[:-3]}_prompt_sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("verifier.md", "router.md", "reviser.md")
    }
    review_config = dict(config)
    for key in ("prompt_dir", "repeat", "output_root", "experiment_root"):
        review_config.pop(key, None)
    review_config["llm"] = {
        key: value
        for key, value in dict(review_config.get("llm", {}) or {}).items()
        if key not in {"api_key", "api_key_env"}
    }
    registry = {
        name: {key: value for key, value in metadata.items() if key != "function"}
        for name, metadata in TOOL_REGISTRY.items()
    }
    payload = {"prompts": prompt_hashes, "config": review_config, "tool_registry": registry}
    signature = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**prompt_hashes, "review_signature": signature}


def parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, Mapping):
        return dict(content)
    parsed = extract_json_object(str(content or ""))
    if not isinstance(parsed, dict) or parsed.get("_parse_error"):
        raise RuntimeError("LLM returned invalid JSON object")
    return parsed


def normalize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    message_type = str(getattr(message, "type", "assistant"))
    result = {
        "role": {"human": "user", "ai": "assistant"}.get(message_type, message_type),
        "content": getattr(message, "content", ""),
    }
    if getattr(message, "tool_call_id", None):
        result["tool_call_id"] = str(message.tool_call_id)
    if getattr(message, "tool_calls", None):
        result["tool_calls"] = [dict(call) for call in message.tool_calls]
    return result


def message_history(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        normalize_message(message)
        for message in [
            *(payload.get("message_history", []) or []),
            *(payload.get("tool_messages", []) or []),
        ]
    ]


def audit_tool_messages(messages: list[Any]) -> list[dict[str, Any]]:
    payloads = []
    for message in messages:
        normalized = normalize_message(message)
        if normalized.get("role") != "tool":
            continue
        content = normalized.get("content", {})
        payload = content if isinstance(content, Mapping) else json.loads(str(content))
        if not isinstance(payload, Mapping):
            raise ValueError("ToolMessage content must be a JSON object")
        payloads.append(dict(payload))
    return payloads


class JsonStructuredModel:
    def __init__(
        self,
        llm_config: dict[str, Any],
        schema: type,
        system_prompt: str,
        usage_tracker: LLMUsageTracker | None = None,
    ):
        self.config = dict(llm_config)
        self.schema = schema
        self.prompt = (
            f"{system_prompt.rstrip()}\n\nReturn exactly one valid JSON object."
        )
        self.usage_tracker = usage_tracker

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(
            api_key=resolve_api_key(self.config),
            base_url=str(self.config["base_url"]),
            timeout=float(self.config.get("timeout", 120)),
        )
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        attempts = int(self.config.get("json_retries", 1) or 1) + 1
        last_error = ""
        for _ in range(attempts):
            content = None
            if self.usage_tracker:
                self.usage_tracker.before_request()
            try:
                response = client.chat.completions.create(
                    model=str(self.config["model_name"]),
                    messages=messages,
                    temperature=float(self.config.get("temperature", 0.0)),
                    max_tokens=int(self.config["max_new_tokens"]),
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                if self.usage_tracker:
                    self.usage_tracker.record_response(response)
                if getattr(response.choices[0], "finish_reason", None) == "length":
                    raise LLMOutputLengthError(
                        "LLM output reached max_new_tokens before completing JSON"
                    )
                content = response.choices[0].message.content
                return dict(
                    self.schema.model_validate(parse_json_content(content)).model_dump()
                )
            except LLMOutputLengthError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if content is not None:
                    messages.extend(
                        [
                            {"role": "assistant", "content": str(content)},
                            {
                                "role": "user",
                                "content": f"Return corrected JSON only. Error: {last_error[:2000]}",
                            },
                        ]
                    )
        raise RuntimeError(last_error or "structured LLM call failed")


class LocalStructuredModel:
    def __init__(
        self,
        config: dict[str, Any],
        schema: type,
        system_prompt: str,
        usage_tracker: LLMUsageTracker | None = None,
    ):
        self.client = LocalLLMClient({**config, "api_key": resolve_api_key(config)})
        self.schema = schema
        self.prompt = system_prompt
        self.usage_tracker = usage_tracker

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not local_llm_server_available(self.client.base_url):
            raise RuntimeError(
                f"Local LLM server is unavailable: {self.client.base_url}"
            )
        if self.usage_tracker:
            self.usage_tracker.before_request()
        response = self.client.chat(
            [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        if self.usage_tracker:
            self.usage_tracker.record_response(response)
        return dict(
            self.schema.model_validate(
                parse_json_content(response.get("content"))
            ).model_dump()
        )


class VerifierChatModel:
    def __init__(
        self,
        acquire_model: Any,
        audit_model: Any,
        system_prompt: str,
        tools: list[Any],
        usage_tracker: LLMUsageTracker | None = None,
    ):
        self.acquire_model = acquire_model
        self.audit_model = audit_model
        self.system_prompt = system_prompt
        self.tools = {str(item.name): item for item in tools}
        self.usage_tracker = usage_tracker

    def invoke_model(self, model: Any, messages: list[Any]) -> Any:
        if self.usage_tracker:
            self.usage_tracker.before_request()
        response = model.invoke(messages)
        if self.usage_tracker:
            self.usage_tracker.record_response(response)
        return response

    def invoke(self, payload: dict[str, Any]) -> Any:
        request = dict(payload)
        mode = str(request.get("mode", "audit"))
        if mode == "acquire":
            history = message_history(request)
            request.pop("message_history", None)
            request.pop("tool_messages", None)
            tool_names = [str(name) for name in request.get("eligible_tools", {})]
            model = self.acquire_model.bind_tools(
                [self.tools[name] for name in dict.fromkeys(tool_names)],
                tool_choice="auto",
            )
            return self.invoke_model(
                model,
                [
                    {"role": "system", "content": self.system_prompt},
                    *history,
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
            )
        audit_request = dict(request)
        audit_request.pop("message_history", None)
        audit_request["tool_messages"] = audit_tool_messages(
            list(request.get("tool_messages", []) or [])
        )
        return self.audit_model.invoke(audit_request)


class LocalVerifierModel:
    def __init__(
        self,
        config: dict[str, Any],
        system_prompt: str,
        tools: list[Any],
        usage_tracker: LLMUsageTracker | None = None,
    ):
        self.client = LocalLLMClient({**config, "api_key": resolve_api_key(config)})
        self.system_prompt = system_prompt
        self.tools = {str(item.name): item for item in tools}
        self.retries = int(config.get("json_retries", 1) or 1)
        self.usage_tracker = usage_tracker

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        if self.usage_tracker:
            self.usage_tracker.before_request()
        response = self.client.chat(messages, tools=tools)
        if self.usage_tracker:
            self.usage_tracker.record_response(response)
        return response

    def invoke(self, payload: dict[str, Any]) -> Any:
        if not local_llm_server_available(self.client.base_url):
            raise RuntimeError(
                f"Local LLM server is unavailable: {self.client.base_url}"
            )
        mode = str(payload.get("mode", "audit"))
        request = dict(payload)
        history = message_history(request)
        request.pop("message_history", None)
        request.pop("tool_messages", None)
        tools = None
        if mode == "acquire":
            tools = []
            for name in dict.fromkeys(str(name) for name in request.get("eligible_tools", {})):
                tool = self.tools[name]
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": str(getattr(tool, "description", "") or ""),
                            "parameters": getattr(
                                tool, "args_schema", None
                            ).model_json_schema()
                            if getattr(tool, "args_schema", None)
                            else {"type": "object"},
                        },
                    }
                )
        response = self.chat(
            [
                {"role": "system", "content": self.system_prompt},
                *history,
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            tools=tools,
        )
        if mode == "acquire" or response.get("tool_calls"):
            return response
        try:
            EvidenceReportBatch.model_validate(
                parse_json_content(response.get("content"))
            )
            return response
        except Exception as exc:
            for _ in range(self.retries):
                response = self.chat(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    **request,
                                    "proposed_report": response.get("content"),
                                    "validation_error": f"{type(exc).__name__}: {exc}",
                                    "instruction": "Return corrected EvidenceReportBatch JSON only.",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ]
                )
                try:
                    EvidenceReportBatch.model_validate(
                        parse_json_content(response.get("content"))
                    )
                    return response
                except Exception as next_exc:
                    exc = next_exc
            raise RuntimeError("Verifier report failed schema validation")


def build_structured_model(
    config: dict[str, Any],
    schema: type,
    prompt: str,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    if str(config.get("structured_output", "json_object")) == "json_prompt":
        return LocalStructuredModel(config, schema, prompt, usage_tracker)
    return JsonStructuredModel(config, schema, prompt, usage_tracker)


def build_default_verifier(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = load_prompt(prompt_dir(config, config_dir), "verifier.md")
    tools = build_validation_tools()
    if str(cfg.get("structured_output", "json_object")) == "json_prompt":
        return LocalVerifierModel(cfg, prompt, tools, usage_tracker)
    from langchain_openai import ChatOpenAI

    acquire_model = ChatOpenAI(
        model=str(cfg["model_name"]),
        base_url=str(cfg["base_url"]),
        api_key=resolve_api_key(cfg),
        temperature=float(cfg.get("temperature", 0.0)),
        timeout=float(cfg.get("timeout", 120)),
        extra_body={"thinking": {"type": "disabled"}},
    )
    audit_model = JsonStructuredModel(
        cfg, EvidenceReportBatch, prompt, usage_tracker
    )
    return VerifierChatModel(
        acquire_model,
        audit_model,
        prompt,
        tools,
        usage_tracker,
    )


def build_default_router(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    cfg = dict(config["llm"])
    return build_structured_model(
        cfg,
        RouterPlan,
        load_prompt(prompt_dir(config, config_dir), "router.md"),
        usage_tracker,
    )


def build_default_reviser(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    cfg = dict(config["llm"])
    return build_structured_model(
        cfg,
        RevisionPlan,
        load_prompt(prompt_dir(config, config_dir), "reviser.md"),
        usage_tracker,
    )


def parse_router_plan(value: Any) -> RouterPlan:
    if hasattr(value, "model_dump"):
        return RouterPlan.model_validate(value.model_dump())
    return RouterPlan.model_validate(parse_json_content(value))


def parse_revision_plan(value: Any) -> RevisionPlan:
    if hasattr(value, "model_dump"):
        return RevisionPlan.model_validate(value.model_dump())
    return RevisionPlan.model_validate(parse_json_content(value))
