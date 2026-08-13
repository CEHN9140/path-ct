from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.subtype_review.schemas import ReviserOutput, RouterAction
from agents.subtype_review.tools import build_validation_tools
from utils.llm_utils import (
    LocalLLMClient,
    extract_json_object,
    local_llm_server_available,
    resolve_api_key,
)


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


class JsonStructuredModel:
    def __init__(self, llm_config: dict[str, Any], schema: type, system_prompt: str):
        self.config = dict(llm_config)
        self.schema = schema
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
            try:
                response = client.chat.completions.create(
                    model=str(self.config["model_name"]),
                    messages=messages,
                    temperature=float(self.config.get("temperature", 0.1)),
                    max_tokens=int(self.config.get("max_new_tokens", 2048)),
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
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
    def __init__(self, config: dict[str, Any], schema: type, system_prompt: str):
        client_config = {**config, "api_key": resolve_api_key(config)}
        self.client = LocalLLMClient(client_config)
        self.schema = schema
        self.system_prompt = system_prompt

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not local_llm_server_available(self.client.base_url):
            raise RuntimeError(f"Local LLM server is unavailable: {self.client.base_url}")
        response = self.client.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
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
    def __init__(self, model: Any, system_prompt: str, tools: list[Any]):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {str(item.name): item for item in tools}

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
        response = model.invoke(messages)
        if mode == "acquire":
            return response
        review_request = {
            **request,
            "mode": "protocol_self_review",
            "proposed_audit": parse_json_content(getattr(response, "content", response)),
        }
        return model.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(review_request, ensure_ascii=False)},
        ])


class LocalVerifierModel:
    def __init__(self, config: dict[str, Any], system_prompt: str, tools: list[Any]):
        client_config = {**config, "api_key": resolve_api_key(config)}
        self.client = LocalLLMClient(client_config)
        self.system_prompt = system_prompt
        self.tools = {str(item.name): item for item in tools}

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
        response = self.client.chat(
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
        reviewed = self.client.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(review_request, ensure_ascii=False)},
        ])
        return parse_json_content(reviewed.get("content"))


def build_structured_model(config: dict[str, Any], schema: type, prompt: str) -> Any:
    if str(config.get("structured_output", "json_object")) == "json_prompt":
        return LocalStructuredModel(config, schema, prompt)
    return JsonStructuredModel(config, schema, prompt)


def build_default_verifier(config: dict[str, Any], config_dir: str | Path) -> Any:
    cfg = dict(config["llm"])
    prompt = load_prompt_with_protocol(prompt_dir(config, config_dir), "verifier.md")
    tools = build_validation_tools()
    if str(cfg.get("structured_output", "json_object")) == "json_prompt":
        return LocalVerifierModel(cfg, prompt, tools)
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=str(cfg["model_name"]),
        base_url=str(cfg["base_url"]),
        api_key=resolve_api_key(cfg),
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(cfg.get("max_new_tokens", 2048)),
        extra_body={"thinking": {"type": "disabled"}},
    )
    return VerifierChatModel(model, prompt, tools)


def build_default_reviser(config: dict[str, Any], config_dir: str | Path) -> Any:
    cfg = dict(config["llm"])
    return build_structured_model(
        cfg,
        ReviserOutput,
        load_prompt_with_protocol(prompt_dir(config, config_dir), "reviser.md"),
    )


def build_default_router(
    config: dict[str, Any],
    config_dir: str | Path,
) -> Any:
    cfg = dict(config["llm"])
    model = build_structured_model(
        cfg,
        RouterAction,
        load_prompt_with_protocol(prompt_dir(config, config_dir), "router.md"),
    )
    return ProtocolSelfReviewModel(model, model)


def parse_router_action(value: Any) -> RouterAction:
    if hasattr(value, "model_dump"):
        return RouterAction.model_validate(value.model_dump())
    return RouterAction.model_validate(parse_json_content(value))
