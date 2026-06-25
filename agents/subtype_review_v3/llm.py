from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.subtype_review_v3.schemas import GlobalRouterDecision, GlobalVerifierReview, RouterDecision, VerifierReview
from utils.llm_utils import LocalLLMClient, extract_json_object


class StructuredChatModel:
    def __init__(self, runnable: Any, system_prompt: str):
        self.runnable = runnable
        self.system_prompt = system_prompt

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.runnable.invoke(
            [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ]
        )
        if hasattr(result, "model_dump"):
            return dict(result.model_dump())
        return dict(result or {})


class LocalStructuredModel:
    def __init__(self, client: LocalLLMClient, schema: type, system_prompt: str):
        self.client = client
        self.schema = schema
        self.system_prompt = system_prompt

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ]
        )
        parsed = extract_json_object(str(response.get("content", "") or "")) or {}
        if parsed.get("_parse_error"):
            raise RuntimeError(str(parsed.get("raw_output", "LLM parse error.")))
        if self.schema is VerifierReview and "accept_ready" not in parsed:
            empty_blocks = ["set_reliability"] if not list(payload.get("tool_results", []) or []) else []
            parsed = {
                "accept_ready": False,
                "verification_vector": {},
                "evidence_gaps": empty_blocks[:1] or ["set_reliability"],
                "confidence_level": "low",
                "reason_codes": [],
                "metric_refs": [],
                "reasoning_summary": str(parsed.get("reasoning_summary", "") or ""),
            }
        if self.schema is GlobalVerifierReview and "set_reviews" not in parsed:
            parsed = {
                "ready_for_revision": False,
                "set_reviews": [],
                "global_evidence_gaps": ["set_reliability"],
                "cross_set_findings": [],
                "reasoning_summary": str(parsed.get("reasoning_summary", "") or ""),
            }
        return dict(self.schema.model_validate(parsed).model_dump())


def load_prompt(prompt_dir: str | Path, name: str) -> str:
    return (Path(prompt_dir) / name).read_text(encoding="utf-8")


def load_prompt_with_protocol(prompt_dir: str | Path, name: str) -> str:
    prompt_dir = Path(prompt_dir)
    prompt = load_prompt(prompt_dir, name)
    protocol_path = prompt_dir / "protocol.md"
    if not protocol_path.exists():
        return prompt
    return f"{prompt}\n\n{protocol_path.read_text(encoding='utf-8')}"


def build_structured_model(llm_config: dict[str, Any], schema: type, system_prompt: str):
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return LocalStructuredModel(LocalLLMClient(llm_config), schema, system_prompt)
    model = ChatOpenAI(
        model=str(llm_config.get("model_name", llm_config.get("model", "qwen2.5"))),
        base_url=str(llm_config.get("base_url", "")),
        api_key=str(llm_config.get("api_key", "EMPTY")),
        temperature=float(llm_config.get("temperature", 0.0) or 0.0),
        max_tokens=int(llm_config.get("max_new_tokens", 1024) or 1024),
    )
    return StructuredChatModel(model.with_structured_output(schema), system_prompt)


def build_default_verifier(config: dict[str, Any], config_dir: str | Path) -> Any:
    llm_config = dict(config.get("llm", {}) or {})
    if not llm_config:
        return None
    prompt_dir = Path(
        config.get("prompt_dir")
        or Path(__file__).resolve().parent / "prompts"
    )
    if not prompt_dir.is_absolute():
        prompt_dir = Path(config_dir).resolve().parent / prompt_dir
    return build_structured_model(
        llm_config,
        GlobalVerifierReview,
        load_prompt_with_protocol(prompt_dir, "verifier.md"),
    )


def build_default_router(config: dict[str, Any], config_dir: str | Path) -> Any:
    llm_config = dict(config.get("llm", {}) or {})
    if not llm_config:
        return None
    prompt_dir = Path(
        config.get("prompt_dir")
        or Path(__file__).resolve().parent / "prompts"
    )
    if not prompt_dir.is_absolute():
        prompt_dir = Path(config_dir).resolve().parent / prompt_dir
    return build_structured_model(
        llm_config,
        GlobalRouterDecision,
        load_prompt_with_protocol(prompt_dir, "router.md"),
    )
