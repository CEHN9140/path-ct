from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from agents.subtype_review.schemas import (
    EvidenceReportBatch,
    RevisionPlan,
    RouterPlan,
)
from agents.subtype_review.tools import TOOL_REGISTRY, build_validation_tools


class LLMOutputLengthError(RuntimeError):
    pass


class LLMUsageTracker:
    def __init__(self, output_path: str | Path | None = None):
        self.api_calls = 0
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None
        self.prompt_cache_hit_tokens: int | None = None
        self.prompt_cache_miss_tokens: int | None = None
        self.cache_observed_requests = 0
        self.usage_observed_requests = 0
        self.output_path = Path(output_path) if output_path is not None else None
        self.current_request: dict[str, Any] = {}
        self.started_at = perf_counter()

    def before_request(self, *, role: str = "unknown", model: str = "", messages: list | None = None) -> None:
        self.api_calls += 1
        self.started_at = perf_counter()
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":")) if messages is not None else ""
        self.current_request = {
            "api_call": self.api_calls,
            "role": role,
            "requested_model": model,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_chars": len(serialized),
            "messages_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        }

    def record_response(self, response: Any) -> None:
        def get(obj: Any, key: str) -> Any:
            return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)

        metadata = get(response, "response_metadata") or {}
        # LangChain may expose normalized usage AND provider-specific cache counters.
        sources = [get(response, "usage"), get(metadata, "token_usage"), get(response, "usage_metadata")]
        counts: dict[str, int | None] = {}
        for key, names in {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
            "prompt_cache_hit_tokens": ("prompt_cache_hit_tokens",),
            "prompt_cache_miss_tokens": ("prompt_cache_miss_tokens",),
        }.items():
            value = next((get(source, name) for source in sources for name in names
                          if get(source, name) is not None), None)
            counts[key] = int(value) if value is not None else None
        if counts["prompt_cache_hit_tokens"] is None:
            cached = next((get(get(source, field), key) for source in sources for field, key in (
                ("prompt_tokens_details", "cached_tokens"), ("input_token_details", "cache_read"),
            ) if get(get(source, field), key) is not None), None)
            counts["prompt_cache_hit_tokens"] = int(cached) if cached is not None else None
        hit, miss, prompt = (counts[key] for key in (
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "prompt_tokens"
        ))
        if miss is None and hit is not None and prompt is not None and prompt >= hit:
            counts["prompt_cache_miss_tokens"] = prompt - hit
        cache_observed = all(counts[key] is not None for key in (
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"
        ))
        self.cache_observed_requests += int(cache_observed)
        self.usage_observed_requests += int(prompt is not None)
        for key, value in counts.items():
            # Cache totals use only requests with both counters known; missing != zero.
            if key.startswith("prompt_cache_") and not cache_observed:
                continue
            if value is not None:
                setattr(self, key, int(getattr(self, key) or 0) + int(value))
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                **self.current_request,
                "response_model": get(response, "model") or get(metadata, "model_name"),
                "elapsed_seconds": perf_counter() - self.started_at,
                **counts,
            }
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def snapshot(self) -> dict[str, int | float | None]:
        measured_tokens = (self.prompt_cache_hit_tokens or 0) + (self.prompt_cache_miss_tokens or 0)
        return {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens,
            "prompt_cache_hit_rate": self.prompt_cache_hit_tokens / measured_tokens if measured_tokens else None,
            "cache_observed_requests": self.cache_observed_requests,
            "usage_observed_requests": self.usage_observed_requests,
        }


def api_extra_body(config: Mapping[str, Any]) -> dict[str, Any]:
    base_url = str(config.get("base_url", "")).lower()
    model_name = str(config.get("model_name", "")).lower()
    if "dashscope.aliyuncs.com" in base_url and model_name.startswith("qwen3.8-"):
        return {"enable_thinking": False}
    if "deepseek" in base_url or model_name.startswith("deepseek-"):
        return {"thinking": {"type": "disabled"}}
    return {}


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
    review_config["multi_k"] = {
        key: value
        for key, value in dict(review_config.get("multi_k", {}) or {}).items()
        if key not in {"initial_ks", "repeats"}
    }
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
    return json.loads(str(content))


def serialize_llm_payload(payload: Mapping[str, Any]) -> str:
    """Stable data precedes changing instructions; list order and values are preserved."""
    stable = (
        "partition", "round_evidence", "available_metric_refs", "evidence_reports",
        "current_evidence", "prior_reports",
    )
    keys = [key for key in stable if key in payload] + sorted(set(payload) - set(stable))
    return "{" + ",".join(
        json.dumps(key) + ":" + json.dumps(payload[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for key in keys
    ) + "}"


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
        api_key_env = str(self.config.get("api_key_env", "") or "").strip()
        if not api_key_env:
            raise ValueError(
                "llm.api_key_env must name the API key environment variable"
            )
        api_key = str(os.environ.get(api_key_env, "") or "").strip()
        if not api_key:
            raise RuntimeError(
                f"Required API key environment variable is unset: {api_key_env}"
            )
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=str(self.config["base_url"]),
            timeout=float(self.config.get("timeout", 120)),
        )

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": serialize_llm_payload(payload)},
        ]
        request = {
            "model": str(self.config["model_name"]),
            "messages": messages,
            "temperature": float(self.config.get("temperature", 0.0)),
            "max_tokens": int(self.config["max_new_tokens"]),
            "response_format": {"type": "json_object"},
        }
        extra_body = api_extra_body(self.config)
        if extra_body:
            request["extra_body"] = extra_body
        if self.usage_tracker:
            role = {RouterPlan: "router", RevisionPlan: "reviser", EvidenceReportBatch: "verifier_audit"}.get(self.schema, "structured")
            self.usage_tracker.before_request(role=role, model=str(self.config["model_name"]), messages=messages)
        response = self.client.chat.completions.create(**request)
        if self.usage_tracker:
            self.usage_tracker.record_response(response)
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise LLMOutputLengthError("LLM output reached max_new_tokens before completing JSON")
        return self.schema.model_validate(parse_json_content(response.choices[0].message.content)).model_dump()


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

    def invoke(self, payload: dict[str, Any]) -> Any:
        mode = payload["mode"]
        if mode == "acquire":
            tool_names = sorted(payload["eligible_tools"])
            model = self.acquire_model.bind_tools(
                [self.tools[name] for name in tool_names],
                tool_choice="auto",
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": serialize_llm_payload(payload)},
            ]
            if self.usage_tracker:
                self.usage_tracker.before_request(
                    role="verifier_acquire", model=self.acquire_model.model_name, messages=messages,
                )
            response = model.invoke(messages)
            if self.usage_tracker:
                self.usage_tracker.record_response(response)
            return response
        if mode == "audit":
            request = dict(payload)
            request["round_evidence"] = sorted(
                [{key: value for key, value in row.items()
                  if key not in {"metric_refs", "artifact_paths", "partition_signature"}}
                 for row in request["round_evidence"]],
                key=lambda row: (row.get("tool_name", ""), tuple(row.get("target_ids", []))),
            )
            request["prior_reports"] = [
                {key: value for key, value in row.items() if key != "metric_refs"}
                for row in request.get("prior_reports", [])
            ]
            return self.audit_model.invoke(request)
        raise ValueError(f"Unknown verifier mode: {mode}")


def build_default_verifier(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = (prompt_dir(config, config_dir) / "verifier.md").read_text(encoding="utf-8")
    tools = build_validation_tools()
    from langchain_openai import ChatOpenAI

    api_key_env = str(cfg.get("api_key_env", "") or "").strip()
    if not api_key_env:
        raise ValueError(
            "llm.api_key_env must name the API key environment variable"
        )
    api_key = str(os.environ.get(api_key_env, "") or "").strip()
    if not api_key:
        raise RuntimeError(
            f"Required API key environment variable is unset: {api_key_env}"
        )
    model_kwargs = {
        "model": str(cfg["model_name"]),
        "base_url": str(cfg["base_url"]),
        "api_key": api_key,
        "temperature": float(cfg.get("temperature", 0.0)),
        "timeout": float(cfg.get("timeout", 120)),
    }
    extra_body = api_extra_body(cfg)
    if extra_body:
        model_kwargs["extra_body"] = extra_body
    return VerifierChatModel(
        ChatOpenAI(**model_kwargs),
        JsonStructuredModel(cfg, EvidenceReportBatch, prompt, usage_tracker),
        prompt, tools, usage_tracker,
    )


def build_default_router(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = (prompt_dir(config, config_dir) / "router.md").read_text(encoding="utf-8")
    return JsonStructuredModel(cfg, RouterPlan, prompt, usage_tracker)


def build_default_reviser(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = (prompt_dir(config, config_dir) / "reviser.md").read_text(encoding="utf-8")
    return JsonStructuredModel(cfg, RevisionPlan, prompt, usage_tracker)


def summarize_reports(reports: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dimension": row.get("dimension", ""),
            "aspect": row.get("aspect", ""),
            "scope": row.get("scope", ""),
            "target_ids": list(row.get("target_ids", []) or []),
            "observations": list(row.get("observations", []) or []),
            "statistical_interpretation": row.get("statistical_interpretation", ""),
            "medical_interpretation": row.get("medical_interpretation", ""),
            "limitations": list(row.get("limitations", []) or []),
            "tool_refs": list(row.get("tool_refs", []) or []),
            "internal_structure_assessment": row.get("internal_structure_assessment"),
            "pair_boundary_assessment": row.get("pair_boundary_assessment"),
            "suggested_k": row.get("suggested_k"),
        }
        for row in reports
    ]
