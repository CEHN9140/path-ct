from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from pydantic import ValidationError

from agents.subtype_review.schemas import (
    EvidenceReportBatch,
    RevisionPlan,
    RouterDecisionAudit,
    RouterPlan,
)
from agents.subtype_review.runtime_trace import append_runtime_trace
from agents.subtype_review.tools import TOOL_REGISTRY, build_selection_tools


class LLMOutputLengthError(RuntimeError):
    pass


def structured_role(schema: type) -> str:
    return {
        RouterPlan: "router",
        RouterDecisionAudit: "router_audit",
        RevisionPlan: "reviser",
        EvidenceReportBatch: "verifier_audit",
    }.get(schema, "structured")


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
        for name in ("verifier.md", "router.md", "router_audit.md", "reviser.md")
    }
    review_config = dict(config)
    for key in ("prompt_dir", "repeat", "output_root", "experiment_root"):
        review_config.pop(key, None)
    review_config.pop("multi_k", None)
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
        runtime_trace_path: str | Path | None = None,
    ):
        self.config = dict(llm_config)
        self.schema = schema
        self.runtime_trace_path = Path(runtime_trace_path) if runtime_trace_path is not None else None
        allowed_fields = ", ".join(sorted(self.schema.model_fields))
        self.prompt = (
            f"{system_prompt.rstrip()}\n\n"
            "Return exactly one valid JSON object.\n"
            f"Allowed top-level JSON fields: {allowed_fields}.\n"
            "Do not output any additional top-level wrapper fields."
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
        base_messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": serialize_llm_payload(payload)},
        ]
        messages = list(base_messages)
        max_retries = int(self.config.get("structured_output_retries", 1))
        role = structured_role(self.schema)

        for attempt in range(max_retries + 1):
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
                request_role = role if attempt == 0 else f"{role}_schema_repair"
                self.usage_tracker.before_request(
                    role=request_role,
                    model=str(self.config["model_name"]),
                    messages=messages,
                )
            response = self.client.chat.completions.create(**request)
            if self.usage_tracker:
                self.usage_tracker.record_response(response)
            if getattr(response.choices[0], "finish_reason", None) == "length":
                if attempt >= max_retries:
                    raise LLMOutputLengthError("LLM output reached max_new_tokens before completing JSON")
                append_runtime_trace(
                    self.runtime_trace_path,
                    node=role,
                    event="structured_output_truncated",
                    round_id=payload.get("round"),
                    payload={"schema": self.schema.__name__, "attempt": attempt + 1},
                )
                messages = [
                    *base_messages,
                    {"role": "user", "content": (
                        "Return the same required JSON more compactly. Preserve every required "
                        "schema item, report, action, and scientific conclusion; compress prose "
                        "only and do not omit report coverage."
                    )},
                ]
                continue

            raw_content = response.choices[0].message.content
            parsed = None
            try:
                parsed = parse_json_content(raw_content)
                validated = self.schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                validation_errors = (
                    exc.errors()
                    if isinstance(exc, ValidationError)
                    else [{"type": "json_decode_error", "msg": str(exc)}]
                )
                append_runtime_trace(
                    self.runtime_trace_path,
                    node=role,
                    event="structured_output_invalid",
                    round_id=payload.get("round"),
                    payload={
                        "schema": self.schema.__name__,
                        "attempt": attempt + 1,
                        "raw_output": raw_content,
                        "parsed_output": parsed,
                        "validation_errors": validation_errors,
                    },
                )
                if attempt >= max_retries:
                    raise

                error_summary = [
                    {
                        "path": ".".join(str(part) for part in error.get("loc", ())),
                        "type": error.get("type", "validation_error"),
                        "message": error.get("msg", "Invalid value"),
                    }
                    for error in validation_errors
                ]
                repair_instruction = (
                    "Your previous response was valid JSON or attempted JSON, but it did not satisfy "
                    "the required output schema. The only allowed top-level fields are: "
                    f"{sorted(self.schema.model_fields)}. Do not output wrapper fields such as 'type', "
                    "'format', 'json_object', 'response', or 'schema'. Preserve the scientific decision "
                    "and all substantive content from your previous response. Repair only the JSON/schema "
                    "structure needed to satisfy the required schema. Fix these validation errors: "
                    f"{json.dumps(error_summary, ensure_ascii=False)}. Return exactly one corrected "
                    "JSON object and no commentary."
                )
                messages = [
                    *base_messages,
                    {"role": "assistant", "content": str(raw_content)},
                    {"role": "user", "content": repair_instruction},
                ]
                continue

            if attempt > 0:
                append_runtime_trace(
                    self.runtime_trace_path,
                    node=role,
                    event="structured_output_repaired",
                    round_id=payload.get("round"),
                    payload={"schema": self.schema.__name__, "attempt": attempt + 1},
                )
            return validated.model_dump()

        raise RuntimeError("Unreachable structured-output state")


class VerifierChatModel:
    def __init__(
        self,
        acquire_model: Any,
        audit_model: Any,
        system_prompt: str,
        tools: list[Any],
        usage_tracker: LLMUsageTracker | None = None,
        runtime_trace_path: str | Path | None = None,
        selection_retries: int = 1,
    ):
        self.acquire_model = acquire_model
        self.audit_model = audit_model
        self.system_prompt = system_prompt
        self.tools = {str(item.name): item for item in tools}
        self.usage_tracker = usage_tracker
        self.runtime_trace_path = Path(runtime_trace_path) if runtime_trace_path is not None else None
        self.selection_retries = max(0, int(selection_retries))

    def invoke(self, payload: dict[str, Any]) -> Any:
        mode = payload["mode"]
        if mode == "select":
            return self._select_next_tool(
                payload,
                tool_names=list(payload["remaining_tools"]),
                require_tool=bool(payload["require_tool"]),
            )
        if mode == "audit":
            request = dict(payload)
            request["round_evidence"] = sorted(
                [{key: value for key, value in row.items()
                  if key not in {"metric_refs", "artifact_paths", "partition_signature", "request_ref"}}
                 for row in request["round_evidence"]],
                key=lambda row: (row.get("tool_name", ""), tuple(row.get("target_ids", []))),
            )
            request["prior_reports"] = [
                {key: value for key, value in row.items() if key != "metric_refs"}
                for row in request.get("prior_reports", [])
            ]
            return self.audit_model.invoke(request)
        raise ValueError(f"Unknown verifier mode: {mode}")

    def _select_next_tool(
        self,
        payload: dict[str, Any],
        *,
        tool_names: list[str],
        require_tool: bool,
    ) -> dict[str, Any]:
        if not tool_names:
            return {"selected_tool": None, "stop_reason": "no_remaining_eligible_tools"}
        unknown = set(tool_names) - set(self.tools)
        if unknown:
            raise ValueError(f"Selection requested unregistered tools: {sorted(unknown)}")

        tools = [self.tools[name] for name in sorted(tool_names)]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": serialize_llm_payload(payload)},
        ]
        last_error = ""
        for attempt in range(self.selection_retries + 1):
            bind_args = {"tool_choice": "required"} if require_tool else {}
            model = self.acquire_model.bind_tools(tools, **bind_args)
            if self.usage_tracker:
                self.usage_tracker.before_request(
                    role="verifier_select", model=self.acquire_model.model_name, messages=messages,
                )
            response = model.invoke(messages)
            if self.usage_tracker:
                self.usage_tracker.record_response(response)
            calls = list(getattr(response, "tool_calls", None) or [])
            if not calls:
                additional = getattr(response, "additional_kwargs", {}) or {}
                calls = list(additional.get("tool_calls", []) or [])
            if not calls:
                if require_tool:
                    last_error = "selection returned no tool call when exactly one eligible tool call was required"
                else:
                    return {
                        "selected_tool": None,
                        "stop_reason": "verifier_no_further_tool_call",
                    }
            elif len(calls) == 1:
                call = calls[0]
                name = str(call.get("name", "")) if isinstance(call, Mapping) else str(getattr(call, "name", ""))
                args = call.get("args", {}) if isinstance(call, Mapping) else getattr(call, "args", {})
                if name in tool_names and args in ({}, None):
                    return {"selected_tool": name}
                last_error = (
                    f"selected tool {name!r} is not eligible" if name not in tool_names
                    else f"selection tool arguments must be empty, got {args!r}"
                )
            else:
                last_error = f"selection returned {len(calls)} tool calls; at most one is allowed"

            append_runtime_trace(
                self.runtime_trace_path,
                node="verifier",
                event="verifier_tool_selection_retry",
                round_id=payload.get("round"),
                payload={"attempt": attempt + 1, "error": last_error},
            )
            if attempt < self.selection_retries:
                messages = [
                    *messages,
                    {"role": "assistant", "content": "The previous selection was invalid."},
                    {"role": "user", "content": (
                        f"Your previous response was invalid: {last_error}. Call at most one of the currently "
                        "eligible zero-argument tools. Choose exactly one if additional evidence is needed; "
                        "if stopping is allowed, make no tool call."
                    )},
                ]
        message = f"Verifier tool selection remained invalid after retries: {last_error}"
        if last_error.startswith("selection returned no tool call"):
            raise RuntimeError(message)
        raise ValueError(message)


def build_default_verifier(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
    runtime_trace_path: str | Path | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = (prompt_dir(config, config_dir) / "verifier.md").read_text(encoding="utf-8")
    tools = build_selection_tools()
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
        JsonStructuredModel(
            cfg, EvidenceReportBatch, prompt, usage_tracker,
            runtime_trace_path=runtime_trace_path,
        ),
        prompt, tools, usage_tracker, runtime_trace_path,
        int(cfg.get("tool_selection_retries", cfg.get("structured_output_retries", 1))),
    )


def build_default_router(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
    runtime_trace_path: str | Path | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = (prompt_dir(config, config_dir) / "router.md").read_text(encoding="utf-8")
    return JsonStructuredModel(
        cfg, RouterPlan, prompt, usage_tracker,
        runtime_trace_path=runtime_trace_path,
    )


def build_default_router_audit(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
    runtime_trace_path: str | Path | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = (prompt_dir(config, config_dir) / "router_audit.md").read_text(encoding="utf-8")
    return JsonStructuredModel(
        cfg, RouterDecisionAudit, prompt, usage_tracker,
        runtime_trace_path=runtime_trace_path,
    )


def build_default_reviser(
    config: dict[str, Any],
    config_dir: str | Path,
    *,
    usage_tracker: LLMUsageTracker | None = None,
    runtime_trace_path: str | Path | None = None,
) -> Any:
    cfg = dict(config["llm"])
    prompt = (prompt_dir(config, config_dir) / "reviser.md").read_text(encoding="utf-8")
    return JsonStructuredModel(
        cfg, RevisionPlan, prompt, usage_tracker,
        runtime_trace_path=runtime_trace_path,
    )


def summarize_reports(reports: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "report_ref": row.get("report_ref", ""),
            "dimension": row.get("dimension", ""),
            "aspect": row.get("aspect", ""),
            "scope": row.get("scope", ""),
            "target_ids": list(row.get("target_ids", []) or []),
            "observations": list(row.get("observations", []) or []),
            "dimension_interpretation": row.get("dimension_interpretation", ""),
            "cross_evidence_context": row.get("cross_evidence_context", ""),
            "limitations": list(row.get("limitations", []) or []),
        }
        for row in reports
    ]
