from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from utils.tool_utils import to_jsonable


def parse_simple_value(value: str) -> Any:
    value = str(value).strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [parse_simple_value(item) for item in value.strip("[]").split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    content = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(content) or {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any, Any, str]] = [(-1, root, None, "")]
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if text.startswith("- "):
            value = parse_simple_value(text[2:])
            if not isinstance(parent, list):
                container = []
                owner, key = stack[-1][2], stack[-1][3]
                if isinstance(owner, dict) and key:
                    owner[key] = container
                    stack[-1] = (stack[-1][0], container, owner, key)
                parent = container
            if isinstance(parent, list):
                parent.append(value)
            continue
        if ":" not in text:
            continue
        key, _, value = text.partition(":")
        key = key.strip()
        if value.strip() in {">", "|"}:
            parent[key] = ""
        elif value.strip():
            parent[key] = parse_simple_value(value)
        else:
            parent[key] = {}
            stack.append((indent, parent[key], parent, key))
    return root


def extract_json_object(text: str) -> dict[str, Any] | None:
    content = str(text or "").strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json\n", "", 1).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def local_llm_server_available(base_url: str, timeout: float = 2.0) -> bool:
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/models", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


class LocalLLMClient:
    def __init__(self, llm_config: dict[str, Any]):
        self.llm_config = dict(llm_config)
        self.base_url = str(llm_config["base_url"])
        self.model_name = str(llm_config["model_name"])
        self.temperature = float(llm_config["temperature"])
        self.max_tokens = int(llm_config["max_new_tokens"])

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not local_llm_server_available(self.base_url):
            prompt_payload = extract_json_object(messages[-1]["content"] if messages else "") or {}
            if isinstance(prompt_payload.get("fallback_report"), dict):
                return {"content": json.dumps(prompt_payload["fallback_report"], ensure_ascii=False), "tool_calls": []}
            has_validation = bool(dict(prompt_payload.get("validation_results", {}) or {}))
            action = ""
            tools_to_call = [] if has_validation else [
                {"tool_name": "tool_stability_check"},
                {"tool_name": "tool_mutation_enrichment"},
                {"tool_name": "tool_pathway_enrichment"},
                {"tool_name": "tool_confound_test"},
                {"tool_name": "tool_multimodal_consistency_check"},
            ]
            fallback_audit = {
                "cluster_id": str(prompt_payload.get("cluster_id", "") or ""),
                "current_hypothesis": "candidate subtype pending validation",
                "hypothesis_type": "candidate_subtype",
                "strong_evidence": [],
                "supportive_evidence": [],
                "weak_evidence": [],
                "missing_evidence": ["LLM server is unavailable; fallback audit used."],
                "contradictory_evidence": [],
                "fatal_issues": {
                    "unstable_cluster": False,
                    "confounder_driven": False,
                    "severe_missingness_bias": False,
                    "no_biological_interpretability": False,
                    "insufficient_sample_context": False,
                },
                "decision_sufficiency": {
                    "is_sufficient": False,
                    "sufficient_for": None,
                    "why": "Local LLM server was unavailable.",
                    "missing_evidence_that_may_change_decision": [item["tool_name"] for item in tools_to_call],
                    "missing_evidence_only_as_limitation": ["LLM audit fallback"],
                },
                "recommended_action": action,
                "tools_to_call": tools_to_call,
                "split_plan": None,
                "merge_plan": None,
                "limitations_to_report": ["Local LLM server was unavailable."],
                "review_unavailable_reason": "llm_unavailable" if has_validation else "",
                "reasoning_summary": f"Fallback audit selected {action or 'tool_plan'}.",
            }
            return {
                "content": json.dumps(fallback_audit, ensure_ascii=False),
                "tool_calls": [
                    {"id": f"fallback_{index}", "name": item["tool_name"], "arguments": {"tool_name": item["tool_name"]}}
                    for index, item in enumerate(tools_to_call)
                ],
            }
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.base_url,
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                timeout=120.0,
            )
            kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                message = str(exc)
                if tools and "tool choice requires" in message:
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            message = response.choices[0].message
            tool_calls = []
            for call in list(getattr(message, "tool_calls", None) or []):
                function = getattr(call, "function", None)
                raw_arguments = str(getattr(function, "arguments", "") or "{}")
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": raw_arguments}
                tool_calls.append(
                    {
                        "id": str(getattr(call, "id", "") or ""),
                        "name": str(getattr(function, "name", "") or ""),
                        "arguments": arguments,
                    }
                )
            return {"content": str(message.content or ""), "tool_calls": tool_calls}
        except Exception as exc:
            return {"content": json.dumps({"_parse_error": True, "raw_output": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), "tool_calls": []}


def load_llm_client(config_dir: str):
    config_path = Path(config_dir).expanduser() if config_dir else Path("configs")
    config = load_yaml_file(config_path / "subtype_review.yaml")
    return LocalLLMClient(dict(config["llm"]))


def call_llm_json(
    messages: list[dict[str, str]],
    llm_client=None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    client = llm_client or LocalLLMClient({})
    response = client.chat(messages, tools=tools)
    raw_text = str(response.get("content", "") or "")
    parsed = extract_json_object(raw_text)
    retried = False
    if parsed is None:
        retry_messages = list(messages) + [
            {
                "role": "user",
                "content": "Your previous response was not valid JSON. Retry once and return valid JSON only.",
            }
        ]
        response = client.chat(retry_messages, tools=tools)
        raw_text = str(response.get("content", "") or "")
        parsed = extract_json_object(raw_text)
        retried = True
    if parsed is None:
        parsed = {"_parse_error": True}
    parsed.setdefault("raw_output", raw_text)
    parsed["tool_calls"] = list(response.get("tool_calls", []) or [])
    parsed["_retry_used"] = retried
    return dict(to_jsonable(parsed))
