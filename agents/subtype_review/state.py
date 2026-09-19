from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agents.subtype_review.schemas import ReviewState, RouterPlan, set_id
from agents.subtype_review.tools import TOOL_REGISTRY


def partition_signature(sets: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "set_id": set_id(item),
                "member_ids": sorted(str(member) for member in item.get("member_ids", [])),
            }
            for item in sorted(sets, key=set_id)
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def partition_artifact_id(signature: str) -> str:
    return "p_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def current_sets(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(item) for item in dict(state.get("partition", {}) or {}).get("sets", []) or []],
        key=set_id,
    )


def compact_partition_for_llm(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sets": [
            {"set_id": set_id(item), "member_n": len(item.get("member_ids", []) or [])}
            for item in current_sets(state)
        ]
    }


def context_values(runtime: Any) -> dict[str, Any]:
    if hasattr(runtime, "context"):
        return dict(runtime.context or {})
    return dict(runtime or {})


def registry(runtime: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(runtime.get("tool_registry") or TOOL_REGISTRY)


def append_trace(state: dict[str, Any], event: dict[str, Any]) -> None:
    control = state.setdefault("control", {})
    control.setdefault("trace", []).append(
        {"round": control.get("round", 0), **event}
    )


def is_length_finish_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in {"LengthFinishReasonError", "LLMOutputLengthError"}:
            return True
        current = current.__cause__ or current.__context__
    return any(name in str(exc) for name in ("LengthFinishReasonError", "LLMOutputLengthError"))


def mark_failure(
    state: dict[str, Any], node: str, exc: Exception, immediate: bool = False
) -> None:
    control = dict(state.get("control", {}) or {})
    control["failures"] = int(control.get("failures", 0)) + 1
    control["error"] = f"{type(exc).__name__}: {exc}"
    if node != "verifier":
        control["next"] = node
    if immediate or control["failures"] >= int(control.get("max_failures", 3)):
        control["status"] = "review_unavailable"
        control["next"] = "end"
    state["control"] = control
    append_trace(state, {"node": node, "event": "failure", "error": control["error"]})


def mark_success(state: dict[str, Any]) -> None:
    state["control"].update(failures=0, error=None)


def initial_review_state(candidate_sets: list[dict[str, Any]]) -> ReviewState:
    sets = []
    for item in candidate_sets:
        identifier = set_id(dict(item))
        if identifier:
            sets.append({
                "set_id": identifier,
                "member_ids": sorted(str(member) for member in item.get("member_ids", [])),
                "revision_lineage": list(item.get("revision_lineage", []) or []),
            })
    identifiers = [set_id(item) for item in sets]
    members = [member for item in sets for member in item["member_ids"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Initial candidate sets contain duplicate identifiers")
    if len(members) != len(set(members)):
        raise ValueError("Initial candidate sets contain overlapping patients")
    return {
        "partition": {"sets": sets},
        "round_evidence": [],
        "reports": [],
        "evidence_memory": {},
        "messages": [],
        "router_plan": None,
        "revision_plan": None,
        "revision_result": None,
        "history": [],
        "control": {
            "round": 0,
            "failures": 0,
            "status": "reviewing",
            "next": "router",
            "error": None,
            "max_rounds": 10,
            "max_failures": 3,
            "pending_evidence_requests": [],
            "router_validation_error": None,
            "router_correction_attempts": 0,
            "previous_invalid_plan": None,
            "eligible_tools": {},
            "trace": [],
        },
    }


def history_entry(
    state: Mapping[str, Any], plan: RouterPlan, terminal_only: bool = False
) -> dict[str, Any]:
    return {
        "round": int(state["control"].get("round", 0)) + (0 if terminal_only else 1),
        "terminal_only": terminal_only,
        "partition_signature": partition_signature(current_sets(state)),
        "partition": copy.deepcopy(state["partition"]),
        "round_evidence": copy.deepcopy(state.get("round_evidence", [])),
        "evidence_reports": copy.deepcopy(state.get("reports", [])),
        "router_plan": plan.model_dump(),
        "revision_plan": None,
        "revision_result": None,
    }


def reset_verifier_round_state(state: dict[str, Any]) -> None:
    state["round_evidence"] = []
    state["messages"] = []
