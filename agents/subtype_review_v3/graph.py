from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from agents.subtype_review_v3.schemas import (
    REVIEW_UNAVAILABLE,
    ReviewState,
    global_metric_refs,
    parse_global_router_decision,
    parse_global_verifier_review,
    parse_router_decision,
    parse_verifier_review,
)
from agents.subtype_review_v3.tools import (
    execute_requested_tools,
    normalize_tool_definitions,
)
from utils.tool_utils import safe_identifier, to_jsonable


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def model_invoke(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if model is None:
        role = str(payload.get("model_role", "LLM") or "LLM")
        raise RuntimeError(f"{role} model is unavailable.")
    if hasattr(model, "invoke"):
        result = model.invoke(payload)
    elif callable(model):
        result = model(payload)
    else:
        result = {}
    if hasattr(result, "model_dump"):
        return dict(result.model_dump())
    return dict(result or {})


def normalize_reason_codes(reason_codes: list[str]) -> list[str]:
    normalized = []
    for reason in reason_codes:
        text = str(reason or "").strip()
        lowered = text.lower()
        if "manufacturer" in lowered and "confound" in lowered:
            code = "strong_ct_manufacturer_confounding"
        elif "stage" in lowered and ("echo" in lowered or "known_label" in lowered or "association" in lowered):
            code = "stage_label_echo"
        elif "insufficient" in lowered and "evidence" in lowered:
            code = "insufficient_evidence"
        elif "confounder_exclusion" in lowered:
            code = "confounder_exclusion"
        elif "known_label_echo" in lowered:
            code = "known_label_echo"
        else:
            code = re.sub(r"[^a-z0-9]+", "_", lowered.split(":", 1)[0]).strip("_")
            if len(code) > 80:
                code = code[:80].rstrip("_")
        if code and code not in normalized:
            normalized.append(code)
    return normalized


def review_init(input_state: Mapping[str, Any]) -> ReviewState:
    cluster = dict(input_state.get("cluster", {}) or {})
    member_ids = [str(item) for item in list(cluster.get("member_ids", []) or [])]
    return {
        "cluster_id": str(cluster.get("cluster_id", "unknown_set") or "unknown_set"),
        "member_ids": member_ids,
        "parent_ids": list(cluster.get("parent_cluster_ids", []) or [str(cluster.get("cluster_id", "unknown_set"))]),
        "round_index": 0,
        "budget": dict(input_state.get("budget", {}) or {}),
        "status": "under_review",
        "final_action": "",
        "reason_codes": [],
        "confidence_level": "",
        "verifier_gap": {},
        "verification_vector": {},
        "evidence_gaps": [],
        "router_decision": {},
        "requested_tools": [],
        "tool_results": [],
        "executed_tools": [],
        "round_trace": [],
        "generated_sets": [],
        "absorbed_set_ids": [],
        "lineage": list(cluster.get("lineage", []) or []),
        "artifact_paths": {},
        "system_error": {},
    }


def global_review_init(input_state: Mapping[str, Any]) -> dict[str, Any]:
    candidate_sets = []
    for item in list(input_state.get("candidate_sets", []) or []):
        cluster = dict(item or {})
        candidate_sets.append(
            {
                "cluster_id": str(cluster.get("cluster_id", "") or ""),
                "member_ids": [str(member) for member in list(cluster.get("member_ids", []) or [])],
                "lineage": list(cluster.get("lineage", []) or []),
            }
        )
    return {
        "candidate_sets": candidate_sets,
        "round_index": 0,
        "status": "under_review",
        "executed_tools": [],
        "requested_tools": [],
        "tool_results": [],
        "verifier_review": {},
        "router_decision": {},
        "revision_plan": {},
        "final_sets": [],
        "dropped_set_ids": [],
        "round_trace": [],
        "artifact_paths": {},
        "system_error": {},
    }


def global_verifier_node(state: dict[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "rules": "Review all candidate sets together. Identify evidence gaps and cross-set findings.",
        "candidate_sets": [
            {
                "cluster_id": item.get("cluster_id", ""),
                "member_count": len(list(item.get("member_ids", []) or [])),
            }
            for item in list(state.get("candidate_sets", []) or [])
        ],
        "round_index": state["round_index"],
        "budget": dict(runtime.get("budget", {}) or {}),
        "executed_tools": list(state.get("executed_tools", []) or []),
        "tool_results": list(state.get("tool_results", []) or []),
        "model_role": "Global Verifier",
    }
    try:
        review = parse_global_verifier_review(model_invoke(runtime.get("verifier_model"), payload))
    except Exception as exc:
        error = {"node": "global_verifier", "error_type": type(exc).__name__, "message": str(exc)}
        state["status"] = REVIEW_UNAVAILABLE
        state["system_error"] = error
        state["round_trace"] = list(state.get("round_trace", []) or []) + [
            {"round_index": int(state.get("round_index", 0) or 0), "node": "global_verifier", "status": REVIEW_UNAVAILABLE, **error}
        ]
        return state
    issues = review.contract_issues(
        list(state.get("candidate_sets", []) or []),
        list(state.get("tool_results", []) or []),
    )
    state["verifier_review"] = review.model_dump()
    state["status"] = "needs_routing"
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "global_verifier",
            "ready_for_revision": review.ready_for_revision,
            "global_evidence_gaps": list(review.global_evidence_gaps),
            "contract_issues": issues,
            "reasoning_summary": review.reasoning_summary,
        }
    ]
    return state


def global_router_node(state: dict[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    remaining_budget = max(
        int(dict(runtime.get("budget", {}) or {}).get("max_tool_calls", 0) or 0)
        - len(list(state.get("executed_tools", []) or [])),
        0,
    )
    max_tools_this_round = min(
        remaining_budget,
        max(int(dict(runtime.get("budget", {}) or {}).get("max_tools_per_round", 3) or 3), 1),
    )
    tool_definitions = normalize_tool_definitions(runtime.get("tool_definitions"))
    executed = set(str(item) for item in list(state.get("executed_tools", []) or []))
    available_tool_definitions = {
        name: definition
        for name, definition in tool_definitions.items()
        if name not in executed
    }
    payload = {
        "candidate_sets": [
            {"cluster_id": item.get("cluster_id", ""), "member_count": len(list(item.get("member_ids", []) or []))}
            for item in list(state.get("candidate_sets", []) or [])
        ],
        "round_index": state["round_index"],
        "set_reviews": list(dict(state.get("verifier_review", {}) or {}).get("set_reviews", []) or []),
        "global_evidence_gaps": list(dict(state.get("verifier_review", {}) or {}).get("global_evidence_gaps", []) or []),
        "cross_set_findings": list(dict(state.get("verifier_review", {}) or {}).get("cross_set_findings", []) or []),
        "executed_tools": list(state.get("executed_tools", []) or []),
        "remaining_tool_budget": remaining_budget,
        "max_tools_this_round": max_tools_this_round,
        "tool_catalog": {
            name: {"evidence_blocks": definition.get("evidence_blocks", [])}
            for name, definition in available_tool_definitions.items()
        },
        "available_metric_refs": sorted(
            global_metric_refs(
                list(state.get("candidate_sets", []) or []),
                list(state.get("tool_results", []) or []),
            )
        )[:200],
        "model_role": "Global Router",
    }
    try:
        decision = parse_global_router_decision(model_invoke(runtime.get("router_model"), payload))
    except Exception as exc:
        error = {"node": "global_router", "error_type": type(exc).__name__, "message": str(exc)}
        state["status"] = REVIEW_UNAVAILABLE
        state["system_error"] = error
        state["round_trace"] = list(state.get("round_trace", []) or []) + [
            {"round_index": int(state.get("round_index", 0) or 0), "node": "global_router", "status": REVIEW_UNAVAILABLE, **error}
        ]
        return state
    requested = []
    rejected = []
    for tool_name in decision.requested_tools:
        tool_name = str(tool_name)
        if tool_name not in tool_definitions or tool_name in executed or tool_name in requested:
            rejected.append(tool_name)
            continue
        if len(requested) >= max_tools_this_round:
            rejected.append(tool_name)
            continue
        requested.append(tool_name)
    payload = decision.model_dump()
    payload["requested_tools"] = requested
    payload["rejected_tools"] = rejected
    payload["reason_codes"] = normalize_reason_codes(list(payload.get("reason_codes", []) or []))
    checked = parse_global_router_decision(payload)
    issues = checked.contract_issues(
        list(state.get("candidate_sets", []) or []),
        list(state.get("tool_results", []) or []),
    )
    state["router_decision"] = payload
    state["requested_tools"] = requested
    state["status"] = "continue_review" if checked.action == "continue_review" else "revision_ready"
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "global_router",
            "action": checked.action,
            "requested_tools": requested,
            "rejected_tools": rejected,
            "reason_codes": list(payload["reason_codes"]),
            "contract_issues": issues,
            "reasoning_summary": checked.reasoning_summary,
        }
    ]
    return state


def global_tool_executor_node(state: dict[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    candidate_sets = list(state.get("candidate_sets", []) or [])
    member_ids = sorted({str(member) for item in candidate_sets for member in list(item.get("member_ids", []) or [])})
    new_results = execute_requested_tools(
        list(state.get("requested_tools", []) or []),
        dict(runtime.get("tool_functions", {}) or {}),
        {"cluster_id": "GLOBAL", "member_ids": member_ids},
        dict(runtime.get("patient_states_by_id", {}) or {}),
        str(runtime.get("output_root", "")),
        str(runtime.get("config_dir", "") or ""),
        candidate_sets,
    )
    state["tool_results"] = list(state.get("tool_results", []) or []) + new_results
    state["executed_tools"] = list(state.get("executed_tools", []) or []) + [
        str(result.get("tool_name", "")) for result in new_results if str(result.get("tool_name", ""))
    ]
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "tool_executor",
            "executed_tools": [result["tool_name"] for result in new_results],
            "failed_tools": [result["tool_name"] for result in new_results if str(result.get("status", "")) == "failure"],
        }
    ]
    return state


def global_reviser_node(state: dict[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(dict(state.get("router_decision", {}) or {}).get("revision_plan", {}) or {})
    accept_ids = [str(item) for item in list(plan.get("accept", []) or [])]
    drop_ids = [str(item) for item in list(plan.get("drop", []) or [])]
    sets_by_id = {str(item.get("cluster_id", "")): dict(item) for item in list(state.get("candidate_sets", []) or [])}
    final_sets = []
    for cluster_id in accept_ids:
        if cluster_id in sets_by_id:
            item = dict(sets_by_id[cluster_id])
            item["status"] = "accept"
            final_sets.append(item)
    state["revision_plan"] = plan
    state["final_sets"] = final_sets
    state["dropped_set_ids"] = drop_ids
    state["status"] = "complete"
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "global_reviser",
            "accepted_set_ids": accept_ids,
            "dropped_set_ids": drop_ids,
        }
    ]
    return state


def verifier_node(state: ReviewState, runtime: Mapping[str, Any]) -> ReviewState:
    payload = {
        "rules": "Accept requires accept_ready=true and high confidence. Otherwise emit evidence gaps.",
        "cluster_id": state["cluster_id"],
        "member_count": len(list(state.get("member_ids", []) or [])),
        "round_index": state["round_index"],
        "budget": dict(state.get("budget", {}) or {}),
        "tool_results": list(state.get("tool_results", []) or []),
        "model_role": "Verifier",
    }
    try:
        review = parse_verifier_review(model_invoke(runtime.get("verifier_model"), payload))
    except Exception as exc:
        error = {"node": "verifier", "error_type": type(exc).__name__, "message": str(exc)}
        state["status"] = REVIEW_UNAVAILABLE
        state["reason_codes"] = ["verifier_failure"]
        state["verifier_gap"] = {"reason": f"{error['error_type']}: {error['message']}"}
        state["system_error"] = error
        state["round_trace"] = list(state.get("round_trace", []) or []) + [
            {
                "round_index": int(state.get("round_index", 0) or 0),
                "node": "verifier",
                "status": REVIEW_UNAVAILABLE,
                "error_type": error["error_type"],
                "message": error["message"],
            }
        ]
        return state
    issues = review.contract_issues(list(state.get("tool_results", []) or []))
    state["verification_vector"] = dict(review.verification_vector)
    state["evidence_gaps"] = list(review.evidence_gaps)
    state["confidence_level"] = review.confidence_level
    state["reason_codes"] = normalize_reason_codes(list(review.reason_codes))
    if review.accept_ready and not issues:
        state["status"] = "accept"
        state["final_action"] = "accept"
    else:
        state["status"] = "needs_routing"
        state["verifier_gap"] = {
            "blocks": list(review.evidence_gaps),
            "reason": review.reasoning_summary or "Verifier requires routing.",
            "contract_issues": issues,
            "metric_refs": list(review.metric_refs),
        }
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "verifier",
            "accept_ready": review.accept_ready,
            "confidence_level": review.confidence_level,
            "evidence_gaps": list(review.evidence_gaps),
            "reason_codes": normalize_reason_codes(list(review.reason_codes)),
            "contract_issues": issues,
            "reasoning_summary": review.reasoning_summary,
        }
    ]
    return state


def router_node(state: ReviewState, runtime: Mapping[str, Any]) -> ReviewState:
    remaining_budget = max(
        int(dict(state.get("budget", {}) or {}).get("max_tool_calls", 0) or 0)
        - len(list(state.get("executed_tools", []) or [])),
        0,
    )
    tool_definitions = normalize_tool_definitions(runtime.get("tool_definitions"))
    payload = {
        "cluster_id": state["cluster_id"],
        "round_index": state["round_index"],
        "evidence_gaps": list(state.get("evidence_gaps", []) or []),
        "verification_vector": dict(state.get("verification_vector", {}) or {}),
        "executed_tools": list(state.get("executed_tools", []) or []),
        "remaining_tool_budget": remaining_budget,
        "tool_catalog": {
            name: {"evidence_blocks": definition.get("evidence_blocks", [])}
            for name, definition in tool_definitions.items()
        },
        "tool_results": list(state.get("tool_results", []) or []),
        "model_role": "Router",
    }
    try:
        decision = parse_router_decision(model_invoke(runtime.get("router_model"), payload))
    except Exception as exc:
        error = {"node": "router", "error_type": type(exc).__name__, "message": str(exc)}
        state["status"] = REVIEW_UNAVAILABLE
        state["reason_codes"] = ["router_failure"]
        state["system_error"] = error
        state["round_trace"] = list(state.get("round_trace", []) or []) + [
            {
                "round_index": int(state.get("round_index", 0) or 0),
                "node": "router",
                "status": REVIEW_UNAVAILABLE,
                "error_type": error["error_type"],
                "message": error["message"],
            }
        ]
        return state

    executed = set(str(item) for item in list(state.get("executed_tools", []) or []))
    requested = []
    rejected = []
    for tool_name in decision.requested_tools:
        tool_name = str(tool_name)
        if tool_name not in tool_definitions or tool_name in executed or tool_name in requested:
            rejected.append(tool_name)
            continue
        if len(requested) >= remaining_budget:
            rejected.append(tool_name)
            continue
        requested.append(tool_name)
    payload = decision.model_dump()
    payload["requested_tools"] = requested
    payload["rejected_tools"] = rejected
    payload["reason_codes"] = normalize_reason_codes(list(payload.get("reason_codes", []) or []))
    checked_decision = parse_router_decision(payload)
    issues = checked_decision.contract_issues(list(state.get("tool_results", []) or []))
    if issues and not (checked_decision.action == "continue_review" and not requested):
        error = {"node": "router", "error_type": "ContractError", "message": ";".join(issues)}
        state["status"] = REVIEW_UNAVAILABLE
        state["reason_codes"] = ["router_failure"]
        state["system_error"] = error
        state["round_trace"] = list(state.get("round_trace", []) or []) + [
            {
                "round_index": int(state.get("round_index", 0) or 0),
                "node": "router",
                "status": REVIEW_UNAVAILABLE,
                "contract_issues": issues,
            }
        ]
        return state
    state["router_decision"] = payload
    state["requested_tools"] = requested
    state["reason_codes"] = list(payload["reason_codes"])
    if decision.action == "continue_review":
        state["status"] = "continue_review"
        state["verifier_gap"] = {
            "blocks": list(decision.target_blocks),
            "reason": decision.continue_review_reason,
            "metric_refs": list(decision.metric_refs),
            "rejected_tools": rejected,
        }
    else:
        state["status"] = "revision_ready"
        state["final_action"] = decision.action
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "router",
            "action": decision.action,
            "requested_tools": requested,
            "rejected_tools": rejected,
            "reason_codes": list(payload["reason_codes"]),
            "reasoning_summary": decision.reasoning_summary,
        }
    ]
    return state


def tool_executor_node(state: ReviewState, runtime: Mapping[str, Any]) -> ReviewState:
    requested_tools = list(state.get("requested_tools", []) or [])
    new_results = execute_requested_tools(
        requested_tools,
        dict(runtime.get("tool_functions", {}) or {}),
        {"cluster_id": state["cluster_id"], "member_ids": state["member_ids"]},
        dict(runtime.get("patient_states_by_id", {}) or {}),
        str(runtime.get("output_root", "")),
        str(runtime.get("config_dir", "") or ""),
        list(runtime.get("all_cluster_states", []) or []),
    )
    tool_results = list(state.get("tool_results", []) or []) + new_results
    executed_tools = list(state.get("executed_tools", []) or []) + [
        str(result.get("tool_name", "")) for result in new_results if str(result.get("tool_name", ""))
    ]
    state["tool_results"] = tool_results
    state["executed_tools"] = executed_tools
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "tool_executor",
            "executed_tools": [result["tool_name"] for result in new_results],
            "failed_tools": [result["tool_name"] for result in new_results if str(result.get("status", "")) == "failure"],
        }
    ]
    return state


def reviser_node(state: ReviewState, runtime: Mapping[str, Any]) -> ReviewState:
    action = str(state.get("final_action", "") or "")
    cluster = dict(runtime.get("cluster", {}) or {})
    if action:
        state["status"] = action
    if action == "split":
        groups = list(dict(cluster.get("split_plan", {}) or {}).get("groups", []) or [])
        if not groups:
            members = list(state.get("member_ids", []) or [])
            midpoint = max(1, len(members) // 2)
            groups = [members[:midpoint], members[midpoint:]]
        generated = []
        for index, group in enumerate(groups, start=1):
            members = [str(item) for item in list(group or []) if str(item) in state["member_ids"]]
            if not members:
                continue
            generated.append(
                {
                    "cluster_id": f"{state['cluster_id']}_S{index}",
                    "member_ids": members,
                    "parent_cluster_ids": [state["cluster_id"]],
                    "lineage": list(state.get("lineage", []) or [])
                    + [{"action": "split", "source_cluster_id": state["cluster_id"]}],
                }
            )
        state["generated_sets"] = generated
    elif action == "merge":
        plan = dict(cluster.get("merge_plan", {}) or {})
        target_ids = [str(item) for item in list(plan.get("target_cluster_ids", []) or []) if str(item)]
        all_sets = {
            str(item.get("cluster_id", "")): dict(item)
            for item in list(runtime.get("all_cluster_states", []) or [])
        }
        members = list(state.get("member_ids", []) or [])
        for target_id in target_ids:
            members.extend(list(all_sets.get(target_id, {}).get("member_ids", []) or []))
        merged_id = str(plan.get("cluster_id", "") or f"{state['cluster_id']}_M1")
        state["absorbed_set_ids"] = target_ids
        state["generated_sets"] = [
            {
                "cluster_id": merged_id,
                "member_ids": sorted(set(str(item) for item in members)),
                "parent_cluster_ids": [state["cluster_id"], *target_ids],
                "lineage": list(state.get("lineage", []) or [])
                + [{"action": "merge", "source_cluster_id": state["cluster_id"], "target_cluster_ids": target_ids}],
            }
        ]
    state["round_trace"] = list(state.get("round_trace", []) or []) + [
        {
            "round_index": int(state.get("round_index", 0) or 0),
            "node": "reviser",
            "action": action,
            "generated_set_count": len(list(state.get("generated_sets", []) or [])),
        }
    ]
    return state


def reporter_node(state: ReviewState, runtime: Mapping[str, Any]) -> ReviewState:
    output_root = Path(str(runtime.get("output_root", "")))
    set_dir = output_root / "subtype_review_v3" / safe_identifier(state["cluster_id"])
    report = {
        "metadata": {
            "cluster_id": state["cluster_id"],
            "member_ids": list(state.get("member_ids", []) or []),
            "member_count": len(list(state.get("member_ids", []) or [])),
            "status": state.get("status", ""),
            "final_action": state.get("final_action", ""),
            "confidence_level": state.get("confidence_level", ""),
            "reason_codes": list(state.get("reason_codes", []) or []),
            "rounds_used": int(state.get("round_index", 0) or 0),
        },
        "tool_results": list(state.get("tool_results", []) or []),
        "verification_vector": dict(state.get("verification_vector", {}) or {}),
        "evidence_gaps": list(state.get("evidence_gaps", []) or []),
        "router_decision": dict(state.get("router_decision", {}) or {}),
        "round_trace": list(state.get("round_trace", []) or []),
        "lineage": list(state.get("lineage", []) or []),
        "generated_sets": list(state.get("generated_sets", []) or []),
        "absorbed_set_ids": list(state.get("absorbed_set_ids", []) or []),
        "verifier_gap": dict(state.get("verifier_gap", {}) or {}),
        "system_error": dict(state.get("system_error", {}) or {}),
        "tool_load_errors": dict(runtime.get("tool_load_errors", {}) or {}),
    }
    paths = dict(state.get("artifact_paths", {}) or {})
    paths["report"] = write_json(set_dir / "report.json", report)
    paths["round_trace"] = write_json(set_dir / "round_trace.json", report["round_trace"])
    paths["tool_metrics"] = write_json(set_dir / "tool_metrics.json", state.get("tool_results", []))
    state["artifact_paths"] = paths
    return state


def global_reporter_node(state: dict[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(str(runtime.get("output_root", "")))
    review_dir = output_root / "subtype_review_v3"
    set_reviews = {
        str(item.get("cluster_id", "")): dict(item)
        for item in list(dict(state.get("verifier_review", {}) or {}).get("set_reviews", []) or [])
    }
    sets = []
    accepted = {str(item.get("cluster_id", "")) for item in list(state.get("final_sets", []) or [])}
    dropped = set(str(item) for item in list(state.get("dropped_set_ids", []) or []))
    for item in list(state.get("candidate_sets", []) or []):
        cluster_id = str(item.get("cluster_id", "") or "")
        review = set_reviews.get(cluster_id, {})
        if state.get("status") == REVIEW_UNAVAILABLE:
            final_action = REVIEW_UNAVAILABLE
        else:
            final_action = "accept" if cluster_id in accepted else "drop" if cluster_id in dropped else "drop"
        sets.append(
            {
                "cluster_id": cluster_id,
                "member_count": len(list(item.get("member_ids", []) or [])),
                "status": final_action,
                "final_action": final_action,
                "confidence_level": str(review.get("confidence_level", "")),
                "reason_codes": (
                    ["system_failure"]
                    if state.get("status") == REVIEW_UNAVAILABLE
                    else normalize_reason_codes(list(review.get("reason_codes", []) or [])) or ["insufficient_evidence"]
                ),
                "rounds_used": int(state.get("round_index", 0) or 0),
                "executed_tools": list(state.get("executed_tools", []) or []),
                "report_path": str(review_dir / safe_identifier(cluster_id) / "report.json"),
            }
        )
    summary = {
        "stage": "subtype_review_v3",
        "mode": "global",
        "set_count": len(sets),
        "sets": sets,
        "action_counts": {},
    }
    for item in sets:
        action = str(item.get("final_action", "") or item.get("status", ""))
        summary["action_counts"][action] = int(summary["action_counts"].get(action, 0)) + 1
    paths = dict(state.get("artifact_paths", {}) or {})
    paths["final_review_summary"] = write_json(review_dir / "final_review_summary.json", summary)
    paths["global_decision_trace"] = write_json(review_dir / "global_decision_trace.json", state.get("round_trace", []))
    paths["global_tool_metrics"] = write_json(review_dir / "global_tool_metrics.json", state.get("tool_results", []))
    for item in sets:
        cluster_id = str(item.get("cluster_id", ""))
        set_dir = review_dir / safe_identifier(cluster_id)
        write_json(
            set_dir / "report.json",
            {
                "metadata": item,
                "set_review": set_reviews.get(cluster_id, {}),
                "tool_results": list(state.get("tool_results", []) or []),
                "round_trace": list(state.get("round_trace", []) or []),
                "revision_plan": dict(state.get("revision_plan", {}) or {}),
            },
        )
    state["summary"] = summary
    state["artifact_paths"] = paths
    return state


def route_after_verifier(state: ReviewState) -> str:
    status = str(state.get("status", "") or "")
    if status == "needs_routing":
        return "router"
    return "reporter"


def route_after_router(state: ReviewState) -> str:
    status = str(state.get("status", "") or "")
    if status == "continue_review":
        return "tool_executor"
    if status == "revision_ready":
        return "reviser"
    return "reporter"


def build_review_graph(runtime: Mapping[str, Any] | None = None) -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return None
    graph_runtime = dict(runtime or {})
    builder = StateGraph(dict)
    builder.add_node(
        "review_init",
        lambda state: {
            **review_init(state),
            "runtime": {**graph_runtime, **dict(state).get("runtime", {})},
        },
    )
    builder.add_node("verifier", lambda state: verifier_node(state, state.get("runtime", {})))
    builder.add_node("router", lambda state: router_node(state, state.get("runtime", {})))
    builder.add_node("tool_executor", lambda state: tool_executor_node(state, state.get("runtime", {})))
    builder.add_node("reviser", lambda state: reviser_node(state, state.get("runtime", {})))
    builder.add_node("reporter", lambda state: reporter_node(state, state.get("runtime", {})))
    builder.add_edge(START, "review_init")
    builder.add_edge("review_init", "verifier")
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"router": "router", "reporter": "reporter"},
    )
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"tool_executor": "tool_executor", "reviser": "reviser", "reporter": "reporter"},
    )
    builder.add_edge("tool_executor", "verifier")
    builder.add_edge("reviser", "reporter")
    builder.add_edge("reporter", END)
    return builder.compile()


def run_review_graph(input_state: Mapping[str, Any]) -> ReviewState:
    runtime = dict(input_state)
    state = review_init(input_state)
    max_rounds = int(dict(state.get("budget", {}) or {}).get("max_rounds", 3) or 3)
    must_verify_new_tool_results = False
    while True:
        if int(state.get("round_index", 0) or 0) >= max_rounds and not must_verify_new_tool_results:
            state["status"] = "drop"
            state["final_action"] = "drop"
            state["confidence_level"] = ""
            state["reason_codes"] = ["insufficient_evidence"]
            state["round_trace"] = list(state.get("round_trace", []) or []) + [
                {
                    "round_index": int(state.get("round_index", 0) or 0),
                    "node": "budget",
                    "reason": "Maximum review rounds reached without high-confidence final action.",
                }
            ]
            break
        reviewed_new_tool_results = must_verify_new_tool_results
        state["round_index"] = int(state.get("round_index", 0) or 0) + 1
        must_verify_new_tool_results = False
        state = verifier_node(state, runtime)
        if state.get("status") == REVIEW_UNAVAILABLE:
            break
        if state.get("status") == "accept":
            break
        if reviewed_new_tool_results and int(state.get("round_index", 0) or 0) >= max_rounds:
            state["status"] = "drop"
            state["final_action"] = "drop"
            state["confidence_level"] = ""
            state["reason_codes"] = ["insufficient_evidence"]
            state["round_trace"] = list(state.get("round_trace", []) or []) + [
                {
                    "round_index": int(state.get("round_index", 0) or 0),
                    "node": "budget",
                    "reason": "Maximum review rounds reached after reviewing the latest tool results.",
                    "verifier_gap": dict(state.get("verifier_gap", {}) or {}),
                }
            ]
            break
        remaining_budget = max(
            int(dict(state.get("budget", {}) or {}).get("max_tool_calls", 0) or 0)
            - len(list(state.get("executed_tools", []) or [])),
            0,
        )
        if remaining_budget <= 0:
            state["status"] = "drop"
            state["final_action"] = "drop"
            state["confidence_level"] = ""
            state["reason_codes"] = ["insufficient_evidence"]
            state["round_trace"] = list(state.get("round_trace", []) or []) + [
                {
                    "round_index": int(state.get("round_index", 0) or 0),
                    "node": "budget",
                    "reason": "No remaining tool budget for verifier evidence gaps.",
                    "verifier_gap": dict(state.get("verifier_gap", {}) or {}),
                }
            ]
            break
        state = router_node(state, runtime)
        if state.get("status") == REVIEW_UNAVAILABLE:
            break
        if state.get("status") == "revision_ready":
            state = reviser_node(state, runtime)
            break
        if state.get("status") == "continue_review":
            if not list(state.get("requested_tools", []) or []):
                state["status"] = "drop"
                state["final_action"] = "drop"
                state["confidence_level"] = ""
                state["reason_codes"] = ["insufficient_evidence", "no_additional_evidence"]
                state["round_trace"] = list(state.get("round_trace", []) or []) + [
                    {
                        "round_index": int(state.get("round_index", 0) or 0),
                        "node": "no_additional_evidence",
                        "reason": "Router requested more evidence, but no executable new tools remained.",
                        "verifier_gap": dict(state.get("verifier_gap", {}) or {}),
                    }
                ]
                break
            state = tool_executor_node(state, runtime)
            must_verify_new_tool_results = True
            continue
        break
    return reporter_node(state, runtime)


def run_global_review_graph(input_state: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(input_state)
    state = global_review_init(input_state)
    max_rounds = int(dict(runtime.get("budget", {}) or {}).get("max_rounds", 3) or 3)
    must_verify_new_tool_results = False
    while True:
        if int(state.get("round_index", 0) or 0) >= max_rounds and not must_verify_new_tool_results:
            state["status"] = "complete"
            state["revision_plan"] = {"accept": [], "drop": [str(item.get("cluster_id", "")) for item in state["candidate_sets"]], "merge": [], "split": []}
            state["dropped_set_ids"] = list(state["revision_plan"]["drop"])
            state["round_trace"] = list(state.get("round_trace", []) or []) + [
                {"round_index": int(state.get("round_index", 0) or 0), "node": "budget", "reason": "Maximum global review rounds reached."}
            ]
            break
        reviewed_new_tool_results = must_verify_new_tool_results
        must_verify_new_tool_results = False
        state["round_index"] = int(state.get("round_index", 0) or 0) + 1
        state = global_verifier_node(state, runtime)
        if state.get("status") == REVIEW_UNAVAILABLE:
            break
        if reviewed_new_tool_results and int(state.get("round_index", 0) or 0) > max_rounds:
            state["status"] = "complete"
            state["revision_plan"] = {"accept": [], "drop": [str(item.get("cluster_id", "")) for item in state["candidate_sets"]], "merge": [], "split": []}
            state["dropped_set_ids"] = list(state["revision_plan"]["drop"])
            state["round_trace"] = list(state.get("round_trace", []) or []) + [
                {"round_index": int(state.get("round_index", 0) or 0), "node": "budget", "reason": "Maximum global review rounds reached after reviewing the latest tool results."}
            ]
            break
        state = global_router_node(state, runtime)
        if state.get("status") == REVIEW_UNAVAILABLE:
            break
        if state.get("status") == "revision_ready":
            state = global_reviser_node(state, runtime)
            break
        if state.get("status") == "continue_review":
            if not list(state.get("requested_tools", []) or []):
                state["status"] = "complete"
                state["revision_plan"] = {"accept": [], "drop": [str(item.get("cluster_id", "")) for item in state["candidate_sets"]], "merge": [], "split": []}
                state["dropped_set_ids"] = list(state["revision_plan"]["drop"])
                state["round_trace"] = list(state.get("round_trace", []) or []) + [
                    {"round_index": int(state.get("round_index", 0) or 0), "node": "no_additional_evidence"}
                ]
                break
            state = global_tool_executor_node(state, runtime)
            must_verify_new_tool_results = True
            continue
        break
    return global_reporter_node(state, runtime)
