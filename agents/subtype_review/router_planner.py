from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from utils.cluster_flow import (
    REVIEW_UNAVAILABLE_STATUS,
    agentic_router_decision,
    generate_cluster_report,
    save_debug_artifacts,
    save_subtype_review_json,
)
from utils.llm_utils import load_yaml_file
from utils.tool_utils import to_jsonable


def route_cluster_action(state: Mapping[str, Any]) -> str:
    cluster_state = dict(
        dict(state.get("inventory", {}) or {}).get("cluster_state", {}) or {}
    )
    action = str(cluster_state.get("next_action", "") or "")
    if action in {"accept", "drop", REVIEW_UNAVAILABLE_STATUS}:
        return "end"
    if action in {"call_tools", "continue_review", "split", "merge"}:
        return "revision_engine_node"
    return "end"


def route_after_revision(state: Mapping[str, Any]) -> str:
    cluster_state = dict(
        dict(state.get("inventory", {}) or {}).get("cluster_state", {}) or {}
    )
    if list(cluster_state.get("generated_clusters", []) or []):
        return "end"
    if str(cluster_state.get("status", "") or "") == "under_review":
        return "compute_verification_vector"
    return "end"


def router_planner_node(state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    cluster_state = dict(inventory.get("cluster_state", {}) or {})
    config_dir = str(inventory.get("config_dir", "") or "")
    router_decision = agentic_router_decision(cluster_state, config_dir)
    audit = dict(cluster_state.get("llm_audit", {}) or {})
    subtype_review_config = load_yaml_file(
        Path(config_dir).expanduser() / "subtype_review.yaml"
    )
    allowed_tools = list(subtype_review_config["allowed_tools"])
    recommended_tools = []
    for item in list(
        router_decision.get("tools_to_call", [])
        or audit.get("tools_to_call", [])
        or []
    ):
        tool_plan = {"tool_name": str(item)} if isinstance(item, str) else dict(item)
        tool_name = str(tool_plan.get("tool_name", tool_plan.get("name", "")) or "")
        if tool_name in allowed_tools:
            recommended_tools.append({"tool_name": tool_name, **tool_plan})

    if str(router_decision.get("route_action", "")) == "send_to_revision":
        action = str(router_decision.get("send_to_revision_action", "") or "drop")
        forced_drop_reason = str(router_decision.get("forced_drop_reason", "") or "")
        if action == REVIEW_UNAVAILABLE_STATUS:
            cluster_state["review_unavailable_reason"] = forced_drop_reason or action
        elif action == "drop" and forced_drop_reason:
            cluster_state["drop_reason"] = forced_drop_reason
        elif action == "drop" and not str(cluster_state.get("drop_reason", "") or ""):
            reason_codes = list(
                dict(cluster_state.get("verifier_decision", {}) or {}).get(
                    "reason_codes", []
                )
                or []
            )
            if reason_codes:
                cluster_state["drop_reason"] = str(reason_codes[0])
    else:
        action = "call_tools" if recommended_tools else "continue_review"

    cluster_state["next_action"] = action
    cluster_state["final_action"] = action if action in {"accept", "drop"} else ""
    cluster_state["router_decision"] = router_decision
    cluster_state["recommended_tools"] = recommended_tools
    cluster_state["split_plan"] = audit.get("split_plan")
    cluster_state["merge_plan"] = audit.get("merge_plan")
    cluster_state["limitations"] = list(audit.get("limitations_to_report", []) or [])
    if action in {"accept", "drop"}:
        cluster_state["final_decision"] = action
        cluster_state["status"] = action
    elif action == REVIEW_UNAVAILABLE_STATUS:
        cluster_state["final_action"] = ""
        cluster_state["final_decision"] = ""
        cluster_state["status"] = REVIEW_UNAVAILABLE_STATUS

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "router_planner",
        "level": "info",
        "message": "Router / Planner selected bounded action.",
        "extra": to_jsonable(
            {"action": action, "recommended_tools": recommended_tools}
        ),
    }
    log_entry["action"] = action
    cluster_state["action_history"] = list(cluster_state.get("action_history", [])) + [
        to_jsonable(log_entry)
    ]
    round_id = int(cluster_state.get("review_round", 0) or 0)
    round_summaries = [
        dict(item)
        for item in list(cluster_state.get("round_summaries", []) or [])
        if int(dict(item).get("round_index", -1) or -1) != round_id
    ]
    verifier_decision = dict(cluster_state.get("verifier_decision", {}) or {})
    round_summaries.append(
        {
            "round_index": round_id,
            "verifier_summary": str(verifier_decision.get("rationale", "") or ""),
            "router_summary": str(router_decision.get("route_reason", "") or ""),
            "blocks_updated_next": list(
                router_decision.get("blocks_to_update", []) or []
            ),
        }
    )
    cluster_state["round_summaries"] = sorted(
        round_summaries, key=lambda item: int(item.get("round_index", 0) or 0)
    )
    artifacts = dict(cluster_state.get("artifacts", {}) or {})
    artifacts[f"round_{round_id}_router"] = save_subtype_review_json(
        str(inventory.get("output_root", "")),
        cluster_state,
        f"round_{round_id}/router.json",
        router_decision,
    )
    cluster_state["artifacts"] = artifacts
    if save_debug_artifacts(config_dir):
        artifacts["decision"] = save_subtype_review_json(
            str(inventory.get("output_root", "")),
            cluster_state,
            "decision.json",
            {
                "next_action": action,
                "recommended_tools": recommended_tools,
                "budget_state": dict(cluster_state.get("budget_state", {}) or {}),
            },
        )
        artifacts["budget_state"] = save_subtype_review_json(
            str(inventory.get("output_root", "")),
            cluster_state,
            "budget_state.json",
            dict(cluster_state.get("budget_state", {}) or {}),
        )
        cluster_state["artifacts"] = artifacts
    if action in {"accept", "drop", REVIEW_UNAVAILABLE_STATUS}:
        cluster_state = generate_cluster_report(
            cluster_state,
            output_root=str(inventory.get("output_root", "")),
            config_dir=config_dir,
        )
    inventory["cluster_state"] = cluster_state
    return {"inventory": inventory}
