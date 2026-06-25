from __future__ import annotations

from typing import Any, Mapping

from utils.cluster_flow import (
    build_agentic_evidence_matrix,
    build_evidence_catalog,
    build_structured_evidence,
    build_verifier_decision,
    llm_action_decision,
    llm_evidence_audit,
    save_debug_artifacts,
    save_subtype_review_json,
)


def verifier_node(state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    cluster_state = dict(inventory.get("cluster_state", {}) or {})
    config_dir = str(inventory.get("config_dir", "") or "")
    output_root = str(inventory.get("output_root", ""))
    round_id = int(cluster_state.get("review_round", 0) or 0)

    cluster_state["structured_evidence"] = build_structured_evidence(cluster_state)
    if save_debug_artifacts(config_dir):
        artifacts = dict(cluster_state.get("artifacts", {}) or {})
        artifacts[f"structured_evidence_round_{round_id}"] = save_subtype_review_json(
            output_root,
            cluster_state,
            f"structured_evidence_round_{round_id}.json",
            cluster_state["structured_evidence"],
        )
        cluster_state["artifacts"] = artifacts

    cluster_state["subtype_set_context"] = [
        {
            "cluster_id": str(item.get("cluster_id", "") or ""),
            "member_count": len(list(item.get("member_ids", []) or [])),
            "status": str(item.get("status", "") or ""),
            "final_action": str(item.get("final_action", "") or ""),
        }
        for item in list(inventory.get("all_cluster_states", []) or [])
    ]

    # Planner: decide whether targeted evidence could change the set-level action.
    llm_evidence_audit(
        cluster_state,
        output_root=output_root,
        config_dir=config_dir,
    )

    tools_to_call = list(
        cluster_state.get("tool_plan", [])
        or dict(cluster_state.get("llm_audit", {}) or {}).get("tools_to_call", [])
        or []
    )
    blocks_to_update = list(
        dict(cluster_state.get("router_decision", {}) or {}).get(
            "blocks_to_update", []
        )
        or []
    )
    cluster_state["evidence_matrix"] = build_agentic_evidence_matrix(
        cluster_state,
        dict(inventory.get("patient_states_by_id", {}) or {}),
        previous_evidence_matrix=dict(cluster_state.get("evidence_matrix", {}) or {}),
        blocks_to_update=blocks_to_update or None,
    )
    cluster_state["evidence_catalog"] = build_evidence_catalog(
        str(cluster_state.get("cluster_id", "")),
        dict(cluster_state.get("evidence_matrix", {}) or {}),
    )

    # Decider: only run when the Planner has no targeted tools left to request.
    if not tools_to_call:
        llm_action_decision(
            cluster_state, output_root=output_root, config_dir=config_dir
        )
        decider_decision = dict(cluster_state.get("decider_decision", {}) or {})
        decider_blocks = list(decider_decision.get("blocks_to_update", []) or [])
        if (
            str(decider_decision.get("decision_state", "") or "") == "continue_review"
            and decider_blocks
        ):
            cluster_state["decider_requested_evidence"] = {
                "blocks_to_update": decider_blocks,
                "continue_review_reason": str(
                    decider_decision.get("continue_review_reason", "") or ""
                ),
                "reasoning_summary": str(
                    decider_decision.get("reasoning_summary", "") or ""
                ),
                "metric_refs": list(decider_decision.get("metric_refs", []) or []),
            }
            llm_evidence_audit(
                cluster_state,
                output_root=output_root,
                config_dir=config_dir,
            )
    cluster_state["verifier_decision"] = build_verifier_decision(cluster_state)
    artifacts = dict(cluster_state.get("artifacts", {}) or {})
    matrix_artifact_name = (
        f"round_{round_id}_pre_tool_evidence_matrix"
        if tools_to_call
        else f"round_{round_id}_evidence_matrix"
    )
    matrix_filename = (
        f"round_{round_id}/pre_tool_evidence_matrix.json"
        if tools_to_call
        else f"round_{round_id}/evidence_matrix.json"
    )
    artifacts[matrix_artifact_name] = save_subtype_review_json(
        output_root,
        cluster_state,
        matrix_filename,
        cluster_state["evidence_matrix"],
    )
    artifacts[f"round_{round_id}_verifier"] = save_subtype_review_json(
        output_root,
        cluster_state,
        f"round_{round_id}/verifier.json",
        cluster_state["verifier_decision"],
    )
    cluster_state["artifacts"] = artifacts

    if save_debug_artifacts(config_dir):
        artifacts = dict(cluster_state.get("artifacts", {}) or {})
        artifacts[f"structured_evidence_llm_round_{round_id}"] = (
            save_subtype_review_json(
                output_root,
                cluster_state,
                f"structured_evidence_llm_round_{round_id}.json",
                list(cluster_state.get("structured_evidence", []) or []),
            )
        )
        cluster_state["artifacts"] = artifacts

    inventory["cluster_state"] = cluster_state
    return {"inventory": inventory}
