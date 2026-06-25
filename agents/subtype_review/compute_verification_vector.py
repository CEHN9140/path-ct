from __future__ import annotations

from typing import Any, Mapping

from utils.cluster_flow import compute_verification_vector, save_debug_artifacts, save_subtype_review_json


def compute_verification_vector_node(state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    cluster_state = dict(inventory.get("cluster_state", {}) or {})
    cluster_state["review_round"] = int(cluster_state.get("review_round", 0) or 0) + 1
    budget_state = dict(cluster_state.get("budget_state", {}) or {})
    budget_state["review_rounds_used"] = int(cluster_state["review_round"])
    cluster_state["budget_state"] = budget_state
    cluster_state["verification_vector"] = compute_verification_vector(
        cluster_state,
        dict(inventory.get("patient_states_by_id", {}) or {}),
    )
    if save_debug_artifacts(str(inventory.get("config_dir", "") or "")):
        round_id = int(cluster_state.get("review_round", 0) or 0)
        artifacts = dict(cluster_state.get("artifacts", {}) or {})
        artifacts[f"verification_vector_round_{round_id}"] = save_subtype_review_json(
            str(inventory.get("output_root", "")),
            cluster_state,
            f"verification_vector_round_{round_id}.json",
            cluster_state["verification_vector"],
        )
        cluster_state["artifacts"] = artifacts
    inventory["cluster_state"] = cluster_state
    return {"inventory": inventory}
