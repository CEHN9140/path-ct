from __future__ import annotations

from typing import Any, Mapping

from utils.cluster_flow import generate_cluster_report, revision_engine


def revision_engine_node(state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    patient_states_by_id = {
        str(patient_id): dict(patient_state)
        for patient_id, patient_state in dict(inventory.get("patient_states_by_id", {}) or {}).items()
    }
    cluster_state = revision_engine(
        dict(inventory.get("cluster_state", {}) or {}),
        patient_states_by_id,
        output_root=str(inventory.get("output_root", "")),
        config_dir=str(inventory.get("config_dir", "") or ""),
        all_cluster_states=list(inventory.get("all_cluster_states", []) or []),
    )
    if str(cluster_state.get("next_action", "") or "") in {"accept", "drop"}:
        cluster_state = generate_cluster_report(
            cluster_state,
            output_root=str(inventory.get("output_root", "")),
            config_dir=str(inventory.get("config_dir", "") or ""),
        )
    inventory["cluster_state"] = cluster_state
    inventory["patient_states_by_id"] = patient_states_by_id
    return {"inventory": inventory}
