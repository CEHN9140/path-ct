from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts_2026_7_20.experiment_05_five_evidence_two_scope_review import (
    cross_fitted_membership_evidence,
)


def tool_set_identifiability(
    cluster_state: dict[str, Any],
    patient_states_by_id: dict[str, Any],
    output_root: str,
    config_dir: str = "",
    all_cluster_states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    memberships = {
        str(state.get("set_id", state.get("cluster_id", ""))): [
            str(case_id) for case_id in list(state.get("member_ids", []) or [])
        ]
        for state in list(all_cluster_states or [])
    }
    active = {
        case_id for members in memberships.values() for case_id in members
    }
    checkpoint = json.loads(
        (
            Path(output_root)
            / "storage"
            / "pipeline_checkpoints"
            / "evidence_ready.json"
        ).read_text(encoding="utf-8")
    )
    all_ids = [
        str(state["case_id"]) for state in checkpoint["patient_states"]
    ]
    active_ids = [case_id for case_id in all_ids if case_id in active]
    index = {case_id: position for position, case_id in enumerate(all_ids)}
    positions = [index[case_id] for case_id in active_ids]
    fused = np.load(Path(output_root) / "fused_similarity.npy")
    evidence = cross_fitted_membership_evidence(
        fused[np.ix_(positions, positions)],
        active_ids,
        memberships,
    )
    return {
        "tool_name": "tool_set_identifiability",
        "status": "success",
        "cluster_id": "GLOBAL",
        "results": {
            "summary": (
                "Current membership was predicted by fold-heldout assignment "
                "on the fixed fused patient network."
            ),
            "metrics": {
                "set_identifiability": {
                    "global": evidence["global"],
                    "candidate_sets": evidence["candidate_sets"],
                }
            },
            "warnings": [
                "This is internal assignability, not external reproducibility."
            ],
            "missing_reason": "",
        },
        "artifacts": {},
        "errors": [],
    }
