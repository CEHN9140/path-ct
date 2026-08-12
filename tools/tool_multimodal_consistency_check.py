from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools.subtype_review_common import (
    member_case_ids,
    tool_result,
)


MODALITIES = ("ct", "wsi", "rna", "genomic")


def modality_affinity_path(output_root: str, modality: str) -> Path:
    """Return the canonical saved affinity path for a review modality."""
    if modality == "genomic":
        return Path(output_root) / "wxs" / "genomic_affinity.npy"
    return Path(output_root) / "candidate_subtype" / f"{modality}_affinity.npy"


def round_value(value: float) -> float | None:
    return round(float(value), 6) if np.isfinite(value) else None


def candidate_set_members(
    cluster_state: Mapping[str, Any],
    all_cluster_states: Any,
) -> dict[str, list[str]]:
    states = list(all_cluster_states or []) or [cluster_state]
    return {
        str(state.get("set_id") or state.get("cluster_id")): sorted(
            member_case_ids(state)
        )
        for state in states
        if member_case_ids(state)
    }


def normalize_affinity(network: np.ndarray) -> np.ndarray:
    matrix = np.asarray(network, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Affinity matrix must be square")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise ValueError("Affinity diagonal must be positive")
    matrix = matrix / np.sqrt(np.outer(diagonal, diagonal))
    matrix = np.maximum((matrix + matrix.T) / 2.0, 0.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def separation_score(within: float, between: float) -> float:
    denominator = abs(within) + abs(between)
    return (within - between) / denominator if denominator else 0.0


def partition_separation(
    similarity: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    upper = np.triu_indices(len(labels), 1)
    same_set = labels[upper[0]] == labels[upper[1]]
    values = similarity[upper]
    within = float(values[same_set].mean())
    between = float(values[~same_set].mean())
    global_row = {
        "mean_within_affinity": round_value(within),
        "mean_between_affinity": round_value(between),
        "normalized_affinity_separation": round_value(
            separation_score(within, between)
        ),
    }
    per_set = {}
    for set_id in np.unique(labels):
        inside = np.flatnonzero(labels == set_id)
        if len(inside) < 2:
            raise ValueError(
                f"Cross-modal validation requires at least two members in {set_id}"
            )
        within_matrix = similarity[np.ix_(inside, inside)]
        set_within = float(
            within_matrix[~np.eye(len(inside), dtype=bool)].mean()
        )
        between_by_set = {
            str(other): float(
                similarity[
                    np.ix_(inside, np.flatnonzero(labels == other))
                ].mean()
            )
            for other in np.unique(labels)
            if other != set_id
        }
        nearest = max(between_by_set, key=between_by_set.get)
        set_between = between_by_set[nearest]
        per_set[str(set_id)] = {
            "member_n": int(len(inside)),
            "mean_within_affinity": round_value(set_within),
            "nearest_other_set": nearest,
            "mean_nearest_between_affinity": round_value(set_between),
            "normalized_affinity_separation": round_value(
                separation_score(set_within, set_between)
            ),
        }
    return global_row, per_set


def gower_matrix(distance: np.ndarray) -> np.ndarray:
    center = np.eye(len(distance)) - np.ones_like(distance) / len(distance)
    return -0.5 * center @ (distance**2) @ center


def permanova_statistic(
    gower: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    design = pd.get_dummies(pd.Series(labels, dtype=str), dtype=float).to_numpy()
    hat = design @ np.linalg.pinv(design)
    rank = int(np.linalg.matrix_rank(design))
    total = float(np.trace(gower))
    between = max(float(np.trace(hat @ gower)), 0.0)
    residual = max(total - between, 0.0)
    numerator_df = rank - 1
    denominator_df = len(labels) - rank
    pseudo_f = (
        (between / numerator_df) / (residual / denominator_df)
        if numerator_df > 0 and denominator_df > 0 and residual > 0
        else 0.0
    )
    return (between / total if total > 0 else 0.0), pseudo_f


def permanova_metrics(
    similarity: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    gower = gower_matrix(distance)
    r2, _ = permanova_statistic(gower, labels)
    return {
        "available_n": len(labels),
        "set_count": int(len(np.unique(labels))),
        "permanova_r2": round_value(r2),
    }


def compute_cross_modal_consistency(
    affinities: Mapping[str, np.ndarray],
    case_ids: list[str],
    memberships: Mapping[str, list[str]],
    *,
    min_per_set_separation: float | None = None,
) -> dict[str, Any]:
    labels_by_case = {
        case_id: set_id
        for set_id, members in memberships.items()
        for case_id in members
    }
    if set(case_ids) != set(labels_by_case):
        raise ValueError("Affinity patients and candidate members differ")
    labels = np.asarray([labels_by_case[case_id] for case_id in case_ids])
    if len(np.unique(labels)) < 2:
        raise ValueError("Cross-modal validation requires at least two sets")

    rows = {}
    for modality in MODALITIES:
        similarity = normalize_affinity(affinities[modality])
        global_row, per_set = partition_separation(similarity, labels)
        global_row.update(
            {
                **permanova_metrics(similarity, labels),
                "per_set": per_set,
            }
        )
        rows[modality] = global_row
    return {
        "modality_partition_support": rows,
        "analysis_scope": (
            "fixed candidate memberships on CT, WSI, RNA, and WXS+CNV genomic affinity "
            "networks; raw within/between affinity, normalized separation and "
            "PERMANOVA R2 are reported for protocol-based interpretation"
        ),
    }


def tool_multimodal_consistency_check(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "GLOBAL"))
    memberships = candidate_set_members(cluster_state, all_cluster_states)
    candidate_dir = Path(output_root) / "candidate_subtype"
    patient_order = json.loads(
        (candidate_dir / "affinity_patient_order.json").read_text()
    )
    active = {
        case_id for members in memberships.values() for case_id in members
    }
    positions = [
        index for index, case_id in enumerate(patient_order) if case_id in active
    ]
    case_ids = [patient_order[index] for index in positions]
    affinities = {
        modality: np.load(modality_affinity_path(output_root, modality))[
            np.ix_(positions, positions)
        ]
        for modality in MODALITIES
    }
    metrics = compute_cross_modal_consistency(
        affinities,
        case_ids,
        memberships,
    )
    return tool_result(
        tool_name="tool_multimodal_consistency_check",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary="Fixed candidate memberships were evaluated in four modality affinity networks.",
        metrics=metrics,
        decision_metrics=metrics,
        support_level="informational",
        concern_level="none",
        figures={},
    )
