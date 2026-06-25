from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.subtype_review_common import member_case_ids, tool_result


def round_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return round(number, 6)


def summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "q25": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": round_value(np.mean(array)),
        "median": round_value(np.median(array)),
        "q25": round_value(np.quantile(array, 0.25)),
        "min": round_value(np.min(array)),
        "max": round_value(np.max(array)),
    }


def candidate_set_members(
    cluster_state: Mapping[str, Any], all_cluster_states: Any
) -> dict[str, list[str]]:
    states = list(all_cluster_states or []) or [cluster_state]
    memberships: dict[str, list[str]] = {}
    assigned = set()
    for index, state in enumerate(states):
        item = dict(state or {})
        set_id = str(
            item.get("cluster_id") or item.get("candidate_set_id") or f"C{index + 1}"
        )
        members = []
        for case_id in member_case_ids(item):
            case_key = str(case_id)
            if case_key not in assigned:
                members.append(case_key)
                assigned.add(case_key)
        if members:
            memberships[set_id] = sorted(members)
    return memberships


def consensus_payload(cluster_state: Mapping[str, Any], all_cluster_states: Any) -> dict[str, Any]:
    for item in [cluster_state] + list(all_cluster_states or []):
        consensus = dict(dict(item or {}).get("consensus", {}) or {})
        if consensus:
            return consensus
    return {}


def load_consensus_matrix(
    consensus: Mapping[str, Any],
) -> tuple[np.ndarray | None, list[str], str | None]:
    matrix_path = str(consensus.get("matrix_path", "") or "")
    metadata_path = str(consensus.get("metadata_path", "") or "")
    if not matrix_path or not metadata_path:
        return None, [], "missing_consensus_matrix_or_metadata"
    if not Path(matrix_path).exists() or not Path(metadata_path).exists():
        return None, [], "missing_consensus_matrix_or_metadata"
    try:
        matrix = np.asarray(np.load(matrix_path), dtype=float)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        patient_ids = [str(item) for item in list(metadata.get("patient_ids", []) or [])]
    except Exception:
        return None, [], "failed_to_load_consensus_matrix_or_metadata"
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] != len(patient_ids)
    ):
        return None, [], "invalid_consensus_matrix_or_metadata"
    return matrix, patient_ids, None


def global_consensus_metrics(
    memberships: Mapping[str, list[str]],
    consensus: Mapping[str, Any],
    matrix_index_by_case: Mapping[str, int],
    missing_reason: str | None,
) -> dict[str, Any]:
    all_members = [case_id for members in memberships.values() for case_id in members]
    matrix_available = sum(1 for case_id in all_members if case_id in matrix_index_by_case)
    return {
        "candidate_set_count": len(memberships),
        "candidate_set_sizes": {
            set_id: len(members) for set_id, members in memberships.items()
        },
        "total_candidate_set_n": len(all_members),
        "matrix_available_n": matrix_available,
        "matrix_missing_n": len(all_members) - matrix_available,
        "partition_count": consensus.get("partition_count"),
        "attempted_partition_count": consensus.get("attempted_partition_count"),
        "saved_partition_count": consensus.get("saved_partition_count"),
        "best_n_clusters": consensus.get("best_n_clusters") or consensus.get("n_clusters"),
        "secondary_algorithm": consensus.get("secondary_algorithm"),
        "selection_metric": consensus.get("selection_metric"),
        "pac": consensus.get("pac") or consensus.get("best_pac"),
        "cdf_area": consensus.get("cdf_area") or consensus.get("best_cdf_area"),
        "delta_area": consensus.get("delta_area") or consensus.get("best_delta_area"),
        "pac_gain": consensus.get("pac_gain"),
        "next_pac_gain": consensus.get("next_pac_gain"),
        "missing_reason": missing_reason,
    }


def empty_set_consensus_row(
    set_id: str,
    set_n: int,
    matrix_available_n: int,
    missing_reason: str | None,
) -> dict[str, Any]:
    return {
        "candidate_set_id": set_id,
        "set_n": set_n,
        "matrix_available_n": matrix_available_n,
        "matrix_missing_n": set_n - matrix_available_n,
        "within_consensus_mean": None,
        "within_consensus_median": None,
        "within_consensus_q25": None,
        "within_consensus_min": None,
        "within_consensus_max": None,
        "outside_consensus_mean": None,
        "outside_consensus_median": None,
        "outside_consensus_max": None,
        "nearest_neighbor_candidate_set_id": None,
        "nearest_other_consensus": None,
        "consensus_margin": None,
        "between_consensus_by_candidate_set": {},
        "silhouette_mean": None,
        "silhouette_median": None,
        "silhouette_q25": None,
        "silhouette_min": None,
        "per_member_consensus_support": {},
        "per_member_silhouette": {},
        "missing_reason": missing_reason,
    }


def silhouette_by_case(
    matrix: np.ndarray,
    memberships: Mapping[str, list[str]],
    matrix_index_by_case: Mapping[str, int],
) -> dict[str, float]:
    case_ids = []
    labels = []
    for set_id, members in memberships.items():
        for case_id in members:
            if case_id in matrix_index_by_case:
                case_ids.append(case_id)
                labels.append(set_id)
    if len(case_ids) < 3 or len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return {}
    indices = [matrix_index_by_case[case_id] for case_id in case_ids]
    distance = 1.0 - np.clip(matrix[np.ix_(indices, indices)], 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    try:
        from sklearn.metrics import silhouette_samples

        scores = silhouette_samples(distance, labels, metric="precomputed")
    except Exception:
        return {}
    return {case_ids[index]: round_value(score) for index, score in enumerate(scores)}


def set_consensus_metrics(
    memberships: Mapping[str, list[str]],
    matrix: np.ndarray | None,
    matrix_index_by_case: Mapping[str, int],
    missing_reason: str | None,
) -> dict[str, dict[str, Any]]:
    silhouettes = (
        silhouette_by_case(matrix, memberships, matrix_index_by_case)
        if matrix is not None
        else {}
    )
    rows = {}
    for set_id, members in memberships.items():
        available_members = [
            case_id for case_id in members if case_id in matrix_index_by_case
        ]
        row = empty_set_consensus_row(
            set_id,
            len(members),
            len(available_members),
            missing_reason if matrix is None else None,
        )
        if matrix is None:
            rows[set_id] = row
            continue
        if len(available_members) < 2:
            row["missing_reason"] = "insufficient_matrix_covered_members"
            rows[set_id] = row
            continue

        indices = [matrix_index_by_case[case_id] for case_id in available_members]
        submatrix = matrix[np.ix_(indices, indices)]
        within_values = (
            submatrix[np.triu_indices(len(indices), k=1)].astype(float).tolist()
        )
        within = summary_stats(within_values)
        row.update(
            {
                "within_consensus_mean": within["mean"],
                "within_consensus_median": within["median"],
                "within_consensus_q25": within["q25"],
                "within_consensus_min": within["min"],
                "within_consensus_max": within["max"],
            }
        )
        row["per_member_consensus_support"] = {
            case_id: round_value(
                np.mean(
                    matrix[
                        matrix_index_by_case[case_id],
                        [
                            matrix_index_by_case[other]
                            for other in available_members
                            if other != case_id
                        ],
                    ]
                )
            )
            for case_id in available_members
        }

        outside_values = []
        between_by_set = {}
        for other_id, other_members in memberships.items():
            if other_id == set_id:
                continue
            other_available = [
                case_id
                for case_id in other_members
                if case_id in matrix_index_by_case
            ]
            if not other_available:
                continue
            other_indices = [matrix_index_by_case[case_id] for case_id in other_available]
            values = matrix[np.ix_(indices, other_indices)].reshape(-1).astype(float)
            outside_values.extend(values.tolist())
            between_by_set[other_id] = round_value(np.mean(values))
        outside = summary_stats(outside_values)
        row.update(
            {
                "outside_consensus_mean": outside["mean"],
                "outside_consensus_median": outside["median"],
                "outside_consensus_max": outside["max"],
                "between_consensus_by_candidate_set": dict(
                    sorted(between_by_set.items())
                ),
            }
        )
        if between_by_set:
            nearest_id, nearest_value = max(
                between_by_set.items(), key=lambda item: item[1]
            )
            row["nearest_neighbor_candidate_set_id"] = nearest_id
            row["nearest_other_consensus"] = nearest_value
            if row["within_consensus_mean"] is not None:
                row["consensus_margin"] = round_value(
                    float(row["within_consensus_mean"]) - float(nearest_value)
                )

        member_silhouettes = {
            case_id: silhouettes[case_id]
            for case_id in available_members
            if case_id in silhouettes
        }
        silhouette_values = list(member_silhouettes.values())
        silhouette = summary_stats(silhouette_values)
        row.update(
            {
                "silhouette_mean": silhouette["mean"],
                "silhouette_median": silhouette["median"],
                "silhouette_q25": silhouette["q25"],
                "silhouette_min": silhouette["min"],
                "per_member_silhouette": member_silhouettes,
            }
        )
        rows[set_id] = row
    return rows


def tool_stability_check(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    memberships = candidate_set_members(cluster_state, all_cluster_states)
    consensus = consensus_payload(cluster_state, all_cluster_states)
    matrix, patient_ids, missing_reason = load_consensus_matrix(consensus)
    matrix_index_by_case = {case_id: index for index, case_id in enumerate(patient_ids)}
    metrics = {
        "set_reliability_global_consensus": global_consensus_metrics(
            memberships,
            consensus,
            matrix_index_by_case,
            missing_reason,
        ),
        "set_reliability_set_consensus": set_consensus_metrics(
            memberships,
            matrix,
            matrix_index_by_case,
            missing_reason,
        ),
    }
    status = "success" if matrix is not None else "warning"
    return tool_result(
        tool_name="tool_stability_check",
        status=status,
        cluster_id=cluster_id,
        output_root=output_root,
        summary=f"Set reliability metrics were computed for {len(memberships)} candidate sets.",
        metrics=metrics,
        missing_reason=missing_reason,
        evidence_hints=[
            {
                "evidence_type": "set_reliability",
                "summary": "Final candidate sets were projected onto the consensus matrix.",
            }
        ],
        artifact_key="stability_json",
        support_level="informational",
        concern_level="low",
        figures={},
    )
