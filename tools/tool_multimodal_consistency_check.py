from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

from tools.subtype_review_common import (
    member_case_ids,
    read_case_feature_table,
    tool_result,
)

MODALITY_CONFIG = {
    "ct": {"key": "ct", "metric": "euclidean"},
    "wsi": {"key": "wsi", "metric": "cosine"},
    "rna": {"key": "rna", "metric": "spearman"},
    "wxs": {"key": "wxs", "metric": "jaccard"},
}
MODALITY_PAIRS = list(combinations(["ct", "wsi", "rna", "wxs"], 2))
MANTEL_PERMUTATIONS = 999
MANTEL_SEED = 123
MANTEL_ALTERNATIVE = "two-sided"
SIMILARITY_SCALING = "p95_distance_to_similarity"


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


def vector_from_state(patient_state, modality):
    if modality in ("ct", "wsi"):
        evidence_key = f"{'ct' if modality == 'ct' else 'wsi'}_evidence"
        evidence = dict(patient_state.get(evidence_key, {}) or {})
        values = evidence.get("features", [])
        if isinstance(values, list) and values:
            vector = np.asarray(values, dtype=float)
            return vector if np.isfinite(vector).all() else None
        path = Path(str(evidence.get("feature_path", "") or ""))
        if path.exists():
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                vector = np.asarray(list(dict(payload).values()), dtype=float)
                return vector if np.isfinite(vector).all() else None
            if path.suffix == ".npy":
                vector = np.asarray(np.load(path), dtype=float).reshape(-1)
                return vector if np.isfinite(vector).all() else None
    return None


def table_vectors(patient_states_by_id, modality):
    if modality not in {"rna", "wxs"}:
        return {}
    path = next(
        (
            str(
                dict(item.get("omics_evidence", {}) or {}).get(
                    f"{modality}_feature_path", ""
                )
                or ""
            )
            for item in patient_states_by_id.values()
            if dict(item.get("omics_evidence", {}) or {}).get(
                f"{modality}_feature_path"
            )
        ),
        "",
    )
    _, table = read_case_feature_table(path)
    vectors = {}
    for case_id, values in table.items():
        vector = np.asarray(list(values.values()), dtype=float)
        if np.isfinite(vector).all():
            vectors[str(case_id)] = vector
    return vectors


def distance_matrix(matrix, metric):
    matrix = np.asarray(matrix, dtype=float)
    if metric == "spearman":
        ranked = np.asarray(
            [rankdata(row, method="average") for row in matrix], dtype=float
        )
        return np.asarray(cdist(ranked, ranked, metric="correlation"), dtype=float)
    if metric == "jaccard":
        return np.asarray(cdist(matrix > 0, matrix > 0, metric="jaccard"), dtype=float)
    return np.asarray(cdist(matrix, matrix, metric=metric), dtype=float)


def get_modality_vectors(patient_states_by_id, modality):
    cached = table_vectors(patient_states_by_id, modality)
    vectors = {}
    for case_id, patient_state in patient_states_by_id.items():
        vector = cached.get(str(case_id))
        if vector is None:
            vector = vector_from_state(patient_state, modality)
        if vector is not None:
            vectors[str(case_id)] = vector
    return vectors


def candidate_set_members(cluster_state: Mapping[str, Any], all_cluster_states: Any) -> dict[str, list[str]]:
    states = list(all_cluster_states or []) or [cluster_state]
    memberships: dict[str, list[str]] = {}
    assigned = set()
    for index, state in enumerate(states):
        row = dict(state or {})
        set_id = str(row.get("cluster_id") or row.get("candidate_set_id") or f"C{index + 1}")
        members = []
        for case_id in sorted(member_case_ids(row)):
            case_key = str(case_id)
            if case_key not in assigned:
                members.append(case_key)
                assigned.add(case_key)
        if members:
            memberships[set_id] = members
    return memberships


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu_indices(matrix.shape[0], k=1)], dtype=float)


def finite_nonconstant(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(
        values.size
        and np.isfinite(values).all()
        and not np.allclose(values, values[0])
    )


def distance_to_similarity(distance: np.ndarray) -> np.ndarray | None:
    values = upper_triangle_values(distance)
    if not finite_nonconstant(values):
        return None
    scale = float(np.percentile(values, 95))
    if not math.isfinite(scale) or scale <= 0:
        return None
    scaled = np.clip(distance / scale, 0.0, 1.0)
    similarity = 1.0 - scaled
    np.fill_diagonal(similarity, 1.0)
    return similarity


def centered_kernel_alignment(left: np.ndarray, right: np.ndarray) -> Any:
    left_similarity = distance_to_similarity(left)
    right_similarity = distance_to_similarity(right)
    if left_similarity is None or right_similarity is None:
        return None
    n = left_similarity.shape[0]
    centerer = np.eye(n) - np.ones((n, n), dtype=float) / float(n)
    left_centered = centerer @ left_similarity @ centerer
    right_centered = centerer @ right_similarity @ centerer
    numerator = float(np.sum(left_centered * right_centered))
    denominator = math.sqrt(
        float(np.sum(left_centered * left_centered))
        * float(np.sum(right_centered * right_centered))
    )
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    return numerator / denominator


def mantel_p_value(left: np.ndarray, right: np.ndarray) -> Any:
    left_values = upper_triangle_values(left)
    right_values = upper_triangle_values(right)
    if not finite_nonconstant(left_values) or not finite_nonconstant(right_values):
        return None
    observed = float(np.corrcoef(left_values, right_values)[0, 1])
    if not math.isfinite(observed):
        return None
    rng = np.random.default_rng(MANTEL_SEED)
    extreme = 0
    for _ in range(MANTEL_PERMUTATIONS):
        order = rng.permutation(left.shape[0])
        permuted = right[np.ix_(order, order)]
        permuted_values = upper_triangle_values(permuted)
        statistic = float(np.corrcoef(left_values, permuted_values)[0, 1])
        if math.isfinite(statistic) and abs(statistic) >= abs(observed):
            extreme += 1
    return (extreme + 1) / float(MANTEL_PERMUTATIONS + 1)


def pair_alignment_row(
    *,
    scope: str,
    candidate_set_id: str | None,
    case_ids: list[str],
    modality_a: str,
    modality_b: str,
    vectors_by_modality: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    metric_a = str(MODALITY_CONFIG[modality_a]["metric"])
    metric_b = str(MODALITY_CONFIG[modality_b]["metric"])
    common = [
        case_id
        for case_id in case_ids
        if case_id in vectors_by_modality.get(modality_a, {})
        and case_id in vectors_by_modality.get(modality_b, {})
    ]
    row = {
        "scope": scope,
        "candidate_set_id": candidate_set_id,
        "modality_a": modality_a,
        "modality_b": modality_b,
        "distance_metric_a": metric_a,
        "distance_metric_b": metric_b,
        "available_n": len(common),
        "pair_count": int(len(common) * (len(common) - 1) / 2),
        "low_pair_count": len(common) < 5,
        "spearman_distance_correlation": None,
        "mantel_p_value": None,
        "cka_similarity_alignment": None,
        "similarity_scaling": SIMILARITY_SCALING,
        "mantel_permutations": MANTEL_PERMUTATIONS,
        "mantel_alternative": MANTEL_ALTERNATIVE,
        "missing_reason": None,
    }
    if len(common) < 3:
        row["missing_reason"] = "insufficient_common_cases"
        return row
    lengths_a = {len(vectors_by_modality[modality_a][case_id]) for case_id in common}
    lengths_b = {len(vectors_by_modality[modality_b][case_id]) for case_id in common}
    if len(lengths_a) != 1 or len(lengths_b) != 1:
        row["missing_reason"] = "ragged_feature_vectors"
        return row
    matrix_a = np.asarray(
        [vectors_by_modality[modality_a][case_id] for case_id in common], dtype=float
    )
    matrix_b = np.asarray(
        [vectors_by_modality[modality_b][case_id] for case_id in common], dtype=float
    )
    if not np.isfinite(matrix_a).all() or not np.isfinite(matrix_b).all():
        row["missing_reason"] = "non_finite_feature_vectors"
        return row
    distance_a = distance_matrix(matrix_a, metric_a)
    distance_b = distance_matrix(matrix_b, metric_b)
    if not np.isfinite(distance_a).all() or not np.isfinite(distance_b).all():
        row["missing_reason"] = "non_finite_distance_matrix"
        return row
    values_a = upper_triangle_values(distance_a)
    values_b = upper_triangle_values(distance_b)
    if not finite_nonconstant(values_a) or not finite_nonconstant(values_b):
        row["missing_reason"] = "constant_distance_matrix"
        return row
    spearman = spearmanr(values_a, values_b).statistic
    cka = centered_kernel_alignment(distance_a, distance_b)
    if cka is None:
        row["missing_reason"] = "zero_cka_denominator"
        return row
    row.update(
        {
            "spearman_distance_correlation": round_value(spearman),
            "mantel_p_value": round_value(mantel_p_value(distance_a, distance_b)),
            "cka_similarity_alignment": round_value(cka),
        }
    )
    return row


def alignment_rows(
    *,
    scope: str,
    candidate_set_id: str | None,
    case_ids: list[str],
    vectors_by_modality: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    return [
        pair_alignment_row(
            scope=scope,
            candidate_set_id=candidate_set_id,
            case_ids=case_ids,
            modality_a=left,
            modality_b=right,
            vectors_by_modality=vectors_by_modality,
        )
        for left, right in MODALITY_PAIRS
    ]


def tool_multimodal_consistency_check(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    memberships = candidate_set_members(cluster_state, all_cluster_states)
    final_case_ids = [case_id for members in memberships.values() for case_id in members]
    scoped_states = {
        case_id: patient_states_by_id[case_id]
        for case_id in final_case_ids
        if case_id in patient_states_by_id
    }
    vectors_by_modality = {
        name: get_modality_vectors(scoped_states, cfg["key"])
        for name, cfg in MODALITY_CONFIG.items()
    }
    global_rows = alignment_rows(
        scope="global",
        candidate_set_id=None,
        case_ids=final_case_ids,
        vectors_by_modality=vectors_by_modality,
    )
    set_rows = {
        set_id: alignment_rows(
            scope="candidate_set",
            candidate_set_id=set_id,
            case_ids=members,
            vectors_by_modality=vectors_by_modality,
        )
        for set_id, members in memberships.items()
    }

    return tool_result(
        tool_name="tool_multimodal_consistency_check",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary="Cross-modal distance/kernel alignment metrics were computed.",
        metrics={
            "crossmodal_global_alignment": global_rows,
            "crossmodal_set_alignment": set_rows,
        },
        evidence_hints=[
            {
                "evidence_type": "multimodal_consistency",
                "summary": "Cross-modal alignment is reported as complete modality-pair metric tables.",
            }
        ],
        support_level="informational",
        concern_level="none",
        figures={},
    )
