from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools.subtype_review_common import (
    bh_fdr,
    scoped_candidate_sets,
    tool_result,
)


MODALITIES = ("ct", "wsi", "rna", "genomic")
DEFAULT_EFFECT_EPSILON = 0.001


def modality_affinity_path(output_root: str, modality: str) -> Path:
    """Return the canonical saved affinity path for a review modality."""
    if modality == "genomic":
        return Path(output_root) / "wxs" / "genomic_affinity.npy"
    return Path(output_root) / "candidate_subtype" / f"{modality}_affinity.npy"


def round_value(value: float) -> float | None:
    return round(float(value), 6) if np.isfinite(value) else None


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
        margins = []
        for patient in inside:
            own = float(np.delete(similarity[patient, inside], np.where(inside == patient)[0][0]).mean())
            other = max(
                float(similarity[patient, np.flatnonzero(labels == other_set)].mean())
                for other_set in np.unique(labels)
                if other_set != set_id
            )
            margins.append(own - other)
        per_set[str(set_id)] = {
            "member_n": int(len(inside)),
            "mean_within_affinity": round_value(set_within),
            "nearest_other_set": nearest,
            "mean_nearest_between_affinity": round_value(set_between),
            "normalized_affinity_separation": round_value(
                separation_score(set_within, set_between)
            ),
            "mean_affinity_margin": round_value(float(np.mean(margins))),
            "median_affinity_margin": round_value(float(np.median(margins))),
            "fraction_affinity_margin_positive": round_value(float(np.mean(np.asarray(margins) > 0))),
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
    permutations: int = 199,
    seed: int = 0,
) -> dict[str, Any]:
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    gower = gower_matrix(distance)
    r2, pseudo_f = permanova_statistic(gower, labels)
    rng = np.random.default_rng(seed)
    null = [permanova_statistic(gower, rng.permutation(labels))[0] for _ in range(permutations)]
    p_value = (1 + sum(value >= r2 for value in null)) / (permutations + 1)
    return {
        "available_n": len(labels),
        "set_count": int(len(np.unique(labels))),
        "permanova_r2": round_value(r2),
        "pseudo_f": round_value(pseudo_f),
        "permanova_p_value": round_value(p_value),
        "permutations": int(permutations),
    }


def compute_cross_modal_consistency(
    affinities: Mapping[str, np.ndarray],
    case_ids: list[str],
    memberships: Mapping[str, list[str]],
    *,
    min_per_set_separation: float | None = None,
    permanova_permutations: int = 199,
    effect_epsilon: float = DEFAULT_EFFECT_EPSILON,
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
        distance = 1.0 - similarity
        np.fill_diagonal(distance, 0.0)
        if len(np.unique(labels)) > 1 and min(np.sum(labels == label) for label in np.unique(labels)) > 1:
            from sklearn.metrics import silhouette_samples
            silhouettes = silhouette_samples(distance, labels, metric="precomputed")
            global_row["mean_silhouette"] = round_value(float(np.mean(silhouettes)))
            global_row["median_silhouette"] = round_value(float(np.median(silhouettes)))
            global_row["fraction_silhouette_positive"] = round_value(float(np.mean(silhouettes > 0)))
            for label in np.unique(labels):
                per_set[str(label)]["mean_silhouette"] = round_value(float(np.mean(silhouettes[labels == label])))
                per_set[str(label)]["median_silhouette"] = round_value(float(np.median(silhouettes[labels == label])))
                per_set[str(label)]["fraction_silhouette_positive"] = round_value(float(np.mean(silhouettes[labels == label] > 0)))
        global_row.update(
            {
                **permanova_metrics(similarity, labels, permanova_permutations, MODALITIES.index(modality)),
                "per_set": per_set,
            }
        )
        rows[modality] = global_row
    permanova_rows = [rows[modality] for modality in MODALITIES]
    for row, q_value in zip(
        permanova_rows,
        bh_fdr([float(row.get("permanova_p_value", 1.0) or 1.0) for row in permanova_rows]),
    ):
        row["permanova_q_value"] = round_value(q_value)
    modality_flags = {}
    identity_modalities_by_set = {set_id: [] for set_id in memberships}
    for modality, row in rows.items():
        per_set = row.get("per_set", {})
        weak_boundary = all(
            float(values.get("median_silhouette", 0) or 0) <= 0
            and float(values.get("median_affinity_margin", 0) or 0) <= 0
            for values in per_set.values()
        )
        strong_boundary = all(
            float(values.get("median_silhouette", 0) or 0) > effect_epsilon
            and float(values.get("median_affinity_margin", 0) or 0) > effect_epsilon
            for values in per_set.values()
        )
        modality_flags[modality] = {
            "identity_support": all(
                float(values.get(key, 0) or 0) > effect_epsilon
                for values in per_set.values()
                for key in ("median_silhouette", "normalized_affinity_separation")
            ) and all(
                float(values.get("fraction_affinity_margin_positive", 0) or 0) > 0.5
                for values in per_set.values()
            ),
            "split_support": (
                float(row.get("normalized_affinity_separation", 0) or 0) > effect_epsilon
                and float(row.get("permanova_q_value", 1) or 1) <= 0.05
                and all(float(values.get("median_silhouette", 0) or 0) > effect_epsilon for values in per_set.values())
            ),
            "merge_support": weak_boundary,
            "merge_strong_boundary": strong_boundary,
        }
        for set_id, values in per_set.items():
            if (
                float(values.get("median_silhouette", 0) or 0) > effect_epsilon
                and float(values.get("normalized_affinity_separation", 0) or 0) > effect_epsilon
                and float(values.get("fraction_affinity_margin_positive", 0) or 0) > 0.5
            ):
                identity_modalities_by_set.setdefault(set_id, []).append(modality)
    return {
        "modality_partition_support": rows,
        "decision_metrics": {
            "modality_flags": modality_flags,
            "identity_supporting_modalities": [modality for modality, flags in modality_flags.items() if flags["identity_support"]],
            "identity_supporting_modalities_by_set": identity_modalities_by_set,
            "split_supporting_modalities": [modality for modality, flags in modality_flags.items() if flags["split_support"]],
            "merge_supporting_modalities": [modality for modality, flags in modality_flags.items() if flags["merge_support"]],
            "merge_strong_boundary_modalities": [modality for modality, flags in modality_flags.items() if flags["merge_strong_boundary"]],
            "effect_epsilon": effect_epsilon,
        },
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
    scope="set_identity",
    target_ids=None,
    proposal=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "GLOBAL"))
    memberships = {
        key: sorted(value)
        for key, value in scoped_candidate_sets(scope, cluster_state, all_cluster_states, proposal).items()
    }
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
    parameters = {}
    if config_dir:
        from tools.subtype_review_common import tool_parameters
        parameters = tool_parameters(config_dir, "cross_modal")
    metrics = compute_cross_modal_consistency(
        affinities,
        case_ids,
        memberships,
        permanova_permutations=int(parameters.get("permanova_permutations", 199)),
        effect_epsilon=float(parameters.get("effect_epsilon", DEFAULT_EFFECT_EPSILON)),
    )
    return tool_result(
        tool_name="tool_multimodal_consistency_check",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary="Fixed candidate memberships were evaluated in four modality affinity networks.",
        metrics=metrics,
        decision_metrics=metrics.get("decision_metrics", {}),
        support_level="informational",
        concern_level="none",
        figures={},
    )
