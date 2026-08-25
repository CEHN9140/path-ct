from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from tools.multimodal_consistency_check import (
    fixed_membership_silhouettes,
    normalize_affinity,
    round_value,
)


MODALITIES = ("ct", "wsi", "rna", "genomic")


def binary_spectral_probe(similarity: np.ndarray) -> np.ndarray:
    matrix = normalize_affinity(similarity)
    if len(matrix) < 2:
        raise ValueError("Binary probe requires at least two patients")
    return SpectralClustering(
        n_clusters=2,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=0,
    ).fit_predict(matrix)


def normalized_cut(similarity: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) != 2:
        return None
    weights = normalize_affinity(similarity).copy()
    np.fill_diagonal(weights, 0.0)
    cut = sum(
        float(weights[np.ix_(labels == left, labels == right)].sum())
        for left, right in [(0, 1), (1, 0)]
    ) / 2.0
    values = []
    for label in np.unique(labels):
        indices = labels == label
        association = float(weights[indices].sum())
        if association <= 0:
            return None
        values.append(cut / association)
    return round_value(sum(values))


def probe_silhouette(similarity: np.ndarray, labels: np.ndarray, case_ids: list[str]) -> dict[str, Any]:
    values = fixed_membership_silhouettes(1.0 - normalize_affinity(similarity), labels)
    by_case = {case_ids[int(index)]: value for index, value in values.items()}
    available = [value for value in by_case.values() if value is not None]
    return {
        "median_silhouette": round_value(np.median(available)) if available else None,
        "mean_silhouette": round_value(np.mean(available)) if available else None,
        "fraction_silhouette_positive": round_value(
            np.mean(np.asarray(available) > 0)
        ) if available else None,
        "patient_silhouette": by_case,
        "comparison_status": "estimable" if available else "not_estimable",
        "not_estimable_reason": "insufficient_set_size" if not available else "",
    }


def resampling_consensus(
    similarity: np.ndarray,
    full_labels: np.ndarray,
    *,
    fraction: float = 0.8,
    iterations: int = 200,
    pac_lower: float = 0.1,
    pac_upper: float = 0.9,
    seed: int = 0,
) -> dict[str, Any]:
    n = len(full_labels)
    sample_size = int(np.floor(n * fraction))
    if sample_size < 4:
        return {
            "iterations": int(iterations),
            "sample_fraction": fraction,
            "median_resample_ari": None,
            "consensus_separation": None,
            "pac": None,
            "degenerate_resample_fraction": None,
            "comparison_status": "not_estimable",
            "not_estimable_reason": "subsample_too_small_for_binary_probe",
        }
    rng = np.random.default_rng(seed)
    sampled = np.zeros((n, n), dtype=int)
    co_clustered = np.zeros((n, n), dtype=int)
    aris = []
    degenerate = 0
    matrix = normalize_affinity(similarity)
    for _ in range(iterations):
        indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        labels = binary_spectral_probe(matrix[np.ix_(indices, indices)])
        counts = np.bincount(labels, minlength=2)
        if min(counts) < 2:
            degenerate += 1
        aris.append(adjusted_rand_score(full_labels[indices], labels))
        for left, right in combinations(indices, 2):
            sampled[left, right] += 1
            sampled[right, left] += 1
            if labels[np.where(indices == left)[0][0]] == labels[np.where(indices == right)[0][0]]:
                co_clustered[left, right] += 1
                co_clustered[right, left] += 1
    valid = sampled > 0
    consensus = np.divide(
        co_clustered,
        sampled,
        out=np.zeros_like(co_clustered, dtype=float),
        where=valid,
    )
    same = (full_labels[:, None] == full_labels[None, :]) & valid
    different = (full_labels[:, None] != full_labels[None, :]) & valid
    same_values = consensus[same]
    different_values = consensus[different]
    observed = consensus[valid]
    pac = np.mean((observed > pac_lower) & (observed < pac_upper)) if len(observed) else None
    return {
        "iterations": int(iterations),
        "sample_fraction": fraction,
        "median_resample_ari": round_value(np.median(aris)) if aris else None,
        "consensus_separation": round_value(
            np.mean(same_values) - np.mean(different_values)
            if len(same_values) and len(different_values) else None
        ),
        "pac": round_value(pac),
        "degenerate_resample_fraction": round_value(degenerate / iterations) if iterations else None,
        "comparison_status": "estimable",
        "not_estimable_reason": "",
    }


def fixed_probe_support(
    similarity: np.ndarray | None,
    labels: np.ndarray,
    case_ids: list[str],
) -> dict[str, Any]:
    if similarity is None:
        return {
            "comparison_status": "scientific_unavailable",
            "not_estimable_reason": "missing_affinity_matrix",
            "median_silhouette": None,
        }
    return probe_silhouette(similarity, labels, case_ids)


def characterize_internal_structure(
    fused_similarity: np.ndarray | None,
    modality_affinities: Mapping[str, np.ndarray | None],
    case_ids: list[str],
    members: list[str],
    *,
    resampling_fraction: float = 0.8,
    resampling_iterations: int = 200,
    pac_lower: float = 0.1,
    pac_upper: float = 0.9,
    random_seed: int = 0,
) -> dict[str, Any]:
    member_positions = [case_ids.index(case_id) for case_id in members]
    supports: dict[str, Any] = {}
    if fused_similarity is None:
        return {
            "comparison_status": "scientific_unavailable",
            "member_n": len(members),
            "not_estimable_reason": "missing_fused_affinity_matrix",
            "probe_support_by_modality": {
                modality: {"comparison_status": "scientific_unavailable", "not_estimable_reason": "missing_fused_affinity_matrix"}
                for modality in MODALITIES
            },
            "limitations": ["The actual SNF fused similarity is unavailable."],
        }
    local_fused = normalize_affinity(fused_similarity)[np.ix_(member_positions, member_positions)]
    if len(members) < 4:
        return {
            "comparison_status": "not_estimable",
            "member_n": len(members),
            "not_estimable_reason": "degenerate_binary_probe",
            "probe_support_by_modality": {},
            "limitations": ["A binary probe cannot produce two estimable children for this set."],
        }
    labels = binary_spectral_probe(local_fused)
    child_sizes = sorted(np.bincount(labels, minlength=2).tolist())
    if child_sizes[0] < 2:
        status = "not_estimable"
        probe = {
            "child_sizes": child_sizes,
            "comparison_status": status,
            "not_estimable_reason": "degenerate_binary_probe",
        }
        return {
            "comparison_status": status,
            "member_n": len(members),
            "fused_binary_probe": probe,
            "probe_support_by_modality": {},
            "limitations": ["The binary probe produced a singleton child."],
        }
    probe = probe_silhouette(local_fused, labels, members)
    probe.update({
        "child_sizes": child_sizes,
        "normalized_cut": normalized_cut(local_fused, labels),
        "probe_labels_by_case": {case_id: int(label) for case_id, label in zip(members, labels)},
        "resampling": resampling_consensus(
            local_fused,
            labels,
            fraction=resampling_fraction,
            iterations=resampling_iterations,
            pac_lower=pac_lower,
            pac_upper=pac_upper,
            seed=random_seed,
        ),
    })
    for modality in MODALITIES:
        matrix = modality_affinities.get(modality)
        local = None if matrix is None else matrix[np.ix_(member_positions, member_positions)]
        supports[modality] = fixed_probe_support(local, labels, members)
    return {
        "comparison_status": "estimable",
        "member_n": len(members),
        "fused_binary_probe": probe,
        "probe_support_by_modality": supports,
        "limitations": [
            "Resampling was performed on the existing SNF affinity subgraph; modality affinities and SNF were not re-estimated."
        ],
    }


def pair_boundary_metrics(
    similarity: np.ndarray | None,
    case_ids: list[str],
    left_members: list[str],
    right_members: list[str],
) -> dict[str, Any]:
    if similarity is None:
        return {"comparison_status": "scientific_unavailable", "not_estimable_reason": "missing_affinity_matrix"}
    members = [*left_members, *right_members]
    labels = np.asarray([0] * len(left_members) + [1] * len(right_members))
    positions = [case_ids.index(case_id) for case_id in members]
    local = normalize_affinity(similarity)[np.ix_(positions, positions)]
    silhouette = probe_silhouette(local, labels, members)
    rows = {}
    for side, own, other in (("left", left_members, right_members), ("right", right_members, left_members)):
        own_indices = [members.index(case_id) for case_id in own]
        other_indices = [members.index(case_id) for case_id in other]
        own_values = local[np.ix_(own_indices, own_indices)]
        own_values = own_values[~np.eye(len(own_indices), dtype=bool)]
        between = local[np.ix_(own_indices, other_indices)]
        margins = []
        for index in own_indices:
            own_mean = np.mean([local[index, other_index] for other_index in own_indices if other_index != index]) if len(own_indices) > 1 else None
            margins.append(None if own_mean is None else float(own_mean - local[index, other_indices].mean()))
        margins = [value for value in margins if value is not None]
        within = float(own_values.mean()) if len(own_values) else None
        between_mean = float(between.mean()) if len(between) else None
        rows[side] = {
            "within_affinity": round_value(within),
            "between_affinity": round_value(between_mean),
            "median_margin": round_value(np.median(margins)) if margins else None,
            "fraction_margin_positive": round_value(np.mean(np.asarray(margins) > 0)) if margins else None,
            "boundary_separation": round_value(
                (within - between_mean) / (abs(within) + abs(between_mean))
                if within is not None and between_mean is not None and abs(within) + abs(between_mean) else None
            ),
            "patient_margins": {case_id: round_value(value) for case_id, value in zip(own, margins)},
        }
    return {
        "comparison_status": (
            "estimable"
            if all(value is not None for value in silhouette["patient_silhouette"].values())
            else "partially_estimable"
        ),
        "not_estimable_reason": "singleton_or_insufficient_side" if any(
            value is None for value in silhouette["patient_silhouette"].values()
        ) else "",
        "pair_median_silhouette": silhouette["median_silhouette"],
        "patient_silhouette": silhouette["patient_silhouette"],
        "left_median_margin": rows["left"]["median_margin"],
        "right_median_margin": rows["right"]["median_margin"],
        "left_fraction_margin_positive": rows["left"]["fraction_margin_positive"],
        "right_fraction_margin_positive": rows["right"]["fraction_margin_positive"],
        "left_within_affinity": rows["left"]["within_affinity"],
        "right_within_affinity": rows["right"]["within_affinity"],
        "between_affinity": rows["left"]["between_affinity"],
        "left_boundary_separation": rows["left"]["boundary_separation"],
        "right_boundary_separation": rows["right"]["boundary_separation"],
        "patient_margins": {**rows["left"]["patient_margins"], **rows["right"]["patient_margins"]},
    }


def compute_structural_characterization(
    fused_similarity: np.ndarray | None,
    modality_affinities: Mapping[str, np.ndarray | None],
    case_ids: list[str],
    memberships: Mapping[str, list[str]],
    *,
    resampling_fraction: float = 0.8,
    resampling_iterations: int = 200,
    pac_lower: float = 0.1,
    pac_upper: float = 0.9,
    random_seed: int = 0,
) -> dict[str, Any]:
    internal = {
        set_id: characterize_internal_structure(
            fused_similarity,
            modality_affinities,
            case_ids,
            members,
            resampling_fraction=resampling_fraction,
            resampling_iterations=resampling_iterations,
            pac_lower=pac_lower,
            pac_upper=pac_upper,
            random_seed=random_seed,
        )
        for set_id, members in memberships.items()
    }
    boundaries = {}
    for left, right in combinations(sorted(memberships), 2):
        pair = {"targets": [left, right], "fused": pair_boundary_metrics(fused_similarity, case_ids, memberships[left], memberships[right])}
        pair["modalities"] = {
            modality: pair_boundary_metrics(
                modality_affinities.get(modality), case_ids, memberships[left], memberships[right]
            )
            for modality in MODALITIES
        }
        boundaries[f"{left}+{right}"] = pair
    return {
        "internal_structure_by_set": internal,
        "boundary_by_pair": boundaries,
        "limitations": [
            "Structural characterization describes the current partition and does not independently validate biological subtype identity."
        ],
    }


def execute_split_membership(
    output_root: str,
    member_ids: list[str],
    n_children: int,
    strategy: str,
    structural_basis: list[str],
) -> list[list[str]]:
    if n_children != 2 or strategy != "fused_similarity_spectral" or structural_basis != ["fused"]:
        raise ValueError("Split execution requires n_children=2, structural_basis=['fused'], and fused_similarity_spectral")
    candidate = Path(output_root) / "candidate_subtype"
    case_ids = json.loads((candidate / "affinity_patient_order.json").read_text(encoding="utf-8"))
    positions = [case_ids.index(case_id) for case_id in member_ids]
    matrix = normalize_affinity(np.load(candidate / "fused_similarity.npy"))
    labels = binary_spectral_probe(matrix[np.ix_(positions, positions)])
    groups = [
        sorted(member_ids[index] for index in np.flatnonzero(labels == label))
        for label in sorted(set(labels))
    ]
    if len(groups) != 2 or any(len(group) < 2 for group in groups):
        raise ValueError("Deterministic binary split produced a non-estimable child")
    return groups
