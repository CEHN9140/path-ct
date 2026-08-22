from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from tools.multimodal_consistency_check import normalize_affinity, round_value, separation_score


MODALITIES = ("fused", "ct", "wsi", "rna", "genomic")
INDEPENDENT_MODALITIES = MODALITIES[1:]


def modality_metrics(matrix: np.ndarray, k: int) -> dict[str, Any]:
    labels = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=0,
    ).fit_predict(matrix)
    upper = np.triu_indices(len(labels), 1)
    same = labels[upper[0]] == labels[upper[1]]
    values = matrix[upper]
    within = float(values[same].mean()) if np.any(same) else 0.0
    between = float(values[~same].mean()) if np.any(~same) else 0.0
    distance = 1.0 - matrix
    silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
    subsampling_scores = []
    sample_size = max(k + 1, int(round(len(labels) * 0.8)))
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(len(labels), size=min(sample_size, len(labels)), replace=False))
        if len(sample) <= k:
            subsampling_scores.append(1.0)
            continue
        repeat = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=0,
        ).fit_predict(matrix[np.ix_(sample, sample)])
        subsampling_scores.append(adjusted_rand_score(labels[sample], repeat))
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    return {
        "separation": round_value(separation_score(within, between)),
        "silhouette": round_value(silhouette),
        "subsampling_stability": round_value(float(np.mean(subsampling_scores))),
        "eigengap": round_value(float(eigenvalues[k - 1] - eigenvalues[k])),
        "within_affinity": round_value(within),
        "between_affinity": round_value(between),
    }


def structure_diagnostics(
    affinities: Mapping[str, np.ndarray],
    case_ids: list[str],
    memberships: Mapping[str, list[str]],
) -> dict[str, Any]:
    normalized = {
        modality: normalize_affinity(matrix)
        for modality, matrix in affinities.items()
    }
    if "fused" not in normalized:
        normalized["fused"] = normalize_affinity(
            np.mean([normalized[modality] for modality in INDEPENDENT_MODALITIES], axis=0)
        )
    index = {case_id: position for position, case_id in enumerate(case_ids)}
    diagnostics = {}
    for set_name, members in sorted(memberships.items()):
        positions = np.asarray([index[case_id] for case_id in members], dtype=int)
        rows = []
        for k in range(2, len(members)):
            modality_rows = {}
            for modality in MODALITIES:
                local = normalized[modality][np.ix_(positions, positions)]
                modality_rows[modality] = modality_metrics(local, k)
            positive_modalities = [
                modality
                for modality in INDEPENDENT_MODALITIES
                if modality_rows[modality]["separation"] is not None
                and modality_rows[modality]["separation"] > 0
                and modality_rows[modality]["silhouette"] is not None
                and modality_rows[modality]["silhouette"] > 0
                and modality_rows[modality]["subsampling_stability"] is not None
                and modality_rows[modality]["subsampling_stability"] >= 0.6
            ]
            rows.append({
                "k": k,
                "modalities": modality_rows,
                "positive_modalities": positive_modalities,
                "positive_internal_heterogeneity": len(positive_modalities) >= 2,
            })
        diagnostics[set_name] = {
            "member_n": len(members),
            "k_diagnostics": rows,
            "positive_internal_heterogeneity": any(
                row["positive_internal_heterogeneity"] for row in rows
            ),
        }

    pair_rows = {}
    weak_pairs = {}
    for left, right in combinations(sorted(memberships), 2):
        left_positions = np.asarray([index[case_id] for case_id in memberships[left]], dtype=int)
        right_positions = np.asarray([index[case_id] for case_id in memberships[right]], dtype=int)
        modality_rows = {}
        for modality in MODALITIES:
            matrix = normalized[modality]
            left_block = matrix[np.ix_(left_positions, left_positions)]
            right_block = matrix[np.ix_(right_positions, right_positions)]
            left_values = left_block[~np.eye(len(left_positions), dtype=bool)]
            right_values = right_block[~np.eye(len(right_positions), dtype=bool)]
            within = float(np.concatenate([left_values, right_values]).mean())
            between = float(matrix[np.ix_(left_positions, right_positions)].mean())
            modality_rows[modality] = {
                "within": round_value(within),
                "between": round_value(between),
                "boundary_separation": round_value(separation_score(within, between)),
                "weak_boundary": bool(between >= within),
            }
        key = f"{left}+{right}"
        supporting = [
            modality for modality in INDEPENDENT_MODALITIES
            if modality_rows[modality]["weak_boundary"]
        ]
        pair_rows[key] = {"targets": [left, right], "modalities": modality_rows}
        weak_pairs[key] = supporting
    return {
        "internal_structure_by_set": diagnostics,
        "boundary_by_pair": pair_rows,
        "positive_weak_boundary_pairs": {
            key: modalities for key, modalities in weak_pairs.items()
            if len(modalities) >= 2
        },
    }


def execute_split_membership(
    output_root: str,
    member_ids: list[str],
    n_children: int,
    strategy: str,
    structural_basis: list[str],
) -> list[list[str]]:
    if n_children < 2 or n_children > len(member_ids):
        raise ValueError("n_children must be between 2 and the source-set size")
    basis = sorted(set(structural_basis))
    if strategy == "fused_similarity_spectral":
        if basis != ["fused"]:
            raise ValueError("fused_similarity_spectral requires structural_basis=['fused']")
    elif strategy == "multimodal_consensus":
        if len(basis) < 2 or any(modality not in INDEPENDENT_MODALITIES for modality in basis):
            raise ValueError("multimodal_consensus requires at least two independent modalities")
    else:
        raise ValueError(f"Unknown split execution strategy: {strategy}")

    candidate_dir = Path(output_root) / "candidate_subtype"
    case_ids = json.loads((candidate_dir / "affinity_patient_order.json").read_text(encoding="utf-8"))
    positions = [case_ids.index(case_id) for case_id in member_ids]
    paths = {
        "fused": candidate_dir / "fused_similarity.npy",
        "ct": candidate_dir / "ct_affinity.npy",
        "wsi": candidate_dir / "wsi_affinity.npy",
        "rna": candidate_dir / "rna_affinity.npy",
        "genomic": Path(output_root) / "wxs" / "genomic_affinity.npy",
    }
    matrices = [normalize_affinity(np.load(paths[modality])) for modality in basis]
    local = np.mean([matrix[np.ix_(positions, positions)] for matrix in matrices], axis=0)
    labels = SpectralClustering(
        n_clusters=n_children,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=0,
    ).fit_predict(local)
    groups = [
        sorted(member_ids[index] for index in np.flatnonzero(labels == label))
        for label in sorted(set(labels))
    ]
    if len(groups) != n_children or any(not group for group in groups):
        raise ValueError("Deterministic split did not produce the requested children")
    return groups
