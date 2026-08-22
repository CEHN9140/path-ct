from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from tools.multimodal_consistency_check import normalize_affinity, round_value, separation_score


def structure_diagnostics(
    affinities: Mapping[str, np.ndarray],
    case_ids: list[str],
    memberships: Mapping[str, list[str]],
    *,
    max_k: int = 8,
) -> dict[str, Any]:
    """Measure internal structure without producing membership candidates."""
    diagnostics = {}
    fused = normalize_affinity(affinities["fused"])
    index = {case_id: position for position, case_id in enumerate(case_ids)}
    for set_name, members in sorted(memberships.items()):
        positions = np.asarray([index[case_id] for case_id in members], dtype=int)
        local = fused[np.ix_(positions, positions)]
        rows = []
        for k in range(2, min(max_k, len(members) - 1) + 1):
            labels = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="cluster_qr",
                random_state=0,
            ).fit_predict(local)
            if len(set(labels)) < 2:
                continue
            upper = np.triu_indices(len(labels), 1)
            same = labels[upper[0]] == labels[upper[1]]
            values = local[upper]
            within = float(values[same].mean()) if np.any(same) else 0.0
            between = float(values[~same].mean()) if np.any(~same) else 0.0
            separation = separation_score(within, between)
            silhouette = float(silhouette_score(1.0 - local, labels, metric="precomputed"))
            stability = []
            for seed in range(1, 6):
                repeat = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    assign_labels="cluster_qr",
                    random_state=seed,
                ).fit_predict(local)
                stability.append(adjusted_rand_score(labels, repeat))
            eigenvalues = np.linalg.eigvalsh(local)[::-1]
            eigengap = float(eigenvalues[k - 1] - eigenvalues[k]) if k < len(eigenvalues) else 0.0
            rows.append({
                "k": k,
                "separation": round_value(separation),
                "silhouette": round_value(silhouette),
                "stability": round_value(float(np.mean(stability))),
                "eigengap": round_value(eigengap),
                "positive_internal_heterogeneity": bool(
                    separation > 0 and silhouette > 0 and np.mean(stability) >= 0.6
                ),
            })
        positive = [row for row in rows if row["positive_internal_heterogeneity"]]
        best = max(
            positive or rows,
            key=lambda row: (
                bool(row["positive_internal_heterogeneity"]),
                float(row.get("stability") or 0),
                float(row.get("separation") or 0),
                float(row.get("silhouette") or 0),
            ),
            default=None,
        )
        diagnostics[set_name] = {
            "member_n": len(members),
            "candidate_k_diagnostics": rows,
            "positive_internal_heterogeneity": bool(positive),
            "recommended_k": best.get("k") if best else None,
        }

    pair_rows = {}
    weak_pairs = {}
    for left, right in combinations(sorted(memberships), 2):
        left_positions = np.asarray([index[case_id] for case_id in memberships[left]], dtype=int)
        right_positions = np.asarray([index[case_id] for case_id in memberships[right]], dtype=int)
        modality_rows = {}
        for modality, matrix in affinities.items():
            matrix = normalize_affinity(matrix)
            left_block = matrix[np.ix_(left_positions, left_positions)]
            right_block = matrix[np.ix_(right_positions, right_positions)]
            left_values = left_block[~np.eye(len(left_positions), dtype=bool)]
            right_values = right_block[~np.eye(len(right_positions), dtype=bool)]
            within_values = np.concatenate([left_values, right_values])
            within = float(within_values.mean()) if len(within_values) else 0.0
            between = float(matrix[np.ix_(left_positions, right_positions)].mean())
            modality_rows[modality] = {
                "within": round_value(within),
                "between": round_value(between),
                "boundary_separation": round_value(separation_score(within, between)),
                "weak_boundary": bool(between >= within),
            }
        key = f"{left}+{right}"
        supporting = [
            modality
            for modality, row in modality_rows.items()
            if modality != "fused" and row["weak_boundary"]
        ]
        pair_rows[key] = {"targets": [left, right], "modalities": modality_rows}
        weak_pairs[key] = supporting
    return {
        "internal_structure_by_set": diagnostics,
        "boundary_by_pair": pair_rows,
        "positive_weak_boundary_pairs": {
            key: modalities for key, modalities in weak_pairs.items() if len(modalities) >= 2
        },
    }


def execute_split_membership(
    output_root: str,
    member_ids: list[str],
    n_children: int,
    strategy: str,
) -> list[list[str]]:
    if strategy not in {"multimodal_consensus", "fused_similarity_spectral"}:
        raise ValueError(f"Unknown split execution strategy: {strategy}")
    if n_children < 2 or n_children > len(member_ids):
        raise ValueError("n_children must be between 2 and the source-set size")
    candidate_dir = Path(output_root) / "candidate_subtype"
    case_ids = json.loads((candidate_dir / "affinity_patient_order.json").read_text(encoding="utf-8"))
    positions = [case_ids.index(case_id) for case_id in member_ids]
    fused = normalize_affinity(np.load(candidate_dir / "fused_similarity.npy"))
    local = fused[np.ix_(positions, positions)]
    labels = SpectralClustering(
        n_clusters=n_children,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=0,
    ).fit_predict(local)
    groups = [sorted(member_ids[index] for index in np.flatnonzero(labels == label)) for label in sorted(set(labels))]
    if len(groups) != n_children or any(not group for group in groups):
        raise ValueError("Deterministic split did not produce the requested children")
    return groups
