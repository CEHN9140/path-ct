#!/usr/bin/env python3
"""Offline quality diagnostics for patient-resampled multi-K candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ("hierarchical", "spectral", "kmedoids")
PAIR_NAMES = (
    ("hierarchical", "spectral", "hierarchical_spectral_ari"),
    ("hierarchical", "kmedoids", "hierarchical_kmedoids_ari"),
    ("spectral", "kmedoids", "spectral_kmedoids_ari"),
)


def summarize_candidate_k(k, final, algorithm_matrices, labels, patient_ids, pac_lower, pac_upper):
    final = np.asarray(final, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if final.shape != (len(patient_ids), len(patient_ids)) or labels.shape != (len(patient_ids),):
        raise ValueError(f"K={k}: matrix, labels, and patient order have inconsistent dimensions")
    if not np.allclose(final, final.T, atol=1e-8):
        raise ValueError(f"K={k}: final consensus matrix is not symmetric")

    upper = final[np.triu_indices(len(patient_ids), k=1)]
    pac = float(np.mean((upper > pac_lower) & (upper < pac_upper)))
    distance = 1.0 - np.clip(final, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    silhouette = float(silhouette_score(distance, labels, metric="precomputed"))

    cluster_rows, item_rows, cluster_consensus = [], [], []
    for cluster_id in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cluster_id)
        members = final[np.ix_(indices, indices)]
        if len(indices) > 1:
            within = members[np.triu_indices(len(indices), k=1)]
            cluster_mean = float(within.mean())
            item_values = (members.sum(axis=1) - np.diag(members)) / (len(indices) - 1)
            cluster_consensus.append(cluster_mean)
        else:
            cluster_mean, item_values = None, np.array([np.nan])
        cluster_rows.append({
            "k": int(k),
            "cluster_id": int(cluster_id),
            "cluster_size": int(len(indices)),
            "within_cluster_consensus": cluster_mean,
        })
        for index, item_value in zip(indices, item_values):
            item_rows.append({
                "k": int(k),
                "case_id": patient_ids[int(index)],
                "cluster_id": int(cluster_id),
                "item_consensus": float(item_value) if np.isfinite(item_value) else None,
            })

    valid_items = [row["item_consensus"] for row in item_rows if row["item_consensus"] is not None]
    algorithm_labels = {}
    for algorithm, matrix in algorithm_matrices.items():
        values = np.asarray(matrix, dtype=float)
        if values.shape != final.shape or not np.allclose(values, values.T, atol=1e-8):
            raise ValueError(f"K={k}: invalid {algorithm} consensus matrix")
        algorithm_distance = 1.0 - np.clip(values, 0.0, 1.0)
        np.fill_diagonal(algorithm_distance, 0.0)
        algorithm_labels[algorithm] = cut_tree(
            linkage(squareform(algorithm_distance, checks=True), method="average"),
            n_clusters=[int(k)],
        ).reshape(-1)

    summary = {
        "k": int(k),
        "pac": pac,
        "consensus_silhouette": silhouette,
        "mean_cluster_consensus": float(np.mean(cluster_consensus)) if cluster_consensus else None,
        "min_cluster_consensus": float(np.min(cluster_consensus)) if cluster_consensus else None,
        "item_consensus_mean": float(np.mean(valid_items)) if valid_items else None,
        "item_consensus_p10": float(np.quantile(valid_items, 0.1)) if valid_items else None,
        "min_cluster_size": int(min(np.sum(labels == label) for label in np.unique(labels))),
    }
    for left, right, column in PAIR_NAMES:
        summary[column] = float(adjusted_rand_score(algorithm_labels[left], algorithm_labels[right]))
    return summary, cluster_rows, item_rows


def run(candidate_root, output_root, config_dir, force=False):
    candidate_root, output_root, config_dir = map(Path, (candidate_root, output_root, config_dir))
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    consensus_dir = candidate_root / "consensus_cluster"
    patient_ids = json.loads((candidate_root / "affinity_patient_order.json").read_text(encoding="utf-8"))
    clustering = yaml.safe_load((config_dir / "candidate_proposer.yaml").read_text(encoding="utf-8"))["clustering"]
    records, cluster_rows, item_rows = [], [], []

    for k in range(2, 9):
        partition = json.loads((consensus_dir / f"consensus_hierarchical_K{k}.json").read_text(encoding="utf-8"))
        if set(partition["labels"]) != set(patient_ids):
            raise ValueError(f"K={k}: candidate labels do not match the saved patient order")
        labels = np.array([partition["labels"][case_id] for case_id in patient_ids], dtype=int)
        final = np.load(consensus_dir / f"consensus_matrix_K{k}.npy")
        algorithm_matrices = {
            name: np.load(consensus_dir / f"{name}_consensus_matrix_K{k}.npy")
            for name in ALGORITHMS
        }
        summary, clusters, items = summarize_candidate_k(
            k, final, algorithm_matrices, labels, patient_ids,
            float(clustering["pac_lower"]), float(clustering["pac_upper"]),
        )
        records.append(summary)
        cluster_rows.extend(clusters)
        item_rows.extend(items)

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_root / "candidate_quality_by_k.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(output_root / "candidate_cluster_consensus.csv", index=False)
    pd.DataFrame(item_rows).to_csv(output_root / "candidate_item_consensus.csv", index=False)
    manifest = {
        "experiment": "offline_patient_resampled_candidate_consensus_diagnostics",
        "candidate_root": str(candidate_root.resolve()),
        "patient_count": len(patient_ids),
        "k_values": list(range(2, 9)),
        "pac_interval": [float(clustering["pac_lower"]), float(clustering["pac_upper"])],
        "metrics": {
            "pac": "fraction of upper-triangle off-diagonal final-consensus values strictly inside the configured PAC interval",
            "consensus_silhouette": "silhouette on precomputed distance 1 - final equal-weight consensus, using saved final candidate labels",
            "cluster_consensus": "mean off-diagonal coassignment within each final cluster; K summary is the unweighted mean and minimum across clusters",
            "item_consensus": "per-patient mean coassignment with other members of its assigned final cluster; singleton clusters are missing",
            "algorithm_agreement": "pairwise ARI after average-linkage clustering of each algorithm-specific consensus matrix at the same K",
        },
        "agent_review_run": False,
        "resampling_recomputed": False,
    }
    (output_root / "diagnostics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root", type=Path,
        default=ROOT / "output_kirc_v15/01_four_view_feature_engineering/inputs/four_view_feature_engineering/candidate_subtype",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v15/02_candidate_consensus_diagnostics")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.candidate_root, args.output_root, args.config_dir, args.force), indent=2))


if __name__ == "__main__":
    main()
