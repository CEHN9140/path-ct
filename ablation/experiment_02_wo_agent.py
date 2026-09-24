#!/usr/bin/env python3
"""Deterministic conventional consensus-clustering baseline without Agent review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
PAC_INTERVAL = (0.1, 0.9)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_inputs(output_root: Path) -> tuple[list[str], dict, np.ndarray, dict[int, dict]]:
    candidate_root = output_root / "candidate_subtype"
    consensus_root = candidate_root / "consensus_cluster"
    order_path = candidate_root / "affinity_patient_order.json"
    fused_distance_path = candidate_root / "fused_distance.npy"
    manifest_path = consensus_root / "resampling_manifest.json"
    patient_ids = [str(value) for value in json.loads(order_path.read_text(encoding="utf-8"))]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Patient order contains duplicate IDs.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_ks = tuple(int(value) for value in manifest.get("candidate_ks", []))
    if candidate_ks != tuple(range(2, 9)):
        raise ValueError(f"Expected candidate K=2..8, got {candidate_ks}.")
    manifest_ids = [str(value) for value in manifest.get("patient_ids", [])]
    if manifest_ids != patient_ids:
        raise ValueError("Manifest patient order does not match affinity patient order.")

    fused_distance = np.asarray(np.load(fused_distance_path), dtype=float)
    n_patients = len(patient_ids)
    if fused_distance.shape != (n_patients, n_patients):
        raise ValueError("Fused distance matrix shape does not match patient order.")
    if not np.isfinite(fused_distance).all():
        raise ValueError("Fused distance matrix contains non-finite values.")
    if np.any(fused_distance < 0) or not np.allclose(fused_distance, fused_distance.T):
        raise ValueError("Fused distance matrix must be finite, non-negative, and symmetric.")
    if not np.allclose(np.diag(fused_distance), 0):
        raise ValueError("Fused distance matrix diagonal must be approximately zero.")

    partitions = {}
    for k in candidate_ks:
        matrix_path = consensus_root / f"consensus_matrix_K{k}.npy"
        labels_path = consensus_root / f"consensus_hierarchical_K{k}.json"
        consensus = np.asarray(np.load(matrix_path), dtype=float)
        if consensus.shape != (n_patients, n_patients):
            raise ValueError(f"K={k} consensus matrix shape is invalid.")
        if not np.isfinite(consensus).all():
            raise ValueError(f"K={k} consensus matrix contains non-finite values.")
        if not np.allclose(consensus, consensus.T) or not np.allclose(np.diag(consensus), 1):
            raise ValueError(f"K={k} consensus matrix must be symmetric with diagonal one.")

        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        labels_by_id = {str(key): int(value) for key, value in payload["labels"].items()}
        if int(payload.get("initial_k")) != k or set(labels_by_id) != set(patient_ids):
            raise ValueError(f"K={k} labels do not match the candidate patient order.")
        labels = np.asarray([labels_by_id[patient_id] for patient_id in patient_ids])
        if len(np.unique(labels)) != k:
            raise ValueError(f"K={k} labels contain {len(np.unique(labels))} clusters.")
        partitions[k] = {"consensus": consensus, "labels": labels, "labels_by_id": labels_by_id}
    return patient_ids, manifest, fused_distance, partitions


def compute_k_metrics(k: int, partition: dict, fused_distance: np.ndarray) -> dict:
    consensus = partition["consensus"]
    labels = partition["labels"]
    upper = np.triu_indices(len(labels), k=1)
    pair_values = consensus[upper]
    pac = float(np.mean((pair_values > PAC_INTERVAL[0]) & (pair_values < PAC_INTERVAL[1])))
    same = labels[:, None] == labels[None, :]
    within = same[upper]
    if not within.any() or (~within).sum() == 0:
        raise ValueError(f"K={k} does not provide both within- and between-cluster pairs.")
    within_mean = float(pair_values[within].mean())
    between_mean = float(pair_values[~within].mean())
    sizes = np.unique(labels, return_counts=True)[1]
    return {
        "k": k,
        "pac": pac,
        "silhouette_fused": float(silhouette_score(fused_distance, labels, metric="precomputed")),
        "mean_within_consensus": within_mean,
        "mean_between_consensus": between_mean,
        "consensus_gap": within_mean - between_mean,
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cluster_size_mean": float(sizes.mean()),
        "cluster_size_std": float(sizes.std()),
    }


def select_k(metrics: list[dict]) -> int:
    minimum_pac = min(row["pac"] for row in metrics)
    pac_ties = [row for row in metrics if np.isclose(row["pac"], minimum_pac, rtol=1e-12, atol=1e-12)]
    maximum_silhouette = max(row["silhouette_fused"] for row in pac_ties)
    silhouette_ties = [
        row for row in pac_ties
        if np.isclose(row["silhouette_fused"], maximum_silhouette, rtol=1e-12, atol=1e-12)
    ]
    return min(row["k"] for row in silhouette_ties)


def build_final_subtypes(selected_k: int, partition: dict, patient_ids: list[str]) -> tuple[list[dict], list[dict]]:
    labels = partition["labels"]
    final_subtypes = []
    membership = []
    for index, label in enumerate(sorted(np.unique(labels)), 1):
        members = [patient_id for patient_id, item in zip(patient_ids, labels) if item == label]
        subtype_id = f"WO_AGENT_SUBTYPE{index:02d}"
        final_subtypes.append({
            "subtype_id": subtype_id,
            "source_set_id": f"K{selected_k}_C{index:04d}",
            "selected_k": selected_k,
            "member_count": len(members),
            "member_ids": members,
        })
        membership.extend({"patient_id": patient_id, "subtype_id": subtype_id} for patient_id in members)
    if sorted(row["patient_id"] for row in membership) != sorted(patient_ids):
        raise ValueError("Final subtype membership does not cover the candidate cohort exactly once.")
    return final_subtypes, membership


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "ablation/results/02_wo_agent")
    args = parser.parse_args()

    patient_ids, manifest, fused_distance, partitions = load_and_validate_inputs(args.output_root)
    metrics = [compute_k_metrics(k, partitions[k], fused_distance) for k in sorted(partitions)]
    selected_k = select_k(metrics)
    for row in metrics:
        row["selected"] = row["k"] == selected_k
    final_subtypes, membership = build_final_subtypes(selected_k, partitions[selected_k], patient_ids)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = args.output_root / "candidate_subtype"
    consensus_root = candidate_root / "consensus_cluster"
    input_paths = [
        candidate_root / "affinity_patient_order.json",
        candidate_root / "fused_distance.npy",
        consensus_root / "resampling_manifest.json",
        *[consensus_root / f"consensus_matrix_K{k}.npy" for k in partitions],
        *[consensus_root / f"consensus_hierarchical_K{k}.json" for k in partitions],
    ]
    selected_metrics = next(row for row in metrics if row["selected"])
    pd.DataFrame(metrics).to_csv(args.result_dir / "k_selection_metrics.csv", index=False)
    (args.result_dir / "selected_k.json").write_text(json.dumps({
        "analysis": "wo_agent_consensus_clustering",
        "selection_method": "minimum_PAC",
        "pac_interval": list(PAC_INTERVAL),
        "selected_k": selected_k,
        "tie_break_rule": "higher_fused_silhouette_then_smaller_k",
        "candidate_ks": sorted(partitions),
        "input_manifest": str(consensus_root / "resampling_manifest.json"),
        "input_sha256": {str(path.relative_to(args.output_root)): sha256(path) for path in input_paths},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.result_dir / "final_subtypes.json").write_text(json.dumps(final_subtypes, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(membership).to_csv(args.result_dir / "membership.csv", index=False)
    (args.result_dir / "summary.json").write_text(json.dumps({
        "selected_k": selected_k,
        "patient_count": len(patient_ids),
        "subtype_count": len(final_subtypes),
        "subtype_sizes": [row["member_count"] for row in final_subtypes],
        "selected_k_pac": selected_metrics["pac"],
        "selected_k_silhouette": selected_metrics["silhouette_fused"],
        "selected_k_consensus_gap": selected_metrics["consensus_gap"],
        "result_dir": str(args.result_dir.resolve()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_k": selected_k, "patient_count": len(patient_ids), "subtype_count": len(final_subtypes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
