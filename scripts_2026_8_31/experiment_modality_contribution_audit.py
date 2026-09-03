#!/usr/bin/env python3
"""Offline modality-contribution audit for the existing candidate pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.candidate_proposer import fuse_affinities
from utils.candidate_clustering_outputs import canonical_partition, consensus_matrix_from_partitions, fit_kmedoids
from utils.llm_utils import load_candidate_proposer_config
from utils.io import write_json


VIEWS = ("ALL", "minus_RNA", "minus_WSI", "minus_CT", "minus_GENOMIC", "RNA_only", "WSI_only", "CT_only", "GENOMIC_only")
MODALITIES = ("ct", "wsi", "rna", "genomic")


def write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def rounded(value):
    return round(float(value), 8) if value is not None and np.isfinite(value) else None


def load_affinities(data_root):
    order = json.loads((data_root / "candidate_subtype/affinity_patient_order.json").read_text(encoding="utf-8"))
    paths = {name: data_root / f"candidate_subtype/{name}_affinity.npy" for name in ("ct", "wsi", "rna")}; paths["genomic"] = data_root / "wxs/genomic_affinity.npy"; paths["fused"] = data_root / "candidate_subtype/fused_similarity.npy"
    matrices = {name: np.asarray(np.load(path), dtype=float) for name, path in paths.items()}
    if any(matrix.shape != (len(order), len(order)) for matrix in matrices.values()): raise ValueError("Affinity matrix shape does not match affinity_patient_order")
    return [str(patient) for patient in order], matrices


def build_views(matrices, snf_config):
    views = {"ALL": matrices["fused"]}
    for view, excluded in (("minus_RNA", "rna"), ("minus_WSI", "wsi"), ("minus_CT", "ct"), ("minus_GENOMIC", "genomic")):
        views[view] = fuse_affinities([matrices[name] for name in MODALITIES if name != excluded], snf_config)
    for name in MODALITIES: views[f"{name.upper()}_only"] = matrices[name]
    return views


def load_all_partition(data_root, k, patient_ids):
    payload = json.loads((data_root / f"candidate_subtype/consensus_cluster/consensus_hierarchical_K{k}.json").read_text(encoding="utf-8"))
    labels = [int(payload["labels"][patient]) for patient in patient_ids]
    return np.asarray(canonical_partition(np.asarray(labels)), dtype=int)


def candidate_partition(matrix, k, config, seed):
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    rng = np.random.default_rng(seed); distance = np.maximum(1 - np.clip(np.asarray(matrix, float), 0, 1), 0); np.fill_diagonal(distance, 0)
    jobs = []
    algorithms = dict(config["clustering"]["algorithms"])
    for algorithm in ("hierarchical", "spectral", "kmedoids"):
        options = algorithms[algorithm]
        for repeat in range(int(config["clustering"]["repeat_count"])):
            if algorithm == "hierarchical": params = {"linkage": str(rng.choice(options["linkage_options"]))}
            elif algorithm == "spectral": params = {"assign_labels": str(rng.choice(options["assign_labels_options"]))}
            else: params = {"init": str(rng.choice(options["init_options"]))}
            try:
                if algorithm == "hierarchical": labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=params["linkage"]).fit_predict(distance)
                elif algorithm == "spectral": labels = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels=params["assign_labels"], random_state=int(rng.integers(0, 2**31 - 1))).fit_predict(np.clip(matrix, 0, 1))
                else: labels = fit_kmedoids(distance, n_clusters=k, init=params["init"], seed=int(rng.integers(0, 2**31 - 1)))
                if len(set(labels)) == k: jobs.append(tuple(canonical_partition(np.asarray(labels))))
            except Exception:
                continue
    if not jobs: raise RuntimeError(f"No valid candidate partitions for K={k}")
    consensus = consensus_matrix_from_partitions(jobs); consensus_distance = 1 - consensus; np.fill_diagonal(consensus_distance, 0)
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=str(config["clustering"]["consensus_linkage"])).fit_predict(consensus_distance)
    return np.asarray(canonical_partition(np.asarray(labels)), dtype=int)


def core_recovery(cores, partitions, view):
    rows = []
    for core, members in sorted(cores.items()):
        member_set = set(members); values = []
        for partition_id, clusters in partitions.items():
            best = max((set(cluster) for cluster in clusters.values()), key=lambda cluster: (len(member_set & cluster) / len(member_set | cluster) if member_set | cluster else 0, len(member_set & cluster), -len(cluster)), default=set()); intersection = len(member_set & best); union = len(member_set | best)
            values.append({"view": view, "core_id": core, "partition": partition_id, "best_jaccard": rounded(intersection / union if union else 0), "best_overlap_coefficient": rounded(intersection / min(len(member_set), len(best))) if member_set and best else 0, "best_recall": rounded(intersection / len(member_set)) if member_set else None, "best_precision": rounded(intersection / len(best)) if best else None})
        rows.extend(values)
    return rows


def affinity_audit(views, patient_ids, k=10):
    from scipy.stats import spearmanr
    rows = []; all_matrix = views["ALL"]; upper = np.triu_indices(len(patient_ids), 1); all_values = all_matrix[upper]
    def neighbors(matrix, index): return set(np.argsort(matrix[index])[::-1][1:k + 1])
    for view, matrix in views.items():
        values = matrix[upper]; rho = 1.0 if view == "ALL" or np.array_equal(all_values, values) else spearmanr(all_values, values).statistic; overlaps = []
        if view != "ALL":
            for index in range(len(patient_ids)):
                left, right = neighbors(all_matrix, index), neighbors(matrix, index); overlaps.append(len(left & right) / len(left | right) if left | right else 1.0)
        rows.append({"view": view, "spearman_vs_all": rounded(rho), "mean_knn_jaccard_vs_all": rounded(np.mean(overlaps)) if overlaps else 1.0, "median_knn_jaccard_vs_all": rounded(np.median(overlaps)) if overlaps else 1.0, "knn": k})
    return rows


def conditional_core_boundary(affinities, patient_ids, cores, pairs):
    from tools.multimodal_consistency_check import normalized_affinity_with_audit

    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    rows = []
    for modality, matrix in affinities.items():
        matrix = normalized_affinity_with_audit(matrix)[0]
        for core_a, core_b in pairs:
            left = [patient_index[patient] for patient in cores[core_a] if patient in patient_index]
            right = [patient_index[patient] for patient in cores[core_b] if patient in patient_index]
            if not left or not right: continue
            left_within = matrix[np.ix_(left, left)]
            right_within = matrix[np.ix_(right, right)]
            between = matrix[np.ix_(left, right)]
            left_values = left_within[~np.eye(len(left), dtype=bool)] if len(left) > 1 else np.array([])
            right_values = right_within[~np.eye(len(right), dtype=bool)] if len(right) > 1 else np.array([])
            left_margins = [matrix[index, [other for other in left if other != index]].mean() - matrix[index, right].mean() for index in left] if len(left) > 1 else []
            right_margins = [matrix[index, [other for other in right if other != index]].mean() - matrix[index, left].mean() for index in right] if len(right) > 1 else []
            margins = np.asarray(left_margins + right_margins)
            rows.append({
                "modality": modality, "core_a": core_a, "core_b": core_b,
                "core_a_n": len(left), "core_b_n": len(right),
                "within_a": rounded(np.mean(left_values)) if left_values.size else None,
                "within_b": rounded(np.mean(right_values)) if right_values.size else None,
                "between": rounded(between.mean()),
                "core_a_median_margin": rounded(np.median(left_margins)) if left_margins else None,
                "core_b_median_margin": rounded(np.median(right_margins)) if right_margins else None,
                "core_a_fraction_margin_positive": rounded(np.mean(np.asarray(left_margins) > 0)) if left_margins else None,
                "core_b_fraction_margin_positive": rounded(np.mean(np.asarray(right_margins) > 0)) if right_margins else None,
                "median_margin": rounded(np.median(margins)) if margins.size else None,
                "fraction_margin_positive": rounded(np.mean(margins > 0)) if margins.size else None,
            })
    return rows


def run(data_root=Path("output_kirc"), stable_root=Path("output_kirc_v12/03_multi_k_accepted_core_stability_v11"), output_root=Path("output_kirc_v12/07_modality_contribution_audit"), config_dir=Path("configs"), k_values=range(2, 9), knn=10, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force: raise FileExistsError(f"Output exists; pass --force: {output_root}")
    if force and output_root.exists(): shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True); patient_ids, matrices = load_affinities(data_root); config = load_candidate_proposer_config(config_dir); views = build_views(matrices, config["snf"]); cores = {}
    for row in csv_rows(stable_root / "stable_core_membership.csv"): cores.setdefault(str(row["core_id"]), []).append(str(row["patient_id"]))
    partitions = {}; partition_rows = []
    for view in VIEWS:
        partitions[view] = {}
        for k in k_values:
            labels = load_all_partition(data_root, k, patient_ids) if view == "ALL" else candidate_partition(views[view], k, config, 20260614)
            clusters = {f"C{i + 1:04d}": [patient_ids[index] for index, label in enumerate(labels) if label == value] for i, value in enumerate(sorted(set(labels)))}; partitions[view][f"K{k}"] = clusters; partition_rows.append({"view": view, "initial_k": k, "cluster_count": len(clusters), "cluster_sizes": json.dumps([len(members) for members in clusters.values()]), "labels": json.dumps(labels.tolist())})
    write_csv(output_root / "modality_ablation_partitions.csv", partition_rows)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows = []
    for view in VIEWS:
        for k in k_values:
            all_labels = json.loads(next(row["labels"] for row in partition_rows if row["view"] == "ALL" and row["initial_k"] == k)); labels = json.loads(next(row["labels"] for row in partition_rows if row["view"] == view and row["initial_k"] == k)); rows.append({"view": view, "initial_k": k, "ari_vs_all": rounded(adjusted_rand_score(all_labels, labels)), "nmi_vs_all": rounded(normalized_mutual_info_score(all_labels, labels))})
    write_csv(output_root / "modality_ablation_partition_similarity.csv", rows)
    baseline = {(int(row["partition"][1:]), row["core_id"]): row["best_jaccard"] for row in core_recovery(cores, partitions["ALL"], "ALL")}
    recovery = []
    for view in VIEWS:
        current = core_recovery(cores, partitions[view], view)
        for row in current:
            base_value = baseline.get((int(row["partition"][1:]), row["core_id"]))
            row["jaccard_retention"] = rounded(row["best_jaccard"] / base_value) if base_value not in (None, 0) else None
            row["delta_jaccard"] = rounded(row["best_jaccard"] - base_value) if base_value is not None else None
        recovery.extend(current)
    affinity_rows = affinity_audit(views, patient_ids, knn); write_csv(output_root / "modality_ablation_core_recovery.csv", recovery); write_csv(output_root / "modality_fused_affinity_similarity.csv", affinity_rows); write_csv(output_root / "conditional_core_boundary.csv", conditional_core_boundary({name: matrices[name] for name in MODALITIES}, patient_ids, cores, (("CORE01", "CORE03"), ("CORE02", "CORE05"))))
    recomputed = []
    for k in k_values:
        labels = candidate_partition(views["ALL"], k, config, 20260614); original = json.loads(next(row["labels"] for row in partition_rows if row["view"] == "ALL" and row["initial_k"] == k)); recomputed.append({"initial_k": k, "ari_recomputed_vs_original": rounded(adjusted_rand_score(original, labels)), "nmi_recomputed_vs_original": rounded(normalized_mutual_info_score(original, labels))})
    write_csv(output_root / "all_recomputed_vs_original.csv", recomputed)
    summary_rows = []
    for view in VIEWS:
        similarity_rows = [row for row in rows if row["view"] == view]; recovery_rows = [row for row in recovery if row["view"] == view]; affinity_row = next(row for row in affinity_rows if row["view"] == view); summary_rows.append({"view": view, "mean_ari_vs_all": rounded(np.mean([row["ari_vs_all"] for row in similarity_rows])), "mean_nmi_vs_all": rounded(np.mean([row["nmi_vs_all"] for row in similarity_rows])), "mean_core_best_jaccard": rounded(np.mean([row["best_jaccard"] for row in recovery_rows])), "mean_core_best_recall": rounded(np.mean([row["best_recall"] for row in recovery_rows])), "spearman_vs_all": affinity_row["spearman_vs_all"], "mean_knn_jaccard_vs_all": affinity_row["mean_knn_jaccard_vs_all"]})
    write_csv(output_root / "modality_contribution_summary.csv", summary_rows)
    summary = {"views": list(VIEWS), "k_values": list(k_values), "patient_count": len(patient_ids), "core_count": len(cores), "snf_config": {key: config["snf"][key] for key in ("neighbor_count", "iterations", "mu", "alpha")}, "scope": "offline audit; no feature extraction, Agent calls, or hyperparameter optimization"}; write_json(output_root / "modality_contribution_summary.json", summary); return summary


def csv_rows(path):
    import csv
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path("output_kirc")); parser.add_argument("--stable-root", type=Path, default=Path("output_kirc_v12/03_multi_k_accepted_core_stability_v11")); parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v12/07_modality_contribution_audit")); parser.add_argument("--config-dir", type=Path, default=Path("configs")); parser.add_argument("--knn", type=int, default=10); parser.add_argument("--force", action="store_true"); args = parser.parse_args(); print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
