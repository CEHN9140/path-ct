#!/usr/bin/env python3
"""Offline sensitivity audit for WXS/CNV genomic fusion granularity."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.candidate_proposer import fuse_affinities
from scripts_2026_8_31.experiment_modality_contribution_audit import (
    candidate_partition,
    conditional_core_boundary,
    core_recovery,
    load_all_partition,
    write_csv,
)
from tools.wxs import combine_genomic_affinities, wxs_distance_affinity
from utils.io import write_json
from utils.llm_utils import load_candidate_proposer_config

VARIANTS = {
    "M4_original": ("ct", "wsi", "rna", "genomic"),
    "M4_recomputed": ("ct", "wsi", "rna", "genomic"),
    "M5_independent_wxs_cnv": ("ct", "wsi", "rna", "wxs", "cnv"),
    "M_CNV": ("ct", "wsi", "rna", "cnv"),
    "M_WXS": ("ct", "wsi", "rna", "wxs"),
    "M_dupGenomic": ("ct", "wsi", "rna", "genomic", "genomic"),
}
NETWORK_NAMES = ("ct", "wsi", "rna", "wxs", "cnv", "genomic")


def load_inputs(data_root, stable_root, config_dir):
    order = [str(x) for x in json.loads((data_root / "candidate_subtype/affinity_patient_order.json").read_text())]
    paths = {name: data_root / f"candidate_subtype/{name}_affinity.npy" for name in ("ct", "wsi", "rna")}
    paths.update({"genomic": data_root / "wxs/genomic_affinity.npy", "fused": data_root / "candidate_subtype/fused_similarity.npy"})
    matrices = {name: np.asarray(np.load(path), dtype=float) for name, path in paths.items()}
    if any(matrix.shape != (len(order), len(order)) for matrix in matrices.values()):
        raise ValueError("Affinity shape does not match the common patient order")
    wxs_order = [str(x) for x in json.loads((data_root / "wxs/wxs_discovery_patient_order.json").read_text())]
    table = pd.read_csv(data_root / "wxs/wxs_discovery_features.csv").set_index("case_id").reindex(wxs_order)
    features = table[[column for column in table if column.startswith("mutation::")]].fillna(0).to_numpy(bool)
    intersection = features.astype(int) @ features.astype(int).T
    counts = features.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    config = load_candidate_proposer_config(config_dir)
    import yaml
    wxs_config = yaml.safe_load((config_dir / "wxs.yaml").read_text()) or {}
    distance = np.divide(union - intersection, union, out=np.full(intersection.shape, float(wxs_config["empty_mutation_distance"]), dtype=float), where=union > 0)
    np.fill_diagonal(distance, 0)
    wxs = wxs_distance_affinity(distance, config["snf"])
    if wxs_order != order:
        indices = [wxs_order.index(patient) for patient in order]
        wxs = wxs[np.ix_(indices, indices)]
    cnv_table = pd.read_csv(data_root / "cnv/case_features.csv").set_index("case_id").reindex(order).to_numpy(float)
    median = np.median(cnv_table, axis=0, keepdims=True)
    iqr = np.quantile(cnv_table, .75, axis=0, keepdims=True) - np.quantile(cnv_table, .25, axis=0, keepdims=True)
    cnv_table = np.divide(cnv_table - median, iqr, out=np.zeros_like(cnv_table), where=iqr > 0)
    cnv = wxs_distance_affinity(cdist(cnv_table, cnv_table), config["snf"])
    saved_cnv = np.asarray(np.load(data_root / "wxs/cnv_affinity.npy"), dtype=float)
    cores = {}
    with (stable_root / "stable_core_membership.csv").open(encoding="utf-8", newline="") as handle:
        import csv
        for row in csv.DictReader(handle):
            cores.setdefault(row["core_id"], []).append(row["patient_id"])
    return order, matrices | {"wxs": wxs, "cnv": cnv}, cores, config, config["snf"], saved_cnv


def upper_metrics(left, right, k=10):
    from scipy.stats import pearsonr, spearmanr
    upper = np.triu_indices(len(left), 1)
    a, b = left[upper], right[upper]
    neighbors = lambda matrix, i: set(np.argsort(matrix[i])[::-1][1:k + 1])
    overlaps = [len(neighbors(left, i) & neighbors(right, i)) / len(neighbors(left, i) | neighbors(right, i)) for i in range(len(left))]
    return {"spearman": float(spearmanr(a, b).statistic) if not np.array_equal(a, b) else 1.0, "pearson": float(pearsonr(a, b).statistic) if np.std(a) and np.std(b) else 1.0, "mean_knn_jaccard": float(np.mean(overlaps)), "median_knn_jaccard": float(np.median(overlaps))}


def coassignment(labels_by_k):
    labels = list(labels_by_k.values()); n = len(labels[0]); matrix = np.zeros((n, n), dtype=float)
    for labels_k in labels: matrix += np.equal.outer(labels_k, labels_k)
    return matrix / len(labels)


def coassignment_rows(variant, matrix, cores, patient_ids):
    index = {patient: i for i, patient in enumerate(patient_ids)}; rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        a = [index[p] for p in cores[core_a] if p in index]; b = [index[p] for p in cores[core_b] if p in index]
        a_pairs = matrix[np.ix_(a, a)][~np.eye(len(a), dtype=bool)] if len(a) > 1 else []
        b_pairs = matrix[np.ix_(b, b)][~np.eye(len(b), dtype=bool)] if len(b) > 1 else []
        between = matrix[np.ix_(a, b)] if a and b else np.array([])
        rows.append({"variant": variant, "core_a": core_a, "core_b": core_b, "within_core_a": float(np.mean(a_pairs)) if len(a_pairs) else None, "within_core_b": float(np.mean(b_pairs)) if len(b_pairs) else None, "between_core": float(np.mean(between)) if between.size else None})
    return rows


def knn_rows(left_name, left, right_name, right, patient_ids, k):
    rows = []
    for index, patient_id in enumerate(patient_ids):
        left_neighbors = set(np.argsort(left[index])[::-1][1:k + 1])
        right_neighbors = set(np.argsort(right[index])[::-1][1:k + 1])
        rows.append({"patient_id": patient_id, "left": left_name, "right": right_name, "knn_jaccard": len(left_neighbors & right_neighbors) / len(left_neighbors | right_neighbors) if left_neighbors | right_neighbors else 1.0})
    return rows


def bootstrap_boundary(matrix_a, matrix_b, left, right, rng, n_bootstrap):
    def median_margin(matrix, a, b):
        a = list(dict.fromkeys(a)); b = list(dict.fromkeys(b))
        margins = [matrix[index, [other for other in a if other != index]].mean() - matrix[index, b].mean() for index in a if len(a) > 1]
        margins += [matrix[index, [other for other in b if other != index]].mean() - matrix[index, a].mean() for index in b if len(b) > 1]
        return float(np.median(margins)) if margins else np.nan
    observed = median_margin(matrix_a, left, right) - median_margin(matrix_b, left, right)
    values = []
    for _ in range(n_bootstrap):
        sampled_left = rng.choice(left, len(left), replace=True).tolist()
        sampled_right = rng.choice(right, len(right), replace=True).tolist()
        values.append(median_margin(matrix_a, sampled_left, sampled_right) - median_margin(matrix_b, sampled_left, sampled_right))
    values = [value for value in values if np.isfinite(value)]
    return observed, float(np.quantile(values, .025)) if values else np.nan, float(np.quantile(values, .975)) if values else np.nan


def candidate_sets(labels, patient_ids, initial_k):
    return [{"cluster_id": f"C{i + 1:04d}", "member_ids": [patient_ids[j] for j, label in enumerate(labels) if label == value], "source_views": ["ct", "wsi", "rna", "wxs", "cnv"], "generator": {"algorithm": "consensus_clustering", "n_clusters": initial_k, "partition_id": f"five_view_K{initial_k}"}} for i, value in enumerate(sorted(set(labels)))]


def run_agent_reviews(output_root, data_root, config_dir, patient_ids, partitions, repeats, initial_ks, force):
    from agents.subtype_review.graph import save_review_outputs
    from agents.subtype_review.runner import run_subtype_review
    from scripts_2026_8_31.experiment_initial_k_review_sensitivity import load_patient_states

    patient_states = list(load_patient_states(data_root).values())
    rows = []
    for repeat in repeats:
        for k in initial_ks:
            run_root = output_root / "agent_discovery" / f"run{repeat}" / f"K{k}"
            summary_path = run_root / "final_review_summary.json"
            if summary_path.exists() and not force:
                rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            if run_root.exists(): shutil.rmtree(run_root)
            run_root.mkdir(parents=True)
            initial_sets = candidate_sets(partitions["M5_independent_wxs_cnv"][k], patient_ids, k)
            write_json(run_root / "initial_partition.json", {"initial_k": k, "repeat": repeat, "view": "M5_independent_wxs_cnv", "candidate_sets": initial_sets})
            state = run_subtype_review(initial_sets, patient_states, str(data_root), str(config_dir), artifact_root=str(run_root))
            summary = save_review_outputs(state, str(run_root), direct=True)
            rows.append(summary)
    write_json(output_root / "agent_discovery_summary.json", {"view": "M5_independent_wxs_cnv", "initial_k": list(initial_ks), "repeats": list(repeats), "runs": rows, "note": "Forced initial-K Agent discovery; no best-K selector is applied in robustness mode."})


def run(data_root=Path("output_kirc"), stable_root=Path("output_kirc_v12/03_multi_k_accepted_core_stability_v11"), output_root=Path("output_kirc_v12/13_genomic_fusion_sensitivity"), config_dir=Path("configs"), k_values=range(2, 9), knn=10, bootstrap=1000, force=False, run_agent=False, repeats=(1, 2, 3)):
    if output_root.exists() and any(output_root.iterdir()) and not force: raise FileExistsError(f"Output exists; pass --force: {output_root}")
    if force and output_root.exists(): shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ids, matrices, cores, config, snf, saved_cnv = load_inputs(data_root, stable_root, config_dir)
    variants = {"M4_original": matrices["fused"]}
    variants["M4_recomputed"] = fuse_affinities([matrices[x] for x in VARIANTS["M4_recomputed"]], snf)
    variants["M5_independent_wxs_cnv"] = fuse_affinities([matrices[x] for x in VARIANTS["M5_independent_wxs_cnv"]], snf)
    variants["M_CNV"] = fuse_affinities([matrices[x] for x in VARIANTS["M_CNV"]], snf)
    variants["M_WXS"] = fuse_affinities([matrices[x] for x in VARIANTS["M_WXS"]], snf)
    variants["M_dupGenomic"] = fuse_affinities([matrices[x] for x in VARIANTS["M_dupGenomic"]], snf)
    affinity_dir = output_root / "affinity"; affinity_dir.mkdir()
    for name, matrix in variants.items(): np.save(affinity_dir / f"{name}.npy", matrix)
    np.save(affinity_dir / "WXS_reconstructed.npy", matrices["wxs"])
    reconstructed_genomic = combine_genomic_affinities(matrices["wxs"], matrices["cnv"], snf)
    reconstruction_error = np.abs(reconstructed_genomic - matrices["genomic"])
    cnv_error = np.abs(matrices["cnv"] - saved_cnv)
    reconstruction_audit = {"patient_count": len(ids), "cnv_max_abs_diff_recomputed_vs_saved": float(cnv_error.max()), "cnv_mean_abs_diff_recomputed_vs_saved": float(cnv_error.mean()), "cnv_allclose": bool(np.allclose(matrices["cnv"], saved_cnv, rtol=1e-6, atol=1e-8)), "max_abs_diff_reconstructed_genomic_vs_saved": float(reconstruction_error.max()), "mean_abs_diff": float(reconstruction_error.mean()), "allclose": bool(np.allclose(reconstructed_genomic, matrices["genomic"], rtol=1e-6, atol=1e-8))}
    reconstruction_audit.update(upper_metrics(reconstructed_genomic, matrices["genomic"], knn))
    write_json(output_root / "wxs_reconstruction_audit.json", reconstruction_audit)
    if not reconstruction_audit["cnv_allclose"] or not reconstruction_audit["allclose"]: raise RuntimeError("Recomputed WXS/CNV affinity does not match the saved pipeline affinity")
    reconstruction = np.load(affinity_dir / "M4_original.npy")
    m4_control = upper_metrics(variants["M4_recomputed"], reconstruction, knn); write_csv(output_root / "m4_recomputed_vs_original_affinity.csv", [m4_control])
    partitions = {}; partition_rows = []; similarity_rows = []; recovery_rows = []; stability_rows = []; coassignment_all = {}
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    for variant, matrix in variants.items():
        partitions[variant] = {}
        for k in k_values:
            labels = load_all_partition(data_root, k, ids) if variant == "M4_original" else candidate_partition(matrix, k, config, 20260614)
            partitions[variant][k] = labels; partition_rows.append({"variant": variant, "initial_k": k, "cluster_count": len(set(labels)), "cluster_sizes": json.dumps([int(np.sum(labels == x)) for x in sorted(set(labels))]), "labels": json.dumps(labels.tolist())})
        coassignment_all[variant] = coassignment(partitions[variant]); np.save(output_root / f"coassignment_{variant}.npy", coassignment_all[variant])
        for a, b in combinations(k_values, 2): stability_rows.append({"variant": variant, "k_a": a, "k_b": b, "ari": float(adjusted_rand_score(partitions[variant][a], partitions[variant][b])), "nmi": float(normalized_mutual_info_score(partitions[variant][a], partitions[variant][b]))})
        recovery_rows.extend(core_recovery(cores, {f"K{k}": {f"C{i + 1:04d}": [ids[j] for j, label in enumerate(labels) if label == value] for i, value in enumerate(sorted(set(labels)))} for k, labels in partitions[variant].items()}, variant))
    baseline = {(row["partition"], row["core_id"]): row["best_jaccard"] for row in core_recovery(cores, {f"K{k}": {f"C{i + 1:04d}": [ids[j] for j, label in enumerate(partitions["M4_recomputed"][k]) if label == value] for i, value in enumerate(sorted(set(partitions["M4_recomputed"][k])))} for k in k_values}, "M4_recomputed")}
    for row in recovery_rows:
        base = baseline.get((row["partition"], row["core_id"])); row["jaccard_retention"] = float(row["best_jaccard"]) / float(base) if base not in (None, 0) else None; row["delta_jaccard"] = float(row["best_jaccard"]) - float(base) if base is not None else None
    write_csv(output_root / "genomic_fusion_partitions.csv", partition_rows); write_csv(output_root / "genomic_fusion_partition_similarity.csv", [{"variant": v, "baseline": "M4_recomputed", "initial_k": k, "ari": float(adjusted_rand_score(partitions[v][k], partitions["M4_recomputed"][k])), "nmi": float(normalized_mutual_info_score(partitions[v][k], partitions["M4_recomputed"][k]))} for v in variants for k in k_values]); write_csv(output_root / "genomic_fusion_core_recovery.csv", recovery_rows); write_csv(output_root / "genomic_fusion_multik_stability.csv", stability_rows)
    write_csv(output_root / "genomic_fusion_core_coassignment.csv", [row | {"variant": variant} for variant, matrix in coassignment_all.items() for row in coassignment_rows(variant, matrix, cores, ids)])
    relationships = [dict(left=a, right=b, **upper_metrics(matrices[a], matrices[b], knn)) for a, b in (("wxs", "cnv"), ("wxs", "genomic"), ("cnv", "genomic"))]; write_csv(output_root / "wxs_cnv_affinity_relationship.csv", relationships)
    knn_pairs = [("wxs", "genomic"), ("cnv", "genomic"), ("wxs", "cnv")]
    knn_rows_all = [row for left, right in knn_pairs for row in knn_rows(left, matrices[left], right, matrices[right], ids, knn)]
    write_csv(output_root / "wxs_cnv_knn_overlap.csv", knn_rows_all)
    pairs = list(combinations(sorted(cores), 2)); boundary = conditional_core_boundary({name: matrices[name] for name in ("wxs", "cnv", "genomic")}, ids, cores, pairs); write_csv(output_root / "genomic_fusion_boundary_support.csv", boundary)
    rng = np.random.default_rng(20260614); boundary_bootstrap = []
    for row in boundary:
        left = [ids.index(patient) for patient in cores[row["core_a"]] if patient in ids]; right = [ids.index(patient) for patient in cores[row["core_b"]] if patient in ids]
        if len(left) < 2 or len(right) < 2: continue
        for comparison, first, second in (("CNV_minus_Genomic", matrices["cnv"], matrices["genomic"]), ("WXS_minus_Genomic", matrices["wxs"], matrices["genomic"])):
            effect, low, high = bootstrap_boundary(first, second, left, right, rng, bootstrap); boundary_bootstrap.append({"core_a": row["core_a"], "core_b": row["core_b"], "comparison": comparison, "bootstrap_median": effect, "ci_low": low, "ci_high": high})
    write_csv(output_root / "genomic_fusion_boundary_bootstrap.csv", boundary_bootstrap)
    modality_similarity = [dict(variant=variant, modality=modality, **upper_metrics(matrix, variants[variant], knn)) for variant, matrix in variants.items() for modality, matrix in matrices.items() if modality in NETWORK_NAMES]; write_csv(output_root / "genomic_fusion_modality_affinity_similarity.csv", modality_similarity)
    write_csv(output_root / "genomic_fusion_modality_knn_overlap.csv", [{"variant": variant, "modality": modality, **upper_metrics(matrix, variants[variant], knn)} for variant, matrix in variants.items() for modality, matrix in matrices.items() if modality in NETWORK_NAMES])
    write_json(output_root / "experiment_manifest.json", {"experiment": "genomic_fusion_sensitivity", "patient_count": len(ids), "k_values": list(k_values), "seed": 20260614, "knn": knn, "variants": VARIANTS, "snf_config": dict(snf), "feature_reextraction": False, "hyperparameter_optimization": False, "agent_calls": bool(run_agent)})
    np.save(output_root / "fused_similarity_5view.npy", variants["M5_independent_wxs_cnv"])
    if run_agent:
        run_agent_reviews(output_root, data_root, config_dir, ids, partitions, repeats, k_values, force)
    summary = [{"variant": variant, "mean_ari_vs_M4": float(np.mean([adjusted_rand_score(partitions[variant][k], partitions["M4_recomputed"][k]) for k in k_values])), "mean_cross_k_ari": float(np.mean([row["ari"] for row in stability_rows if row["variant"] == variant])), "mean_cross_k_nmi": float(np.mean([row["nmi"] for row in stability_rows if row["variant"] == variant])), "spearman_with_wxs": upper_metrics(variants[variant], matrices["wxs"], knn)["spearman"], "spearman_with_cnv": upper_metrics(variants[variant], matrices["cnv"], knn)["spearman"]} for variant in variants]; write_csv(output_root / "genomic_fusion_summary.csv", summary); write_json(output_root / "genomic_fusion_summary.json", {"variants": list(variants), "patient_count": len(ids), "k_values": list(k_values), "bootstrap_requested": bootstrap, "note": "Descriptive representation sensitivity audit; no winner or same-subtype probability is computed."}); return {"output_root": str(output_root), "variants": list(variants), "patient_count": len(ids), "k_values": list(k_values)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path("output_kirc")); parser.add_argument("--stable-root", type=Path, default=Path("output_kirc_v12/03_multi_k_accepted_core_stability_v11")); parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v12/13_genomic_fusion_sensitivity")); parser.add_argument("--config-dir", type=Path, default=Path("configs")); parser.add_argument("--knn", type=int, default=10); parser.add_argument("--bootstrap", type=int, default=1000); parser.add_argument("--repeat", dest="repeats", type=int, action="append", default=[1, 2, 3]); parser.add_argument("--run-agent", action="store_true"); parser.add_argument("--force", action="store_true"); print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
