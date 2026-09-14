#!/usr/bin/env python3
"""Offline sensitivity audit for the canonical five-view feature engineering."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts_2026_9_7 import analyze_multi_k_stable_cores as core_analysis
from tools.evidence_features import distance_to_affinity
from tools.multimodal_consistency_check import normalize_affinity
from utils.llm_utils import load_candidate_proposer_config


def load_inputs(data_root, multi_k_root):
    candidate = data_root / "candidate_subtype"
    patient_ids = [str(x) for x in json.loads((candidate / "affinity_patient_order.json").read_text())]
    affinities = {name: np.load(candidate / f"{name}_affinity.npy") for name in ("ct", "wsi", "rna")}
    affinities.update({"wxs": np.load(data_root / "wxs/wxs_affinity.npy"), "cnv": np.load(data_root / "wxs/cnv_affinity.npy")})
    cores, core_ids = core_analysis.load_cores(multi_k_root)
    if len(cores) != 4 or len(core_ids) != 56:
        raise ValueError(f"Expected four stable cores covering 56 patients, got {len(cores)} and {len(core_ids)}")
    if any(matrix.shape != (len(patient_ids), len(patient_ids)) for matrix in affinities.values()):
        raise ValueError("Canonical affinity shapes do not match patient order")
    return patient_ids, affinities, cores


def affinity(values, metric, snf_config):
    distance = cdist(values, values, metric=metric)
    return distance_to_affinity(distance, snf_config)


def robust_scale_cnv(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values, axis=0, keepdims=True)
    iqr = np.quantile(values, 0.75, axis=0, keepdims=True) - np.quantile(values, 0.25, axis=0, keepdims=True)
    return np.divide(values - median, iqr, out=np.zeros_like(values), where=iqr > 0)


def rna_affinity_from_full_table(table, count, snf_config):
    numeric = table.apply(pd.to_numeric).fillna(0.0)
    mad = numeric.sub(numeric.median()).abs().median().sort_values(ascending=False)
    selected = mad.head(min(count, len(mad))).index
    values = numeric[selected].to_numpy(float)
    values -= values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    values = np.divide(values, std, out=np.zeros_like(values), where=std > 0)
    return affinity(values, "correlation", snf_config), len(selected)


def mutation_affinity(values, empty_distance, snf_config):
    binary = np.asarray(values, dtype=bool)
    intersection = binary.astype(int) @ binary.astype(int).T
    union = binary.sum(1)[:, None] + binary.sum(1)[None, :] - intersection
    distance = np.divide(union - intersection, union, out=np.full_like(union, empty_distance, dtype=float), where=union > 0)
    np.fill_diagonal(distance, 0.0)
    return distance_to_affinity(distance, snf_config)


def fuse(views, snf_config):
    import snf
    fused = snf.snf(*[views[name] for name in ("ct", "wsi", "rna", "wxs", "cnv")], K=int(snf_config["neighbor_count"]), t=int(snf_config["iterations"]), alpha=float(snf_config["alpha"]))
    fused = np.maximum((np.asarray(fused) + np.asarray(fused).T) / 2, 0)
    np.fill_diagonal(fused, 1.0)
    return fused


def core_jaccard(reference, candidate):
    left = [set(values) for values in reference.values()]
    right = [set(values) for values in candidate.values()]
    scores = np.asarray([[len(a & b) / len(a | b) for b in right] for a in left], dtype=float)
    size = max(len(left), len(right))
    padded = np.zeros((size, size), dtype=float)
    padded[:len(left), :len(right)] = scores
    rows, columns = linear_sum_assignment(1.0 - padded)
    matched = padded[rows, columns]
    return float(matched.mean()), float(matched.min())


def recovery_metrics(fused, patient_ids, cores):
    core_ids = [case for members in cores.values() for case in members]
    positions = [patient_ids.index(case) for case in core_ids]
    matrix = normalize_affinity(fused)[np.ix_(positions, positions)]
    labels = np.asarray([next(i for i, members in enumerate(cores.values()) if case in members) for case in core_ids])
    distance = np.maximum(0.0, 1.0 - matrix)
    predicted = AgglomerativeClustering(n_clusters=4, metric="precomputed", linkage="average").fit_predict(distance)
    candidate = {str(label): [case for case, value in zip(core_ids, predicted) if value == label] for label in sorted(set(predicted))}
    mean_jaccard, min_jaccard = core_jaccard(cores, candidate)
    return {
        "fixed_core_mean_jaccard": mean_jaccard,
        "fixed_core_min_jaccard": min_jaccard,
        "fixed_core_ari": float(adjusted_rand_score(labels, predicted)),
        "fixed_core_nmi": float(normalized_mutual_info_score(labels, predicted)),
        "fixed_core_mean_silhouette": float(silhouette_score(distance, labels, metric="precomputed")),
    }


def load_variants(data_root, patient_ids, canonical, snf_config):
    variants = {"canonical": (fuse(canonical, snf_config), {})}
    rna = pd.read_csv(data_root / "rna/case_pathway_features.csv").set_index("case_id").reindex(patient_ids)
    for count in (1000, 2000, 3000, 5000):
        rna_view, feature_count = rna_affinity_from_full_table(rna, count, snf_config)
        views = dict(canonical); views["rna"] = rna_view
        variants[f"rna_top_{count}"] = (fuse(views, snf_config), {"feature_count": feature_count, "view": "rna", "normalization": "cohort_zscore"})
    wxs = pd.read_csv(data_root / "wxs/wxs_discovery_features.csv").set_index("case_id").reindex(patient_ids).fillna(0.0)
    prevalence = wxs.mean()
    selected = prevalence[prevalence >= .05].index
    for name, empty_distance, columns in (("wxs_prevalence_only", 0.0, selected), ("wxs_zero_distance_1", 1.0, wxs.columns)):
        views = dict(canonical); views["wxs"] = mutation_affinity(wxs[columns].to_numpy(), empty_distance, snf_config)
        variants[name] = (fuse(views, snf_config), {"feature_count": len(columns), "view": "wxs", "empty_mutation_distance": empty_distance})
    cnv = pd.read_csv(data_root / "cnv/case_features.csv").set_index("case_id").reindex(patient_ids).fillna(0.0)
    for name, columns in (("cnv_arm_only", [x for x in cnv if x.startswith("chr")]), ("cnv_arm_locus", [x for x in cnv if x.startswith("chr") or x.startswith("locus::")])):
        views = dict(canonical); views["cnv"] = affinity(robust_scale_cnv(cnv[columns].to_numpy()), "euclidean", snf_config)
        variants[name] = (fuse(views, snf_config), {"feature_count": len(columns), "view": "cnv"})
    return variants


def run(data_root, multi_k_root, config_dir, output_root, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    patient_ids, canonical, cores = load_inputs(data_root, multi_k_root)
    snf_config = load_candidate_proposer_config(config_dir)["snf"]
    variants = load_variants(data_root, patient_ids, canonical, snf_config)
    rows = []
    for name, (fused, metadata) in variants.items():
        metrics = recovery_metrics(fused, patient_ids, cores)
        similarity = normalize_affinity(fused)
        offdiag = ~np.eye(len(patient_ids), dtype=bool)
        metrics["fused_correlation_with_canonical"] = float(np.corrcoef(similarity[offdiag], normalize_affinity(variants["canonical"][0])[offdiag])[0, 1])
        rows.append({"variant": name, **metadata, **metrics})
    wxs = pd.read_csv(data_root / "wxs/wxs_discovery_features.csv").set_index("case_id").reindex(patient_ids).fillna(0.0)
    zero = wxs.sum(axis=1).eq(0)
    core_ids = set().union(*cores.values())
    zero_rows = [{"group": group, "zero_vector_n": int(zero.loc[list(ids)].sum()), "total_n": len(ids), "zero_vector_fraction": float(zero.loc[list(ids)].mean())} for group, ids in (("core", core_ids), ("non_core", set(patient_ids) - core_ids))]
    core_analysis.write_csv(output_root / "feature_engineering_sensitivity_summary.csv", rows)
    core_analysis.write_csv(output_root / "wxs_zero_vector_audit.csv", zero_rows)
    core_analysis.write_json(output_root / "feature_engineering_sensitivity_manifest.json", {"experiment": "feature_engineering_sensitivity", "patient_count": len(patient_ids), "stable_core_count": len(cores), "stable_core_patient_count": len(core_ids), "variants": list(variants), "wxs_prevalence_only_scope": "prevalence-only selection within the canonical 22-gene discovery table; the canonical construction is prevalence union curated drivers", "core_recovery_definition": "fixed-core partition recovery diagnostic: deterministic K=4 average-linkage clustering on the 56 fixed core patients; not full stable-core rediscovery", "jaccard_matching": "Hungarian one-to-one with zero penalty for unmatched cores", "no_agent_rerun": True, "no_partition_replacement": True})
    return {"variant_count": len(rows), "output_root": str(output_root), "stable_core_patient_count": len(core_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("output_kirc"))
    parser.add_argument("--multi-k-root", type=Path, default=Path("output_kirc_v13/00_five_view_multi_k_agent_review"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v14/05_feature_engineering_sensitivity"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
