#!/usr/bin/env python3
"""Run an isolated four-view feature-engineering variant without changing main."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.candidate_proposer import consensus_records_from_similarity
from tools.confound import confounder_values
from tools.evidence_features import distance_to_affinity, fuse_affinities
from tools.wxs import binary_mutation_distance
from utils.io import write_json
from utils.llm_utils import load_candidate_proposer_config

ACTIVE_MODALITIES = ("ct", "wsi", "rna", "wxs")
CORRELATION_THRESHOLD = 0.95
EMPTY_MUTATION_DISTANCE = 1.0
RUNNER_PATH = ROOT / "scripts_2026_9_7" / "00_experiment_five_view_multi_k_agent_review.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("feature_variant_multi_k_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform_ct_features(matrix, feature_names, technical_design, low_variance_threshold, correlation_threshold):
    matrix = np.asarray(matrix, dtype=float)
    keep = np.var(matrix, axis=0) > float(low_variance_threshold)
    low_variance_removed = [name for name, selected in zip(feature_names, keep) if not selected]
    matrix = matrix[:, keep]
    names = [name for name, selected in zip(feature_names, keep) if selected]

    pruned = []
    if matrix.shape[1] > 1:
        correlation = np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0)
        selected = []
        for index, name in enumerate(names):
            if any(abs(correlation[index, prior]) > correlation_threshold for prior in selected):
                pruned.append(name)
            else:
                selected.append(index)
        matrix = matrix[:, selected]
        names = [names[index] for index in selected]

    design = np.asarray(technical_design, dtype=float)
    residual = matrix - design @ np.linalg.lstsq(design, matrix, rcond=None)[0]
    means = residual.mean(axis=0, keepdims=True)
    stds = residual.std(axis=0, keepdims=True)
    zero_variance = stds <= 1e-12
    standardized = np.divide(residual - means, stds, out=np.zeros_like(residual), where=~zero_variance)
    audit = {
        "raw_feature_count": int(len(feature_names)),
        "low_variance_threshold": float(low_variance_threshold),
        "low_variance_removed": low_variance_removed,
        "after_low_variance_count": int(len(feature_names) - len(low_variance_removed)),
        "correlation_threshold": float(correlation_threshold),
        "correlation_pruned": pruned,
        "after_correlation_count": int(len(names)),
        "retained_features": names,
        "technical_design_shape": list(design.shape),
        "technical_design_rank": int(np.linalg.matrix_rank(design)),
        "residual_degrees_of_freedom": int(len(matrix) - np.linalg.matrix_rank(design)),
        "post_residual_zero_variance_features": [
            name for name, is_zero in zip(names, zero_variance.ravel()) if is_zero
        ],
        "steps": [
            "PyRadiomics cached feature extraction",
            "low variance filter",
            "absolute Pearson correlation pruning",
            "technical-covariate residualization",
            "column-wise z-score",
            "Euclidean distance",
        ],
    }
    return standardized, names, audit


def technical_design_matrix(patient_ids, states, data_root, ct_config):
    correction = dict(ct_config.get("confound_correction", {}) or {})
    fields = list(correction.get("fields") or []) if correction.get("enabled", True) else []
    values = confounder_values(states, str(data_root))
    parts = [np.ones((len(patient_ids), 1))]
    audit = {}
    for field in fields:
        raw = [dict(values.get(case_id, {}) or {}).get(field, "") for case_id in patient_ids]
        if field == "ct_slice_thickness":
            numeric = pd.to_numeric(pd.Series(raw), errors="coerce")
            missing = int(numeric.isna().sum())
            numeric = numeric.fillna(float(numeric.median()) if numeric.notna().any() else 0.0)
            parts.append(numeric.to_numpy(float)[:, None])
            audit[field] = {"type": "numeric", "missing_count": missing}
        else:
            missing = sum(not str(value or "").strip() for value in raw)
            series = pd.Series([str(value or "missing") for value in raw])
            encoded = pd.get_dummies(series, dtype=float)
            parts.append(encoded.to_numpy(float))
            audit[field] = {
                "type": "categorical",
                "levels": sorted(series.unique().tolist()),
                "level_counts": {str(key): int(value) for key, value in series.value_counts().sort_index().items()},
                "singleton_levels": sorted(series.value_counts()[series.value_counts() == 1].index.tolist()),
                "rare_levels_lt_5": sorted(series.value_counts()[series.value_counts() < 5].index.tolist()),
                "missing_count": missing,
            }
    return np.column_stack(parts), {"fields": audit}


def load_ct_features(patient_ids, states):
    vectors, names = [], []
    for case_id in patient_ids:
        evidence = dict(states[case_id].get("ct_evidence", {}) or {})
        path = Path(str(evidence.get("feature_path", "")))
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_names = list(payload)
        if names and names[0] != current_names:
            raise ValueError(f"CT feature names differ for {case_id}")
        names.append(current_names)
        vectors.append([float(payload[name]) for name in current_names])
    return np.asarray(vectors, dtype=float), names[0]


def build_wxs_view(features, snf_config):
    distance = binary_mutation_distance(np.asarray(features, dtype=bool), EMPTY_MUTATION_DISTANCE)
    affinity = distance_to_affinity(distance, snf_config)
    return distance, affinity


def prevalence_only_wxs_features(source_table, patient_ids, min_prevalence):
    table = source_table.copy()
    table["case_id"] = table["case_id"].astype(str)
    if table["case_id"].duplicated().any() or set(table["case_id"]) != set(patient_ids):
        raise ValueError("WXS feature table does not contain exactly one row per discovery patient")
    table = table.set_index("case_id").loc[patient_ids].reset_index()
    feature_columns = [column for column in table.columns if column != "case_id"]
    prevalence = table[feature_columns].astype(float).mean(axis=0)
    selected = prevalence[prevalence >= float(min_prevalence)].index.tolist()
    return table[["case_id", *selected]], selected


def prepare_variant_input(data_root, output_root, patient_ids, views, fused, wxs_table, ct_audit, wxs_audit, config, force):
    input_root = output_root / "inputs" / "four_view_feature_engineering"
    if input_root.exists():
        if not force:
            raise FileExistsError(f"Variant input exists; pass --force to rebuild: {input_root}")
        shutil.rmtree(input_root)
    candidate_dir = input_root / "candidate_subtype"
    candidate_dir.mkdir(parents=True)
    for name in ("storage", "ct_radiomics", "ct_qc", "rna", "wsi_tumor_seg"):
        (input_root / name).symlink_to((data_root / name).resolve(), target_is_directory=True)
    shutil.copytree(data_root / "wxs", input_root / "wxs")
    wxs_root = input_root / "wxs"
    wxs_table.to_csv(wxs_root / "wxs_discovery_features.csv", index=False)

    write_json(candidate_dir / "affinity_patient_order.json", patient_ids)
    paths = {}
    for name in ACTIVE_MODALITIES:
        if name == "wxs":
            target = wxs_root / "wxs_affinity.npy"
            paths[name] = "wxs/wxs_affinity.npy"
        else:
            target = candidate_dir / f"{name}_affinity.npy"
            paths[name] = target.name
        np.save(target, views[name])
    write_json(candidate_dir / "affinity_cache.json", {
        "cache_version": 1,
        "variant": "four_view_feature_engineering",
        "active_modalities": list(ACTIVE_MODALITIES),
        "disabled_modalities": ["cnv"],
        "patient_ids": patient_ids,
        "paths": paths,
    })
    np.save(candidate_dir / "fused_similarity.npy", fused)
    records, _ = consensus_records_from_similarity(fused, config["clustering"])
    consensus_dir = candidate_dir / "consensus_cluster"
    consensus_dir.mkdir()
    for record in records:
        k = int(record["n_clusters"])
        write_json(consensus_dir / f"consensus_hierarchical_K{k}.json", {
            "n_clusters": k,
            "labels": dict(zip(patient_ids, map(int, record["labels"]))),
            "partition_source": "production_consensus_records_from_similarity",
            "active_modalities": list(ACTIVE_MODALITIES),
        })
    write_json(candidate_dir / "feature_engineering_variant.json", {
        "variant": "four_view_feature_engineering",
        "active_modalities": list(ACTIVE_MODALITIES),
        "disabled_modalities": ["cnv"],
        "ct": ct_audit,
        "wxs": wxs_audit,
    })
    return input_root


def run(data_root, config_dir, output_root, initial_ks, repeats, force=False, preflight=False):
    data_root, config_dir, output_root = map(Path, (data_root, config_dir, output_root))
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    patient_ids, source_views, _ = runner.load_main_inputs(data_root, ACTIVE_MODALITIES)
    states_map = runner.load_patient_states(data_root)
    states = {case_id: states_map[case_id] for case_id in patient_ids}

    raw_ct, feature_names = load_ct_features(patient_ids, states)
    technical_design, technical_audit = technical_design_matrix(patient_ids, states, data_root, yaml.safe_load((config_dir / "ct_radiomics.yaml").read_text(encoding="utf-8")))
    snf_config = load_candidate_proposer_config(config_dir)["snf"]
    ct_matrix, retained_ct_features, ct_audit = transform_ct_features(
        raw_ct, feature_names, technical_design,
        snf_config["ct_low_variance_threshold"], CORRELATION_THRESHOLD,
    )
    ct_distance = cdist(ct_matrix, ct_matrix, metric="euclidean")
    source_views["ct"] = distance_to_affinity(ct_distance, snf_config)
    ct_audit.update(technical_audit)
    ct_audit.update({"patient_count": len(patient_ids), "retained_feature_count": len(retained_ct_features)})

    source_table = pd.read_csv(data_root / "wxs" / "wxs_discovery_features.csv")
    min_prevalence = float(yaml.safe_load((config_dir / "wxs.yaml").read_text(encoding="utf-8"))["min_gene_prevalence"])
    wxs_table, selected_features = prevalence_only_wxs_features(source_table, patient_ids, min_prevalence)
    wxs_distance, source_views["wxs"] = build_wxs_view(wxs_table[selected_features].to_numpy(), snf_config)
    empty_rows = np.flatnonzero(wxs_table[selected_features].astype(float).sum(axis=1).to_numpy() == 0)
    wxs_audit = {
        "source": "prevalence_only_nonsynonymous_binary_mutations",
        "min_gene_prevalence": min_prevalence,
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "empty_mutation_distance": EMPTY_MUTATION_DISTANCE,
        "distance_diagonal": 0.0,
        "empty_patient_count": int((wxs_table[selected_features].astype(float).sum(axis=1) == 0).sum()),
        "empty_pair_affinity_override": False,
        "empty_empty_distance_example": (
            float(wxs_distance[empty_rows[0], empty_rows[1]]) if len(empty_rows) > 1 else None
        ),
        "empty_empty_affinity": (
            {
                "mean": float(source_views["wxs"][np.ix_(empty_rows, empty_rows)][np.triu_indices(len(empty_rows), 1)].mean()),
                "minimum": float(source_views["wxs"][np.ix_(empty_rows, empty_rows)][np.triu_indices(len(empty_rows), 1)].min()),
                "maximum": float(source_views["wxs"][np.ix_(empty_rows, empty_rows)][np.triu_indices(len(empty_rows), 1)].max()),
            }
            if len(empty_rows) > 1 else None
        ),
    }
    fused = fuse_affinities({name: source_views[name] for name in ACTIVE_MODALITIES}, snf_config)
    input_root = prepare_variant_input(
        data_root, output_root, patient_ids, source_views, fused, wxs_table,
        ct_audit, wxs_audit, load_candidate_proposer_config(config_dir), force,
    )
    write_json(output_root / "ct_feature_audit.json", ct_audit)
    write_json(output_root / "wxs_feature_audit.json", wxs_audit)
    manifest = {
        "experiment": "four_view_feature_engineering_variant",
        "active_modalities": list(ACTIVE_MODALITIES),
        "disabled_modalities": ["cnv"],
        "patient_count": len(patient_ids),
        "initial_ks": list(initial_ks),
        "repeats": list(repeats),
        "ct_pipeline": "PyRadiomics -> low variance -> |r| > 0.95 pruning -> technical residualization -> z-score -> Euclidean",
        "modality_normalization": {
            "ct": "feature-wise z-score after technical residualization; Euclidean",
            "wsi": "existing row-wise L2-normalized GigaPath embeddings; cosine; affinity reused unchanged",
            "rna": "existing log2(TPM+1), top-MAD genes, gene-wise cohort z-score; Pearson correlation; affinity reused unchanged",
            "wxs": "prevalence-filtered nonsynonymous binary events; no scaling; Jaccard",
        },
        "shared_affinity_construction": {
            "function": "distance_to_affinity",
            "neighbor_count": int(snf_config["neighbor_count"]),
            "mu": float(snf_config["mu"]),
        },
        "wxs_driver_forced_inclusion": False,
        "wxs_empty_mutation_distance": EMPTY_MUTATION_DISTANCE,
        "rna_unchanged": True,
        "wsi_unchanged": True,
        "input_root": str(input_root),
    }
    write_json(output_root / "experiment_manifest.json", manifest)
    if preflight:
        return {"preflight": "passed", **manifest}
    result = runner.run(
        input_root, config_dir, output_root / "agent_review", initial_ks, repeats,
        force=force, active_modalities=ACTIVE_MODALITIES,
    )
    return {"agent": result, **manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc_v14/11_four_view_no_cnv/inputs/four_view_no_cnv")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v15/01_four_view_feature_engineering")
    parser.add_argument("--initial-k", dest="initial_ks", type=int, choices=range(2, 9), action="append")
    parser.add_argument("--repeat", dest="repeats", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    initial_ks = tuple(sorted(set(args.initial_ks or range(2, 9))))
    repeats = tuple(sorted(set(args.repeats or (1, 2, 3))))
    print(json.dumps(run(args.data_root, args.config_dir, args.output_root, initial_ks, repeats, args.force, args.preflight), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
