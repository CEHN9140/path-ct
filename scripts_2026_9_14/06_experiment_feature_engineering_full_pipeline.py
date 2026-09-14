#!/usr/bin/env python3
"""Run isolated full-pipeline feature-engineering sensitivity experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.candidate_proposer import consensus_records_from_similarity
from utils.io import write_json
from utils.llm_utils import load_candidate_proposer_config

SENSITIVITY = ROOT / "scripts_2026_9_14" / "05_experiment_feature_engineering_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("feature_sensitivity", SENSITIVITY)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

VARIANTS = ("rna_top_3000", "wxs_prevalence_only", "wxs_zero_distance_05")


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_consensus_partitions(candidate_dir, fused, patient_ids, config):
    records, _ = consensus_records_from_similarity(fused, config["clustering"])
    consensus_dir = candidate_dir / "consensus_cluster"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        k = int(record["n_clusters"])
        labels = [int(value) for value in record["labels"]]
        write_json(consensus_dir / f"consensus_hierarchical_K{k}.json", {
            "n_clusters": k,
            "labels": dict(zip(patient_ids, labels)),
            "variant": True,
            "partition_source": "production_consensus_records_from_similarity",
        })


def link_or_copy(source, target):
    if target.exists() or target.is_symlink():
        target.unlink() if target.is_symlink() else shutil.rmtree(target)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_input(data_root, input_root, variant, patient_ids, affinities, fused, config):
    input_root.mkdir(parents=True, exist_ok=True)
    candidate_source = data_root / "candidate_subtype"
    candidate_dir = input_root / "candidate_subtype"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(candidate_source, candidate_dir)

    for name in ("storage", "ct_radiomics", "ct_qc", "rna", "cnv"):
        link_or_copy(data_root / name, input_root / name)
    shutil.copytree(data_root / "wxs", input_root / "wxs")

    for name in ("ct", "wsi", "rna", "wxs", "cnv"):
        path = candidate_dir / f"{name}_affinity.npy" if name in {"ct", "wsi", "rna"} else input_root / "wxs" / f"{name}_affinity.npy"
        np.save(path, affinities[name])
    np.save(candidate_dir / "fused_similarity.npy", fused)

    if variant == "wxs_prevalence_only":
        wxs_dir = input_root / "wxs"
        table = pd.read_csv(data_root / "wxs" / "wxs_discovery_features.csv")
        feature_columns = [column for column in table.columns if column != "case_id"]
        prevalence = table[feature_columns].mean()
        selected = prevalence[prevalence >= 0.05].index.tolist()
        table[["case_id", *selected]].to_csv(wxs_dir / "wxs_discovery_features.csv", index=False)

    write_json(candidate_dir / "feature_engineering_variant.json", {
        "variant": variant,
        "views": ["ct", "wsi", "rna", "wxs", "cnv"],
        "patient_count": len(patient_ids),
        "input_source": str(data_root.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "full_pipeline": True,
    })
    write_consensus_partitions(candidate_dir, fused, patient_ids, config)


def run_variant(variant, data_root, config_dir, output_root, initial_ks, repeats, force):
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    multi_k = importlib.util.spec_from_file_location(
        "multi_k_runner", ROOT / "scripts_2026_9_7" / "00_experiment_five_view_multi_k_agent_review.py"
    )
    multi_k_module = importlib.util.module_from_spec(multi_k)
    multi_k.loader.exec_module(multi_k_module)
    patient_ids, canonical, _ = multi_k_module.load_main_inputs(data_root)
    config = load_candidate_proposer_config(config_dir)
    variants = MODULE.load_variants(data_root, patient_ids, canonical, config["snf"])
    fused, _ = variants[variant]

    variant_root = output_root / "inputs" / variant
    review_root = output_root / variant / "agent_review"
    if force:
        shutil.rmtree(variant_root, ignore_errors=True)
        shutil.rmtree(review_root, ignore_errors=True)
    prepare_input(data_root, variant_root, variant, patient_ids, canonical, fused, config)

    result = multi_k_module.run(
        data_root=variant_root,
        config_dir=config_dir,
        output_root=review_root,
        initial_ks=initial_ks,
        repeats=repeats,
        force=force,
    )
    write_json(output_root / variant / "variant_manifest.json", {
        "variant": variant,
        "data_root": str(variant_root),
        "review_root": str(review_root),
        "initial_ks": list(initial_ks),
        "repeats": list(repeats),
        "result": result,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/06_feature_engineering_full_pipeline")
    parser.add_argument("--initial-k", dest="initial_ks", type=int, choices=range(2, 9), action="append")
    parser.add_argument("--repeat", dest="repeats", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    variants = tuple(dict.fromkeys(args.variant or VARIANTS))
    initial_ks = tuple(sorted(set(args.initial_ks or range(2, 9))))
    repeats = tuple(sorted(set(args.repeats or (1, 2, 3))))
    if args.force and args.output_root.exists():
        for variant in variants:
            shutil.rmtree(args.output_root / "inputs" / variant, ignore_errors=True)
            shutil.rmtree(args.output_root / variant, ignore_errors=True)
    results = [run_variant(variant, args.data_root, args.config_dir, args.output_root, initial_ks, repeats, args.force) for variant in variants]
    print(json.dumps({"variants": variants, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
