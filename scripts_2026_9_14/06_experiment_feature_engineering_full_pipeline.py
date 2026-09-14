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


def prepare_input(data_root, input_root, variant, patient_ids, views, fused, config, force):
    manifest_path = input_root / "candidate_subtype" / "feature_engineering_variant.json"
    expected = {
        "variant": variant,
        "patient_count": len(patient_ids),
        "fused_sha256": hashlib.sha256(np.asarray(fused).tobytes()).hexdigest(),
        "patient_order": patient_ids,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            return False
        if not force:
            raise RuntimeError(f"Variant input identity mismatch: {input_root}; use --force to rebuild it")
    if force:
        shutil.rmtree(input_root, ignore_errors=True)
    input_root.mkdir(parents=True, exist_ok=True)
    candidate_source = data_root / "candidate_subtype"
    candidate_dir = input_root / "candidate_subtype"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(candidate_source, candidate_dir)

    for name in ("storage", "ct_radiomics", "ct_qc", "rna", "cnv"):
        link_or_copy(data_root / name, input_root / name)
    shutil.copytree(data_root / "wxs", input_root / "wxs")

    for name in ("ct", "wsi", "rna"):
        path = candidate_dir / f"{name}_affinity.npy" if name in {"ct", "wsi", "rna"} else input_root / "wxs" / f"{name}_affinity.npy"
        np.save(path, views[name])
    for name in ("wxs", "cnv"):
        np.save(input_root / "wxs" / f"{name}_affinity.npy", views[name])
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
        **expected,
        "input_source": str(data_root.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "full_pipeline": True,
    })
    write_consensus_partitions(candidate_dir, fused, patient_ids, config)
    return True


def validate_variant_input(input_root, runner, patient_ids, views, fused, initial_ks):
    loaded_ids, loaded_views, loaded_fused = runner.load_main_inputs(input_root)
    if loaded_ids != patient_ids:
        raise ValueError("Variant patient order changed during input preparation")
    for name in ("ct", "wsi", "rna", "wxs", "cnv"):
        if not np.allclose(loaded_views[name], views[name], atol=1e-10):
            raise ValueError(f"Variant {name} affinity does not match the constructed view")
    if not np.allclose(loaded_fused, fused, atol=1e-10):
        raise ValueError("Variant fused similarity does not match the constructed views")
    expected = set(patient_ids)
    for initial_k in initial_ks:
        groups = runner.load_initial_partition(input_root, initial_k, patient_ids)
        if len(groups) != initial_k or set().union(*(set(group["member_ids"]) for group in groups)) != expected:
            raise ValueError(f"Variant K={initial_k} partition does not cover the patient universe")
        if sum(len(group["member_ids"]) for group in groups) != len(patient_ids):
            raise ValueError(f"Variant K={initial_k} partition contains duplicate patients")


def remove_incomplete_runs(review_root, initial_ks, repeats):
    for repeat in repeats:
        for initial_k in initial_ks:
            run_root = review_root / f"run{repeat}" / f"K{initial_k}"
            summary_path = run_root / "final_review_summary.json"
            metadata_path = run_root / "run_metadata.json"
            complete = False
            if summary_path.is_file() and metadata_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                complete = summary.get("status") == "review_complete" and summary.get("raw_control_status") == "complete"
            if run_root.exists() and not complete:
                shutil.rmtree(run_root)


def run_variant(variant, data_root, config_dir, output_root, initial_ks, repeats, force, preflight):
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
    views, _ = variants[variant]
    fused = MODULE.fuse(views, config["snf"])

    variant_root = output_root / "inputs" / variant
    review_root = output_root / variant / "agent_review"
    views_changed = prepare_input(data_root, variant_root, variant, patient_ids, views, fused, config, force)
    validate_variant_input(variant_root, multi_k_module, patient_ids, views, fused, initial_ks)
    if views_changed and review_root.exists() and not preflight:
        shutil.rmtree(review_root)
    elif force and not preflight:
        remove_incomplete_runs(review_root, initial_ks, repeats)
    if preflight:
        return {"variant": variant, "preflight": "passed", "input_root": str(variant_root)}

    result = multi_k_module.run(
        data_root=variant_root,
        config_dir=config_dir,
        output_root=review_root,
        initial_ks=initial_ks,
        repeats=repeats,
        force=False,
    )
    write_json(output_root / variant / "variant_manifest.json", {
        "variant": variant,
        "data_root": str(variant_root),
        "review_root": str(review_root),
        "initial_ks": list(initial_ks),
        "repeats": list(repeats),
        "input_rebuilt": views_changed,
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
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    variants = tuple(dict.fromkeys(args.variant or VARIANTS))
    initial_ks = tuple(sorted(set(args.initial_ks or range(2, 9))))
    repeats = tuple(sorted(set(args.repeats or (1, 2, 3))))
    results = [run_variant(variant, args.data_root, args.config_dir, args.output_root, initial_ks, repeats, args.force, args.preflight) for variant in variants]
    print(json.dumps({"variants": variants, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
