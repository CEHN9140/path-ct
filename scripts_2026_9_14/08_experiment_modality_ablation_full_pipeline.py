#!/usr/bin/env python3
"""Leave-CT-out five-view ablation with an optional Agent run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.candidate_proposer import consensus_records_from_similarity
from tools.evidence_features import fuse_affinities
from utils.io import write_json
from utils.llm_utils import load_candidate_proposer_config

ACTIVE_MODALITIES = ("wsi", "rna", "wxs", "cnv")
ALL_MODALITIES = ("ct", *ACTIVE_MODALITIES)


def load_runner():
    path = ROOT / "scripts_2026_9_7" / "00_experiment_five_view_multi_k_agent_review.py"
    spec = importlib.util.spec_from_file_location("multi_k_agent_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_canonical_inputs(data_root):
    runner = load_runner()
    patient_ids, views, _, = runner.load_main_inputs(data_root)
    return patient_ids, views


def fuse_active_views(views, snf_config):
    return fuse_affinities({name: views[name] for name in ACTIVE_MODALITIES}, snf_config)


def variant_manifest(patient_ids, fused_sha256, patient_order_sha256):
    return {
        "variant": "leave_ct_out",
        "active_modalities": list(ACTIVE_MODALITIES),
        "disabled_modalities": ["ct"],
        "cohort_n": len(patient_ids),
        "fused_sha256": fused_sha256,
        "patient_order_sha256": patient_order_sha256,
    }


def write_consensus_partitions(candidate_dir, fused, patient_ids, clustering_config):
    records, _ = consensus_records_from_similarity(fused, clustering_config)
    consensus_dir = candidate_dir / "consensus_cluster"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        k = int(record["n_clusters"])
        labels = [int(value) for value in record["labels"]]
        write_json(consensus_dir / f"consensus_hierarchical_K{k}.json", {
            "n_clusters": k,
            "labels": dict(zip(patient_ids, labels)),
            "partition_source": "production_consensus_records_from_similarity",
            "active_modalities": list(ACTIVE_MODALITIES),
        })


def link_directory(source, target):
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.symlink_to(source.resolve(), target_is_directory=True)


def prepare_variant(data_root, output_root, patient_ids, views, fused, config, force):
    input_root = output_root / "inputs" / "leave_ct_out"
    candidate_source = data_root / "candidate_subtype"
    candidate_dir = input_root / "candidate_subtype"
    manifest_path = candidate_dir / "modality_ablation_manifest.json"
    expected = variant_manifest(
        patient_ids,
        hashlib.sha256(np.asarray(fused).tobytes()).hexdigest(),
        file_sha256(data_root / "candidate_subtype" / "affinity_patient_order.json"),
    )
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) == expected:
        return input_root, False
    if input_root.exists() and not force:
        raise FileExistsError(f"Variant input exists with a different identity: {input_root}")
    shutil.rmtree(input_root, ignore_errors=True)
    input_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate_source, candidate_dir)
    for name in ("storage", "ct_radiomics", "ct_qc", "rna", "cnv", "wsi_tumor_seg"):
        link_directory(data_root / name, input_root / name)
    shutil.copytree(data_root / "wxs", input_root / "wxs")
    for name in ALL_MODALITIES:
        source = views[name]
        target = candidate_dir / f"{name}_affinity.npy" if name in {"ct", "wsi", "rna"} else input_root / "wxs" / f"{name}_affinity.npy"
        np.save(target, source)
    np.save(candidate_dir / "fused_similarity.npy", fused)
    write_consensus_partitions(candidate_dir, fused, patient_ids, config["clustering"])
    write_json(manifest_path, expected)
    return input_root, True


def validate_preflight(input_root, patient_ids, views, fused, initial_ks):
    runner = load_runner()
    loaded_ids, loaded_views, loaded_fused = runner.load_main_inputs(input_root)
    if loaded_ids != patient_ids:
        raise ValueError("Leave-CT-out patient order differs from canonical cohort")
    for name in ALL_MODALITIES:
        if not np.allclose(loaded_views[name], views[name], atol=1e-10):
            raise ValueError(f"Variant {name} affinity differs from canonical input")
    if not np.allclose(loaded_fused, fused, atol=1e-10):
        raise ValueError("Leave-CT-out fused matrix differs from four-view fusion")
    if not np.isfinite(fused).all() or not np.allclose(fused, fused.T, atol=1e-8):
        raise ValueError("Leave-CT-out fused matrix is not finite and symmetric")
    if not np.allclose(fused.diagonal(), 1.0, atol=1e-8):
        raise ValueError("Leave-CT-out fused diagonal is invalid")
    expected = set(patient_ids)
    for initial_k in initial_ks:
        groups = runner.load_initial_partition(input_root, initial_k, patient_ids)
        members = [case for group in groups for case in group["member_ids"]]
        if len(groups) != initial_k or len(members) != len(set(members)) or set(members) != expected:
            raise ValueError(f"Leave-CT-out K={initial_k} partition is invalid")


def compare_cores(canonical, variant):
    canonical_ids = sorted(canonical)
    variant_ids = sorted(variant)
    reference = [set(canonical[core_id]) for core_id in canonical_ids]
    candidate = [set(variant[core_id]) for core_id in variant_ids]
    if not reference:
        return []
    if not candidate:
        return [
            {
                "canonical_core": core_id,
                "variant_core": None,
                "canonical_size": len(members),
                "variant_size": 0,
                "intersection": 0,
                "canonical_retention": 0.0,
                "variant_purity": 0.0,
                "jaccard": 0.0,
            }
            for core_id, members in zip(canonical_ids, reference)
        ]
    scores = np.asarray([[len(left & right) / len(left | right) for right in candidate] for left in reference])
    rows, columns = linear_sum_assignment(1.0 - scores)
    matched = {row: (columns[index], scores[row, columns[index]]) for index, row in enumerate(rows)}
    result = []
    for index, members in enumerate(reference):
        candidate_index, jaccard = matched.get(index, (None, 0.0))
        other = candidate[candidate_index] if candidate_index is not None else set()
        result.append({
            "canonical_core": canonical_ids[index],
            "variant_core": variant_ids[candidate_index] if candidate_index is not None else None,
            "canonical_size": len(members),
            "variant_size": len(other),
            "intersection": len(members & other),
            "canonical_retention": len(members & other) / len(members),
            "variant_purity": len(members & other) / len(other) if other else 0.0,
            "jaccard": float(jaccard),
        })
    return result


def summarize_core_comparison(canonical, variant, comparison):
    canonical_n = sum(map(len, canonical.values()))
    variant_n = sum(map(len, variant.values()))
    matched_rows = [row for row in comparison if row["variant_core"] is not None]
    intersections = sum(row["intersection"] for row in comparison)
    matched_variant_ids = {row["variant_core"] for row in matched_rows}
    return {
        "canonical_core_count": len(canonical),
        "variant_core_count": len(variant),
        "canonical_core_coverage": intersections / canonical_n if canonical_n else None,
        "variant_core_coverage": intersections / variant_n if variant_n else None,
        "matched_core_mean_jaccard": (
            float(np.mean([row["jaccard"] for row in matched_rows]))
            if matched_rows else None
        ),
        "penalized_core_mean_jaccard": (
            float(np.mean([row["jaccard"] for row in comparison]))
            if comparison else None
        ),
        "unmatched_canonical_cores": [
            row["canonical_core"] for row in comparison if row["variant_core"] is None
        ],
        "extra_variant_cores": sorted(set(variant) - matched_variant_ids),
    }


def run(data_root, config_dir, output_root, initial_ks, repeats, force=False, preflight=False, run_agent=False):
    config = load_candidate_proposer_config(config_dir)
    patient_ids, views = load_canonical_inputs(data_root)
    fused = fuse_active_views(views, config["snf"])
    input_root, rebuilt = prepare_variant(data_root, output_root, patient_ids, views, fused, config, force)
    validate_preflight(input_root, patient_ids, views, fused, initial_ks)
    manifest = variant_manifest(
        patient_ids,
        hashlib.sha256(np.asarray(fused).tobytes()).hexdigest(),
        file_sha256(data_root / "candidate_subtype" / "affinity_patient_order.json"),
    )
    manifest.update({"input_root": str(input_root), "input_rebuilt": rebuilt, "initial_ks": list(initial_ks), "repeats": list(repeats)})
    write_json(output_root / "leave_ct_out_manifest.json", manifest)
    if preflight or not run_agent:
        return {"preflight": "passed", **manifest}
    runner = load_runner()
    review_root = output_root / "leave_ct_out" / "agent_review"
    result = runner.run(
        data_root=input_root,
        config_dir=config_dir,
        output_root=review_root,
        initial_ks=initial_ks,
        repeats=repeats,
        force=force,
        active_modalities=ACTIVE_MODALITIES,
    )
    canonical_root = ROOT / "output_kirc_v13" / "00_five_view_multi_k_agent_review"
    variant_summary = review_root / "stable_core_summary.csv"
    if variant_summary.is_file():
        canonical_cores, _ = runner.core_analysis.load_cores(canonical_root)
        variant_cores, _ = runner.core_analysis.load_cores(review_root)
        comparison = compare_cores(canonical_cores, variant_cores)
        runner.core_analysis.write_csv(
            output_root / "canonical_vs_leave_ct_out_cores.csv", comparison
        )
        write_json(
            output_root / "canonical_vs_leave_ct_out_summary.json",
            summarize_core_comparison(canonical_cores, variant_cores, comparison),
        )
    return {"agent": result, **manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/08_modality_ablation")
    parser.add_argument("--initial-k", dest="initial_ks", type=int, choices=range(2, 9), action="append")
    parser.add_argument("--repeat", dest="repeats", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    initial_ks = tuple(sorted(set(args.initial_ks or range(2, 9))))
    repeats = tuple(sorted(set(args.repeats or (1, 2, 3))))
    print(json.dumps(run(args.data_root, args.config_dir, args.output_root, initial_ks, repeats, args.force, args.preflight, args.run_agent), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
