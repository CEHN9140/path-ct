#!/usr/bin/env python3
"""Run the complete CT/WSI/RNA/WXS pipeline without any CNV input."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.candidate_proposer import consensus_records_from_similarity
from tools.evidence_features import fuse_affinities
from utils.io import write_json
from utils.llm_utils import load_candidate_proposer_config

ACTIVE_MODALITIES = ("ct", "wsi", "rna", "wxs")


def load_runner():
    path = ROOT / "scripts_2026_9_7" / "00_experiment_five_view_multi_k_agent_review.py"
    spec = importlib.util.spec_from_file_location("four_view_agent_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def link_directory(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.symlink_to(source.resolve(), target_is_directory=True)


def write_consensus(candidate_dir, fused, patient_ids, clustering):
    records, _ = consensus_records_from_similarity(fused, clustering)
    consensus_dir = candidate_dir / "consensus_cluster"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        k = int(record["n_clusters"])
        write_json(consensus_dir / f"consensus_hierarchical_K{k}.json", {
            "n_clusters": k,
            "labels": dict(zip(patient_ids, map(int, record["labels"]))),
            "partition_source": "production_consensus_records_from_similarity",
            "active_modalities": list(ACTIVE_MODALITIES),
        })


def prepare_input(data_root, output_root, patient_ids, views, fused, config, force):
    input_root = output_root / "inputs" / "four_view_no_cnv"
    candidate_dir = input_root / "candidate_subtype"
    if input_root.exists() and not force:
        raise FileExistsError(f"Input exists; pass --force to overwrite: {input_root}")
    shutil.rmtree(input_root, ignore_errors=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for name in ("storage", "ct_radiomics", "ct_qc", "rna", "wsi_tumor_seg"):
        link_directory(data_root / name, input_root / name)
    wxs_dir = input_root / "wxs"
    wxs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("wxs_discovery_features.csv", "wxs_validation_features.csv", "wxs_affinity.npy"):
        shutil.copy2(data_root / "wxs" / name, wxs_dir / name)

    write_json(candidate_dir / "affinity_patient_order.json", patient_ids)
    paths = {}
    for name in ACTIVE_MODALITIES:
        source = views[name]
        if name in {"ct", "wsi", "rna"}:
            target = candidate_dir / f"{name}_affinity.npy"
            paths[name] = target.name
        else:
            target = wxs_dir / f"{name}_affinity.npy"
            paths[name] = f"wxs/{target.name}"
        np.save(target, source)
    write_json(candidate_dir / "affinity_cache.json", {
        "cache_version": 1,
        "variant": "four_view_no_cnv",
        "active_modalities": list(ACTIVE_MODALITIES),
        "disabled_modalities": ["cnv"],
        "patient_ids": patient_ids,
        "paths": paths,
    })
    np.save(candidate_dir / "fused_similarity.npy", fused)
    write_consensus(candidate_dir, fused, patient_ids, config["clustering"])
    return input_root


def run(data_root, config_dir, output_root, initial_ks, repeats, force=False, preflight=False):
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    patient_ids, views, _ = runner.load_main_inputs(Path(data_root), ACTIVE_MODALITIES)
    config = load_candidate_proposer_config(Path(config_dir))
    fused = fuse_affinities(
        {name: views[name] for name in ACTIVE_MODALITIES}, config["snf"]
    )
    input_root = prepare_input(
        Path(data_root), output_root, patient_ids, views, fused, config, force
    )
    manifest = {
        "experiment": "four_view_no_cnv",
        "active_modalities": list(ACTIVE_MODALITIES),
        "disabled_modalities": ["cnv"],
        "patient_count": len(patient_ids),
        "initial_ks": list(initial_ks),
        "repeats": list(repeats),
        "input_root": str(input_root),
        "cnv_artifact_present": (input_root / "cnv").exists(),
    }
    write_json(output_root / "experiment_manifest.json", manifest)
    if preflight:
        return {"preflight": "passed", **manifest}
    result = runner.run(
        input_root,
        Path(config_dir),
        output_root / "agent_review",
        initial_ks,
        repeats,
        force=force,
        active_modalities=ACTIVE_MODALITIES,
    )
    return {"agent": result, **manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/11_four_view_no_cnv")
    parser.add_argument("--initial-k", dest="initial_ks", type=int, choices=range(2, 9), action="append")
    parser.add_argument("--repeat", dest="repeats", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    initial_ks = tuple(sorted(set(args.initial_ks or range(2, 9))))
    repeats = tuple(sorted(set(args.repeats or (1, 2, 3))))
    print(json.dumps(run(
        args.data_root, args.config_dir, args.output_root,
        initial_ks, repeats, args.force, args.preflight,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
