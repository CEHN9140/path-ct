#!/usr/bin/env python3
"""Five-view primary discovery: exact consensus clustering, K selection, and one review."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.candidate_proposer import select_k_with_llm
from agents.subtype_review.graph import save_review_outputs
from agents.subtype_review.runner import run_subtype_review
from scripts_2026_8_31.experiment_initial_k_review_sensitivity import load_patient_states
from scripts_2026_8_31.five_view_experiment import (
    build_consensus,
    candidate_sets,
    load_five_view_inputs,
)
from utils.candidate_clustering_outputs import save_candidate_clustering_outputs
from utils.io import write_json
from utils.llm_utils import load_candidate_proposer_config

def run(data_root: Path, config_dir: Path, output_root: Path, force: bool = False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    stable_root = ROOT / "output_kirc_v12/03_multi_k_accepted_core_stability_v11"
    patient_ids, _, fused, config = load_five_view_inputs(data_root, stable_root, output_root, config_dir)
    records, partition_records = build_consensus(fused, config, output_root, patient_ids)
    decision = select_k_with_llm(
        records,
        output_root=str(output_root),
        config_dir=str(config_dir),
        min_cluster_size=int(config["clustering"]["min_cluster_size"]),
    )
    selected_k = int(decision["selected_k"])
    selected = next(record for record in records if int(record["n_clusters"]) == selected_k)
    save_candidate_clustering_outputs(
        str(output_root), patient_ids, partition_records, records, selected, snf_matrix=fused
    )
    initial_sets = candidate_sets(selected["labels"], patient_ids, ["ct", "wsi", "rna", "wxs", "cnv"])
    write_json(output_root / "initial_partition.json", {
        "initial_k": selected_k,
        "view": "ct_wsi_rna_wxs_cnv",
        "candidate_sets": initial_sets,
        "k_selection": decision,
    })
    patient_states = load_patient_states(data_root)
    state = run_subtype_review(initial_sets, patient_states, str(data_root), str(config_dir), artifact_root=str(output_root))
    summary = save_review_outputs(state, str(output_root), direct=True)
    write_json(output_root / "run_metadata.json", {
        "experiment": "five_view_primary_discovery",
        "view": "ct_wsi_rna_wxs_cnv",
        "selected_k": selected_k,
        "patient_count": len(patient_ids),
        "terminal_status": summary.get("status"),
        "llm_usage": summary.get("llm_usage", {}),
    })
    return {"output_root": str(output_root), "selected_k": selected_k, "status": summary.get("status")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v12/14_five_view_primary_discovery")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
