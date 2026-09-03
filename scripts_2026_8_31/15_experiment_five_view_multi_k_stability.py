#!/usr/bin/env python3
"""Five-view K=2..8 repeated Agent review followed by stable-core analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import save_review_outputs
from agents.subtype_review.runner import run_subtype_review
from scripts_2026_8_31.experiment_initial_k_review_sensitivity import load_patient_states
from scripts_2026_8_31.experiment_multi_k_accepted_core_stability import analyze, scientifically_terminal
from scripts_2026_8_31.five_view_experiment import build_consensus, candidate_sets, load_five_view_inputs
from utils.io import write_json

INITIAL_KS = tuple(range(2, 9))


def parse_repeats(values):
    return tuple(sorted(set(values or (1, 2, 3))))


def run(data_root: Path, config_dir: Path, output_root: Path, repeats: tuple[int, ...], force: bool = False, run_agent: bool = False):
    output_root.mkdir(parents=True, exist_ok=True)
    stable_root = ROOT / "output_kirc_v12/03_multi_k_accepted_core_stability_v11"
    patient_ids, _, fused, config = load_five_view_inputs(data_root, stable_root, output_root, config_dir)
    records, _ = build_consensus(fused, config, output_root, patient_ids)
    record_by_k = {int(record["n_clusters"]): record for record in records}
    write_json(output_root / "experiment_manifest.json", {
        "experiment": "five_view_multi_k_stability",
        "view": "ct_wsi_rna_wxs_cnv",
        "initial_k": list(INITIAL_KS),
        "repeats": list(repeats),
        "patient_count": len(patient_ids),
        "agent_calls_enabled": run_agent,
    })
    if run_agent:
        patient_states = load_patient_states(data_root)
        rows = []
        for repeat in repeats:
            for initial_k in INITIAL_KS:
                run_root = output_root / f"run{repeat}" / f"K{initial_k}"
                if force and run_root.exists():
                    shutil.rmtree(run_root)
                if run_root.exists() and not force:
                    summary_path = run_root / "final_review_summary.json"
                    metadata_path = run_root / "run_metadata.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                    if summary_path.exists() and metadata.get("repeat") == repeat and metadata.get("initial_k") == initial_k:
                        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                        continue
                run_root.mkdir(parents=True, exist_ok=True)
                initial_sets = candidate_sets(record_by_k[initial_k]["labels"], patient_ids, ["ct", "wsi", "rna", "wxs", "cnv"])
                write_json(run_root / "initial_partition.json", {
                    "initial_k": initial_k, "repeat": repeat,
                    "view": "ct_wsi_rna_wxs_cnv", "candidate_sets": initial_sets,
                })
                state = run_subtype_review(initial_sets, patient_states, str(data_root), str(config_dir), artifact_root=str(run_root))
                summary = save_review_outputs(state, str(run_root), direct=True)
                write_json(run_root / "run_metadata.json", {
                    "experiment": "five_view_multi_k_stability",
                    "repeat": repeat, "initial_k": initial_k,
                    "patient_count": len(patient_ids),
                    "valid_for_analysis": scientifically_terminal(summary),
                    "terminal_status": summary.get("status"),
                })
                rows.append(summary)
        write_json(output_root / "agent_discovery_summary.json", {"runs": rows})
        analyze(
            output_root,
            patient_ids,
            INITIAL_KS,
            repeats,
            min_core_size=5,
        )
    return {"output_root": str(output_root), "initial_k": list(INITIAL_KS), "repeats": list(repeats), "agent_calls_enabled": run_agent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v12/15_five_view_multi_k_stability")
    parser.add_argument("--repeat", dest="repeats", type=int, action="append")
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.repeats = parse_repeats(args.repeats)
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
