#!/usr/bin/env python3
"""Run and summarize full leave-one-modality-out ablations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.io import write_json
ALL_MODALITIES = ("ct", "wsi", "rna", "wxs", "cnv")
VARIANTS = {name: tuple(value for value in ALL_MODALITIES if value != name) for name in ALL_MODALITIES}


def load_ablation():
    path = ROOT / "scripts_2026_9_14" / "08_experiment_modality_ablation_full_pipeline.py"
    spec = importlib.util.spec_from_file_location("modality_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runner():
    return load_ablation().load_runner()


def canonical_row(canonical):
    n = sum(map(len, canonical.values()))
    return {
        "setting": "full_5view",
        "disabled_modality": None,
        "stable_core_count": len(canonical),
        "stable_core_patient_count": n,
        "canonical_matched_jaccard": 1.0,
        "canonical_matched_core_coverage": 1.0,
        "any_variant_stable_core_coverage": 1.0,
        "no_variant_stable_core_fraction": 0.0,
        "loco_mean_matched_jaccard": None,
        "loco_min_matched_jaccard": None,
    }


def variant_row(name, root, runner):
    summary_path = root / f"canonical_vs_leave_{name}_out_summary.json"
    review_root = root / f"leave_{name}_out" / "agent_review"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing completed {name}-out comparison: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    variant, _ = runner.core_analysis.load_cores(review_root)
    summary_path_loco = review_root / "leave_one_k_out_summary.json"
    loco = json.loads(summary_path_loco.read_text(encoding="utf-8"))["results"] if summary_path_loco.is_file() else []
    return {
        "setting": f"leave_{name}_out",
        "disabled_modality": name,
        "stable_core_count": summary["variant_core_count"],
        "stable_core_patient_count": sum(map(len, variant.values())),
        "canonical_matched_jaccard": summary["matched_core_mean_jaccard"],
        "canonical_matched_core_coverage": summary["canonical_core_coverage"],
        "any_variant_stable_core_coverage": summary["any_variant_stable_core_coverage"],
        "no_variant_stable_core_fraction": summary["no_variant_stable_core_fraction"],
        "loco_mean_matched_jaccard": (
            sum(row["matched_core_mean_jaccard"] for row in loco) / len(loco) if loco else None
        ),
        "loco_min_matched_jaccard": (
            min(row["matched_core_mean_jaccard"] for row in loco) if loco else None
        ),
    }


def run(data_root, config_dir, output_root, variants, initial_ks, repeats, force=False, preflight=False, run_agent=False):
    ablation = load_ablation()
    runner = load_runner()
    patient_ids, views = ablation.load_canonical_inputs(data_root)
    canonical_root = ROOT / "output_kirc_v13" / "00_five_view_multi_k_agent_review"
    canonical, _ = runner.core_analysis.load_cores(canonical_root)
    rows = [canonical_row(canonical)]
    for name in variants:
        variant_root = output_root / f"leave_{name}_out"
        active = VARIANTS[name]
        if name == "ct":
            source_root = ROOT / "output_kirc_v14" / "08_modality_ablation"
            if not (source_root / "leave_ct_out" / "agent_review" / "stable_core_summary.csv").is_file():
                raise FileNotFoundError("Existing 08 leave-CT-out result is required for 09 aggregation")
            rows.append(variant_row(name, source_root, runner))
            continue
        ablation.run(
            data_root=data_root,
            config_dir=config_dir,
            output_root=variant_root,
            initial_ks=initial_ks,
            repeats=repeats,
            force=force,
            preflight=preflight or not run_agent,
            run_agent=run_agent,
            active_modalities=active,
        )
        if not preflight and run_agent:
            rows.append(variant_row(name, variant_root, runner))
    write_json(output_root / "modality_ablation_summary.json", {
        "initial_ks": list(initial_ks),
        "repeats": list(repeats),
        "rows": rows,
    })
    return {"output_root": str(output_root), "variants": list(variants), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/09_full_leave_one_modality_out")
    parser.add_argument("--variant", choices=ALL_MODALITIES, action="append")
    parser.add_argument("--initial-k", dest="initial_ks", type=int, choices=range(2, 9), action="append")
    parser.add_argument("--repeat", dest="repeats", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    variants = tuple(args.variant or ALL_MODALITIES)
    initial_ks = tuple(sorted(set(args.initial_ks or range(2, 9))))
    repeats = tuple(sorted(set(args.repeats or (1, 2, 3))))
    print(json.dumps(run(args.data_root, args.config_dir, args.output_root, variants, initial_ks, repeats, args.force, args.preflight, args.run_agent), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
