#!/usr/bin/env python3
"""Unified multimodal characterization of the 5-view stable cores."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts_2026_8_31.analyze_multi_k_stable_cores import (
    load_cores,
    run as characterize_stable_cores,
    wsi_embedding_table,
)
from scripts_2026_8_31.five_view_experiment import load_five_view_inputs


def validate_stable_cores(stable_core_root, patient_ids):
    cores, _ = load_cores(stable_core_root)
    members = [patient_id for values in cores.values() for patient_id in values]
    if len(cores) != 6 or sorted(map(len, cores.values())) != [5, 9, 10, 14, 14, 17]:
        raise ValueError("15号stable-core名单不是预期的6组固定规模")
    if len(members) != 69 or len(set(members)) != 69:
        raise ValueError("15号stable-core病例必须是互不重叠的69例")
    if not set(members).issubset(patient_ids):
        raise ValueError("stable-core病例不完全存在于5-view patient order中")
    return cores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument(
        "--stable-core-root",
        type=Path,
        default=ROOT / "output_kirc_v12/15_five_view_multi_k_stability",
    )
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output_kirc_v12/16_five_view_stable_core_characterization",
    )
    parser.add_argument("--top-pathways", type=int, default=25)
    parser.add_argument("--top-cnv", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"Output exists; pass --force: {args.output_root}")
    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    validate_stable_cores(args.stable_core_root, set(np.load(args.stable_core_root / "affinity_patient_order.npy", allow_pickle=True).tolist()))
    patient_ids, matrices, _, _ = load_five_view_inputs(
        args.data_root,
        args.stable_core_root,
        args.output_root,
        args.config_dir,
    )
    fused_path = args.stable_core_root / "fused_similarity_5view.npy"
    order_path = args.stable_core_root / "affinity_patient_order.npy"
    if not fused_path.is_file() or not order_path.is_file():
        raise FileNotFoundError("15号5-view实验缺少fused similarity或patient order")
    saved_ids = [str(item) for item in np.load(order_path, allow_pickle=True).tolist()]
    if saved_ids != patient_ids:
        raise ValueError("15号5-view patient order does not match affinity inputs")
    fused = np.load(fused_path)
    affinities = {
        name: matrices[name]
        for name in ("ct", "wsi", "rna", "wxs", "cnv")
    }
    affinities["fused"] = fused
    result = characterize_stable_cores(
        args.data_root,
        args.stable_core_root,
        args.output_root,
        args.config_dir,
        args.top_pathways,
        args.top_cnv,
        args.random_state,
        True,
        affinities,
        patient_ids,
        ROOT / "output_kirc_v12/14_five_view_primary_discovery",
    )
    old_mapping = args.output_root / "stable_core_main_mapping.csv"
    if old_mapping.exists():
        old_mapping.replace(args.output_root / "five_view_core_to_four_view_primary_mapping.csv")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
