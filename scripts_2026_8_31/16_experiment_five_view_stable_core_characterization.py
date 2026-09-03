#!/usr/bin/env python3
"""Unified multimodal characterization of the 5-view stable cores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts_2026_8_31.analyze_multi_k_stable_cores import (
    run as characterize_stable_cores,
    wsi_embedding_table,
)


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
    result = characterize_stable_cores(
        args.data_root,
        args.stable_core_root,
        args.output_root,
        args.config_dir,
        args.top_pathways,
        args.top_cnv,
        args.random_state,
        args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
