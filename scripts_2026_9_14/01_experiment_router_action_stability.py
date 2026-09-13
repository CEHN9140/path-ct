#!/usr/bin/env python3
"""Repeat each controlled Router action case without within-round aggregation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "router_action_sanity",
    Path(__file__).with_name("00_experiment_router_action_sanity.py"),
)
SANITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SANITY)

from agents.subtype_review.llm import LLMUsageTracker, build_default_router, review_signature_manifest
from utils.io import write_json
from utils.llm_utils import load_yaml_file


DEFAULT_REPEATS = 5


def summarize_replays(name: str, results: list[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    signatures = {
        json.dumps(result.get("actual_actions", {}), sort_keys=True)
        for result in successful
    }
    return {
        "case": name,
        "repeats": len(results),
        "successful_replays": len(successful),
        "correct_replays": sum(bool(result.get("passed")) for result in results),
        "router_errors": sum(result.get("status") == "router_error" for result in results),
        "action_consistency": len(successful) == len(results) and len(signatures) <= 1,
        "expected_actions": dict(SANITY.CASES[name]["expected"]),
        "action_counts": {
            action: sum(
                result.get("actual_actions", {}).get(target) == action
                for result in successful
                for target in result.get("actual_actions", {})
            )
            for action in ("accept", "drop", "split", "merge")
        },
        "results": [
            {
                key: value
                for key, value in result.items()
                if key != "payload"
            }
            for result in results
        ],
    }


def run(
    config_dir: Path,
    output_root: Path,
    repeats: int = DEFAULT_REPEATS,
    force: bool = False,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_yaml_file(config_dir / "subtype_review.yaml")
    usage = LLMUsageTracker()
    router = build_default_router(config, config_dir, usage_tracker=usage)
    cases = {
        name: summarize_replays(
            name,
            [SANITY.run_case(name, router) for _ in range(repeats)],
        )
        for name in SANITY.CASES
    }
    for name, result in cases.items():
        write_json(output_root / f"{name}_stability.json", result)
    summary = {
        "experiment": "router_action_stability",
        "repeats_per_case": repeats,
        "cases": cases,
        "total_replays": repeats * len(cases),
        "successful_replays": sum(item["successful_replays"] for item in cases.values()),
        "correct_replays": sum(item["correct_replays"] for item in cases.values()),
        "router_errors": sum(item["router_errors"] for item in cases.values()),
        "llm_usage": usage.snapshot(),
        "review_signature": review_signature_manifest(config, config_dir),
    }
    write_json(output_root / "router_action_stability_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14" / "01_router_action_stability")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config_dir, args.output_root, args.repeats, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
