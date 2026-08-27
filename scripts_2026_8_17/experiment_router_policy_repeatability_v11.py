#!/usr/bin/env python3
"""Replay the V11 Router on saved V10 evidence without rerunning tools or Verifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.subtype_review.llm import (  # noqa: E402
    LLMUsageTracker,
    build_default_router,
    review_signature_manifest,
)
from scripts_2026_8_17.experiment_structural_index_router_replay import (  # noqa: E402
    action_summary,
    replay_entry,
)
from utils.llm_utils import load_yaml_file  # noqa: E402

INITIAL_KS = tuple(range(2, 9))
REPEATS = (1, 2, 3)


def plan_signature(plan: Mapping[str, Any]) -> str:
    actions = []
    for action in plan.get("actions", []) or []:
        actions.append({
            "action": action.get("action"),
            "target_ids": sorted(action.get("target_ids", []) or []),
            "tool_requests": sorted(
                (
                    request.get("tool_name"),
                    sorted(request.get("target_ids", []) or []),
                )
                for request in action.get("tool_requests", []) or []
            ),
        })
    return json.dumps(sorted(actions, key=lambda item: (item["action"], item["target_ids"])), sort_keys=True)


def load_history(run_root: Path) -> dict[str, Any]:
    path = run_root / "review_history.json"
    if not path.exists():
        return {"status": "missing"}
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        return {"status": "failed", "error": "empty review_history.json"}
    return {"status": "available", "entry": history[-1]}


def calibrate(
    experiment_root: Path,
    config_dir: Path,
    initial_ks: tuple[int, ...] = INITIAL_KS,
    repeats: tuple[int, ...] = REPEATS,
    output_root: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root or ROOT / "output_kirc_v11" / "router_policy_calibration"
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_yaml_file(config_dir / "subtype_review.yaml")
    router = build_default_router(config, config_dir, usage_tracker=LLMUsageTracker())
    manifest = review_signature_manifest(config, config_dir)
    (output_root / "prompt_manifest.json").write_text(
        json.dumps({**manifest, "model": config.get("llm", {}).get("model_name"), "temperature": config.get("llm", {}).get("temperature")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = {}
    for repeat in repeats:
        repeat_root = output_root / f"run{repeat}"
        repeat_root.mkdir(exist_ok=True)
        for initial_k in initial_ks:
            run_root = experiment_root / f"run{repeat}" / f"K{initial_k}"
            loaded = load_history(run_root)
            result = {"initial_k": initial_k, "repeat": repeat, **loaded}
            if loaded["status"] == "available":
                try:
                    payload, plan = replay_entry(loaded["entry"], router)
                    result.update({"status": "success", "plan": plan, "payload": payload})
                except Exception as exc:
                    result.update({"status": "router_error", "error": f"{type(exc).__name__}: {exc}"})
            results[(initial_k, repeat)] = result
            run_output = repeat_root / f"K{initial_k}"
            run_output.mkdir(exist_ok=True)
            (run_output / "router_plan.json").write_text(
                json.dumps({key: value for key, value in result.items() if key not in {"entry", "payload"}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    set_rows = []
    partition_rows = []
    for initial_k in initial_ks:
        successful = [
            result for repeat in repeats
            if (result := results[(initial_k, repeat)]).get("status") == "success"
        ]
        set_ids = sorted({
            set_id
            for result in successful
            for set_id in result["entry"].get("partition", {}).get("sets", [])
            for set_id in [str(set_id.get("set_id") or set_id.get("cluster_id") or "")]
            if set_id
        })
        signatures = [plan_signature(result["plan"]) for result in successful]
        pair_count = len(signatures) * (len(signatures) - 1) // 2
        partition_rows.append({
            "initial_k": initial_k,
            "successful_repeats": len(successful),
            "partition_plan_exact_match_fraction": (
                sum(left == right for left, right in combinations(signatures, 2))
                / pair_count if pair_count else None
            ),
        })
        for set_id in set_ids:
            actions = [action_summary(result["plan"]).get(set_id, "") for result in successful]
            counts = Counter(action for action in actions if action)
            set_rows.append({
                "initial_k": initial_k,
                "set_id": set_id,
                **{
                    f"repeat{repeat}_action": action_summary(results[(initial_k, repeat)].get("plan", {})).get(set_id, "")
                    if results[(initial_k, repeat)].get("status") == "success" else ""
                    for repeat in repeats
                },
                **{f"{action}_count": counts[action] for action in ("accept", "drop", "split", "merge", "need_more_evidence")},
                "decision_agreement_fraction": max(counts.values()) / len(actions) if actions else None,
                "decision_discordant": len(counts) > 1,
                "v10_action_used_as_ground_truth": False,
            })

    fields = list(set_rows[0]) if set_rows else []
    if fields:
        with (output_root / "router_repeatability.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(set_rows)
    discordant = [row for row in set_rows if row["decision_discordant"]]
    if discordant:
        with (output_root / "router_discordant_cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(discordant[0]))
            writer.writeheader()
            writer.writerows(discordant)

    summary = {
        "experiment": "router_policy_calibration_v11",
        "run_count": len(results),
        "successful_run_count": sum(result.get("status") == "success" for result in results.values()),
        "router_error_count": sum(result.get("status") == "router_error" for result in results.values()),
        "missing_run_count": sum(result.get("status") == "missing" for result in results.values()),
        "set_row_count": len(set_rows),
        "discordant_set_count": len(discordant),
        "partition_plan_exact_match_fraction": (
            sum(row["partition_plan_exact_match_fraction"] for row in partition_rows if row["partition_plan_exact_match_fraction"] is not None)
            / sum(row["partition_plan_exact_match_fraction"] is not None for row in partition_rows)
            if any(row["partition_plan_exact_match_fraction"] is not None for row in partition_rows) else None
        ),
        "partition_rows": partition_rows,
        "set_rows": set_rows,
        "v10_actions_are_not_ground_truth": True,
    }
    (output_root / "router_repeatability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "output_kirc_v10" / "experiment_multi_k_accepted_core_stability")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v11" / "router_policy_calibration")
    parser.add_argument("--initial-k", type=int, choices=INITIAL_KS, action="append")
    parser.add_argument("--repeat", type=int, choices=REPEATS, action="append")
    args = parser.parse_args()
    print(json.dumps(calibrate(args.experiment_root, args.config_dir, tuple(args.initial_k or INITIAL_KS), tuple(args.repeat or REPEATS), args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
