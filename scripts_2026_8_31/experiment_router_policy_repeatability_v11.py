#!/usr/bin/env python3
"""Replay the V11 Router on one saved V10 entry per K."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import validate_router_plan  # noqa: E402
from agents.subtype_review.llm import (  # noqa: E402
    LLMUsageTracker,
    build_default_router,
    parse_router_plan,
    review_signature_manifest,
)
from agents.subtype_review.tools import TOOL_REGISTRY  # noqa: E402
from scripts_2026_8_31.experiment_structural_index_router_replay import (  # noqa: E402
    action_summary,
    replay_payload,
)
from utils.llm_utils import load_yaml_file  # noqa: E402

INITIAL_KS = tuple(range(2, 9))
SOURCE_REPEATS = (1, 2, 3)


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def plan_signature(plan: Mapping[str, Any]) -> str:
    actions = []
    for action in plan.get("actions", []) or []:
        actions.append({
            "action": action.get("action"),
            "target_ids": sorted(action.get("target_ids", []) or []),
            "evidence_requests": sorted(
                (
                    request.get("dimension"),
                    tuple(sorted(request.get("target_ids", []) or [])),
                )
                for request in action.get("evidence_requests", []) or []
            ),
        })
    return json.dumps(sorted(actions, key=lambda item: (item["action"], item["target_ids"])), sort_keys=True)


def load_source_entry(experiment_root: Path, initial_k: int, source_repeat: int) -> tuple[Path, dict[str, Any] | None, str | None]:
    path = experiment_root / f"run{source_repeat}" / f"K{initial_k}" / "review_history.json"
    if not path.exists():
        return path, None, "missing_history"
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        return path, None, "empty_history"
    return path, history[-1], None


def calibrate(
    experiment_root: Path,
    config_dir: Path,
    initial_ks: tuple[int, ...] = INITIAL_KS,
    source_repeat: int = 2,
    replay_count: int = 3,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if replay_count < 1:
        raise ValueError("replay_count must be positive")
    output_root = output_root or ROOT / "output_kirc_v12" / "10_router_policy_calibration_v11"
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_yaml_file(config_dir / "subtype_review.yaml")
    router = build_default_router(config, config_dir, usage_tracker=LLMUsageTracker())
    manifest = review_signature_manifest(config, config_dir)
    (output_root / "prompt_manifest.json").write_text(
        json.dumps({
            **manifest,
            "model": config.get("llm", {}).get("model_name"),
            "temperature": config.get("llm", {}).get("temperature"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = {}
    source_entries = {}
    for initial_k in initial_ks:
        history_path, entry, error = load_source_entry(experiment_root, initial_k, source_repeat)
        source_entries[initial_k] = entry
        payload = state = None
        if entry is not None:
            payload, state = replay_payload(entry)
        entry_hash = json_hash(entry) if entry is not None else None
        for replay_number in range(1, replay_count + 1):
            replay_payload_instance = copy.deepcopy(payload) if payload is not None else None
            payload_hash = json_hash(replay_payload_instance) if replay_payload_instance is not None else None
            result = {
                "initial_k": initial_k,
                "replay": replay_number,
                "source_repeat": source_repeat,
                "source_history_path": str(history_path),
                "source_entry_hash": entry_hash,
                "router_payload_hash": payload_hash,
                "status": "missing" if error == "missing_history" else "failed" if error else "pending",
            }
            if error:
                result["error"] = error
            else:
                try:
                    raw_plan = router.invoke(replay_payload_instance)
                    plan = parse_router_plan(raw_plan)
                    validate_router_plan(plan, state, {"tool_registry": TOOL_REGISTRY})
                    result.update({"status": "success", "plan": plan.model_dump()})
                except Exception as exc:
                    result.update({"status": "router_error", "error": f"{type(exc).__name__}: {exc}"})
                finally:
                    result["payload_unchanged_after_invoke"] = (
                        payload_hash == json_hash(replay_payload_instance)
                    )
            results[(initial_k, replay_number)] = result
            run_output = output_root / f"run{replay_number}" / f"K{initial_k}"
            run_output.mkdir(parents=True, exist_ok=True)
            (run_output / "router_plan.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    set_rows = []
    partition_rows = []
    for initial_k in initial_ks:
        entry = source_entries[initial_k]
        successful = [results[(initial_k, replay)] for replay in range(1, replay_count + 1) if results[(initial_k, replay)]["status"] == "success"]
        set_ids = sorted({
            str(item.get("set_id") or item.get("cluster_id") or "")
            for item in (entry or {}).get("partition", {}).get("sets", []) or []
            if str(item.get("set_id") or item.get("cluster_id") or "")
        })
        payload_hashes = [results[(initial_k, replay)]["router_payload_hash"] for replay in range(1, replay_count + 1)]
        identical_payload = bool(
            payload_hashes
            and len(set(payload_hashes)) == 1
            and None not in payload_hashes
            and all(
                results[(initial_k, replay)].get("payload_unchanged_after_invoke") is True
                for replay in range(1, replay_count + 1)
            )
        )
        signatures = [plan_signature(result["plan"]) for result in successful]
        pair_count = len(signatures) * (len(signatures) - 1) // 2
        partition_rows.append({
            "initial_k": initial_k,
            "source_repeat": source_repeat,
            "source_entry_hash": json_hash(entry) if entry is not None else None,
            "source_history_path": str(experiment_root / f"run{source_repeat}" / f"K{initial_k}" / "review_history.json"),
            "successful_replay_count": len(successful),
            "identical_payload_all_replays": identical_payload,
            "invalid_for_repeatability": not identical_payload or len(successful) != replay_count,
            "partition_plan_exact_match_fraction": (
                sum(left == right for left, right in combinations(signatures, 2)) / pair_count
                if pair_count else None
            ),
        })
        for set_id in set_ids:
            actions = [action_summary(result["plan"]).get(set_id, "") for result in successful]
            counts = Counter(action for action in actions if action)
            set_rows.append({
                "initial_k": initial_k,
                "set_id": set_id,
                **{
                    f"replay{replay}_action": action_summary(results[(initial_k, replay)].get("plan", {})).get(set_id, "")
                    if results[(initial_k, replay)]["status"] == "success" else ""
                    for replay in range(1, replay_count + 1)
                },
                **{f"{action}_count": counts[action] for action in ("accept", "drop", "split", "merge", "need_more_evidence")},
                "decision_agreement_fraction": max(counts.values()) / len(actions) if actions else None,
                "decision_discordant": len(counts) > 1,
                "identical_payload_all_replays": identical_payload,
                "invalid_for_repeatability": not identical_payload or len(successful) != replay_count,
                "v10_action_used_as_ground_truth": False,
            })

    fields = list(set_rows[0]) if set_rows else []
    if fields:
        with (output_root / "router_repeatability.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(set_rows)
    discordant = [row for row in set_rows if row["decision_discordant"]]
    valid_discordant = [row for row in discordant if not row["invalid_for_repeatability"]]
    invalid_set_count = sum(row["invalid_for_repeatability"] for row in set_rows)
    if discordant:
        with (output_root / "router_discordant_cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(discordant[0]))
            writer.writeheader()
            writer.writerows(discordant)

    valid_partition_rows = [row for row in partition_rows if not row["invalid_for_repeatability"] and row["partition_plan_exact_match_fraction"] is not None]
    summary = {
        "experiment": "router_policy_calibration_v11",
        "source_repeat": source_repeat,
        "replay_count": replay_count,
        "run_count": len(results),
        "successful_run_count": sum(result["status"] == "success" for result in results.values()),
        "router_error_count": sum(result["status"] == "router_error" for result in results.values()),
        "missing_run_count": sum(result["status"] == "missing" for result in results.values()),
        "set_row_count": len(set_rows),
        "discordant_set_count": len(discordant),
        "valid_discordant_set_count": len(valid_discordant),
        "invalid_set_count": invalid_set_count,
        "identical_payload_all_replays": {
            f"K{row['initial_k']}": row["identical_payload_all_replays"] for row in partition_rows
        },
        "partition_plan_exact_match_fraction": (
            sum(row["partition_plan_exact_match_fraction"] for row in valid_partition_rows) / len(valid_partition_rows)
            if valid_partition_rows else None
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
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "output_kirc_v12" / "02_multi_k_accepted_core_stability_v10")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v12" / "10_router_policy_calibration_v11")
    parser.add_argument("--initial-k", type=int, choices=INITIAL_KS, action="append")
    parser.add_argument("--source-repeat", type=int, choices=SOURCE_REPEATS, default=2)
    parser.add_argument("--replay-count", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(calibrate(
        args.experiment_root,
        args.config_dir,
        tuple(args.initial_k or INITIAL_KS),
        args.source_repeat,
        args.replay_count,
        args.output_root,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
