#!/usr/bin/env python3
"""Replay one Router decision per existing run using saved evidence only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import compact_structural_index, validate_router_plan
from agents.subtype_review.llm import LLMUsageTracker, build_default_router, parse_router_plan
from agents.subtype_review.llm_summary import summarize_reports
from agents.subtype_review.tools import TOOL_REGISTRY
from utils.llm_utils import load_yaml_file

INITIAL_KS = tuple(range(2, 9))


def action_summary(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(target): str(action.get("action", ""))
        for action in plan.get("actions", []) or []
        for target in action.get("target_ids", []) or []
    }


def replay_entry(entry: Mapping[str, Any], router_model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    state = {
        "partition": entry["partition"],
        "round_evidence": entry.get("round_evidence", []),
        "reports": entry.get("evidence_reports", []),
        "control": {"round": int(entry.get("round", 1)) - 1},
    }
    registry_payload = {
        name: {key: value for key, value in metadata.items() if key != "function"}
        for name, metadata in TOOL_REGISTRY.items()
    }
    payload = {
        "partition": state["partition"],
        "evidence_reports": summarize_reports(state["reports"]),
        "structural_index": compact_structural_index(state),
        "tool_registry": registry_payload,
        "available_extra_evidence": [],
        "round": entry.get("round", 1),
        "instruction": "Router-only replay: use saved evidence and structural_index; do not request new evidence.",
    }
    raw_plan = router_model.invoke(payload)
    plan = parse_router_plan(raw_plan)
    validate_router_plan(plan, state, {"tool_registry": TOOL_REGISTRY})
    return payload, plan.model_dump()


def replay_run(
    run_root: Path,
    router_model: Any,
) -> dict[str, Any]:
    history_path = run_root / "review_history.json"
    if not history_path.exists():
        return {"status": "missing_history"}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return {"status": "empty_history"}
    entry = history[-1]
    old_plan = entry.get("router_plan", {})
    try:
        payload, new_plan = replay_entry(entry, router_model)
        return {
            "status": "success",
            "round": entry.get("round"),
            "old_actions": action_summary(old_plan),
            "new_actions": action_summary(new_plan),
            "changed": action_summary(old_plan) != action_summary(new_plan),
            "structural_index": payload["structural_index"],
            "plan": new_plan,
        }
    except Exception as exc:
        return {
            "status": "router_error",
            "round": entry.get("round"),
            "old_actions": action_summary(old_plan),
            "error": f"{type(exc).__name__}: {exc}",
        }


def replay_experiment(
    experiment_root: Path,
    config_dir: Path,
    initial_ks: list[int] | tuple[int, ...] = INITIAL_KS,
    repeat: int = 1,
    output_root: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root or experiment_root / "structural_index_router_replay"
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_yaml_file(config_dir / "subtype_review.yaml")
    router = build_default_router(config, config_dir, usage_tracker=LLMUsageTracker())
    rows = []
    for initial_k in initial_ks:
        run_root = experiment_root / f"run{repeat}" / f"K{initial_k}"
        result = replay_run(run_root, router)
        rows.append({"initial_k": initial_k, "repeat": repeat, **result})
    summary = {
        "experiment": "structural_index_router_replay",
        "runs": len(rows),
        "success": sum(row["status"] == "success" for row in rows),
        "router_error": sum(row["status"] == "router_error" for row in rows),
        "changed": sum(row.get("changed", False) for row in rows),
        "rows": rows,
    }
    (output_root / "router_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "output_kirc_v10" / "experiment_multi_k_accepted_core_stability",
    )
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--initial-k", type=int, action="append")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(
        replay_experiment(
            args.experiment_root,
            args.config_dir,
            args.initial_k or list(INITIAL_KS),
            args.repeat,
            args.output_root,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
