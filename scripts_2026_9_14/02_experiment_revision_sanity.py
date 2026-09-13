#!/usr/bin/env python3
"""Exercise Split and Merge through the real Graph with deterministic mock agents."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import (
    build_review_graph,
    current_sets,
    partition_signature,
)
from agents.subtype_review.tools import TOOL_REGISTRY
from utils.io import write_json

from importlib.util import module_from_spec, spec_from_file_location

SPEC = spec_from_file_location(
    "router_action_sanity",
    Path(__file__).with_name("00_experiment_router_action_sanity.py"),
)
SANITY = module_from_spec(SPEC)
SPEC.loader.exec_module(SANITY)


def action_plan(action: str, targets: list[str]) -> dict:
    state = SANITY.assessed_state(
        structure="incompatible" if action in {"split", "merge"} else "unassessed",
        identity="unassessed",
        alternative_explanation="unassessed",
        uncertainty="yes",
    )
    return {
        "action": action,
        "target_ids": targets,
        "decision_state": state,
        "reason": f"controlled {action} action",
    }


class MockRouter:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            if self.name == "split":
                actions = [action_plan("split", ["C1"]), action_plan("accept", ["C2"])]
            else:
                actions = [action_plan("merge", ["C1", "C2"])]
        else:
            actions = [action_plan("accept", [item["set_id"]]) for item in payload["partition"]["sets"]]
        return {"actions": actions, "evidence_requests": []}


class MockReviser:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.name == "split":
            return {
                "split_plans": [{
                    "target_id": "C1",
                    "n_children": 2,
                    "structural_basis": ["fused"],
                    "execution_strategy": "fused_similarity_spectral",
                    "metric_refs": [],
                    "rationale": "deterministic controlled split",
                }],
                "merge_plans": [],
                "rationale": "controlled split revision",
            }
        return {
            "split_plans": [],
            "merge_plans": [{
                "target_ids": ["C1", "C2"],
                "metric_refs": [],
                "rationale": "deterministic controlled merge",
            }],
            "rationale": "controlled merge revision",
        }


def run_mock_case(name: str) -> dict:
    state = SANITY.build_case(name)
    initial_signature = partition_signature(current_sets(state))
    router = MockRouter(name)
    reviser = MockReviser(name)
    runtime = {
        "data_root": str(ROOT),
        "artifact_root": str(ROOT),
        "config_dir": str(ROOT / "configs"),
        "tool_registry": TOOL_REGISTRY,
        "router_model": router,
        "reviser_model": reviser,
        "verifier_model": None,
    }

    def split_membership(output_root, member_ids, n_children, strategy, structural_basis):
        midpoint = len(member_ids) // 2
        return [sorted(member_ids[:midpoint]), sorted(member_ids[midpoint:])]

    with patch("tools.cross_modal_structure.execute_split_membership", split_membership):
        final_state = build_review_graph().invoke(state, context=runtime)
    final_sets = current_sets(final_state)
    final_signature = partition_signature(final_sets)
    return {
        "case": name,
        "status": final_state["control"]["status"],
        "router_calls": router.calls,
        "reviser_calls": reviser.calls,
        "revision_applied": bool(final_state.get("revision_result")),
        "partition_changed": initial_signature != final_signature,
        "revisited_after_revision": router.calls == 2,
        "initial_set_ids": [item["set_id"] for item in SANITY.build_case(name)["partition"]["sets"]],
        "current_set_ids": [item["set_id"] for item in final_sets],
        "revision_result": final_state.get("revision_result"),
        "trace": final_state["control"].get("trace", []),
    }


def run(output_root: Path, force: bool = False) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {name: run_mock_case(name) for name in ("split", "merge")}
    summary = {
        "experiment": "revision_sanity",
        "mode": "deterministic_mock_agents",
        "results": results,
        "passed_cases": sum(
            result["status"] == "complete"
            and result["revision_applied"]
            and result["partition_changed"]
            and result["revisited_after_revision"]
            for result in results.values()
        ),
    }
    for name, result in results.items():
        write_json(output_root / f"{name}_revision_result.json", result)
    write_json(output_root / "revision_sanity_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14" / "02_revision_sanity")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
