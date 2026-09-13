#!/usr/bin/env python3
"""Test whether the current Router selects structurally supported actions."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import (
    compact_structural_index,
    current_sets,
    evidence_coverage,
    initial_review_state,
    partition_signature,
    validate_router_plan,
)
from agents.subtype_review.llm import (
    LLMUsageTracker,
    build_default_router,
    parse_router_plan,
    review_signature_manifest,
)
from agents.subtype_review.llm_summary import summarize_reports
from agents.subtype_review.tools import TOOL_REGISTRY
from utils.io import write_json
from utils.llm_utils import load_yaml_file


def assessed_state(
    identity: str = "supported",
    structure: str = "compatible",
    alternative_explanation: str = "not_supported",
    uncertainty: str = "no",
) -> dict[str, str]:
    return {
        "identity": identity,
        "structure": structure,
        "alternative_explanation": alternative_explanation,
        "uncertainty": uncertainty,
    }


CASES = {
    "split": {
        "expected": {"C1": "split", "C2": "accept"},
        "description": "C1 has a reproducible positive internal binary structure; C2 is a retained control set.",
    },
    "merge": {
        "expected": {"C1+C2": "merge"},
        "description": "C1 and C2 have a reproducibly weak pairwise boundary.",
    },
    "drop": {
        "expected": {"C1": "drop"},
        "description": "C1 lacks a defensible identity and has a substantial competing technical explanation.",
    },
}


def build_case(name: str) -> dict[str, Any]:
    if name not in CASES:
        raise ValueError(f"Unknown sanity case: {name}")
    groups = {
        "split": [("C1", 12), ("C2", 12)],
        "merge": [("C1", 12), ("C2", 12)],
        "drop": [("C1", 12)],
    }[name]
    sets = [
        {"set_id": set_id, "member_ids": [f"P{i}" for i in range(start, start + size)]}
        for start, (set_id, size) in zip((1, 13), groups)
    ]
    state = initial_review_state(sets)
    signature = partition_signature(current_sets(state))
    rows = []
    for item in sets:
        target = item["set_id"]
        rows.extend([
            {
                "tool_name": "pathway_enrichment",
                "dimension": "biological_support",
                "scope": "set_identity",
                "target_ids": [target],
                "status": "success",
                "partition_signature": signature,
                "observations": [{"metric": "coherent_signal", "finding": "controlled biological evidence"}],
            },
            {
                "tool_name": "multimodal_consistency_check",
                "dimension": "cross_modal_consistency",
                "scope": "set_identity",
                "target_ids": [target],
                "status": "success",
                "partition_signature": signature,
                "observations": [{"metric": "controlled_structure", "finding": CASES[name]["description"]}],
            },
            {
                "tool_name": "confound_test",
                "dimension": "confounder_exclusion",
                "scope": "set_identity",
                "target_ids": [target],
                "status": "success",
                "partition_signature": signature,
                "observations": [{"metric": "technical_context", "finding": "controlled technical context"}],
            },
        ])
    rows.append({
        "tool_name": "known_label_echo_test",
        "dimension": "known_label_echo",
        "scope": "partition",
        "target_ids": [],
        "status": "success",
        "partition_signature": signature,
        "observations": [{"metric": "stage_grade_echo", "finding": "controlled partition context"}],
    })
    cross_rows = [row for row in rows if row["tool_name"] == "multimodal_consistency_check"]
    if name == "split":
        cross_rows[0]["full_metrics"] = {"structural_characterization": {"internal_structure_by_set": {
            "C1": {"fused_binary_probe": {"child_sizes": [6, 6], "median_silhouette": 0.65, "resampling": {
                "median_resample_ari": 0.92, "consensus_separation": 0.81, "pac": 0.12, "degenerate_resample_fraction": 0.01
            }}}
        }}}
    elif name == "merge":
        cross_rows[0]["full_metrics"] = {"structural_characterization": {"boundary_by_pair": {
            "C1+C2": {"targets": ["C1", "C2"], "fused": {
                "pair_median_silhouette": -0.21, "left_median_margin": -0.14,
                "right_median_margin": -0.18, "left_boundary_separation": -0.12,
                "right_boundary_separation": -0.16,
            }}
        }}}
    state["round_evidence"] = rows
    state["reports"] = copy.deepcopy(rows)
    return state


def action_summary(plan: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for action in plan.get("actions", []) or []:
        targets = sorted(str(target) for target in action.get("target_ids", []) or [])
        result["+".join(targets)] = str(action.get("action", ""))
    return result


def build_payload(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    state = build_case(name)
    payload = {
        "partition": state["partition"],
        "evidence_reports": summarize_reports(state["reports"]),
        "evidence_coverage": evidence_coverage(state, {"tool_registry": TOOL_REGISTRY}),
        "structural_index": compact_structural_index(state),
        "available_evidence_requests": [],
        "round": 1,
        "instruction": "This is a controlled action sanity test. Select the scientifically justified action mode from the supplied evidence.",
    }
    return payload, state


def run_case(name: str, router: Any) -> dict[str, Any]:
    payload, state = build_payload(name)
    try:
        response = router.invoke(copy.deepcopy(payload))
        plan = parse_router_plan(response)
        validate_router_plan(plan, state, {"tool_registry": TOOL_REGISTRY})
        actual = action_summary(plan.model_dump())
        expected = dict(CASES[name]["expected"])
        return {
            "status": "success",
            "expected_actions": expected,
            "actual_actions": actual,
            "passed": actual == expected,
            "router_plan": plan.model_dump(),
            "payload": payload,
        }
    except Exception as exc:
        return {
            "status": "router_error",
            "expected_actions": dict(CASES[name]["expected"]),
            "actual_actions": {},
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "payload": payload,
        }


def run(
    config_dir: Path,
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_yaml_file(config_dir / "subtype_review.yaml")
    usage = LLMUsageTracker()
    router = build_default_router(config, config_dir, usage_tracker=usage)
    results = {}
    for name in CASES:
        result = run_case(name, router)
        results[name] = result
        write_json(output_root / f"{name}_result.json", result)
    summary = {
        "experiment": "router_action_sanity",
        "cases": list(CASES),
        "passed_cases": sum(result["passed"] for result in results.values()),
        "router_errors": sum(result["status"] == "router_error" for result in results.values()),
        "results": {
            name: {key: value for key, value in result.items() if key != "payload"}
            for name, result in results.items()
        },
        "llm_usage": usage.snapshot(),
        "review_signature": review_signature_manifest(config, config_dir),
    }
    write_json(output_root / "router_action_sanity_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output_kirc_v14" / "00_router_action_sanity",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config_dir, args.output_root, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
