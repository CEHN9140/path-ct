#!/usr/bin/env python3
"""Verify real Router/Reviser revision followed by real Verifier reacquisition."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import build_review_graph, current_sets, initial_review_state, partition_signature
from agents.subtype_review.llm import LLMUsageTracker, build_default_reviser, build_default_router, build_default_verifier, review_signature_manifest
from agents.subtype_review.tools import TOOL_REGISTRY
from tools.subtype_review_common import tool_result
from utils.io import write_json
from utils.llm_utils import load_yaml_file


def load_sanity_module():
    path = Path(__file__).with_name("00_experiment_router_action_sanity.py")
    spec = importlib.util.spec_from_file_location("router_action_sanity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SANITY = load_sanity_module()
REAL_REVISION = None


def write_synthetic_input(root: Path) -> None:
    global REAL_REVISION
    if REAL_REVISION is None:
        path = Path(__file__).with_name("03_experiment_real_llm_revision_sanity.py")
        spec = importlib.util.spec_from_file_location("real_revision_sanity", path)
        REAL_REVISION = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(REAL_REVISION)
    REAL_REVISION.write_synthetic_input(root)


def build_initial_state(name: str) -> dict:
    state = SANITY.build_case(name)
    row = next(
        item for item in state["round_evidence"]
        if item["tool_name"] == "multimodal_consistency_check"
    )
    if name == "split":
        base = "tool_results.multimodal_consistency_check.metrics.structural_characterization.internal_structure_by_set.C1.fused_binary_probe"
        row["metric_refs"] = [
            f"{base}.{field}"
            for field in ("median_silhouette", "resampling.median_resample_ari", "resampling.consensus_separation", "resampling.pac")
        ]
    else:
        base = "tool_results.multimodal_consistency_check.metrics.structural_characterization.boundary_by_pair.C1+C2.fused"
        row["metric_refs"] = [
            f"{base}.{field}"
            for field in ("pair_median_silhouette", "left_median_margin", "right_median_margin", "left_boundary_separation", "right_boundary_separation")
        ]
    for item in state["reports"]:
        if item["tool_name"] == "multimodal_consistency_check":
            item["metric_refs"] = list(row["metric_refs"])
    return state


def synthetic_tool(name: str):
    def run_tool(
        cluster_state, patient_states_by_id, output_root, config_dir="",
        all_cluster_states=None, scope="set_identity", target_ids=None,
        artifact_root=None,
    ):
        targets = [str(target) for target in target_ids or []]
        if scope == "partition":
            metrics = {"known_label_echo": {"stage": {"ari": 0.1}, "grade": {"ari": 0.1}}}
        else:
            metrics = {
                "set_metrics": {
                    target: {"effect": 0.5, "q_value": 0.2}
                    for target in targets
                }
            }
        return tool_result(
            tool_name=name,
            status="success",
            cluster_id=str(cluster_state.get("cluster_id", "GLOBAL")),
            output_root=artifact_root or output_root,
            summary=f"Controlled reacquisition output for {name}.",
            metrics=metrics,
            decision_metrics=metrics,
        )
    return run_tool


def synthetic_registry() -> dict:
    return {
        name: {**metadata, "function": synthetic_tool(name)}
        for name, metadata in TOOL_REGISTRY.items()
    }


def summarize_state(name: str, state: dict, initial_signature: str) -> dict:
    trace = list(state.get("control", {}).get("trace", []) or [])
    revision_index = next(
        (index for index, event in enumerate(trace)
         if event.get("event") == "revision_applied"),
        -1,
    )
    post_revision_trace = trace[revision_index + 1:] if revision_index >= 0 else []
    verifier_events = [event for event in post_revision_trace if event.get("node") == "verifier"]
    tool_events = [event for event in verifier_events if event.get("event") == "tool_selection"]
    report_events = [event for event in verifier_events if event.get("event") == "reports"]
    request_events = [
        event for event in post_revision_trace
        if event.get("node") == "router"
        and event.get("event") == "decision"
        and (event.get("plan") or {}).get("evidence_requests")
    ]
    report_index = next(
        (index for index, event in enumerate(trace)
         if index > revision_index and event.get("node") == "verifier"
         and event.get("event") == "reports"),
        -1,
    )
    final_router = any(
        index > report_index and event.get("node") == "router"
        and event.get("event") == "decision"
        for index, event in enumerate(trace)
    )
    current_signature = partition_signature(current_sets(state))
    current_reports = state.get("evidence_memory", {}).get(current_signature, [])
    current_ids = {item["set_id"] for item in current_sets(state)}
    reports = bool(report_events) and bool(current_reports) and state.get("reports") == current_reports and all(
        set(report.get("target_ids", []) or []).issubset(current_ids)
        for report in current_reports
    )
    reacquired = bool(request_events) and bool(tool_events)
    revision = bool(state.get("revision_result"))
    changed = initial_signature != current_signature
    revision_event = revision_index >= 0
    return {
        "case": name,
        "status": state["control"].get("status"),
        "router_calls": sum(event.get("node") == "router" and event.get("event") == "decision" for event in trace),
        "reviser_calls": sum(event.get("node") == "reviser" for event in trace),
        "verifier_tool_selection_calls": sum(event.get("event") == "tool_selection" for event in verifier_events),
        "verifier_report_batches": sum(event.get("event") == "reports" for event in verifier_events),
        "reacquired_tools": [name for event in tool_events for name in event.get("tools", [])],
        "final_report_count": len(state.get("reports", []) or []),
        "revision_applied": revision,
        "partition_changed": changed,
        "verifier_reacquired_after_revision": revision_event and reacquired,
        "reports_reacquired_after_revision": revision_event and reports,
        "full_loop_verified": revision_event and revision and changed and reacquired and reports and final_router,
        "initial_set_ids": [item["set_id"] for item in SANITY.build_case(name)["partition"]["sets"]],
        "current_set_ids": [item["set_id"] for item in current_sets(state)],
        "trace": trace,
    }


def run_case(name: str, config: dict, config_dir: Path, output_root: Path) -> dict:
    case_root = output_root / name
    input_root = case_root / "synthetic_input"
    write_synthetic_input(input_root)
    initial = build_initial_state(name)
    initial_signature = partition_signature(current_sets(initial))
    usage = LLMUsageTracker()
    registry = synthetic_registry()
    with patch.dict(TOOL_REGISTRY, registry, clear=True):
        runtime = {
            "data_root": str(input_root),
            "artifact_root": str(case_root),
            "config_dir": str(config_dir),
            "tool_registry": TOOL_REGISTRY,
            "patient_states_by_id": {},
            "router_model": build_default_router(config, config_dir, usage_tracker=usage),
            "reviser_model": build_default_reviser(config, config_dir, usage_tracker=usage),
            "verifier_model": build_default_verifier(config, config_dir, usage_tracker=usage),
        }
        final_state = build_review_graph().invoke(initial, context=runtime)
    result = summarize_state(name, final_state, initial_signature)
    result["llm_usage"] = usage.snapshot()
    return result


def run(config_dir: Path, output_root: Path, case_names: tuple[str, ...], force: bool = False) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_yaml_file(config_dir / "subtype_review.yaml")
    results = {}
    for name in case_names:
        try:
            results[name] = run_case(name, config, config_dir, output_root)
        except Exception as exc:
            results[name] = {
                "case": name,
                "status": "experiment_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_json(output_root / f"{name}_full_loop_result.json", results[name])
    summary = {
        "experiment": "full_revision_evidence_loop",
        "cases": list(case_names),
        "results": results,
        "verified_cases": sum(item.get("full_loop_verified", False) for item in results.values()),
        "review_signature": review_signature_manifest(config, config_dir),
    }
    write_json(output_root / "full_revision_evidence_loop_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14" / "04_full_revision_evidence_loop")
    parser.add_argument("--case", choices=("split", "merge"), action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config_dir, args.output_root, tuple(args.case or ("split", "merge")), args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
