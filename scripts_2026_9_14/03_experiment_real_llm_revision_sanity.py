#!/usr/bin/env python3
"""Run controlled Split/Merge reviews with real Router and Reviser models."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import (
    build_review_graph,
    current_sets,
    initial_review_state,
    partition_signature,
)
from agents.subtype_review.llm import (
    LLMUsageTracker,
    build_default_reviser,
    build_default_router,
    review_signature_manifest,
)
from agents.subtype_review.tools import TOOL_REGISTRY
from utils.io import write_json
from utils.llm_utils import load_yaml_file

from importlib.util import module_from_spec, spec_from_file_location

SPEC = spec_from_file_location(
    "router_action_sanity",
    Path(__file__).with_name("00_experiment_router_action_sanity.py"),
)
SANITY = module_from_spec(SPEC)
SPEC.loader.exec_module(SANITY)


def write_synthetic_input(root: Path) -> None:
    candidate = root / "candidate_subtype"
    candidate.mkdir(parents=True, exist_ok=True)
    patient_ids = [f"P{i}" for i in range(1, 25)]
    similarity = np.full((24, 24), 0.05, dtype=float)
    for start, end in ((0, 6), (6, 12), (12, 24)):
        similarity[start:end, start:end] = 0.9
    np.fill_diagonal(similarity, 1.0)
    np.save(candidate / "fused_similarity.npy", similarity)
    (candidate / "affinity_patient_order.json").write_text(
        json.dumps(patient_ids), encoding="utf-8"
    )


def reports_for_sets(sets: list[dict], signature: str) -> list[dict]:
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
                "observations": [{"metric": "identity", "finding": "coherent controlled biological identity"}],
            },
            {
                "tool_name": "multimodal_consistency_check",
                "dimension": "cross_modal_consistency",
                "scope": "set_identity",
                "target_ids": [target],
                "status": "success",
                "partition_signature": signature,
                "observations": [{"metric": "post_revision_review", "finding": "the revised membership is available for Router review"}],
            },
            {
                "tool_name": "confound_test",
                "dimension": "confounder_exclusion",
                "scope": "set_identity",
                "target_ids": [target],
                "status": "success",
                "partition_signature": signature,
                "observations": [{"metric": "technical_context", "finding": "no plausible competing technical explanation"}],
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
    return rows


def build_post_revision_state(name: str, root: Path) -> dict:
    initial = SANITY.build_case(name)
    sets = current_sets(initial)
    if name == "split":
        from tools.cross_modal_structure import execute_split_membership

        groups = execute_split_membership(
            str(root), sets[0]["member_ids"], 2,
            "fused_similarity_spectral", ["fused"],
        )
        revised = [
            {"set_id": f"C1_S{i}", "member_ids": members}
            for i, members in enumerate(groups, 1)
        ] + [sets[1]]
    else:
        revised = [{
            "set_id": "C1_M_C2",
            "member_ids": sorted(member for item in sets for member in item["member_ids"]),
        }]
    state = initial_review_state(revised)
    signature = partition_signature(current_sets(state))
    state["reports"] = reports_for_sets(revised, signature)
    return state


def run_case(name: str, config: dict, config_dir: Path, output_root: Path) -> dict:
    input_root = output_root / name / "synthetic_input"
    write_synthetic_input(input_root)
    initial = SANITY.build_case(name)
    initial_signature = partition_signature(current_sets(initial))
    post = build_post_revision_state(name, input_root)
    post_signature = partition_signature(current_sets(post))
    initial["evidence_memory"][post_signature] = post["reports"]
    usage = LLMUsageTracker()
    runtime = {
        "data_root": str(input_root),
        "artifact_root": str(output_root / name),
        "config_dir": str(config_dir),
        "tool_registry": TOOL_REGISTRY,
        "router_model": build_default_router(config, config_dir, usage_tracker=usage),
        "reviser_model": build_default_reviser(config, config_dir, usage_tracker=usage),
        "verifier_model": None,
    }
    final_state = build_review_graph().invoke(initial, context=runtime)
    return {
        "case": name,
        "status": final_state["control"]["status"],
        "router_calls": sum(event.get("node") == "router" and event.get("event") == "decision" for event in final_state["control"].get("trace", [])),
        "reviser_calls": sum(event.get("node") == "reviser" for event in final_state["control"].get("trace", [])),
        "revision_applied": bool(final_state.get("revision_result")),
        "partition_changed": initial_signature != partition_signature(current_sets(final_state)),
        "revisited_after_revision": any(
            event.get("node") == "router" and event.get("event") == "decision"
            and event.get("round", 0) >= 1
            for event in final_state["control"].get("trace", [])
        ),
        "initial_set_ids": [item["set_id"] for item in SANITY.build_case(name)["partition"]["sets"]],
        "current_set_ids": [item["set_id"] for item in current_sets(final_state)],
        "trace": final_state["control"].get("trace", []),
        "llm_usage": usage.snapshot(),
    }


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
            results[name] = {"case": name, "status": "experiment_error", "error": f"{type(exc).__name__}: {exc}"}
        write_json(output_root / f"{name}_real_llm_result.json", results[name])
    summary = {
        "experiment": "real_llm_revision_sanity",
        "cases": list(case_names),
        "results": results,
        "completed_cases": sum(result.get("status") == "complete" for result in results.values()),
        "review_signature": review_signature_manifest(config, config_dir),
    }
    write_json(output_root / "real_llm_revision_sanity_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14" / "03_real_llm_revision_sanity")
    parser.add_argument("--case", choices=("split", "merge"), action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config_dir, args.output_root, tuple(args.case or ("split", "merge")), args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
