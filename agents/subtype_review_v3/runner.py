from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from agents.subtype_review_v3.graph import run_global_review_graph
from agents.subtype_review_v3.llm import build_default_router, build_default_verifier
from agents.subtype_review_v3.tools import DEFAULT_TOOL_DEFINITIONS, load_available_tool_functions
from utils.llm_utils import load_yaml_file


def load_evidence_ready(source_output_root: str | Path) -> list[dict[str, Any]]:
    path = Path(source_output_root) / "storage" / "pipeline_checkpoints" / "evidence_ready.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in list(payload.get("patient_states", []) or [])]


def load_candidate_clusters(source_output_root: str | Path) -> list[dict[str, Any]]:
    candidate_dir = Path(source_output_root) / "candidate_subtype"
    combined = candidate_dir / "candidate_clusters.json"
    if combined.exists():
        return [dict(item) for item in json.loads(combined.read_text(encoding="utf-8"))]
    return [
        dict(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(candidate_dir.glob("C*.json"))
    ]


def load_v3_config(config_dir: str | Path) -> dict[str, Any]:
    path = Path(config_dir) / "subtype_review_v3.yaml"
    if path.exists():
        return load_yaml_file(path)
    return {
        "budget": {"max_rounds": 3, "max_tool_calls": 8},
        "tools": DEFAULT_TOOL_DEFINITIONS,
    }


def run_subtype_review_v3_from_candidate(
    source_output_root: str,
    output_root: str,
    config_dir: str,
    *,
    verifier_model: Any = None,
    router_model: Any = None,
) -> dict[str, Any]:
    output_path = Path(output_root)
    final_review_dir = output_path / "subtype_review_v3"
    temp_output_root = output_path / ".subtype_review_v3_tmp"
    shutil.rmtree(temp_output_root, ignore_errors=True)
    patients = load_evidence_ready(source_output_root)
    patient_states_by_id = {
        str(item.get("case_id", item.get("Case_ID", ""))): dict(item)
        for item in patients
        if str(item.get("case_id", item.get("Case_ID", "")))
    }
    clusters = load_candidate_clusters(source_output_root)
    config = load_v3_config(config_dir)
    if verifier_model is None:
        verifier_model = build_default_verifier(config, config_dir)
    if router_model is None:
        router_model = build_default_router(config, config_dir)
    tool_definitions = dict(config.get("tools", {}) or DEFAULT_TOOL_DEFINITIONS)
    tool_functions, tool_load_errors = load_available_tool_functions(tool_definitions)
    active_tool_definitions = {
        name: definition
        for name, definition in tool_definitions.items()
        if name in tool_functions
    }
    try:
        state = run_global_review_graph(
            {
                "candidate_sets": clusters,
                "patient_states_by_id": patient_states_by_id,
                "tool_definitions": active_tool_definitions,
                "tool_functions": tool_functions,
                "tool_load_errors": tool_load_errors,
                "output_root": str(temp_output_root),
                "config_dir": config_dir,
                "budget": dict(config.get("budget", {}) or {}),
                "llm": dict(config.get("llm", {}) or {}),
                "verifier_model": verifier_model,
                "router_model": router_model,
            }
        )
    except Exception:
        shutil.rmtree(temp_output_root, ignore_errors=True)
        raise
    shutil.rmtree(final_review_dir, ignore_errors=True)
    shutil.move(str(temp_output_root / "subtype_review_v3"), str(final_review_dir))
    shutil.rmtree(temp_output_root, ignore_errors=True)
    old_prefix = str(temp_output_root)
    new_prefix = str(output_path)
    for path in final_review_dir.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if old_prefix in text:
            path.write_text(text.replace(old_prefix, new_prefix), encoding="utf-8")
    summary = dict(state.get("summary", {}) or {})
    return json.loads(json.dumps(summary).replace(old_prefix, new_prefix))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run subtype review v3 from existing candidate outputs.")
    parser.add_argument("source_output_root", default='output_kirc')
    parser.add_argument("output_root", nargs="?", default="output_kirc_v3")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    summary = run_subtype_review_v3_from_candidate(
        args.source_output_root,
        args.output_root,
        args.config_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
