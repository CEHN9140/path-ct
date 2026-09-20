from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from agents.subtype_review.graph import build_review_graph, initial_review_state, save_review_outputs
from agents.subtype_review.llm import (
    LLMUsageTracker,
    build_default_reviser,
    build_default_router,
    build_default_verifier,
    review_signature_manifest,
)
from agents.subtype_review.tools import TOOL_REGISTRY
from utils.cache_utils import file_identity, hash_payload
from utils.io import write_json
from utils.llm_utils import load_yaml_file


def run_subtype_review(
    candidate_sets: list[dict[str, Any]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    data_root: str,
    config_dir: str,
    artifact_root: str,
) -> dict[str, Any]:
    review_config = load_yaml_file(Path(config_dir) / "subtype_review.yaml")
    budget = review_config["budget"]
    usage_tracker = LLMUsageTracker(Path(artifact_root) / "llm_requests.jsonl")
    runtime = {
        "patient_states_by_id": {str(key): dict(value) for key, value in patient_states_by_id.items()},
        "data_root": str(data_root),
        "config_dir": str(config_dir),
        "tool_registry": TOOL_REGISTRY,
        "verifier_model": build_default_verifier(review_config, config_dir, usage_tracker=usage_tracker),
        "router_model": build_default_router(review_config, config_dir, usage_tracker=usage_tracker),
        "reviser_model": build_default_reviser(review_config, config_dir, usage_tracker=usage_tracker),
    }
    state = initial_review_state(candidate_sets)
    state["control"]["max_rounds"] = int(budget["max_rounds"])
    final_state = build_review_graph().invoke(state, context=runtime)
    final_state["control"]["llm_usage"] = usage_tracker.snapshot()
    return final_state


def run_review_grid(
    candidate_partitions: Mapping[int, list[dict[str, Any]]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    initial_ks: tuple[int, ...],
    repeats: tuple[int, ...],
    candidate_signature: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root) / "subtype_review" / "runs"
    config_path = Path(config_dir) / "subtype_review.yaml"
    review_config = load_yaml_file(config_path)
    evidence_paths = {
        str(state["omics_evidence"][key])
        for state in patient_states_by_id.values()
        for key in ("rna_pathway_feature_path", "wxs_discovery_feature_path")
        if state.get("omics_evidence", {}).get(key)
    }
    evidence_paths.update(str(review_config["known_label_echo"][key]) for key in (
        "mrna_m1_m4_path", "clearcode34_path"
    ))
    evidence_paths.add(str(review_config["rna"]["hallmark_gene_sets_path"]))
    technical_metadata = sorted(
        str(path) for path in (Path(output_root) / "ct_qc").rglob("*.json")
    )
    input_signature = hash_payload({
        "candidate_signature": candidate_signature,
        "review_config": file_identity(str(config_path)),
        "review_signature": review_signature_manifest(review_config, config_dir),
        "evidence_inputs": {path: file_identity(path) for path in sorted(evidence_paths)},
        "technical_metadata": {path: file_identity(path) for path in technical_metadata},
        "clinical_labels": {
            case_id: state.get("inventory", {}).get("Clinical", {})
            for case_id, state in sorted(patient_states_by_id.items())
        },
        "implementation": {
            name: file_identity(str(Path(__file__).with_name(name)))
            for name in ("runner.py", "graph.py", "llm.py", "schemas.py", "tools.py")
        },
    })
    ks = tuple(sorted(set(int(k) for k in initial_ks)))
    repeat_ids = tuple(sorted(set(int(repeat) for repeat in repeats)))
    if not ks or not repeat_ids or any(k not in candidate_partitions for k in ks):
        raise ValueError("Every requested initial K must have a candidate partition")
    run_summaries = []

    for k in ks:
        for repeat in repeat_ids:
            run_root = root / f"K{k}" / f"repeat{repeat}"
            metadata_path = run_root / "run_metadata.json"
            summary_path = run_root / "final_review_summary.json"
            sets_path = run_root / "final_subtype_sets.json"
            if run_root.exists():
                metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
                summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
                if (
                    sets_path.is_file()
                    and metadata.get("input_signature") == input_signature
                    and metadata.get("status") == "complete"
                    and summary.get("raw_control_status") == "complete"
                    and summary.get("status") == "review_complete"
                ):
                    run_summaries.append(summary)
                    continue
                if not force:
                    raise FileExistsError(f"Incomplete or stale Agent run exists: {run_root}; pass --force to replace it")
                shutil.rmtree(run_root)

            run_root.mkdir(parents=True)
            write_json(metadata_path, {
                "status": "running",
                "initial_k": k,
                "repeat": repeat,
                "input_signature": input_signature,
                "candidate_signature": candidate_signature,
            })
            final_state = run_subtype_review(
                candidate_partitions[k],
                patient_states_by_id,
                output_root,
                config_dir,
                str(run_root),
            )
            summary = save_review_outputs(final_state, str(run_root), direct=True)
            complete = (
                (run_root / "final_subtype_sets.json").is_file()
                and summary["raw_control_status"] == "complete"
                and summary["status"] == "review_complete"
            )
            write_json(metadata_path, {
                "status": "complete" if complete else "incomplete",
                "initial_k": k,
                "repeat": repeat,
                "input_signature": input_signature,
                "candidate_signature": candidate_signature,
            })
            run_summaries.append(summary)

    full_grid = True
    for k in range(2, 9):
        for repeat in (1, 2, 3):
            run_root = root / f"K{k}" / f"repeat{repeat}"
            metadata_path = run_root / "run_metadata.json"
            summary_path = run_root / "final_review_summary.json"
            sets_path = run_root / "final_subtype_sets.json"
            if not metadata_path.is_file() or not summary_path.is_file() or not sets_path.is_file():
                full_grid = False
                break
            metadata = json.loads(metadata_path.read_text())
            summary = json.loads(summary_path.read_text())
            if (
                metadata.get("input_signature") != input_signature
                or metadata.get("status") != "complete"
                or summary.get("raw_control_status") != "complete"
                or summary.get("status") != "review_complete"
            ):
                full_grid = False
                break
        if not full_grid:
            break
    return {
        "runs": run_summaries,
        "run_root": str(root),
        "input_signature": input_signature,
        "multi_k_ready": full_grid,
    }
