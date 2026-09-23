from __future__ import annotations

import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from agents.subtype_review.runtime_trace import append_runtime_trace, partition_snapshot
from agents.subtype_review.tools import TOOL_REGISTRY
from utils.cache_utils import file_content_identity, hash_payload
from utils.io import write_json
from utils.llm_utils import load_yaml_file


def run_subtype_review(
    candidate_sets: list[dict[str, Any]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    data_root: str,
    config_dir: str,
    artifact_root: str,
    runtime_trace_path: str | None = None,
) -> dict[str, Any]:
    review_config = load_yaml_file(Path(config_dir) / "subtype_review.yaml")
    budget = review_config["budget"]
    runtime_trace_path = runtime_trace_path or str(Path(artifact_root) / "runtime_trace.jsonl")
    usage_tracker = LLMUsageTracker(Path(artifact_root) / "llm_requests.jsonl")
    runtime = {
        "patient_states_by_id": {str(key): dict(value) for key, value in patient_states_by_id.items()},
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
        "config_dir": str(config_dir),
        "tool_registry": TOOL_REGISTRY,
        "verifier_model": build_default_verifier(
            review_config, config_dir,
            usage_tracker=usage_tracker,
            runtime_trace_path=runtime_trace_path,
        ),
        "router_model": build_default_router(
            review_config, config_dir,
            usage_tracker=usage_tracker,
            runtime_trace_path=runtime_trace_path,
        ),
        "reviser_model": build_default_reviser(
            review_config, config_dir,
            usage_tracker=usage_tracker,
            runtime_trace_path=runtime_trace_path,
        ),
        "runtime_trace_path": runtime_trace_path,
        "verifier_audit_coverage_retries": int(
            review_config["llm"].get("verifier_audit_coverage_retries", 2)
        ),
    }
    state = initial_review_state(candidate_sets)
    state["control"]["max_rounds"] = int(budget["max_rounds"])
    final_state = build_review_graph().invoke(state, context=runtime)
    final_state["control"]["llm_usage"] = usage_tracker.snapshot()
    return final_state


def run_single_review_job(
    k: int,
    repeat: int,
    candidate_sets: list[dict[str, Any]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    run_root: str,
    input_signature: str,
    candidate_signature: str,
) -> dict[str, Any]:
    run_path = Path(run_root)
    trace_path = run_path / "runtime_trace.jsonl"
    metadata_path = run_path / "run_metadata.json"
    metadata = {
        "signature_version": 2,
        "initial_k": k,
        "repeat": repeat,
        "input_signature": input_signature,
        "candidate_signature": candidate_signature,
        "runtime_trace_path": str(trace_path),
    }
    try:
        trace_path.write_text("", encoding="utf-8")
        write_json(metadata_path, {"status": "running", **metadata})
        append_runtime_trace(
            trace_path, node="runner", event="run_started", round_id=0,
            payload={"initial_k": k, "repeat": repeat, "input_signature": input_signature,
                     "candidate_signature": candidate_signature,
                     "initial_partition": partition_snapshot(candidate_sets)},
        )
        final_state = run_subtype_review(
            candidate_sets, patient_states_by_id, output_root, config_dir,
            str(run_path), str(trace_path),
        )
        summary = save_review_outputs(final_state, str(run_path), direct=True)
        complete = (
            (run_path / "final_subtype_sets.json").is_file()
            and summary["raw_control_status"] == "complete"
            and summary["status"] == "review_complete"
        )
        status = "complete" if complete else "incomplete"
        append_runtime_trace(
            trace_path, node="runner", event="run_completed",
            payload={"status": summary.get("status"), "raw_control_status": summary.get("raw_control_status"),
                     "rounds_used": summary.get("rounds_used")},
        )
        write_json(metadata_path, {"status": status, **metadata})
        return {"initial_k": k, "repeat": repeat, "status": status,
                "summary": summary, "run_root": str(run_path)}
    except Exception as exc:
        failure = {"initial_k": k, "repeat": repeat, "status": "failed",
                   "error_type": type(exc).__name__, "error_message": str(exc),
                   "run_root": str(run_path)}
        append_runtime_trace(trace_path, node="runner", event="run_failed", payload=failure)
        write_json(metadata_path, {**failure, **metadata})
        return failure


def build_review_input_manifest(
    *,
    candidate_signature: str,
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
) -> dict[str, Any]:
    review_config = load_yaml_file(Path(config_dir) / "subtype_review.yaml")
    evidence_paths = {
        str(state["omics_evidence"][key])
        for state in patient_states_by_id.values()
        for key in (
            "rna_pathway_feature_path", "rna_raw_counts_path",
            "wxs_discovery_feature_path", "wxs_interpretation_feature_path",
        )
        if state.get("omics_evidence", {}).get(key)
    }
    evidence_paths.update(str(review_config["known_label_echo"][key]) for key in (
        "mrna_m1_m4_path", "clearcode34_path"
    ))
    evidence_paths.add(str(review_config["rna"]["hallmark_gene_sets_path"]))
    wxs_config = Path(config_dir) / "wxs.yaml"
    if wxs_config.is_file():
        evidence_paths.add(str(wxs_config))
    technical_metadata = sorted(str(path) for path in (Path(output_root) / "ct_qc").rglob("*.json"))
    return {
        "signature_version": 2,
        "candidate_signature": candidate_signature,
        "review_signature": review_signature_manifest(review_config, config_dir),
        "evidence_inputs": sorted(
            (file_content_identity(path) for path in evidence_paths),
            key=lambda row: (row["sha256"], row["size"]),
        ),
        "technical_metadata": sorted(
            (file_content_identity(path) for path in technical_metadata),
            key=lambda row: (row["sha256"], row["size"]),
        ),
        "clinical_labels": {
            case_id: state.get("inventory", {}).get("Clinical", {})
            for case_id, state in sorted(patient_states_by_id.items())
        },
        "implementation": {
            name: file_content_identity(str(Path(__file__).with_name(name)))
            for name in (
                "runner.py", "graph.py", "llm.py", "schemas.py", "tools.py", "evidence_semantics.py"
            )
        },
    }


def build_review_input_signature(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    manifest = build_review_input_manifest(**kwargs)
    return hash_payload(manifest), manifest


def summarize_review_grid(
    *, output_root: str, config: Mapping[str, Any], active_input_signature: str,
) -> dict[str, Any]:
    root = Path(output_root) / "subtype_review" / "runs"
    complete_runs, failed_runs, incomplete_runs, missing_runs, stale_runs, invalid_runs = [], [], [], [], [], []
    configured_ks = sorted(set(map(int, config["multi_k"]["initial_ks"])))
    configured_repeats = sorted(set(map(int, config["multi_k"]["repeats"])))
    for initial_k in configured_ks:
        for repeat in configured_repeats:
            run_root = root / f"K{initial_k}" / f"repeat{repeat}"
            metadata_path = run_root / "run_metadata.json"
            summary_path = run_root / "final_review_summary.json"
            sets_path = run_root / "final_subtype_sets.json"
            record = {"initial_k": initial_k, "repeat": repeat, "run_root": str(run_root)}
            if not metadata_path.is_file():
                missing_runs.append({**record, "reason": "missing_run_metadata"}); continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid_runs.append({**record, "reason": "invalid_run_metadata_json"}); continue
            if not isinstance(metadata, dict):
                invalid_runs.append({**record, "reason": "invalid_run_metadata_json"}); continue
            if metadata.get("input_signature") != active_input_signature:
                stale_runs.append({**record, "input_signature": metadata.get("input_signature")}); continue
            if metadata.get("status") == "failed":
                failed_runs.append({**record, "error_type": metadata.get("error_type"), "error_message": metadata.get("error_message")}); continue
            if not summary_path.is_file() or not sets_path.is_file():
                incomplete_runs.append({**record, "reason": "missing_run_output"}); continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                sets = json.loads(sets_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid_runs.append({**record, "reason": "invalid_run_output_json"}); continue
            if (metadata.get("status") == "complete" and isinstance(summary, dict)
                    and summary.get("raw_control_status") == "complete"
                    and summary.get("status") == "review_complete"
                    and isinstance(sets, list)):
                complete_runs.append({"initial_k": initial_k, "repeat": repeat})
            else:
                incomplete_runs.append({**record, "reason": "review_not_complete"})
    return {
        "input_signature": active_input_signature,
        "configured_run_count": len(configured_ks) * len(configured_repeats),
        "complete_run_count": len(complete_runs),
        "failed_run_count": len(failed_runs),
        "incomplete_run_count": len(incomplete_runs),
        "missing_run_count": len(missing_runs),
        "stale_run_count": len(stale_runs),
        "invalid_run_count": len(invalid_runs),
        "complete_runs": complete_runs,
        "failed_runs": failed_runs,
        "incomplete_runs": incomplete_runs,
        "missing_runs": missing_runs,
        "stale_runs": stale_runs,
        "invalid_runs": invalid_runs,
        "usable_runs_by_k": {
            str(k): sum(row["initial_k"] == k for row in complete_runs)
            for k in configured_ks
        },
    }


def run_review_grid(
    candidate_partitions: Mapping[int, list[dict[str, Any]]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    initial_ks: tuple[int, ...],
    repeats: tuple[int, ...],
    candidate_signature: str,
    *,
    run_pairs: tuple[tuple[int, int], ...] | None = None,
    force: bool = False,
    parallel_runs: int | None = None,
) -> dict[str, Any]:
    root = Path(output_root) / "subtype_review" / "runs"
    config_path = Path(config_dir) / "subtype_review.yaml"
    review_config = load_yaml_file(config_path)
    input_signature, signature_manifest = build_review_input_signature(
        candidate_signature=candidate_signature,
        patient_states_by_id=patient_states_by_id,
        output_root=output_root,
        config_dir=config_dir,
    )
    if run_pairs:
        requested_pairs = tuple(sorted(set((int(k), int(repeat)) for k, repeat in run_pairs)))
    else:
        ks = tuple(sorted(set(int(k) for k in initial_ks)))
        repeat_ids = tuple(sorted(set(int(repeat) for repeat in repeats)))
        requested_pairs = tuple((k, repeat) for k in ks for repeat in repeat_ids)
    if not requested_pairs or any(k not in candidate_partitions for k, _ in requested_pairs):
        raise ValueError("Every requested initial K must have a candidate partition")
    if any(repeat < 1 for _, repeat in requested_pairs):
        raise ValueError("Every requested repeat must be >= 1")
    run_summaries = []
    run_failures = []
    incomplete_runs = []

    jobs = []
    for k, repeat in requested_pairs:
        run_root = root / f"K{k}" / f"repeat{repeat}"
        metadata_path = run_root / "run_metadata.json"
        summary_path = run_root / "final_review_summary.json"
        sets_path = run_root / "final_subtype_sets.json"
        if run_root.exists():
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            if (sets_path.is_file() and metadata.get("input_signature") == input_signature
                    and metadata.get("status") == "complete"
                    and summary.get("raw_control_status") == "complete"
                    and summary.get("status") == "review_complete"):
                run_summaries.append(summary)
                continue
            if not force and not (metadata.get("input_signature") == input_signature
                                  and metadata.get("status") == "failed"):
                raise FileExistsError(
                    f"Incomplete or stale Agent run exists: {run_root}; pass --force to replace it"
                )
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True)
        jobs.append((k, repeat, candidate_partitions[k], str(run_root)))

    if jobs:
        workers = max(1, int(parallel_runs or review_config["multi_k"].get("parallel_runs", 1)))
        if workers == 1:
            results = [run_single_review_job(
                k, repeat, candidate_sets, patient_states_by_id, output_root, config_dir,
                run_root, input_signature, candidate_signature,
            ) for k, repeat, candidate_sets, run_root in jobs]
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        run_single_review_job, k, repeat, candidate_sets, patient_states_by_id,
                        output_root, config_dir, run_root, input_signature, candidate_signature,
                    ): (k, repeat, run_root)
                    for k, repeat, candidate_sets, run_root in jobs
                }
                results = []
                for future in as_completed(futures):
                    k, repeat, run_root = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append({"initial_k": k, "repeat": repeat, "status": "failed",
                                        "error_type": type(exc).__name__, "error_message": str(exc),
                                        "run_root": run_root})
        for result in results:
            print(f"[subtype_review] K={result['initial_k']} repeat={result['repeat']} status={result['status']}", flush=True)
            if result["status"] == "complete":
                run_summaries.append(result["summary"])
            elif result["status"] == "failed":
                run_failures.append(result)
            else:
                incomplete_runs.append(result)

    requested_run_count = len(requested_pairs)
    complete_run_count = sum(
        1 for summary in run_summaries
        if summary.get("raw_control_status") == "complete"
        and summary.get("status") == "review_complete"
    )
    return {
        "runs": run_summaries,
        "run_root": str(root),
        "input_signature": input_signature,
        "signature_manifest": signature_manifest,
        "requested_pairs": [{"initial_k": k, "repeat": repeat} for k, repeat in requested_pairs],
        "failed_runs": run_failures,
        "incomplete_runs": incomplete_runs,
        "requested_run_count": requested_run_count,
        "complete_run_count": complete_run_count,
    }
