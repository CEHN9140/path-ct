from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from utils.cache_utils import hash_payload
from utils.io import write_json
from utils.llm_utils import load_yaml_file


def collect_accept_observations(run_files: list[tuple[int, int, Path]], patient_ids: set[str]) -> list[dict[str, Any]]:
    observations = []
    for initial_k, repeat, path in run_files:
        assigned: set[str] = set()
        for item in json.loads(path.read_text(encoding="utf-8")):
            members = [str(member) for member in item["member_ids"]]
            member_set = set(members)
            if len(member_set) != len(members):
                raise ValueError(f"Accepted set contains duplicate patients: {path}")
            if not member_set.issubset(patient_ids):
                raise ValueError(f"Accepted set contains patients outside candidate cohort: {path}")
            if assigned.intersection(member_set):
                raise ValueError(f"Accepted sets overlap within run: {path}")
            assigned.update(member_set)
            observations.append({
                "observation_id": f"K{initial_k}_repeat{repeat}_{item['set_id']}",
                "initial_k": initial_k, "repeat": repeat, "set_id": str(item["set_id"]),
                "member_ids": member_set, "patient_count": len(member_set),
            })
    return observations


def inspect_agent_run(run_root: Path, active_input_signature: str) -> tuple[bool, dict[str, Any]]:
    metadata_path = run_root / "run_metadata.json"
    summary_path = run_root / "final_review_summary.json"
    sets_path = run_root / "final_subtype_sets.json"
    if not metadata_path.is_file():
        return False, {"status": "excluded", "reason": "missing_run_metadata"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, {"status": "excluded", "reason": "invalid_run_metadata_json"}
    if not isinstance(metadata, dict):
        return False, {"status": "excluded", "reason": "invalid_run_metadata_json"}
    if metadata.get("input_signature") != active_input_signature:
        return False, {"status": "excluded", "reason": "input_signature_mismatch", "input_signature": metadata.get("input_signature")}
    if metadata.get("status") != "complete":
        return False, {"status": "excluded", "reason": f"run_status_{metadata.get('status', 'unknown')}"}
    if not summary_path.is_file():
        return False, {"status": "excluded", "reason": "missing_final_review_summary"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, {"status": "excluded", "reason": "invalid_final_review_summary_json"}
    if not isinstance(summary, dict):
        return False, {"status": "excluded", "reason": "invalid_final_review_summary_json"}
    if summary.get("raw_control_status") != "complete" or summary.get("status") != "review_complete":
        return False, {"status": "excluded", "reason": "review_not_complete"}
    if not sets_path.is_file():
        return False, {"status": "excluded", "reason": "missing_final_subtype_sets"}
    try:
        sets = json.loads(sets_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, {"status": "excluded", "reason": "invalid_final_subtype_sets_json"}
    if not isinstance(sets, list):
        return False, {"status": "excluded", "reason": "invalid_final_subtype_sets"}
    return True, {"status": "included", "input_signature": metadata["input_signature"], "sets_path": str(sets_path)}


def collect_usable_runs(
    output_root: str,
    config: Mapping[str, Any],
    active_input_signature: str,
) -> tuple[list[tuple[int, int, Path]], list[dict[str, Any]]]:
    runs_root = Path(output_root) / "subtype_review" / "runs"
    params = config["multi_k"]
    usable_runs = []
    audit = []
    for initial_k in sorted(set(map(int, params["initial_ks"]))):
        for repeat in sorted(set(map(int, params["repeats"]))):
            run_root = runs_root / f"K{initial_k}" / f"repeat{repeat}"
            usable, detail = inspect_agent_run(run_root, active_input_signature)
            audit.append({"initial_k": initial_k, "repeat": repeat, "run_root": str(run_root), **detail})
            if usable:
                usable_runs.append((initial_k, repeat, run_root / "final_subtype_sets.json"))
    return usable_runs, audit


def enumerate_intersection_candidates(observations: list[dict[str, Any]], min_subtype_size: int) -> set[frozenset[str]]:
    unique_sets = {frozenset(row["member_ids"]) for row in observations}
    candidates = {members for members in unique_sets if len(members) >= min_subtype_size}
    frontier = list(candidates)
    while frontier:
        current = frontier.pop()
        for accepted in unique_sets:
            intersection = frozenset(current.intersection(accepted))
            if len(intersection) >= min_subtype_size and intersection not in candidates:
                candidates.add(intersection)
                frontier.append(intersection)
    return candidates


def build_recurrent_sets(observations: list[dict[str, Any]], min_subtype_size: int, usable_run_count: int | None = None) -> list[dict[str, Any]]:
    records = []
    total = len(observations)
    usable_run_count = usable_run_count or len({(row["initial_k"], row["repeat"]) for row in observations})
    for members in enumerate_intersection_candidates(observations, min_subtype_size):
        supporting = [row for row in observations if members.issubset(row["member_ids"])]
        closure = frozenset.intersection(*(frozenset(row["member_ids"]) for row in supporting)) if supporting else frozenset()
        if closure != members:
            continue
        occurrence_count = len(supporting)
        supporting_ks = sorted({int(row["initial_k"]) for row in supporting})
        supporting_runs = [f"K{row['initial_k']}_repeat{row['repeat']}" for row in supporting]
        records.append({
            "member_ids": sorted(members), "member_count": len(members),
            "occurrence_count": occurrence_count, "total_accept_set_count": total,
            "accept_set_frequency": occurrence_count / total if total else 0.0,
            "supporting_observation_ids": [row["observation_id"] for row in supporting],
            "supporting_ks": supporting_ks, "supporting_k_count": len(supporting_ks),
            "supporting_runs": supporting_runs, "supporting_run_count": len(supporting_runs),
            "usable_run_count": usable_run_count,
            "run_support_frequency": len(supporting_runs) / usable_run_count if usable_run_count else 0.0,
        })
    records.sort(key=lambda row: (-row["accept_set_frequency"], -row["member_count"], tuple(row["member_ids"])))
    for rank, row in enumerate(records, 1):
        row["recurrent_set_id"] = f"RS{rank:03d}"
    return records


def write_recurrent_outputs(output_dir: Path, observations: list[dict[str, Any]], recurrent_sets: list[dict[str, Any]], *, aggregation_manifest: Mapping[str, Any], aggregation_signature: str, run_count: int, min_subtype_size: int, run_audit: list[dict[str, Any]]) -> dict[str, Any]:
    pd.DataFrame([
        {"observation_id": row["observation_id"], "initial_k": row["initial_k"], "repeat": row["repeat"], "set_id": row["set_id"], "patient_count": row["patient_count"], "member_ids": json.dumps(sorted(row["member_ids"]), ensure_ascii=False)}
        for row in observations
    ], columns=["observation_id", "initial_k", "repeat", "set_id", "patient_count", "member_ids"]).to_csv(output_dir / "accept_set_observations.csv", index=False)
    pd.DataFrame([
        {"rank": rank, "recurrent_set_id": row["recurrent_set_id"], "member_count": row["member_count"], "occurrence_count": row["occurrence_count"], "total_accept_set_count": row["total_accept_set_count"], "accept_set_frequency": row["accept_set_frequency"], "supporting_k_count": row["supporting_k_count"], "supporting_ks": json.dumps(row["supporting_ks"]), "supporting_run_count": row["supporting_run_count"], "usable_run_count": row["usable_run_count"], "run_support_frequency": row["run_support_frequency"], "member_ids": json.dumps(row["member_ids"], ensure_ascii=False)}
        for rank, row in enumerate(recurrent_sets, 1)
    ], columns=["rank", "recurrent_set_id", "member_count", "occurrence_count", "total_accept_set_count", "accept_set_frequency", "supporting_k_count", "supporting_ks", "supporting_run_count", "usable_run_count", "run_support_frequency", "member_ids"]).to_csv(output_dir / "recurrent_set_summary.csv", index=False)
    write_json(output_dir / "recurrent_sets.json", recurrent_sets)
    pd.DataFrame([
        {"recurrent_set_id": row["recurrent_set_id"], "patient_id": patient_id}
        for row in recurrent_sets for patient_id in row["member_ids"]
    ], columns=["recurrent_set_id", "patient_id"]).to_csv(output_dir / "recurrent_set_membership.csv", index=False)
    included = [row for row in run_audit if row["status"] == "included"]
    excluded = [row for row in run_audit if row["status"] == "excluded"]
    usable_runs_by_k = {
        str(initial_k): sum(
            row["status"] == "included" and row["initial_k"] == initial_k
            for row in run_audit
        )
        for initial_k in sorted({row["initial_k"] for row in run_audit})
    }
    summary = {"status": "complete", "analysis_type": "closed_recurrent_accept_set_statistics", "aggregation_signature": aggregation_signature, "run_count": run_count, "included_run_count": len(included), "excluded_run_count": len(excluded), "total_accept_set_count": len(observations), "min_subtype_size": min_subtype_size, "recurrent_set_count": len(recurrent_sets), "automatic_subtype_selection": False}
    write_json(output_dir / "aggregation_manifest.json", {
        **aggregation_manifest,
        "aggregation_signature": aggregation_signature,
        "configured_run_count": len(run_audit),
        "included_run_count": len(included),
        "excluded_run_count": len(excluded),
        "excluded_runs": excluded,
        "usable_runs_by_k": usable_runs_by_k,
        "run_audit": run_audit,
    })
    write_json(output_dir / "aggregation_summary.json", summary)
    return summary


def run_multi_k_aggregation(
    output_root: str,
    config: Mapping[str, Any],
    input_signature: str,
) -> dict[str, Any]:
    root = Path(output_root)
    patient_order_path = root / "candidate_subtype" / "affinity_patient_order.json"
    patient_values = json.loads(patient_order_path.read_text(encoding="utf-8"))
    patient_ids = [str(value) for value in patient_values]
    patient_set = set(patient_ids)
    if len(patient_set) != len(patient_ids):
        raise ValueError("Candidate patient order contains duplicate IDs")
    params = config["multi_k"]
    run_files, run_audit = collect_usable_runs(output_root, config, input_signature)
    if not run_files:
        raise ValueError("No usable completed Agent runs were found for the active input signature.")
    configured_initial_ks = sorted(set(map(int, params["initial_ks"])))
    configured_repeats = sorted(set(map(int, params["repeats"])))
    included_runs = [
        {"initial_k": initial_k, "repeat": repeat}
        for initial_k, repeat, _ in run_files
    ]
    included_initial_ks = sorted({row["initial_k"] for row in included_runs})
    included_repeats = sorted({row["repeat"] for row in included_runs})
    aggregation_params = {
        **params,
        "initial_ks": configured_initial_ks,
        "repeats": configured_repeats,
    }
    run_manifest = []
    for initial_k, repeat, sets_path in run_files:
        metadata = json.loads((sets_path.parent / "run_metadata.json").read_text(encoding="utf-8"))
        run_manifest.append({"initial_k": initial_k, "repeat": repeat, "agent_input_signature": metadata["input_signature"], "final_subtype_sets_path": str(sets_path.resolve()), "final_subtype_sets_sha256": hashlib.sha256(sets_path.read_bytes()).hexdigest()})
    aggregation_manifest = {
        "aggregation_version": 2,
        "agent_input_signature": input_signature,
        "configured_initial_ks": configured_initial_ks,
        "configured_repeats": configured_repeats,
        "included_initial_ks": included_initial_ks,
        "included_repeats": included_repeats,
        "included_runs": included_runs,
        "parameters": aggregation_params,
        "patient_order": {
            "path": str(patient_order_path.resolve()),
            "sha256": hashlib.sha256(patient_order_path.read_bytes()).hexdigest(),
        },
        "agent_runs": run_manifest,
        "aggregation_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    aggregation_signature = hash_payload(aggregation_manifest)
    output_dir = root / "subtype_review" / "multi_k"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = collect_accept_observations(run_files, patient_set)
    recurrent_sets = build_recurrent_sets(observations, int(params["min_subtype_size"]), usable_run_count=len(run_files))
    return write_recurrent_outputs(output_dir, observations, recurrent_sets, aggregation_manifest=aggregation_manifest, aggregation_signature=aggregation_signature, run_count=len(run_files), min_subtype_size=int(params["min_subtype_size"]), run_audit=run_audit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute recurrent ACCEPT-set statistics from completed Agent runs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    config = load_yaml_file(Path(args.config_dir) / "subtype_review.yaml")
    experiment_path = Path(args.output_root) / "subtype_review" / "active_review_experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    input_signature = experiment["input_signature"]
    print(json.dumps(
        run_multi_k_aggregation(args.output_root, config, input_signature),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
