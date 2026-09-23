from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from agents.subtype_review.run_io import inspect_review_run
from utils.cache_utils import hash_payload
from utils.io import write_json
from utils.llm_utils import load_yaml_file

DEFAULT_MULTI_K_CONFIG: dict[str, Any] = {
    "initial_ks": [2, 3, 4, 5, 6, 7, 8],
    "repeats": [1, 2, 3],
    "parallel_runs": 4,
    "min_subtype_size": 10,
    "patient_recurrence": {
        "similarity_threshold": 0.9,
        "min_subtype_size": 10,
        "min_common_occurrences": 2,
        "min_supporting_ks": 2,
    },
}


def collect_accept_observations(run_files: list[dict[str, Any]], patient_ids: set[str]) -> list[dict[str, Any]]:
    observations = []
    for run in run_files:
        initial_k = run["initial_k"]
        repeat = run["repeat"]
        path = Path(run["sets_path"])
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


def collect_usable_runs(
    output_root: str,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs_root = Path(output_root) / "subtype_review" / "runs"
    params = config.get("multi_k", DEFAULT_MULTI_K_CONFIG)
    usable_runs = []
    audit = []
    for initial_k in sorted(set(map(int, params["initial_ks"]))):
        for repeat in sorted(set(map(int, params["repeats"]))):
            run_root = runs_root / f"K{initial_k}" / f"repeat{repeat}"
            inspection = inspect_review_run(run_root)
            detail = {
                "initial_k": initial_k, "repeat": repeat, "run_root": str(run_root),
                "status": inspection["status"], "reason": inspection["reason"],
                "input_signature": inspection["input_signature"],
                "candidate_signature": inspection["candidate_signature"],
                "sets_path": inspection["sets_path"],
                "error_type": inspection["error_type"],
                "error_message": inspection["error_message"],
            }
            audit.append(detail)
            if inspection["status"] == "complete":
                usable_runs.append({
                    "initial_k": initial_k, "repeat": repeat,
                    "run_root": str(run_root),
                    "sets_path": inspection["sets_path"],
                    "input_signature": inspection["input_signature"],
                    "candidate_signature": inspection["candidate_signature"],
                })
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


def mutual_recurrence_similarity(
    observations: list[dict[str, Any]],
    patient_ids: list[str],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], float], list[dict[str, Any]]]:
    occurrence_sets = {patient_id: set() for patient_id in patient_ids}
    for row in observations:
        for patient_id in row["member_ids"]:
            occurrence_sets.setdefault(str(patient_id), set()).add(row["observation_id"])
    similarities: dict[tuple[str, str], float] = {}
    pair_rows = []
    for index, left_id in enumerate(patient_ids):
        for right_id in patient_ids[index + 1:]:
            left = occurrence_sets[left_id]
            right = occurrence_sets[right_id]
            shared = sorted(left.intersection(right))
            denominator = max(len(left), len(right))
            similarity = len(shared) / denominator if denominator else 0.0
            similarities[(left_id, right_id)] = similarity
            pair_rows.append({
                "patient_id_left": left_id,
                "patient_id_right": right_id,
                "left_occurrence_count": len(left),
                "right_occurrence_count": len(right),
                "shared_accept_set_count": len(shared),
                "similarity": similarity,
                "shared_accept_set_ids": json.dumps(shared, ensure_ascii=False),
            })
    return occurrence_sets, similarities, pair_rows


def select_patient_recurrence_subtypes(
    observations: list[dict[str, Any]],
    patient_ids: list[str],
    *,
    similarity_threshold: float,
    min_subtype_size: int,
    min_common_occurrences: int,
    min_supporting_ks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    occurrence_sets, similarities, pair_rows = mutual_recurrence_similarity(observations, patient_ids)
    n_patients = len(patient_ids)
    if n_patients < 2:
        labels = [1] * n_patients
    else:
        distances = [[0.0] * n_patients for _ in range(n_patients)]
        for left_index, left_id in enumerate(patient_ids):
            for right_index in range(left_index + 1, n_patients):
                right_id = patient_ids[right_index]
                similarity = similarities[(left_id, right_id)]
                distances[left_index][right_index] = 1.0 - similarity
                distances[right_index][left_index] = 1.0 - similarity
        labels = fcluster(
            linkage(squareform(distances, checks=True), method="complete"),
            t=1.0 - similarity_threshold,
            criterion="distance",
        ).tolist()
    patient_by_label: dict[int, list[str]] = {}
    for patient_id, label in zip(patient_ids, labels):
        patient_by_label.setdefault(int(label), []).append(patient_id)
    observation_map = {row["observation_id"]: row for row in observations}
    candidates = []
    for members in patient_by_label.values():
        member_sets = [occurrence_sets[patient_id] for patient_id in members]
        common_observations = sorted(set.intersection(*member_sets)) if member_sets else []
        supporting_ks = sorted({int(observation_map[observation_id]["initial_k"]) for observation_id in common_observations})
        candidates.append({
            "member_ids": sorted(members),
            "member_count": len(members),
            "common_accept_set_count": len(common_observations),
            "supporting_observation_ids": common_observations,
            "supporting_k_count": len(supporting_ks),
            "supporting_ks": supporting_ks,
            "supporting_runs": sorted({
                f"K{observation_map[observation_id]['initial_k']}_repeat{observation_map[observation_id]['repeat']}"
                for observation_id in common_observations
            }),
        })
    candidates.sort(key=lambda row: (-row["common_accept_set_count"], -row["supporting_k_count"], -row["member_count"], tuple(row["member_ids"])))
    subtypes = []
    for candidate in candidates:
        if (
            candidate["member_count"] < min_subtype_size
            or candidate["common_accept_set_count"] < min_common_occurrences
            or candidate["supporting_k_count"] < min_supporting_ks
        ):
            continue
        subtype = {
            "subtype_id": f"SUBTYPE{len(subtypes) + 1:02d}",
            **candidate,
            "occurrence_frequency": candidate["common_accept_set_count"] / len(observations) if observations else 0.0,
        }
        subtypes.append(subtype)
    return subtypes, pair_rows


def write_patient_recurrence_outputs(
    output_dir: Path,
    subtypes: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    write_json(output_dir / "patient_recurrence_subtypes.json", subtypes)
    pd.DataFrame([
        {
            "subtype_rank": rank,
            "subtype_id": row["subtype_id"],
            "member_count": row["member_count"],
            "common_accept_set_count": row["common_accept_set_count"],
            "occurrence_frequency": row["occurrence_frequency"],
            "supporting_k_count": row["supporting_k_count"],
            "supporting_ks": json.dumps(row["supporting_ks"]),
            "supporting_observation_ids": json.dumps(row["supporting_observation_ids"]),
            "supporting_runs": json.dumps(row["supporting_runs"]),
            "member_ids": json.dumps(row["member_ids"], ensure_ascii=False),
        }
        for rank, row in enumerate(subtypes, 1)
    ], columns=["subtype_rank", "subtype_id", "member_count", "common_accept_set_count", "occurrence_frequency", "supporting_k_count", "supporting_ks", "supporting_observation_ids", "supporting_runs", "member_ids"]).to_csv(output_dir / "patient_recurrence_summary.csv", index=False)
    pd.DataFrame([
        {"subtype_id": row["subtype_id"], "patient_id": patient_id}
        for row in subtypes for patient_id in row["member_ids"]
    ], columns=["subtype_id", "patient_id"]).to_csv(output_dir / "patient_recurrence_membership.csv", index=False)
    pd.DataFrame(pair_rows, columns=["patient_id_left", "patient_id_right", "left_occurrence_count", "right_occurrence_count", "shared_accept_set_count", "similarity", "shared_accept_set_ids"]).to_csv(output_dir / "patient_recurrence_pair_similarity.csv", index=False)


def write_recurrent_outputs(output_dir: Path, observations: list[dict[str, Any]], recurrent_sets: list[dict[str, Any]], *, aggregation_manifest: Mapping[str, Any], aggregation_signature: str, run_count: int, min_subtype_size: int, run_audit: list[dict[str, Any]], patient_subtypes: list[dict[str, Any]], patient_recurrence: Mapping[str, Any]) -> dict[str, Any]:
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
    included = [row for row in run_audit if row["status"] == "complete"]
    excluded = [row for row in run_audit if row["status"] != "complete"]
    usable_runs_by_k = {
        str(initial_k): sum(
            row["status"] == "complete" and row["initial_k"] == initial_k
            for row in run_audit
        )
        for initial_k in sorted({row["initial_k"] for row in run_audit})
    }
    summary = {
        "status": "complete", "analysis_type": "patient_recurrence_subtype_clustering",
        "aggregation_signature": aggregation_signature, "run_count": run_count,
        "included_run_count": len(included), "excluded_run_count": len(excluded),
        "total_accept_set_count": len(observations), "min_subtype_size": min_subtype_size,
        "recurrent_set_count": len(recurrent_sets),
        "patient_recurrence": {
            "method": "complete_linkage_patient_occurrence_clustering",
            "similarity_metric": "mutual_recurrence_similarity",
            "similarity_threshold": patient_recurrence["similarity_threshold"],
            "min_subtype_size": patient_recurrence["min_subtype_size"],
            "min_common_occurrences": patient_recurrence["min_common_occurrences"],
            "min_supporting_ks": patient_recurrence["min_supporting_ks"],
            "selected_subtype_count": len(patient_subtypes),
        },
    }
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
) -> dict[str, Any]:
    root = Path(output_root)
    patient_order_path = root / "candidate_subtype" / "affinity_patient_order.json"
    patient_values = json.loads(patient_order_path.read_text(encoding="utf-8"))
    patient_ids = [str(value) for value in patient_values]
    patient_set = set(patient_ids)
    if len(patient_set) != len(patient_ids):
        raise ValueError("Candidate patient order contains duplicate IDs")
    params = config.get("multi_k", DEFAULT_MULTI_K_CONFIG)
    run_files, run_audit = collect_usable_runs(output_root, config)
    if not run_files:
        raise ValueError("No usable completed Agent runs were found.")
    candidate_signatures = {
        run.get("candidate_signature") for run in run_files if run.get("candidate_signature")
    }
    if len(candidate_signatures) > 1:
        raise ValueError(
            "Included Agent runs reference multiple candidate signatures: "
            + ", ".join(sorted(map(str, candidate_signatures)))
        )
    configured_initial_ks = sorted(set(map(int, params["initial_ks"])))
    configured_repeats = sorted(set(map(int, params["repeats"])))
    included_runs = [
        {"initial_k": run["initial_k"], "repeat": run["repeat"]}
        for run in run_files
    ]
    included_initial_ks = sorted({row["initial_k"] for row in included_runs})
    included_repeats = sorted({row["repeat"] for row in included_runs})
    aggregation_params = {
        **params,
        "initial_ks": configured_initial_ks,
        "repeats": configured_repeats,
    }
    run_manifest = []
    for run in run_files:
        sets_path = Path(run["sets_path"])
        run_manifest.append({
            "initial_k": run["initial_k"],
            "repeat": run["repeat"],
            "agent_input_signature": run.get("input_signature"),
            "candidate_signature": run.get("candidate_signature"),
            "final_subtype_sets_path": str(sets_path.resolve()),
            "final_subtype_sets_sha256": hashlib.sha256(sets_path.read_bytes()).hexdigest(),
        })
    input_signatures = sorted({str(run["input_signature"]) for run in run_files if run.get("input_signature")})
    aggregation_manifest = {
        "aggregation_version": 2,
        "agent_input_signatures": input_signatures,
        "candidate_signature": next(iter(candidate_signatures), None),
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
    patient_params = params.get("patient_recurrence", {})
    patient_recurrence = {
        "similarity_threshold": float(patient_params.get("similarity_threshold", 0.9)),
        "min_subtype_size": int(patient_params.get("min_subtype_size", params["min_subtype_size"])),
        "min_common_occurrences": int(patient_params.get("min_common_occurrences", 2)),
        "min_supporting_ks": int(patient_params.get("min_supporting_ks", 2)),
    }
    patient_subtypes, pair_rows = select_patient_recurrence_subtypes(
        observations, patient_ids, **patient_recurrence
    )
    write_patient_recurrence_outputs(output_dir, patient_subtypes, pair_rows)
    return write_recurrent_outputs(
        output_dir, observations, recurrent_sets,
        aggregation_manifest=aggregation_manifest,
        aggregation_signature=aggregation_signature,
        run_count=len(run_files), min_subtype_size=int(params["min_subtype_size"]),
        run_audit=run_audit, patient_subtypes=patient_subtypes,
        patient_recurrence=patient_recurrence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute recurrent ACCEPT-set statistics from completed Agent runs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    config = load_yaml_file(Path(args.config_dir) / "subtype_review.yaml")
    print(json.dumps(
        run_multi_k_aggregation(args.output_root, config),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
