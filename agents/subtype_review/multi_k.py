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
    params = config["multi_k"]
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


def overlap_coefficient(
    left_members: list[str] | set[str],
    right_members: list[str] | set[str],
) -> tuple[int, float]:
    left = set(left_members)
    right = set(right_members)
    denominator = min(len(left), len(right))
    if denominator == 0:
        return 0, 0.0
    intersection_n = len(left.intersection(right))
    return intersection_n, intersection_n / denominator


def select_stable_cores(
    recurrent_sets: list[dict[str, Any]],
    *,
    min_occurrences: int,
    min_supporting_ks: int,
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        recurrent_sets,
        key=lambda row: (
            -int(row["occurrence_count"]),
            -int(row["supporting_k_count"]),
            -int(row["member_count"]),
            str(row["recurrent_set_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for candidate in ranked:
        occurrence_ok = int(candidate["occurrence_count"]) >= min_occurrences
        k_support_ok = int(candidate["supporting_k_count"]) >= min_supporting_ks
        base = {
            "recurrent_set_id": candidate["recurrent_set_id"],
            "member_count": candidate["member_count"],
            "occurrence_count": candidate["occurrence_count"],
            "supporting_k_count": candidate["supporting_k_count"],
            "eligible": occurrence_ok and k_support_ok,
            "status": "insufficient_recurrence",
            "core_id": None,
            "representative_recurrent_set_id": None,
            "shared_member_count": 0,
            "overlap_coefficient": 0.0,
            "reason": None,
        }
        if not occurrence_ok or not k_support_ok:
            if not occurrence_ok and not k_support_ok:
                base["reason"] = "occurrence_and_k_support_below_minimum"
            elif not occurrence_ok:
                base["reason"] = "occurrence_count_below_minimum"
            else:
                base["reason"] = "supporting_k_count_below_minimum"
            mapping.append(base)
            continue

        best_core = None
        best_overlap = -1.0
        best_intersection = 0
        for core in selected:
            intersection_n, overlap = overlap_coefficient(candidate["member_ids"], core["member_ids"])
            if overlap > best_overlap:
                best_core = core
                best_overlap = overlap
                best_intersection = intersection_n
        if best_core is not None and best_overlap >= overlap_threshold:
            base.update({
                "status": "redundant_variant",
                "core_id": best_core["core_id"],
                "representative_recurrent_set_id": best_core["representative_recurrent_set_id"],
                "shared_member_count": best_intersection,
                "overlap_coefficient": best_overlap,
            })
            best_core.setdefault("variant_recurrent_set_ids", []).append(candidate["recurrent_set_id"])
            mapping.append(base)
            continue

        core = {
            "core_id": f"CORE{len(selected) + 1:02d}",
            "representative_recurrent_set_id": candidate["recurrent_set_id"],
            "member_ids": list(candidate["member_ids"]),
            "member_count": candidate["member_count"],
            "occurrence_count": candidate["occurrence_count"],
            "total_accept_set_count": candidate["total_accept_set_count"],
            "accept_set_frequency": candidate["accept_set_frequency"],
            "supporting_run_count": candidate["supporting_run_count"],
            "usable_run_count": candidate["usable_run_count"],
            "run_support_frequency": candidate["run_support_frequency"],
            "supporting_k_count": candidate["supporting_k_count"],
            "supporting_ks": list(candidate["supporting_ks"]),
            "supporting_runs": list(candidate["supporting_runs"]),
            "supporting_observation_ids": list(candidate["supporting_observation_ids"]),
            "variant_recurrent_set_ids": [],
            "variant_count": 0,
        }
        selected.append(core)
        base.update({
            "status": "selected_core",
            "core_id": core["core_id"],
            "representative_recurrent_set_id": core["representative_recurrent_set_id"],
            "shared_member_count": candidate["member_count"],
            "overlap_coefficient": 1.0,
        })
        mapping.append(base)
    for core in selected:
        core["variant_count"] = len(core["variant_recurrent_set_ids"])
    mapping.sort(key=lambda row: str(row["recurrent_set_id"]))
    return selected, mapping


def write_core_outputs(output_dir: Path, selected_cores: list[dict[str, Any]], core_mapping: list[dict[str, Any]]) -> None:
    write_json(output_dir / "stable_core_subtypes.json", selected_cores)
    pd.DataFrame([
        {
            "core_rank": rank, "core_id": core["core_id"],
            "representative_recurrent_set_id": core["representative_recurrent_set_id"],
            "member_count": core["member_count"], "occurrence_count": core["occurrence_count"],
            "total_accept_set_count": core["total_accept_set_count"],
            "accept_set_frequency": core["accept_set_frequency"],
            "supporting_run_count": core["supporting_run_count"], "usable_run_count": core["usable_run_count"],
            "run_support_frequency": core["run_support_frequency"], "supporting_k_count": core["supporting_k_count"],
            "supporting_ks": json.dumps(core["supporting_ks"]), "variant_count": core["variant_count"],
            "member_ids": json.dumps(core["member_ids"], ensure_ascii=False),
        }
        for rank, core in enumerate(selected_cores, 1)
    ], columns=["core_rank", "core_id", "representative_recurrent_set_id", "member_count", "occurrence_count", "total_accept_set_count", "accept_set_frequency", "supporting_run_count", "usable_run_count", "run_support_frequency", "supporting_k_count", "supporting_ks", "variant_count", "member_ids"]).to_csv(output_dir / "stable_core_summary.csv", index=False)
    pd.DataFrame(core_mapping, columns=["recurrent_set_id", "member_count", "occurrence_count", "supporting_k_count", "eligible", "status", "core_id", "representative_recurrent_set_id", "shared_member_count", "overlap_coefficient", "reason"]).to_csv(output_dir / "recurrent_set_core_map.csv", index=False)
    pairs = []
    for index, left in enumerate(selected_cores):
        for right in selected_cores[index + 1:]:
            shared = sorted(set(left["member_ids"]).intersection(right["member_ids"]))
            _, overlap = overlap_coefficient(left["member_ids"], right["member_ids"])
            pairs.append({
                "left_core_id": left["core_id"], "right_core_id": right["core_id"],
                "left_member_count": left["member_count"], "right_member_count": right["member_count"],
                "shared_member_count": len(shared), "overlap_coefficient": overlap,
                "shared_member_ids": json.dumps(shared, ensure_ascii=False),
            })
    pd.DataFrame(pairs, columns=["left_core_id", "right_core_id", "left_member_count", "right_member_count", "shared_member_count", "overlap_coefficient", "shared_member_ids"]).to_csv(output_dir / "stable_core_pair_overlap.csv", index=False)


def patient_recurrence_similarity(
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
            denominator = min(len(left), len(right))
            shared = sorted(left.intersection(right))
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
    occurrence_sets, similarities, pair_rows = patient_recurrence_similarity(observations, patient_ids)
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


def write_recurrent_outputs(output_dir: Path, observations: list[dict[str, Any]], recurrent_sets: list[dict[str, Any]], *, aggregation_manifest: Mapping[str, Any], aggregation_signature: str, run_count: int, min_subtype_size: int, run_audit: list[dict[str, Any]], selected_cores: list[dict[str, Any]], core_selection: Mapping[str, Any], patient_subtypes: list[dict[str, Any]], patient_recurrence: Mapping[str, Any]) -> dict[str, Any]:
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
        "eligible_core_candidate_count": sum(row["eligible"] for row in core_selection["mapping"]),
        "selected_core_count": len(selected_cores),
        "core_selection": {
            "method": "frequency_ranked_overlap_pruning",
            "min_occurrences": core_selection["min_occurrences"],
            "min_supporting_ks": core_selection["min_supporting_ks"],
            "overlap_metric": "overlap_coefficient",
            "overlap_threshold": core_selection["overlap_threshold"],
        },
        "patient_recurrence": {
            "method": "complete_linkage_patient_occurrence_clustering",
            "similarity_metric": "overlap_coefficient",
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
    params = config["multi_k"]
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
    core_params = params.get("core_selection", {})
    core_selection = {
        "min_occurrences": int(core_params.get("min_occurrences", 2)),
        "min_supporting_ks": int(core_params.get("min_supporting_ks", 2)),
        "overlap_threshold": float(core_params.get("overlap_threshold", 0.8)),
    }
    selected_cores, core_mapping = select_stable_cores(recurrent_sets, **core_selection)
    write_core_outputs(output_dir, selected_cores, core_mapping)
    core_selection["mapping"] = core_mapping
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
        run_audit=run_audit, selected_cores=selected_cores,
        core_selection=core_selection, patient_subtypes=patient_subtypes,
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
