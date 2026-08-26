#!/usr/bin/env python3
"""Review saved initial-K partitions and analyze accepted-set families and patient cores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import save_review_outputs  # noqa: E402
from agents.subtype_review.runner import run_subtype_review  # noqa: E402
from scripts_2026_8_17.experiment_initial_k_review_sensitivity import (  # noqa: E402
    labels_to_candidate_sets,
    load_affinity_patient_ids,
    load_patient_states,
)
from utils.io import write_json  # noqa: E402

INITIAL_KS = tuple(range(2, 9))
REPEATS = (1, 2, 3)
THRESHOLDS = (0.5, 2 / 3, 0.75, 0.8)
FAMILY_JACCARD_THRESHOLD = 0.5
FAMILY_OVERLAP_THRESHOLD = 0.8
FAMILY_THRESHOLD_GRID = (
    (0.5, 0.8),
    (0.5, 1.0),
    (2 / 3, 0.8),
    (2 / 3, 1.0),
    (0.75, 0.8),
    (0.75, 1.0),
)
MIN_CONCURRENT_ACCEPTED_RUNS_FOR_CONDITIONAL_PLOT = 2
DEFAULT_DATA_ROOT = ROOT / "output_kirc"
DEFAULT_EXPERIMENT_ROOT = (
    ROOT / "output_kirc_v9" / "experiment_multi_k_accepted_core_stability"
)


def accepted_assignments(final_sets: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    assignments = {}
    for item in final_sets:
        set_id = str(item.get("set_id", item.get("cluster_id", "")) or "")
        if not set_id:
            raise ValueError("Accepted set has no set_id")
        for patient_id in item.get("member_ids", []) or []:
            patient_id = str(patient_id)
            if patient_id in assignments:
                raise ValueError(f"Patient occurs in multiple accepted sets: {patient_id}")
            assignments[patient_id] = set_id
    return dict(sorted(assignments.items()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_data_signature(paths: Sequence[Path]) -> str:
    return hashlib.sha256(
        json.dumps(
            {str(path): sha256_file(path) for path in sorted(paths)},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def scientifically_terminal(summary: Mapping[str, Any]) -> bool:
    raw_status = str(summary.get("raw_control_status", ""))
    review_status = str(summary.get("status", ""))
    return (
        raw_status == "complete"
        and review_status == "review_complete"
    )


def set_similarity(left: set[str], right: set[str]) -> dict[str, float]:
    intersection = len(left & right)
    union = len(left | right)
    smaller = min(len(left), len(right))
    return {
        "jaccard": intersection / union if union else 0.0,
        "dice": 2 * intersection / (len(left) + len(right)) if left or right else 0.0,
        "overlap": intersection / smaller if smaller else 0.0,
    }


def compare_runs(left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, Any]:
    left_ids, right_ids = set(left), set(right)
    shared = sorted(left_ids & right_ids)
    union = left_ids | right_ids
    result = {
        "left_accepted_count": len(left_ids),
        "right_accepted_count": len(right_ids),
        "accepted_intersection_count": len(shared),
        "accepted_union_count": len(union),
        "both_empty": not union,
        "accepted_coverage_jaccard": len(shared) / len(union) if union else None,
        "adjusted_rand_index": None,
        "adjusted_mutual_information": None,
    }
    if len(shared) >= 2:
        left_labels = [left[item] for item in shared]
        right_labels = [right[item] for item in shared]
        result["adjusted_rand_index"] = float(
            adjusted_rand_score(left_labels, right_labels)
        )
        result["adjusted_mutual_information"] = float(
            adjusted_mutual_info_score(left_labels, right_labels)
        )

    left_groups = [
        (group, {patient for patient, label in left.items() if label == group})
        for group in sorted(set(left.values()))
    ]
    right_groups = [
        (group, {patient for patient, label in right.items() if label == group})
        for group in sorted(set(right.values()))
    ]
    denominator = max(len(left_groups), len(right_groups))
    if not denominator:
        result.update({"matched_mean_jaccard": None, "matched_mean_dice": None, "matched_set_pairs": []})
        return result
    scores = np.zeros((len(left_groups), len(right_groups)), dtype=float)
    dice = np.zeros_like(scores)
    overlaps = np.zeros_like(scores)
    for i, (_, left_group) in enumerate(left_groups):
        for j, (_, right_group) in enumerate(right_groups):
            metrics = set_similarity(left_group, right_group)
            scores[i, j] = metrics["jaccard"]
            dice[i, j] = metrics["dice"]
            overlaps[i, j] = metrics["overlap"]
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        result["matched_mean_jaccard"] = float(scores[rows, columns].sum() / denominator)
        result["matched_mean_dice"] = float(dice[rows, columns].sum() / denominator)
        result["matched_set_pairs"] = [
            {
                "left_set_id": left_groups[i][0],
                "right_set_id": right_groups[j][0],
                "jaccard": float(scores[i, j]),
                "dice": float(dice[i, j]),
                "overlap": float(overlaps[i, j]),
            }
            for i, j in zip(rows, columns)
        ]
    else:
        result.update({"matched_mean_jaccard": 0.0, "matched_mean_dice": 0.0, "matched_set_pairs": []})
    return result


def accepted_set_catalog(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    catalog = []
    for run in runs:
        groups: dict[str, set[str]] = {}
        for patient_id, set_id in dict(run["assignments"]).items():
            groups.setdefault(str(set_id), set()).add(str(patient_id))
        for set_id, members in sorted(groups.items()):
            catalog.append(
                {
                    "node_id": f"{run['run_id']}::{set_id}",
                    "run_id": str(run["run_id"]),
                    "initial_k": int(run["initial_k"]),
                    "repeat": int(run["repeat"]),
                    "set_id": set_id,
                    "member_ids": sorted(members),
                    "member_n": len(members),
                }
            )
    return catalog


def accepted_set_similarity(
    catalog: Sequence[Mapping[str, Any]],
    *,
    jaccard_threshold: float = FAMILY_JACCARD_THRESHOLD,
    overlap_threshold: float = FAMILY_OVERLAP_THRESHOLD,
) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(catalog, 2):
        if left["run_id"] == right["run_id"]:
            continue
        rows.append(
            {
                "left_node_id": left["node_id"],
                "right_node_id": right["node_id"],
                "left_run_id": left["run_id"],
                "right_run_id": right["run_id"],
                "left_initial_k": left["initial_k"],
                "right_initial_k": right["initial_k"],
                **set_similarity(set(left["member_ids"]), set(right["member_ids"])),
            }
        )
    for row in rows:
        if row["jaccard"] < jaccard_threshold or row["overlap"] < overlap_threshold:
            row["relation_type"] = "below_threshold"
            continue
        run_pair = tuple(sorted((row["left_run_id"], row["right_run_id"])))
        node_a = (
            row["left_node_id"]
            if row["left_run_id"] == run_pair[0]
            else row["right_node_id"]
        )
        node_b = (
            row["right_node_id"]
            if row["right_run_id"] == run_pair[1]
            else row["left_node_id"]
        )
        left_matches = set()
        right_matches = set()
        for other in rows:
            if (
                tuple(sorted((other["left_run_id"], other["right_run_id"]))) != run_pair
                or other["jaccard"] < jaccard_threshold
                or other["overlap"] < overlap_threshold
            ):
                continue
            for node_id, matches in (
                (node_a, left_matches),
                (node_b, right_matches),
            ):
                if node_id == other["left_node_id"]:
                    matches.add(other["right_node_id"])
                elif node_id == other["right_node_id"]:
                    matches.add(other["left_node_id"])
        row["relation_type"] = {
            (1, 1): "one_to_one",
            (1, 2): "many_to_one",
            (2, 1): "one_to_many",
        }.get((min(len(left_matches), 2), min(len(right_matches), 2)), "many_to_many")
    return rows


def accepted_set_families(
    catalog: Sequence[Mapping[str, Any]],
    similarities: Sequence[Mapping[str, Any]],
    *,
    jaccard_threshold: float,
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    node_ids = [str(item["node_id"]) for item in catalog]
    adjacency = {node_id: set() for node_id in node_ids}
    for row in similarities:
        if (
            float(row["jaccard"]) >= jaccard_threshold
            and float(row["overlap"]) >= overlap_threshold
        ):
            left, right = str(row["left_node_id"]), str(row["right_node_id"])
            adjacency[left].add(right)
            adjacency[right].add(left)

    components = []
    unseen = set(node_ids)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = []
        unseen.remove(start)
        while stack:
            node_id = stack.pop()
            component.append(node_id)
            for neighbor in sorted(adjacency[node_id] & unseen):
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))

    catalog_by_id = {str(item["node_id"]): item for item in catalog}
    similarity_by_pair = {
        frozenset((row["left_node_id"], row["right_node_id"])): row
        for row in similarities
    }
    families = []
    for family_number, node_group in enumerate(sorted(components, key=lambda group: group[0]), 1):
        items = [catalog_by_id[node_id] for node_id in node_group]
        same_run_conflict_count = sum(
            left["run_id"] == right["run_id"]
            for left, right in combinations(items, 2)
        )
        pair_metrics = [
            similarity_by_pair[frozenset((left, right))]
            for left, right in combinations(node_group, 2)
            if frozenset((left, right)) in similarity_by_pair
        ]
        k_coverage = len({int(item["initial_k"]) for item in items})
        is_recurrent = len(node_group) >= 2 and k_coverage >= 2
        is_cohesive = (
            is_recurrent
            and same_run_conflict_count == 0
            and len(pair_metrics) == len(node_group) * (len(node_group) - 1) // 2
            and all(
                float(row["jaccard"]) >= jaccard_threshold
                and float(row["overlap"]) >= overlap_threshold
                for row in pair_metrics
            )
        )
        frequencies = Counter(
            patient_id
            for item in items
            for patient_id in item["member_ids"]
        )
        k_frequencies = Counter()
        for initial_k in sorted({int(item["initial_k"]) for item in items}):
            k_members = {
                patient_id
                for item in items
                if int(item["initial_k"]) == initial_k
                for patient_id in item["member_ids"]
            }
            k_frequencies.update(k_members)
        families.append(
            {
                "family_id": f"FAMILY{family_number:02d}",
                "node_ids": node_group,
                "run_ids": sorted({item["run_id"] for item in items}),
                "initial_k_values": sorted({int(item["initial_k"]) for item in items}),
                "repeat_values": sorted({int(item["repeat"]) for item in items}),
                "k_coverage": k_coverage,
                "repeat_coverage": len({int(item["repeat"]) for item in items}),
                "member_ids": sorted(frequencies),
                "member_frequency": dict(sorted(frequencies.items())),
                "k_member_frequency": dict(sorted(k_frequencies.items())),
                "same_run_conflict": same_run_conflict_count > 0,
                "same_run_conflict_count": same_run_conflict_count,
                "is_recurrent_relation_component": is_recurrent,
                "is_cohesive_family": is_cohesive,
                "is_orphan_set": len(node_group) == 1,
                "mean_jaccard": float(np.mean([row["jaccard"] for row in pair_metrics])) if pair_metrics else None,
                "minimum_jaccard": float(np.min([row["jaccard"] for row in pair_metrics])) if pair_metrics else None,
                "mean_dice": float(np.mean([row["dice"] for row in pair_metrics])) if pair_metrics else None,
                "minimum_dice": float(np.min([row["dice"] for row in pair_metrics])) if pair_metrics else None,
                "mean_overlap": float(np.mean([row["overlap"] for row in pair_metrics])) if pair_metrics else None,
                "minimum_overlap": float(np.min([row["overlap"] for row in pair_metrics])) if pair_metrics else None,
            }
        )
    return families


def cohesive_set_families(
    catalog: Sequence[Mapping[str, Any]],
    similarities: Sequence[Mapping[str, Any]],
    relation_components: Sequence[Mapping[str, Any]],
    *,
    jaccard_threshold: float,
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    catalog_by_id = {str(item["node_id"]): item for item in catalog}
    similarity_by_pair = {
        frozenset((row["left_node_id"], row["right_node_id"])): row
        for row in similarities
    }
    families = []
    for component in relation_components:
        node_ids = [str(node_id) for node_id in component["node_ids"]]
        if len(node_ids) < 2:
            continue
        distances = np.ones((len(node_ids), len(node_ids)), dtype=float)
        np.fill_diagonal(distances, 0)
        for left_index, left_id in enumerate(node_ids):
            for right_index in range(left_index + 1, len(node_ids)):
                right_id = node_ids[right_index]
                row = similarity_by_pair.get(frozenset((left_id, right_id)))
                if (
                    row
                    and row.get("relation_type") == "one_to_one"
                    and float(row["jaccard"]) >= jaccard_threshold
                    and float(row["overlap"]) >= overlap_threshold
                ):
                    distances[left_index, right_index] = 0
                    distances[right_index, left_index] = 0
        labels = fcluster(
            linkage(squareform(distances, checks=False), method="complete"),
            t=1 - 1e-12,
            criterion="distance",
        )
        for label in sorted(set(labels)):
            group = [node_ids[index] for index, value in enumerate(labels) if value == label]
            items = [catalog_by_id[node_id] for node_id in group]
            if len(group) < 2 or len({int(item["initial_k"]) for item in items}) < 2:
                continue
            pair_metrics = [
                similarity_by_pair[frozenset((left, right))]
                for left, right in combinations(group, 2)
            ]
            if any(
                row.get("relation_type") != "one_to_one"
                for row in pair_metrics
            ):
                continue
            node_count_by_k = Counter(int(item["initial_k"]) for item in items)
            member_frequency = Counter(
                patient_id
                for item in items
                for patient_id in item["member_ids"]
            )
            families.append(
                {
                    "family_id": f"FAMILY{len(families) + 1:02d}",
                    "relation_component_id": component["family_id"],
                    "node_ids": sorted(group),
                    "run_ids": sorted({item["run_id"] for item in items}),
                    "initial_k_values": sorted(node_count_by_k),
                    "repeat_values": sorted({int(item["repeat"]) for item in items}),
                    "k_coverage": len(node_count_by_k),
                    "repeat_coverage": len({int(item["repeat"]) for item in items}),
                    "member_ids": sorted(member_frequency),
                    "member_frequency": dict(sorted(member_frequency.items())),
                    "same_run_conflict": False,
                    "same_run_conflict_count": 0,
                    "is_recurrent_relation_component": True,
                    "is_cohesive_family": True,
                    "is_orphan_set": False,
                    "mean_jaccard": float(np.mean([row["jaccard"] for row in pair_metrics])),
                    "minimum_jaccard": float(np.min([row["jaccard"] for row in pair_metrics])),
                    "mean_dice": float(np.mean([row["dice"] for row in pair_metrics])),
                    "minimum_dice": float(np.min([row["dice"] for row in pair_metrics])),
                    "mean_overlap": float(np.mean([row["overlap"] for row in pair_metrics])),
                    "minimum_overlap": float(np.min([row["overlap"] for row in pair_metrics])),
                }
            )
    return families


def family_layer(
    runs: Sequence[Mapping[str, Any]],
    *,
    jaccard_threshold: float,
    overlap_threshold: float,
) -> dict[str, Any]:
    catalog = accepted_set_catalog(runs)
    similarities = accepted_set_similarity(
        catalog,
        jaccard_threshold=jaccard_threshold,
        overlap_threshold=overlap_threshold,
    )
    components = accepted_set_families(
        catalog,
        similarities,
        jaccard_threshold=jaccard_threshold,
        overlap_threshold=overlap_threshold,
    )
    families = cohesive_set_families(
        catalog,
        similarities,
        components,
        jaccard_threshold=jaccard_threshold,
        overlap_threshold=overlap_threshold,
    )
    runs_by_k: dict[int, list[Mapping[str, Any]]] = {}
    catalog_by_id = {str(item["node_id"]): item for item in catalog}
    for run in runs:
        runs_by_k.setdefault(int(run["initial_k"]), []).append(run)
    for family in families:
        family_nodes = {
            node_id: catalog_by_id[node_id]
            for node_id in family["node_ids"]
        }
        family_nodes_by_run = {
            item["run_id"]: item for item in family_nodes.values()
        }
        presence_count_by_k = {}
        valid_count_by_k = {}
        conditional_by_patient: dict[str, dict[str, float | None]] = {
            patient_id: {} for patient_id in family["member_ids"]
        }
        unconditional_by_patient: dict[str, dict[str, float]] = {
            patient_id: {} for patient_id in family["member_ids"]
        }
        for initial_k, k_runs in sorted(runs_by_k.items()):
            present_items = [
                family_nodes_by_run[run["run_id"]]
                for run in k_runs
                if run["run_id"] in family_nodes_by_run
            ]
            valid_count_by_k[str(initial_k)] = len(k_runs)
            presence_count_by_k[str(initial_k)] = len(present_items)
            for patient_id in family["member_ids"]:
                present_count = sum(
                    patient_id in item["member_ids"] for item in present_items
                )
                conditional_by_patient[patient_id][str(initial_k)] = (
                    present_count / len(present_items) if present_items else None
                )
                unconditional_by_patient[patient_id][str(initial_k)] = (
                    present_count / len(k_runs) if k_runs else 0.0
                )
        family["family_presence_count_by_k"] = presence_count_by_k
        family["family_valid_run_count_by_k"] = valid_count_by_k
        family["family_presence_fraction_by_k"] = {
            initial_k: presence_count_by_k[initial_k] / valid_count_by_k[initial_k]
            if valid_count_by_k[initial_k]
            else 0.0
            for initial_k in valid_count_by_k
        }
        family["patient_membership_given_family_present_by_k"] = conditional_by_patient
        family["patient_unconditional_membership_by_k"] = unconditional_by_patient
    return {
        "catalog": catalog,
        "similarities": similarities,
        "components": components,
        "families": families,
    }


def stability_matrices(
    assignments_by_run: Sequence[Mapping[str, str]], patient_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not assignments_by_run:
        raise ValueError("No completed Review runs were provided")
    patient_ids = list(patient_ids)
    index = {patient_id: i for i, patient_id in enumerate(patient_ids)}
    same_set = np.zeros((len(patient_ids), len(patient_ids)), dtype=float)
    coaccepted = np.zeros_like(same_set)
    acceptance = np.zeros(len(patient_ids), dtype=float)
    for assignments in assignments_by_run:
        accepted = []
        groups: dict[str, list[int]] = {}
        for patient_id, label in assignments.items():
            if patient_id not in index:
                raise ValueError(f"Unknown patient in accepted set: {patient_id}")
            patient_index = index[patient_id]
            acceptance[patient_index] += 1
            accepted.append(patient_index)
            groups.setdefault(str(label), []).append(patient_index)
        coaccepted[np.ix_(accepted, accepted)] += 1
        for members in groups.values():
            same_set[np.ix_(members, members)] += 1
    denominator = len(assignments_by_run)
    conditional = np.divide(
        same_set,
        coaccepted,
        out=np.zeros_like(same_set),
        where=coaccepted > 0,
    )
    return same_set / denominator, coaccepted / denominator, conditional, acceptance / denominator


def aggregate_k_levels(
    runs: Sequence[Mapping[str, Any]],
    patient_ids: Sequence[str],
    initial_ks: Sequence[int],
    expected_repeat_count: int,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]]:
    levels = []
    matrices = {"primary": [], "strict": [], "exploratory": []}
    primary_minimum = 2 if expected_repeat_count >= 2 else 1
    for initial_k in initial_ks:
        k_runs = [run for run in runs if int(run["initial_k"]) == int(initial_k)]
        level = {
            "initial_k": int(initial_k),
            "expected_repeat_count": int(expected_repeat_count),
            "valid_run_count": len(k_runs),
            "valid_run_ids": [str(run["run_id"]) for run in k_runs],
            "status": (
                "complete"
                if len(k_runs) == expected_repeat_count
                else "low_confidence"
                if k_runs
                else "unavailable"
            ),
            "primary_eligible": len(k_runs) >= primary_minimum,
            "strict_eligible": len(k_runs) == expected_repeat_count,
        }
        levels.append(level)
        if k_runs:
            matrix = stability_matrices([run["assignments"] for run in k_runs], patient_ids)
            matrices["exploratory"].append(matrix)
            if len(k_runs) >= primary_minimum:
                matrices["primary"].append(matrix)
            if len(k_runs) == expected_repeat_count:
                matrices["strict"].append(matrix)
    return levels, matrices


def combine_k_matrices(
    matrices: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not matrices:
        raise ValueError("No K-level matrices are available")
    joint = np.mean([matrix[0] for matrix in matrices], axis=0)
    coacceptance = np.mean([matrix[1] for matrix in matrices], axis=0)
    acceptance = np.mean([matrix[3] for matrix in matrices], axis=0)
    conditional = np.divide(
        joint,
        coacceptance,
        out=np.zeros_like(joint),
        where=coacceptance > 0,
    )
    return joint, coacceptance, conditional, acceptance


def extract_cores(
    joint: np.ndarray,
    acceptance: np.ndarray,
    patient_ids: Sequence[str],
    *,
    threshold: float,
    min_size: int,
) -> list[dict[str, Any]]:
    eligible = np.flatnonzero(acceptance >= threshold)
    if len(eligible) < min_size:
        return []
    if len(eligible) == 1:
        labels = np.ones(1, dtype=int)
    else:
        similarity = joint[np.ix_(eligible, eligible)].clip(0, 1)
        np.fill_diagonal(similarity, 1)
        tree = linkage(squareform(1 - similarity, checks=False), method="complete")
        labels = fcluster(tree, t=1 - threshold + 1e-12, criterion="distance")
    groups = []
    for label in sorted(set(labels)):
        members = sorted(str(patient_ids[index]) for index in eligible[labels == label])
        if len(members) >= min_size:
            groups.append({"member_ids": members})
    return sorted(groups, key=lambda item: (-len(item["member_ids"]), item["member_ids"]))


def core_recurrence(
    member_ids: Sequence[str], runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    recovered_by_k: Counter[int] = Counter()
    all_accepted = 0
    same_set = 0
    for run in runs:
        assignments = dict(run["assignments"])
        labels = [assignments.get(patient_id) for patient_id in member_ids]
        if labels and None not in labels:
            all_accepted += 1
        if labels and None not in labels and len(set(labels)) == 1:
            same_set += 1
            recovered_by_k[int(run["initial_k"])] += 1
    return {
        "all_members_accepted_run_count": all_accepted,
        "all_members_accepted_run_fraction": all_accepted / len(runs) if runs else 0.0,
        "same_set_run_count": same_set,
        "same_set_run_fraction": same_set / len(runs) if runs else 0.0,
        "conditional_same_set_fraction": same_set / all_accepted if all_accepted else 0.0,
        "k_coverage_any": sum(count >= 1 for count in recovered_by_k.values()),
        "k_coverage_majority": sum(count >= 2 for count in recovered_by_k.values()),
        "recovered_runs_by_k": dict(sorted(recovered_by_k.items())),
    }


def analyze(
    experiment_root: Path,
    patient_ids: Sequence[str],
    initial_ks: Sequence[int],
    repeats: Sequence[int],
    min_core_size: int,
    family_jaccard_threshold: float = FAMILY_JACCARD_THRESHOLD,
    family_overlap_threshold: float = FAMILY_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    runs = []
    execution_rows = []
    for repeat in repeats:
        for initial_k in initial_ks:
            run_root = experiment_root / f"run{repeat}" / f"K{initial_k}"
            summary_path = run_root / "final_review_summary.json"
            metadata_path = run_root / "run_metadata.json"
            if not summary_path.exists() or not metadata_path.exists():
                execution_rows.append({
                    "run_id": f"run{repeat}_K{initial_k}", "initial_k": initial_k,
                    "repeat": repeat, "valid_for_analysis": False,
                    "raw_control_status": "missing", "review_status": "missing",
                    "partition_assessment": "missing",
                    "rounds_used": None, "accepted_set_count": None,
                    "accepted_patient_count": None, "api_calls": None, "total_tokens": None,
                })
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            raw_status = str(summary.get("raw_control_status", ""))
            review_status = str(summary.get("status", ""))
            valid_for_analysis = scientifically_terminal(summary)
            assignments = accepted_assignments(summary.get("accepted_subtype_sets", []))
            usage = dict(summary.get("llm_usage", {}) or {})
            row = {
                "run_id": f"run{repeat}_K{initial_k}", "initial_k": initial_k,
                "repeat": repeat, "valid_for_analysis": valid_for_analysis,
                "raw_control_status": raw_status, "review_status": review_status,
                "partition_assessment": summary.get("partition_assessment", "unavailable"),
                "rounds_used": summary.get("rounds_used"),
                "accepted_set_count": len(set(assignments.values())),
                "accepted_patient_count": len(assignments),
                "api_calls": usage.get("api_calls"), "total_tokens": usage.get("total_tokens"),
            }
            execution_rows.append(row)
            if valid_for_analysis:
                runs.append({**row, "assignments": assignments})

    experiment_root.mkdir(parents=True, exist_ok=True)
    with (experiment_root / "run_execution_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(execution_rows[0]) if execution_rows else [])
        if execution_rows:
            writer.writeheader()
            writer.writerows(execution_rows)
    write_json(experiment_root / "run_execution_status.json", {"runs": execution_rows})
    if not runs:
        summary = {
            "experiment": "multi_k_accepted_core_stability",
            "analysis_status": "unavailable",
            "expected_run_count": len(initial_ks) * len(repeats),
            "valid_run_count": 0,
            "invalid_run_count": len(execution_rows),
            "interpretation": "No scientifically complete Review run is available.",
        }
        write_json(experiment_root / "summary.json", summary)
        return summary

    run_rows = [
        {key: value for key, value in run.items() if key != "assignments"}
        for run in runs
    ]
    k_levels, k_matrices = aggregate_k_levels(
        runs, patient_ids, initial_ks, expected_repeat_count=len(repeats)
    )
    primary_ks = {
        int(level["initial_k"])
        for level in k_levels
        if level["primary_eligible"]
    }
    primary_runs = [
        run for run in runs if int(run["initial_k"]) in primary_ks
    ]
    strict_ks = {
        int(level["initial_k"])
        for level in k_levels
        if level["strict_eligible"]
    }
    family_layers = {
        "primary": family_layer(
            [run for run in runs if int(run["initial_k"]) in primary_ks],
            jaccard_threshold=family_jaccard_threshold,
            overlap_threshold=family_overlap_threshold,
        ),
        "strict": family_layer(
            [run for run in runs if int(run["initial_k"]) in strict_ks],
            jaccard_threshold=family_jaccard_threshold,
            overlap_threshold=family_overlap_threshold,
        ),
        "exploratory": family_layer(
            runs,
            jaccard_threshold=family_jaccard_threshold,
            overlap_threshold=family_overlap_threshold,
        ),
    }
    primary_layer = family_layers["primary"]
    catalog = primary_layer["catalog"]
    set_similarity_rows = primary_layer["similarities"]
    relation_components = primary_layer["components"]
    cohesive_families = primary_layer["families"]
    recurrent_families = [
        component
        for component in relation_components
        if component["is_recurrent_relation_component"]
    ]
    family_threshold_rows = []
    for jaccard_threshold, overlap_threshold in sorted({
        *FAMILY_THRESHOLD_GRID,
        (family_jaccard_threshold, family_overlap_threshold),
    }):
        threshold_layer = family_layer(
            [run for run in runs if int(run["initial_k"]) in primary_ks],
            jaccard_threshold=jaccard_threshold,
            overlap_threshold=overlap_threshold,
        )
        family_threshold_rows.append(
            {
                "jaccard_threshold": jaccard_threshold,
                "overlap_threshold": overlap_threshold,
                "relation_component_count": len(threshold_layer["components"]),
                "recurrent_relation_component_count": sum(
                    item["is_recurrent_relation_component"]
                    for item in threshold_layer["components"]
                ),
                "cohesive_family_count": len(threshold_layer["families"]),
                "cohesive_family_sizes": json.dumps(
                    sorted(
                        (len(item["member_ids"]) for item in threshold_layer["families"]),
                        reverse=True,
                    )
                ),
            }
        )
    pairwise_rows = []
    for left, right in combinations(runs, 2):
        pairwise_rows.append(
            {
                "left_run": left["run_id"],
                "right_run": right["run_id"],
                "same_initial_k": left["initial_k"] == right["initial_k"],
                **compare_runs(left["assignments"], right["assignments"]),
            }
        )

    primary_k_matrices = k_matrices["primary"]
    if not primary_k_matrices:
        summary = {
            "experiment": "multi_k_accepted_core_stability",
            "analysis_status": "primary_unavailable",
            "expected_run_count": len(initial_ks) * len(repeats),
            "valid_run_count": len(runs),
            "k_level_summary": k_levels,
            "interpretation": "No K level has enough valid repeats for primary consensus analysis.",
        }
        write_json(experiment_root / "summary.json", summary)
        return summary
    joint, coacceptance, conditional, acceptance = combine_k_matrices(primary_k_matrices)
    valid_k_count = len(primary_k_matrices)
    exploratory_k_count = len(k_matrices["exploratory"])
    strict_k_count = len(k_matrices["strict"])
    strict_cores = []
    if k_matrices["strict"]:
        strict_joint, _, _, strict_acceptance = combine_k_matrices(k_matrices["strict"])
        strict_cores = extract_cores(
            strict_joint,
            strict_acceptance,
            patient_ids,
            threshold=2 / 3,
            min_size=min_core_size,
        )
    patient_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    threshold_rows, cores_by_threshold = [], {}
    for threshold in THRESHOLDS:
        cores = extract_cores(
            joint,
            acceptance,
            patient_ids,
            threshold=threshold,
            min_size=min_core_size,
        )
        cores_by_threshold[threshold] = cores
        threshold_rows.append(
            {
                "threshold": threshold,
                "core_count": len(cores),
                "core_patient_count": sum(len(core["member_ids"]) for core in cores),
                "core_sizes": json.dumps([len(core["member_ids"]) for core in cores]),
            }
        )

    primary_threshold = 2 / 3
    primary_cores = cores_by_threshold[primary_threshold]
    core_rows, membership_rows = [], []
    for core_number, core in enumerate(primary_cores, 1):
        core_id = f"CORE{core_number:02d}"
        members = core["member_ids"]
        indices = [patient_index[member] for member in members]
        within_joint = joint[np.ix_(indices, indices)]
        within_conditional = conditional[np.ix_(indices, indices)]
        within_coacceptance = coacceptance[np.ix_(indices, indices)]
        pair_indices = np.triu_indices(len(indices), k=1)
        joint_values = within_joint[pair_indices]
        within_values = within_conditional[pair_indices]
        coacceptance_values = within_coacceptance[pair_indices]
        outside_indices = [index for index in range(len(patient_ids)) if index not in indices]
        outside_joint = joint[np.ix_(indices, outside_indices)].ravel()
        outside_values = conditional[np.ix_(indices, outside_indices)].ravel()
        recurrence = core_recurrence(members, primary_runs)
        core_rows.append(
            {
                "core_id": core_id,
                "core_size": len(members),
                "recurrence_denominator": "primary_eligible_runs",
                **{
                    **recurrence,
                    "recovered_runs_by_k": json.dumps(
                        recurrence["recovered_runs_by_k"], ensure_ascii=False
                    ),
                },
                "mean_within_joint_coassignment": float(joint_values.mean()),
                "min_within_joint_coassignment": float(joint_values.min()),
                "mean_within_conditional_coassignment": float(within_values.mean()),
                "min_within_conditional_coassignment": float(within_values.min()),
                "mean_within_coacceptance": float(coacceptance_values.mean()),
                "min_within_coacceptance": float(coacceptance_values.min()),
                "mean_outside_conditional_coassignment": float(outside_values.mean()) if outside_values.size else 0.0,
                "max_outside_conditional_coassignment": float(outside_values.max()) if outside_values.size else 0.0,
                "mean_outside_joint_coassignment": float(outside_joint.mean()) if outside_joint.size else 0.0,
                "max_outside_joint_coassignment": float(outside_joint.max()) if outside_joint.size else 0.0,
                "mean_acceptance_frequency_across_k": float(acceptance[indices].mean()),
                "min_acceptance_frequency_across_k": float(acceptance[indices].min()),
                "member_ids": json.dumps(members, ensure_ascii=False),
            }
        )
        membership_rows.extend(
            {
                "core_id": core_id,
                "patient_id": member,
                "acceptance_frequency_across_k": float(acceptance[patient_index[member]]),
            }
            for member in members
        )

    catalog_by_id = {str(item["node_id"]): item for item in catalog}
    family_membership_rows = [
        {
            "family_id": family["family_id"],
            **{
                key: catalog_by_id[node_id][key]
                for key in ("node_id", "run_id", "initial_k", "repeat", "set_id", "member_n")
            },
            "member_ids": json.dumps(catalog_by_id[node_id]["member_ids"], ensure_ascii=False),
        }
        for family in cohesive_families
        for node_id in family["node_ids"]
    ]
    family_patient_rows = [
        {
            "family_id": family["family_id"],
            "patient_id": patient_id,
            "node_occurrence_count": count,
            "node_count": len(family["node_ids"]),
            "node_membership_fraction": count / len(family["node_ids"]),
            "family_present_k_count": sum(
                value > 0 for value in family["family_presence_count_by_k"].values()
            ),
            "family_presence_fraction_by_k": json.dumps(
                family["family_presence_fraction_by_k"], ensure_ascii=False
            ),
            "patient_membership_given_family_present_by_k": json.dumps(
                family["patient_membership_given_family_present_by_k"][patient_id],
                ensure_ascii=False,
            ),
            "patient_unconditional_membership_by_k": json.dumps(
                family["patient_unconditional_membership_by_k"][patient_id],
                ensure_ascii=False,
            ),
        }
        for family in cohesive_families
        for patient_id, count in sorted(family["member_frequency"].items())
    ]
    family_csv_rows = [
        {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            for key, value in family.items()
            if key != "node_ids"
        }
        for family in cohesive_families
    ]
    catalog_csv_rows = [
        {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            for key, value in item.items()
        }
        for item in catalog
    ]
    similarity_csv_rows = [dict(row) for row in set_similarity_rows]
    pairwise_csv_rows = [
        {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        }
        for row in pairwise_rows
    ]
    write_json(experiment_root / "run_summary.json", {"runs": run_rows})
    write_json(experiment_root / "accepted_set_catalog.json", {"sets": catalog})
    write_json(experiment_root / "accepted_set_relation_components.json", {"components": relation_components})
    write_json(experiment_root / "accepted_set_families.json", {"families": cohesive_families})
    for level, layer in family_layers.items():
        if level == "primary":
            continue
        write_json(
            experiment_root / f"{level}_accepted_set_catalog.json",
            {"sets": layer["catalog"]},
        )
        write_json(
            experiment_root / f"{level}_accepted_set_relation_components.json",
            {"components": layer["components"]},
        )
        write_json(
            experiment_root / f"{level}_accepted_set_families.json",
            {"families": layer["families"]},
        )
    for name, rows in (
        ("run_summary.csv", run_rows),
        ("pairwise_partition_consistency.csv", pairwise_csv_rows),
        ("accepted_set_catalog.csv", catalog_csv_rows),
        ("accepted_set_similarity.csv", similarity_csv_rows),
        ("accepted_set_relation_components.csv", [
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in component.items()
                if key != "node_ids"
            }
            for component in relation_components
        ]),
        ("accepted_set_families.csv", family_csv_rows),
        ("accepted_set_family_threshold_sensitivity.csv", family_threshold_rows),
        ("accepted_set_family_membership.csv", family_membership_rows),
        ("accepted_set_family_patient_frequency.csv", family_patient_rows),
        ("k_level_summary.csv", k_levels),
        (
            "patient_acceptance_frequency.csv",
            [
                {"patient_id": patient_id, "acceptance_frequency": float(acceptance[index])}
                for index, patient_id in enumerate(patient_ids)
            ],
        ),
        ("stable_core_membership.csv", membership_rows),
        ("stable_core_summary.csv", core_rows),
        ("threshold_sensitivity.csv", threshold_rows),
    ):
        fields = list(rows[0]) if rows else []
        with (experiment_root / name).open("w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    for name, matrix in (
        ("joint_accepted_coassignment_matrix.csv", joint),
        ("pairwise_coacceptance_matrix.csv", coacceptance),
        ("conditional_membership_coassignment_matrix.csv", conditional),
    ):
        with (experiment_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", *patient_ids])
            for patient_id, row in zip(patient_ids, matrix):
                writer.writerow([patient_id, *map(float, row)])

    import matplotlib.pyplot as plt

    plot_matrix = joint.copy()
    np.fill_diagonal(plot_matrix, 1)
    if len(patient_ids) > 1:
        order = leaves_list(
            linkage(squareform(1 - plot_matrix, checks=False), method="complete")
        )
    else:
        order = np.arange(len(patient_ids))
    conditional_plot = conditional.copy()
    coaccepted_k_support_count = np.sum(
        [matrix[1] > 0 for matrix in primary_k_matrices], axis=0
    )
    conditional_plot[
        coaccepted_k_support_count < MIN_CONCURRENT_ACCEPTED_RUNS_FOR_CONDITIONAL_PLOT
    ] = np.nan
    for filename, matrix, title, label in (
        (
            "coassignment_heatmap.png",
            joint,
            f"Joint accepted-set co-assignment across {valid_k_count} equally weighted K levels",
            "P(same accepted set)",
        ),
        (
            "conditional_coassignment_heatmap.png",
            conditional_plot,
            f"Conditional co-assignment diagnostic across {valid_k_count} K levels",
            "P(same set | both accepted; masked if <2 co-accepted K levels)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(10, 9))
        image = axis.imshow(matrix[np.ix_(order, order)], vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Patients (complete-linkage order)")
        axis.set_ylabel("Patients (complete-linkage order)")
        figure.colorbar(image, ax=axis, label=label)
        figure.tight_layout()
        figure.savefig(experiment_root / filename, dpi=180)
        plt.close(figure)

    same_k = [row for row in pairwise_rows if row["same_initial_k"]]
    different_k = [row for row in pairwise_rows if not row["same_initial_k"]]
    same_k_ari = [
        row["adjusted_rand_index"]
        for row in same_k
        if row["adjusted_rand_index"] is not None
    ]
    different_k_ari = [
        row["adjusted_rand_index"]
        for row in different_k
        if row["adjusted_rand_index"] is not None
    ]
    same_k_informative = [row for row in same_k if not row["both_empty"]]
    different_k_informative = [row for row in different_k if not row["both_empty"]]
    summary = {
        "experiment": "multi_k_accepted_core_stability",
        "analysis_status": "complete" if len(runs) == len(initial_ks) * len(repeats) else "partial",
        "initial_k_values": list(initial_ks),
        "review_repeats": list(repeats),
        "expected_run_count": len(initial_ks) * len(repeats),
        "valid_run_count": len(runs),
        "invalid_run_count": len(execution_rows) - len(runs),
        "valid_k_count": valid_k_count,
        "exploratory_k_count": exploratory_k_count,
        "strict_k_count": strict_k_count,
        "strict_sensitivity": {
            "core_count": len(strict_cores),
            "core_patient_count": sum(len(core["member_ids"]) for core in strict_cores),
            "cores": strict_cores,
        },
        "k_level_summary": k_levels,
        "partition_assessment_counts": dict(sorted(Counter(
            str(run.get("partition_assessment", "unavailable")) for run in runs
        ).items())),
        "patient_count": len(patient_ids),
        "accepted_only": True,
        "scientific_denominator": "K levels with at least two valid repeats are primary; valid repeats are aggregated within K, then each primary K receives equal weight",
        "accepted_set_family_definition": {
            "jaccard_threshold": family_jaccard_threshold,
            "overlap_threshold": family_overlap_threshold,
            "cross_run_only": True,
            "cohesive_family_requires_no_same_run_conflict": True,
            "cohesive_family_requires_all_pairwise_edges": True,
            "cohesive_family_requires_one_to_one_pair_relations": True,
        },
        "accepted_set_relation_level": "run_level_cross_run; K-level consensus is reported separately in k_level_summary",
        "accepted_set_relation_component_count": len(relation_components),
        "recurrent_relation_component_count": len(recurrent_families),
        "cohesive_family_count": len(cohesive_families),
        "strict_cohesive_family_count": len(family_layers["strict"]["families"]),
        "exploratory_cohesive_family_count": len(family_layers["exploratory"]["families"]),
        "primary_family_k_values": sorted(primary_ks),
        "strict_family_k_values": sorted(strict_ks),
        "exploratory_family_k_values": sorted({int(run["initial_k"]) for run in runs}),
        "family_level_denominator": {
            "primary": "K levels with at least two valid repeats",
            "strict": "K levels with all expected repeats valid",
            "exploratory": "all scientifically terminal runs",
        },
        "orphan_set_count": sum(component["is_orphan_set"] for component in relation_components),
        "accepted_set_families": cohesive_families,
        "family_threshold_sensitivity": family_threshold_rows,
        "primary_core_definition": {
            "acceptance_frequency": primary_threshold,
            "minimum_pairwise_joint_coassignment": primary_threshold,
            "conditional_coassignment": "diagnostic_only",
            "coacceptance": "denominator_reliability_diagnostic",
            "minimum_core_size": min_core_size,
        },
        "primary_core_count": len(core_rows),
        "primary_core_patient_count": len(membership_rows),
        "primary_cores": core_rows,
        "threshold_sensitivity": threshold_rows,
        "within_k_repeatability": {
            "pair_count": len(same_k),
            "informative_pair_count": len(same_k_informative),
            "mean_ari": float(np.mean(same_k_ari)) if same_k_ari else None,
            "mean_coverage_jaccard": float(np.mean([row["accepted_coverage_jaccard"] for row in same_k_informative])) if same_k_informative else None,
            "mean_matched_jaccard": float(np.mean([row["matched_mean_jaccard"] for row in same_k_informative])) if same_k_informative else None,
            "mean_matched_dice": float(np.mean([row["matched_mean_dice"] for row in same_k_informative])) if same_k_informative else None,
        },
        "between_k_sensitivity": {
            "pair_count": len(different_k),
            "informative_pair_count": len(different_k_informative),
            "mean_ari": float(np.mean(different_k_ari)) if different_k_ari else None,
            "mean_coverage_jaccard": float(np.mean([row["accepted_coverage_jaccard"] for row in different_k_informative])) if different_k_informative else None,
            "mean_matched_jaccard": float(np.mean([row["matched_mean_jaccard"] for row in different_k_informative])) if different_k_informative else None,
            "mean_matched_dice": float(np.mean([row["matched_mean_dice"] for row in different_k_informative])) if different_k_informative else None,
        },
        "interpretation": (
            "Recurrent accepted cores indicate cross-K structural repeatability, "
            "not biological subtype validation."
        ),
    }
    write_json(experiment_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--initial-k", type=int, choices=INITIAL_KS, action="append")
    parser.add_argument("--repeat", type=int, choices=REPEATS, action="append")
    parser.add_argument("--min-core-size", type=int, default=5)
    parser.add_argument("--family-jaccard-threshold", type=float, default=FAMILY_JACCARD_THRESHOLD)
    parser.add_argument("--family-overlap-threshold", type=float, default=FAMILY_OVERLAP_THRESHOLD)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    initial_ks = sorted(set(args.initial_k or INITIAL_KS))
    repeats = sorted(set(args.repeat or REPEATS))
    patient_ids = sorted(load_affinity_patient_ids(args.data_root))
    patient_states = load_patient_states(args.data_root)
    review_files = [
        *sorted((ROOT / "agents" / "subtype_review").rglob("*.py")),
        *sorted((ROOT / "agents" / "subtype_review").rglob("*.md")),
        args.config_dir / "subtype_review.yaml",
        *[
            ROOT / "tools" / name
            for name in (
                "cnv_characterization.py", "confound.py", "known_label_echo.py",
                "multimodal_consistency_check.py", "mutation_enrichment.py",
                "pathway_enrichment.py", "cross_modal_structure.py",
            )
        ],
    ]
    review_signature = hashlib.sha256(
        b"".join(path.read_bytes() for path in review_files)
    ).hexdigest()
    source_paths = {
        initial_k: (
            args.data_root
            / "candidate_subtype"
            / "consensus_cluster"
            / f"consensus_hierarchical_K{initial_k}.json"
        )
        for initial_k in initial_ks
    }
    source_sha256 = (
        {initial_k: sha256_file(path) for initial_k, path in source_paths.items()}
        if not args.analyze_only
        else {}
    )
    input_signature = (
        input_data_signature(
            [
                args.data_root / "storage" / "patient_states" / "patient_states.jsonl",
                args.data_root / "candidate_subtype" / "affinity_patient_order.json",
                *(
                    path
                    for directory in ("candidate_subtype", "wxs", "cnv")
                    for path in sorted((args.data_root / directory).rglob("*"))
                    if path.is_file()
                ),
                *source_paths.values(),
            ]
        )
        if not args.analyze_only
        else ""
    )

    if not args.analyze_only:
        for repeat in repeats:
            for initial_k in initial_ks:
                run_root = args.experiment_root / f"run{repeat}" / f"K{initial_k}"
                source_path = source_paths[initial_k]
                metadata_path = run_root / "run_metadata.json"
                summary_path = run_root / "final_review_summary.json"
                reusable = False
                if metadata_path.exists() and summary_path.exists() and not args.force:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    reusable = (
                        metadata.get("initial_k") == initial_k
                        and metadata.get("repeat") == repeat
                        and metadata.get("source") == str(source_path)
                        and metadata.get("source_sha256") == source_sha256[initial_k]
                        and metadata.get("input_data_signature") == input_signature
                        and metadata.get("review_signature") == review_signature
                        and scientifically_terminal(summary)
                    )
                if reusable:
                    print(f"[reuse] run{repeat}/K{initial_k}")
                    continue
                if run_root.exists():
                    shutil.rmtree(run_root)
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                initial_sets = labels_to_candidate_sets(
                    payload,
                    initial_k,
                    expected_patient_count=len(patient_ids),
                    expected_patient_ids=set(patient_ids),
                )
                run_root.mkdir(parents=True, exist_ok=True)
                write_json(
                    run_root / "initial_partition.json",
                    {
                        "initial_k": initial_k,
                        "repeat": repeat,
                        "source": str(source_path),
                        "source_sha256": source_sha256[initial_k],
                        "input_data_signature": input_signature,
                        "review_signature": review_signature,
                        "patient_count": len(patient_ids),
                        "candidate_sets": initial_sets,
                    },
                )
                state = run_subtype_review(
                    initial_sets,
                    patient_states,
                    str(args.data_root),
                    str(args.config_dir),
                    artifact_root=str(run_root),
                )
                summary = save_review_outputs(state, str(run_root), direct=True)
                write_json(
                    metadata_path,
                    {
                        "experiment": "multi_k_accepted_core_stability",
                        "initial_k": initial_k,
                        "repeat": repeat,
                        "source": str(source_path),
                        "source_sha256": source_sha256[initial_k],
                        "input_data_signature": input_signature,
                        "review_signature": review_signature,
                        "patient_count": len(patient_ids),
                        "status": summary["status"],
                        "rounds_used": summary["rounds_used"],
                        "llm_usage": summary["llm_usage"],
                    },
                )
                print(f"[complete] run{repeat}/K{initial_k}: {summary['status']}")

    summary = analyze(
        args.experiment_root,
        patient_ids,
        initial_ks,
        repeats,
        args.min_core_size,
        args.family_jaccard_threshold,
        args.family_overlap_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
