from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import SpectralClustering

from tools.subtype_review_common import tool_parameters, tool_result
from tools.tool_multimodal_consistency_check import (
    modality_affinity_path,
    normalize_affinity,
    partition_separation,
    round_value,
    separation_score,
)
from utils.llm_utils import load_yaml_file


MODALITIES = ("ct", "wsi", "rna", "genomic", "fused")


def local_split_plans(
    memberships: Mapping[str, list[str]],
    affinities: Mapping[str, np.ndarray],
    case_ids: list[str],
    *,
    min_size: int,
    max_children: int,
) -> list[dict[str, Any]]:
    index_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
    candidates = {}
    for set_id, members in sorted(memberships.items()):
        positions = np.asarray([index_by_case[case_id] for case_id in members])
        maximum = min(max_children, len(members) // min_size)
        for child_count in range(2, maximum + 1):
            similarity = normalize_affinity(
                affinities["fused"][np.ix_(positions, positions)]
            )
            labels = SpectralClustering(
                n_clusters=child_count,
                affinity="precomputed",
                assign_labels="cluster_qr",
                random_state=0,
            ).fit_predict(similarity)
            groups = tuple(
                sorted(
                    tuple(
                        sorted(
                            members[index]
                            for index in np.flatnonzero(labels == label)
                        )
                    )
                    for label in np.unique(labels)
                )
            )
            if min(map(len, groups), default=0) < min_size:
                continue
            candidates[(set_id, groups)] = {
                "source_set_id": set_id,
                "child_count": child_count,
                "child_sizes": [len(group) for group in groups],
                "groups": [list(group) for group in groups],
                "source_networks": ["fused"],
            }

    plans = []
    counters = {}
    for plan in sorted(
        candidates.values(),
        key=lambda item: (
            item["source_set_id"],
            item["child_count"],
            -len(item["source_networks"]),
            item["groups"],
        ),
    ):
        key = (plan["source_set_id"], plan["child_count"])
        counters[key] = counters.get(key, 0) + 1
        plan["plan_id"] = (
            f"split:{plan['source_set_id']}:k{plan['child_count']}:"
            f"p{counters[key]}"
        )
        plans.append(plan)
    return plans


def split_plan_evidence(
    plan: Mapping[str, Any],
    affinities: Mapping[str, np.ndarray],
    case_ids: list[str],
) -> dict[str, Any]:
    group_by_case = {
        str(case_id): f"G{index}"
        for index, group in enumerate(list(plan.get("groups", []) or []), 1)
        for case_id in group
    }
    local_case_ids = [case_id for case_id in case_ids if case_id in group_by_case]
    positions = [case_ids.index(case_id) for case_id in local_case_ids]
    labels = np.asarray([group_by_case[case_id] for case_id in local_case_ids])
    if len(local_case_ids) != sum(
        len(group) for group in list(plan.get("groups", []) or [])
    ):
        raise ValueError(f"Split plan has patients outside affinity order: {plan}")

    similarity = normalize_affinity(
        np.asarray(affinities["fused"])[np.ix_(positions, positions)]
    )
    global_row, per_set = partition_separation(similarity, labels)
    children = [
        {
            "child_id": child_id,
            **row,
        }
        for child_id, row in per_set.items()
    ]
    rows = {
        "fused": {
            "normalized_internal_separation": global_row[
                "normalized_affinity_separation"
            ],
            "minimum_child_separation": min(
                float(row["normalized_affinity_separation"])
                for row in per_set.values()
            ),
            "children": children,
        }
    }

    return {
        "plan_id": plan.get("plan_id"),
        "source_set_id": plan.get("source_set_id"),
        "child_count": plan.get("child_count"),
        "child_sizes": list(plan.get("child_sizes", []) or []),
        "groups": [list(group) for group in list(plan.get("groups", []) or [])],
        "source_networks": list(plan.get("source_networks", []) or []),
        "internal_community": rows,
    }


def boundary_values(
    similarity: np.ndarray,
    set_indices: list[np.ndarray],
) -> tuple[float, float, float]:
    within_values = []
    between_values = []
    for indices in set_indices:
        block = similarity[np.ix_(indices, indices)]
        distinct_patients = indices[:, None] != indices[None, :]
        within_values.extend(block[distinct_patients].tolist())
    for left, right in combinations(set_indices, 2):
        between_values.extend(similarity[np.ix_(left, right)].ravel().tolist())
    within = float(np.mean(within_values)) if within_values else float("nan")
    between = float(np.mean(between_values))
    return within, between, separation_score(within, between)


def boundary_bootstrap_interval(
    similarity: np.ndarray,
    set_indices: list[np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        sampled = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in set_indices
        ]
        score = boundary_values(similarity, sampled)[2]
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        raise ValueError("Boundary bootstrap produced no valid resamples")
    return tuple(float(value) for value in np.quantile(scores, [0.025, 0.975]))


def merge_group_evidence(
    set_ids: list[str],
    memberships: Mapping[str, list[str]],
    affinities: Mapping[str, np.ndarray],
    case_ids: list[str],
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    index_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
    set_indices = [
        np.asarray([index_by_case[case_id] for case_id in memberships[set_id]])
        for set_id in set_ids
    ]
    rows = {}
    for index, modality in enumerate(MODALITIES):
        similarity = normalize_affinity(affinities[modality])
        within, between, separation = boundary_values(similarity, set_indices)
        lower, upper = boundary_bootstrap_interval(
            similarity,
            set_indices,
            iterations=bootstrap_iterations,
            seed=6100 + index,
        )
        rows[modality] = {
            "mean_within_affinity": round_value(within),
            "mean_between_affinity": round_value(between),
            "normalized_boundary_separation": round_value(separation),
            "bootstrap_ci95": [round_value(lower), round_value(upper)],
        }
    return {
        "plan_id": "merge:" + "+".join(sorted(set_ids)),
        "set_ids": sorted(set_ids),
        "set_sizes": [len(memberships[set_id]) for set_id in sorted(set_ids)],
        "current_boundaries": rows,
    }


def tool_structural_adequacy(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    memberships = {
        str(state.get("set_id") or state.get("cluster_id")): sorted(
            str(case_id) for case_id in list(state.get("member_ids", []) or [])
        )
        for state in list(all_cluster_states or [cluster_state])
    }
    candidate_dir = Path(output_root) / "candidate_subtype"
    case_ids = json.loads(
        (candidate_dir / "affinity_patient_order.json").read_text(encoding="utf-8")
    )
    active = {case_id for members in memberships.values() for case_id in members}
    positions = [index for index, case_id in enumerate(case_ids) if case_id in active]
    case_ids = [case_ids[index] for index in positions]
    affinities = {
        modality: np.load(modality_affinity_path(output_root, modality))[
            np.ix_(positions, positions)
        ]
        for modality in MODALITIES
        if modality != "fused"
    }
    affinities["fused"] = np.load(candidate_dir / "fused_similarity.npy")[
        np.ix_(positions, positions)
    ]
    review_config = load_yaml_file(Path(config_dir) / "subtype_review.yaml")
    budget = dict(review_config["budget"])
    parameters = tool_parameters(config_dir, "structural_adequacy")
    split_plans = local_split_plans(
        memberships,
        affinities,
        case_ids,
        min_size=int(budget["min_split_size"]),
        max_children=int(budget["max_split_children"]),
    )
    split_evidence = [
        split_plan_evidence(
            plan,
            affinities,
            case_ids,
        )
        for plan in split_plans
    ]
    max_split_plans = int(parameters["max_split_plans_per_set"])
    selected_split_evidence = []
    for set_id in sorted(memberships):
        options = [
            row for row in split_evidence if row["source_set_id"] == set_id
        ]
        options.sort(
            key=lambda row: (
                int(row.get("child_count", 0) or 0),
                -float(
                    dict(row["internal_community"].get("fused", {}) or {}).get(
                        "minimum_child_separation", 0.0
                    )
                    or 0.0
                ),
                -float(
                    dict(row["internal_community"].get("fused", {}) or {}).get(
                        "normalized_internal_separation", 0.0
                    )
                    or 0.0
                ),
            )
        )
        selected_split_evidence.extend(options[:max_split_plans])
    split_evidence = selected_split_evidence
    bootstrap_iterations = int(parameters["bootstrap_iterations"])
    merge_evidence = [
        merge_group_evidence(
            list(set_ids),
            memberships,
            affinities,
            case_ids,
            bootstrap_iterations=bootstrap_iterations,
        )
        for set_ids in combinations(sorted(memberships), 2)
    ]
    merge_evidence.sort(
        key=lambda row: (
            abs(
                float(
                    row["current_boundaries"]["fused"][
                        "normalized_boundary_separation"
                    ]
                    or 0.0
                )
            ),
        )
    )
    metrics = {
        "split_candidates": split_evidence,
        "merge_candidates": merge_evidence,
        "analysis_scope": (
            "Split candidates use only each current set's fused affinity "
            "submatrix. Single-modality support is evaluated only after a "
            "split through a new on-demand review. Merge evidence covers every "
            "current set pair; the LLM may compose a multi-set Merge only when "
            "evidence exists for every pair. Candidate-Proposer partitions "
            "and alternative cluster counts are not used."
        ),
        "decision_scope": (
            "The tool reports measurements and exact executable plans only. "
            "It does not qualify or reject any scientific action."
        ),
    }
    return tool_result(
        tool_name="tool_structural_adequacy",
        status="success",
        cluster_id="GLOBAL",
        output_root=output_root,
        summary="Current subtype structure was evaluated for supported split and merge candidates.",
        metrics=metrics,
        decision_metrics=metrics,
        support_level="informational",
        concern_level="none",
        figures={},
    )
