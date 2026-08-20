from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import SpectralClustering

from tools.subtype_review_common import assign_groupwise_fdr, tool_parameters, tool_result
from tools.multimodal_consistency_check import (
    modality_affinity_path,
    normalize_affinity,
    partition_separation,
    round_value,
    separation_score,
)
from utils.llm_utils import load_yaml_file


MODALITIES = ("ct", "wsi", "rna", "genomic", "fused")


def split_plan_seed(plan: Mapping[str, Any]) -> int:
    groups = sorted(
        sorted(str(case_id) for case_id in group)
        for group in list(plan.get("groups", []) or [])
    )
    signature = json.dumps(groups, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return 7100 + int(digest[:8], 16) % 1_000_000


def merge_bootstrap_seed(
    memberships: Mapping[str, list[str]], set_ids: list[str], modality: str
) -> int:
    groups = sorted(
        sorted(str(case_id) for case_id in memberships[set_id])
        for set_id in set_ids
    )
    signature = json.dumps(
        {"groups": groups, "modality": modality},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return 6100 + int(digest[:8], 16) % 1_000_000


def apply_split_modality_fdr(split_evidence: list[dict[str, Any]]) -> None:
    for modality in MODALITIES:
        if modality == "fused":
            continue
        rows = [
            {
                "group": str(row["source_set_id"]),
                "p": row["modality_community"][modality]["permutation_p_value"],
                "metrics": row["modality_community"][modality],
            }
            for row in split_evidence
        ]
        assign_groupwise_fdr(rows, "group", "p", "q")
        for item in rows:
            metrics = item["metrics"]
            metrics["q_value"] = round_value(item["q"])


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
    *,
    permutations: int,
    seed: int,
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

    rows = {}
    fused_similarity = None
    for modality in MODALITIES:
        similarity = normalize_affinity(
            np.asarray(affinities[modality])[np.ix_(positions, positions)]
        )
        global_row, per_set = partition_separation(similarity, labels)
        rows[modality] = {
            "normalized_separation": global_row["normalized_affinity_separation"],
            "minimum_child_separation": min(
                float(row["normalized_affinity_separation"])
                for row in per_set.values()
            ),
            **fixed_partition_null_calibration(
                similarity,
                labels,
                permutations=permutations,
                seed=seed + 1000 + MODALITIES.index(modality),
            ),
        }
        if modality == "fused":
            fused_similarity = similarity

    return {
        "plan_id": plan.get("plan_id"),
        "source_set_id": plan.get("source_set_id"),
        "child_count": plan.get("child_count"),
        "child_sizes": list(plan.get("child_sizes", []) or []),
        "groups": [list(group) for group in list(plan.get("groups", []) or [])],
        "source_networks": list(plan.get("source_networks", []) or []),
        "metric_direction": "larger_positive_separation_supports_the_proposed_split",
        "selection_adjusted_null": split_null_calibration(
            fused_similarity,
            labels,
            permutations=permutations,
            seed=seed,
        ),
        "modality_community": rows,
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


def split_null_calibration(
    similarity: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    matrix = normalize_affinity(similarity)
    labels = np.asarray(labels)
    child_count = len(np.unique(labels))
    indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    observed = boundary_values(matrix, indices)[2]
    upper = np.triu_indices(len(matrix), 1)
    weights = matrix[upper].copy()
    rng = np.random.default_rng(seed)
    null_scores = []
    for iteration in range(permutations):
        shuffled = weights.copy()
        rng.shuffle(shuffled)
        null_matrix = np.eye(len(matrix), dtype=float)
        null_matrix[upper] = shuffled
        null_matrix[(upper[1], upper[0])] = shuffled
        null_labels = SpectralClustering(
            n_clusters=child_count,
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=seed + iteration,
        ).fit_predict(null_matrix)
        null_indices = [
            np.flatnonzero(null_labels == label)
            for label in np.unique(null_labels)
        ]
        null_scores.append(boundary_values(null_matrix, null_indices)[2])
    null_values = np.asarray(null_scores, dtype=float)
    null_mean = float(null_values.mean())
    lower, upper_bound = np.quantile(null_values, [0.025, 0.975])
    p_value = (1 + int(np.sum(null_values >= observed))) / (permutations + 1)
    return {
        "observed_separation": round_value(observed),
        "null_mean_separation": round_value(null_mean),
        "null_ci95": [round_value(lower), round_value(upper_bound)],
        "separation_gain_over_null": round_value(observed - null_mean),
        "permutation_p_value": round_value(p_value),
        "permutations": int(permutations),
    }


def fixed_partition_null_calibration(
    similarity: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    matrix = normalize_affinity(similarity)
    labels = np.asarray(labels)
    observed_indices = [
        np.flatnonzero(labels == label) for label in np.unique(labels)
    ]
    observed = boundary_values(matrix, observed_indices)[2]
    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        shuffled_indices = [
            np.flatnonzero(shuffled == label) for label in np.unique(shuffled)
        ]
        null_scores.append(boundary_values(matrix, shuffled_indices)[2])
    null_values = np.asarray(null_scores, dtype=float)
    null_mean = float(null_values.mean())
    lower, upper = np.quantile(null_values, [0.025, 0.975])
    p_value = (1 + int(np.sum(null_values >= observed))) / (permutations + 1)
    return {
        "null_mean_separation": round_value(null_mean),
        "null_ci95": [round_value(lower), round_value(upper)],
        "separation_gain_over_null": round_value(observed - null_mean),
        "permutation_p_value": round_value(p_value),
    }


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
    canonical_groups = sorted(
        (tuple(sorted(str(case_id) for case_id in memberships[set_id])), set_id)
        for set_id in set_ids
    )
    set_indices = [
        np.asarray([index_by_case[case_id] for case_id in group])
        for group, _ in canonical_groups
    ]
    rows = {}
    for modality in MODALITIES:
        similarity = normalize_affinity(affinities[modality])
        within, between, separation = boundary_values(similarity, set_indices)
        lower, upper = boundary_bootstrap_interval(
            similarity,
            set_indices,
            iterations=bootstrap_iterations,
            seed=merge_bootstrap_seed(memberships, set_ids, modality),
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


def generate_structure_proposal_metrics(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
    artifact_root=None,
):
    artifact_root = artifact_root or output_root
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
    parameters = tool_parameters(config_dir, "revision_candidates")
    split_plans = local_split_plans(
        memberships,
        affinities,
        case_ids,
        min_size=int(budget["min_split_size"]),
        max_children=int(budget["max_split_children"]),
    )
    split_permutations = int(parameters["split_null_permutations"])
    split_evidence = [
        split_plan_evidence(
            plan,
            affinities,
            case_ids,
            permutations=split_permutations,
            seed=split_plan_seed(plan),
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
                    dict(row["modality_community"].get("fused", {}) or {}).get(
                        "minimum_child_separation", 0.0
                    )
                    or 0.0
                ),
                -float(
                    dict(row["modality_community"].get("fused", {}) or {}).get(
                        "normalized_separation", 0.0
                    )
                    or 0.0
                ),
            )
        )
        selected_split_evidence.extend(options[:max_split_plans])
    split_evidence = selected_split_evidence
    for row in split_evidence:
        row["_fused_p"] = row["selection_adjusted_null"]["permutation_p_value"]
    assign_groupwise_fdr(split_evidence, "source_set_id", "_fused_p", "_fused_q")
    for row in split_evidence:
        row["selection_adjusted_null"]["q_value"] = round_value(row.pop("_fused_q"))
        row.pop("_fused_p", None)
    apply_split_modality_fdr(split_evidence)
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
        "split_evidence_semantics": (
            "Every split_candidates entry is an executable legal plan, not an "
            "endorsed Split. normalized_separation is directional: larger positive "
            "values indicate stronger child separation. selection_adjusted_null "
            "compares observed fused separation with edge-permuted networks "
            "reclustered by the same algorithm. Original modalities use a "
            "fixed-membership label-permutation null. The protocol defines how "
            "these measurements inform scientific actions."
        ),
        "analysis_scope": (
            "Split plans are generated from each current set's fused affinity "
            "submatrix, calibrated against a selection-adjusted permutation null, "
            "and described across CT, WSI, RNA, genomic and fused networks. Merge "
            "evidence covers every current set pair. Candidate-Proposer alternative "
            "cluster counts are not used."
        ),
        "decision_scope": (
            "The tool reports measurements and exact executable plans only. "
            "It does not qualify or reject any scientific action."
        ),
    }
    return tool_result(
        tool_name="revision_candidates",
        status="success",
        cluster_id=str(cluster_state.get("cluster_id", "GLOBAL")),
        output_root=artifact_root,
        summary="Current subtype structure was measured and exact legal Split and Merge plans were generated.",
        metrics=metrics,
        decision_metrics=metrics,
        support_level="informational",
        concern_level="none",
        figures={},
    )


def generate_revision_candidates(
    action,
    target_ids,
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
    artifact_root=None,
):
    artifact_root = artifact_root or output_root
    states = [dict(item) for item in list(all_cluster_states or [cluster_state])]
    memberships = {
        str(item.get("set_id") or item.get("cluster_id")): sorted(
            str(member) for member in item.get("member_ids", []) or []
        )
        for item in states
    }
    signature = json.dumps(sorted(memberships.values()), ensure_ascii=False)
    raw = generate_structure_proposal_metrics(
        dict(cluster_state),
        {str(key): dict(value) for key, value in patient_states_by_id.items()},
        output_root,
        config_dir=config_dir,
        all_cluster_states=states,
        artifact_root=artifact_root,
    )
    metrics = dict(dict(raw.get("results", {}) or {}).get("metrics", {}) or {})
    targets = sorted(str(item) for item in target_ids)
    if action == "split":
        candidates = [
            dict(item)
            for item in metrics.get("split_candidates", []) or []
            if str(item.get("source_set_id", "")) == targets[0]
        ]
        for item in candidates:
            item["proposal_id"] = str(item.get("plan_id", ""))
            item["parent_members"] = memberships.get(targets[0], [])
            item["eligible_for_review"] = (
                int(item.get("child_count", 0) or 0) == 2
                and float(dict(item.get("selection_adjusted_null", {}) or {}).get("q_value", 1) or 1) <= 0.05
                and float(dict(item.get("selection_adjusted_null", {}) or {}).get("separation_gain_over_null", 0) or 0) > 0
            )
    elif action == "merge":
        candidates = [
            dict(item)
            for item in metrics.get("merge_candidates", []) or []
            if sorted(str(value) for value in item.get("set_ids", []) or []) == targets
        ]
        for item in candidates:
            item["proposal_id"] = str(item.get("plan_id", ""))
            item["set_ids"] = targets
            item["memberships"] = [memberships[target] for target in targets]
            item["eligible_for_review"] = True
    else:
        raise ValueError(f"Unknown revision action: {action}")
    for item in candidates:
        item["partition_signature"] = signature
    return sorted(candidates, key=lambda item: str(item.get("proposal_id", "")))
