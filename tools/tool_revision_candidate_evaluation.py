from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from agents.subtype_review.graph import build_global_revision_candidates
from scripts_2026_7_20 import (
    experiment_03_four_dimension_metric_review as metric_review,
)
from scripts_2026_7_20 import (
    experiment_05_five_evidence_two_scope_review as five_evidence,
)
from scripts_2026_7_20 import (
    experiment_06_strict_crossmodal_consistency as strict_crossmodal,
)
from utils.llm_utils import load_yaml_file


SCREENING_PERMUTATIONS = 199
ACTIVE_SOURCE_OUTPUT_ROOT: Path | None = None


def resolve_source_output_root(output_root: str | Path) -> Path:
    return (
        Path(ACTIVE_SOURCE_OUTPUT_ROOT)
        if ACTIVE_SOURCE_OUTPUT_ROOT is not None
        else Path(output_root)
    )


def candidate_memberships(
    all_cluster_states: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    return {
        str(state.get("set_id", state.get("cluster_id", ""))): sorted(
            str(case_id) for case_id in list(state.get("member_ids", []) or [])
        )
        for state in list(all_cluster_states or [])
    }


def select_representative_split_plans(
    plans: list[dict[str, Any]], max_per_set: int = 2
) -> list[dict[str, Any]]:
    best_by_resolution: dict[tuple[str, int], dict[str, Any]] = {}
    for plan in plans:
        key = (str(plan["source_set_id"]), int(plan["child_count"]))
        current = best_by_resolution.get(key)
        rank = (
            int(plan.get("support_count", 0) or 0),
            float(plan.get("support_fraction", 0.0) or 0.0),
        )
        current_rank = (
            int(current.get("support_count", 0) or 0),
            float(current.get("support_fraction", 0.0) or 0.0),
        ) if current else (-1, -1.0)
        if rank > current_rank:
            best_by_resolution[key] = plan

    selected = []
    source_ids = sorted({key[0] for key in best_by_resolution})
    for source_id in source_ids:
        options = [
            plan
            for (plan_source, _), plan in best_by_resolution.items()
            if plan_source == source_id
        ]
        options.sort(
            key=lambda plan: (
                -int(plan.get("support_count", 0) or 0),
                -float(plan.get("support_fraction", 0.0) or 0.0),
                int(plan.get("child_count", 0) or 0),
            )
        )
        selected.extend(options[:max_per_set])
    return selected


def split_memberships(
    memberships: Mapping[str, list[str]], plan: Mapping[str, Any]
) -> dict[str, list[str]]:
    source_id = str(plan["source_set_id"])
    proposed = {
        set_id: list(members)
        for set_id, members in memberships.items()
        if set_id != source_id
    }
    for index, group in enumerate(list(plan["groups"]), start=1):
        proposed[f"{source_id}_S{index}"] = sorted(str(case_id) for case_id in group)
    return dict(sorted(proposed.items()))


def merge_memberships(
    memberships: Mapping[str, list[str]], set_ids: list[str]
) -> dict[str, list[str]]:
    merged_ids = sorted(str(set_id) for set_id in set_ids)
    merged_name = "+".join(merged_ids)
    proposed = {
        set_id: list(members)
        for set_id, members in memberships.items()
        if set_id not in merged_ids
    }
    proposed[merged_name] = sorted(
        case_id for set_id in merged_ids for case_id in memberships[set_id]
    )
    return dict(sorted(proposed.items()))


def select_merge_plans(
    memberships: Mapping[str, list[str]],
    fused_similarity: np.ndarray,
    case_ids: list[str],
    max_plans: int = 3,
) -> list[dict[str, Any]]:
    if len(memberships) <= 2:
        return []
    index_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
    plans = []
    for left, right in combinations(sorted(memberships), 2):
        left_indices = [index_by_case[case_id] for case_id in memberships[left]]
        right_indices = [index_by_case[case_id] for case_id in memberships[right]]
        similarity = float(
            np.mean(fused_similarity[np.ix_(left_indices, right_indices)])
        )
        plans.append(
            {
                "plan_id": f"merge:{left}+{right}",
                "set_ids": [left, right],
                "between_set_similarity": round(similarity, 6),
            }
        )
    plans.sort(key=lambda plan: -float(plan["between_set_similarity"]))
    return plans[:max_plans]


def cluster_states(memberships: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {"cluster_id": set_id, "set_id": set_id, "member_ids": list(members)}
        for set_id, members in memberships.items()
    ]


def summarize_identifiability(evidence: Mapping[str, Any]) -> dict[str, Any]:
    global_row = dict(evidence.get("global", {}) or {})
    return {
        "status": global_row.get("status"),
        "macro_f1": global_row.get("macro_f1"),
        "minimum_set_recall": global_row.get("minimum_set_recall"),
        "boundary_rate": global_row.get("boundary_rate"),
        "per_set": {
            set_id: {
                key: row.get(key)
                for key in ("status", "recall", "f1", "boundary_fraction")
            }
            for set_id, row in dict(evidence.get("candidate_sets", {}) or {}).items()
        },
    }


def summarize_rna_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [
        row for row in rows if five_evidence.biological_row_supported(row)
    ]
    by_set: dict[str, list[dict[str, Any]]] = {}
    for row in supported:
        by_set.setdefault(str(row.get("candidate_set_id", "")), []).append(row)
    return {
        "supported_set_count": len(by_set),
        "per_set": {
            set_id: {
                "supported_pathway_count": len(set_rows),
                "top_pathways": [
                    {
                        "pathway": row.get("pathway"),
                        "q_value": row.get("q_value"),
                        "standardized_mean_difference": row.get(
                            "standardized_mean_difference"
                        ),
                    }
                    for row in sorted(
                        set_rows,
                        key=lambda item: (
                            float(item.get("q_value", 1.0) or 1.0),
                            -abs(
                                float(
                                    item.get(
                                        "standardized_mean_difference", 0.0
                                    )
                                    or 0.0
                                )
                            ),
                        ),
                    )[:3]
                ],
            }
            for set_id, set_rows in sorted(by_set.items())
        },
    }


def summarize_confounder(metrics: Mapping[str, Any]) -> dict[str, Any]:
    associations = dict(metrics.get("confounder_global_association", {}) or {})
    adjusted = dict(metrics.get("adjusted_modality_partial_r2", {}) or {})
    return {
        "confounding_non_identifiable": list(
            metrics.get("confounding_non_identifiable", []) or []
        ),
        "strong_associations": five_evidence.strong_confounders(associations),
        "adjusted_modalities": {
            modality: {
                key: row.get(key)
                for key in ("status", "partial_r2", "q_value")
                if row.get(key) is not None
            }
            for modality, row in adjusted.items()
        },
    }


def evaluate_structure(
    memberships: Mapping[str, list[str]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    fused_similarity: np.ndarray,
    case_ids: list[str],
    output_root: Path,
    config_dir: str,
) -> dict[str, Any]:
    states = cluster_states(memberships)
    identifiability = five_evidence.cross_fitted_membership_evidence(
        fused_similarity,
        case_ids,
        memberships,
    )
    pathway_result = metric_review.tool_heldout_rna_pathways(
        {"cluster_id": "GLOBAL"},
        dict(patient_states_by_id),
        str(output_root),
        config_dir=config_dir,
        all_cluster_states=states,
    )
    pathway_rows = list(
        pathway_result["results"]["metrics"].get("rna_pathway_enrichment", [])
        or []
    )

    original_summary = five_evidence.summarize_multimodal_rows
    original_permutations = metric_review.ACTIVE_PERMUTATIONS
    five_evidence.summarize_multimodal_rows = (
        strict_crossmodal.strict_crossmodal_summary
    )
    metric_review.ACTIVE_PERMUTATIONS = SCREENING_PERMUTATIONS
    try:
        multimodal = five_evidence.compute_multimodal_support(
            patient_states_by_id,
            memberships,
            permutations=SCREENING_PERMUTATIONS,
        )
        confound_result = metric_review.tool_technical_confound_safety(
            {"cluster_id": "GLOBAL"},
            dict(patient_states_by_id),
            str(output_root),
            config_dir=config_dir,
            all_cluster_states=states,
        )
    finally:
        five_evidence.summarize_multimodal_rows = original_summary
        metric_review.ACTIVE_PERMUTATIONS = original_permutations

    return {
        "set_count": len(memberships),
        "set_sizes": {
            set_id: len(members) for set_id, members in memberships.items()
        },
        "identifiability": summarize_identifiability(identifiability),
        "heldout_rna_biology": summarize_rna_support(pathway_rows),
        "multimodal_support": multimodal,
        "confounder_exclusion": summarize_confounder(
            confound_result["results"]["metrics"]
        ),
    }


def identifiability_deltas(
    current: Mapping[str, Any], proposed: Mapping[str, Any]
) -> dict[str, float | None]:
    current_ident = dict(current.get("identifiability", {}) or {})
    proposed_ident = dict(proposed.get("identifiability", {}) or {})
    deltas = {}
    for key in ("macro_f1", "minimum_set_recall", "boundary_rate"):
        before = current_ident.get(key)
        after = proposed_ident.get(key)
        deltas[key] = (
            round(float(after) - float(before), 6)
            if before is not None and after is not None
            else None
        )
    return deltas


def tool_revision_candidate_evaluation(
    cluster_state: dict[str, Any],
    patient_states_by_id: dict[str, Any],
    output_root: str,
    config_dir: str = "",
    all_cluster_states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifact_output_root = Path(output_root)
    source_output_root = resolve_source_output_root(output_root)
    memberships = candidate_memberships(all_cluster_states)
    checkpoint = json.loads(
        (
            source_output_root
            / "storage"
            / "pipeline_checkpoints"
            / "evidence_ready.json"
        ).read_text(encoding="utf-8")
    )
    case_ids = [
        str(state["case_id"]) for state in checkpoint.get("patient_states", [])
    ]
    fused_similarity = np.load(source_output_root / "fused_similarity.npy")
    if sorted(case_ids) != sorted(
        case_id for members in memberships.values() for case_id in members
    ):
        raise ValueError("Current candidate sets do not cover the fused cohort")

    review_config = load_yaml_file(
        Path(config_dir or "configs") / "subtype_review.yaml"
    ) or {}
    revision_budget = dict(review_config["budget"])
    candidates = build_global_revision_candidates(
        cluster_states(memberships),
        {},
        source_output_root,
        revision_budget,
    )
    split_plans = select_representative_split_plans(candidates["split"])
    merge_plans = select_merge_plans(memberships, fused_similarity, case_ids)
    artifact_root = artifact_output_root / "subtype_review" / "GLOBAL"
    screening_root = artifact_root / "revision_candidate_screening"
    current = evaluate_structure(
        memberships,
        patient_states_by_id,
        fused_similarity,
        case_ids,
        screening_root / "current",
        config_dir,
    )

    split_results = []
    for plan in split_plans:
        proposed = evaluate_structure(
            split_memberships(memberships, plan),
            patient_states_by_id,
            fused_similarity,
            case_ids,
            screening_root / str(plan["plan_id"]).replace(":", "_"),
            config_dir,
        )
        split_results.append(
            {
                key: plan.get(key)
                for key in (
                    "plan_id",
                    "source_set_id",
                    "child_count",
                    "child_sizes",
                    "support_count",
                    "support_fraction",
                )
            }
            | {
                "proposed_structure": proposed,
                "identifiability_delta_vs_current": identifiability_deltas(
                    current, proposed
                ),
            }
        )

    merge_results = []
    for plan in merge_plans:
        proposed = evaluate_structure(
            merge_memberships(memberships, list(plan["set_ids"])),
            patient_states_by_id,
            fused_similarity,
            case_ids,
            screening_root / str(plan["plan_id"]).replace(":", "_"),
            config_dir,
        )
        merge_results.append(
            dict(plan)
            | {
                "proposed_structure": proposed,
                "identifiability_delta_vs_current": identifiability_deltas(
                    current, proposed
                ),
            }
        )

    metrics = {
        "current_structure": current,
        "split_candidates": split_results,
        "merge_candidates": merge_results,
        "screening_permutations": SCREENING_PERMUTATIONS,
        "analysis_scope": (
            "Counterfactual internal screening of current versus selected "
            "revision structures. Any selected revision must undergo the "
            "standard five-evidence review again."
        ),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / "revision_candidate_evaluation.json"
    artifact_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "tool_name": "tool_revision_candidate_evaluation",
        "status": "success",
        "cluster_id": "GLOBAL",
        "results": {
            "summary": (
                "Current structure was compared with representative split and "
                "merge structures before an action decision."
            ),
            "metrics": {"revision_candidate_evaluation": metrics},
            "warnings": [
                "This is internal screening, not external validation.",
                "Historical split support is not evidence of biological validity.",
            ],
            "missing_reason": "",
        },
        "artifacts": {"tables": {"evaluation": str(artifact_path)}},
        "errors": [],
    }
