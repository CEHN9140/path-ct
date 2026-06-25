from __future__ import annotations

from typing import Any, Mapping

from tools.subtype_review_common import subtype_review_tool_definitions
from utils.cluster_state import SubtypeReviewRecord
from utils.candidate_clustering_outputs import (
    canonical_partition,
    cluster_sizes_from_labels,
    consensus_matrix_from_partitions,
    consensus_scatter_coordinates,
    consensus_silhouette_score,
    save_candidate_clustering_outputs,
    select_pac_stability_k,
)
from utils.llm_utils import call_llm_json, load_llm_client
from utils.subtype_review_contract import (
    EVIDENCE_BLOCK_NAMES,
    FINAL_ACTION_REASON_CODES,
    FINAL_REVIEW_ACTIONS,
    action_decision_contract_issues,
    build_agentic_evidence_matrix,
    build_evidence_catalog,
    build_verifier_decision,
    compute_verification_vector,
    decision_consistency_check,
    evidence_blocks_missing_metrics,
    evidence_catalog_by_id,
    metric_refs_for_blocks,
    metric_refs_from_evidence_ids,
    normalize_action_decision,
    normalize_confidence_level,
    normalize_decision_state,
    reason_codes_for_action,
)
from utils.subtype_review_runtime import (
    action_decision_messages,
    agentic_router_decision,
    allowed_review_tools,
    build_pipeline_output,
    build_structured_evidence,
    choose_router_action,
    compact_cluster_state,
    compact_tool_result,
    filtered_tool_plan,
    force_drop,
    generate_cluster_report,
    init_review_budget_state,
    llm_action_decision,
    filter_confounder_metrics_for_review,
    llm_evidence_audit,
    load_prompt,
    load_subtype_review_budget,
    load_subtype_review_config,
    merge_cluster_members,
    normalize_review_action,
    report_messages,
    review_messages,
    review_tool_schemas,
    revision_budget_exhausted,
    revision_engine,
    save_debug_artifacts,
    save_subtype_review_json,
    split_cluster_members,
    split_cluster_specs_from_consensus,
    subtype_review_artifact_policy,
    update_review_budget_after_revision,
)

INTERNAL_REVIEW_ACTIONS = FINAL_REVIEW_ACTIONS | {"call_tools", "continue_review"}
REVIEW_UNAVAILABLE_STATUS = "review_unavailable"

DEFAULT_BUDGET_STATE = {
    "review_rounds_used": 0,
    "supplement_rounds_used": 0,
    "llm_audit_calls_used": 0,
    "report_calls_used": 0,
    "tool_calls_used": 0,
    "split_plans_used": 0,
    "merge_plans_used": 0,
    "revisions_used": 0,
    "llm_parse_failures": 0,
    "agent_failures": 0,
    "tool_failures": 0,
    "no_new_evidence_rounds": 0,
    "budget_exhausted": False,
    "budget_exhausted_reason": "",
}

def tools_for_evidence_blocks(
    blocks: list[str] | tuple[str, ...],
    config_dir: str = "",
) -> list[dict[str, str]]:
    requested = {str(item) for item in list(blocks or []) if str(item)}
    if not requested:
        requested = set(EVIDENCE_BLOCK_NAMES)
    tool_definitions = subtype_review_tool_definitions(config_dir)
    tool_plan = [
        {"tool_name": tool_name}
        for tool_name, definition in tool_definitions.items()
        if requested.intersection(set(definition.get("evidence_blocks", []) or []))
    ]
    return filtered_tool_plan(tool_plan, config_dir) if config_dir else tool_plan


def initial_subtype_review_state(
    cluster: Mapping[str, Any],
) -> SubtypeReviewRecord:
    return {
        "cluster_id": str(cluster.get("cluster_id", "unknown_cluster")),
        "member_ids": [str(item) for item in list(cluster.get("member_ids", []))],
        "status": str(cluster.get("status", "under_review")),
        "parent_cluster_ids": [str(cluster.get("cluster_id", "unknown_cluster"))],
        "revision_round": 0,
        "source_views": [str(item) for item in list(cluster.get("source_views", []))],
        "generator": dict(cluster.get("generator", {}) or {}),
        "consensus": dict(cluster.get("consensus", {}) or {}),
        "evidence_gap": [],
        "tool_plan": [],
        "tool_results": [],
        "verification_vector": {},
        "structured_evidence": [],
        "llm_audit": {},
        "missing_evidence": [],
        "budget_state": dict(DEFAULT_BUDGET_STATE),
        "action_history": [],
        "validation_results": {},
        "report_draft": {},
        "final_decision": "",
        "final_action": "",
        "drop_reason": "",
        "review_round": 0,
        "artifacts": {},
        "recommended_tools": [],
        "split_plan": None,
        "merge_plan": None,
        "limitations": [],
        "revision_history": [],
        "previous_evidence": [],
    }
