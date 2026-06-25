from __future__ import annotations

from typing import TypedDict

from utils.tool_utils import JSONValue


class SubtypeReviewRecord(TypedDict, total=False):
    cluster_id: str
    member_ids: list[str]
    status: str
    source_views: list[str]
    generator: dict[str, JSONValue]
    consensus: dict[str, JSONValue]
    verification_vector: dict[str, JSONValue]
    structured_evidence: list[dict[str, JSONValue]]
    llm_audit: dict[str, JSONValue]
    missing_evidence: list[str]
    tool_budget: dict[str, JSONValue]
    budget_state: dict[str, JSONValue]
    action_history: list[dict[str, JSONValue]]
    validation_results: dict[str, JSONValue]
    report_draft: dict[str, JSONValue]
    final_decision: str
    review_round: int
    next_action: str
    recommended_tools: list[dict[str, JSONValue]]
    split_plan: dict[str, JSONValue] | None
    merge_plan: dict[str, JSONValue] | None
    limitations: list[str]
    artifacts: dict[str, JSONValue]
