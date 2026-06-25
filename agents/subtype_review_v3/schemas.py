from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

EVIDENCE_BLOCKS = (
    "set_reliability",
    "biological_support",
    "multimodal_support",
    "clinical_context",
    "known_label_echo",
    "confounder_exclusion",
)

FINAL_ACTIONS = {"accept", "drop", "split", "merge"}
ROUTER_ACTIONS = {"continue_review", "drop", "split", "merge"}
REVIEW_UNAVAILABLE = "review_unavailable"


class CompactToolResult(BaseModel):
    tool_name: str
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    missing_reason: str = ""
    artifact_paths: dict[str, Any] = Field(default_factory=dict)
    metric_refs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class VerifierReview(BaseModel):
    accept_ready: bool = False
    verification_vector: dict[str, Any] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)
    confidence_level: Literal["high", "moderate", "low", ""] = ""
    reason_codes: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""

    def contract_issues(self, tool_results: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        known_refs = tool_metric_refs(tool_results)
        if self.accept_ready:
            if self.confidence_level != "high":
                issues.append("accept_requires_high_confidence")
            if not self.reason_codes:
                issues.append("missing_reason_codes")
            if not self.metric_refs:
                issues.append("missing_metric_refs")
        for metric_ref in self.metric_refs:
            if metric_ref not in known_refs:
                issues.append(f"missing_metric_ref:{metric_ref}")
        return issues

    def is_contract_valid(self, tool_results: list[dict[str, Any]]) -> bool:
        return not self.contract_issues(tool_results)


class RouterDecision(BaseModel):
    action: Literal["continue_review", "drop", "split", "merge"] = "continue_review"
    requested_tools: list[str] = Field(default_factory=list)
    target_blocks: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    continue_review_reason: str = ""
    revision_signal: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: str = ""
    rejected_tools: list[str] = Field(default_factory=list)

    def contract_issues(self, tool_results: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        known_refs = tool_metric_refs(tool_results)
        if self.action == "continue_review":
            if not self.requested_tools:
                issues.append("missing_requested_tools")
            valid_blocks = [block for block in self.target_blocks if block in EVIDENCE_BLOCKS]
            if not valid_blocks:
                issues.append("missing_target_blocks")
            if not self.continue_review_reason:
                issues.append("missing_continue_review_reason")
        else:
            if not self.reason_codes:
                issues.append("missing_reason_codes")
            if not self.metric_refs:
                issues.append("missing_metric_refs")
        for metric_ref in self.metric_refs:
            if metric_ref not in known_refs:
                issues.append(f"missing_metric_ref:{metric_ref}")
        return issues


class GlobalSetReview(BaseModel):
    cluster_id: str
    accept_ready: bool = False
    confidence_level: Literal["high", "moderate", "low", ""] = ""
    verification_vector: dict[str, Any] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class GlobalVerifierReview(BaseModel):
    ready_for_revision: bool = False
    set_reviews: list[GlobalSetReview] = Field(default_factory=list)
    global_evidence_gaps: list[str] = Field(default_factory=list)
    cross_set_findings: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_summary: str = ""

    def contract_issues(self, candidate_sets: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        known_clusters = {str(item.get("cluster_id", "")) for item in candidate_sets}
        known_refs = global_metric_refs(candidate_sets, tool_results)
        for review in self.set_reviews:
            if review.cluster_id not in known_clusters:
                issues.append(f"unknown_cluster_id:{review.cluster_id}")
            if review.accept_ready and review.confidence_level != "high":
                issues.append(f"accept_requires_high_confidence:{review.cluster_id}")
            for metric_ref in review.metric_refs:
                if metric_ref not in known_refs:
                    issues.append(f"missing_metric_ref:{metric_ref}")
        return issues


class GlobalRouterDecision(BaseModel):
    action: Literal["continue_review", "revise"] = "continue_review"
    requested_tools: list[str] = Field(default_factory=list)
    target_sets: list[str] = Field(default_factory=list)
    target_blocks: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    continue_review_reason: str = ""
    revision_plan: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: str = ""
    rejected_tools: list[str] = Field(default_factory=list)

    def contract_issues(self, candidate_sets: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        known_clusters = {str(item.get("cluster_id", "")) for item in candidate_sets}
        known_refs = global_metric_refs(candidate_sets, tool_results)
        if self.action == "continue_review":
            if not self.requested_tools:
                issues.append("missing_requested_tools")
            if not self.target_blocks:
                issues.append("missing_target_blocks")
            if not self.continue_review_reason:
                issues.append("missing_continue_review_reason")
        else:
            if not self.revision_plan:
                issues.append("missing_revision_plan")
        for cluster_id in self.target_sets:
            if str(cluster_id) not in known_clusters:
                issues.append(f"unknown_target_set:{cluster_id}")
        for metric_ref in self.metric_refs:
            if metric_ref not in known_refs:
                issues.append(f"missing_metric_ref:{metric_ref}")
        return issues


class ReviewState(TypedDict, total=False):
    cluster_id: str
    member_ids: list[str]
    parent_ids: list[str]
    round_index: int
    budget: dict[str, Any]
    status: str
    final_action: str
    reason_codes: list[str]
    confidence_level: str
    verifier_gap: dict[str, Any]
    verification_vector: dict[str, Any]
    evidence_gaps: list[str]
    router_decision: dict[str, Any]
    requested_tools: list[str]
    tool_results: list[dict[str, Any]]
    executed_tools: list[str]
    round_trace: list[dict[str, Any]]
    generated_sets: list[dict[str, Any]]
    absorbed_set_ids: list[str]
    lineage: list[dict[str, Any]]
    artifact_paths: dict[str, str]
    system_error: dict[str, Any]


def tool_metric_refs(tool_results: list[dict[str, Any]]) -> set[str]:
    refs = {"cluster.member_count"}

    def add_refs(prefix: str, value: Any) -> None:
        refs.add(prefix)
        if isinstance(value, dict):
            for key, item in value.items():
                add_refs(f"{prefix}.{key}", item)

    for result in tool_results:
        tool_name = str(dict(result).get("tool_name", "") or "")
        if not tool_name:
            continue
        refs.update(str(item) for item in list(dict(result).get("metric_refs", []) or []))
        for metric_name, metric_value in dict(dict(result).get("metrics", {}) or {}).items():
            add_refs(f"tool_results.{tool_name}.metrics.{metric_name}", metric_value)
    return refs


def global_metric_refs(candidate_sets: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> set[str]:
    refs = tool_metric_refs(tool_results)
    for item in candidate_sets:
        cluster_id = str(dict(item).get("cluster_id", "") or "")
        if cluster_id:
            refs.add(f"cluster.{cluster_id}.member_count")
    return refs


def parse_verifier_review(payload: Any) -> VerifierReview:
    if isinstance(payload, VerifierReview):
        return payload
    return VerifierReview.model_validate(dict(payload or {}))


def parse_router_decision(payload: Any) -> RouterDecision:
    if isinstance(payload, RouterDecision):
        return payload
    return RouterDecision.model_validate(dict(payload or {}))


def parse_global_verifier_review(payload: Any) -> GlobalVerifierReview:
    if isinstance(payload, GlobalVerifierReview):
        return payload
    return GlobalVerifierReview.model_validate(dict(payload or {}))


def parse_global_router_decision(payload: Any) -> GlobalRouterDecision:
    if isinstance(payload, GlobalRouterDecision):
        return payload
    return GlobalRouterDecision.model_validate(dict(payload or {}))
