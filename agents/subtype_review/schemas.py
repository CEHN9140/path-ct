from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

EVIDENCE_DIMENSIONS = (
    "biological_support",
    "cross_modal_consistency",
    "confounder_exclusion",
    "known_label_echo",
    "structural_adequacy",
)
FINDING_STATUSES = (
    "supporting",
    "conflicting",
    "mixed",
    "inconclusive",
    "unavailable",
)
ROUTER_ACTIONS = ("accept", "drop", "split", "merge")


class AuditFinding(BaseModel):
    target_ids: list[str] = Field(default_factory=list)
    dimension: str
    status: str
    summary: str = ""
    metric_refs: list[str] = Field(default_factory=list)

    @field_validator("dimension")
    @classmethod
    def valid_dimension(cls, value: str) -> str:
        if value not in EVIDENCE_DIMENSIONS:
            raise ValueError(f"unknown evidence dimension: {value}")
        return value

    @field_validator("status")
    @classmethod
    def valid_status(cls, value: str) -> str:
        if value not in FINDING_STATUSES:
            raise ValueError(f"unknown audit status: {value}")
        return value


class EvidenceGap(BaseModel):
    target_ids: list[str] = Field(default_factory=list)
    dimension: str
    reason: str = ""

    @field_validator("dimension")
    @classmethod
    def valid_dimension(cls, value: str) -> str:
        if value not in EVIDENCE_DIMENSIONS:
            raise ValueError(f"unknown evidence dimension: {value}")
        return value


class VerifierOutput(BaseModel):
    findings: list[AuditFinding] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)


class RouterAction(BaseModel):
    action: Literal["accept", "drop", "split", "merge"]
    target_id: str
    reason: str = ""
    metric_refs: list[str] = Field(default_factory=list)


class ReviserOutput(BaseModel):
    plan_id: str | None = None
    reason: str = ""
    metric_refs: list[str] = Field(default_factory=list)


class ReviewState(TypedDict, total=False):
    sets: list[dict[str, Any]]
    evidence: dict[str, Any]
    audit: dict[str, Any]
    action: dict[str, Any] | None
    messages: list[Any]
    control: dict[str, Any]


def set_id(item: dict[str, Any]) -> str:
    return str(item.get("set_id") or item.get("cluster_id") or "")


def active_sets(sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in sets if str(item.get("status", "active")) != "retired"]


def action_target_ids(action: dict[str, Any]) -> list[str]:
    target = str(action.get("target_id", "") or "")
    return [item for item in target.split("+") if item]
