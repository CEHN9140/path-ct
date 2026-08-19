from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

EVIDENCE_DIMENSIONS = (
    "biological_support",
    "cross_modal_consistency",
    "confounder_exclusion",
    "known_label_echo",
)
EVIDENCE_SCOPES = (
    "set_identity",
    "split_proposal",
    "merge_proposal",
    "partition",
)
FINDING_STATUSES = (
    "supporting",
    "conflicting",
    "mixed",
    "inconclusive",
    "unavailable",
)
class AuditFinding(BaseModel):
    target_ids: list[str] = Field(default_factory=list)
    dimension: str
    scope: str
    proposal_id: str | None = None
    subject_signature: str = ""
    status: str
    summary: str = ""
    metric_refs: list[str] = Field(default_factory=list)

    @field_validator("dimension")
    @classmethod
    def valid_dimension(cls, value: str) -> str:
        if value not in EVIDENCE_DIMENSIONS:
            raise ValueError(f"unknown evidence dimension: {value}")
        return value

    @field_validator("scope")
    @classmethod
    def valid_scope(cls, value: str) -> str:
        if value not in EVIDENCE_SCOPES:
            raise ValueError(f"unknown evidence scope: {value}")
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
    scope: str
    proposal_id: str | None = None
    subject_signature: str = ""
    reason: str = ""

    @field_validator("dimension")
    @classmethod
    def valid_dimension(cls, value: str) -> str:
        if value not in EVIDENCE_DIMENSIONS:
            raise ValueError(f"unknown evidence dimension: {value}")
        return value

    @field_validator("scope")
    @classmethod
    def valid_scope(cls, value: str) -> str:
        if value not in EVIDENCE_SCOPES:
            raise ValueError(f"unknown evidence scope: {value}")
        return value


class VerifierOutput(BaseModel):
    findings: list[AuditFinding] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)


class RouterAction(BaseModel):
    action: Literal["need_more_evidence", "accept", "drop", "split", "merge"]
    target_ids: list[str] = Field(default_factory=list)
    dimension: str | None = None
    scope: str | None = None
    proposal_id: str | None = None
    reason: str = ""
    metric_refs: list[str] = Field(default_factory=list)

    @field_validator("dimension")
    @classmethod
    def valid_dimension(cls, value: str | None) -> str | None:
        if value is not None and value not in EVIDENCE_DIMENSIONS:
            raise ValueError(f"unknown evidence dimension: {value}")
        return value

    @field_validator("scope")
    @classmethod
    def valid_scope(cls, value: str | None) -> str | None:
        if value is not None and value not in EVIDENCE_SCOPES:
            raise ValueError(f"unknown evidence scope: {value}")
        return value

    @model_validator(mode="after")
    def valid_action_shape(self) -> "RouterAction":
        self.target_ids = [str(item) for item in self.target_ids if str(item)]
        if self.action == "need_more_evidence":
            if self.dimension is None or self.scope is None:
                raise ValueError("need_more_evidence requires dimension and scope")
            if self.scope in {"split_proposal", "merge_proposal"} and not self.proposal_id:
                raise ValueError("proposal evidence requests require proposal_id")
        else:
            if self.dimension is not None or self.scope is not None or self.proposal_id is not None:
                raise ValueError("scientific actions must clear evidence fields")
            if len(self.target_ids) != 1:
                raise ValueError("scientific actions require one target_id")
        return self


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
    structure_proposals: dict[str, Any]


def set_id(item: dict[str, Any]) -> str:
    return str(item.get("set_id") or item.get("cluster_id") or "")


def active_sets(sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in sets if str(item.get("status", "active")) != "retired"]
