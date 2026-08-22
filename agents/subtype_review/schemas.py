from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EVIDENCE_DIMENSIONS = (
    "biological_support",
    "cross_modal_consistency",
    "confounder_exclusion",
    "known_label_echo",
)
EVIDENCE_SCOPES = ("set_identity", "partition")


class EvidenceObservation(BaseModel):
    metric: str
    finding: str
    metric_refs: list[str] = Field(default_factory=list)


class EvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    scope: str
    analysis: str
    target_ids: list[str] = Field(default_factory=list)
    observations: list[EvidenceObservation] = Field(default_factory=list)
    statistical_interpretation: str = ""
    medical_interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    subject_signature: str = ""

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

    @field_validator("analysis")
    @classmethod
    def valid_analysis(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("analysis must not be empty")
        return value.strip()

    @field_validator("target_ids")
    @classmethod
    def normalize_targets(cls, value: list[str]) -> list[str]:
        return sorted({str(item) for item in value if str(item)})


class EvidenceReportBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reports: list[EvidenceReport] = Field(default_factory=list)


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    scope: str
    analysis: str
    target_ids: list[str] = Field(default_factory=list)

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

    @field_validator("analysis")
    @classmethod
    def valid_analysis(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("analysis must not be empty")
        return value.strip()

    @field_validator("target_ids")
    @classmethod
    def normalize_targets(cls, value: list[str]) -> list[str]:
        return sorted({str(item) for item in value if str(item)})

    @model_validator(mode="after")
    def validate_scope_targets(self) -> "EvidenceRequest":
        if self.scope == "partition" and self.target_ids:
            raise ValueError("partition evidence must use empty target_ids")
        if self.scope == "set_identity" and not self.target_ids:
            raise ValueError("set_identity evidence requires target_ids")
        return self


class RouterAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["need_more_evidence", "accept", "drop", "split", "merge"]
    target_ids: list[str] = Field(min_length=1)
    requests: list[EvidenceRequest] = Field(default_factory=list)
    reason: str = ""

    @field_validator("target_ids")
    @classmethod
    def normalize_targets(cls, value: list[str]) -> list[str]:
        return sorted({str(item) for item in value if str(item)})

    @model_validator(mode="after")
    def validate_shape(self) -> "RouterAction":
        if self.action == "need_more_evidence":
            if not self.requests:
                raise ValueError("need_more_evidence requires requests")
            requested_sets = {
                target for request in self.requests for target in request.target_ids
            }
            if not requested_sets.issubset(self.target_ids):
                raise ValueError("evidence request targets must belong to the action targets")
            return self
        if self.requests:
            raise ValueError("scientific actions cannot contain evidence requests")
        if self.action in {"accept", "drop", "split"} and len(self.target_ids) != 1:
            raise ValueError(f"{self.action} requires one target_id")
        if self.action == "merge" and len(self.target_ids) < 2:
            raise ValueError("merge requires at least two target_ids")
        return self


class RouterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[RouterAction] = Field(min_length=1)


class ReviserOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["split", "merge"]
    target_ids: list[str] = Field(min_length=1)
    n_children: int | None = Field(default=None, ge=2)
    structural_basis: list[str] = Field(default_factory=list)
    execution_strategy: Literal[
        "multimodal_consensus", "fused_similarity_spectral"
    ] | None = None
    metric_refs: list[str] = Field(default_factory=list)
    rationale: str = ""


class ReviewContext(TypedDict, total=False):
    patient_states_by_id: dict[str, dict[str, Any]]
    output_root: str
    data_root: str
    review_output_root: str
    config_dir: str


class ReviewState(TypedDict, total=False):
    sets: list[dict[str, Any]]
    round_evidence: list[dict[str, Any]]
    evidence_history: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    messages: list[Any]
    revision: dict[str, Any] | None
    control: dict[str, Any]


def set_id(item: dict[str, Any]) -> str:
    return str(item.get("set_id") or item.get("cluster_id") or "")


def active_sets(sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in sets
        if str(item.get("status", "active"))
        not in {"superseded_by_split", "superseded_by_merge"}
    ]
