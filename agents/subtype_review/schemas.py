from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EVIDENCE_DIMENSIONS = (
    "cross_modal_consistency",
    "biological_support",
    "known_label_echo",
    "confounder_exclusion",
)


class EvidenceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    value: Any
    meaning: str = Field(min_length=1)
    finding: str = Field(min_length=1)


class EvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal[
        "biological_support",
        "cross_modal_consistency",
        "confounder_exclusion",
        "known_label_echo",
    ]
    aspect: str = Field(min_length=1)
    scope: Literal["set", "pair", "partition"]
    target_ids: list[str] = Field(default_factory=list)
    observations: list[EvidenceObservation] = Field(default_factory=list)
    dimension_interpretation: str = Field(min_length=1)
    cross_evidence_context: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    tool_refs: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)

    @field_validator("target_ids", "tool_refs", "metric_refs")
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})

    @model_validator(mode="after")
    def valid_report_shape(self) -> "EvidenceReport":
        expected = {"set": 1, "pair": 2, "partition": 0}[self.scope]
        if len(self.target_ids) != expected:
            raise ValueError(f"{self.scope} reports require exactly {expected} target_ids")

        return self


class EvidenceReportBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reports: list[EvidenceReport] = Field(default_factory=list)


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal[
        "biological_support",
        "cross_modal_consistency",
        "confounder_exclusion",
        "known_label_echo",
    ]
    scope: Literal["set", "pair", "partition"]
    target_ids: list[str] = Field(default_factory=list)
    focus: str = Field(min_length=1)
    question: str

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("question must not be empty")
        return value

    @field_validator("target_ids")
    @classmethod
    def unique_targets(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})

    @model_validator(mode="after")
    def valid_scope(self) -> "EvidenceRequest":
        expected = {"set": 1, "pair": 2, "partition": 0}[self.scope]
        if len(self.target_ids) != expected:
            raise ValueError(f"{self.scope} requests require exactly {expected} target_ids")
        return self


class RouterAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["accept", "drop", "split", "merge"]
    target_ids: list[str] = Field(min_length=1)
    n_children: int | None = Field(default=None, ge=2)
    evidence_report_refs: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("target_ids", "evidence_report_refs")
    @classmethod
    def unique_targets(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})

    @model_validator(mode="after")
    def valid_shape(self) -> "RouterAction":
        if self.action in {"accept", "drop", "split"} and len(self.target_ids) != 1:
            raise ValueError(f"{self.action} requires one target")
        if self.action == "merge" and len(self.target_ids) != 2:
            raise ValueError("merge requires exactly two targets")
        if (self.action == "split") != (self.n_children is not None):
            raise ValueError("n_children is required only for split actions")
        return self


class RouterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[RouterAction] = Field(default_factory=list)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_mode(self) -> "RouterPlan":
        if bool(self.actions) == bool(self.evidence_requests):
            raise ValueError(
                "RouterPlan must contain exactly one mode: "
                "scientific actions or evidence requests"
            )
        return self


class SplitPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["split"] = "split"
    target_id: str
    n_children: int = Field(ge=2)
    structural_basis: list[str] = Field(min_length=1)
    execution_strategy: Literal["candidate_consensus_spectral"]
    metric_refs: list[str] = Field(default_factory=list)
    rationale: str = ""


class MergePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["merge"] = "merge"
    target_ids: list[str] = Field(min_length=2, max_length=2)
    metric_refs: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("target_ids", "metric_refs")
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})


class RevisionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_plans: list[SplitPlan] = Field(default_factory=list)
    merge_plans: list[MergePlan] = Field(default_factory=list)
    rationale: str = ""


class ReviewContext(TypedDict, total=False):
    patient_states_by_id: dict[str, dict[str, Any]]
    data_root: str
    config_dir: str
    tool_registry: dict[str, dict[str, Any]]
    verifier_model: Any
    router_model: Any
    reviser_model: Any
    runtime_trace_path: str


class ReviewControl(TypedDict, total=False):
    round: int
    status: Literal["reviewing", "complete", "incomplete_due_to_round_budget"]
    next: Literal["router", "verifier", "reviser", "end"]
    max_rounds: int
    pending_evidence_requests: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    llm_usage: dict[str, int | float | None]


class ReviewState(TypedDict, total=False):
    """Shared partition, evidence ledger, and run control."""

    partition: dict[str, Any]
    tool_evidence: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    router_plan: dict[str, Any] | None
    revision_plan: dict[str, Any] | None
    revision_result: dict[str, Any] | None
    control: ReviewControl
    history: list[dict[str, Any]]


def set_id(item: dict[str, Any]) -> str:
    return str(item.get("set_id") or item.get("cluster_id") or "")
