from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EVIDENCE_DIMENSIONS = (
    "biological_support",
    "cross_modal_consistency",
    "confounder_exclusion",
    "known_label_echo",
)


class EvidenceObservation(BaseModel):
    metric: str
    finding: str
    metric_refs: list[str] = Field(default_factory=list)


class EvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal[
        "biological_support",
        "cross_modal_consistency",
        "confounder_exclusion",
        "known_label_echo",
    ]
    scope: Literal["set_identity", "partition"]
    target_ids: list[str] = Field(default_factory=list)
    observations: list[EvidenceObservation] = Field(default_factory=list)
    statistical_interpretation: str = ""
    medical_interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    tool_refs: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)

    @field_validator("target_ids", "tool_refs", "metric_refs")
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})

    @model_validator(mode="after")
    def valid_scope(self) -> "EvidenceReport":
        if self.scope == "partition" and self.target_ids:
            raise ValueError("partition reports cannot target sets")
        if self.scope == "set_identity" and not self.target_ids:
            raise ValueError("set reports require target_ids")
        return self


class EvidenceReportBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reports: list[EvidenceReport] = Field(default_factory=list)


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    target_ids: list[str] = Field(default_factory=list)

    @field_validator("tool_name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("tool_name must not be empty")
        return value

    @field_validator("target_ids")
    @classmethod
    def unique_targets(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})


class RouterAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["need_more_evidence", "accept", "drop", "split", "merge"]
    target_ids: list[str] = Field(min_length=1)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    reason: str = ""

    @field_validator("target_ids")
    @classmethod
    def unique_targets(cls, values: list[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})

    @model_validator(mode="after")
    def valid_shape(self) -> "RouterAction":
        if self.action == "need_more_evidence":
            if not self.tool_requests:
                raise ValueError("need_more_evidence requires tool_requests")
            if any(
                not set(request.target_ids).issubset(self.target_ids)
                for request in self.tool_requests
            ):
                raise ValueError("tool request targets must belong to the action")
        elif self.tool_requests:
            raise ValueError("scientific actions cannot contain tool_requests")
        if self.action in {"accept", "drop", "split"} and len(self.target_ids) != 1:
            raise ValueError(f"{self.action} requires one target")
        if self.action == "merge" and len(self.target_ids) != 2:
            raise ValueError("merge requires exactly two targets")
        return self


class RouterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[RouterAction] = Field(min_length=1)


class SplitPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["split"] = "split"
    target_id: str
    n_children: int = Field(ge=2)
    structural_basis: list[str] = Field(min_length=1)
    execution_strategy: Literal["fused_similarity_spectral"]
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
    artifact_root: str
    config_dir: str
    tool_registry: dict[str, dict[str, Any]]
    verifier_model: Any
    router_model: Any
    reviser_model: Any


class ReviewState(TypedDict, total=False):
    partition: dict[str, Any]
    round_evidence: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    messages: list[Any]
    router_plan: dict[str, Any] | None
    revision_plan: dict[str, Any] | None
    revision_result: dict[str, Any] | None
    control: dict[str, Any]
    history: list[dict[str, Any]]


def set_id(item: dict[str, Any]) -> str:
    return str(item.get("set_id") or item.get("cluster_id") or "")
