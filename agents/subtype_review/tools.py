from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.cnv_characterization import cnv_characterization
from tools.confound import confound_test
from tools.known_label_echo import known_label_echo_test
from tools.multimodal_consistency_check import multimodal_consistency_check
from tools.mutation_enrichment import mutation_enrichment
from tools.pathway_enrichment import pathway_enrichment
from tools.subtype_review_common import clinical_table, tool_result
from utils.tool_utils import to_jsonable


def clinical_characterization(
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str = "",
    all_cluster_states: list[dict[str, Any]] | None = None,
    scope: str = "set_identity",
    target_ids: list[str] | None = None,
    artifact_root: str | None = None,
) -> dict[str, Any]:
    groups = {
        str(item.get("set_id") or item.get("cluster_id")): list(item.get("member_ids", []))
        for item in list(all_cluster_states or [cluster_state])
    }
    requested = {str(target) for target in target_ids or []}
    if requested:
        groups = {group_id: members for group_id, members in groups.items() if group_id in requested}
    clinical = clinical_table(patient_states_by_id)
    rows = {}
    for group_id, members in groups.items():
        records = [clinical.get(case_id, {}) for case_id in members]
        rows[group_id] = {
            "member_n": len(members),
            "stage_values": sorted({str(record.get("stage", "")) for record in records if record.get("stage")}),
            "grade_values": sorted({str(record.get("grade", "")) for record in records if record.get("grade")}),
        }
    return tool_result(
        tool_name="clinical_characterization",
        status="success",
        cluster_id=str(cluster_state.get("cluster_id", "GLOBAL")),
        output_root=artifact_root or output_root,
        summary="Clinical stage and grade descriptors were summarized for the current partition.",
        metrics={"clinical_by_set": rows},
        decision_metrics={"clinical_by_set": rows},
    )


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "pathway_enrichment": {
        "tool_name": "pathway_enrichment",
        "dimension": "biological_support",
        "verifier_selectable": True,
        "scope": "set_identity",
        "targeting": "all_sets",
        "description": "RNA pathway enrichment for every current set.",
        "function": pathway_enrichment,
    },
    "mutation_enrichment": {
        "tool_name": "mutation_enrichment",
        "dimension": "biological_support",
        "verifier_selectable": True,
        "scope": "set_identity",
        "targeting": "all_sets",
        "description": "WXS mutation enrichment for every current set.",
        "function": mutation_enrichment,
    },
    "cnv_characterization": {
        "tool_name": "cnv_characterization",
        "dimension": "biological_support",
        "verifier_selectable": True,
        "scope": "set_identity",
        "targeting": "all_sets",
        "description": "CNV characterization for every current set.",
        "function": cnv_characterization,
    },
    "multimodal_consistency_check": {
        "tool_name": "multimodal_consistency_check",
        "dimension": "cross_modal_consistency",
        "verifier_selectable": True,
        "scope": "set_identity",
        "targeting": "all_sets",
        "description": "CT, WSI, RNA, WXS and CNV affinity diagnostics for the current partition.",
        "function": multimodal_consistency_check,
    },
    "confound_test": {
        "tool_name": "confound_test",
        "dimension": "confounder_exclusion",
        "verifier_selectable": True,
        "scope": "set_identity",
        "targeting": "all_sets",
        "description": "Technical confounder diagnostics for every current set.",
        "function": confound_test,
    },
    "known_label_echo_test": {
        "tool_name": "known_label_echo_test",
        "dimension": "known_label_echo",
        "verifier_selectable": True,
        "scope": "partition",
        "targeting": "partition",
        "description": "Whole-partition comparison with known stage and grade labels.",
        "function": known_label_echo_test,
    },
    "clinical_characterization": {
        "tool_name": "clinical_characterization",
        "dimension": "biological_support",
        "verifier_selectable": False,
        "scope": "set_identity",
        "targeting": "all_sets",
        "description": "Optional clinical descriptors for the current sets.",
        "function": clinical_characterization,
    },
}

def compact_tool_result(raw: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    payload = dict(raw or {})
    results = dict(payload.get("results", {}) or {})
    full_metrics = to_jsonable(results.get("metrics", {}) or {})
    metrics = to_jsonable(results.get("decision_metrics", results.get("metrics", {})) or {})
    errors = list(payload.get("errors", []) or [])
    missing_reason = str(results.get("missing_reason", "") or "")
    raw_status = str(payload.get("status", "")).lower()
    if raw_status == "success":
        status = "success"
    elif raw_status in {"unavailable", "scientific_unavailable", "missing"} or (
        missing_reason and not errors
    ):
        status = "scientific_unavailable"
    else:
        status = "runtime_failure"
    metric_refs = []

    def collect_leaves(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                collect_leaves(value[key], f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect_leaves(item, f"{path}[{index}]")
        else:
            metric_refs.append(path)

    collect_leaves(metrics, f"tool_results.{tool_name}.metrics")
    if tool_name == "multimodal_consistency_check":
        def prune_patient_metrics(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: prune_patient_metrics(item)
                    for key, item in value.items()
                    if key not in {"probe_labels_by_case", "patient_silhouette", "patient_margins"}
                }
            if isinstance(value, list):
                return [prune_patient_metrics(item) for item in value]
            return value

        full_metrics = {
            "structural_characterization": prune_patient_metrics(
                full_metrics.get("structural_characterization", {})
            )
        }
    else:
        full_metrics = {}
    return {
        "tool_name": tool_name,
        "status": status,
        "metrics": metrics,
        "full_metrics": full_metrics,
        "metric_refs": metric_refs,
        "warnings": list(results.get("warnings", []) or []),
        "missing_reason": missing_reason,
        "errors": errors,
        "artifact_paths": dict(payload.get("artifacts", {}) or {}),
    }


def build_validation_tools(registry: Mapping[str, Mapping[str, Any]] | None = None) -> list[Any]:
    from langchain_core.tools import tool

    registry = registry or TOOL_REGISTRY
    result = []
    for name, metadata in registry.items():
        def request_tool(
            target_ids: list[str] | None = None, _tool_name: str = name
        ) -> str:
            return f"Python will execute {_tool_name} for targets {target_ids or []}."

        result.append(tool(name, description=str(metadata["description"]))(request_tool))
    return result
