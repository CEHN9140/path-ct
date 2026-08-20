from __future__ import annotations

from typing import Any, Callable, Mapping

from tools.cnv_characterization import cnv_characterization
from tools.confound_test import confound_test
from tools.known_label_echo_test import known_label_echo_test
from tools.multimodal_consistency_check import multimodal_consistency_check
from tools.mutation_enrichment import mutation_enrichment
from tools.pathway_enrichment import pathway_enrichment
from utils.tool_utils import to_jsonable


def compact_tool_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(raw or {})
    results = dict(payload.get("results", {}) or {})
    metrics = results.get("decision_metrics")
    if metrics is None:
        metrics = results.get("metrics", {})
    metrics = to_jsonable(metrics or {})
    tool_name = str(payload.get("tool_name", "") or "")
    return {
        "tool_name": tool_name,
        "status": "success" if str(payload.get("status", "")).lower() == "success" else "failure",
        "metrics": metrics,
        "metric_refs": [f"tool_results.{tool_name}.metrics.{key}" for key in metrics],
        "warnings": list(results.get("warnings", []) or []),
        "missing_reason": str(results.get("missing_reason", "") or ""),
        "errors": list(payload.get("errors", []) or []),
        "artifact_paths": dict(payload.get("artifacts", {}) or {}),
    }


def validation_result(
    dimension: str,
    functions: list[tuple[str, Callable[..., Any]]],
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[dict[str, Any]],
    *,
    scope: str = "set_identity",
    target_ids: list[str] | None = None,
    proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    results = []
    for name, function in functions:
        try:
            raw = function(
                dict(cluster_state),
                {str(key): dict(value) for key, value in patient_states_by_id.items()},
                output_root,
                config_dir=config_dir,
                all_cluster_states=all_cluster_states,
                scope=scope,
                target_ids=list(target_ids or []),
                proposal=dict(proposal or {}),
            )
        except Exception as exc:
            raw = {
                "tool_name": name,
                "status": "failure",
                "results": {"metrics": {}, "warnings": [], "missing_reason": ""},
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(compact_tool_result(raw))
    return {
        "dimension": dimension,
        "status": "success" if all(item["status"] == "success" for item in results) else "failure",
        "results": results,
    }


def biological_support(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return validation_result(
        "biological_support",
        [
            ("pathway_enrichment", pathway_enrichment),
            ("mutation_enrichment", mutation_enrichment),
            ("cnv_characterization", cnv_characterization),
        ],
        *args,
        **kwargs,
    )


def cross_modal_consistency(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return validation_result(
        "cross_modal_consistency",
        [("multimodal_consistency_check", multimodal_consistency_check)],
        *args,
        **kwargs,
    )


def confounder_exclusion(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return validation_result(
        "confounder_exclusion",
        [("confound_test", confound_test)],
        *args,
        **kwargs,
    )


def known_label_echo(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return validation_result(
        "known_label_echo",
        [("known_label_echo_test", known_label_echo_test)],
        *args,
        **kwargs,
    )


VALIDATION_FUNCTIONS = {
    "biological_support": biological_support,
    "cross_modal_consistency": cross_modal_consistency,
    "confounder_exclusion": confounder_exclusion,
    "known_label_echo": known_label_echo,
}


def build_validation_tools() -> list[Any]:
    from langchain_core.tools import tool

    descriptions = {
        "biological_support": "Compute RNA, WXS and CNV biological evidence for the requested scope.",
        "cross_modal_consistency": "Compute CT, WSI, RNA and genomic consistency for the requested scope.",
        "confounder_exclusion": "Compute CT acquisition-confounder evidence for the requested scope.",
        "known_label_echo": "Compute whole-partition stage and grade echo evidence.",
    }

    def request_validation() -> str:
        return "Use the pending evidence request supplied by Python."

    return [
        tool(name, description=description)(request_validation)
        for name, description in descriptions.items()
    ]
