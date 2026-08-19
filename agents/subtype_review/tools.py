from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

from utils.tool_utils import to_jsonable

DEFAULT_TOOL_DEFINITIONS = {
    "tool_mutation_enrichment": {
        "module": "tools.tool_mutation_enrichment",
        "function": "tool_mutation_enrichment",
    },
    "tool_pathway_enrichment": {
        "module": "tools.tool_pathway_enrichment",
        "function": "tool_pathway_enrichment",
    },
    "tool_confound_test": {
        "module": "tools.tool_confound_test",
        "function": "tool_confound_test",
    },
    "tool_known_label_echo_test": {
        "module": "tools.tool_known_label_echo_test",
        "function": "tool_known_label_echo_test",
    },
    "tool_multimodal_consistency_check": {
        "module": "tools.tool_multimodal_consistency_check",
        "function": "tool_multimodal_consistency_check",
    },
    "tool_cnv_characterization": {
        "module": "tools.tool_cnv_characterization",
        "function": "tool_cnv_characterization",
    },
}


def normalize_tool_definitions(
    raw: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    definitions = dict(raw or DEFAULT_TOOL_DEFINITIONS)
    return {
        str(name): {
            "module": str(dict(item or {}).get("module", "") or ""),
            "function": str(dict(item or {}).get("function", "") or ""),
        }
        for name, item in definitions.items()
    }


def load_available_tool_functions(
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Callable[..., Any]]:
    functions: dict[str, Callable[..., Any]] = {}
    for name, item in definitions.items():
        module = importlib.import_module(str(item.get("module", "")))
        functions[str(name)] = getattr(module, str(item.get("function", "")))
    return functions


def compact_tool_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(raw or {})
    results = dict(payload.get("results", {}) or {})
    metrics = results.get("decision_metrics")
    if metrics is None:
        metrics = results.get("metrics", {})
    metrics = to_jsonable(metrics or {})
    tool_name = str(payload.get("tool_name", "") or "")
    metric_refs = [
        f"tool_results.{tool_name}.metrics.{key}"
        for key in metrics
    ]
    return {
        "tool_name": tool_name,
        "status": "success" if str(payload.get("status", "")).lower() == "success" else "failure",
        "metrics": metrics,
        "metric_refs": metric_refs,
        "warnings": list(results.get("warnings", []) or []),
        "missing_reason": str(results.get("missing_reason", "") or ""),
        "errors": list(payload.get("errors", []) or []),
        "artifact_paths": dict(payload.get("artifacts", {}) or {}),
    }


def run_validation_function(
    tool_name: str,
    tool_function: Callable[..., Any],
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
    try:
        raw = tool_function(
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
            "tool_name": tool_name,
            "status": "failure",
            "results": {"metrics": {}, "warnings": [], "missing_reason": ""},
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    return compact_tool_result(raw)


def execute_capability(
    capability: str,
    tool_functions: Mapping[str, Callable[..., Any]],
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
    mapping = {
        "biological_support": [
            "tool_mutation_enrichment",
            "tool_pathway_enrichment",
            "tool_cnv_characterization",
        ],
        "cross_modal_consistency": ["tool_multimodal_consistency_check"],
        "confounder_exclusion": ["tool_confound_test"],
        "known_label_echo": ["tool_known_label_echo_test"],
    }
    names = mapping.get(str(capability), [])
    results = [
        run_validation_function(
            name,
            tool_functions[name],
            cluster_state,
            patient_states_by_id,
            output_root,
            config_dir,
            all_cluster_states,
            scope=scope,
            target_ids=target_ids,
            proposal=proposal,
        )
        for name in names
        if name in tool_functions
    ]
    return {
        "capability": str(capability),
        "status": "success" if results and all(
            item["status"] == "success" for item in results
        ) else "failure",
        "results": results,
    }


def build_validation_tools() -> list[Any]:
    from langchain_core.tools import tool

    def make(name: str, description: str, func: Callable[..., Any]) -> Any:
        return tool(name, description=description)(func)

    def biological_support() -> str:
        """Compute RNA, WXS and CNV biological evidence for the requested scope."""
        return "biological_support"

    def cross_modal_consistency() -> str:
        """Compute CT, WSI, RNA and genomic consistency for the current partition."""
        return "cross_modal_consistency"

    def confounder_exclusion() -> str:
        """Compute CT acquisition-confounder evidence for the current partition."""
        return "confounder_exclusion"

    def known_label_echo() -> str:
        """Compute whole-partition stage and grade echo evidence."""
        return "known_label_echo"

    return [
        make("biological_support", biological_support.__doc__ or "", biological_support),
        make("cross_modal_consistency", cross_modal_consistency.__doc__ or "", cross_modal_consistency),
        make("confounder_exclusion", confounder_exclusion.__doc__ or "", confounder_exclusion),
        make("known_label_echo", known_label_echo.__doc__ or "", known_label_echo),
    ]
