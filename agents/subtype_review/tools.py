from __future__ import annotations

import importlib
import json
from typing import Any, Callable, Mapping, Literal

from agents.subtype_review.schemas import EVIDENCE_DIMENSIONS
from utils.tool_utils import to_jsonable

DEFAULT_TOOL_DEFINITIONS = {
    "tool_mutation_enrichment": {
        "module": "tools.tool_mutation_enrichment",
        "function": "tool_mutation_enrichment",
        "evidence_blocks": ["biological_support"],
    },
    "tool_pathway_enrichment": {
        "module": "tools.tool_pathway_enrichment",
        "function": "tool_pathway_enrichment",
        "evidence_blocks": ["biological_support"],
    },
    "tool_confound_test": {
        "module": "tools.tool_confound_test",
        "function": "tool_confound_test",
        "evidence_blocks": ["confounder_exclusion"],
    },
    "tool_known_label_echo_test": {
        "module": "tools.tool_known_label_echo_test",
        "function": "tool_known_label_echo_test",
        "evidence_blocks": ["known_label_echo"],
    },
    "tool_multimodal_consistency_check": {
        "module": "tools.tool_multimodal_consistency_check",
        "function": "tool_multimodal_consistency_check",
        "evidence_blocks": ["cross_modal_consistency"],
    },
    "tool_structural_adequacy": {
        "module": "tools.tool_structural_adequacy",
        "function": "tool_structural_adequacy",
        "evidence_blocks": ["structural_adequacy"],
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
            "evidence_blocks": [
                str(block)
                for block in list(dict(item or {}).get("evidence_blocks", []) or [])
                if str(block) in EVIDENCE_DIMENSIONS
            ],
        }
        for name, item in definitions.items()
    }


def load_available_tool_functions(
    definitions: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Callable[..., Any]], dict[str, str]]:
    functions: dict[str, Callable[..., Any]] = {}
    errors: dict[str, str] = {}
    for name, item in definitions.items():
        try:
            module = importlib.import_module(str(item.get("module", "")))
            functions[str(name)] = getattr(module, str(item.get("function", "")))
        except Exception as exc:
            errors[str(name)] = f"{type(exc).__name__}: {exc}"
    return functions, errors


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
        "status": str(payload.get("status", "failure") or "failure"),
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
) -> dict[str, Any]:
    try:
        raw = tool_function(
            dict(cluster_state),
            {str(key): dict(value) for key, value in patient_states_by_id.items()},
            output_root,
            config_dir=config_dir,
            all_cluster_states=all_cluster_states,
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
) -> dict[str, Any]:
    mapping = {
        "biological_support": [
            "tool_mutation_enrichment",
            "tool_pathway_enrichment",
        ],
        "cross_modal_consistency": ["tool_multimodal_consistency_check"],
        "confounder_exclusion": ["tool_confound_test"],
        "known_label_echo": ["tool_known_label_echo_test"],
        "structural_adequacy": ["tool_structural_adequacy"],
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


def build_validation_tools(executor: Callable[..., dict[str, Any]]) -> list[Any]:
    from langchain_core.tools import tool

    def make(name: str, description: str, func: Callable[..., Any]) -> Any:
        return tool(name, description=description)(func)

    def biological_support(target_set_ids: list[str] | None = None) -> str:
        """Compute RNA and WXS biological-support evidence for target sets."""
        return json.dumps(executor("biological_support", target_set_ids or []), ensure_ascii=False)

    def cross_modal_consistency(target_set_ids: list[str] | None = None) -> str:
        """Compute CT, WSI, RNA and genomic affinity consistency evidence."""
        return json.dumps(executor("cross_modal_consistency", target_set_ids or []), ensure_ascii=False)

    def confounder_exclusion(target_set_ids: list[str] | None = None) -> str:
        """Compute CT acquisition-confounder evidence for target sets."""
        return json.dumps(executor("confounder_exclusion", target_set_ids or []), ensure_ascii=False)

    def known_label_echo(target_set_ids: list[str] | None = None) -> str:
        """Compute whole-partition stage and grade echo evidence."""
        return json.dumps(executor("known_label_echo", target_set_ids or []), ensure_ascii=False)

    def structural_adequacy(
        target_set_ids: list[str] | None = None,
        scope: Literal["internal", "external", "both"] = "both",
    ) -> str:
        """Compute legal Split and Merge candidates for current sets."""
        return json.dumps(executor("structural_adequacy", target_set_ids or [], scope), ensure_ascii=False)

    return [
        make("biological_support", biological_support.__doc__ or "", biological_support),
        make("cross_modal_consistency", cross_modal_consistency.__doc__ or "", cross_modal_consistency),
        make("confounder_exclusion", confounder_exclusion.__doc__ or "", confounder_exclusion),
        make("known_label_echo", known_label_echo.__doc__ or "", known_label_echo),
        make("structural_adequacy", structural_adequacy.__doc__ or "", structural_adequacy),
    ]
