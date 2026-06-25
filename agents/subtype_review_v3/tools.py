from __future__ import annotations

import importlib
import json
import math
from typing import Any, Callable, Mapping

from agents.subtype_review_v3.schemas import (
    CompactToolResult,
    EVIDENCE_BLOCKS,
)
from utils.tool_utils import to_jsonable

KIRC_DRIVER_GENES = {"VHL", "PBRM1", "BAP1", "SETD2", "KDM5C", "MTOR", "PTEN", "TSC1", "TSC2"}

DEFAULT_TOOL_DEFINITIONS = {
    "tool_stability_check": {
        "module": "tools.tool_stability_check",
        "function": "tool_stability_check",
        "evidence_blocks": ["set_reliability"],
    },
    "tool_survival_analysis": {
        "module": "tools.tool_survival_analysis",
        "function": "tool_survival_analysis",
        "evidence_blocks": ["clinical_context"],
    },
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
        "evidence_blocks": ["multimodal_support"],
    },
}


def normalize_tool_definitions(raw: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    definitions = dict(raw or DEFAULT_TOOL_DEFINITIONS)
    normalized = {}
    for tool_name, definition in definitions.items():
        item = dict(definition or {})
        normalized[str(tool_name)] = {
            "module": str(item.get("module", "") or ""),
            "function": str(item.get("function", "") or ""),
            "evidence_blocks": [
                str(block)
                for block in list(item.get("evidence_blocks", []) or [])
                if str(block) in EVIDENCE_BLOCKS
            ],
        }
    return normalized


def load_tool_functions(
    tool_definitions: Mapping[str, Mapping[str, Any]],
    tool_names: list[str] | None = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    functions = {}
    requested = set(tool_names or tool_definitions.keys())
    for tool_name, definition in tool_definitions.items():
        if str(tool_name) not in requested:
            continue
        module_name = str(dict(definition).get("module", "") or "")
        function_name = str(dict(definition).get("function", "") or "")
        if not module_name or not function_name:
            continue
        functions[str(tool_name)] = getattr(importlib.import_module(module_name), function_name)
    return functions


def load_available_tool_functions(
    tool_definitions: Mapping[str, Mapping[str, Any]],
    tool_names: list[str] | None = None,
) -> tuple[dict[str, Callable[..., dict[str, Any]]], dict[str, str]]:
    functions = {}
    errors = {}
    requested = set(tool_names or tool_definitions.keys())
    for tool_name in requested:
        try:
            functions.update(load_tool_functions(tool_definitions, [str(tool_name)]))
        except Exception as exc:
            errors[str(tool_name)] = f"{type(exc).__name__}: {exc}"
    return functions, errors


def build_validation_tools(
    tool_functions: Mapping[str, Callable[..., Any]],
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[dict[str, Any]],
) -> list[Any]:
    tools = []
    try:
        from langchain_core.tools import StructuredTool
    except Exception:
        StructuredTool = None

    def make_run_tool(tool_name: str, tool_function: Callable[..., Any]) -> Callable[[], str]:
        def run_tool() -> str:
            try:
                raw_result = tool_function(
                    dict(cluster_state),
                    {str(key): dict(value) for key, value in patient_states_by_id.items()},
                    output_root,
                    config_dir=config_dir,
                    all_cluster_states=all_cluster_states,
                )
            except Exception as exc:
                raw_result = {
                    "tool_name": tool_name,
                    "status": "failure",
                    "results": {"metrics": {}, "warnings": [], "missing_reason": ""},
                    "artifacts": {},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            return json.dumps(compact_tool_result(raw_result), ensure_ascii=False)

        return run_tool

    for tool_name, tool_function in tool_functions.items():
        run_tool = make_run_tool(str(tool_name), tool_function)
        description = f"Compute subtype-review validation metrics for {tool_name}."
        if StructuredTool is None:
            run_tool.name = tool_name
            run_tool.description = description
            tools.append(run_tool)
        else:
            tools.append(StructuredTool.from_function(func=run_tool, name=tool_name, description=description))
    return tools


def collect_tool_results_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    results = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        try:
            payload = json.loads(str(content or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("tool_name"):
            results.append(CompactToolResult.model_validate(payload).model_dump())
    return results


def number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def slim_enrichment_row(row: Mapping[str, Any], name_field: str) -> dict[str, Any]:
    kept = {
        name_field: row.get(name_field),
        "q_value": row.get("q_value"),
        "p_value": row.get("p_value"),
        "delta_frequency": row.get("delta_frequency"),
        "standardized_mean_difference": row.get("standardized_mean_difference"),
        "odds_ratio": row.get("odds_ratio"),
    }
    return {key: to_jsonable(value) for key, value in kept.items() if value is not None}


def summarize_enrichment_rows(rows: list[Any], name_field: str, cluster_id: str = "") -> dict[str, Any]:
    table = [dict(row) for row in rows if isinstance(row, Mapping)]
    if cluster_id:
        scoped = [row for row in table if str(row.get("candidate_set_id", "") or "") == cluster_id]
        if scoped:
            table = scoped
    q_values = [number_or_none(row.get("q_value")) for row in table]
    p_values = [number_or_none(row.get("p_value")) for row in table]
    q_values = [value for value in q_values if value is not None]
    p_values = [value for value in p_values if value is not None]

    def q_sort_key(row: Mapping[str, Any]) -> float:
        value = number_or_none(row.get("q_value"))
        return value if value is not None else float("inf")

    def effect_value(row: Mapping[str, Any]) -> float:
        for key in ("delta_frequency", "standardized_mean_difference", "delta_mean_score"):
            value = number_or_none(row.get(key))
            if value is not None:
                return abs(value)
        odds_ratio = number_or_none(row.get("odds_ratio"))
        if odds_ratio is not None and odds_ratio > 0:
            return abs(math.log2(odds_ratio))
        return 0.0

    return {
        "summary": {
            "tested_count": len(table),
            "significant_count_q05": sum(1 for value in q_values if value <= 0.05),
            "significant_count_q10": sum(1 for value in q_values if value <= 0.10),
            "min_q_value": min(q_values) if q_values else None,
            "min_p_value": min(p_values) if p_values else None,
            "positive_effect_count": sum(
                1
                for row in table
                if (number_or_none(row.get("delta_frequency")) or number_or_none(row.get("standardized_mean_difference")) or 0) > 0
            ),
            "negative_effect_count": sum(
                1
                for row in table
                if (number_or_none(row.get("delta_frequency")) or number_or_none(row.get("standardized_mean_difference")) or 0) < 0
            ),
        },
        "top_by_q": [slim_enrichment_row(row, name_field) for row in sorted(table, key=q_sort_key)[:10]],
        "top_by_effect": [
            slim_enrichment_row(row, name_field)
            for row in sorted(table, key=effect_value, reverse=True)[:10]
        ],
    }


def summarize_enrichment_by_set(rows: list[Any], name_field: str) -> dict[str, Any]:
    cluster_ids = sorted(
        {
            str(dict(row).get("candidate_set_id", "") or "")
            for row in rows
            if isinstance(row, Mapping) and str(dict(row).get("candidate_set_id", "") or "")
        }
    )
    return {
        cluster_id: summarize_enrichment_rows(rows, name_field, cluster_id)
        for cluster_id in cluster_ids
    }


def compact_llm_metrics(tool_name: str, metrics: Mapping[str, Any], cluster_id: str = "") -> dict[str, Any]:
    compacted = dict(metrics or {})
    if tool_name == "tool_mutation_enrichment":
        compacted = {}
        gene_rows = list(metrics.get("wxs_gene_enrichment", []) or [])
        pathway_rows = list(metrics.get("wxs_pathway_enrichment", []) or [])
        if cluster_id == "GLOBAL":
            compacted["per_set_gene_enrichment"] = summarize_enrichment_by_set(gene_rows, "gene")
            compacted["per_set_pathway_enrichment"] = summarize_enrichment_by_set(pathway_rows, "pathway")
            return to_jsonable(compacted)
        gene_summary = summarize_enrichment_rows(gene_rows, "gene", cluster_id)
        pathway_summary = summarize_enrichment_rows(pathway_rows, "pathway", cluster_id)
        compacted["gene_enrichment_summary"] = gene_summary["summary"]
        compacted["top_genes_by_q"] = gene_summary["top_by_q"]
        compacted["top_genes_by_effect"] = gene_summary["top_by_effect"]
        compacted["known_driver_hits"] = [
            row for row in gene_summary["top_by_q"] + gene_summary["top_by_effect"]
            if str(row.get("gene", "") or "").upper() in KIRC_DRIVER_GENES
        ][:10]
        compacted["pathway_enrichment_summary"] = pathway_summary["summary"]
        compacted["top_pathways_by_q"] = pathway_summary["top_by_q"]
        compacted["top_pathways_by_effect"] = pathway_summary["top_by_effect"]
    elif tool_name == "tool_pathway_enrichment" and "rna_pathway_enrichment" in metrics:
        pathway_rows = list(metrics.get("rna_pathway_enrichment", []) or [])
        if cluster_id == "GLOBAL":
            compacted = {
                "per_set_rna_pathway_enrichment": summarize_enrichment_by_set(pathway_rows, "pathway")
            }
            return to_jsonable(compacted)
        pathway_summary = summarize_enrichment_rows(
            pathway_rows,
            "pathway",
            cluster_id,
        )
        compacted = {
            "rna_pathway_enrichment_summary": pathway_summary["summary"],
            "top_rna_pathways_by_q": pathway_summary["top_by_q"],
            "top_rna_pathways_by_effect": pathway_summary["top_by_effect"],
        }
    return to_jsonable(compacted)


def compact_tool_result(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(result or {})
    results = dict(payload.get("results", {}) or {})
    artifacts = dict(payload.get("artifacts", {}) or {})
    tool_name = str(payload.get("tool_name", "") or "")
    metrics = compact_llm_metrics(
        tool_name,
        dict(results.get("metrics", {}) or {}),
        str(payload.get("cluster_id", "") or ""),
    )
    return CompactToolResult(
        tool_name=tool_name,
        status=str(payload.get("status", "") or "failure"),
        metrics=metrics,
        warnings=[str(item) for item in list(results.get("warnings", []) or [])],
        missing_reason=str(results.get("missing_reason", "") or ""),
        artifact_paths=to_jsonable(artifacts),
        metric_refs=[f"tool_results.{tool_name}.metrics.{metric_name}" for metric_name in metrics],
        errors=[str(item) for item in list(payload.get("errors", []) or [])],
    ).model_dump()


def execute_requested_tools(
    requested_tools: list[str],
    tool_functions: Mapping[str, Callable[..., dict[str, Any]]],
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for tool_name in requested_tools:
        try:
            raw_result = tool_functions[str(tool_name)](
                dict(cluster_state),
                {str(key): dict(value) for key, value in patient_states_by_id.items()},
                output_root,
                config_dir=config_dir,
                all_cluster_states=all_cluster_states,
            )
        except Exception as exc:
            raw_result = {
                "tool_name": str(tool_name),
                "status": "failure",
                "results": {"metrics": {}, "warnings": [], "missing_reason": ""},
                "artifacts": {},
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(compact_tool_result(raw_result))
    return results
