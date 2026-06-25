from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from tools.subtype_review_common import (
    clinical_table,
    subtype_review_tool_definitions,
)
from utils.tool_utils import safe_identifier, to_jsonable

FINAL_REVIEW_ACTIONS = {
    "accept",
    "drop",
    "split",
    "merge",
}

EVIDENCE_BLOCK_NAMES = (
    "set_reliability",
    "biological_support",
    "multimodal_support",
    "clinical_context",
    "known_label_echo",
    "confounder_exclusion",
)

TABLE_PREVIEW_ROW_LIMIT = 20

FINAL_ACTION_REASON_CODES = {
    "accept": ["reliable_biological_nonconfounded"],
    "drop": [
        "unstable_or_boundary_poor",
        "biologically_unsupported",
        "known_label_echo",
        "confounder_driven",
        "major_multimodal_contradiction",
        "not_evaluable_from_available_data",
        "insufficient_evidence",
        "insufficient_after_max_rounds",
        "review_failed_after_max_failures",
    ],
    "split": ["internal_heterogeneity_supported"],
    "merge": ["weak_boundary_compatible_evidence"],
}


def normalize_review_action(action: str) -> str:
    action = str(action or "").strip().lower()
    if action == "reject":
        return "drop"
    return action if action in FINAL_REVIEW_ACTIONS else "drop"


def compute_verification_vector(
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    member_ids = [str(item) for item in list(cluster_state.get("member_ids", []))]
    member_states = [
        dict(patient_states_by_id[member_id])
        for member_id in member_ids
        if member_id in patient_states_by_id
    ]
    member_count = len(member_states)
    validation_results = dict(cluster_state.get("validation_results", {}) or {})
    tool_results = {
        key: dict(
            dict(validation_results.get(tool_name, {}) or {}).get("results", {}) or {}
        )
        for key, tool_name in {
            "stability": "tool_stability_check",
            "multimodal_consistency": "tool_multimodal_consistency_check",
            "survival": "tool_survival_analysis",
            "mutation": "tool_mutation_enrichment",
            "pathway": "tool_pathway_enrichment",
            "confounder": "tool_confound_test",
            "known_label_echo": "tool_known_label_echo_test",
        }.items()
    }
    feature_counts = {
        key: sum(
            1
            for item in member_states
            if dict(item.get(evidence_key, {}) or {}).get("features")
        )
        for key, evidence_key in {
            "ct": "ct_evidence",
            "pathology": "wsi_evidence",
        }.items()
    }
    clinical = clinical_table(patient_states_by_id)
    member_clinical = [clinical.get(member_id, {}) for member_id in member_ids]
    return {
        **tool_results,
        "ct": {
            "members_with_ct_features": feature_counts["ct"],
            "member_count": member_count,
        },
        "pathology": {
            "members_with_wsi_features": feature_counts["pathology"],
            "member_count": member_count,
        },
        "clinical": {
            "members_with_clinical": sum(
                1
                for item in member_states
                if dict(item.get("inventory", {}) or {}).get("Clinical")
            ),
            "members_with_os": sum(
                1 for item in member_clinical if item.get("os_time") is not None
            ),
            "os_events": sum(
                1 for item in member_clinical if int(item.get("os_event") or 0) == 1
            ),
            "members_with_stage": sum(
                1 for item in member_clinical if str(item.get("stage_group", "") or "")
            ),
            "members_with_grade": sum(
                1 for item in member_clinical if str(item.get("grade", "") or "")
            ),
            "member_count": member_count,
        },
        "raw": {
            "member_ids": member_ids,
            "member_count": member_count,
            "validation_result_tools": sorted(validation_results.keys()),
            "source_views": list(cluster_state.get("source_views", [])),
        },
    }


def evidence_block(metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "metrics": dict(metrics or {}),
        "llm_assessment": {
            "assessment": "unavailable",
            "rationale": "No LLM assessment has been recorded for this evidence block.",
            "metric_refs": [],
        },
        "figures": {},
        "source_paths": {},
    }


def tool_metrics(cluster_state: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    validation = dict(cluster_state.get("validation_results", {}) or {})
    payload = dict(validation.get(tool_name, {}) or {})
    return dict(dict(payload.get("results", {}) or {}).get("metrics", {}) or {})


def metric_table_view(
    rows: list[Any],
    *,
    full_table_ref: str,
    row_limit: int = TABLE_PREVIEW_ROW_LIMIT,
) -> dict[str, Any]:
    normalized_rows = [dict(row or {}) for row in list(rows or [])]
    schema = sorted(
        {
            str(key)
            for row in normalized_rows[: max(row_limit, 1)]
            for key in dict(row).keys()
        }
    )
    return {
        "row_count": len(normalized_rows),
        "schema": schema,
        "preview_rows": normalized_rows[:row_limit],
        "preview_row_count": min(len(normalized_rows), row_limit),
        "omitted_row_count": max(0, len(normalized_rows) - row_limit),
        "sort_order": "q_value ascending, then absolute effect size descending",
        "full_table_ref": full_table_ref,
    }


def compact_biological_metrics_for_review(metrics: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(metrics or {})
    refs = {
        "wxs_gene_enrichment": (
            "validation_results.tool_mutation_enrichment.results.metrics."
            "wxs_gene_enrichment"
        ),
        "wxs_pathway_enrichment": (
            "validation_results.tool_mutation_enrichment.results.metrics."
            "wxs_pathway_enrichment"
        ),
        "rna_pathway_enrichment": (
            "validation_results.tool_pathway_enrichment.results.metrics."
            "rna_pathway_enrichment"
        ),
    }
    return {
        key: metric_table_view(
            list(metrics.get(key, []) or []),
            full_table_ref=ref,
        )
        for key, ref in refs.items()
        if key in metrics
    }


def tool_figures(cluster_state: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    validation = dict(cluster_state.get("validation_results", {}) or {})
    payload = dict(validation.get(tool_name, {}) or {})
    return dict(dict(payload.get("artifacts", {}) or {}).get("figures", {}) or {})


def association_rows(rows: Any) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        iterable = rows.values()
    else:
        iterable = list(rows or [])
    normalized = []
    for item in iterable:
        row = dict(item or {})
        normalized.append(
            {
                "field": str(
                    row.get("field", row.get("label", row.get("name", ""))) or ""
                ),
                "test": str(row.get("test", row.get("method", "")) or ""),
                "p_value": row.get("p_value"),
                "q_value": row.get("q_value"),
                "effect_size": row.get("effect_size"),
                "effect_size_name": str(row.get("effect_size_name", "") or ""),
                "level": row.get("level"),
                "candidate_set_id": row.get("candidate_set_id"),
                "dominant_level": row.get("dominant_level"),
                "available_n": row.get("available_n"),
                "missing_n": row.get("missing_n"),
                "overlap_count": row.get("overlap_count"),
                "set_available_n": row.get("set_available_n"),
                "level_total_n": row.get("level_total_n"),
                "set_fraction": row.get("set_fraction"),
                "level_recall": row.get("level_recall"),
                "odds_ratio": row.get("odds_ratio"),
                "n_cases": row.get("n_cases"),
                "member_count": row.get("member_count"),
                "member_total": row.get("member_total"),
                "rest_count": row.get("rest_count"),
                "rest_total": row.get("rest_total"),
                "cramers_v": row.get("cramers_v"),
                "cluster_distribution": dict(row.get("cluster_distribution", {}) or {}),
                "rest_distribution": dict(row.get("rest_distribution", {}) or {}),
            }
        )
    return normalized


def build_set_reliability_block(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    stability = tool_metrics(cluster_state, "tool_stability_check")
    metrics = {
        "set_reliability_global_consensus": dict(
            stability.get("set_reliability_global_consensus", {}) or {}
        ),
        "set_reliability_set_consensus": dict(
            stability.get("set_reliability_set_consensus", {}) or {}
        ),
    }
    block = evidence_block(metrics)
    block["figures"] = {}
    consensus = dict(cluster_state.get("consensus", {}) or {})
    block["source_paths"] = {
        key: value
        for key, value in {
            "consensus_matrix": consensus.get("matrix_path"),
            "consensus_metadata": consensus.get("metadata_path"),
            "partition_records": consensus.get("partition_records_path"),
        }.items()
        if value
    }
    return block


def build_biological_support_block(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    mutation = tool_metrics(cluster_state, "tool_mutation_enrichment")
    pathway = tool_metrics(cluster_state, "tool_pathway_enrichment")
    full_metrics = {
        "wxs_gene_enrichment": list(mutation.get("wxs_gene_enrichment", []) or []),
        "wxs_pathway_enrichment": list(
            mutation.get("wxs_pathway_enrichment", []) or []
        ),
        "rna_pathway_enrichment": list(pathway.get("rna_pathway_enrichment", []) or []),
    }
    block = evidence_block(compact_biological_metrics_for_review(full_metrics))
    block["figures"] = {}
    return block


def build_multimodal_support_block(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    multimodal = tool_metrics(cluster_state, "tool_multimodal_consistency_check")
    metrics = {
        "crossmodal_global_alignment": list(
            multimodal.get("crossmodal_global_alignment", []) or []
        ),
        "crossmodal_set_alignment": dict(
            multimodal.get("crossmodal_set_alignment", {}) or {}
        ),
    }
    block = evidence_block(metrics)
    return block


def build_clinical_context_block(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    survival = tool_metrics(cluster_state, "tool_survival_analysis")
    metrics = {
        "survival_global_association": dict(
            survival.get("survival_global_association", {}) or {}
        ),
        "survival_set_association": dict(
            survival.get("survival_set_association", {}) or {}
        ),
    }
    block = evidence_block(metrics)
    return block


def build_known_label_echo_block(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    known = tool_metrics(cluster_state, "tool_known_label_echo_test")
    metrics = {
        "known_label_global_association": dict(
            known.get("known_label_global_association", {}) or {}
        ),
        "known_label_set_enrichment": dict(
            known.get("known_label_set_enrichment", {}) or {}
        ),
    }
    block = evidence_block(metrics)
    return block


def build_confounder_exclusion_block(
    cluster_state: Mapping[str, Any],
) -> dict[str, Any]:
    confound = tool_metrics(cluster_state, "tool_confound_test")
    metrics = {
        "confounder_global_association": dict(
            confound.get("confounder_global_association", {}) or {}
        ),
        "confounder_set_association": dict(
            confound.get("confounder_set_association", {}) or {}
        ),
    }
    block = evidence_block(metrics)
    return block


def apply_structured_evidence_assessments(
    matrix: dict[str, Any], cluster_state: Mapping[str, Any]
) -> dict[str, Any]:
    aliases = {
        "stability": "set_reliability",
        "set_reliability": "set_reliability",
        "biological_support": "biological_support",
        "multimodal_consistency": "multimodal_support",
        "multimodal_support": "multimodal_support",
        "clinical_relevance": "clinical_context",
        "clinical_context": "clinical_context",
        "known_label_echo": "known_label_echo",
        "confounder_exclusion": "confounder_exclusion",
    }
    for item in list(cluster_state.get("structured_evidence", []) or []):
        evidence = dict(item or {})
        block_name = aliases.get(str(evidence.get("dimension", "") or ""))
        if block_name not in matrix:
            continue
        text = " ".join(
            str(evidence.get(key, "") or "")
            for key in ("evidence", "limitations", "metric")
        ).lower()
        source_metric_text = str(evidence.get("metric", "") or "").lower()
        unavailable_text = any(
            phrase in text
            for phrase in (
                "no evidence available",
                "no known label echo evidence",
                "no cluster-level validation tool evidence",
                "evidence is important",
                "testing is important",
                "entirely missing",
                "not run",
            )
        )
        assessment = "unavailable" if unavailable_text or any(
            marker in source_metric_text
            for marker in (
                "not run",
                "no evidence",
                "entirely missing",
                "no data",
            )
        ) else "reviewed"
        metric_ref = f"evidence_matrix.{block_name}.metrics"
        matrix[block_name]["llm_assessment"] = {
            "assessment": assessment,
            "rationale": " ".join(
                str(evidence.get(key, "") or "")
                for key in ("evidence", "limitations")
                if str(evidence.get(key, "") or "")
            ),
            "metric_refs": [metric_ref],
            "source_metric_text": str(evidence.get("metric", "") or ""),
        }
    return matrix


def significant_metric_present(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        value = row.get("q_value", row.get("p_value"))
        if value is not None:
            try:
                if float(value) <= 0.05:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def known_label_echo_supported(known: Mapping[str, Any], text: str) -> bool:
    negative_echo = any(
        phrase in text
        for phrase in [
            "not a known label echo",
            "not strongly echo",
            "not dominated",
            "no known-label dominance",
            "no known label dominance",
            "not a simple echo",
        ]
    )
    explicit_echo = any(
        phrase in text
        for phrase in [
            "known label echo",
            "known-label echo",
            "dominant known",
            "simple echo",
            "entirely explained by",
        ]
    )
    return explicit_echo and not negative_echo


def known_label_metrics_indicate_echo(known: Mapping[str, Any]) -> bool:
    raw_set_rows = known.get("known_label_set_enrichment", {}) or {}
    set_rows = []
    if isinstance(raw_set_rows, Mapping):
        for rows in raw_set_rows.values():
            if isinstance(rows, Mapping):
                set_rows.extend(list(rows.values()))
            else:
                set_rows.extend(list(rows or []))
    else:
        set_rows = list(raw_set_rows or [])
    if significant_metric_present(set_rows):
        return True
    for row in set_rows:
        purity = row.get("set_fraction")
        recall = row.get("level_recall")
        try:
            if (
                purity is not None
                and recall is not None
                and float(purity) >= 0.8
                and float(recall) >= 0.5
            ):
                return True
        except (TypeError, ValueError):
            pass
    association_rows_for_labels = list(
        dict(known.get("known_label_global_association", {}) or {}).values()
    )
    for row in association_rows_for_labels:
        value = dict(row or {}).get("cramers_v")
        if value is not None:
            try:
                if float(value) >= 0.3:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def biological_pathway_support_present(metrics: Mapping[str, Any]) -> bool:
    rows = []
    for key in (
        "wxs_gene_enrichment",
        "wxs_pathway_enrichment",
        "rna_pathway_enrichment",
    ):
        value = metrics.get(key, [])
        if isinstance(value, Mapping):
            rows += list(dict(value).get("preview_rows", []) or [])
        else:
            rows += list(value or [])
    return significant_metric_present(rows)


def build_agentic_evidence_matrix(
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    previous_evidence_matrix: Mapping[str, Any] | None = None,
    blocks_to_update: list[str] | None = None,
) -> dict[str, Any]:
    previous = {
        str(key): dict(value)
        for key, value in dict(previous_evidence_matrix or {}).items()
    }
    requested = set(blocks_to_update or EVIDENCE_BLOCK_NAMES)
    if not previous:
        requested = set(EVIDENCE_BLOCK_NAMES)
    builders = {
        "set_reliability": build_set_reliability_block,
        "biological_support": build_biological_support_block,
        "multimodal_support": build_multimodal_support_block,
        "clinical_context": build_clinical_context_block,
        "known_label_echo": build_known_label_echo_block,
        "confounder_exclusion": build_confounder_exclusion_block,
    }
    matrix = {}
    for name in EVIDENCE_BLOCK_NAMES:
        if name in requested:
            try:
                matrix[name] = builders[name](cluster_state)
            except Exception as exc:
                matrix[name] = {
                    **evidence_block({}),
                    "status": "unavailable",
                    "error": {
                        "failed_source": name,
                        "error_message": f"{type(exc).__name__}: {exc}",
                    },
                }
        else:
            matrix[name] = previous.get(name, evidence_block({}))
        matrix[name].setdefault("metrics", {})
        matrix[name].setdefault("llm_assessment", {})
        matrix[name]["llm_assessment"].setdefault("assessment", "unavailable")
        matrix[name]["llm_assessment"].setdefault("rationale", "")
        matrix[name]["llm_assessment"].setdefault("metric_refs", [])
        matrix[name].setdefault("figures", {})
        matrix[name].setdefault("source_paths", {})
    return apply_structured_evidence_assessments(matrix, cluster_state)


def blocks_from_tool_plan(tool_plan: list[Any], config_dir: str = "") -> list[str]:
    tool_definitions = subtype_review_tool_definitions(config_dir)
    blocks = []
    for item in tool_plan:
        tool_name = str(
            (
                dict(item).get("tool_name", dict(item).get("name", ""))
                if isinstance(item, Mapping)
                else item
            )
            or ""
        )
        tool_blocks = list(
            dict(tool_definitions.get(tool_name, {}) or {}).get("evidence_blocks", [])
            or []
        )
        for block_name in tool_blocks:
            if block_name not in blocks:
                blocks.append(block_name)
    return blocks


def metric_refs_for_blocks(block_names: list[str]) -> list[str]:
    refs = []
    for block_name in block_names:
        refs.append(f"evidence_matrix.{block_name}.metrics")
    return refs


def evidence_blocks_missing_metrics(evidence_matrix: Mapping[str, Any]) -> list[str]:
    missing = []
    for block_name in EVIDENCE_BLOCK_NAMES:
        block = dict(evidence_matrix.get(block_name, {}) or {})
        metrics = dict(block.get("metrics", {}) or {})
        if not metrics:
            missing.append(block_name)
    return missing


def evidence_id_part(value: Any) -> str:
    return safe_identifier(str(value or "item")).replace("-", "_")


def metric_ref_for_path(path: list[Any]) -> str:
    return ".".join(["evidence_matrix"] + [str(item) for item in path])


def compact_metric_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): compact_metric_snapshot(item)
            for key, item in list(value.items())[:8]
        }
    if isinstance(value, list):
        return [compact_metric_snapshot(item) for item in value[:5]]
    return to_jsonable(value)


def build_evidence_catalog(
    cluster_id: str,
    evidence_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    cluster_prefix = safe_identifier(str(cluster_id or "cluster"))
    for block_name in EVIDENCE_BLOCK_NAMES:
        metrics = dict(
            dict(evidence_matrix.get(block_name, {}) or {}).get("metrics", {}) or {}
        )
        if not metrics:
            continue
        for metric_name, metric_value in metrics.items():
            metric_path = [block_name, "metrics", metric_name]
            if isinstance(metric_value, list):
                for index, row in enumerate(metric_value[:5]):
                    catalog.append(
                        {
                            "evidence_id": (
                                f"{cluster_prefix}__{block_name}__"
                                f"{evidence_id_part(metric_name)}__{index}"
                            ),
                            "block": block_name,
                            "statement": f"{block_name}.{metric_name}[{index}]",
                            "raw_metric_snapshot": compact_metric_snapshot(row),
                            "metric_refs": [metric_ref_for_path(metric_path + [index])],
                        }
                    )
            else:
                catalog.append(
                    {
                        "evidence_id": (
                            f"{cluster_prefix}__{block_name}__"
                            f"{evidence_id_part(metric_name)}"
                        ),
                        "block": block_name,
                        "statement": f"{block_name}.{metric_name}",
                        "raw_metric_snapshot": compact_metric_snapshot(metric_value),
                        "metric_refs": [metric_ref_for_path(metric_path)],
                    }
                )
    return catalog


def evidence_catalog_by_id(
    evidence_catalog: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("evidence_id", "")): dict(item)
        for item in evidence_catalog
        if str(item.get("evidence_id", ""))
    }


def metric_refs_from_evidence_ids(
    evidence_ids: list[str],
    evidence_catalog: list[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    catalog = evidence_catalog_by_id(evidence_catalog)
    refs: list[str] = []
    missing: list[str] = []
    for evidence_id in evidence_ids:
        item = catalog.get(str(evidence_id))
        if not item:
            missing.append(str(evidence_id))
            continue
        for ref in list(item.get("metric_refs", []) or []):
            ref = str(ref)
            if ref and ref not in refs:
                refs.append(ref)
    return refs, missing


def reason_codes_for_action(action: str, drop_reason: str = "") -> list[str]:
    action = normalize_review_action(action)
    if action == "drop":
        reason = str(drop_reason or "")
        if reason in FINAL_ACTION_REASON_CODES["drop"]:
            return [reason]
        return ["other_with_explanation"]
    return list(FINAL_ACTION_REASON_CODES.get(action, ["other_with_explanation"]))


def allowed_reason_codes_for_action(action: str) -> set[str]:
    normalized = str(action or "").strip().lower()
    if normalized == "reject":
        normalized = "drop"
    codes = set(FINAL_ACTION_REASON_CODES.get(normalized, []))
    if normalized == "drop":
        codes.add("other_with_explanation")
    return codes


def normalize_confidence_level(value: Any) -> str:
    level = str(value or "").strip().lower()
    return level if level in {"high", "moderate", "low"} else ""


def normalize_decision_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    return state if state in {"final", "continue_review"} else ""


def metric_ref_exists(metric_ref: str, evidence_matrix: Mapping[str, Any]) -> bool:
    parts = [part for part in str(metric_ref or "").split(".") if part]
    if parts and parts[0] in {"evidence_matrix", "final_evidence_matrix"}:
        parts = parts[1:]
    value: Any = evidence_matrix
    for part in parts:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
            continue
        if isinstance(value, list):
            try:
                value = value[int(part)]
                continue
            except (ValueError, IndexError):
                return False
        return False
    return True


def nearest_existing_metric_ref(
    metric_ref: str,
    evidence_matrix: Mapping[str, Any],
) -> str:
    ref = str(metric_ref or "")
    if metric_ref_exists(ref, evidence_matrix):
        return ref
    parts = [part for part in ref.split(".") if part]
    prefix = ""
    if parts and parts[0] in {"evidence_matrix", "final_evidence_matrix"}:
        prefix = parts[0]
        parts = parts[1:]
    for end in range(len(parts) - 1, 0, -1):
        if "metrics" not in parts[:end]:
            continue
        candidate = ".".join(([prefix] if prefix else []) + parts[:end])
        if metric_ref_exists(candidate, evidence_matrix):
            return candidate
    return ref


def normalize_metric_refs_for_matrix(
    metric_refs: list[str],
    evidence_matrix: Mapping[str, Any],
) -> list[str]:
    normalized = []
    for ref in [str(item) for item in list(metric_refs or []) if str(item)]:
        resolved = nearest_existing_metric_ref(ref, evidence_matrix)
        if resolved not in normalized:
            normalized.append(resolved)
    return normalized


def normalize_reason_codes_for_final_action(
    action: str,
    reason_codes: list[str],
    drop_reason: str = "",
) -> list[str]:
    normalized_action = normalize_review_action(action)
    if normalized_action not in FINAL_REVIEW_ACTIONS:
        return list(reason_codes or [])
    allowed = allowed_reason_codes_for_action(normalized_action)
    kept = [str(code) for code in list(reason_codes or []) if str(code) in allowed]
    if kept:
        return kept
    if normalized_action == "accept":
        all_reason_codes = {
            code
            for codes in FINAL_ACTION_REASON_CODES.values()
            for code in list(codes or [])
        }
        invalid_codes = [str(code) for code in list(reason_codes or []) if str(code)]
        looks_like_dimension_or_free_text = all(
            code in EVIDENCE_BLOCK_NAMES or code not in all_reason_codes
            for code in invalid_codes
        )
        if looks_like_dimension_or_free_text:
            return reason_codes_for_action(normalized_action, drop_reason)
    return list(reason_codes or [])


def decision_consistency_check(
    final_action: str | None,
    reason_codes: list[str],
    rationale: str,
    metric_refs: list[str],
    evidence_matrix: Mapping[str, Any],
    confidence_level: str = "",
    evidence_ids: list[str] | None = None,
    evidence_catalog: list[Mapping[str, Any]] | None = None,
    decision_state: str = "final",
    blocks_to_update: list[str] | None = None,
    continue_review_reason: str = "",
) -> dict[str, Any]:
    decision_state = normalize_decision_state(decision_state) or "final"
    action = str(final_action or "").strip().lower()
    if action == "reject":
        action = "drop"
    issues = []
    if decision_state == "final":
        if not action:
            issues.append("missing_recommended_action")
        elif action not in FINAL_REVIEW_ACTIONS:
            issues.append("invalid_final_action")
    confidence_level = normalize_confidence_level(confidence_level)
    if not confidence_level:
        issues.append("missing_confidence_level")
    elif (
        decision_state == "final"
        and action == "accept"
        and confidence_level != "high"
    ):
        issues.append(f"non_high_confidence:{confidence_level}")
    if decision_state == "final" and not [code for code in reason_codes if str(code)]:
        issues.append("missing_reason_codes")
    allowed_codes = allowed_reason_codes_for_action(action)
    for code in [str(item) for item in reason_codes if str(item)]:
        if decision_state == "final" and code not in allowed_codes:
            issues.append(f"reason_code_not_allowed_for_action:{code}")
    if not str(rationale or "").strip():
        issues.append("missing_rationale")
    if decision_state == "continue_review":
        requested_blocks = [
            str(item)
            for item in list(blocks_to_update or [])
            if str(item) in EVIDENCE_BLOCK_NAMES
        ]
        if not requested_blocks and not str(continue_review_reason or "").strip():
            issues.append("missing_continue_review_reason")
    lower = str(rationale or "").lower()
    if decision_state == "final" and action == "drop" and any(
        phrase in lower
        for phrase in (
            "supports accepting",
            "support accepting",
            "supports accept",
            "should be accepted",
            "accept this candidate",
        )
    ):
        issues.append("action_rationale_conflict")
    if decision_state == "final" and action == "accept" and any(
        phrase in lower
        for phrase in (
            "should be dropped",
            "must be dropped",
            "drop this candidate",
            "not reliable enough to accept",
        )
    ):
        issues.append("action_rationale_conflict")
    checked_refs = normalize_metric_refs_for_matrix(
        [str(ref) for ref in list(metric_refs or []) if str(ref)],
        evidence_matrix,
    )
    checked_evidence_ids = [
        str(item) for item in list(evidence_ids or []) if str(item)
    ]
    if evidence_catalog:
        known_ids = set(evidence_catalog_by_id(list(evidence_catalog)).keys())
        if not checked_evidence_ids and not checked_refs:
            issues.append("missing_evidence_ids")
        for evidence_id in checked_evidence_ids:
            if evidence_id not in known_ids:
                issues.append(f"missing_evidence_id:{evidence_id}")
    if not checked_refs:
        issues.append("missing_metric_refs")
    for ref in checked_refs:
        if not metric_ref_exists(ref, evidence_matrix):
            issues.append(f"missing_metric_ref:{ref}")
    return {
        "passed": not issues,
        "issues": issues,
        "metric_refs_checked": checked_refs,
        "evidence_ids_checked": checked_evidence_ids,
    }


def normalize_confidence_basis(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return {"summary": list(value)}
    text = str(value or "").strip()
    return {"summary": text} if text else {}


def build_verifier_decision(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    audit = dict(cluster_state.get("llm_audit", {}) or {})
    decision_state = normalize_decision_state(audit.get("decision_state"))
    action = str(audit.get("recommended_action", "") or "").strip().lower()
    if action == "reject":
        action = "drop"
    if action not in FINAL_REVIEW_ACTIONS:
        action = ""
    if not decision_state:
        decision_state = (
            "continue_review"
            if not action
            and (
                list(audit.get("blocks_to_update", []) or [])
                or str(audit.get("continue_review_reason", "") or "").strip()
            )
            else "final"
        )
    tool_plan = list(
        cluster_state.get("tool_plan", [])
        or audit.get("tool_plan", [])
        or audit.get("tools_to_call", [])
        or []
    )
    blocks_to_update = blocks_from_tool_plan(tool_plan)
    evidence_matrix = dict(cluster_state.get("evidence_matrix", {}) or {})
    evidence_catalog = list(
        cluster_state.get("evidence_catalog", [])
        or build_evidence_catalog(str(cluster_state.get("cluster_id", "")), evidence_matrix)
    )
    dimension_assessments = {
        name: str(
            dict(
                dict(evidence_matrix.get(name, {}) or {}).get("llm_assessment", {})
                or {}
            ).get("assessment", "unavailable")
        )
        for name in EVIDENCE_BLOCK_NAMES
    }
    rationale = str(audit.get("reasoning_summary", "") or "")
    reason_codes = [
        str(item)
        for item in list(audit.get("reason_codes", []) or [])
        if str(item)
    ]
    if decision_state == "final":
        reason_codes = normalize_reason_codes_for_final_action(
            action,
            reason_codes,
            str(audit.get("drop_reason", "") or ""),
        )
    has_valid_reason = bool(reason_codes)
    evidence_review = []
    for name in EVIDENCE_BLOCK_NAMES:
        assessment = dict(
            dict(evidence_matrix.get(name, {}) or {}).get("llm_assessment", {}) or {}
        )
        evidence_review.append(
            {
                "block": name,
                "assessment": str(assessment.get("assessment", "unavailable") or ""),
                "rationale": str(assessment.get("rationale", "") or ""),
                "metric_refs": list(assessment.get("metric_refs", []) or []),
            }
        )
    evidence_ids = [
        str(item) for item in list(audit.get("evidence_ids", []) or []) if str(item)
    ]
    metric_refs, missing_evidence_ids = metric_refs_from_evidence_ids(
        evidence_ids, evidence_catalog
    )
    if not metric_refs:
        metric_refs = [
            str(item) for item in list(audit.get("metric_refs", []) or []) if str(item)
        ]
    if not metric_refs:
        metric_refs = (
            metric_refs_for_blocks(list(EVIDENCE_BLOCK_NAMES))
            if action in FINAL_REVIEW_ACTIONS and not tool_plan
            else metric_refs_for_blocks(blocks_to_update)
        )
    if not metric_refs:
        metric_refs = [
            "evidence_matrix.set_reliability.metrics.set_reliability_global_consensus.total_candidate_set_n"
        ]
    metric_refs = normalize_metric_refs_for_matrix(metric_refs, evidence_matrix)
    confidence_level = normalize_confidence_level(audit.get("confidence_level"))
    confidence_basis = normalize_confidence_basis(audit.get("confidence_basis"))
    audit_blocks = [
        str(item)
        for item in list(audit.get("blocks_to_update", []) or [])
        if str(item) in EVIDENCE_BLOCK_NAMES
    ]
    blocks_to_update = blocks_to_update or audit_blocks
    consistency = decision_consistency_check(
        action,
        reason_codes,
        rationale,
        metric_refs,
        evidence_matrix,
        confidence_level=confidence_level,
        evidence_ids=evidence_ids,
        evidence_catalog=evidence_catalog,
        decision_state=decision_state,
        blocks_to_update=blocks_to_update,
        continue_review_reason=str(audit.get("continue_review_reason", "") or ""),
    )
    if missing_evidence_ids:
        consistency["issues"].extend(
            f"missing_evidence_id:{item}" for item in missing_evidence_ids
        )
        consistency["passed"] = False
    final_action_ready = bool(
        decision_state == "final"
        and action in FINAL_REVIEW_ACTIONS
        and not tool_plan
        and confidence_level in ({"high"} if action == "accept" else {"high", "moderate"})
        and rationale
        and has_valid_reason
        and consistency["passed"]
    )
    return {
        "cluster_id": str(cluster_state.get("cluster_id", "")),
        "round_index": int(cluster_state.get("review_round", 0) or 0),
        "dimension_assessments": dimension_assessments,
        "decision_state": decision_state,
        "confidence_level": confidence_level,
        "confidence_basis": confidence_basis,
        "final_action_ready": final_action_ready,
        "final_action": action if final_action_ready else None,
        "reason_codes": reason_codes if final_action_ready else reason_codes,
        "evidence_review": evidence_review,
        "blocks_to_update": blocks_to_update,
        "continue_review_reason": str(audit.get("continue_review_reason", "") or ""),
        "rationale": rationale,
        "evidence_ids": evidence_ids,
        "metric_refs": metric_refs,
        "decision_consistency_check": consistency,
    }


def normalize_action_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(decision or {})
    action = str(normalized.get("recommended_action", "") or "").strip().lower()
    if action == "reject":
        action = "drop"
    normalized["recommended_action"] = action if action else ""
    decision_state = normalize_decision_state(normalized.get("decision_state"))
    if not decision_state:
        decision_state = (
            "continue_review"
            if not normalized["recommended_action"]
            and (
                list(normalized.get("blocks_to_update", []) or [])
                or str(normalized.get("continue_review_reason", "") or "").strip()
            )
            else "final"
        )
    normalized["decision_state"] = decision_state
    normalized["reason_codes"] = [
        str(item) for item in list(normalized.get("reason_codes", []) or []) if str(item)
    ]
    normalized["evidence_ids"] = [
        str(item) for item in list(normalized.get("evidence_ids", []) or []) if str(item)
    ]
    normalized["metric_refs"] = [
        str(item) for item in list(normalized.get("metric_refs", []) or []) if str(item)
    ]
    normalized["confidence_level"] = normalize_confidence_level(
        normalized.get("confidence_level")
    )
    normalized["blocks_to_update"] = [
        str(item)
        for item in list(normalized.get("blocks_to_update", []) or [])
        if str(item) in EVIDENCE_BLOCK_NAMES
    ]
    return normalized


def action_decision_contract_issues(
    decision: Mapping[str, Any],
    evidence_matrix: Mapping[str, Any],
    evidence_catalog: list[Mapping[str, Any]],
) -> list[str]:
    payload = normalize_action_decision(decision)
    refs, _missing = metric_refs_from_evidence_ids(
        list(payload.get("evidence_ids", []) or []),
        evidence_catalog,
    )
    if not refs:
        refs = list(payload.get("metric_refs", []) or [])
    refs = normalize_metric_refs_for_matrix(refs, evidence_matrix)
    reason_codes = list(payload.get("reason_codes", []) or [])
    if str(payload.get("decision_state", "") or "final") == "final":
        reason_codes = normalize_reason_codes_for_final_action(
            str(payload.get("recommended_action", "") or ""),
            reason_codes,
            str(payload.get("drop_reason", "") or ""),
        )
    check = decision_consistency_check(
        final_action=str(payload.get("recommended_action", "") or ""),
        reason_codes=reason_codes,
        rationale=str(payload.get("reasoning_summary", "") or ""),
        metric_refs=refs,
        evidence_matrix=evidence_matrix,
        confidence_level=str(payload.get("confidence_level", "") or ""),
        evidence_ids=list(payload.get("evidence_ids", []) or []),
        evidence_catalog=evidence_catalog,
        decision_state=str(payload.get("decision_state", "") or "final"),
        blocks_to_update=list(payload.get("blocks_to_update", []) or []),
        continue_review_reason=str(payload.get("continue_review_reason", "") or ""),
    )
    return list(check.get("issues", []) or [])
