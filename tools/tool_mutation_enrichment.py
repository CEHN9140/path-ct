from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from tools.subtype_review_common import (
    bh_fdr,
    enrichment_decision_metrics,
    fisher_exact_result,
    member_case_ids,
    read_case_feature_table,
    standardized_mean_difference,
    tool_result,
)


def candidate_set_members(all_cluster_states, fallback_cluster_state):
    states = list(all_cluster_states or [fallback_cluster_state])
    return {
        str(state.get("cluster_id", "") or ""): member_case_ids(state)
        for state in states
        if str(state.get("cluster_id", "") or "")
    }


def round_value(value):
    if value is None:
        return None
    number = float(value)
    return round(number, 6) if math.isfinite(number) else None


def feature_values(table, case_ids, feature):
    return [
        float(table[case_id][feature])
        for case_id in case_ids
        if case_id in table
        and feature in table[case_id]
        and math.isfinite(float(table[case_id][feature]))
    ]


def enrichment_rows(feature_names, table, candidate_sets):
    all_case_ids = sorted(
        {case_id for members in candidate_sets.values() for case_id in members}
    )
    rows = []
    for set_id, members in candidate_sets.items():
        set_ids = [case_id for case_id in all_case_ids if case_id in members]
        rest_ids = [case_id for case_id in all_case_ids if case_id not in members]
        for feature in feature_names:
            set_values = feature_values(table, set_ids, feature)
            rest_values = feature_values(table, rest_ids, feature)
            unique_values = set(set_values + rest_values)
            binary = unique_values.issubset({0.0, 1.0})
            p_value = None
            odds_ratio = None
            if set_values and rest_values:
                if binary:
                    set_positive = int(sum(value > 0 for value in set_values))
                    rest_positive = int(sum(value > 0 for value in rest_values))
                    odds_ratio, p_value = fisher_exact_result(
                        set_positive,
                        len(set_values) - set_positive,
                        rest_positive,
                        len(rest_values) - rest_positive,
                    )
                elif len(unique_values) > 1:
                    p_value = float(
                        mannwhitneyu(
                            set_values,
                            rest_values,
                            alternative="two-sided",
                        ).pvalue
                    )
            set_mean = float(np.mean(set_values)) if set_values else None
            rest_mean = float(np.mean(rest_values)) if rest_values else None
            rows.append(
                {
                    "candidate_set_id": set_id,
                    "gene": feature,
                    "feature": feature,
                    "wxs_block": feature.split("::", 1)[0],
                    "feature_type": "binary" if binary else "continuous",
                    "available_n": len(set_values) + len(rest_values),
                    "missing_n": len(all_case_ids)
                    - len(set_values)
                    - len(rest_values),
                    "set_mean": round_value(set_mean),
                    "rest_mean": round_value(rest_mean),
                    "delta_mean": round_value(
                        set_mean - rest_mean
                        if set_mean is not None and rest_mean is not None
                        else None
                    ),
                    "standardized_mean_difference": round_value(
                        standardized_mean_difference(set_values, rest_values)
                    ),
                    "odds_ratio": round_value(odds_ratio),
                    "p_value": p_value,
                    "q_value": None,
                }
            )
    for row, q_value in zip(
        rows,
        bh_fdr(
            [
                float(row["p_value"]) if row["p_value"] is not None else 1.0
                for row in rows
            ]
        ),
    ):
        row["p_value"] = (
            float(row["p_value"]) if row["p_value"] is not None else None
        )
        row["q_value"] = float(q_value)
    return sorted(
        rows,
        key=lambda row: (
            float(row["q_value"] if row["q_value"] is not None else 1.0),
            -abs(
                float(
                    row["standardized_mean_difference"]
                    if row["standardized_mean_difference"] is not None
                    else row["delta_mean"] or 0.0
                )
            ),
            row["feature"],
            row["candidate_set_id"],
        ),
    )


def empty_result(cluster_id, output_root, reason):
    return tool_result(
        tool_name="tool_mutation_enrichment",
        status="failure",
        cluster_id=cluster_id,
        output_root=output_root,
        summary="WXS discovery and validation features are unavailable.",
        metrics={"wxs_gene_enrichment": []},
        decision_metrics={"per_set_wxs_feature_enrichment": {}},
        missing_reason=reason,
        support_level="none",
        concern_level="moderate",
        figures={},
    )


def tool_mutation_enrichment(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    wxs_dir = Path(output_root) / "wxs"
    discovery_names, discovery_table = read_case_feature_table(
        wxs_dir / "wxs_discovery_features.csv"
    )
    validation_names, validation_table = read_case_feature_table(
        wxs_dir / "wxs_validation_features.csv"
    )
    feature_names = discovery_names + validation_names
    table = {
        case_id: {
            **discovery_table.get(case_id, {}),
            **validation_table.get(case_id, {}),
        }
        for case_id in set(discovery_table) | set(validation_table)
    }
    if not feature_names or not table:
        return empty_result(
            cluster_id,
            output_root,
            "missing_wxs_feature_tables",
        )
    candidate_sets = candidate_set_members(all_cluster_states, cluster_state)
    if not candidate_sets:
        return empty_result(cluster_id, output_root, "missing_candidate_sets")
    rows = enrichment_rows(feature_names, table, candidate_sets)
    return tool_result(
        tool_name="tool_mutation_enrichment",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary=(
            "Discovery mutations and validation-only WXS summaries were "
            "tested by candidate set."
        ),
        metrics={"wxs_gene_enrichment": rows},
        decision_metrics={
            "per_set_wxs_feature_enrichment": enrichment_decision_metrics(
                rows, "gene"
            )
        },
        evidence_hints=[],
        support_level="informational",
        concern_level="none",
        figures={},
    )
