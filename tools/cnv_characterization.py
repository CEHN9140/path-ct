from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu

from tools.subtype_review_common import (
    assign_groupwise_fdr,
    cliffs_delta,
    fisher_exact_result,
    odds_ratio_ci,
    read_case_feature_table,
    ranked_decision_summary,
    scoped_candidate_sets,
    tool_parameters,
    tool_result,
)


def cnv_characterization(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
    scope="set_identity",
    target_ids=None,
    artifact_root=None,
):
    artifact_root = artifact_root or output_root
    path = Path(output_root) / "cnv" / "case_features.csv"
    feature_names, table = read_case_feature_table(str(path))
    if not feature_names or not table:
        return tool_result(
            tool_name="cnv_characterization",
            status="failure",
            cluster_id=str(cluster_state.get("cluster_id", "unknown_cluster")),
            output_root=artifact_root,
            summary="CNV feature table is unavailable.",
            metrics={"cnv_characterization": []},
            missing_reason="missing_cnv_feature_table",
        )
    groups = scoped_candidate_sets(scope, cluster_state, all_cluster_states)
    if len(groups) < 2:
        return tool_result(
            tool_name="cnv_characterization",
            status="failure",
            cluster_id=str(cluster_state.get("cluster_id", "unknown_cluster")),
            output_root=artifact_root,
            summary="CNV comparison requires at least two groups.",
            metrics={"cnv_characterization": []},
            missing_reason="insufficient_cnv_groups",
        )
    threshold = (
        float(tool_parameters(config_dir, "cnv").get("alteration_threshold", 0.2))
        if config_dir
        else 0.2
    )
    rows = []
    all_ids = sorted({case_id for members in groups.values() for case_id in members})
    comparisons = [(group_id, "rest") for group_id in sorted(groups)]
    for feature in feature_names:
        for left, right in comparisons:
            left_ids = sorted(groups[left])
            right_ids = (
                sorted(groups[right])
                if right != "rest"
                else [case_id for case_id in all_ids if case_id not in groups[left]]
            )
            a = [
                table[c][feature]
                for c in left_ids
                if c in table and math.isfinite(table[c].get(feature, float("nan")))
            ]
            b = [
                table[c][feature]
                for c in right_ids
                if c in table and math.isfinite(table[c].get(feature, float("nan")))
            ]
            if not a or not b:
                continue
            tests = [(feature, a, b)]
            if feature.startswith("chr") or feature.startswith("locus::"):
                tests.extend(
                    [
                        (
                            f"{feature}::loss",
                            [float(value <= -threshold) for value in a],
                            [float(value <= -threshold) for value in b],
                        ),
                        (
                            f"{feature}::gain",
                            [float(value >= threshold) for value in a],
                            [float(value >= threshold) for value in b],
                        ),
                    ]
                )
            for tested_feature, left_values, right_values in tests:
                unique = set(left_values + right_values)
                binary = unique.issubset({0.0, 1.0})
                p_value = None
                odds_ratio = None
                odds_ci = None
                left_frequency = None
                right_frequency = None
                left_positive = None
                right_positive = None
                if binary:
                    left_positive = int(sum(value > 0 for value in left_values))
                    right_positive = int(sum(value > 0 for value in right_values))
                    left_frequency = left_positive / len(left_values)
                    right_frequency = right_positive / len(right_values)
                    odds_ratio, p_value = fisher_exact_result(
                        left_positive, len(left_values) - left_positive,
                        right_positive, len(right_values) - right_positive,
                    )
                    odds_ratio, odds_ci = odds_ratio_ci(
                        left_positive, len(left_values) - left_positive,
                        right_positive, len(right_values) - right_positive,
                    )
                elif len(unique) > 1:
                    p_value = float(
                        mannwhitneyu(
                            left_values, right_values, alternative="two-sided"
                        ).pvalue
                    )
                direction = None
                effect = cliffs_delta(left_values, right_values)
                if effect is not None:
                    direction = "higher_in_set" if effect > 0 else "lower_in_set" if effect < 0 else "neutral"
                rows.append({
                    "comparison": f"{left}_vs_{right}",
                    "candidate_set_id": left,
                    "feature": tested_feature,
                    "feature_type": "binary_event" if binary else "continuous",
                    "left_mean": float(np.mean(left_values)),
                    "right_mean": float(np.mean(right_values)),
                    "delta_mean": float(np.mean(left_values) - np.mean(right_values)),
                    "cliffs_delta": effect,
                    "direction": direction,
                    "alteration_frequency": (
                        left_frequency if binary else None
                    ),
                    "rest_alteration_frequency": (
                        right_frequency if binary else None
                    ),
                    "alteration_frequency_difference": (
                        left_frequency - right_frequency
                        if binary and left_frequency is not None and right_frequency is not None
                        else None
                    ),
                    "left_altered_n": left_positive,
                    "left_total_n": len(left_values),
                    "right_altered_n": right_positive,
                    "right_total_n": len(right_values),
                    "odds_ratio": odds_ratio,
                    "odds_ratio_ci95": odds_ci,
                    "p_value": p_value,
                    "q_value": None,
                })
    assign_groupwise_fdr(rows, "candidate_set_id", "p_value", "q_value")
    summaries = {}
    for comparison in sorted({row["comparison"] for row in rows}):
        comparison_rows = [row for row in rows if row["comparison"] == comparison]
        decision = {}
        for feature_type, name, effect_key in (
            ("continuous", "continuous", "cliffs_delta"),
            ("binary_event", "gain_loss", "alteration_frequency_difference"),
        ):
            selected = [row for row in comparison_rows if row["feature_type"] == feature_type]
            core_rows = [
                (
                    {
                        "feature": row["feature"],
                        "cliffs_delta": row["cliffs_delta"],
                        "direction": row["direction"],
                        "q_value": row["q_value"],
                    }
                    if feature_type == "continuous"
                    else {
                        "feature": row["feature"],
                        "alteration_frequency": row["alteration_frequency"],
                        "rest_alteration_frequency": row["rest_alteration_frequency"],
                        "alteration_frequency_difference": row["alteration_frequency_difference"],
                        "odds_ratio": row["odds_ratio"],
                        "odds_ratio_ci95": row["odds_ratio_ci95"],
                        "q_value": row["q_value"],
                        "left_altered_n": row["left_altered_n"],
                        "left_total_n": row["left_total_n"],
                        "right_altered_n": row["right_altered_n"],
                        "right_total_n": row["right_total_n"],
                    }
                )
                for row in selected
            ]
            decision[name] = ranked_decision_summary(core_rows, effect_key)
        summaries[comparison] = decision
    return tool_result(
        tool_name="cnv_characterization",
        status="success",
        cluster_id=str(cluster_state.get("cluster_id", "unknown_cluster")),
        output_root=artifact_root,
        summary=f"CNV characterization computed for {scope}.",
        metrics={"cnv_characterization": rows},
        decision_metrics={"per_comparison_cnv": summaries},
        warnings=[] if rows else ["No comparable CNV features were available."],
    )
