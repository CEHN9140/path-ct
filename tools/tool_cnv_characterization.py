from __future__ import annotations

import math
from itertools import combinations
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
    scoped_candidate_sets,
    tool_result,
)


def tool_cnv_characterization(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
    scope="set_identity",
    target_ids=None,
    proposal=None,
):
    path = Path(output_root) / "cnv" / "case_features.csv"
    feature_names, table = read_case_feature_table(str(path))
    if not feature_names or not table:
        return tool_result(
            tool_name="tool_cnv_characterization",
            status="failure",
            cluster_id=str(cluster_state.get("cluster_id", "unknown_cluster")),
            output_root=output_root,
            summary="CNV feature table is unavailable.",
            metrics={"cnv_characterization": []},
            missing_reason="missing_cnv_feature_table",
        )
    groups = scoped_candidate_sets(scope, cluster_state, all_cluster_states, proposal)
    if len(groups) < 2:
        return tool_result(
            tool_name="tool_cnv_characterization",
            status="failure",
            cluster_id=str(cluster_state.get("cluster_id", "unknown_cluster")),
            output_root=output_root,
            summary="CNV comparison requires at least two groups.",
            metrics={"cnv_characterization": []},
            missing_reason="insufficient_cnv_groups",
        )
    rows = []
    all_ids = sorted({case_id for members in groups.values() for case_id in members})
    if scope == "merge_proposal":
        comparisons = list(combinations(sorted(groups), 2))
    else:
        comparisons = [(group_id, "rest") for group_id in sorted(groups)]
    for feature in feature_names:
        for left, right in comparisons:
            left_ids = sorted(groups[left])
            right_ids = (
                sorted(groups[right])
                if right != "rest"
                else [case_id for case_id in all_ids if case_id not in groups[left]]
            )
            a = [table[c][feature] for c in left_ids if c in table and math.isfinite(table[c].get(feature, float("nan")))]
            b = [table[c][feature] for c in right_ids if c in table and math.isfinite(table[c].get(feature, float("nan")))]
            if not a or not b:
                continue
            unique = set(a + b)
            p_value = None
            odds_ratio = None
            if unique.issubset({0.0, 1.0}):
                a_pos, a_neg = sum(x > 0 for x in a), sum(x == 0 for x in a)
                b_pos, b_neg = sum(x > 0 for x in b), sum(x == 0 for x in b)
                odds_ratio, p_value = fisher_exact_result(a_pos, a_neg, b_pos, b_neg)
                odds_ratio, odds_ci = odds_ratio_ci(a_pos, a_neg, b_pos, b_neg)
            elif len(unique) > 1:
                p_value = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            rows.append({
                "comparison": f"{left}_vs_{right}",
                "feature": feature,
                "feature_type": "binary" if unique.issubset({0.0, 1.0}) else "continuous",
                "left_mean": float(np.mean(a)),
                "right_mean": float(np.mean(b)),
                "delta_mean": float(np.mean(a) - np.mean(b)),
                "cliffs_delta": cliffs_delta(a, b),
                "odds_ratio": odds_ratio,
                "odds_ratio_ci95": odds_ci if unique.issubset({0.0, 1.0}) else None,
                "p_value": p_value,
                "q_value": None,
            })
    assign_groupwise_fdr(rows, "comparison", "p_value", "q_value")
    return tool_result(
        tool_name="tool_cnv_characterization",
        status="success",
        cluster_id=str(cluster_state.get("cluster_id", "unknown_cluster")),
        output_root=output_root,
        summary=f"CNV characterization computed for {scope}.",
        metrics={"cnv_characterization": rows},
        decision_metrics={"cnv_characterization": rows[:100]},
        warnings=[] if rows else ["No comparable CNV features were available."],
    )
