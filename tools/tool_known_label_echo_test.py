from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from tools.subtype_review_common import (
    bh_fdr,
    clinical_table,
    member_case_ids,
    tool_result,
)


KNOWN_LABEL_FIELDS = {"stage": "stage_group", "grade": "grade"}
MISSING_LABELS = {"", "not reported", "unknown", "not applicable", "nan", "none"}


def valid_label(value):
    return str(value or "").strip().lower() not in MISSING_LABELS


def cramers_v_from_table(table):
    table = np.asarray(table, dtype=float)
    if table.size == 0 or table.sum() <= 0:
        return None
    chi2 = float(chi2_contingency(table, correction=False).statistic)
    denom = float(table.sum()) * max(min(table.shape[0] - 1, table.shape[1] - 1), 1)
    return round(float(math.sqrt(chi2 / denom)), 6) if denom else None


def finite_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return None
    return round(value, 6)


def cluster_labels_by_case(all_cluster_states, fallback_cluster_state):
    labels = {}
    states = list(all_cluster_states or []) or [fallback_cluster_state]
    for state in states:
        cluster_id = str(state.get("cluster_id", "") or "")
        for case_id in member_case_ids(state):
            labels[str(case_id)] = cluster_id
    return labels


def global_label_redundancy(label_name, clinical_field, cluster_labels, clinical):
    case_ids = [
        case_id
        for case_id in cluster_labels
        if valid_label(clinical.get(case_id, {}).get(clinical_field, ""))
    ]
    missing_n = len(cluster_labels) - len(case_ids)
    if len(case_ids) < 2:
        return {
            "label": label_name,
            "available_n": len(case_ids),
            "missing_n": missing_n,
            "contingency_table": {},
            "chi_square_p_value": None,
            "low_expected_count": None,
            "cramers_v": None,
        }
    set_levels = sorted({cluster_labels[case_id] for case_id in case_ids})
    known_levels = sorted(
        {str(clinical[case_id].get(clinical_field, "") or "") for case_id in case_ids}
    )
    table = np.asarray(
        [
            [
                sum(
                    cluster_labels[case_id] == set_level
                    and str(clinical[case_id].get(clinical_field, "") or "")
                    == known_level
                    for case_id in case_ids
                )
                for known_level in known_levels
            ]
            for set_level in set_levels
        ],
        dtype=int,
    )
    contingency = {
        set_level: {
            known_level: int(table[row_idx, col_idx])
            for col_idx, known_level in enumerate(known_levels)
        }
        for row_idx, set_level in enumerate(set_levels)
    }
    if min(table.shape) > 1:
        chi2 = chi2_contingency(table, correction=False)
        chi_square_p_value = round(float(chi2.pvalue), 6)
        low_expected_count = bool(np.any(chi2.expected_freq < 5))
    else:
        chi_square_p_value = None
        low_expected_count = None
    return {
        "label": label_name,
        "available_n": len(case_ids),
        "missing_n": missing_n,
        "contingency_table": contingency,
        "chi_square_p_value": chi_square_p_value,
        "low_expected_count": low_expected_count,
        "cramers_v": cramers_v_from_table(table),
    }


def set_label_redundancy(label_name, clinical_field, cluster_id, cluster_labels, clinical):
    case_ids = [
        case_id
        for case_id in cluster_labels
        if valid_label(clinical.get(case_id, {}).get(clinical_field, ""))
    ]
    member_ids = [
        case_id for case_id in case_ids if cluster_labels[case_id] == cluster_id
    ]
    raw_member_ids = [
        case_id for case_id, set_label in cluster_labels.items() if set_label == cluster_id
    ]
    missing_n = len(raw_member_ids) - len(member_ids)
    set_available_n = len(member_ids)
    if not case_ids or not member_ids:
        return {
            "candidate_set_id": cluster_id,
            "label": label_name,
            "available_n": set_available_n,
            "missing_n": missing_n,
            "dominant_level": None,
            "overlap_count": 0,
            "set_available_n": set_available_n,
            "level_total_n": 0,
            "set_fraction": None,
            "level_recall": None,
            "odds_ratio": None,
            "p_value": None,
            "q_value": None,
        }
    level_counts = {
        value: sum(
            str(clinical[case_id].get(clinical_field, "") or "") == value
            for case_id in member_ids
        )
        for value in sorted(
            {
                str(clinical[case_id].get(clinical_field, "") or "")
                for case_id in case_ids
            }
        )
    }
    dominant_level, overlap_count = max(
        level_counts.items(), key=lambda item: (item[1], item[0])
    )
    level_total_n = sum(
        str(clinical[case_id].get(clinical_field, "") or "") == dominant_level
        for case_id in case_ids
    )
    rest_n = len(case_ids) - set_available_n
    rest_level_n = level_total_n - overlap_count
    table = [
        [overlap_count, set_available_n - overlap_count],
        [rest_level_n, rest_n - rest_level_n],
    ]
    try:
        fisher = fisher_exact(table)
        odds_ratio = finite_float(fisher.statistic)
        p_value = round(float(fisher.pvalue), 6)
    except Exception:
        odds_ratio = None
        p_value = None
    return {
        "candidate_set_id": cluster_id,
        "label": label_name,
        "available_n": set_available_n,
        "missing_n": missing_n,
        "dominant_level": dominant_level,
        "overlap_count": int(overlap_count),
        "set_available_n": set_available_n,
        "level_total_n": int(level_total_n),
        "set_fraction": (
            round(float(overlap_count / set_available_n), 6)
            if set_available_n
            else None
        ),
        "level_recall": (
            round(float(overlap_count / level_total_n), 6) if level_total_n else None
        ),
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "q_value": None,
    }


def all_set_label_redundancy(label_name, clinical_field, cluster_labels, clinical):
    rows = {
        cluster_id: set_label_redundancy(
            label_name, clinical_field, cluster_id, cluster_labels, clinical
        )
        for cluster_id in sorted(set(cluster_labels.values()))
    }
    p_values = [row["p_value"] for row in rows.values() if row["p_value"] is not None]
    q_values = bh_fdr(p_values)
    for row in rows.values():
        if row["p_value"] is not None:
            row["q_value"] = round(float(q_values.pop(0)), 6)
    return rows


def tool_known_label_echo_test(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    clinical = clinical_table(patient_states_by_id)
    cluster_labels = cluster_labels_by_case(all_cluster_states, cluster_state)
    global_association = {}
    set_enrichment = {
        set_id: {} for set_id in sorted(set(cluster_labels.values()))
    }
    for label_name, clinical_field in KNOWN_LABEL_FIELDS.items():
        global_association[label_name] = global_label_redundancy(
            label_name, clinical_field, cluster_labels, clinical
        )
        label_rows = all_set_label_redundancy(
            label_name, clinical_field, cluster_labels, clinical
        )
        for set_id, row in label_rows.items():
            set_enrichment.setdefault(set_id, {})[label_name] = row
    return tool_result(
        tool_name="tool_known_label_echo_test",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary=f"{cluster_id} known-label stage/grade association and set enrichment were summarized.",
        metrics={
            "known_label_global_association": global_association,
            "known_label_set_enrichment": set_enrichment,
        },
        evidence_hints=[
            {
                "evidence_type": "known_label_echo",
                "summary": f"{cluster_id} stage and grade redundancy was evaluated.",
            }
        ],
        support_level="informational",
        concern_level="low",
        figures={},
    )
