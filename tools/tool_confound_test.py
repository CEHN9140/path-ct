from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu

from tools.subtype_review_common import (
    bh_fdr,
    clinical_table,
    tool_result,
)

CATEGORICAL_FIELDS = ("gender", "race", "ct_manufacturer")
NUMERIC_FIELDS = ("age_at_index", "year_of_diagnosis")
CONFOUNDER_FIELDS = CATEGORICAL_FIELDS + NUMERIC_FIELDS


def round_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return round(number, 6)


def cramers_v(contingency: Any) -> float:
    table = np.asarray(contingency, dtype=float)
    if table.size == 0 or table.sum() <= 0:
        return 0.0
    chi2 = float(chi2_contingency(table, correction=False).statistic)
    n = float(table.sum())
    rows, cols = table.shape
    denom = n * max(min(rows - 1, cols - 1), 1)
    return math.sqrt(chi2 / denom) if denom else 0.0


def normalized_manufacturer(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "PHILIPS"
    return upper


def selected_ct_manufacturer(case_id: str, patient_state: Mapping[str, Any], output_root: str) -> str:
    qc_dir = Path(output_root) / "ct_qc" / str(case_id)
    selected_ct_id = ""
    selected_source_file = ""
    selection_path = qc_dir / "selection_summary.json"
    if selection_path.exists():
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selected_files = list(selection.get("selected_files", []) or [])
            selected = dict(selected_files[0] if selected_files else {})
            selected_ct_id = str(selected.get("selected_ct_id", "") or "")
            selected_source_file = str(selected.get("selected_source_file", "") or "")
        except Exception:
            selected_ct_id = ""
            selected_source_file = ""
    if selected_ct_id:
        sidecar_path = qc_dir / "dcm2nii" / f"{selected_ct_id}.json"
        if sidecar_path.exists():
            try:
                value = json.loads(sidecar_path.read_text(encoding="utf-8")).get("Manufacturer", "")
                manufacturer = normalized_manufacturer(value)
                if manufacturer:
                    return manufacturer
            except Exception:
                pass
    ct_entries = [
        dict(item)
        for item in list(dict(patient_state.get("inventory", {}) or {}).get("CT", []) or [])
    ]
    if selected_source_file:
        for item in ct_entries:
            if str(item.get("File Path", "") or "") == selected_source_file:
                manufacturer = normalized_manufacturer(item.get("Manufacturer", ""))
                if manufacturer:
                    return manufacturer
    for item in ct_entries:
        manufacturer = normalized_manufacturer(item.get("Manufacturer", ""))
        if manufacturer:
            return manufacturer
    return ""


def candidate_set_members(cluster_state: Mapping[str, Any], all_cluster_states: Any) -> dict[str, list[str]]:
    clusters = list(all_cluster_states or [])
    if not clusters:
        clusters = [cluster_state]
    memberships: dict[str, list[str]] = {}
    assigned = set()
    for index, cluster in enumerate(clusters):
        item = dict(cluster or {})
        cluster_id = str(item.get("cluster_id") or item.get("candidate_set_id") or f"C{index + 1}")
        members = []
        for case_id in list(item.get("member_ids", []) or []):
            case_key = str(case_id)
            if case_key not in assigned:
                members.append(case_key)
                assigned.add(case_key)
        if members:
            memberships[cluster_id] = members
    return memberships


def confounder_values(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
) -> dict[str, dict[str, Any]]:
    clinical = clinical_table(patient_states_by_id)
    values: dict[str, dict[str, Any]] = {}
    for case_id, patient_state in patient_states_by_id.items():
        record = dict(clinical.get(str(case_id), {}) or {})
        values[str(case_id)] = {
            "gender": str(record.get("gender", "") or "").strip(),
            "race": str(record.get("race", "") or "").strip(),
            "ct_manufacturer": selected_ct_manufacturer(str(case_id), patient_state, output_root),
            "age_at_index": record.get("age"),
            "year_of_diagnosis": record.get("year_of_diagnosis"),
        }
    return values


def field_case_values(
    memberships: Mapping[str, list[str]],
    values: Mapping[str, Mapping[str, Any]],
    field: str,
    numeric: bool = False,
) -> tuple[dict[str, list[tuple[str, Any]]], int, int]:
    total = sum(len(members) for members in memberships.values())
    per_set: dict[str, list[tuple[str, Any]]] = {}
    available = 0
    for set_id, members in memberships.items():
        rows = []
        for case_id in members:
            value = dict(values.get(case_id, {}) or {}).get(field)
            if numeric:
                try:
                    value = None if value is None or value == "" else float(value)
                except (TypeError, ValueError):
                    value = None
            else:
                value = str(value or "").strip()
            if value is not None and value != "":
                rows.append((case_id, value))
                available += 1
        per_set[set_id] = rows
    return per_set, available, total - available


def global_categorical(field: str, memberships: Mapping[str, list[str]], values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_set, available_n, missing_n = field_case_values(memberships, values, field)
    levels = sorted({str(value) for rows in per_set.values() for _, value in rows})
    table_dict = {
        set_id: {level: sum(1 for _, value in rows if value == level) for level in levels}
        for set_id, rows in per_set.items()
    }
    table = np.asarray([[table_dict[set_id][level] for level in levels] for set_id in per_set], dtype=int)
    p_value = None
    low_expected = False
    effect = 0.0
    if table.size and table.shape[0] > 1 and table.shape[1] > 1 and table.sum() > 0:
        chi2 = chi2_contingency(table, correction=False)
        p_value = float(chi2.pvalue)
        low_expected = bool(np.any(chi2.expected_freq < 5))
        effect = cramers_v(table)
    return {
        "field": field,
        "field_type": "categorical",
        "available_n": available_n,
        "missing_n": missing_n,
        "contingency_table": table_dict,
        "chi_square_p_value": round_value(p_value),
        "q_value": None,
        "low_expected_count": low_expected,
        "cramers_v": round_value(effect),
    }


def max_pairwise_smd(per_set: Mapping[str, list[tuple[str, Any]]]) -> Any:
    values = {set_id: [float(value) for _, value in rows] for set_id, rows in per_set.items()}
    effects = []
    for left, right in combinations(values, 2):
        if values[left] and values[right]:
            effect = smd_or_none(values[left], values[right])
            if effect is not None:
                effects.append(abs(float(effect)))
    return max(effects) if effects else None


def smd_or_none(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    array_a = np.asarray(values_a, dtype=float)
    array_b = np.asarray(values_b, dtype=float)
    var_a = float(np.var(array_a, ddof=1))
    var_b = float(np.var(array_b, ddof=1))
    denom = len(array_a) + len(array_b) - 2
    if denom <= 0:
        return None
    pooled_sd = math.sqrt(
        ((len(array_a) - 1) * var_a + (len(array_b) - 1) * var_b) / denom
    )
    if not math.isfinite(pooled_sd) or pooled_sd == 0.0:
        return None
    return float((np.mean(array_a) - np.mean(array_b)) / pooled_sd)


def global_numeric(field: str, memberships: Mapping[str, list[str]], values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_set, available_n, missing_n = field_case_values(memberships, values, field, numeric=True)
    set_values = {set_id: [float(value) for _, value in rows] for set_id, rows in per_set.items()}
    groups = [items for items in set_values.values() if items]
    p_value = None
    if len(groups) > 1 and len(set(value for items in groups for value in items)) > 1:
        p_value = float(kruskal(*groups).pvalue)
    return {
        "field": field,
        "field_type": "numeric",
        "available_n": available_n,
        "missing_n": missing_n,
        "per_set_mean": {set_id: round_value(np.mean(items)) if items else None for set_id, items in set_values.items()},
        "per_set_median": {set_id: round_value(np.median(items)) if items else None for set_id, items in set_values.items()},
        "kruskal_p_value": round_value(p_value),
        "q_value": None,
        "max_pairwise_smd": round_value(max_pairwise_smd(per_set)),
    }


def set_categorical(
    set_id: str,
    field: str,
    memberships: Mapping[str, list[str]],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    per_set, available_n, missing_n = field_case_values(memberships, values, field)
    member_values = [str(value) for _, value in per_set.get(set_id, [])]
    rest_values = [
        str(value)
        for other_id, rows in per_set.items()
        if other_id != set_id
        for _, value in rows
    ]
    levels = sorted(set(member_values + rest_values))
    member_counts = Counter(member_values)
    rest_counts = Counter(rest_values)
    set_total = len(member_values)
    rest_total = len(rest_values)
    rows: dict[str, dict[str, Any]] = {}
    for level in levels:
        set_count = member_counts[level]
        rest_count = rest_counts[level]
        set_fraction = set_count / set_total if set_total else 0.0
        rest_fraction = rest_count / rest_total if rest_total else 0.0
        p_value = None
        odds_ratio = None
        sparse = False
        if set_total and rest_total:
            table = [[set_count, set_total - set_count], [rest_count, rest_total - rest_count]]
            fisher = fisher_exact(table)
            odds_ratio = float(fisher.statistic)
            p_value = float(fisher.pvalue)
            sparse = any(cell < 5 for row in table for cell in row) or (set_count + rest_count) < 5
        rows[level] = {
            "candidate_set_id": set_id,
            "field": field,
            "field_type": "categorical",
            "level": level,
            "available_n": available_n,
            "missing_n": missing_n,
            "set_count": int(set_count),
            "set_total": int(set_total),
            "rest_count": int(rest_count),
            "rest_total": int(rest_total),
            "level_total_n": int(set_count + rest_count),
            "set_fraction": round_value(set_fraction),
            "rest_fraction": round_value(rest_fraction),
            "delta_fraction": round_value(set_fraction - rest_fraction),
            "odds_ratio": round_value(odds_ratio),
            "p_value": round_value(p_value),
            "q_value": None,
            "sparse_level": bool(sparse),
        }
    return rows


def set_numeric(
    set_id: str,
    field: str,
    memberships: Mapping[str, list[str]],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    per_set, available_n, missing_n = field_case_values(memberships, values, field, numeric=True)
    member_values = [float(value) for _, value in per_set.get(set_id, [])]
    rest_values = [
        float(value)
        for other_id, rows in per_set.items()
        if other_id != set_id
        for _, value in rows
    ]
    smd = smd_or_none(member_values, rest_values)
    p_value = None
    if member_values and rest_values and len(set(member_values + rest_values)) > 1:
        p_value = float(mannwhitneyu(member_values, rest_values, alternative="two-sided").pvalue)
    direction = None
    if smd is not None:
        if smd > 0:
            direction = "higher_in_set"
        elif smd < 0:
            direction = "lower_in_set"
    return {
        "candidate_set_id": set_id,
        "field": field,
        "field_type": "numeric",
        "available_n": available_n,
        "missing_n": missing_n,
        "set_mean": round_value(np.mean(member_values)) if member_values else None,
        "rest_mean": round_value(np.mean(rest_values)) if rest_values else None,
        "set_median": round_value(np.median(member_values)) if member_values else None,
        "rest_median": round_value(np.median(rest_values)) if rest_values else None,
        "delta_mean": round_value((np.mean(member_values) - np.mean(rest_values)) if member_values and rest_values else None),
        "standardized_mean_difference": round_value(smd),
        "mannwhitney_p_value": round_value(p_value),
        "q_value": None,
        "direction": direction,
    }


def assign_q_values(global_metrics: dict[str, dict[str, Any]], set_metrics: dict[str, dict[str, dict[str, Any]]]) -> None:
    global_refs = []
    global_p_values = []
    for field, row in global_metrics.items():
        p_value = row.get("chi_square_p_value") if field in CATEGORICAL_FIELDS else row.get("kruskal_p_value")
        if p_value is not None:
            global_refs.append(row)
            global_p_values.append(p_value)
    if global_p_values:
        for row, q_value in zip(global_refs, bh_fdr(global_p_values)):
            row["q_value"] = round_value(q_value)

    set_refs = []
    p_values = []
    for rows in set_metrics.values():
        for row in rows.values():
            if isinstance(row, Mapping) and row.get("field_type") == "numeric":
                p_value = row.get("mannwhitney_p_value")
                if p_value is not None:
                    set_refs.append(row)
                    p_values.append(p_value)
            elif isinstance(row, Mapping):
                for level_row in row.values():
                    if not isinstance(level_row, Mapping):
                        continue
                    p_value = level_row.get("p_value")
                    if p_value is not None:
                        set_refs.append(level_row)
                        p_values.append(p_value)
    if p_values:
        for row, q_value in zip(set_refs, bh_fdr(p_values)):
            row["q_value"] = round_value(q_value)


def tool_confound_test(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    memberships = candidate_set_members(cluster_state, all_cluster_states)
    if not memberships:
        return tool_result(
            tool_name="tool_confound_test",
            status="missing",
            cluster_id=cluster_id,
            output_root=output_root,
            summary="No candidate-set cases are available for confound testing.",
            missing_reason="empty candidate sets",
            support_level="none",
            concern_level="high",
        )

    values = confounder_values(patient_states_by_id, output_root)
    global_metrics = {
        field: global_categorical(field, memberships, values)
        for field in CATEGORICAL_FIELDS
    }
    global_metrics.update(
        {field: global_numeric(field, memberships, values) for field in NUMERIC_FIELDS}
    )
    set_metrics = {
        set_id: {
            **{
                field: set_categorical(set_id, field, memberships, values)
                for field in CATEGORICAL_FIELDS
            },
            **{
                field: set_numeric(set_id, field, memberships, values)
                for field in NUMERIC_FIELDS
            },
        }
        for set_id in memberships
    }
    assign_q_values(global_metrics, set_metrics)

    return tool_result(
        tool_name="tool_confound_test",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary=f"Confounder association metrics were computed for {len(memberships)} candidate sets.",
        metrics={
            "confounder_global_association": global_metrics,
            "confounder_set_association": set_metrics,
        },
        evidence_hints=[
            {
                "evidence_type": "confounder",
                "summary": "Full global and per-set confounder metrics were computed.",
            }
        ],
        support_level="informational",
        concern_level="low",
        figures={},
    )
