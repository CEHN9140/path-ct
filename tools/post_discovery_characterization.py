"""Shared post-discovery statistics for fixed stable groups.

This module never changes memberships. It only analyzes labels supplied by a
caller on the explicitly supplied stable-group universe.
"""

from __future__ import annotations

from itertools import combinations
from math import exp, log, sqrt
from typing import Mapping, Sequence

import numpy as np


def stable_analysis_universe(groups: Mapping[str, Sequence[str]]) -> list[str]:
    flat = [case_id for members in groups.values() for case_id in members]
    if len(flat) != len(set(flat)):
        raise ValueError("Groups overlap in characterization universe")
    return sorted(set(flat))


def bh_adjust(values: Sequence[float | None]) -> list[float | None]:
    valid = sorted(
        ((index, float(value)) for index, value in enumerate(values) if value is not None and np.isfinite(value)),
        key=lambda item: item[1],
    )
    result = [None] * len(values)
    running = 1.0
    for rank, (index, value) in reversed(list(enumerate(valid, 1))):
        running = min(running, value * len(valid) / rank)
        result[index] = min(running, 1.0)
    return result


def holm_adjust(values: Sequence[float | None]) -> list[float | None]:
    valid = sorted(
        ((index, float(value)) for index, value in enumerate(values) if value is not None and np.isfinite(value)),
        key=lambda item: item[1],
    )
    result = [None] * len(values)
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * value))
        result[index] = running
    return result


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) == 0 or len(right) == 0:
        return None
    a, b = np.asarray(left, float), np.asarray(right, float)
    return float((np.greater.outer(a, b).sum() - np.less.outer(a, b).sum()) / (len(a) * len(b)))


def bootstrap_cliffs_delta_ci(left, right, iterations=2000, seed=20260908):
    observed = cliffs_delta(left, right)
    if observed is None or len(left) == 0 or len(right) == 0 or iterations <= 0:
        return observed, None, None
    rng = np.random.default_rng(seed)
    a, b = np.asarray(left, float), np.asarray(right, float)
    a_indices = rng.integers(0, len(a), size=(iterations, len(a)))
    b_indices = rng.integers(0, len(b), size=(iterations, len(b)))
    a_counts = np.eye(len(a), dtype=float)[a_indices].sum(axis=1)
    b_counts = np.eye(len(b), dtype=float)[b_indices].sum(axis=1)
    comparison = np.sign(a[:, None] - b[None, :])
    values = np.einsum("ia,ab,ib->i", a_counts, comparison, b_counts) / (len(a) * len(b))
    return observed, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def _values(table, ids, feature):
    return [float(table[case_id][feature]) for case_id in ids if table.get(case_id, {}).get(feature) is not None and np.isfinite(float(table[case_id][feature]))]


def _binary_value(table, case_id, feature):
    value = table.get(case_id, {}).get(feature)
    if value is None or str(value).strip().upper() in {"", "NA", "N/A", "UNKNOWN"}:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return int(value > 0) if np.isfinite(value) else None


def _rounded(value):
    return round(float(value), 8) if value is not None and np.isfinite(value) else None


def continuous_omnibus(table, features, groups):
    from scipy.stats import kruskal

    rows = []
    for feature in features:
        samples = [_values(table, members, feature) for members in groups.values()]
        total_n = sum(map(len, samples))
        row = {"feature": feature, "group_count": len(groups), "available_n": total_n, "group_sizes": ";".join(map(str, map(len, samples))), "comparison_status": "not_estimable", "not_estimable_reason": "", "kruskal_h": None, "p_value": None, "q_value": None, "epsilon_squared": None}
        if any(not sample for sample in samples):
            row["not_estimable_reason"] = "one_or_more_groups_have_no_valid_values"
        elif len(samples) >= 2 and total_n > len(samples):
            try:
                statistic, p_value = kruskal(*samples)
                row["comparison_status"] = "estimable"
                row["kruskal_h"] = _rounded(statistic)
                row["p_value"] = _rounded(p_value)
                row["epsilon_squared"] = _rounded(max(0.0, min(1.0, (statistic - len(samples) + 1) / (total_n - len(samples)))))
            except ValueError:
                row["not_estimable_reason"] = "kruskal_wallis_failed"
        rows.append(row)
    for row, q_value in zip(rows, bh_adjust([row["p_value"] for row in rows])):
        row["q_value"] = _rounded(q_value)
    return rows


def continuous_posthoc(table, features, groups, bootstrap_iterations=2000, seed=20260908):
    from scipy.stats import mannwhitneyu

    rows = []
    for feature_index, feature in enumerate(features):
        current = []
        for group_a, group_b in combinations(groups, 2):
            left, right = _values(table, groups[group_a], feature), _values(table, groups[group_b], feature)
            delta, low, high = bootstrap_cliffs_delta_ci(left, right, bootstrap_iterations, seed + feature_index)
            p_value = None
            if left and right:
                p_value = float(mannwhitneyu(left, right, alternative="two-sided", method="auto").pvalue)
            current.append({"feature": feature, "group_a": group_a, "group_b": group_b, "n_a": len(left), "n_b": len(right), "median_a": _rounded(np.median(left)) if left else None, "median_b": _rounded(np.median(right)) if right else None, "mean_a": _rounded(np.mean(left)) if left else None, "mean_b": _rounded(np.mean(right)) if right else None, "cliffs_delta": _rounded(delta), "cliffs_delta_ci_low": _rounded(low), "cliffs_delta_ci_high": _rounded(high), "p_value": _rounded(p_value), "p_adjusted_holm": None})
        for row, adjusted in zip(current, holm_adjust([row["p_value"] for row in current])):
            row["p_adjusted_holm"] = _rounded(adjusted)
        rows.extend(current)
    return rows


def continuous_state_vs_rest_descriptive(table, features, groups, bootstrap_iterations=2000, seed=20260908):
    from scipy.stats import mannwhitneyu

    rows = []
    universe = stable_analysis_universe(groups)
    for group, members in groups.items():
        rest = [case_id for case_id in universe if case_id not in members]
        for feature in features:
            left, right = _values(table, members, feature), _values(table, rest, feature)
            p_value = float(mannwhitneyu(left, right, alternative="two-sided", method="auto").pvalue) if left and right else None
            delta, low, high = bootstrap_cliffs_delta_ci(left, right, bootstrap_iterations, seed)
            rows.append({"group_id": group, "feature": feature, "n_group": len(left), "n_rest": len(right), "median_group": _rounded(np.median(left)) if left else None, "median_rest": _rounded(np.median(right)) if right else None, "cliffs_delta": _rounded(delta), "cliffs_delta_ci_low": _rounded(low), "cliffs_delta_ci_high": _rounded(high), "p_value": _rounded(p_value), "analysis_role": "descriptive_secondary"})
    return rows


def _odds_ratio_ci(a, b, c, d):
    odds = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    se = sqrt(sum(1 / (value + .5) for value in (a, b, c, d)))
    return odds, exp(log(odds) - 1.96 * se), exp(log(odds) + 1.96 * se)


def _chi_square(table):
    from scipy.stats import chi2_contingency

    table = np.asarray(table, dtype=float)
    table = table[table.sum(axis=1) > 0]
    table = table[:, table.sum(axis=0) > 0] if table.size else table
    return 0.0 if table.shape[0] < 2 or table.shape[1] < 2 else float(chi2_contingency(table, correction=False)[0])


def binary_permutation_omnibus(table, features, groups, permutations=9999, seed=20260908):
    case_ids = [case_id for members in groups.values() for case_id in members]
    rows = []
    for feature_index, feature in enumerate(features):
        values = [_binary_value(table, case_id, feature) for case_id in case_ids]
        valid = np.asarray([value is not None for value in values])
        values = np.asarray([value for value in values if value is not None], dtype=float)
        valid_ids = [case_id for case_id, keep in zip(case_ids, valid) if keep]
        labels = np.asarray([next(group for group, members in groups.items() if case_id in members) for case_id in valid_ids])
        group_names = [group for group in groups if np.any(labels == group)]
        group_sizes = np.asarray([int(np.sum(labels == group)) for group in group_names], dtype=float)
        group_mask = np.asarray([[int(label == group) for label in labels] for group in group_names], dtype=float)
        observed_mutated = values @ group_mask.T
        contingency = np.column_stack((observed_mutated, group_sizes - observed_mutated)).astype(int)
        statistic = _chi_square(contingency) if len(group_names) > 1 else None
        p_value = None
        status = "estimable" if statistic is not None else "not_estimable"
        if statistic is not None:
            rng = np.random.default_rng(seed + feature_index)
            shuffled = values[rng.random((permutations, len(values))).argsort(axis=1)]
            mutated = shuffled @ group_mask.T
            total_mutated = float(values.sum())
            expected = total_mutated * group_sizes / len(values)
            with np.errstate(divide="ignore", invalid="ignore"):
                chi = ((mutated - expected) ** 2 / expected +
                       ((group_sizes - mutated) - (group_sizes - expected)) ** 2 /
                       (group_sizes - expected)).sum(axis=1)
            chi = np.nan_to_num(chi, nan=0.0, posinf=0.0, neginf=0.0)
            exceed = int(np.count_nonzero(chi >= statistic))
            p_value = (exceed + 1) / (permutations + 1)
        rows.append({"feature": feature, "group_count": len(groups), "available_n": len(values), "missing_n": len(case_ids) - len(values), "mutated_n": int(values.sum()), "total_n": len(values), "comparison_status": status, "chi_square": _rounded(statistic), "p_value": _rounded(p_value), "q_value": None, "permutations": permutations})
    for row, q_value in zip(rows, bh_adjust([row["p_value"] for row in rows])):
        row["q_value"] = _rounded(q_value)
    return rows


def binary_posthoc(table, features, groups):
    from scipy.stats import fisher_exact

    rows = []
    for feature in features:
        current = []
        for group_a, group_b in combinations(groups, 2):
            left = [value for case_id in groups[group_a] if (value := _binary_value(table, case_id, feature)) is not None]
            right = [value for case_id in groups[group_b] if (value := _binary_value(table, case_id, feature)) is not None]
            a, b, c, d = sum(left), len(left) - sum(left), sum(right), len(right) - sum(right)
            p_value = float(fisher_exact([[a, b], [c, d]])[1]) if left and right else None
            odds, low, high = _odds_ratio_ci(a, b, c, d) if left and right else (None, None, None)
            current.append({"feature": feature, "group_a": group_a, "group_b": group_b, "n_a": len(left), "n_b": len(right), "missing_n_a": len(groups[group_a]) - len(left), "missing_n_b": len(groups[group_b]) - len(right), "frequency_a": a / len(left) if left else None, "frequency_b": c / len(right) if right else None, "frequency_difference": a / len(left) - c / len(right) if left and right else None, "odds_ratio": _rounded(odds), "ci_low": _rounded(low), "ci_high": _rounded(high), "fisher_p": _rounded(p_value), "p_adjusted_holm": None})
        for row, adjusted in zip(current, holm_adjust([row["fisher_p"] for row in current])):
            row["p_adjusted_holm"] = _rounded(adjusted)
        rows.extend(current)
    return rows


def categorical_omnibus(records, variable, groups, permutations=9999, seed=20260908):
    valid = {case_id: str(records.get(case_id, {}).get(variable, "")).strip() for case_id in stable_analysis_universe(groups)}
    valid = {case_id: value for case_id, value in valid.items() if value and value.upper() not in {"UNKNOWN", "NA", "N/A", "NX", "MX", "TX"}}
    levels = sorted(set(valid.values()))
    labels = [(group, case_id) for group, members in groups.items() for case_id in members if case_id in valid]
    table = np.asarray([[sum(valid[case_id] == level for observed_group, case_id in labels if observed_group == group_id) for level in levels] for group_id in groups])
    row = {"clinical_variable": variable, "group_count": len(groups), "available_n": len(labels), "levels": ";".join(levels), "chi_square": None, "p_value": None, "q_value": None, "permutations": permutations}
    if len(levels) > 1 and table.shape[0] > 1:
        statistic = _chi_square(table)
        group_labels = np.asarray([group for group, _ in labels]); values = np.asarray([valid[case_id] for _, case_id in labels]); rng = np.random.default_rng(seed); exceed = 0
        for _ in range(permutations):
            shuffled = rng.permutation(values)
            perm = np.asarray([[sum(shuffled[group_labels == group] == level) for level in levels] for group in groups])
            if _chi_square(perm) >= statistic:
                exceed += 1
        row.update({"chi_square": _rounded(statistic), "p_value": _rounded((exceed + 1) / (permutations + 1))})
    return row


def categorical_posthoc(records, variable, groups, permutations=9999, seed=20260908):
    rows = []
    for pair_index, (group_a, group_b) in enumerate(combinations(groups, 2)):
        values = {case_id: str(records.get(case_id, {}).get(variable, "")).strip() for case_id in groups[group_a] + groups[group_b]}
        values = {case_id: value for case_id, value in values.items() if value and value.upper() not in {"UNKNOWN", "NA", "N/A", "NX", "MX", "TX"}}
        left_ids = [case_id for case_id in groups[group_a] if case_id in values]
        right_ids = [case_id for case_id in groups[group_b] if case_id in values]
        levels = sorted(set(values.values()))
        observed_values = np.asarray([values[case_id] for case_id in left_ids + right_ids])
        n_left = len(left_ids)
        table = np.asarray([[sum(values.get(case_id) == level for case_id in groups[group_a]) for level in levels], [sum(values.get(case_id) == level for case_id in groups[group_b]) for level in levels]])
        statistic = _chi_square(table) if len(levels) > 1 and n_left and len(right_ids) else None
        p_value = None
        if statistic is not None:
            rng = np.random.default_rng(seed + pair_index)
            exceed = 0
            for _ in range(permutations):
                shuffled = rng.permutation(observed_values)
                perm = np.asarray([[np.sum(shuffled[:n_left] == level), np.sum(shuffled[n_left:] == level)] for level in levels]).T
                if _chi_square(perm) >= statistic:
                    exceed += 1
            p_value = (exceed + 1) / (permutations + 1)
        rows.append({"clinical_variable": variable, "group_a": group_a, "group_b": group_b, "n_a": n_left, "n_b": len(right_ids), "missing_n_a": len(groups[group_a]) - n_left, "missing_n_b": len(groups[group_b]) - len(right_ids), "levels": ";".join(levels), "chi_square": _rounded(statistic), "p_value": _rounded(p_value), "p_adjusted_holm": None, "permutations": permutations})
    for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
        row["p_adjusted_holm"] = _rounded(adjusted)
    return rows


def stage_adjusted_survival(records, groups):
    import pandas as pd
    from lifelines import CoxPHFitter

    stage_values = {"I": 1, "II": 2, "III": 3, "IV": 4}
    rows = []
    for group, members in groups.items():
        for case_id in members:
            record = records.get(case_id, {})
            stage = str(record.get("stage_group") or "").strip().upper()
            try:
                os_time = float(record.get("os_time"))
            except (TypeError, ValueError):
                continue
            if stage not in stage_values or not np.isfinite(os_time):
                continue
            rows.append({"case_id": case_id, "state": group, "stage_number": stage_values[stage], "os_time": os_time, "os_event": int(record.get("os_event") or 0)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return [{"model": "state_plus_stage", "status": "not_estimable", "reason": "no_usable_survival_records", "analysis_role": "secondary_exploratory"}]
    frame = pd.get_dummies(frame.drop(columns="case_id"), columns=["state"], dtype=float)
    state_columns = [f"state_{group}" for group in sorted(groups) if f"state_{group}" in frame]
    if len(frame) < 10 or frame["os_event"].sum() < 3 or len(state_columns) < 2:
        return [{"model": "state_plus_stage", "status": "not_estimable", "reason": "insufficient_events_or_covariates", "available_n": len(frame), "events": int(frame["os_event"].sum()), "analysis_role": "secondary_exploratory"}]
    reference = state_columns[0]
    frame = frame.drop(columns=reference)
    model = CoxPHFitter(penalizer=.1).fit(frame, duration_col="os_time", event_col="os_event")
    output = []
    for covariate in ["stage_number", *sorted(set(state_columns) - {reference})]:
        interval = model.confidence_intervals_.loc[covariate].to_numpy()
        output.append({"model": "state_plus_stage", "covariate": covariate, "status": "estimable", "hazard_ratio": _rounded(np.exp(model.params_[covariate])), "ci_low": _rounded(np.exp(interval[0])), "ci_high": _rounded(np.exp(interval[1])), "p_value": _rounded(model.summary.loc[covariate, "p"]), "reference_state": reference.removeprefix("state_") if covariate != "stage_number" else None, "adjustment": "stage_group ordinal I=1, II=2, III=3, IV=4", "available_n": len(frame), "events": int(frame["os_event"].sum()), "analysis_role": "secondary_exploratory"})
    return output


def survival_analysis(records, groups):
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    ids = stable_analysis_universe(groups)
    durations = [records[case_id].get("os_time") for case_id in ids if records.get(case_id, {}).get("os_time") is not None]
    usable_ids = [case_id for case_id in ids if records.get(case_id, {}).get("os_time") is not None]
    labels = [next(group for group, members in groups.items() if case_id in members) for case_id in usable_ids]
    durations = [records[case_id]["os_time"] for case_id in usable_ids]
    events = [int(records[case_id].get("os_event") or 0) for case_id in usable_ids]
    global_row = {"analysis": "global_logrank", "group_count": len(groups), "available_n": len(durations), "events": sum(events), "p_value": None, "q_value": None, "analysis_role": "primary"}
    if len(set(labels)) > 1 and sum(events) > 0:
        global_row["p_value"] = _rounded(multivariate_logrank_test(durations, labels, event_observed=events).p_value)
    pairs = []
    for group_a, group_b in combinations(groups, 2):
        a = [case_id for case_id in groups[group_a] if records.get(case_id, {}).get("os_time") is not None]; b = [case_id for case_id in groups[group_b] if records.get(case_id, {}).get("os_time") is not None]
        p_value = logrank_test([records[x]["os_time"] for x in a], [records[x]["os_time"] for x in b], event_observed_A=[int(records[x].get("os_event") or 0) for x in a], event_observed_B=[int(records[x].get("os_event") or 0) for x in b]).p_value if a and b and sum(int(records[x].get("os_event") or 0) for x in a + b) > 0 else None
        pairs.append({"group_a": group_a, "group_b": group_b, "n_a": len(a), "n_b": len(b), "events_a": sum(int(records[x].get("os_event") or 0) for x in a), "events_b": sum(int(records[x].get("os_event") or 0) for x in b), "p_value": _rounded(p_value), "p_adjusted_holm": None})
    for row, adjusted in zip(pairs, holm_adjust([row["p_value"] for row in pairs])):
        row["p_adjusted_holm"] = _rounded(adjusted)
    return global_row, pairs


def affinity_statistics(affinities, patient_ids, groups, permutations=999):
    from tools.multimodal_consistency_check import (
        normalized_affinity_with_audit,
        permanova_metrics,
        permdisp_metrics,
    )

    universe = stable_analysis_universe(groups)
    index_map = {case_id: i for i, case_id in enumerate(patient_ids)}
    positions = [index_map[case_id] for case_id in universe]
    labels = np.asarray([next(group for group, members in groups.items() if case_id in members) for case_id in universe])
    global_rows, pair_rows, audits = [], [], {}
    for modality, matrix in affinities.items():
        normalized, audit = normalized_affinity_with_audit(np.asarray(matrix)[np.ix_(positions, positions)])
        audits[modality] = audit
        global_permanova = permanova_metrics(normalized, labels, permutations=permutations, seed=42)
        global_permdisp = permdisp_metrics(normalized, labels, permutations=permutations, seed=142)
        row = {"modality": modality, **global_permanova, **{"permdisp_" + key: value for key, value in global_permdisp.items()}}
        global_rows.append(row)
    permanova_q = bh_adjust([item.get("permanova_p_value") for item in global_rows])
    permdisp_q = bh_adjust([item.get("permdisp_permdisp_p_value") for item in global_rows])
    for row_index, row in enumerate(global_rows):
        row["permanova_q_value"] = _rounded(permanova_q[row_index])
        row["permdisp_q_value"] = _rounded(permdisp_q[row_index])
        if row.get("permanova_q_value") is None or row["permanova_q_value"] >= .05:
            continue
        matrix = affinities[row["modality"]]
        for group_a, group_b in combinations(groups, 2):
            selected = groups[group_a] + groups[group_b]
            selected_positions = [index_map[case_id] for case_id in selected]
            pair_labels = np.asarray([group_a] * len(groups[group_a]) + [group_b] * len(groups[group_b]))
            pair_similarity, _ = normalized_affinity_with_audit(np.asarray(matrix)[np.ix_(selected_positions, selected_positions)])
            permanova = permanova_metrics(pair_similarity, pair_labels, permutations=permutations, seed=142)
            permdisp = permdisp_metrics(pair_similarity, pair_labels, permutations=permutations, seed=242)
            pair_rows.append({"modality": row["modality"], "group_a": group_a, "group_b": group_b, "n_a": len(groups[group_a]), "n_b": len(groups[group_b]), **permanova, **{"permdisp_" + key: value for key, value in permdisp.items()}})
    for modality in sorted(affinities):
        current = [row for row in pair_rows if row["modality"] == modality]
        for row, q_value in zip(current, bh_adjust([row.get("permanova_p_value") for row in current])):
            row["permanova_q_value"] = _rounded(q_value)
        for row, q_value in zip(current, bh_adjust([row.get("permdisp_permdisp_p_value") for row in current])):
            row["permdisp_q_value"] = _rounded(q_value)
    return global_rows, pair_rows, audits
