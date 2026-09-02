#!/usr/bin/env python3
"""Offline descriptive analysis of recurrent multi-K stable cores.

The script reads existing artifacts only. It never calls an Agent, changes a
partition, or assigns a biological subtype to a core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

MODALITIES = ("ct", "wsi", "rna", "genomic", "fused")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rounded(value):
    value = finite(value)
    return round(value, 8) if value is not None else None


def bh(values: list[float | None]) -> list[float | None]:
    valid = [(index, float(value)) for index, value in enumerate(values) if finite(value) is not None]
    result = [None] * len(values); running = 1.0
    for rank, (index, value) in reversed(list(enumerate(sorted(valid, key=lambda item: item[1]), 1))):
        running = min(running, value * len(valid) / rank); result[index] = rounded(running)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_test(left: list[float], right: list[float]) -> tuple[float | None, float | None]:
    if not left or not right or len(set(left + right)) < 2:
        return None, None
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(left, float), np.asarray(right, float)
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2) if len(a) + len(b) > 2 else 0
    effect = (a.mean() - b.mean()) / math.sqrt(pooled) if pooled > 0 else None
    return rounded(effect), rounded(mannwhitneyu(left, right, alternative="two-sided").pvalue)


def cliffs_delta(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    a, b = np.asarray(left), np.asarray(right)
    return rounded((np.greater.outer(a, b).sum() - np.less.outer(a, b).sum()) / (len(a) * len(b)))


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, list[float]]:
    odds = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5)); se = math.sqrt(sum(1 / (value + .5) for value in (a, b, c, d)))
    return rounded(odds), [rounded(math.exp(math.log(odds) - 1.96 * se)), rounded(math.exp(math.log(odds) + 1.96 * se))]


def load_states(data_root: Path) -> tuple[dict[str, dict], list[str]]:
    states = {}
    with (data_root / "storage/patient_states/patient_states.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                if state.get("qc") == "success": states[str(state["case_id"])] = state
    return states, sorted(states)


def load_cores(experiment_root: Path) -> tuple[dict[str, list[str]], list[str]]:
    cores: dict[str, list[str]] = {}
    for row in read_csv(experiment_root / "stable_core_membership.csv"):
        cores.setdefault(str(row["core_id"]), []).append(str(row["patient_id"]))
    cores = {key: sorted(set(value)) for key, value in sorted(cores.items())}
    return cores, sorted({case_id for members in cores.values() for case_id in members})


def load_main_sets(data_root: Path) -> list[dict]:
    summary = json.loads((data_root / "subtype_review/final_review_summary.json").read_text(encoding="utf-8"))
    accepted = {str(row["set_id"]) for row in summary.get("accepted_subtype_sets", [])}; dropped = {str(row["set_id"]) for row in summary.get("dropped_set_registry", [])}
    return [{"set_id": str(row["set_id"]), "member_ids": sorted(map(str, row.get("member_ids", []))), "decision": "accept" if str(row["set_id"]) in accepted else "drop" if str(row["set_id"]) in dropped else "unknown"} for row in summary.get("partition_sets", [])]


def main_mapping(cores: dict[str, set[str] | list[str]], main_sets: list[dict]) -> tuple[list[dict], dict[str, dict[str, int]]]:
    rows, composition = [], {}
    for core_id, core_members in cores.items():
        core_members = set(core_members); composition[core_id] = {}
        for main in main_sets:
            members = set(main["member_ids"]); intersection = len(core_members & members); composition[core_id][main["set_id"]] = intersection; union = len(core_members | members)
            rows.append({"core_id": core_id, "core_size": len(core_members), "main_set_id": main["set_id"], "main_set_size": len(members), "main_decision": main["decision"], "intersection_n": intersection, "core_fraction_in_main": intersection / len(core_members) if core_members else None, "main_fraction_captured": intersection / len(members) if members else None, "jaccard": intersection / union if union else None, "overlap_coefficient": intersection / min(len(core_members), len(members)) if core_members and members else None})
    return rows, composition


def load_table(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    rows = read_csv(path)
    if not rows: return [], {}
    features = [key for key in rows[0] if key != "case_id"]
    return features, {row["case_id"]: {feature: finite(row.get(feature)) for feature in features} for row in rows}


def ct_radiomics_table(states: dict[str, dict], all_ids: list[str]) -> tuple[list[str], dict[str, dict[str, float]]]:
    features = sorted({str(feature) for case_id in all_ids for feature in dict(states.get(case_id, {}).get("ct_evidence", {}).get("features", {}) or {})})
    table = {case_id: {feature: finite(dict(states.get(case_id, {}).get("ct_evidence", {}).get("features", {}) or {}).get(feature)) for feature in features} for case_id in all_ids}
    return features, table


def values(table: dict[str, dict[str, float]], ids: list[str], feature: str) -> list[float]:
    return [value for case_id in ids if (value := table.get(case_id, {}).get(feature)) is not None]


def core_rest_rows(table: dict[str, dict[str, float]], features: list[str], cores: dict[str, list[str]], all_ids: list[str]) -> list[dict]:
    rows = []
    for core_id, members in cores.items():
        rest = [case_id for case_id in all_ids if case_id not in members]; current = []
        for feature in features:
            left, right = values(table, members, feature), values(table, rest, feature); effect, p_value = numeric_test(left, right)
            current.append({"core_id": core_id, "feature": feature, "n_core": len(left), "n_rest": len(right), "median_core": rounded(np.median(left)) if left else None, "median_rest": rounded(np.median(right)) if right else None, "mean_core": rounded(np.mean(left)) if left else None, "mean_rest": rounded(np.mean(right)) if right else None, "smd": effect, "p_value": p_value, "q_value": None})
        for row, q_value in zip(current, bh([row["p_value"] for row in current])): row["q_value"] = q_value
        rows.extend(current)
    return rows


def test_rows(groups: dict[str, list[float]], all_values: list[float], feature: str) -> list[dict]:
    rows = []
    for core_id, left in groups.items():
        right = list(all_values)
        for value in left: right.remove(value)
        effect, p_value = numeric_test(left, right)
        rows.append({"core_id": core_id, "feature": feature, "smd": effect, "p_value": p_value, "q_value": None})
    for row, q_value in zip(rows, bh([row["p_value"] for row in rows])): row["q_value"] = q_value
    return rows


def pairwise_numeric_rows(table: dict[str, dict[str, float]], cores: dict[str, list[str]], feature: str) -> list[dict]:
    rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        left, right = values(table, cores[core_a], feature), values(table, cores[core_b], feature); effect, p_value = numeric_test(left, right)
        rows.append({"core_a": core_a, "core_b": core_b, "feature": feature, "smd_a_vs_b": effect, "p_value": p_value, "q_value": None, "median_a": rounded(np.median(left)) if left else None, "median_b": rounded(np.median(right)) if right else None})
    for row, q_value in zip(rows, bh([row["p_value"] for row in rows])): row["q_value"] = q_value
    return rows


def radiomics_analysis(table: dict[str, dict[str, float]], features: list[str], cores: dict[str, list[str]], all_ids: list[str]) -> tuple[list[dict], list[dict]]:
    families = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
    domains = {"shape": "tumor morphology and geometry", "firstorder": "intratumoral intensity distribution", "glcm": "gray-level co-occurrence texture", "glrlm": "gray-level run-length texture", "glszm": "gray-level zone-size texture", "gldm": "gray-level dependence texture", "ngtdm": "neighboring gray-tone texture"}
    family = lambda feature: next((name for name in families if f"_{name}_" in feature), "other")
    domain = lambda feature: domains.get(family(feature), "other radiomics descriptor")
    rows = core_rest_rows(table, features, cores, all_ids)
    for row in rows:
        direction = "higher in core" if (row["smd"] or 0) > 0 else "lower in core" if (row["smd"] or 0) < 0 else "not estimable"
        row.update({"effect_size": row["smd"], "feature_family": family(row["feature"]), "medical_imaging_domain": domain(row["feature"]), "interpretation": f"{domain(row['feature'])}; {direction} than the rest of the cohort", "significant_fdr": int(row["q_value"] is not None and row["q_value"] < .05)})
    pair_rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        current = []
        for feature in features:
            left, right = values(table, cores[core_a], feature), values(table, cores[core_b], feature); effect, p_value = numeric_test(left, right)
            direction = "higher in CORE A" if (effect or 0) > 0 else "lower in CORE A" if (effect or 0) < 0 else "not estimable"
            current.append({"core_a": core_a, "core_b": core_b, "feature": feature, "effect_size": effect, "smd_a_vs_b": effect, "p_value": p_value, "q_value": None, "feature_family": family(feature), "medical_imaging_domain": domain(feature), "interpretation": f"{domain(feature)}; {direction} than CORE B"})
        for row, q_value in zip(current, bh([row["p_value"] for row in current])): row["q_value"] = q_value
        pair_rows.extend(current)
    return rows, pair_rows


def clinical_value(record: dict, variable: str):
    value = record.get(variable)
    if variable == "age":
        return finite(value)
    text = str(value or "").strip().upper()
    if text in {"", "NA", "N/A", "NAN", "NONE", "UNKNOWN", "NOT REPORTED", "NOT AVAILABLE", "NX", "MX", "TX"}:
        return None
    if variable == "m_stage":
        return text if text in {"M0", "M1"} else None
    return text


def clinical_availability(records: dict[str, dict], cores: dict[str, list[str]], all_ids: list[str], threshold: float = .8) -> tuple[list[dict], dict]:
    audit_variables = ("age", "gender", "race", "stage_group", "t_stage", "n_stage", "m_stage", "grade", "overall_survival")
    formal_variables = ("age", "gender", "stage_group", "t_stage", "m_stage", "grade", "overall_survival")
    stable_core_ids = sorted(set().union(*(set(members) for members in cores.values()))) if cores else []
    rows, summary = [], {}
    for variable in audit_variables:
        scopes = {"ALL": all_ids, "STABLE_CORES": stable_core_ids, **cores}
        available_by_scope = {}
        for scope, ids in scopes.items():
            available = sum(clinical_value(records.get(case_id, {}), "os_time" if variable == "overall_survival" else variable) is not None for case_id in ids)
            available_by_scope[scope] = available
            rows.append({"clinical_variable": variable, "scope": scope, "available_n": available, "total_n": len(ids), "missing_n": len(ids) - available, "availability_fraction": available / len(ids) if ids else None})
        overall_fraction = available_by_scope["ALL"] / len(all_ids) if all_ids else 0
        core_fractions = [available_by_scope[core] / len(members) if members else 0 for core, members in cores.items()]
        eligible = variable in formal_variables and overall_fraction >= threshold and all(fraction >= threshold for fraction in core_fractions)
        stable_core_fraction = available_by_scope["STABLE_CORES"] / len(stable_core_ids) if stable_core_ids else 0
        summary[variable] = {"overall_available_n": available_by_scope["ALL"], "overall_total_n": len(all_ids), "overall_fraction": overall_fraction, "stable_core_available_n": available_by_scope["STABLE_CORES"], "stable_core_total_n": len(stable_core_ids), "stable_core_fraction": stable_core_fraction, "minimum_core_fraction": min(core_fractions, default=0), "eligible": eligible, "reason": "eligible" if eligible else "below_availability_threshold" if variable in formal_variables else "descriptive_only" if variable == "race" else "not_analyzed_below_availability_threshold"}
    summary["threshold"] = threshold
    summary["eligible_variables"] = [variable for variable in formal_variables if variable != "overall_survival" and summary[variable]["eligible"]]
    summary["excluded_variables"] = [variable for variable in formal_variables if variable != "overall_survival" and not summary[variable]["eligible"]]
    summary["not_analyzed_variables"] = [variable for variable in audit_variables if variable not in formal_variables]
    summary["survival_eligible"] = summary["overall_survival"]["eligible"]
    return rows, summary


def clinical_analysis(records: dict[str, dict], cores: dict[str, list[str]], all_ids: list[str], eligible_variables: list[str] | None = None, survival_eligible: bool = True) -> tuple[list[dict], list[dict], list[dict]]:
    from scipy.stats import fisher_exact
    categorical = {"gender", "stage_group", "t_stage", "m_stage", "grade"}
    variables = tuple(("age", "gender", "stage_group", "t_stage", "m_stage", "grade") if eligible_variables is None else eligible_variables)
    rows, survival = [], []
    for core_id, members in cores.items():
        rest = [case_id for case_id in all_ids if case_id not in members]
        current = []
        for variable in variables:
            left = [clinical_value(records.get(case_id, {}), variable) for case_id in members]
            right = [clinical_value(records.get(case_id, {}), variable) for case_id in rest]
            left, right = [value for value in left if value is not None], [value for value in right if value is not None]
            if not left or not right:
                continue
            if variable in categorical:
                for level in sorted(set(left + right)):
                    set_count, rest_count = left.count(level), right.count(level)
                    if not left or not right:
                        p_value, odds, ci = None, None, [None, None]
                    else:
                        _, p_value = fisher_exact([[set_count, len(left) - set_count], [rest_count, len(right) - rest_count]])
                        odds, ci = odds_ratio_ci(set_count, len(left) - set_count, rest_count, len(right) - rest_count)
                    current.append({"core_id": core_id, "clinical_variable": variable, "level": level, "set_count": set_count, "set_total": len(left), "rest_count": rest_count, "rest_total": len(right), "set_fraction": set_count / len(left) if left else None, "rest_fraction": rest_count / len(right) if right else None, "frequency_difference": set_count / len(left) - rest_count / len(right) if left and right else None, "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1], "p_value": rounded(p_value), "q_value": None, "available_n": len(left) + len(right), "missing_n": len(all_ids) - len(left) - len(right)})
            else:
                effect, p_value = numeric_test(left, right)
                current.append({"core_id": core_id, "clinical_variable": variable, "n_core": len(left), "n_rest": len(right), "median_core": rounded(np.median(left)) if left else None, "median_rest": rounded(np.median(right)) if right else None, "mean_core": rounded(np.mean(left)) if left else None, "mean_rest": rounded(np.mean(right)) if right else None, "smd": effect, "effect_size": effect, "direction": "higher_in_core" if (effect or 0) > 0 else "lower_in_core" if (effect or 0) < 0 else None, "p_value": p_value, "q_value": None, "available_n": len(left) + len(right), "missing_n": len(all_ids) - len(left) - len(right)})
        for row, q_value in zip(current, bh([row["p_value"] for row in current])): row["q_value"] = q_value
        rows.extend(current)
        left = [finite(records.get(case_id, {}).get("os_time")) for case_id in members if finite(records.get(case_id, {}).get("os_time")) is not None]; right = [finite(records.get(case_id, {}).get("os_time")) for case_id in rest if finite(records.get(case_id, {}).get("os_time")) is not None]
        p_value = None
        if survival_eligible and left and right and (sum(records.get(case_id, {}).get("os_event", 0) for case_id in members) + sum(records.get(case_id, {}).get("os_event", 0) for case_id in rest)) > 0:
            from lifelines.statistics import logrank_test
            p_value = logrank_test(left, right, event_observed_A=[records.get(case_id, {}).get("os_event", 0) for case_id in members if finite(records.get(case_id, {}).get("os_time")) is not None], event_observed_B=[records.get(case_id, {}).get("os_event", 0) for case_id in rest if finite(records.get(case_id, {}).get("os_time")) is not None]).p_value
        survival.append({"core_id": core_id, "clinical_variable": "overall_survival", "set_n": len(left), "rest_n": len(right), "set_events": sum(records.get(case_id, {}).get("os_event", 0) for case_id in members if finite(records.get(case_id, {}).get("os_time")) is not None), "rest_events": sum(records.get(case_id, {}).get("os_event", 0) for case_id in rest if finite(records.get(case_id, {}).get("os_time")) is not None), "set_median_time": rounded(np.median(left)) if left else None, "rest_median_time": rounded(np.median(right)) if right else None, "p_value": rounded(p_value), "q_value": None})
    for row, q_value in zip(survival, bh([row["p_value"] for row in survival])): row["q_value"] = q_value
    pair_rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        current = []
        for variable in variables:
            left = [clinical_value(records.get(case_id, {}), variable) for case_id in cores[core_a]]; right = [clinical_value(records.get(case_id, {}), variable) for case_id in cores[core_b]]; left, right = [value for value in left if value is not None], [value for value in right if value is not None]
            if not left or not right:
                continue
            if variable in categorical:
                for level in sorted(set(left + right)):
                    a, c = left.count(level), right.count(level); p_value = fisher_exact([[a, len(left) - a], [c, len(right) - c]])[1] if left and right else None; odds, ci = odds_ratio_ci(a, len(left) - a, c, len(right) - c) if left and right else (None, [None, None]); current.append({"core_a": core_a, "core_b": core_b, "clinical_variable": variable, "level": level, "core_a_count": a, "core_a_total": len(left), "core_b_count": c, "core_b_total": len(right), "core_a_fraction": a / len(left) if left else None, "core_b_fraction": c / len(right) if right else None, "frequency_difference": a / len(left) - c / len(right) if left and right else None, "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1], "p_value": rounded(p_value), "q_value": None})
            else:
                effect, p_value = numeric_test(left, right); current.append({"core_a": core_a, "core_b": core_b, "clinical_variable": variable, "n_core_a": len(left), "n_core_b": len(right), "median_core_a": rounded(np.median(left)) if left else None, "median_core_b": rounded(np.median(right)) if right else None, "smd_a_vs_b": effect, "effect_size": effect, "p_value": p_value, "q_value": None})
        for row, q_value in zip(current, bh([row["p_value"] for row in current])): row["q_value"] = q_value
        pair_rows.extend(current)
        left_ids = [case_id for case_id in cores[core_a] if finite(records.get(case_id, {}).get("os_time")) is not None]; right_ids = [case_id for case_id in cores[core_b] if finite(records.get(case_id, {}).get("os_time")) is not None]; left = [finite(records[case_id]["os_time"]) for case_id in left_ids]; right = [finite(records[case_id]["os_time"]) for case_id in right_ids]; p_value = None
        if survival_eligible and left and right and (sum(records[case_id].get("os_event", 0) for case_id in left_ids) + sum(records[case_id].get("os_event", 0) for case_id in right_ids)) > 0:
            from lifelines.statistics import logrank_test
            p_value = logrank_test(left, right, event_observed_A=[records[case_id].get("os_event", 0) for case_id in left_ids], event_observed_B=[records[case_id].get("os_event", 0) for case_id in right_ids]).p_value
        pair_rows.append({"core_a": core_a, "core_b": core_b, "clinical_variable": "overall_survival", "core_a_n": len(left), "core_b_n": len(right), "core_a_events": sum(records[case_id].get("os_event", 0) for case_id in left_ids), "core_b_events": sum(records[case_id].get("os_event", 0) for case_id in right_ids), "core_a_median_time": rounded(np.median(left)) if left else None, "core_b_median_time": rounded(np.median(right)) if right else None, "p_value": rounded(p_value), "q_value": None})
    survival_pair_rows = [row for row in pair_rows if row["clinical_variable"] == "overall_survival"]
    for row, q_value in zip(survival_pair_rows, bh([row["p_value"] for row in survival_pair_rows])): row["q_value"] = q_value
    return rows, pair_rows, survival


def load_expression_scores(states: dict[str, dict], config_dir: Path):
    from tools.pathway_enrichment import rna_feature_path, ssgsea_scores
    from tools.subtype_review_common import feature_dataframe, read_gmt_gene_sets, tool_parameters
    feature_frame = feature_dataframe(rna_feature_path(states)); params = tool_parameters(str(config_dir), "rna"); pathways, _ = read_gmt_gene_sets(str(params.get("pathway_gene_sets_path", ""))); min_overlap = max(int(params.get("min_pathway_overlap", 15) or 15), 1)
    if feature_frame.empty or not pathways: raise RuntimeError("RNA ssGSEA inputs are unavailable")
    pathways = {pathway: [gene for gene in genes if gene in feature_frame.columns] for pathway, genes in pathways.items()}; pathways = {pathway: genes for pathway, genes in pathways.items() if len(genes) >= min_overlap}
    if not pathways: raise RuntimeError("No Hallmark pathway meets the configured gene-overlap requirement")
    try: scores = ssgsea_scores(feature_frame, pathways, min_overlap)
    except Exception as exc: raise RuntimeError("RNA ssGSEA execution failed") from exc
    if scores.empty: raise RuntimeError("RNA ssGSEA returned no scores")
    return sorted(pathways), scores


def rna_analysis(states: dict[str, dict], config_dir: Path, cores: dict[str, list[str]], all_ids: list[str], pathways=None, scores=None) -> tuple[list[dict], list[dict]]:
    pathways, scores = load_expression_scores(states, config_dir) if pathways is None or scores is None else (pathways, scores); rows = []
    for core_id, members in cores.items():
        rest = [case_id for case_id in all_ids if case_id not in members]; current = []
        for pathway in pathways:
            left = scores.loc[[x for x in members if x in scores.index], pathway].dropna().astype(float).tolist(); right = scores.loc[[x for x in rest if x in scores.index], pathway].dropna().astype(float).tolist(); effect, p_value = numeric_test(left, right)
            current.append({"core_id": core_id, "pathway": pathway, "n_core": len(left), "n_rest": len(right), "median_core": rounded(np.median(left)) if left else None, "median_rest": rounded(np.median(right)) if right else None, "mean_core": rounded(np.mean(left)) if left else None, "mean_rest": rounded(np.mean(right)) if right else None, "smd": effect, "direction": "up" if (effect or 0) > 0 else "down" if (effect or 0) < 0 else None, "p_value": p_value, "q_value": None, "available_n_core": len(left), "available_n_rest": len(right)})
        for row, q_value in zip(current, bh([row["p_value"] for row in current])): row["q_value"] = q_value
        rows.extend(current)
    pair_rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        current = []
        for pathway in pathways:
            left = scores.loc[[x for x in cores[core_a] if x in scores.index], pathway].dropna().astype(float).tolist(); right = scores.loc[[x for x in cores[core_b] if x in scores.index], pathway].dropna().astype(float).tolist(); effect, p_value = numeric_test(left, right)
            current.append({"core_a": core_a, "core_b": core_b, "pathway": pathway, "smd_a_vs_b": effect, "p_value": p_value, "q_value": None, "median_a": rounded(np.median(left)) if left else None, "median_b": rounded(np.median(right)) if right else None})
        for row, q_value in zip(current, bh([row["p_value"] for row in current])): row["q_value"] = q_value
        pair_rows.extend(current)
    return rows, pair_rows


def binary_analysis(table, features, cores, all_ids, kind):
    from scipy.stats import fisher_exact
    rows, pair_rows = [], []
    for core_id, members in cores.items():
        rest = [case_id for case_id in all_ids if case_id not in members]; current = []
        for feature in features:
            left, right = values(table, members, feature), values(table, rest, feature)
            if not left or not right: continue
            set_n, rest_n = int(sum(value > 0 for value in left)), int(sum(value > 0 for value in right)); odds_raw, p_value = fisher_exact([[set_n, len(left) - set_n], [rest_n, len(right) - rest_n]]); odds, ci = odds_ratio_ci(set_n, len(left) - set_n, rest_n, len(right) - rest_n)
            row = {"core_id": core_id, "feature": feature, "q_value": None, "fisher_p": rounded(p_value), "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1], "frequency_difference": set_n / len(left) - rest_n / len(right)}
            if kind == "mutation": row.update({"gene": feature.removeprefix("mutation::"), "set_mutated_n": set_n, "set_total_n": len(left), "rest_mutated_n": rest_n, "rest_total_n": len(right), "mutation_frequency": set_n / len(left), "rest_mutation_frequency": rest_n / len(right), "delta_mutation_frequency": set_n / len(left) - rest_n / len(right)})
            else: row.update({"event": feature.rsplit("::", 1)[-1], "set_altered_n": set_n, "set_total_n": len(left), "rest_altered_n": rest_n, "rest_total_n": len(right), "alteration_frequency": set_n / len(left), "rest_alteration_frequency": rest_n / len(right), "delta_alteration_frequency": set_n / len(left) - rest_n / len(right)})
            current.append(row)
        for row, q_value in zip(current, bh([row["fisher_p"] for row in current])): row["q_value"] = q_value
        rows.extend(current)
    for core_a, core_b in combinations(sorted(cores), 2):
        current = []
        for feature in features:
            left, right = values(table, cores[core_a], feature), values(table, cores[core_b], feature)
            if not left or not right: continue
            a, c = int(sum(value > 0 for value in left)), int(sum(value > 0 for value in right)); b, d = len(left) - a, len(right) - c; _, p_value = fisher_exact([[a, b], [c, d]]); odds, ci = odds_ratio_ci(a, b, c, d)
            current.append({"core_a": core_a, "core_b": core_b, "feature": feature, "fisher_p": rounded(p_value), "q_value": None, "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1], "core_a_frequency": a / len(left), "core_b_frequency": c / len(right), "frequency_difference": a / len(left) - c / len(right)})
        for row, q_value in zip(current, bh([row["fisher_p"] for row in current])): row["q_value"] = q_value
        pair_rows.extend(current)
    return rows, pair_rows


def cnv_analysis(table, features, cores, all_ids, threshold=.2):
    continuous = core_rest_rows(table, features, cores, all_ids)
    for row in continuous:
        left, rest = values(table, cores[row["core_id"]], row["feature"]), values(table, [x for x in all_ids if x not in cores[row["core_id"]]], row["feature"]); row["cliffs_delta"] = cliffs_delta(left, rest); row["direction"] = "higher_in_core" if (row["cliffs_delta"] or 0) > 0 else "lower_in_core" if (row["cliffs_delta"] or 0) < 0 else None
    event_features, event_table = [], {case_id: {} for case_id in table}
    for feature in features:
        if feature.startswith(("chr", "locus::")):
            for event, predicate in (("loss", lambda value: value <= -threshold), ("gain", lambda value: value >= threshold)):
                name = f"{feature}::{event}"; event_features.append(name)
                for case_id, row in table.items():
                    if row.get(feature) is not None: event_table[case_id][name] = float(predicate(row[feature]))
    gain_loss, _ = binary_analysis(event_table, event_features, cores, all_ids, "cnv"); pair_continuous, pair_gain_loss = [], []
    for core_a, core_b in combinations(sorted(cores), 2):
        current_continuous = []
        for feature in features:
            left, right = values(table, cores[core_a], feature), values(table, cores[core_b], feature); effect, p_value = numeric_test(left, right); current_continuous.append({"core_a": core_a, "core_b": core_b, "feature": feature, "cliffs_delta": cliffs_delta(left, right), "p_value": p_value, "q_value": None, "direction": "higher_in_core_a" if (effect or 0) > 0 else "lower_in_core_a" if (effect or 0) < 0 else None})
        for row, q_value in zip(current_continuous, bh([row["p_value"] for row in current_continuous])): row["q_value"] = q_value
        pair_continuous.extend(current_continuous); current_gain_loss = []
        for feature in event_features:
            left, right = values(event_table, cores[core_a], feature), values(event_table, cores[core_b], feature)
            if not left or not right: continue
            a, c = int(sum(left)), int(sum(right)); b, d = len(left) - a, len(right) - c; from scipy.stats import fisher_exact; _, p_value = fisher_exact([[a, b], [c, d]]); odds, ci = odds_ratio_ci(a, b, c, d)
            current_gain_loss.append({"core_a": core_a, "core_b": core_b, "feature": feature, "core_a_altered_n": a, "core_a_total_n": len(left), "core_a_frequency": a / len(left), "core_b_altered_n": c, "core_b_total_n": len(right), "core_b_frequency": c / len(right), "frequency_difference": a / len(left) - c / len(right), "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1], "fisher_p": rounded(p_value), "q_value": None})
        for row, q_value in zip(current_gain_loss, bh([row["fisher_p"] for row in current_gain_loss])): row["q_value"] = q_value
        pair_gain_loss.extend(current_gain_loss)
    return continuous, gain_loss, pair_continuous, pair_gain_loss


def affinity_characterization(similarity, ids, cores):
    from tools.multimodal_consistency_check import fixed_membership_silhouettes, normalized_affinity_with_audit
    normalized, audit = normalized_affinity_with_audit(similarity); labels = {case_id: core for core, members in cores.items() for case_id in members}; ordered = [case_id for case_id in ids if case_id in labels]; indices = [ids.index(case_id) for case_id in ordered]; distance = 1 - normalized[np.ix_(indices, indices)]; silhouettes = fixed_membership_silhouettes(distance, np.asarray([labels[case_id] for case_id in ordered])); per_core = {}
    for core, members in cores.items():
        values_for_core = [silhouettes[str(i)] for i, case_id in enumerate(ordered) if case_id in members and silhouettes[str(i)] is not None]; per_core[core] = {"silhouette": rounded(np.median(values_for_core)) if values_for_core else None, "available_n": len(values_for_core)}
    core_ids = sorted(cores); distances = {core: {other: None for other in core_ids} for core in core_ids}
    for core_a, core_b in combinations(core_ids, 2):
        left, right = [ordered.index(x) for x in cores[core_a] if x in ordered], [ordered.index(x) for x in cores[core_b] if x in ordered]; distances[core_a][core_b] = distances[core_b][core_a] = rounded(distance[np.ix_(left, right)].mean()) if left and right else None
    return {"per_core": per_core, "distance_matrix": distances, "ordered_ids": ordered, "similarity": normalized, "audit": audit}


def core_embedding_analysis(affinities, ids, cores, permutations=999):
    from tools.multimodal_consistency_check import permanova_metrics, permdisp_metrics
    separation, distances, tests, audits = [], {}, {}, {}
    for modality, matrix in affinities.items():
        result = affinity_characterization(matrix, ids, cores); distances[modality], audits[modality] = result["distance_matrix"], result["audit"]; separation.extend({"modality": modality, "core_id": core, **row} for core, row in result["per_core"].items()); ordered = result["ordered_ids"]; labels = np.asarray([next(core for core, members in cores.items() if case_id in members) for case_id in ordered]); sub = result["similarity"][np.ix_([ids.index(x) for x in ordered], [ids.index(x) for x in ordered])]; tests[modality] = {"permanova": permanova_metrics(sub, labels, permutations=permutations, seed=42), "permdisp": permdisp_metrics(sub, labels, permutations=permutations, seed=142)}
    for metric_name, p_key in (("permanova", "permanova_p_value"), ("permdisp", "permdisp_p_value")):
        q_values = bh([tests[modality][metric_name].get(p_key) for modality in tests])
        for modality, q_value in zip(tests, q_values):
            tests[modality][metric_name]["q_value"] = q_value
    return separation, distances, tests, audits


def load_core_runs(experiment_root):
    runs = []
    for path in sorted(experiment_root.glob("run*/K*/final_review_summary.json")):
        report = json.loads(path.read_text(encoding="utf-8")); accepted = {str(row["set_id"]): set(map(str, row.get("member_ids", []))) for row in report.get("accepted_subtype_sets", [])}; runs.append({"run_id": f"{path.parents[1].name}_{path.parents[0].name}", "initial_k": int(path.parents[0].name[1:]), "sets": accepted})
    return runs


def cooccurrence_from_runs(runs, cores):
    rows, by_k = [], []
    for run in runs:
        for core_a, core_b in combinations(sorted(cores), 2):
            details = {}
            for core in (core_a, core_b):
                candidates = [(set_id, len(set(cores[core]) & members)) for set_id, members in run["sets"].items()]; dominant, count = max(candidates, key=lambda item: (item[1], item[0]), default=(None, 0)); details[core] = (dominant, count / len(cores[core]) if cores[core] else None)
            a_set, a_fraction = details[core_a]; b_set, b_fraction = details[core_b]; valid = a_fraction is not None and b_fraction is not None and a_fraction >= .5 and b_fraction >= .5; rows.append({"core_a": core_a, "core_b": core_b, "run_id": run["run_id"], "K": run["initial_k"], "dominant_set_a": a_set, "dominant_fraction_a": rounded(a_fraction), "dominant_set_b": b_set, "dominant_fraction_b": rounded(b_fraction), "valid_assignment": int(valid), "same_parent_set": int(a_set == b_set) if valid else None})
    for core_a, core_b in combinations(sorted(cores), 2):
        current = [row for row in rows if row["core_a"] == core_a and row["core_b"] == core_b]
        for k in sorted({row["K"] for row in current}):
            subset = [row for row in current if row["K"] == k]; valid = [row for row in subset if row["valid_assignment"]]; same = [row for row in valid if row["same_parent_set"] == 1]; by_k.append({"core_a": core_a, "core_b": core_b, "K": k, "valid_repeat_count": len(valid), "not_estimable_repeat_count": len(subset) - len(valid), "same_parent_run_count": len(same), "conditional_same_parent_fraction": rounded(len(same) / len(valid)) if valid else None, "unconditional_same_parent_fraction": rounded(len(same) / len(subset)) if subset else None})
    return rows, by_k


def core_confounds(data_root, config_dir, states, cores):
    from tools.confound import CATEGORICAL_FIELDS, NUMERIC_FIELDS, assign_q_values, confounder_values, global_categorical, global_numeric, set_categorical, set_numeric
    values_by_case = confounder_values(states, str(data_root)); global_rows = {field: global_categorical(field, cores, values_by_case, 999) for field in CATEGORICAL_FIELDS}; global_rows.update({field: global_numeric(field, cores, values_by_case) for field in NUMERIC_FIELDS}); set_rows = {core: {**{field: set_categorical(core, field, cores, values_by_case) for field in CATEGORICAL_FIELDS}, **{field: set_numeric(core, field, cores, values_by_case) for field in NUMERIC_FIELDS}} for core in cores}; assign_q_values(global_rows, set_rows); rows = []
    for core, fields in set_rows.items():
        for field, metric in fields.items():
            if field in CATEGORICAL_FIELDS:
                for level, item in metric.items(): rows.append({"core_id": core, "field": field, "level": level, **{key: item.get(key) for key in ("set_count", "set_total", "rest_count", "rest_total", "set_fraction", "rest_fraction", "frequency_difference", "odds_ratio", "q_value")}})
            else: rows.append({"core_id": core, "field": field, "level": "", **{key: metric.get(key) for key in ("set_mean", "rest_mean", "cliffs_delta", "q_value", "direction")}})
    rows.extend({"core_id": "GLOBAL", "field": field, "level": "", "cramers_v": metric.get("cramers_v"), "epsilon_squared": metric.get("epsilon_squared"), "q_value": metric.get("q_value"), "available_n": metric.get("available_n"), "missing_n": metric.get("missing_n")} for field, metric in global_rows.items()); return rows


def select_pathways(rows, limit):
    grouped = {pathway: [row for row in rows if row["pathway"] == pathway] for pathway in sorted({row["pathway"] for row in rows})}; significant = [pathway for pathway, values_for_pathway in grouped.items() if any((row.get("q_value") or 1) < .05 for row in values_for_pathway)]; order = lambda pathway: (-max(abs(row.get("smd") or 0) for row in grouped[pathway]), pathway); selected = sorted(significant, key=order)[:limit]; selected.extend(pathway for pathway in sorted((x for x in grouped if x not in selected), key=order)[:limit - len(selected)]); return selected


def select_cnv_heatmap_features(rows, limit):
    grouped = {}
    for row in rows: grouped.setdefault(row["feature"], []).append(row)
    return sorted(grouped, key=lambda feature: (not any((row.get("q_value") or 1) < .05 for row in grouped[feature]), -max(abs(row.get("cliffs_delta") or 0) for row in grouped[feature]), feature))[:limit]


def pairwise_similarity(cores, rna_rows, wxs_rows, cnv_rows, distance_matrices, co_run, co_k):
    from scipy.stats import pearsonr, spearmanr
    def corr(left, right, method):
        pairs = [(left[key], right[key]) for key in sorted(set(left) & set(right)) if left[key] is not None and right[key] is not None]; return rounded(method(*zip(*pairs)).statistic) if len(pairs) >= 3 else None
    result = []
    for core_a, core_b in combinations(sorted(cores), 2):
        rna_a = {row["pathway"]: row.get("smd") for row in rna_rows if row["core_id"] == core_a}; rna_b = {row["pathway"]: row.get("smd") for row in rna_rows if row["core_id"] == core_b}; wxs_a = {row["gene"]: row.get("mutation_frequency") for row in wxs_rows if row["core_id"] == core_a}; wxs_b = {row["gene"]: row.get("mutation_frequency") for row in wxs_rows if row["core_id"] == core_b}; cnv_a = {row["feature"]: row.get("cliffs_delta") for row in cnv_rows if row["core_id"] == core_a}; cnv_b = {row["feature"]: row.get("cliffs_delta") for row in cnv_rows if row["core_id"] == core_b}; same = [row["same_parent_set"] for row in co_run if row["core_a"] == core_a and row["core_b"] == core_b and row["same_parent_set"] is not None]; k_rows = [row for row in co_k if row["core_a"] == core_a and row["core_b"] == core_b]
        result.append({"core_a": core_a, "core_b": core_b, "conditional_same_parent_fraction": rounded(np.mean(same)) if same else None, "unconditional_same_parent_fraction": rounded(sum(same) / len([row for row in co_run if row["core_a"] == core_a and row["core_b"] == core_b])) if any(row["core_a"] == core_a and row["core_b"] == core_b for row in co_run) else None, "same_parent_run_count": sum(same), "k_with_any_same_parent": sum(row["same_parent_run_count"] > 0 for row in k_rows), "k_with_majority_same_parent": sum(row["valid_repeat_count"] > 0 and row["same_parent_run_count"] / row["valid_repeat_count"] >= 2 / 3 for row in k_rows), "rna_hallmark_profile_pearson": corr(rna_a, rna_b, pearsonr), "rna_hallmark_profile_spearman": corr(rna_a, rna_b, spearmanr), "mutation_frequency_pearson": corr(wxs_a, wxs_b, pearsonr), "cnv_effect_profile_pearson": corr(cnv_a, cnv_b, pearsonr), "ct_mean_between_core_distance": distance_matrices.get("ct", {}).get(core_a, {}).get(core_b), "wsi_mean_between_core_distance": distance_matrices.get("wsi", {}).get(core_a, {}).get(core_b), "rna_mean_between_core_distance": distance_matrices.get("rna", {}).get(core_a, {}).get(core_b), "genomic_mean_between_core_distance": distance_matrices.get("genomic", {}).get(core_a, {}).get(core_b), "fused_mean_between_core_distance": distance_matrices.get("fused", {}).get(core_a, {}).get(core_b), "cnv_pairwise_fdr_feature_count": None, "wxs_pairwise_fdr_gene_count": None, "rna_pairwise_fdr_pathway_count": None})
    return result


def update_pairwise_similarity(rows, rna_pair, wxs_pair, cnv_pair):
    for row in rows:
        pair = lambda source: [item for item in source if item["core_a"] == row["core_a"] and item["core_b"] == row["core_b"]]; rna, wxs, cnv = pair(rna_pair), pair(wxs_pair), pair(cnv_pair); row["rna_pairwise_fdr_pathway_count"] = sum((item.get("q_value") or 1) < .05 for item in rna); row["wxs_pairwise_fdr_gene_count"] = sum((item.get("q_value") or 1) < .05 for item in wxs); row["cnv_pairwise_fdr_feature_count"] = sum((item.get("q_value") or 1) < .05 for item in cnv)
    return rows


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight"); import matplotlib.pyplot as plt; plt.close(figure)


def plot_heatmap(path, matrix, row_labels, col_labels, title="", annotations=None, diverging=False):
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    figure, axis = plt.subplots(figsize=(max(7, len(col_labels) * .8), max(4, len(row_labels) * .32))); matrix = np.nan_to_num(np.asarray(matrix, float)); kwargs = {"cmap": "RdBu_r", "norm": TwoSlopeNorm(0, vmin=float(matrix.min()), vmax=float(matrix.max()))} if diverging and matrix.size and matrix.min() < 0 < matrix.max() else {"cmap": "viridis"}; image = axis.imshow(matrix, aspect="auto", **kwargs); axis.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right"); axis.set_yticks(range(len(row_labels)), row_labels); axis.set_title(title); figure.colorbar(image, ax=axis, shrink=.8)
    if annotations:
        for i, row in enumerate(annotations):
            for j, text in enumerate(row):
                if text: axis.text(j, i, text, ha="center", va="center", fontsize=8)
    figure.tight_layout(); save_figure(figure, path)


def plot_bubbles(path, rows, cores, pathways):
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(max(7, len(cores) * .9), max(5, len(pathways) * .32))); lookup = {(row["core_id"], row["pathway"]): row for row in rows}; max_effect = max((abs(row.get("smd") or 0) for row in rows), default=1) or 1
    for y, pathway in enumerate(pathways):
        for x, core in enumerate(cores):
            row = lookup.get((core, pathway), {}); effect = float(row.get("smd") or 0); q_value = row.get("q_value"); axis.scatter(x, y, s=max(8, abs(effect) * 90), c=[effect], cmap="RdBu_r", vmin=-max_effect, vmax=max_effect, edgecolors="black" if q_value is not None and q_value < .05 else "none")
    axis.set_xticks(range(len(cores)), cores, rotation=45, ha="right"); axis.set_yticks(range(len(pathways)), pathways); axis.set_xlabel("CORE"); axis.set_ylabel("Hallmark pathway"); axis.set_title("RNA pathway bubble plot (size=|SMD|, color=SMD)"); figure.tight_layout(); save_figure(figure, path)


def plot_oncoplot(path, table, genes, cores, all_ids, main_by_core, tss_by_core):
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    ordered = []
    for core in sorted(cores):
        members = sorted(cores[core], key=lambda case_id: (-sum((table.get(case_id, {}).get(gene) or 0) > 0 for gene in genes), tuple(-int((table.get(case_id, {}).get(gene) or 0) > 0) for gene in genes), case_id)); ordered.extend((case_id, core) for case_id in members)
    case_ids = [case_id for case_id, _ in ordered]; matrix = np.asarray([[int((table.get(case_id, {}).get(gene) or 0) > 0) for case_id in case_ids] for gene in genes]); figure, axis = plt.subplots(figsize=(max(12, len(case_ids) * .16), max(6, len(genes) * .22))); axis.imshow(matrix, aspect="auto", interpolation="none", cmap="Greens", vmin=0, vmax=1); axis.set_yticks(range(len(genes)), [gene.removeprefix("mutation::") for gene in genes]); axis.set_xticks(range(len(case_ids)), case_ids, rotation=90, fontsize=5); start = 0
    for core in sorted(cores):
        count = sum(item[1] == core for item in ordered); axis.axvspan(start - .5, start + count - .5, alpha=.08); axis.text(start + max(count - 1, 0) / 2, -1.8, f"{core} n={count}\nmain={main_by_core.get(core, '')} TSS={tss_by_core.get(core, '')}", ha="center", va="bottom", fontsize=8); start += count
    axis.set_title("WXS discovery mutations: Mutated / WT only"); right = figure.add_axes([.88, .12, .1, .75]); core_ids = sorted(cores); width = .8 / max(len(core_ids), 1)
    for index, core in enumerate(core_ids):
        core_columns = [j for j, item in enumerate(ordered) if item[1] == core]
        frequencies = [matrix[i, core_columns].mean() if core_columns else 0 for i in range(len(genes))]
        right.barh(np.arange(len(genes)) + index * width, frequencies, height=width, label=core)
    right.set_yticks(range(len(genes)), []); right.set_xlabel("Mutation frequency"); right.legend(title="CORE", fontsize=6); figure.subplots_adjust(left=.2, bottom=.25, right=.86, top=.82); save_figure(figure, path)


def plot_survival_km(path, records, cores):
    from lifelines import KaplanMeierFitter
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    groups = {core: members for core, members in sorted(cores.items())}
    figure, axis = plt.subplots(figsize=(9, 6))
    plotted = 0
    for group, members in groups.items():
        usable = [records.get(case_id, {}) for case_id in members if finite(records.get(case_id, {}).get("os_time")) is not None]
        if not usable:
            continue
        durations = [finite(record["os_time"]) for record in usable]
        events = [int(record.get("os_event") or 0) for record in usable]
        KaplanMeierFitter().fit(durations, event_observed=events, label=f"{group} (n={len(usable)}, events={sum(events)})").plot_survival_function(ax=axis)
        plotted += 1
    axis.set_xlabel("Overall survival time (days)"); axis.set_ylabel("Survival probability"); axis.set_title("Overall survival by recurrent stable core (descriptive)"); axis.grid(alpha=.2); axis.legend(title="Group"); figure.tight_layout(); save_figure(figure, path)
    return plotted


def plot_patient_coassignment(path, runs, cores):
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    case_ids = [case_id for core in sorted(cores) for case_id in sorted(cores[core])]; index = {case_id: i for i, case_id in enumerate(case_ids)}; numerator = np.zeros((len(case_ids), len(case_ids))); denominator = np.zeros_like(numerator)
    for run in runs:
        assignment = {case_id: set_id for set_id, members in run["sets"].items() for case_id in members if case_id in index}
        for i, case_a in enumerate(case_ids):
            for j in range(i + 1, len(case_ids)):
                case_b = case_ids[j]
                if case_a not in assignment or case_b not in assignment: continue
                denominator[i, j] += 1; denominator[j, i] += 1
                if assignment[case_a] == assignment[case_b]: numerator[i, j] += 1; numerator[j, i] += 1
    matrix = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0); np.fill_diagonal(matrix, 1); figure, axis = plt.subplots(figsize=(12, 11)); image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", interpolation="none"); axis.set_xticks(range(len(case_ids)), case_ids, rotation=90, fontsize=5); axis.set_yticks(range(len(case_ids)), case_ids, fontsize=5); axis.set_title(f"Patient co-assignment across accepted multi-K runs (stable cores only, n={len(case_ids)})"); figure.colorbar(image, ax=axis, label="Conditional co-assignment fraction")
    start = 0
    for core in sorted(cores):
        start += len(cores[core]); axis.axhline(start - .5, color="white", linewidth=1.2); axis.axvline(start - .5, color="white", linewidth=1.2)
    figure.tight_layout(); save_figure(figure, path); return len(case_ids)


def plot_patient_rna_heatmap(path, scores, pathways, cores, records):
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    case_ids = [case_id for core in sorted(cores) for case_id in sorted(cores[core]) if case_id in scores.index]; pathways = [pathway for pathway in pathways if pathway in scores.columns]; matrix = scores.loc[case_ids, pathways].astype(float).to_numpy().T; matrix = (matrix - np.nanmean(matrix, axis=1, keepdims=True)) / np.where(np.nanstd(matrix, axis=1, keepdims=True) > 0, np.nanstd(matrix, axis=1, keepdims=True), 1); fields = ["CORE", "TSS", "stage_group", "grade", "m_stage", "OS event", "Age"]; annotation = np.zeros((len(fields), len(case_ids)), dtype=float); legend_handles = []; annotation_levels = []
    for row_index, field in enumerate(fields):
        values_for_field = [next((core for core, members in cores.items() if case_id in members), "") if field == "CORE" else case_id.split("-")[1] if field == "TSS" and len(case_id.split("-")) > 2 else "" if field == "TSS" else str(records.get(case_id, {}).get("os_event", "")) if field == "OS event" else records.get(case_id, {}).get("age") if field == "Age" else str(records.get(case_id, {}).get(field, "")) for case_id in case_ids]
        if field == "Age": annotation[row_index] = [finite(value) if finite(value) is not None else np.nan for value in values_for_field]; annotation_levels.append(None)
        else:
            levels = sorted(set(values_for_field)); mapping = {value: index + 1 for index, value in enumerate(levels)}; annotation[row_index] = [mapping[value] for value in values_for_field]; annotation_levels.append(levels); legend_handles.extend(Patch(facecolor=plt.get_cmap("tab20")(i), label=f"{field}: {level}") for i, level in enumerate(levels))
    figure, (annotation_axis, heatmap_axis) = plt.subplots(2, 1, figsize=(max(14, len(case_ids) * .22), max(9, len(pathways) * .36 + 2)), gridspec_kw={"height_ratios": [.9, max(5, len(pathways) * .34)]}, sharex=True); annotation_colors = plt.get_cmap("tab20")
    for row_index, (field, levels) in enumerate(zip(fields, annotation_levels)):
        if field == "Age": annotation_axis.imshow(annotation[row_index][None, :], aspect="auto", interpolation="none", cmap="viridis", extent=(-.5, len(case_ids) - .5, row_index, row_index + 1))
        else: annotation_axis.imshow(annotation[row_index][None, :], aspect="auto", interpolation="none", cmap=ListedColormap(["white"] + [annotation_colors(i) for i in range(len(levels))]), vmin=0, vmax=max(len(levels), 1), extent=(-.5, len(case_ids) - .5, row_index, row_index + 1))
    annotation_axis.set_yticks(np.arange(len(fields)) + .5, fields); annotation_axis.set_ylim(len(fields), 0); annotation_axis.set_xticks([]); annotation_axis.set_title("Clinical and technical annotations (Age uses continuous viridis scale)"); image = heatmap_axis.imshow(matrix, aspect="auto", interpolation="none", cmap="RdBu_r", vmin=-2.5, vmax=2.5); heatmap_axis.set_yticks(range(len(pathways)), pathways); heatmap_axis.set_xticks(range(len(case_ids)), case_ids, rotation=90, fontsize=5); heatmap_axis.set_xlabel("Stable-core patient"); heatmap_axis.set_title("Patient-level Hallmark ssGSEA signatures (row z-score)"); figure.colorbar(image, ax=heatmap_axis, label="ssGSEA z-score", shrink=.7); start = 0
    for core in sorted(cores): start += sum(case_id in case_ids for case_id in cores[core]); heatmap_axis.axvline(start - .5, color="black", linewidth=.8); annotation_axis.axvline(start - .5, color="black", linewidth=.8)
    figure.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(.72, .99), ncol=2, fontsize=6, frameon=False); figure.tight_layout(rect=[0, 0, .7, .96]); save_figure(figure, path); return len(case_ids)


def plot_umap(path, similarity, ids, cores, random_state):
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    normalized = affinity_characterization(similarity, ids, cores)["similarity"]; distance = 1 - normalized; method = "umap_precomputed"
    try:
        import umap
        coordinates = umap.UMAP(n_components=2, metric="precomputed", random_state=random_state).fit_transform(distance)
    except Exception:
        from sklearn.manifold import MDS
        coordinates = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state, normalized_stress="auto").fit_transform(distance); method = "mds_fallback_umap_unavailable"
    labels = {case_id: core for core, members in cores.items() for case_id in members}; figure, axis = plt.subplots(figsize=(8, 6))
    for core in sorted(cores):
        selected = [index for index, case_id in enumerate(ids) if labels.get(case_id) == core]
        if selected: axis.scatter(coordinates[selected, 0], coordinates[selected, 1], label=core, s=22)
    axis.set_title("Exploratory visualization: affinity UMAP"); axis.set_xlabel("UMAP-1"); axis.set_ylabel("UMAP-2"); axis.legend(title="CORE", bbox_to_anchor=(1.02, 1), loc="upper left"); figure.tight_layout(); save_figure(figure, path); return method


def plot_pcoa_mds(path, similarity, ids, cores, random_state):
    from utils.visualization import configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    normalized = affinity_characterization(similarity, ids, cores)["similarity"]; distance = np.maximum(0, 1 - normalized); centered = -.5 * distance ** 2; centered -= centered.mean(axis=0, keepdims=True); centered -= centered.mean(axis=1, keepdims=True); centered += centered.mean(); eigenvalues, eigenvectors = np.linalg.eigh(centered); positive = np.flatnonzero(eigenvalues > 1e-10)[::-1]
    if len(positive) >= 2:
        coordinates = eigenvectors[:, positive[:2]] * np.sqrt(eigenvalues[positive[:2]]); method, axis_labels = "pcoa", ("PCoA-1", "PCoA-2")
    else:
        from sklearn.manifold import MDS
        coordinates = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state, normalized_stress="auto").fit_transform(distance); method, axis_labels = "mds", ("MDS-1", "MDS-2")
    labels = {case_id: core for core, members in cores.items() for case_id in members}; figure, axis = plt.subplots(figsize=(8, 6))
    for core in sorted(cores):
        selected = [index for index, case_id in enumerate(ids) if labels.get(case_id) == core]
        if selected: axis.scatter(coordinates[selected, 0], coordinates[selected, 1], label=core, s=22)
    axis.set_title(f"Exploratory visualization: fused affinity {method.upper()}"); axis.set_xlabel(axis_labels[0]); axis.set_ylabel(axis_labels[1]); axis.legend(title="CORE", bbox_to_anchor=(1.02, 1), loc="upper left"); figure.tight_layout(); save_figure(figure, path); return method


def build_summary(cores, all_ids, mapping_rows, separation_rows, confound_rows, **counts):
    core_ids = set().union(*(set(members) for members in cores.values())) if cores else set()
    return {"patient_count": len(all_ids), "stable_core_patient_count": len(core_ids), "non_core_patient_count": len(set(all_ids) - core_ids), "core_count": len(cores), "mapping_rows": len(mapping_rows), "embedding_rows": len(separation_rows), "confound_rows": len(confound_rows), **counts}


def run(data_root: Path, experiment_root: Path, output_root: Path, config_dir: Path, top_pathways=25, top_cnv=25, random_state=42, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force: raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists(): shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True); figure_root = output_root / "figures"; figure_root.mkdir(); states, all_ids = load_states(data_root); cores, core_ids = load_cores(experiment_root); main_sets = load_main_sets(data_root); mapping_rows, composition = main_mapping(cores, main_sets); write_csv(output_root / "stable_core_main_mapping.csv", mapping_rows); write_csv(output_root / "stable_core_main_composition.csv", [{"core_id": core, **counts} for core, counts in composition.items()])
    ct_features, ct_table = ct_radiomics_table(states, all_ids); ct_rows, ct_pair_rows = radiomics_analysis(ct_table, ct_features, cores, all_ids); write_csv(output_root / "ct_radiomics_core_vs_rest.csv", ct_rows); write_csv(output_root / "ct_radiomics_core_pairwise.csv", ct_pair_rows)
    rna_pathways, rna_scores = load_expression_scores(states, config_dir); rna_rows, rna_pair_rows = rna_analysis(states, config_dir, cores, all_ids, rna_pathways, rna_scores); write_csv(output_root / "rna_hallmark_core_vs_rest.csv", rna_rows); write_csv(output_root / "rna_hallmark_core_pairwise.csv", rna_pair_rows); wxs_features, wxs_table = load_table(data_root / "wxs/wxs_discovery_features.csv"); mutation_features = [x for x in wxs_features if x.startswith("mutation::")]; wxs_rows, wxs_pair_rows = binary_analysis(wxs_table, mutation_features, cores, all_ids, "mutation"); write_csv(output_root / "wxs_core_vs_rest.csv", wxs_rows); write_csv(output_root / "wxs_core_pairwise.csv", wxs_pair_rows); cnv_features, cnv_table = load_table(data_root / "cnv/case_features.csv"); cnv_cont, cnv_event, cnv_pair_cont, cnv_pair_event = cnv_analysis(cnv_table, cnv_features, cores, all_ids); write_csv(output_root / "cnv_continuous_core_vs_rest.csv", cnv_cont); write_csv(output_root / "cnv_gain_loss_core_vs_rest.csv", cnv_event); write_csv(output_root / "cnv_continuous_core_pairwise.csv", cnv_pair_cont); write_csv(output_root / "cnv_gain_loss_core_pairwise.csv", cnv_pair_event)
    affinity_paths = {"ct": data_root / "candidate_subtype/ct_affinity.npy", "wsi": data_root / "candidate_subtype/wsi_affinity.npy", "rna": data_root / "candidate_subtype/rna_affinity.npy", "genomic": data_root / "wxs/genomic_affinity.npy", "fused": data_root / "candidate_subtype/fused_similarity.npy"}; affinity_ids = json.loads((data_root / "candidate_subtype/affinity_patient_order.json").read_text(encoding="utf-8")); affinities = {name: np.load(path) for name, path in affinity_paths.items() if path.is_file()}; separation_rows, distance_matrices, tests, audits = core_embedding_analysis(affinities, affinity_ids, cores); write_csv(output_root / "embedding_core_separation.csv", separation_rows); [write_csv(output_root / f"{modality}_core_distance_matrix.csv", [{"core_id": core, **values} for core, values in matrix.items()]) for modality, matrix in distance_matrices.items()]; write_json(output_root / "affinity_audit.json", audits); write_csv(output_root / "core_permanova.csv", [{"modality": modality, **tests[modality]["permanova"]} for modality in tests]); write_csv(output_root / "core_permdisp.csv", [{"modality": modality, **tests[modality]["permdisp"]} for modality in tests])
    core_runs = load_core_runs(experiment_root); co_run, co_k = cooccurrence_from_runs(core_runs, cores); write_csv(output_root / "stable_core_cooccurrence_by_run.csv", co_run); write_csv(output_root / "stable_core_cooccurrence_by_k.csv", co_k); confounds = core_confounds(data_root, config_dir, states, cores); write_csv(output_root / "stable_core_confounders.csv", confounds)
    from tools.subtype_review_common import clinical_table
    clinical_records = clinical_table(states); write_csv(output_root / "clinical_patient_records.csv", [{"case_id": case_id, "core_id": next((core for core, members in cores.items() if case_id in members), "non_core"), **record} for case_id, record in clinical_records.items()]); clinical_availability_rows, clinical_availability_summary = clinical_availability(clinical_records, cores, all_ids); write_csv(output_root / "clinical_availability.csv", clinical_availability_rows); write_json(output_root / "clinical_availability_summary.json", clinical_availability_summary); clinical_rows, clinical_pair_rows, survival_rows = clinical_analysis(clinical_records, cores, all_ids, clinical_availability_summary["eligible_variables"], clinical_availability_summary["survival_eligible"]); write_csv(output_root / "clinical_core_vs_rest.csv", clinical_rows); write_csv(output_root / "clinical_core_pairwise.csv", clinical_pair_rows); write_csv(output_root / "clinical_survival_core_vs_rest.csv", survival_rows)
    similarity_rows = update_pairwise_similarity(pairwise_similarity(cores, rna_rows, wxs_rows, cnv_cont, distance_matrices, co_run, co_k), rna_pair_rows, wxs_pair_rows, cnv_pair_cont); write_csv(output_root / "stable_core_pairwise_similarity.csv", similarity_rows)
    core_order = sorted(cores)
    overview_rows = []
    for core in core_order:
        top_ct = sorted((row for row in ct_rows if row["core_id"] == core), key=lambda row: (row.get("q_value") is None, row.get("q_value") if row.get("q_value") is not None else 1, -abs(row.get("effect_size") or 0), row["feature"]))
        top_clinical = sorted((row for row in clinical_rows if row["core_id"] == core), key=lambda row: (row.get("q_value") is None, row.get("q_value") if row.get("q_value") is not None else 1, row["clinical_variable"], row.get("level", "")))
        main = max(main_sets, key=lambda item: composition[core].get(item["set_id"], 0), default={})
        top_signal = ""
        if top_clinical:
            top_signal = f'{top_clinical[0]["clinical_variable"]}={top_clinical[0].get("level", "")}' if top_clinical[0].get("level") else top_clinical[0]["clinical_variable"]
        overview_rows.append({"core_id": core, "core_n": len(cores[core]), "main_candidate": main.get("set_id", ""), "top_ct_radiomics_feature": top_ct[0]["feature"] if top_ct else "", "top_ct_radiomics_effect_size": top_ct[0].get("effect_size") if top_ct else None, "top_ct_radiomics_q_value": top_ct[0].get("q_value") if top_ct else None, "top_clinical_signal": top_signal, "top_clinical_q_value": top_clinical[0].get("q_value") if top_clinical else None, "overall_survival_q_value": next((row.get("q_value") for row in survival_rows if row["core_id"] == core), None)})
    write_csv(output_root / "stable_core_summary_multimodal.csv", overview_rows)
    plot_patient_coassignment(figure_root / "patient_core_coassignment_heatmap", core_runs, cores); selected_pathways = select_pathways(rna_rows, top_pathways); plot_patient_rna_heatmap(figure_root / "patient_rna_signature_heatmap", rna_scores, selected_pathways, cores, clinical_records); rna_lookup = {(row["core_id"], row["pathway"]): row for row in rna_rows}; pathway_matrix = np.asarray([[rna_lookup.get((core, pathway), {}).get("smd") or 0 for core in core_order] for pathway in selected_pathways]); stars = [["***" if (rna_lookup.get((core, pathway), {}).get("q_value") or 1) < .001 else "**" if (rna_lookup.get((core, pathway), {}).get("q_value") or 1) < .01 else "*" if (rna_lookup.get((core, pathway), {}).get("q_value") or 1) < .05 else "" for core in core_order] for pathway in selected_pathways]; plot_heatmap(figure_root / "pathway_smd_heatmap", pathway_matrix, selected_pathways, core_order, "Hallmark pathway SMD", stars, True); plot_bubbles(figure_root / "pathway_bubble_plot", rna_rows, core_order, selected_pathways); selected_cnv = select_cnv_heatmap_features(cnv_cont, top_cnv); cnv_lookup = {(row["core_id"], row["feature"]): row for row in cnv_cont}; plot_heatmap(figure_root / "cnv_effect_heatmap", np.asarray([[cnv_lookup.get((core, feature), {}).get("cliffs_delta") or 0 for core in core_order] for feature in selected_cnv]), selected_cnv, core_order, "CNV continuous Cliff's delta", diverging=True); plot_oncoplot(figure_root / "driver_mutation_oncoplot", wxs_table, mutation_features, cores, all_ids, {core: max(main_sets, key=lambda item: composition[core].get(item["set_id"], 0), default={}).get("set_id", "") for core in cores}, {core: next((row.get("level", "") for row in confounds if row["core_id"] == core and row["field"] == "tissue_source_site" and row.get("q_value") is not None), "") for core in cores}); plot_survival_km(figure_root / "clinical_overall_survival_km", clinical_records, cores)
    projection_methods = {modality: plot_pcoa_mds(figure_root / "fused_core_pcoa_mds", matrix, affinity_ids, cores, random_state) if modality == "fused" else plot_umap(figure_root / f"{modality}_core_umap", matrix, affinity_ids, cores, random_state) for modality, matrix in affinities.items()}; generated = sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()); main_partition = data_root / "subtype_review/final_partition_sets.json"; manifest = {"core_count": len(cores), "patient_count": len(all_ids), "core_patient_count": len(core_ids), "non_core_patient_count": len(set(all_ids) - set(core_ids)), "source_multi_k_summary_sha256": file_sha256(experiment_root / "stable_core_summary.csv"), "source_main_partition_sha256": file_sha256(main_partition) if main_partition.exists() else None, "analysis_parameters": {"top_pathways": top_pathways, "top_cnv": top_cnv, "random_state": random_state, "permutations": 999, "projection_methods": projection_methods}, "generated_files": generated}; write_json(output_root / "stable_core_analysis_manifest.json", manifest); summary = build_summary(cores, all_ids, mapping_rows, separation_rows, confounds, ct_radiomics_core_vs_rest_rows=len(ct_rows), ct_radiomics_core_pairwise_rows=len(ct_pair_rows), clinical_patient_record_count=len(clinical_records), clinical_availability_rows=len(clinical_availability_rows), clinical_eligible_variables=clinical_availability_summary["eligible_variables"], clinical_excluded_variables=clinical_availability_summary["excluded_variables"], clinical_core_vs_rest_rows=len(clinical_rows), clinical_core_pairwise_rows=len(clinical_pair_rows), clinical_survival_rows=len(survival_rows)); write_json(output_root / "stable_core_analysis_summary.json", summary); return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path("output_kirc")); parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v11/experiment_multi_k_accepted_core_stability")); parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v11/stable_core_multimodal_analysis")); parser.add_argument("--config-dir", type=Path, default=Path("configs")); parser.add_argument("--top-pathways", type=int, default=25); parser.add_argument("--top-cnv", type=int, default=25); parser.add_argument("--random-state", type=int, default=42); parser.add_argument("--force", action="store_true"); print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
