#!/usr/bin/env python3
"""Offline audit of selection differences between stable cores and non-core cases."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts_2026_9_7 import analyze_multi_k_stable_cores as base
from tools import confound
from tools.subtype_review_common import clinical_table


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def bh(values):
    valid = sorted(
        ((i, float(value)) for i, value in enumerate(values) if value is not None and math.isfinite(value)),
        key=lambda item: item[1],
    )
    result = [None] * len(values)
    for rank, (index, value) in enumerate(valid, 1):
        result[index] = min(1.0, value * len(valid) / rank)
    for i in range(len(valid) - 2, -1, -1):
        result[valid[i][0]] = min(result[valid[i][0]], result[valid[i + 1][0]])
    return result


def load_inputs(data_root, multi_k_root):
    order = json.loads((data_root / "candidate_subtype/affinity_patient_order.json").read_text())
    patient_ids = [str(item) for item in order]
    cores, core_ids = base.load_cores(multi_k_root)
    if not set(core_ids).issubset(patient_ids):
        raise ValueError("Stable-core membership contains patients outside the current affinity cohort")
    if len(core_ids) != 56 or len(patient_ids) - len(core_ids) != 47:
        raise ValueError(f"Expected 56 core and 47 non-core patients, got {len(core_ids)} and {len(patient_ids) - len(core_ids)}")
    states = {}
    with (data_root / "storage/patient_states/patient_states.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                states[str(state["case_id"])] = state
    if set(patient_ids) - set(states):
        raise ValueError("Patient-state records do not cover the affinity cohort")
    return patient_ids, cores, set(core_ids), states


def categorical_rows(records, groups, fields):
    rows = []
    for field in fields:
        levels = sorted({str(records[case].get(field, "")) for case in records if records[case].get(field) is not None and str(records[case].get(field)) != ""})
        table = [[sum(str(records[case].get(field)) == level for case in members) for level in levels] for members in groups.values()]
        table = np.asarray(table, dtype=int)
        p_value, odds_ratio, test = None, None, None
        if len(levels) > 1 and np.all(table.sum(axis=0) > 0):
            if table.shape == (2, 2):
                odds_ratio, p_value = map(float, fisher_exact(table))
                test = "fisher_exact"
            else:
                p_value = float(chi2_contingency(table, correction=False)[1])
                test = "pearson_chi2"
        v = confound.cramers_v(table) if p_value is not None else None
        for level in levels:
            core_n = sum(str(records[case].get(field)) == level for case in groups["core"])
            non_core_n = sum(str(records[case].get(field)) == level for case in groups["non_core"])
            rows.append({"field": field, "level": level, "core_n": core_n, "non_core_n": non_core_n, "core_fraction": core_n / len(groups["core"]), "non_core_fraction": non_core_n / len(groups["non_core"]), "cramers_v": v, "odds_ratio": odds_ratio, "test": test, "p_value": p_value, "q_value": None})
    tests = {}
    for row in rows:
        tests[row["field"]] = row["p_value"]
    q_values = bh([tests.get(field) for field in fields])
    for row in rows:
        row["q_value"] = q_values[fields.index(row["field"])]
    return rows


def numeric_rows(records, groups, fields):
    rows = []
    for field in fields:
        left = [finite(records[case].get(field)) for case in groups["core"]]
        right = [finite(records[case].get(field)) for case in groups["non_core"]]
        left, right = [value for value in left if value is not None], [value for value in right if value is not None]
        p_value = float(mannwhitneyu(left, right, alternative="two-sided").pvalue) if left and right else None
        delta = base.cliffs_delta(left, right)
        rows.append({"field": field, "core_n": len(left), "non_core_n": len(right), "core_median": float(np.median(left)) if left else None, "non_core_median": float(np.median(right)) if right else None, "cliffs_delta": delta, "p_value": p_value, "q_value": None})
    for row, q_value in zip(rows, bh([row["p_value"] for row in rows])):
        row["q_value"] = q_value
    return rows


def run(data_root, multi_k_root, output_root, reference_coverage, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    patient_ids, cores, core_ids, states = load_inputs(data_root, multi_k_root)
    clinical = clinical_table(states)
    technical = confound.confounder_values(states, str(data_root))
    records = {case: {**clinical.get(case, {}), **technical.get(case, {})} for case in patient_ids}
    coverage = {row["case_id"]: row for row in base.read_csv(reference_coverage)}
    if set(patient_ids) - set(coverage):
        raise ValueError("Known-label coverage does not cover the affinity cohort")
    for case in patient_ids:
        records[case].update({
            "has_mrna_label": coverage[case].get("has_mrna_label"),
            "has_clearcode_label": coverage[case].get("has_clearcode_label"),
        })
    groups = {"core": sorted(core_ids), "non_core": sorted(set(patient_ids) - core_ids)}
    categorical_fields = ("stage_group", "t_stage", "m_stage", "grade", "gender", "race", "os_event", "tissue_source_site", "ct_phase", "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel", "has_mrna_label", "has_clearcode_label")
    numeric_fields = ("age", "os_time", "ct_slice_thickness", "ct_z_spacing", "ct_pixel_spacing", "ct_n_images", "ct_study_year")
    categorical = categorical_rows(records, groups, categorical_fields)
    numeric = numeric_rows(records, groups, numeric_fields)
    known_m = {case: record for case, record in records.items() if record.get("m_stage") in {"M0", "M1"}}
    known_groups = {group: [case for case in cases if case in known_m] for group, cases in groups.items()}
    m_known = categorical_rows(known_m, known_groups, ("m_stage",))
    m_missing = [{"group": group, "known_m_n": sum(case in known_m for case in cases), "unknown_m_n": sum(case not in known_m for case in cases), "total_n": len(cases)} for group, cases in groups.items()]
    missing_table = np.asarray([[row["known_m_n"], row["unknown_m_n"]] for row in m_missing])
    missing_or, missing_p = map(float, fisher_exact(missing_table))
    missing_audit = {"comparison": "known_M_vs_unknown_M", "core_known": m_missing[0]["known_m_n"], "core_unknown": m_missing[0]["unknown_m_n"], "non_core_known": m_missing[1]["known_m_n"], "non_core_unknown": m_missing[1]["unknown_m_n"], "odds_ratio": missing_or, "p_value": missing_p, "q_value": None, "test": "fisher_exact"}
    known_m_rows = [row for row in m_known]
    known_m_q = bh([missing_p, m_known[0]["p_value"] if m_known else None])
    missing_audit["q_value"], q_known = known_m_q[0], known_m_q[1]
    for row in known_m_rows:
        row["q_value"] = q_known
    quality = []
    for case in patient_ids:
        state = states[case]
        inventory = state.get("inventory", {}) or {}
        quality.append({"case_id": case, "group": "core" if case in core_ids else "non_core", "qc": state.get("qc"), **{f"{view}_available": bool(inventory.get(view)) for view in ("CT", "WSI", "RNA_Seq", "WXS", "CNV")}})
    base.write_csv(output_root / "core_non_core_categorical.csv", categorical)
    base.write_csv(output_root / "core_non_core_numeric.csv", numeric)
    base.write_csv(output_root / "core_non_core_m_stage_known_only.csv", m_known)
    base.write_csv(output_root / "core_non_core_m_stage_missingness.csv", m_missing)
    base.write_csv(output_root / "core_non_core_m_stage_missingness_test.csv", [missing_audit])
    base.write_csv(output_root / "core_non_core_view_quality.csv", quality)
    summary = {"patient_count": len(patient_ids), "core_count": len(core_ids), "non_core_count": len(patient_ids) - len(core_ids), "core_ids": sorted(cores), "categorical_fields": list(categorical_fields), "numeric_fields": list(numeric_fields), "q_value_family": "within audit table", "interpretation": "descriptive selection-bias audit; no subtype assignment"}
    base.write_json(output_root / "core_non_core_audit_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("output_kirc"))
    parser.add_argument("--multi-k-root", type=Path, default=Path("output_kirc_v13/00_five_view_multi_k_agent_review"))
    parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v13/05_core_non_core_audit"))
    parser.add_argument("--reference-coverage", type=Path, default=Path("output_kirc_v13/03_known_ccrcc_subtype_mapping/reference_coverage_all_cases.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
