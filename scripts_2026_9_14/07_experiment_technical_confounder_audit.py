#!/usr/bin/env python3
"""Offline audit of available technical variables across the five views.

This experiment does not correct data or infer unavailable batch metadata.
It reports measurable QC proxies and explicitly records unavailable fields.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts_2026_9_7 import analyze_multi_k_stable_cores as base
from tools import confound


TECHNICAL_VARIABLES = (
    "tissue_source_site", "ct_phase", "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel",
    "ct_slice_thickness", "ct_z_spacing", "ct_pixel_spacing", "ct_n_images", "ct_study_year",
    "wsi_patch_count", "wsi_tumor_patch_count", "wsi_tumor_patch_fraction",
    "wsi_model_mpp", "wsi_model_tile_size",
    "wsi_scanner_vendor", "wsi_specimen_type", "wsi_slide_count", "wsi_stain_summary",
    "rna_file_presence", "rna_batch", "rna_sequencing_center", "rna_plate", "rna_library_protocol",
    "wxs_file_presence", "wxs_discovery_mutation_count", "wxs_discovery_all_zero_proxy",
    "wxs_sequencing_center", "wxs_capture_platform", "wxs_median_depth", "wxs_callable_bases", "wxs_purity",
    "cnv_missing_feature_count", "cnv_feature_completeness_proxy", "cnv_gain_burden", "cnv_loss_burden",
    "cnv_platform", "cnv_center", "cnv_purity", "cnv_ploidy", "cnv_segment_quality",
)

AVAILABLE_SOURCES = {
    "ct": "tools.confound + ct_qc metadata",
    "wsi": "wsi_tumor_seg/*/summary.json",
    "rna": "patient_states inventory",
    "wxs": "wxs manifest and discovery features",
    "cnv": "cnv/case_features.csv",
}


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def build_availability_rows():
    available = {
        "tissue_source_site": ("available", "ct", "TCGA site identifier; technical/site proxy"),
        "ct_phase": ("available", "ct", "categorical CT acquisition metadata"),
        "ct_manufacturer": ("available", "ct", "categorical CT acquisition metadata"),
        "ct_scanner_model": ("available", "ct", "categorical CT acquisition metadata"),
        "ct_reconstruction_kernel": ("available", "ct", "categorical CT acquisition metadata"),
        "ct_slice_thickness": ("available", "ct", "numeric CT acquisition metadata"),
        "ct_z_spacing": ("available", "ct", "numeric CT acquisition metadata"),
        "ct_pixel_spacing": ("available", "ct", "numeric CT acquisition metadata"),
        "ct_n_images": ("available", "ct", "selected CT metadata"),
        "ct_study_year": ("available", "ct", "selected CT metadata"),
        "wsi_patch_count": ("available", "wsi", "generated patch-count proxy"),
        "wsi_tumor_patch_count": ("available", "wsi", "generated tumor-patch-count proxy"),
        "wsi_tumor_patch_fraction": ("available", "wsi", "derived WSI QC proxy"),
        "wsi_model_mpp": ("available", "wsi", "embedding/segmentation model setting"),
        "wsi_model_tile_size": ("available", "wsi", "embedding/segmentation model setting"),
        "rna_file_presence": ("available", "rna", "file-presence only; not batch QC"),
        "wxs_file_presence": ("available", "wxs", "file-presence only; not sequencing QC"),
        "wxs_discovery_mutation_count": ("available", "wxs", "representation proxy"),
        "wxs_discovery_all_zero_proxy": ("available", "wxs", "representation proxy, not missingness"),
        "cnv_feature_completeness_proxy": ("available", "cnv", "feature-table completeness proxy"),
        "cnv_missing_feature_count": ("available", "cnv", "feature-table completeness proxy"),
        "cnv_gain_burden": ("available", "cnv", "derived CNV representation feature"),
        "cnv_loss_burden": ("available", "cnv", "derived CNV representation feature"),
    }
    biological = {"wxs_discovery_mutation_count", "cnv_gain_burden", "cnv_loss_burden"}
    qc_proxy = {"wsi_patch_count", "wsi_tumor_patch_count", "wsi_tumor_patch_fraction", "wsi_model_mpp", "wsi_model_tile_size", "rna_file_presence", "wxs_file_presence", "wxs_discovery_all_zero_proxy", "cnv_missing_feature_count", "cnv_feature_completeness_proxy"}
    rows = []
    for variable in TECHNICAL_VARIABLES:
        status, modality, interpretation = available.get(
            variable, ("unavailable", variable.split("_", 1)[0], "not present in current artifacts")
        )
        rows.append({
            "variable": variable,
            "modality": modality,
            "status": status,
            "source": AVAILABLE_SOURCES.get(modality, "not available"),
            "role": "biological_representation" if variable in biological else "qc_proxy" if variable in qc_proxy else "technical_metadata",
            "interpretation": interpretation,
        })
    return rows


def build_wsi_proxy(case_id, summary):
    patch_count = finite(summary.get("patch_count"))
    tumor_patch_count = finite(summary.get("tumor_patch_count"))
    return {
        "case_id": case_id,
        "wsi_patch_count": patch_count,
        "wsi_tumor_patch_count": tumor_patch_count,
        "wsi_tumor_patch_fraction": tumor_patch_count / patch_count if patch_count else None,
        "wsi_model_mpp": finite(summary.get("model_mpp")),
        "wsi_model_tile_size": finite(summary.get("model_tile_size")),
    }


def build_wxs_proxy(case_id, row):
    values = [finite(value) for key, value in row.items() if str(key).startswith("mutation::")]
    values = [value for value in values if value is not None]
    all_zero = bool(values) and all(value == 0 for value in values)
    return {
        "case_id": case_id,
        "wxs_discovery_mutation_count": int(sum(value > 0 for value in values)),
        "wxs_discovery_all_zero_proxy": all_zero,
        "wxs_discovery_all_zero_proxy_interpretation": "representation_proxy_not_qc",
    }


def build_cnv_proxy(case_id, row):
    feature_values = [finite(value) for key, value in row.items() if key != "case_id"]
    return {
        "case_id": case_id,
        "cnv_missing_feature_count": int(sum(value is None for value in feature_values)),
        "cnv_feature_completeness_proxy": int(sum(value is not None for value in feature_values)),
        "cnv_gain_burden": finite(row.get("gain_burden")),
        "cnv_loss_burden": finite(row.get("loss_burden")),
    }


def bh(values):
    valid = sorted(
        ((i, float(value)) for i, value in enumerate(values) if value is not None and math.isfinite(float(value))),
        key=lambda item: item[1],
    )
    result = [None] * len(values)
    for rank, (index, value) in enumerate(valid, 1):
        result[index] = min(1.0, value * len(valid) / rank)
    for i in range(len(valid) - 2, -1, -1):
        result[valid[i][0]] = min(result[valid[i][0]], result[valid[i + 1][0]])
    return result


def primary_core_groups(cores):
    names = ("CORE01", "CORE02", "CORE03", "CORE04")
    if set(cores) != set(names):
        raise ValueError(f"Expected exactly four stable cores: {names}")
    return {name: list(cores[name]) for name in names}


def primary_categorical_rows(records, groups, fields):
    rows = []
    for field in fields:
        levels = sorted({str(records[case].get(field, "")) for cases in groups.values() for case in cases if records[case].get(field) not in (None, "")})
        table = np.asarray([[sum(str(records[case].get(field, "")) == level for case in cases) for level in levels] for cases in groups.values()], dtype=int)
        p_value, test = None, "not_testable"
        if len(levels) > 1 and table.sum() > 0:
            expected = chi2_contingency(table, correction=False)[3]
            observed = float(chi2_contingency(table, correction=False)[0])
            if (expected < 1).any() or (expected < 5).mean() > 0.2:
                values = np.asarray([str(records[case].get(field, "")) for cases in groups.values() for case in cases])
                labels = np.concatenate([np.full(len(cases), index) for index, cases in enumerate(groups.values())])
                rng = np.random.default_rng(20260915 + fields.index(field))
                exceed = 0
                for _ in range(10000):
                    shuffled = rng.permutation(labels)
                    permuted = np.asarray([[np.sum((shuffled == index) & (values == level)) for level in levels] for index in range(len(groups))])
                    exceed += chi2_contingency(permuted, correction=False)[0] >= observed
                p_value, test = (exceed + 1) / 10001, "monte_carlo_chi2"
            else:
                p_value, test = float(chi2_contingency(table, correction=False)[1]), "pearson_chi2"
        rows.append({"variable": field, "grouping": "four_stable_cores", "levels": json.dumps(levels), "counts": json.dumps(table.tolist()), "test": test, "cramers_v": confound.cramers_v(table) if p_value is not None else None, "p_value": p_value})
    for row, q_value in zip(rows, bh([row["p_value"] for row in rows])):
        row["q_value"] = q_value
    return rows


def primary_numeric_rows(records, groups, fields):
    rows = []
    for field in fields:
        values = [[finite(records[case].get(field)) for case in cases] for cases in groups.values()]
        values = [[value for value in group if value is not None] for group in values]
        nonempty = [group for group in values if group]
        if len(nonempty) >= 2 and len({value for group in nonempty for value in group}) > 1:
            p_value = float(kruskal(*nonempty).pvalue)
            test = "kruskal_wallis"
            effect = confound.epsilon_squared(nonempty)
        else:
            p_value, test, effect = None, "invariant_or_not_testable", None
        rows.append({"variable": field, "grouping": "four_stable_cores", "group_n": json.dumps([len(group) for group in values]), "group_medians": json.dumps([float(np.median(group)) if group else None for group in values]), "test": test, "epsilon_squared": effect, "p_value": p_value})
    for row, q_value in zip(rows, bh([row["p_value"] for row in rows])):
        row["q_value"] = q_value
    return rows


def compare_proxies(records, core_ids):
    rows = []
    fields = (
        "wsi_patch_count", "wsi_tumor_patch_count", "wsi_tumor_patch_fraction", "wsi_model_mpp",
        "wsi_model_tile_size", "wxs_discovery_mutation_count", "wxs_discovery_all_zero_proxy",
        "cnv_missing_feature_count", "cnv_feature_completeness_proxy", "cnv_gain_burden", "cnv_loss_burden",
    )
    core = set(core_ids)
    for field in fields:
        left = [records[case].get(field) for case in sorted(core)]
        right = [records[case].get(field) for case in sorted(set(records) - core)]
        left = [value for value in left if isinstance(value, (int, float, bool)) and math.isfinite(float(value))]
        right = [value for value in right if isinstance(value, (int, float, bool)) and math.isfinite(float(value))]
        p_value = float(mannwhitneyu(left, right, alternative="two-sided").pvalue) if left and right else None
        rows.append({
            "variable": field,
            "core_n": len(left),
            "non_core_n": len(right),
            "core_median": float(np.median(left)) if left else None,
            "non_core_median": float(np.median(right)) if right else None,
            "test": "mann_whitney_u",
            "p_value": p_value,
        })
    for row, q_value in zip(rows, bh([row["p_value"] for row in rows])):
        row["q_value"] = q_value
    return rows


def load_inputs(data_root, multi_k_root):
    patient_ids = [str(x) for x in json.loads((data_root / "candidate_subtype/affinity_patient_order.json").read_text())]
    cores, core_ids = base.load_cores(multi_k_root)
    states = {}
    with (data_root / "storage/patient_states/patient_states.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                states[str(state["case_id"])] = state
    if set(patient_ids) - set(states) or not set(core_ids).issubset(patient_ids):
        raise ValueError("Current patient-state and stable-core artifacts do not cover the same cohort")
    return patient_ids, cores, core_ids, states


def run(data_root, multi_k_root, output_root, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    patient_ids, cores, core_ids, states = load_inputs(data_root, multi_k_root)
    technical = confound.confounder_values(states, str(data_root))
    wxs = pd.read_csv(data_root / "wxs/wxs_discovery_features.csv").set_index("case_id").to_dict("index")
    cnv = pd.read_csv(data_root / "cnv/case_features.csv").set_index("case_id").to_dict("index")
    records = {}
    for case_id in patient_ids:
        inventory = states[case_id].get("inventory", {}) or {}
        summary_path = data_root / "wsi_tumor_seg" / case_id / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        record = {"case_id": case_id, "group": "core" if case_id in core_ids else "non_core"}
        record.update(technical.get(case_id, {}))
        record.update(build_wsi_proxy(case_id, summary))
        record.update(build_wxs_proxy(case_id, wxs.get(case_id, {})))
        record.update(build_cnv_proxy(case_id, cnv.get(case_id, {})))
        record["rna_file_presence"] = bool(inventory.get("RNA_Seq"))
        record["wxs_file_presence"] = bool(inventory.get("WXS"))
        records[case_id] = record

    base.write_csv(output_root / "technical_variable_availability.csv", build_availability_rows())
    base.write_csv(output_root / "technical_proxy_by_case.csv", list(records.values()))
    base.write_csv(output_root / "technical_proxy_core_non_core.csv", compare_proxies(records, core_ids))
    primary_groups = primary_core_groups(cores)
    primary_categorical_fields = ("tissue_source_site", "ct_phase", "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel")
    primary_numeric_fields = ("ct_slice_thickness", "ct_z_spacing", "ct_pixel_spacing", "ct_n_images", "ct_study_year")
    base.write_csv(output_root / "technical_primary_core_categorical.csv", primary_categorical_rows(records, primary_groups, primary_categorical_fields))
    base.write_csv(output_root / "technical_primary_core_numeric.csv", primary_numeric_rows(records, primary_groups, primary_numeric_fields))
    unavailable = [row for row in build_availability_rows() if row["status"] == "unavailable"]
    summary = {
        "experiment": "five_view_technical_confounder_audit",
        "patient_count": len(patient_ids),
        "core_count": len(core_ids),
        "non_core_count": len(patient_ids) - len(core_ids),
        "technical_audit_is_correction": False,
        "primary_comparison": "CORE01 vs CORE02 vs CORE03 vs CORE04",
        "secondary_comparison": "all stable-core patients vs non-core patients",
        "available_variable_count": len(TECHNICAL_VARIABLES) - len(unavailable),
        "unavailable_variable_count": len(unavailable),
        "unavailable_variables": [row["variable"] for row in unavailable],
        "interpretation": "Measured QC proxies are audited; unavailable batch, platform, depth, purity, and ploidy metadata are not inferred.",
        "stable_core_ids": sorted(cores),
    }
    base.write_json(output_root / "technical_confounder_audit_summary.json", summary)
    base.write_json(output_root / "unavailable_technical_metadata.json", {"variables": unavailable})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("output_kirc"))
    parser.add_argument("--multi-k-root", type=Path, default=Path("output_kirc_v13/00_five_view_multi_k_agent_review"))
    parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v14/07_technical_confounder_audit"))
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
