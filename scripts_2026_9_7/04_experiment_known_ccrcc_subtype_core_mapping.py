#!/usr/bin/env python3
"""Map frozen stable cores to published ccRCC reference taxonomies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("known_ccrcc_mapping", Path(__file__).with_name("03_experiment_known_ccrcc_subtype_mapping.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SEVEN_CORE_SIZES = {"CORE01": 17, "CORE02": 15, "CORE03": 14, "CORE04": 10, "CORE05": 9, "CORE06": 7, "CORE07": 5}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_core_membership(path, expected_sizes):
    frame = pd.read_csv(path, dtype=str)
    if not {"core_id", "patient_id"}.issubset(frame.columns):
        raise ValueError(f"Invalid stable-core membership columns: {path}")
    cores = {}
    for row in frame.to_dict("records"):
        case_id = BASE.normalize_tcga_patient_id(row["patient_id"])
        if case_id is None:
            raise ValueError(f"Invalid core patient ID: {row['patient_id']}")
        cores.setdefault(str(row["core_id"]), set()).add(case_id)
    if set(cores) != set(expected_sizes):
        raise ValueError(f"Unexpected core IDs in {path}: {sorted(cores)}")
    seen = set()
    for core_id, expected_n in expected_sizes.items():
        if len(cores[core_id]) != expected_n:
            raise ValueError(f"{core_id} expected {expected_n} patients, got {len(cores[core_id])}")
        overlap = seen & cores[core_id]
        if overlap:
            raise ValueError(f"Core patient overlap: {sorted(overlap)}")
        seen |= cores[core_id]
    return {core_id: sorted(members) for core_id, members in cores.items()}


def load_macro_core_map(path, core_ids):
    frame = pd.read_csv(path, dtype=str)
    if set(frame.columns) != {"core_id", "state_id"}:
        raise ValueError(f"Invalid macro-state core mapping columns: {path}")
    mapping = {}
    for row in frame.to_dict("records"):
        if row["core_id"] not in core_ids:
            raise ValueError(f"Unknown core in macro-state mapping: {row['core_id']}")
        mapping.setdefault(row["state_id"], []).append(row["core_id"])
    if set().union(*mapping.values()) != set(core_ids):
        raise ValueError("Macro-state mapping does not cover every stable core")
    return {state: tuple(cores) for state, cores in mapping.items()}


def load_universe(path):
    frame = pd.read_csv(path, dtype=str)
    if "case_id" not in frame.columns:
        raise ValueError(f"Analysis universe has no case_id: {path}")
    ids = [BASE.normalize_tcga_patient_id(value) for value in frame["case_id"]]
    if any(case_id is None for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid or duplicate IDs in {path}")
    return set(ids)


def validate_core_union(cores, universe):
    core_union = set().union(*cores.values())
    missing_from_universe = sorted(core_union - set(universe))
    extra_in_universe = sorted(set(universe) - core_union)
    if missing_from_universe or extra_in_universe:
        raise ValueError(f"Core/version mismatch: missing_from_universe={missing_from_universe}, extra_in_universe={extra_in_universe}")


def load_reference(path, allowed_labels):
    frame = pd.read_csv(path, dtype=str)
    required = {"case_id", "reference_subtype", "reference_status"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Reference file missing columns: {path}")
    output = {}
    for row in frame.to_dict("records"):
        case_id = BASE.normalize_tcga_patient_id(row["case_id"])
        raw_label = row.get("reference_subtype")
        raw_status = row.get("reference_status")
        label = "" if pd.isna(raw_label) else str(raw_label).strip()
        status = "" if pd.isna(raw_status) else str(raw_status).strip()
        if case_id is None:
            raise ValueError(f"Invalid reference case ID: {row['case_id']}")
        if label and label not in allowed_labels:
            raise ValueError(f"Invalid reference label {label} in {path}")
        if case_id in output:
            raise ValueError(f"Duplicate reference case ID: {case_id}")
        item = dict(row)
        item["case_id"] = case_id
        item["reference_subtype"] = label or None
        item["reference_status"] = status
        output[case_id] = item
    return output


def core_contingency(cores, labels, reference_labels):
    return np.asarray([[sum(labels.get(case_id) == label for case_id in cores[core_id]) for label in reference_labels] for core_id in cores], dtype=int)


def core_contingency_full(cores, labels, reference_labels):
    rows = []
    for core_id, members in cores.items():
        available = sum(bool(labels.get(case_id)) for case_id in members)
        row = {"core_id": core_id, **{label: sum(labels.get(case_id) == label for case_id in members) for label in reference_labels}}
        row.update({"missing_n": len(members) - available, "total_n": len(members), "coverage": available / len(members) if members else None})
        rows.append(row)
    return rows


def evidence_status(label_n, enrichment_rows):
    if label_n < 5:
        return "low_information"
    if any(row.get("bh_q") is not None and row["bh_q"] < .05 and float(row.get("odds_ratio") or 0) > 1 for row in enrichment_rows):
        return "significant_known_enrichment"
    return "no_fdr_supported_enrichment"


def no_fdr_supported_enrichment_in_either_reference(mrna_label_n, mrna_status, clearcode_label_n, clearcode_status):
    return bool(mrna_label_n >= 5 and clearcode_label_n >= 5 and mrna_status == "no_fdr_supported_enrichment" and clearcode_status == "no_fdr_supported_enrichment")


def core_composition_rows(cores, labels, reference_labels, partition, reference):
    table = core_contingency(cores, labels, reference_labels)
    rows = []
    for core_id, counts in zip(cores, table):
        label_n = int(counts.sum())
        missing_n = len(cores[core_id]) - label_n
        probabilities = counts / label_n if label_n else np.zeros(len(counts))
        entropy = -sum(float(value) * np.log(float(value)) for value in probabilities if value > 0)
        rows.append({"partition": partition, "reference": reference, "core_id": core_id, "core_total_n": len(cores[core_id]), "label_n": label_n, "missing_n": missing_n, "coverage": label_n / len(cores[core_id]), "dominant_reference_subtype": reference_labels[int(np.argmax(counts))] if label_n else None, "purity": BASE.rounded(float(probabilities.max())) if label_n else None, "normalized_entropy": BASE.rounded(entropy / np.log(len(reference_labels))) if label_n and len(reference_labels) > 1 else 0.0, **{f"{label}_n": int(count) for label, count in zip(reference_labels, counts)}})
    return rows


def component_heterogeneity_rows(partition, macros, cores, labels, reference, reference_labels, permutations):
    rows = []
    for macro_state, components in macros.items():
        if len(components) < 2:
            continue
        component_cores = {core_id: cores[core_id] for core_id in components}
        table = core_contingency(component_cores, labels, reference_labels)
        chi, p_value = BASE.permutation_chi_square(table, permutations, 20260940)
        rows.append({"partition": partition, "macro_state": macro_state, "component_cores": "+".join(components), "reference": reference, "matched_n": int(table.sum()), "chi_square": chi, "permutation_p": p_value, "cramers_v_raw": BASE.cramers_v_raw(table), "cramers_v_bias_corrected": BASE.bias_corrected_cramers_v(table)})
    return rows


def attach_component_bh(rows):
    BASE.attach_bh(rows, "permutation_p", "bh_q")


def macro_mixing_row(partition, macro_state, components, component_info, macro_counts, reference_labels, heterogeneity_p, heterogeneity_v):
    matched_n = sum(macro_counts.values())
    macro_purity = max(macro_counts.values()) / matched_n if matched_n else None
    weight = sum(component_info[core_id]["label_n"] for core_id in components)
    weighted_purity = sum(component_info[core_id]["label_n"] * component_info[core_id]["purity"] for core_id in components if component_info[core_id]["purity"] is not None) / weight if weight else None
    return {"partition": partition, "macro_state": macro_state, "reference": None, "component_core_n": len(components), "matched_n": matched_n, "macro_purity": BASE.rounded(macro_purity), "weighted_component_purity": BASE.rounded(weighted_purity), "purity_drop": BASE.rounded(weighted_purity - macro_purity) if weighted_purity is not None and macro_purity is not None else None, "heterogeneity_permutation_p": heterogeneity_p, "heterogeneity_cramers_v": heterogeneity_v}


def analyze_reference(partition, cores, labels, reference, reference_labels, output_root, permutations):
    slug = "mrna" if reference == "mRNA" else "clearcode"
    label_map = {case_id: item["reference_subtype"] for case_id, item in labels.items() if item.get("reference_subtype") and item.get("reference_status") != "conflict"}
    table = core_contingency(cores, label_map, reference_labels)
    full = core_contingency_full(cores, label_map, reference_labels)
    write_csv(output_root / f"{slug}_contingency_full.csv", full)
    write_csv(output_root / f"{slug}_contingency_counts.csv", [{"core_id": core_id, **{label: int(value) for label, value in zip(reference_labels, table[i])}} for i, core_id in enumerate(cores)])
    write_csv(output_root / f"{slug}_contingency_row_pct.csv", [{"core_id": core_id, **{label: BASE.rounded(value) for label, value in zip(reference_labels, row)}} for core_id, row in zip(cores, BASE.percentages(table, 1))])
    write_csv(output_root / f"{slug}_contingency_col_pct.csv", [{"core_id": core_id, **{label: BASE.rounded(value) for label, value in zip(reference_labels, row)}} for core_id, row in zip(cores, BASE.percentages(table, 0))])
    chi, p_value = BASE.permutation_chi_square(table, permutations, 20260941)
    global_row = {"partition": partition, "reference": reference, "matched_n": int(table.sum()), "group_count": len(cores), "reference_class_count": len(reference_labels), "chi_square": chi, "permutation_p": p_value, "cramers_v_raw": BASE.cramers_v_raw(table), "cramers_v_bias_corrected": BASE.bias_corrected_cramers_v(table)}
    state_map = {case_id: core_id for core_id, members in cores.items() for case_id in members}
    matched_ids = sorted(set(state_map) & set(label_map))
    global_row.update({"ari": BASE.rounded(adjusted_rand_score([state_map[x] for x in matched_ids], [label_map[x] for x in matched_ids])) if len(set(state_map[x] for x in matched_ids)) > 1 and len(set(label_map[x] for x in matched_ids)) > 1 else None, "nmi": BASE.rounded(normalized_mutual_info_score([state_map[x] for x in matched_ids], [label_map[x] for x in matched_ids])) if len(set(state_map[x] for x in matched_ids)) > 1 and len(set(label_map[x] for x in matched_ids)) > 1 else None})
    write_csv(output_root / f"{slug}_global_association.csv", [global_row])
    enrichment = BASE.enrichments(table, list(cores), reference_labels, partition, reference)
    for row in enrichment:
        core_id = row.pop("state")
        core_label_n = row.pop("state_n")
        rest_label_n = row.pop("rest_n")
        row.update({
            "core_id": core_id,
            "core_total_n": len(cores[core_id]),
            "core_label_n": core_label_n,
            "core_subtype_n": row.pop("state_subtype_n"),
            "core_subtype_fraction": row.pop("state_subtype_fraction"),
            "rest_label_n": rest_label_n,
        })
    write_csv(output_root / f"{slug}_core_enrichment.csv", enrichment)
    composition = core_composition_rows(cores, label_map, reference_labels, partition, reference)
    statuses = {core_id: evidence_status(row["label_n"], [item for item in enrichment if item["core_id"] == core_id]) for core_id, row in ((item["core_id"], item) for item in composition)}
    for row in composition:
        row["evidence_status"] = statuses[row["core_id"]]
    write_csv(output_root / f"{slug}_core_composition.csv", composition)
    score = None
    posthoc = []
    if reference == "ClearCode34":
        score, posthoc = BASE.clearcode_score_analysis(state_map, labels, partition)
        write_csv(output_root / f"{slug}_score_omnibus.csv", [score])
        write_csv(output_root / f"{slug}_score_posthoc.csv", posthoc)
    return {"global": global_row, "composition": composition, "enrichment": enrichment, "score": score, "posthoc": posthoc}


def run_partition(partition, cores, macro_map, output_root, mrna, clearcode, permutations):
    output_root.mkdir(parents=True, exist_ok=True)
    results = {
        "mRNA": analyze_reference(partition, cores, mrna, "mRNA", ["m1", "m2", "m3", "m4"], output_root, permutations),
        "ClearCode34": analyze_reference(partition, cores, clearcode, "ClearCode34", ["ccA", "ccB"], output_root, permutations),
    }
    component_rows = []
    mixing_rows = []
    component_info = {
        reference: {row["core_id"]: row for row in results[reference]["composition"]}
        for reference in results
    }
    for reference, labels, reference_labels in (("mRNA", mrna, ["m1", "m2", "m3", "m4"]), ("ClearCode34", clearcode, ["ccA", "ccB"])):
        rows = component_heterogeneity_rows(partition, macro_map, cores, {case_id: item["reference_subtype"] for case_id, item in labels.items() if item.get("reference_subtype") and item.get("reference_status") != "conflict"}, reference, reference_labels, permutations)
        component_rows.extend(rows)
        for row in rows:
            macro_state = row["macro_state"]
            components = tuple(row["component_cores"].split("+"))
            counts = core_contingency({core_id: cores[core_id] for core_id in components}, {case_id: item["reference_subtype"] for case_id, item in labels.items() if item.get("reference_subtype") and item.get("reference_status") != "conflict"}, reference_labels).sum(axis=0)
            info = {core_id: component_info[reference][core_id] for core_id in components}
            mixed = macro_mixing_row(partition, macro_state, components, info, dict(zip(reference_labels, counts)), reference_labels, row["permutation_p"], row["cramers_v_bias_corrected"])
            mixed["reference"] = reference
            mixing_rows.append(mixed)
    return results, component_rows, mixing_rows


def run(data_root=ROOT / "data", stable_root=ROOT / "output_kirc_v13/00_five_view_multi_k_agent_review", macro_root=ROOT / "output_kirc_v13/01_five_view_four_state_macro_characterization", output_root=ROOT / "output_kirc_v13/04_known_ccrcc_subtype_core_mapping", discovery_root=ROOT / "output_kirc_v13/02_post_discovery_characterization", permutations=9999, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = {
        "stable_core_membership": Path(stable_root) / "stable_core_membership.csv",
        "stable_core_universe": discovery_root / "5view_7core/analysis_universe.csv",
        "macro_core_mapping": Path(macro_root) / "core_to_macro_state.csv",
        "mrna_reference": data_root / "tcga_kirc_mrna_m1_m4.csv",
        "clearcode_reference": data_root / "tcga_kirc_clearcode34.csv",
    }
    cores = load_core_membership(inputs["stable_core_membership"], SEVEN_CORE_SIZES)
    validate_core_union(cores, load_universe(inputs["stable_core_universe"]))
    macro_map = load_macro_core_map(inputs["macro_core_mapping"], cores)
    mrna = load_reference(inputs["mrna_reference"], {"m1", "m2", "m3", "m4"})
    clearcode = load_reference(inputs["clearcode_reference"], {"ccA", "ccB"})
    write_csv(output_root / "core_membership_audit.csv", [{"partition": "5V-7core", "core_id": core_id, "core_n": len(members), "stable_union_n": 77} for core_id, members in cores.items()])
    coverage_rows = []
    for core_id, members in cores.items():
        mrna_n = sum(bool(mrna.get(case_id, {}).get("reference_subtype")) for case_id in members)
        clear_n = sum(bool(clearcode.get(case_id, {}).get("reference_subtype")) for case_id in members)
        coverage_rows.append({"partition": "5V-7core", "core_id": core_id, "total_n": len(members), "mrna_available_n": mrna_n, "mrna_missing_n": len(members) - mrna_n, "mrna_coverage": mrna_n / len(members), "clearcode_available_n": clear_n, "clearcode_missing_n": len(members) - clear_n, "clearcode_coverage": clear_n / len(members)})
    write_csv(output_root / "core_reference_coverage.csv", coverage_rows)
    missingness = []
    for reference, labels in (("mRNA", mrna), ("ClearCode34", clearcode)):
        valid_ids = {case_id for case_id, item in labels.items() if item.get("reference_subtype") and item.get("reference_status") != "conflict"}
        table = np.asarray([[sum(case_id in valid_ids for case_id in members), sum(case_id not in valid_ids for case_id in members)] for members in cores.values()])
        chi, p_value = BASE.permutation_chi_square(table, permutations, 20260942)
        missingness.append({"partition": "5V-7core", "reference": reference, "available_n": int(table[:, 0].sum()), "missing_n": int(table[:, 1].sum()), "chi_square": chi, "permutation_p": p_value})
    BASE.attach_bh(missingness, "permutation_p", "bh_q")
    write_csv(output_root / "reference_missingness_by_core.csv", missingness)
    results, component_rows, mixing_rows = run_partition("5V-7core", cores, macro_map, output_root / "5view_7core", mrna, clearcode, permutations)
    attach_component_bh(component_rows)
    write_csv(output_root / "macro_component_reference_heterogeneity.csv", component_rows)
    write_csv(output_root / "macro_merge_mixing_diagnostics.csv", mixing_rows)
    summary_rows = []
    for core_id in SEVEN_CORE_SIZES:
        values = {reference: next(row for row in results[reference]["composition"] if row["core_id"] == core_id) for reference in results}
        best = {}
        for reference, slug in (("mRNA", "mRNA"), ("ClearCode34", "ClearCode")):
            rows = [row for row in results[reference]["enrichment"] if row["core_id"] == core_id and float(row.get("odds_ratio") or 0) > 1]
            best_row = min(rows, key=lambda row: (row["bh_q"] is None, row["bh_q"] if row["bh_q"] is not None else 1, row["fisher_p"] if row["fisher_p"] is not None else 1), default={})
            best.update({f"{slug}_best_enriched_subtype": best_row.get("reference_subtype"), f"{slug}_best_OR": best_row.get("odds_ratio"), f"{slug}_best_q": best_row.get("bh_q")})
        summary_rows.append({"partition": "5V-7core", "core_id": core_id, "core_total_n": len(cores[core_id]), "mRNA_label_n": values["mRNA"]["label_n"], "mRNA_missing_n": values["mRNA"]["missing_n"], "mRNA_dominant": values["mRNA"]["dominant_reference_subtype"], "mRNA_purity": values["mRNA"]["purity"], "mRNA_entropy": values["mRNA"]["normalized_entropy"], "mRNA_status": evidence_status(values["mRNA"]["label_n"], [row for row in results["mRNA"]["enrichment"] if row["core_id"] == core_id]), "ClearCode_label_n": values["ClearCode34"]["label_n"], "ClearCode_missing_n": values["ClearCode34"]["missing_n"], "ClearCode_dominant": values["ClearCode34"]["dominant_reference_subtype"], "ClearCode_purity": values["ClearCode34"]["purity"], "ClearCode_entropy": values["ClearCode34"]["normalized_entropy"], "ClearCode_status": evidence_status(values["ClearCode34"]["label_n"], [row for row in results["ClearCode34"]["enrichment"] if row["core_id"] == core_id]), **best})
        summary_rows[-1]["no_fdr_supported_enrichment_in_either_reference"] = no_fdr_supported_enrichment_in_either_reference(values["mRNA"]["label_n"], summary_rows[-1]["mRNA_status"], values["ClearCode34"]["label_n"], summary_rows[-1]["ClearCode_status"])
    write_csv(output_root / "core_mapping_interpretation_summary.csv", summary_rows)
    write_csv(output_root / "mapping_summary.csv", [results[reference]["global"] for reference in results])
    manifest = {"experiment": "known_ccrcc_subtype_core_mapping", "reference_labels_used_in_discovery": False, "stable_core_memberships_frozen": True, "macro_state_definitions_frozen": True, "no_reclustering": True, "no_agent_rerun": True, "reference_labels_reused_from_experiment_03": True, "permutations": permutations, "five_view_stable_n": 77, "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()}}
    write_json(output_root / "core_mapping_manifest.json", manifest)
    write_json(output_root / "input_audit.json", {"five_view_core_sizes": {key: len(value) for key, value in cores.items()}, "macro_state_core_map": macro_map, "reference_mrna_n": len(mrna), "reference_clearcode_n": len(clearcode), "version_consistency": "exact_core_union_equals_analysis_universe"})
    return {"output_root": str(output_root), "partitions": {"5V-7core": {key: len(value) for key, value in cores.items()}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--stable-root", type=Path, default=ROOT / "output_kirc_v13/00_five_view_multi_k_agent_review")
    parser.add_argument("--macro-root", type=Path, default=ROOT / "output_kirc_v13/01_five_view_four_state_macro_characterization")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v13/04_known_ccrcc_subtype_core_mapping")
    parser.add_argument("--discovery-root", type=Path, default=ROOT / "output_kirc_v13/02_post_discovery_characterization")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
