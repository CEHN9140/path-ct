#!/usr/bin/env python3
"""Map frozen discovery states to published ccRCC reference taxonomies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TCGA_ID = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.I)
NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")
MRNA_MAP = {1: "m1", 2: "m2", 3: "m3", 4: "m4"}


def rounded(value):
    return round(float(value), 10) if value is not None and np.isfinite(value) else None


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_tcga_patient_id(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    match = TCGA_ID.search(text)
    return match.group(0).upper() if match else None


def bh_adjust(values):
    valid = sorted(
        ((index, float(value)) for index, value in enumerate(values) if value is not None and np.isfinite(value)),
        key=lambda item: item[1],
    )
    output = [None] * len(values)
    running = 1.0
    for rank, (index, value) in reversed(list(enumerate(valid, 1))):
        running = min(running, value * len(valid) / rank)
        output[index] = rounded(min(running, 1.0))
    return output


def holm_adjust(values):
    valid = sorted(
        ((index, float(value)) for index, value in enumerate(values) if value is not None and np.isfinite(value)),
        key=lambda item: item[1],
    )
    output = [None] * len(values)
    running = 0.0
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        output[index] = rounded(running)
    return output


def attach_bh(rows, p_field, q_field):
    for row, q_value in zip(rows, bh_adjust([row.get(p_field) for row in rows])):
        row[q_field] = q_value


def prepare_mrna_labels(path):
    frame = pd.read_excel(path, sheet_name="mRNA_miRNA_cluster_assignments")
    required = {"Patient", "mRNA_cluster", "microRNA_cluster"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"mRNA reference is missing columns: {sorted(missing)}")
    rows = []
    audit = {
        "source_rows": int(len(frame)),
        "valid_patient_ids": 0,
        "m1_count": 0,
        "m2_count": 0,
        "m3_count": 0,
        "m4_count": 0,
        "missing_cluster": 0,
        "duplicate_consistent": 0,
        "duplicate_conflict": 0,
        "invalid_patient_id": 0,
    }
    grouped = {}
    for record in frame.to_dict("records"):
        case_id = normalize_tcga_patient_id(record["Patient"])
        if case_id is None:
            audit["invalid_patient_id"] += 1
            continue
        raw = record["mRNA_cluster"]
        if pd.isna(raw):
            cluster = None
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if pd.isna(numeric) or float(numeric) not in MRNA_MAP:
                raise ValueError(f"Invalid mRNA_cluster for {case_id}: {raw}")
            cluster = int(float(numeric))
        grouped.setdefault(case_id, []).append((raw, cluster))
    audit["valid_patient_ids"] = len(grouped)
    for case_id in sorted(grouped):
        values = grouped[case_id]
        clusters = {cluster for _, cluster in values if cluster is not None}
        status = "missing" if not clusters else "matched"
        subtype = MRNA_MAP[next(iter(clusters))] if len(clusters) == 1 else None
        if len(values) > 1 and len(clusters) == 1:
            status = "duplicate_consistent"
            audit["duplicate_consistent"] += 1
        elif len(clusters) > 1:
            status = "conflict"
            audit["duplicate_conflict"] += 1
        if not clusters:
            audit["missing_cluster"] += 1
        if subtype:
            audit[f"{subtype}_count"] += 1
        raw_values = sorted({str(raw) for raw, cluster in values if cluster is not None})
        rows.append({
            "case_id": case_id,
            "mrna_cluster_raw": ";".join(raw_values),
            "reference_subtype": subtype,
            "reference_status": status,
            "source": "TCGA_Nature_2013_Data_File_S9",
        })
    return pd.DataFrame(rows), audit


def extract_doc_text(path):
    command = shutil.which("antiword")
    if command:
        result = subprocess.run([command, str(path)], check=True, capture_output=True, text=True)
        return result.stdout, "antiword"
    command = shutil.which("libreoffice") or shutil.which("soffice")
    if command:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile"
            output = Path(temporary) / path.with_suffix(".txt").name
            profile.mkdir()
            try:
                subprocess.run(
                    [command, f"-env:UserInstallation=file://{profile}", "--headless", "--convert-to", "txt:Text", "--outdir", temporary, str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            if output.exists():
                return output.read_text(encoding="utf-8", errors="replace"), "libreoffice"
    command = shutil.which("strings")
    if not command:
        raise RuntimeError("Legacy .doc extraction tool unavailable")
    result = subprocess.run([command, "-n", "3", str(path)], check=True, capture_output=True, text=True)
    return result.stdout, "strings"


def parse_clearcode_text(text):
    start_match = re.search(r"Supplemental Table 3", text, re.I)
    end_match = re.search(r"Supplemental Table 4", text, re.I)
    if not start_match or not end_match or end_match.start() <= start_match.end():
        raise ValueError("Could not isolate ClearCode34 Supplemental Table 3")
    lines = [line.strip() for line in text[start_match.end():end_match.start()].splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", line, re.I)
        if not match:
            continue
        values = lines[index + 1:index + 4]
        if len(values) != 3 or not NUMBER.fullmatch(values[0]) or not NUMBER.fullmatch(values[1]) or values[2].lower() not in {"cca", "ccb"}:
            raise ValueError(f"Malformed ClearCode34 Table 3 row near {line}")
        p_a, p_b = float(values[0]), float(values[1])
        label = values[2].lower()
        if not 0 <= p_a <= 1 or not 0 <= p_b <= 1:
            raise ValueError(f"ClearCode probability out of range for {line}")
        rows.append({
            "case_id": normalize_tcga_patient_id(line),
            "cca_probability": p_a,
            "ccb_probability": p_b,
            "probability_sum_error": rounded(p_a + p_b - 1),
            "classification_consistency": label == ("cca" if p_a >= p_b else "ccb"),
            "reference_subtype": "ccA" if label == "cca" else "ccB",
            "reference_status": "matched" if label == ("cca" if p_a >= p_b else "ccb") else "classification_mismatch",
            "source": "ClearCode34_Supp_Table3",
        })
    if not rows:
        raise ValueError("No TCGA rows parsed from ClearCode34 Supplemental Table 3")
    frame = pd.DataFrame(rows)
    if frame["case_id"].duplicated().any():
        raise ValueError("Duplicate TCGA patient in ClearCode34 Supplemental Table 3")
    audit = {
        "table_used": "Supplemental Table 3",
        "valid_tcga_rows": len(frame),
        "ccA_count": int((frame["reference_subtype"] == "ccA").sum()),
        "ccB_count": int((frame["reference_subtype"] == "ccB").sum()),
        "probability_sum_outside_0.02": int((frame["probability_sum_error"].abs() > .02).sum()),
        "classification_probability_mismatches": int((~frame["classification_consistency"]).sum()),
    }
    return frame, audit


def prepare_clearcode_labels(path):
    text, extraction_method = extract_doc_text(path)
    frame, audit = parse_clearcode_text(text)
    audit["extraction_method"] = extraction_method
    return frame, audit


def load_analysis_universe(path, expected_n, expected_groups):
    frame = pd.read_csv(path, dtype=str)
    if set(frame.columns) != {"case_id", "group_id"}:
        raise ValueError(f"Unexpected analysis universe columns in {path}")
    frame["case_id"] = frame["case_id"].map(normalize_tcga_patient_id)
    if frame["case_id"].isna().any() or frame["case_id"].duplicated().any():
        raise ValueError(f"Invalid or duplicate case IDs in {path}")
    if len(frame) != expected_n or set(frame["group_id"]) != set(expected_groups):
        raise ValueError(f"Unexpected fixed discovery universe in {path}")
    return dict(zip(frame["case_id"], frame["group_id"])), list(expected_groups)


def build_contingency(state_labels, reference_labels, ordered_states, ordered_reference_labels):
    return np.asarray([
        [sum(state_labels.get(case_id) == state and reference_labels.get(case_id) == label for case_id in state_labels if case_id in reference_labels) for label in ordered_reference_labels]
        for state in ordered_states
    ], dtype=int)


def chi_square_stat(table):
    from scipy.stats import chi2_contingency
    table = np.asarray(table, dtype=float)
    table = table[table.sum(axis=1) > 0]
    table = table[:, table.sum(axis=0) > 0] if table.size else table
    return None if table.shape[0] < 2 or table.shape[1] < 2 else float(chi2_contingency(table, correction=False)[0])


def bias_corrected_cramers_v(table):
    table = np.asarray(table, dtype=float)
    table = table[table.sum(axis=1) > 0]
    table = table[:, table.sum(axis=0) > 0] if table.size else table
    n = table.sum()
    if n <= 1 or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi = chi_square_stat(table)
    phi2 = chi / n
    rows, cols = table.shape
    phi2_corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    rows_corr = rows - (rows - 1) ** 2 / (n - 1)
    cols_corr = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_corr - 1, cols_corr - 1)
    return rounded(math.sqrt(phi2_corr / denominator)) if denominator > 0 else None


def cramers_v_raw(table):
    table = np.asarray(table, dtype=float)
    table = table[table.sum(axis=1) > 0]
    table = table[:, table.sum(axis=0) > 0] if table.size else table
    chi = chi_square_stat(table)
    n = table.sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return rounded(math.sqrt(chi / denominator)) if chi is not None and denominator > 0 else None


def permutation_chi_square(table, permutations=9999, seed=20260909):
    observed = chi_square_stat(table)
    if observed is None:
        return None, None
    table = np.asarray(table, dtype=int)
    row_labels = np.repeat(np.arange(table.shape[0]), table.sum(axis=1))
    reference_labels = np.concatenate([np.repeat(j, table[:, j].sum()) for j in range(table.shape[1])])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(reference_labels)
        perm = np.asarray([[np.sum((row_labels == i) & (shuffled == j)) for j in range(table.shape[1])] for i in range(table.shape[0])])
        if (chi_square_stat(perm) or 0) >= observed:
            exceed += 1
    return rounded(observed), rounded((exceed + 1) / (permutations + 1))


def percentages(table, axis):
    table = np.asarray(table, dtype=float)
    denominator = table.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        output = np.divide(table, denominator, where=denominator != 0)
    output[~np.isfinite(output)] = 0
    return output * 100


def odds_ratio_ci(a, b, c, d):
    odds = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    se = math.sqrt(sum(1 / (value + .5) for value in (a, b, c, d)))
    return odds, math.exp(math.log(odds) - 1.96 * se), math.exp(math.log(odds) + 1.96 * se)


def state_composition(table, states, reference_labels, partition, reference):
    rows = []
    for i, state in enumerate(states):
        counts = table[i]
        total = int(counts.sum())
        probs = counts / total if total else np.zeros(len(counts))
        entropy = -sum(float(p) * math.log(float(p)) for p in probs if p > 0)
        rows.append({
            "partition": partition,
            "reference": reference,
            "state": state,
            "state_n": total,
            "dominant_reference_subtype": reference_labels[int(np.argmax(counts))] if total else None,
            "purity": rounded(float(probs.max())) if total else None,
            "normalized_entropy": rounded(entropy / math.log(len(reference_labels))) if total and len(reference_labels) > 1 else 0.0,
            **{f"{label}_n": int(count) for label, count in zip(reference_labels, counts)},
        })
    return rows


def enrichments(table, states, reference_labels, partition, reference):
    rows = []
    total = table.sum(axis=0)
    grand = int(table.sum())
    for i, state in enumerate(states):
        state_n = int(table[i].sum())
        rest = table.sum(axis=0) - table[i]
        for j, label in enumerate(reference_labels):
            a, b = int(table[i, j]), state_n - int(table[i, j])
            c, d = int(rest[j]), int(rest.sum() - rest[j])
            odds, low, high = odds_ratio_ci(a, b, c, d)
            p_value = float(fisher_exact([[a, b], [c, d]])[1]) if state_n and int(rest.sum()) else None
            rows.append({
                "partition": partition,
                "reference": reference,
                "state": state,
                "reference_subtype": label,
                "state_n": state_n,
                "state_subtype_n": a,
                "state_subtype_fraction": a / state_n if state_n else None,
                "rest_n": int(rest.sum()),
                "rest_subtype_n": c,
                "rest_subtype_fraction": c / rest.sum() if rest.sum() else None,
                "odds_ratio": rounded(odds),
                "ci_low": rounded(low),
                "ci_high": rounded(high),
                "fisher_p": rounded(p_value),
            })
    attach_bh(rows, "fisher_p", "bh_q")
    return rows


def missingness_rows(state_labels, reference_labels, partition, reference, permutations):
    states = sorted(set(state_labels.values()))
    table = np.asarray([[sum(state == s and case_id in reference_labels for case_id, state in state_labels.items()), sum(state == s and case_id not in reference_labels for case_id, state in state_labels.items())] for s in states])
    statistic, p_value = permutation_chi_square(table, permutations, 20260920)
    return {"partition": partition, "reference": reference, "available_n": int(table[:, 0].sum()), "missing_n": int(table[:, 1].sum()), "chi_square": statistic, "permutation_p": p_value, "states": ";".join(states)}


def clearcode_score_analysis(state_labels, labels, output_prefix):
    groups = sorted(set(state_labels.values()))
    samples = [[float(labels[case_id]["cca_probability"]) - float(labels[case_id]["ccb_probability"]) for case_id, state in state_labels.items() if state == group and case_id in labels] for group in groups]
    omnibus = {"partition": output_prefix, "analysis": "kruskal_wallis", "available_n": sum(map(len, samples)), "group_count": len(groups), "h_statistic": None, "p_value": None}
    if all(samples) and len(groups) > 1:
        result = kruskal(*samples)
        omnibus.update({"h_statistic": rounded(result.statistic), "p_value": rounded(result.pvalue)})
    posthoc = []
    if omnibus["p_value"] is not None and omnibus["p_value"] < .05:
        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1:]:
                left = samples[groups.index(group_a)]
                right = samples[groups.index(group_b)]
                p_value = float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
                posthoc.append({"partition": output_prefix, "group_a": group_a, "group_b": group_b, "n_a": len(left), "n_b": len(right), "median_a": rounded(np.median(left)), "median_b": rounded(np.median(right)), "p_value": rounded(p_value)})
        for row, value in zip(posthoc, holm_adjust([row["p_value"] for row in posthoc])):
            row["holm_p"] = value
    return omnibus, posthoc


def write_reference_coverage(all_ids, references, output_root):
    rows = []
    for case_id in all_ids:
        mrna = references["mrna"].get(case_id, {})
        clearcode = references["clearcode"].get(case_id, {})
        rows.append({"case_id": case_id, "has_mrna_label": bool(mrna.get("reference_subtype")), "mrna_subtype": mrna.get("reference_subtype"), "has_clearcode_label": bool(clearcode.get("reference_subtype")), "clearcode_subtype": clearcode.get("reference_subtype"), "cca_probability": clearcode.get("cca_probability"), "ccb_probability": clearcode.get("ccb_probability")})
    write_csv(output_root / "reference_coverage_all_cases.csv", rows)
    return rows


def analyze_partition(name, state_labels, mrna, clearcode, output_root, permutations):
    output_root.mkdir(parents=True, exist_ok=True)
    states = sorted(set(state_labels.values()))
    rows = []
    families = [("mRNA", "mrna", mrna, ["m1", "m2", "m3", "m4"]), ("ClearCode34", "clearcode", clearcode, ["ccA", "ccB"])]
    for reference, slug, labels, reference_labels in families:
        usable = {case_id: item["reference_subtype"] for case_id, item in labels.items() if case_id in state_labels and item.get("reference_status") != "conflict" and item.get("reference_subtype")}
        table = build_contingency(state_labels, usable, states, reference_labels)
        prefix = f"{name}_{slug}"
        write_csv(output_root / f"{prefix}_contingency_counts.csv", [{"state": state, **{label: int(value) for label, value in zip(reference_labels, table[i])}} for i, state in enumerate(states)])
        write_csv(output_root / f"{prefix}_contingency_row_pct.csv", [{"state": state, **{label: rounded(value) for label, value in zip(reference_labels, row)}} for state, row in zip(states, percentages(table, 1))])
        write_csv(output_root / f"{prefix}_contingency_col_pct.csv", [{"reference_subtype": label, **{state: rounded(value) for state, value in zip(states, row)}} for label, row in zip(reference_labels, percentages(table, 0).T)])
        chi, p_value = permutation_chi_square(table, permutations, 20260910)
        global_row = {"partition": name, "reference": reference, "matched_n": int(table.sum()), "group_count": len(states), "reference_class_count": len(reference_labels), "chi_square": chi, "permutation_p": p_value, "cramers_v_raw": cramers_v_raw(table), "cramers_v_bias_corrected": bias_corrected_cramers_v(table)}
        write_csv(output_root / f"{prefix}_global_association.csv", [global_row])
        labels_a = [state_labels[case_id] for case_id in usable]
        labels_b = [usable[case_id] for case_id in usable]
        metrics = {"partition": name, "reference": reference, "matched_n": len(usable), "ari": rounded(adjusted_rand_score(labels_a, labels_b)) if len(set(labels_a)) > 1 and len(set(labels_b)) > 1 else None, "nmi": rounded(normalized_mutual_info_score(labels_a, labels_b)) if len(set(labels_a)) > 1 and len(set(labels_b)) > 1 else None}
        write_csv(output_root / f"{prefix}_partition_metrics.csv", [metrics])
        composition = state_composition(table, states, reference_labels, name, reference)
        write_csv(output_root / f"{prefix}_state_composition.csv", composition)
        write_csv(output_root / f"{prefix}_state_subtype_enrichment.csv", enrichments(table, states, reference_labels, name, reference))
        rows.append(global_row | metrics)
        if reference == "ClearCode34":
            omnibus, posthoc = clearcode_score_analysis(state_labels, labels, name)
            write_csv(output_root / f"{prefix}_score_omnibus.csv", [omnibus])
            write_csv(output_root / f"{prefix}_score_posthoc.csv", posthoc)
            sensitivity_rows = []
            for threshold in (.6, .8):
                confident = {case_id: item for case_id, item in labels.items() if max(item["cca_probability"], item["ccb_probability"]) >= threshold}
                confident_table = build_contingency(state_labels, {case_id: item["reference_subtype"] for case_id, item in confident.items()}, states, reference_labels)
                c, q = permutation_chi_square(confident_table, permutations, int(threshold * 1000))
                sensitivity_rows.append({"partition": name, "confidence_threshold": threshold, "matched_n": int(confident_table.sum()), "chi_square": c, "permutation_p": q, "cramers_v_bias_corrected": bias_corrected_cramers_v(confident_table)})
                rows.append({"partition": name, "reference": f"ClearCode34_confidence_{threshold}", "matched_n": int(confident_table.sum()), "group_count": len(states), "reference_class_count": 2, "chi_square": c, "permutation_p": q, "cramers_v_raw": cramers_v_raw(confident_table), "cramers_v_bias_corrected": bias_corrected_cramers_v(confident_table), "ari": None, "nmi": None})
            write_csv(output_root / f"{prefix}_confidence_sensitivity.csv", sensitivity_rows)
    return rows


def run(data_root=ROOT / "data", output_root=ROOT / "output_kirc_v13/03_known_ccrcc_subtype_mapping", discovery_root=ROOT / "output_kirc_v13/02_post_discovery_characterization", permutations=9999, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mrna_frame, mrna_audit = prepare_mrna_labels(data_root / "Data_file_S9_mRNA_miRNA_cluster_assignments.xlsx")
    clearcode_frame, clearcode_audit = prepare_clearcode_labels(data_root / "NIHMS576995-supplement-03.doc")
    mrna_frame.to_csv(data_root / "tcga_kirc_mrna_m1_m4.csv", index=False)
    clearcode_frame.to_csv(data_root / "tcga_kirc_clearcode34.csv", index=False)
    mrna = mrna_frame.set_index("case_id").to_dict("index")
    clearcode = clearcode_frame.set_index("case_id").to_dict("index")
    states = {}
    with (data_root / "tcga_kirc_data.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    all_ids = sorted({normalize_tcga_patient_id(item.get("case_id")) for item in payload if isinstance(item, dict) and item.get("case_id")}) if isinstance(payload, list) else []
    if len(all_ids) != 103:
        from scripts_2026_8_31.analyze_multi_k_stable_cores import load_states
        _, all_ids = load_states(ROOT / "output_kirc")
    all_ids = sorted(set(all_ids))
    coverage_rows = write_reference_coverage(all_ids, {"mrna": mrna, "clearcode": clearcode}, output_root)
    partition_specs = {
        "4V-3state": (discovery_root / "4view_3state/analysis_universe.csv", 59, ["STATE_A", "STATE_B", "STATE_C"]),
        "5V-4state": (discovery_root / "5view_4state/analysis_universe.csv", 69, ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]),
    }
    partition_labels = {name: load_analysis_universe(path, expected_n, groups) for name, (path, expected_n, groups) in partition_specs.items()}
    coverage_summary = []
    missingness = []
    mapping_rows = []
    for reference, reference_labels in (("mRNA", mrna), ("ClearCode34", clearcode)):
        matched = sum(bool(case_id in reference_labels and reference_labels[case_id].get("reference_status") != "conflict" and reference_labels[case_id].get("reference_subtype")) for case_id in all_ids)
        coverage_summary.append({"partition": "all", "reference": reference, "total": len(all_ids), "matched": matched, "unmatched": len(all_ids) - matched, "coverage": matched / len(all_ids)})
    for partition, (labels, groups) in partition_labels.items():
        for reference, reference_labels in (("mRNA", mrna), ("ClearCode34", clearcode)):
            matched = sum(bool(case_id in reference_labels and reference_labels[case_id].get("reference_status") != "conflict" and reference_labels[case_id].get("reference_subtype")) for case_id in labels)
            coverage_summary.append({"partition": partition, "reference": reference, "total": len(labels), "matched": matched, "unmatched": len(labels) - matched, "coverage": matched / len(labels)})
            missingness.append(missingness_rows(labels, {case_id for case_id, item in reference_labels.items() if item.get("reference_status") != "conflict" and item.get("reference_subtype")}, partition, reference, permutations))
        rows = analyze_partition(partition, labels, mrna, clearcode, output_root, permutations)
        mapping_rows.extend(rows)
    all_mrna = {case_id: item["reference_subtype"] for case_id, item in mrna.items() if item.get("reference_subtype")}
    all_clear = {case_id: item["reference_subtype"] for case_id, item in clearcode.items() if item.get("reference_subtype")}
    cross_rows = []
    for label in ["m1", "m2", "m3", "m4"]:
        cross_rows.append({"mrna_subtype": label, "ccA": sum(mrna.get(case_id, {}).get("reference_subtype") == label and clearcode.get(case_id, {}).get("reference_subtype") == "ccA" for case_id in all_ids), "ccB": sum(mrna.get(case_id, {}).get("reference_subtype") == label and clearcode.get(case_id, {}).get("reference_subtype") == "ccB" for case_id in all_ids)})
    cross_table = np.asarray([[row["ccA"], row["ccB"]] for row in cross_rows])
    cross_chi, cross_p = permutation_chi_square(cross_table, permutations, 20260911)
    write_csv(output_root / "reference_taxonomy_crosswalk.csv", cross_rows)
    write_csv(output_root / "reference_taxonomy_crosswalk_metrics.csv", [{"matched_n": sum(case_id in all_mrna and case_id in all_clear for case_id in all_ids), "chi_square": cross_chi, "permutation_p": cross_p, "cramers_v_bias_corrected": bias_corrected_cramers_v(np.asarray([[row["ccA"], row["ccB"]] for row in cross_rows]))}])
    write_csv(output_root / "reference_coverage_summary.csv", coverage_summary)
    attach_bh(missingness, "permutation_p", "bh_q")
    write_csv(output_root / "reference_missingness_by_state.csv", missingness)
    write_json(output_root / "reference_label_preparation_audit.json", {"tcga_mrna": mrna_audit, "clearcode34": clearcode_audit})
    write_csv(output_root / "mapping_summary.csv", mapping_rows)
    dominant = []
    for partition in partition_labels:
        for reference in ("mRNA", "ClearCode34"):
            prefix = f"{partition}_{'mrna' if reference == 'mRNA' else 'clearcode'}_state_composition.csv"
            path = output_root / prefix
            if path.exists():
                dominant.extend(pd.read_csv(path).to_dict("records"))
    write_csv(output_root / "dominant_mapping_summary.csv", dominant)
    write_json(output_root / "mapping_manifest.json", {
        "experiment": "known_ccrcc_subtype_mapping",
        "discovery_labels_frozen_before_reference_mapping": True,
        "reference_labels_used_in_discovery": False,
        "reference_sources": ["Data_file_S9_mRNA_miRNA_cluster_assignments.xlsx", "NIHMS576995-supplement-03.doc: Supplemental Table 3 only"],
        "mRNA_primary_field": "mRNA_cluster",
        "clearcode_primary_field": "published Subtype Classification",
        "permutations": permutations,
        "all_cohort_n": len(all_ids),
        "partition_sizes": {name: len(labels) for name, (labels, _) in partition_labels.items()},
    })
    write_json(output_root / "mapping_summary.json", {"coverage": coverage_summary, "mapping_rows": mapping_rows, "source_sha256": {name: hashlib.sha256((data_root / name).read_bytes()).hexdigest() for name in ["Data_file_S9_mRNA_miRNA_cluster_assignments.xlsx", "NIHMS576995-supplement-03.doc"]}})
    return {"output_root": str(output_root), "all_cohort_n": len(all_ids), "coverage": coverage_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v13/03_known_ccrcc_subtype_mapping")
    parser.add_argument("--discovery-root", type=Path, default=ROOT / "output_kirc_v13/02_post_discovery_characterization")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
