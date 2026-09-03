#!/usr/bin/env python3
"""Offline conditional multimodal support audit for existing stable cores."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts_2026_8_31.analyze_multi_k_stable_cores import (
    binary_analysis,
    clinical_analysis,
    clinical_availability,
    core_confounds,
    core_rest_rows,
    cnv_analysis,
    ct_radiomics_table,
    finite,
    load_expression_scores,
    load_states,
    load_table,
    rna_analysis,
    read_csv,
    wsi_embedding_table,
    write_csv,
    write_json,
)
from scripts_2026_8_31.analyze_multi_k_stable_cores import load_cores as load_membership


def target_groups(five_cores, overlap_rows):
    groups = {f"5V_{core}": sorted(members) for core, members in five_cores.items()}
    for row in overlap_rows:
        if str(row.get("representation_robust_candidate")) != "1":
            continue
        shared = json.loads(row.get("shared_patient_ids") or "[]")
        name = f"ROBUST_4V_{row['core_4v']}_5V_{row['core_5v']}"
        groups[name] = sorted(set(shared))
    return groups


def support_summary(rows):
    output = []
    for target, modality in sorted({(row["target_id"], row["modality"]) for row in rows}):
        current = [row for row in rows if row["target_id"] == target and row["modality"] == modality]
        significant = [row for row in current if (finite(row.get("q_value")) or 1) < 0.05]
        top = sorted(significant or current, key=lambda row: ((finite(row.get("q_value")) is None), finite(row.get("q_value")) or 1, -abs(finite(row.get("effect_size")) or 0), row["feature"]))
        output.append({
            "target_id": target,
            "modality": modality,
            "feature_n": len(current),
            "significant_feature_n": len(significant),
            "top_feature": top[0]["feature"] if top else "",
            "top_effect_size": finite(top[0].get("effect_size")) if top else None,
            "top_q_value": finite(top[0].get("q_value")) if top else None,
        })
    return output


def add_numeric(rows, target, modality, feature_key, effect_key):
    return [
        {"target_id": target, "modality": modality, "feature": str(row.get(feature_key, "")),
         "effect_size": row.get(effect_key), "q_value": row.get("q_value"),
         "n_target": row.get("n_core", row.get("available_n_core", row.get("set_total_n"))),
         "n_rest": row.get("n_rest", row.get("available_n_rest", row.get("rest_total_n")))}
        for row in rows
    ]


def plot_support(output_root, rows):
    from utils.visualization import configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt

    targets = sorted({row["target_id"] for row in rows})
    modalities = ["ct", "wsi", "rna", "wxs", "cnv", "clinical"]
    lookup = {(row["target_id"], row["modality"]): row["significant_feature_n"] for row in rows}
    matrix = np.asarray([[lookup.get((target, modality), 0) for modality in modalities] for target in targets])
    figure, axis = plt.subplots(figsize=(9, max(4, len(targets) * .45)))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
    axis.set_xticks(range(len(modalities)), modalities); axis.set_yticks(range(len(targets)), targets)
    for i in range(len(targets)):
        for j in range(len(modalities)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    axis.set_xlabel("Modality"); axis.set_ylabel("Fixed target group")
    axis.set_title("Conditional support: FDR-significant feature count (q < 0.05)")
    figure.colorbar(image, ax=axis, label="Significant feature count")
    figure.tight_layout(); figure.savefig(output_root / "conditional_support_heatmap.png", dpi=220, bbox_inches="tight"); plt.close(figure)


def run(data_root, five_view_root, robustness_root, output_root, config_dir, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    five_cores, stable_ids = load_membership(five_view_root)
    states, _ = load_states(data_root)
    overlap_rows = read_csv(robustness_root / "representation_robust_core_candidates.csv")
    groups = target_groups(five_cores, overlap_rows)
    stable_ids = sorted(set(stable_ids))
    if len(five_cores) != 6 or len(stable_ids) != 69 or any(not set(members) <= set(stable_ids) for members in groups.values()):
        raise ValueError("实验15 stable-core输入必须是6组、69例且目标病例必须来自该稳定核心集合")

    feature_rows = []
    ct_features, ct_table = ct_radiomics_table(states, stable_ids)
    wsi_features, wsi_table = wsi_embedding_table(states, stable_ids)
    pathways, scores = load_expression_scores(states, config_dir)
    wxs_features, wxs_table = load_table(data_root / "wxs/wxs_discovery_features.csv")
    cnv_features, cnv_table = load_table(data_root / "cnv/case_features.csv")
    mutation_features = [feature for feature in wxs_features if feature.startswith("mutation::")]

    for target, members in groups.items():
        one = {target: members}
        feature_rows.extend(add_numeric(core_rest_rows(ct_table, ct_features, one, stable_ids), target, "ct", "feature", "smd"))
        feature_rows.extend(add_numeric(core_rest_rows(wsi_table, wsi_features, one, stable_ids), target, "wsi", "feature", "smd"))
        rna_rows, _ = rna_analysis(states, config_dir, one, stable_ids, pathways, scores)
        feature_rows.extend(add_numeric(rna_rows, target, "rna", "pathway", "smd"))
        wxs_rows, _ = binary_analysis(wxs_table, mutation_features, one, stable_ids, "mutation")
        feature_rows.extend(add_numeric(wxs_rows, target, "wxs", "gene", "delta_mutation_frequency"))
        cnv_cont, cnv_event, _, _ = cnv_analysis(cnv_table, cnv_features, one, stable_ids)
        feature_rows.extend(add_numeric(cnv_cont, target, "cnv", "feature", "cliffs_delta"))
        feature_rows.extend(add_numeric(cnv_event, target, "cnv", "feature", "frequency_difference"))

    write_csv(output_root / "conditional_support_features.csv", feature_rows)
    support_rows = support_summary(feature_rows)

    from tools.subtype_review_common import clinical_table
    records = clinical_table(states)
    availability_rows, availability = clinical_availability(records, groups, stable_ids)
    clinical_rows, _, survival_rows = clinical_analysis(records, groups, stable_ids, availability["eligible_variables"], availability["survival_eligible"])
    clinical_feature_rows = []
    for row in clinical_rows:
        clinical_feature_rows.append({"target_id": row["core_id"], "modality": "clinical", "feature": f'{row["clinical_variable"]}:{row.get("level", "")}', "effect_size": row.get("effect_size", row.get("frequency_difference")), "q_value": row.get("q_value"), "n_target": row.get("n_core", row.get("set_total")), "n_rest": row.get("n_rest", row.get("rest_total"))})
    write_csv(output_root / "conditional_clinical_features.csv", clinical_feature_rows)
    support_rows.extend(support_summary(clinical_feature_rows))

    confound_rows = core_confounds(data_root, config_dir, states, groups)
    write_csv(output_root / "conditional_technical_confounders.csv", confound_rows)
    write_csv(output_root / "clinical_availability.csv", availability_rows)
    write_csv(output_root / "clinical_survival.csv", [{"target_id": row["core_id"], **row} for row in survival_rows])
    write_csv(output_root / "conditional_support_summary.csv", support_rows)
    write_csv(output_root / "target_group_membership.csv", [{"target_id": target, "patient_id": patient_id} for target, members in groups.items() for patient_id in members])
    plot_support(output_root, support_rows)
    summary = {
        "experiment": "conditional_multimodal_support",
        "stable_core_universe_n": len(stable_ids),
        "five_view_core_count": len(five_cores),
        "target_group_count": len(groups),
        "targets": {target: len(members) for target, members in groups.items()},
        "modalities": ["ct", "wsi", "rna", "wxs", "cnv", "clinical"],
        "interpretation": "Offline conditional support audit; q<0.05 counts are descriptive evidence summaries and do not assign subtype labels.",
    }
    write_json(output_root / "conditional_multimodal_support_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--five-view-root", type=Path, default=ROOT / "output_kirc_v12/15_five_view_multi_k_stability")
    parser.add_argument("--robustness-root", type=Path, default=ROOT / "output_kirc_v12/17_cross_representation_stable_core_robustness")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v12/18_conditional_multimodal_support")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
