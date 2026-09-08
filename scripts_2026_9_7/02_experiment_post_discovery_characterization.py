#!/usr/bin/env python3
"""Unified stable-group characterization for the 4-view and 5-view results."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import post_discovery_characterization as stats
from tools.ct_radiomics import build_ct_affinity
from tools.subtype_review_common import clinical_table
from scripts_2026_8_31 import analyze_multi_k_stable_cores as base
from scripts_2026_8_31.experiment_macro_state_analysis import (
    MACRO_STATE_CORES as FOUR_VIEW_MACRO_CORES,
)
from scripts_2026_8_31.five_view_experiment import load_five_view_inputs

FIVE_VIEW_MACRO_CORES = {
    "STATE_A": ("CORE01",),
    "STATE_B": ("CORE02",),
    "STATE_C": ("CORE03",),
    "STATE_D": ("CORE04", "CORE05", "CORE06"),
}


def groups_from_cores(cores, mapping):
    return {
        state: sorted({case_id for core in source for case_id in cores[core]})
        for state, source in mapping.items()
    }


def load_saved_affinities(data_root, stable_root, output_root, config_dir, five_view):
    if five_view:
        patient_ids, matrices, fused, _ = load_five_view_inputs(data_root, stable_root, output_root, config_dir)
        saved_fused = np.load(stable_root / "fused_similarity_5view.npy")
        if not np.allclose(fused, saved_fused, rtol=1e-6, atol=1e-8):
            raise RuntimeError("Recomputed 5-view fused affinity does not match the saved 15号 artifact")
        affinities = {name: matrices[name] for name in ("ct", "wsi", "rna", "wxs", "cnv")}
        affinities["fused"] = saved_fused
        saved_ids = [str(value) for value in np.load(stable_root / "affinity_patient_order.npy", allow_pickle=True).tolist()]
    else:
        order_path = data_root / "candidate_subtype/affinity_patient_order.json"
        patient_ids = json.loads(order_path.read_text(encoding="utf-8"))
        paths = {
            "ct": data_root / "candidate_subtype/ct_affinity.npy",
            "wsi": data_root / "candidate_subtype/wsi_affinity.npy",
            "rna": data_root / "candidate_subtype/rna_affinity.npy",
            "genomic": data_root / "wxs/genomic_affinity.npy",
            "fused": data_root / "candidate_subtype/fused_similarity.npy",
        }
        affinities = {name: np.load(path) for name, path in paths.items()}
        saved_ids = [str(value) for value in patient_ids]
    if [str(value) for value in patient_ids] != saved_ids:
        raise ValueError("Saved affinity patient order is inconsistent")
    return saved_ids, affinities


def load_ct_table(patient_states, patient_ids, data_root, config_dir):
    payload = build_ct_affinity(patient_states, config_dir=str(config_dir), output_root=str(data_root))
    if payload["patient_ids"] != patient_ids:
        raise ValueError("Production CT feature patient order does not match saved affinity order")
    saved = np.load(data_root / "candidate_subtype/ct_affinity.npy")
    reconstructed = payload["affinity"]
    difference = np.abs(saved - reconstructed)
    audit = dict(payload["audit"])
    audit.update({
        "production_consistent": bool(np.allclose(saved, reconstructed, rtol=1e-6, atol=1e-8)),
        "saved_affinity_max_abs_diff": float(difference.max()) if difference.size else 0.0,
        "saved_affinity_mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
        "preprocessing_fit_universe_n": len(payload["patient_ids"]),
        "characterization_feature_count": len(payload["feature_names"]),
    })
    if not audit["production_consistent"]:
        raise RuntimeError("Reconstructed CT affinity does not match saved production CT affinity")
    table = {
        case_id: {feature: float(payload["matrix"][row, column]) for column, feature in enumerate(payload["feature_names"])}
        for row, case_id in enumerate(payload["patient_ids"])
    }
    return payload["feature_names"], table, audit


def load_event_table(table, features, threshold=.2):
    events = {case_id: {} for case_id in table}
    event_features = []
    for feature in features:
        if not feature.startswith(("chr", "locus::")):
            continue
        for event, predicate in (("loss", lambda value: value <= -threshold), ("gain", lambda value: value >= threshold)):
            name = f"{feature}::{event}"
            event_features.append(name)
            for case_id, values in table.items():
                value = values.get(feature)
                if value is not None:
                    events[case_id][name] = int(predicate(value))
    return event_features, events


def clinical_outputs(records, groups, output_root, permutations, bootstrap_iterations):
    continuous = {"age": {case_id: {"age": records.get(case_id, {}).get("age")} for case_id in records}}
    age_omnibus = stats.continuous_omnibus(continuous["age"], ["age"], groups)
    categorical_variables = ("gender", "stage_group", "t_stage", "m_stage", "grade")
    categorical = [stats.categorical_omnibus(records, variable, groups, permutations, 20260908 + index) for index, variable in enumerate(categorical_variables)]
    all_p = [row["p_value"] for row in age_omnibus + categorical]
    for row, q_value in zip(age_omnibus + categorical, stats.bh_adjust(all_p)):
        row["q_value"] = q_value
    base.write_csv(output_root / "clinical_omnibus.csv", age_omnibus + categorical)
    selected_categorical = {row["clinical_variable"] for row in categorical if row["q_value"] is not None and row["q_value"] < .05}
    posthoc = []
    if age_omnibus[0]["q_value"] is not None and age_omnibus[0]["q_value"] < .05:
        posthoc.extend(stats.continuous_posthoc(continuous["age"], ["age"], groups))
    for variable in selected_categorical:
        posthoc.extend(stats.categorical_posthoc(records, variable, groups))
    base.write_csv(output_root / "clinical_posthoc.csv", posthoc)
    survival_global, survival_pairs = stats.survival_analysis(records, groups)
    survival_global["q_value"] = survival_global["p_value"]
    base.write_csv(output_root / "survival_global.csv", [survival_global])
    base.write_csv(output_root / "survival_posthoc.csv", survival_pairs if survival_global["p_value"] is not None and survival_global["p_value"] < .05 else [])
    return age_omnibus + categorical, posthoc, survival_global


def characterize_groups(data_root, stable_root, output_root, config_dir, groups, label, five_view, top_pathways, permutations, bootstrap_iterations):
    output_root.mkdir(parents=True, exist_ok=True)
    states, all_ids = base.load_states(data_root)
    stable_ids = stats.stable_analysis_universe(groups)
    patient_ids, affinities = load_saved_affinities(data_root, stable_root, output_root, config_dir, five_view)
    patient_states = [states[case_id] for case_id in all_ids]
    ct_features, ct_table_all, ct_audit = load_ct_table(patient_states, patient_ids, data_root, config_dir)
    ct_table = {case_id: ct_table_all[case_id] for case_id in stable_ids}
    rna_pathways, rna_scores = base.load_expression_scores(states, config_dir)
    rna_table = {case_id: {pathway: float(rna_scores.loc[case_id, pathway]) for pathway in rna_pathways if case_id in rna_scores.index and np.isfinite(rna_scores.loc[case_id, pathway])} for case_id in stable_ids}
    wxs_features, wxs_table_all = base.load_table(data_root / "wxs/wxs_discovery_features.csv")
    wxs_table = {case_id: wxs_table_all[case_id] for case_id in stable_ids}
    mutation_features = [feature for feature in wxs_features if feature.startswith("mutation::")]
    cnv_features, cnv_table_all = base.load_table(data_root / "cnv/case_features.csv")
    cnv_table = {case_id: cnv_table_all[case_id] for case_id in stable_ids}

    base.write_csv(output_root / "analysis_universe.csv", [{"case_id": case_id, "group_id": group} for group, members in groups.items() for case_id in members])
    base.write_json(output_root / "ct_feature_preprocessing_audit.json", ct_audit)
    base.write_csv(output_root / "ct_production_feature_matrix.csv", [{"case_id": case_id, **ct_table_all[case_id]} for case_id in stable_ids])

    continuous_specs = (("rna_hallmark", rna_table, rna_pathways), ("ct_radiomics", ct_table, ct_features), ("cnv_continuous", cnv_table, cnv_features))
    summary_rows = []
    for modality, table, features in continuous_specs:
        omnibus = stats.continuous_omnibus(table, features, groups)
        selected = [row["feature"] for row in omnibus if row["q_value"] is not None and row["q_value"] < .05]
        posthoc = stats.continuous_posthoc(table, selected, groups, bootstrap_iterations, 20260908)
        descriptive = stats.continuous_state_vs_rest_descriptive(table, features, groups, 0)
        base.write_csv(output_root / f"{modality}_omnibus.csv", omnibus)
        base.write_csv(output_root / f"{modality}_posthoc.csv", posthoc)
        base.write_csv(output_root / f"{modality}_state_vs_rest_descriptive.csv", descriptive)
        summary_rows.append({"modality": modality, "omnibus_feature_count": len(features), "omnibus_fdr_significant_count": len(selected), "posthoc_row_count": len(posthoc)})

    wxs_omnibus = stats.binary_permutation_omnibus(wxs_table, mutation_features, groups, permutations, 20260908)
    wxs_selected = [row["feature"] for row in wxs_omnibus if row["q_value"] is not None and row["q_value"] < .05]
    base.write_csv(output_root / "wxs_mutation_omnibus.csv", wxs_omnibus)
    base.write_csv(output_root / "wxs_mutation_posthoc.csv", stats.binary_posthoc(wxs_table, wxs_selected, groups))
    summary_rows.append({"modality": "WXS", "omnibus_feature_count": len(mutation_features), "omnibus_fdr_significant_count": len(wxs_selected), "posthoc_row_count": len(wxs_selected) * (len(groups) * (len(groups) - 1) // 2)})

    event_features, event_table = load_event_table(cnv_table, cnv_features)
    event_omnibus = stats.binary_permutation_omnibus(event_table, event_features, groups, permutations, 20260909)
    event_selected = [row["feature"] for row in event_omnibus if row["q_value"] is not None and row["q_value"] < .05]
    base.write_csv(output_root / "cnv_event_omnibus_secondary.csv", event_omnibus)
    base.write_csv(output_root / "cnv_event_posthoc_secondary.csv", stats.binary_posthoc(event_table, event_selected, groups))
    summary_rows.append({"modality": "CNV event secondary", "omnibus_feature_count": len(event_features), "omnibus_fdr_significant_count": len(event_selected), "posthoc_row_count": len(event_selected) * (len(groups) * (len(groups) - 1) // 2)})

    clinical_records = clinical_table(states)
    clinical_omnibus, clinical_posthoc, survival_global = clinical_outputs(clinical_records, groups, output_root, permutations, bootstrap_iterations)
    summary_rows.append({"modality": "clinical", "omnibus_feature_count": len(clinical_omnibus), "omnibus_fdr_significant_count": sum(row.get("q_value") is not None and row["q_value"] < .05 for row in clinical_omnibus), "posthoc_row_count": len(clinical_posthoc)})

    global_affinity, pair_affinity, audits = stats.affinity_statistics(affinities, patient_ids, groups, permutations)
    base.write_csv(output_root / "modality_permanova_global.csv", [{key: value for key, value in row.items() if not key.startswith("permdisp_")} for row in global_affinity])
    base.write_csv(output_root / "modality_permdisp_global.csv", [{"modality": row["modality"], **{key.removeprefix("permdisp_"): value for key, value in row.items() if key.startswith("permdisp_")}} for row in global_affinity])
    base.write_csv(output_root / "modality_permanova_posthoc.csv", [{key: value for key, value in row.items() if not key.startswith("permdisp_")} for row in pair_affinity])
    base.write_csv(output_root / "modality_permdisp_posthoc.csv", [{"modality": row["modality"], "group_a": row["group_a"], "group_b": row["group_b"], **{key.removeprefix("permdisp_"): value for key, value in row.items() if key.startswith("permdisp_")}} for row in pair_affinity])
    base.write_json(output_root / "affinity_audit.json", audits)
    summary_rows.append({"modality": "WSI affinity", "omnibus_feature_count": None, "omnibus_fdr_significant_count": None, "posthoc_row_count": sum(row["modality"] == "wsi" for row in pair_affinity)})

    base.write_csv(output_root / "technical_confounders.csv", base.core_confounds(data_root, config_dir, states, groups))
    base.write_json(output_root / "wsi_primary_evidence.json", {"primary": "saved production WSI affinity PERMANOVA/PERMDISP", "embedding_dimensions_secondary_only": True})
    base.write_csv(output_root / "characterization_summary.csv", summary_rows)
    manifest = {
        "analysis_type": "post_discovery_characterization",
        "evidence_role": "internal_post_discovery_characterization_not_independent_validation",
        "discovery_labels_fixed": True,
        "analysis_universe": "stable_group_patients_only",
        "analysis_universe_n": len(stable_ids),
        "non_core_patients_excluded_from_primary": True,
        "ct_representation": {"production_consistent": ct_audit["production_consistent"], "feature_count": len(ct_features), "preprocessing_fit_universe_n": len(patient_ids)},
        "wsi_primary_evidence": "affinity_permanova",
        "multiple_testing": {"omnibus": "BH within modality", "continuous_posthoc": "Holm within feature across group pairs", "binary_posthoc": "Holm within feature across group pairs", "affinity_posthoc": "BH within modality across group pairs"},
        "permutations": permutations,
        "bootstrap_iterations": bootstrap_iterations,
        "random_seed": 20260908,
        "label": label,
        "generated_feature_count": len(ct_features),
    }
    base.write_json(output_root / "characterization_manifest.json", manifest)
    base.write_json(output_root / "characterization_summary.json", {"label": label, "group_count": len(groups), "group_sizes": {key: len(value) for key, value in groups.items()}, "patient_count": len(stable_ids), "clinical_global_logrank_p": survival_global.get("p_value"), "ct_feature_count": len(ct_features), "taxonomy_mapping_performed": False})
    return manifest


def run(data_root=ROOT / "output_kirc", output_root=ROOT / "output_kirc_v13/02_post_discovery_characterization", config_dir=ROOT / "configs", permutations=9999, bootstrap_iterations=2000, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configs = [
        ("4view_5core", ROOT / "output_kirc_v12/03_multi_k_accepted_core_stability_v11", False, None),
        ("4view_3state", ROOT / "output_kirc_v12/03_multi_k_accepted_core_stability_v11", False, FOUR_VIEW_MACRO_CORES),
        ("5view_6core", ROOT / "output_kirc_v12/15_five_view_multi_k_stability", True, None),
        ("5view_4state", ROOT / "output_kirc_v12/15_five_view_multi_k_stability", True, FIVE_VIEW_MACRO_CORES),
    ]
    summaries = {}
    for name, stable_root, five_view, mapping in configs:
        cores, _ = base.load_cores(stable_root)
        groups = {key: sorted(value) for key, value in cores.items()} if mapping is None else groups_from_cores(cores, mapping)
        summaries[name] = characterize_groups(data_root, stable_root, output_root / name, config_dir, groups, name, five_view, 25, permutations, bootstrap_iterations)
    base.write_json(output_root / "characterization_summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v13/02_post_discovery_characterization")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
