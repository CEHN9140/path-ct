#!/usr/bin/env python3
"""Offline test of three candidate macro-states built from five stable cores."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts_2026_8_17 import analyze_multi_k_stable_cores as base


MACRO_STATE_CORES = {
    "STATE_A": ("CORE01", "CORE03"),
    "STATE_B": ("CORE02", "CORE05"),
    "STATE_C": ("CORE04",),
}


def macro_state_members(cores):
    return {
        state: sorted({patient for core in core_ids for patient in cores.get(core, [])})
        for state, core_ids in MACRO_STATE_CORES.items()
    }


def plot_macro_fused_structure(path, similarity, ids, states, random_state):
    from sklearn.manifold import SpectralEmbedding
    from utils.visualization import configure_matplotlib

    configure_matplotlib()
    import matplotlib.pyplot as plt

    ordered = [patient for state in sorted(states) for patient in states[state] if patient in ids]
    indices = [ids.index(patient) for patient in ordered]
    normalized = base.affinity_characterization(similarity, ids, states)["similarity"]
    normalized = normalized[np.ix_(indices, indices)]
    distance = np.maximum(0, 1 - normalized)
    centered = -.5 * distance ** 2
    centered -= centered.mean(axis=0, keepdims=True)
    centered -= centered.mean(axis=1, keepdims=True)
    centered += centered.mean()
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    positive = np.flatnonzero(eigenvalues > 1e-10)[::-1]
    pcoa = eigenvectors[:, positive[:2]] * np.sqrt(eigenvalues[positive[:2]]) if len(positive) >= 2 else np.zeros((len(ordered), 2))
    explained = eigenvalues[positive[:2]].sum() / eigenvalues[positive].sum() if positive.size else 0
    spectral = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=random_state).fit_transform(normalized)
    degree = normalized.sum(axis=1)
    operator = normalized / np.sqrt(np.outer(degree, degree))
    values, vectors = np.linalg.eigh(operator)
    order = np.argsort(values)[::-1][1:3]
    diffusion = vectors[:, order] * values[order] if len(order) == 2 else np.zeros((len(ordered), 2))
    labels = {patient: state for state, members in states.items() for patient in members}
    colors = {state: plt.get_cmap("tab10")(i) for i, state in enumerate(sorted(states))}
    projections = [(pcoa, f"PCoA (first two positive axes: {explained:.1%})", "PCoA-1", "PCoA-2"), (spectral, "Spectral embedding", "Spectral-1", "Spectral-2"), (diffusion, "Diffusion map", "DC-1", "DC-2")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (coordinates, title, xlabel, ylabel) in zip(axes, projections):
        for state in sorted(states):
            selected = [i for i, patient in enumerate(ordered) if labels.get(patient) == state]
            axis.scatter(coordinates[selected, 0], coordinates[selected, 1], label=state, color=colors[state], s=28)
        axis.set_title(title); axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.grid(alpha=.15)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[state], label=state) for state in sorted(states)]
    figure.suptitle(f"Candidate macro-state structure in fused affinity space (n={len(ordered)}; exploratory)")
    figure.legend(handles=handles, title="Macro-state", bbox_to_anchor=(1.01, .8), loc="upper left")
    figure.tight_layout(); base.save_figure(figure, path)
    return ["pcoa", "spectral", "diffusion"]


def compare_fused_models(five_cores, macro_states, similarity, ids):
    rows = []
    for model, groups in (("five_core", five_cores), ("three_macro_state", macro_states)):
        separation, _, tests, _ = base.core_embedding_analysis({"fused": similarity}, ids, groups)
        test = tests["fused"]
        values = [row.get("silhouette") for row in separation if row["modality"] == "fused" and row.get("silhouette") is not None]
        rows.append({"model": model, "group_count": len(groups), "patient_count": sum(len(x) for x in groups.values()), "median_group_silhouette": base.rounded(np.median(values)) if values else None, **{f"permanova_{key}": value for key, value in test["permanova"].items()}, **{f"permdisp_{key}": value for key, value in test["permdisp"].items()}})
    return rows


def stage_number(value):
    text = str(value or "").upper().replace("STAGE ", "")
    return {"I": 1, "II": 2, "III": 3, "IV": 4}.get(text)


def stage_adjusted_survival(records, states):
    import pandas as pd
    from lifelines import CoxPHFitter

    rows = [{"case_id": patient, "state": state, "stage_number": stage_number(records.get(patient, {}).get("stage_group")), "os_time": base.finite(records.get(patient, {}).get("os_time")), "os_event": int(records.get(patient, {}).get("os_event") or 0)} for state, members in states.items() for patient in members]
    frame = pd.DataFrame(rows).dropna(subset=["stage_number", "os_time"])
    frame = pd.get_dummies(frame.drop(columns="case_id"), columns=["state"], dtype=float)
    state_columns = [f"state_{state}" for state in sorted(states) if f"state_{state}" in frame]
    if len(frame) < 10 or frame.os_event.sum() < 3 or len(state_columns) < 2:
        return [{"model": "state_plus_stage", "status": "not_estimable", "reason": "insufficient_events_or_covariates"}]
    reference = state_columns[0]
    frame = frame.drop(columns=reference)
    model = CoxPHFitter(penalizer=.1).fit(frame, duration_col="os_time", event_col="os_event")
    return [{"model": "state_plus_stage", "covariate": covariate, "status": "estimable", "hazard_ratio": base.rounded(np.exp(model.params_[covariate])), "ci_low": base.rounded(np.exp(model.confidence_intervals_.loc[covariate].iloc[0])), "ci_high": base.rounded(np.exp(model.confidence_intervals_.loc[covariate].iloc[1])), "p_value": base.rounded(model.summary.loc[covariate, "p"]), "reference_state": reference.removeprefix("state_"), "adjustment": "stage_group ordinal I=1, II=2, III=3, IV=4"} for covariate in ["stage_number", *sorted(set(state_columns) - {reference})]]


def evidence_table(five_cores, macro_states, five_rna, macro_rna, five_wxs, macro_wxs, five_cnv, macro_cnv, five_clinical, macro_clinical, five_fused, macro_fused):
    def count(rows):
        return sum(row.get("q_value") is not None and row["q_value"] < .05 for row in rows)
    def pair_count(rows):
        return len({(row.get("core_a"), row.get("core_b")) for row in rows if row.get("q_value") is not None and row["q_value"] < .05})
    return [
        {"evidence": "fused median group silhouette", "five_core": five_fused[0].get("median_group_silhouette"), "three_macro_state": macro_fused[0].get("median_group_silhouette")},
        {"evidence": "fused PERMANOVA R2", "five_core": five_fused[0].get("permanova_permanova_r2"), "three_macro_state": macro_fused[0].get("permanova_permanova_r2")},
        {"evidence": "RNA pairwise FDR-supported pairs", "five_core": pair_count(five_rna), "three_macro_state": pair_count(macro_rna)},
        {"evidence": "WXS pairwise FDR-supported pairs", "five_core": pair_count(five_wxs), "three_macro_state": pair_count(macro_wxs)},
        {"evidence": "CNV pairwise FDR-supported pairs", "five_core": pair_count(five_cnv), "three_macro_state": pair_count(macro_cnv)},
        {"evidence": "clinical pairwise FDR-supported pairs", "five_core": pair_count(five_clinical), "three_macro_state": pair_count(macro_clinical)},
        {"evidence": "minimum group n", "five_core": min(map(len, five_cores.values())), "three_macro_state": min(map(len, macro_states.values()))},
    ]


def rename_rows(rows, key="core_id"):
    return [{"state_id": row[key], **{name: value for name, value in row.items() if name != key}} if key in row else dict(row) for row in rows]


def run(data_root=Path("output_kirc"), stable_root=Path("output_kirc_v11/experiment_multi_k_accepted_core_stability"), output_root=Path("output_kirc_v11/experiment_macro_state_analysis"), config_dir=Path("configs"), top_pathways=25, top_cnv=25, random_state=42, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True); figures = output_root / "figures"; figures.mkdir()
    states, all_ids = base.load_states(data_root); five_cores, _ = base.load_cores(stable_root); macro_states = macro_state_members(five_cores); stable_ids = sorted(set().union(*map(set, macro_states.values())))
    base.write_csv(output_root / "macro_state_definition.csv", [{"state_id": state, "source_cores": "+".join(MACRO_STATE_CORES[state]), "patient_n": len(members)} for state, members in macro_states.items()]); base.write_csv(output_root / "core_to_macro_state.csv", [{"core_id": core, "state_id": state} for state, core_ids in MACRO_STATE_CORES.items() for core in core_ids])

    ct_features, ct_table = base.ct_radiomics_table(states, stable_ids); ct_rows, ct_pairs = base.radiomics_analysis(ct_table, ct_features, macro_states, stable_ids); base.write_csv(output_root / "ct_radiomics_macro_state_vs_rest.csv", rename_rows(ct_rows)); base.write_csv(output_root / "ct_radiomics_macro_state_pairwise.csv", ct_pairs)
    pathways, scores = base.load_expression_scores(states, config_dir); rna_rows, rna_pairs = base.rna_analysis(states, config_dir, macro_states, stable_ids, pathways, scores); base.write_csv(output_root / "rna_hallmark_macro_state_vs_rest.csv", rename_rows(rna_rows)); base.write_csv(output_root / "rna_hallmark_macro_state_pairwise.csv", rna_pairs)
    wxs_features, wxs_table = base.load_table(data_root / "wxs/wxs_discovery_features.csv"); mutation_features = [feature for feature in wxs_features if feature.startswith("mutation::")]; wxs_rows, wxs_pairs = base.binary_analysis(wxs_table, mutation_features, macro_states, stable_ids, "mutation"); base.write_csv(output_root / "wxs_macro_state_vs_rest.csv", rename_rows(wxs_rows)); base.write_csv(output_root / "wxs_macro_state_pairwise.csv", wxs_pairs)
    cnv_features, cnv_table = base.load_table(data_root / "cnv/case_features.csv"); cnv_cont, cnv_event, cnv_pairs_cont, cnv_pairs_event = base.cnv_analysis(cnv_table, cnv_features, macro_states, stable_ids); base.write_csv(output_root / "cnv_continuous_macro_state_vs_rest.csv", rename_rows(cnv_cont)); base.write_csv(output_root / "cnv_gain_loss_macro_state_vs_rest.csv", rename_rows(cnv_event)); base.write_csv(output_root / "cnv_continuous_macro_state_pairwise.csv", cnv_pairs_cont); base.write_csv(output_root / "cnv_gain_loss_macro_state_pairwise.csv", cnv_pairs_event)

    affinity_paths = {"fused": data_root / "candidate_subtype/fused_similarity.npy"}; affinity_ids = json.loads((data_root / "candidate_subtype/affinity_patient_order.json").read_text(encoding="utf-8")); affinities = {name: np.load(path) for name, path in affinity_paths.items() if path.is_file()}; separation, distances, tests, audits = base.core_embedding_analysis(affinities, affinity_ids, macro_states); base.write_csv(output_root / "fused_macro_state_separation.csv", rename_rows(separation)); base.write_csv(output_root / "fused_macro_state_permanova.csv", [{"modality": "fused", **tests["fused"]["permanova"]}]); base.write_csv(output_root / "fused_macro_state_permdisp.csv", [{"modality": "fused", **tests["fused"]["permdisp"]}]); base.write_json(output_root / "fused_affinity_audit.json", audits); fused_comparison = compare_fused_models(five_cores, macro_states, affinities["fused"], affinity_ids); base.write_csv(output_root / "fused_model_comparison.csv", fused_comparison)

    from tools.subtype_review_common import clinical_table
    clinical_records = clinical_table(states); availability_rows, availability = base.clinical_availability(clinical_records, macro_states, stable_ids); base.write_csv(output_root / "clinical_availability.csv", availability_rows); base.write_json(output_root / "clinical_availability_summary.json", availability); clinical_rows, clinical_pairs, survival_rows = base.clinical_analysis(clinical_records, macro_states, stable_ids, availability["eligible_variables"], availability["survival_eligible"]); base.write_csv(output_root / "clinical_macro_state_vs_rest.csv", rename_rows(clinical_rows)); base.write_csv(output_root / "clinical_macro_state_pairwise.csv", clinical_pairs); base.write_csv(output_root / "clinical_survival_macro_state_vs_rest.csv", rename_rows(survival_rows)); base.write_csv(output_root / "clinical_patient_records.csv", [{"case_id": patient, "state_id": next(state for state, members in macro_states.items() if patient in members), **clinical_records[patient]} for patient in stable_ids if patient in clinical_records])
    five_rna, five_rna_pairs = base.rna_analysis(states, config_dir, five_cores, stable_ids, pathways, scores); five_wxs, five_wxs_pairs = base.binary_analysis(wxs_table, mutation_features, five_cores, stable_ids, "mutation"); five_cnv, _, five_cnv_pairs, _ = base.cnv_analysis(cnv_table, cnv_features, five_cores, stable_ids); _, five_clinical_pairs, _ = base.clinical_analysis(clinical_records, five_cores, stable_ids); base.write_csv(output_root / "technical_confounder_macro_state.csv", rename_rows(base.core_confounds(data_root, config_dir, states, macro_states))); base.write_csv(output_root / "stage_adjusted_survival.csv", stage_adjusted_survival(clinical_records, macro_states)); base.write_csv(output_root / "five_core_vs_three_state_evidence.csv", evidence_table(five_cores, macro_states, five_rna_pairs, rna_pairs, five_wxs_pairs, wxs_pairs, five_cnv_pairs, cnv_pairs_cont, five_clinical_pairs, clinical_pairs, fused_comparison[:1], fused_comparison[1:]))

    core_order = sorted(macro_states); selected_pathways = base.select_pathways(rna_rows, top_pathways); lookup = {(row["core_id"], row["pathway"]): row for row in rna_rows}; matrix = np.asarray([[lookup.get((state, pathway), {}).get("smd") or 0 for state in core_order] for pathway in selected_pathways]); stars = [["***" if (lookup.get((state, pathway), {}).get("q_value") or 1) < .001 else "**" if (lookup.get((state, pathway), {}).get("q_value") or 1) < .01 else "*" if (lookup.get((state, pathway), {}).get("q_value") or 1) < .05 else "" for state in core_order] for pathway in selected_pathways]; base.plot_heatmap(figures / "macro_state_pathway_smd_heatmap", matrix, selected_pathways, core_order, "Candidate macro-state Hallmark pathway SMD", stars, True); base.plot_bubbles(figures / "macro_state_pathway_bubble_plot", rna_rows, core_order, selected_pathways)
    selected_cnv = base.select_cnv_heatmap_features(cnv_cont, top_cnv); cnv_lookup = {(row["core_id"], row["feature"]): row for row in cnv_cont}; base.plot_heatmap(figures / "macro_state_cnv_effect_heatmap", np.asarray([[cnv_lookup.get((state, feature), {}).get("cliffs_delta") or 0 for state in core_order] for feature in selected_cnv]), selected_cnv, core_order, "Candidate macro-state CNV Cliff's delta", diverging=True); base.plot_oncoplot(figures / "macro_state_driver_mutation_oncoplot", wxs_table, mutation_features, macro_states, stable_ids, {state: "" for state in core_order}, {state: "" for state in core_order}); base.plot_survival_km(figures / "macro_state_overall_survival_km", clinical_records, macro_states); projection = plot_macro_fused_structure(figures / "macro_state_fused_structure", affinities["fused"], affinity_ids, macro_states, random_state)
    summary = {"state_count": 3, "source_core_count": len(five_cores), "patient_count": len(stable_ids), "state_sizes": {state: len(members) for state, members in macro_states.items()}, "figures": projection, "analysis_scope": "59 stable-core patients", "source_stable_core_sha256": base.file_sha256(stable_root / "stable_core_membership.csv")}; base.write_json(output_root / "macro_state_analysis_summary.json", summary); base.write_json(output_root / "macro_state_analysis_manifest.json", {"analysis_parameters": {"top_pathways": top_pathways, "top_cnv": top_cnv, "random_state": random_state}, "source": str(stable_root), "preserved_core_mapping": str(output_root / "core_to_macro_state.csv")}); return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path("output_kirc")); parser.add_argument("--stable-root", type=Path, default=Path("output_kirc_v11/experiment_multi_k_accepted_core_stability")); parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v11/experiment_macro_state_analysis")); parser.add_argument("--config-dir", type=Path, default=Path("configs")); parser.add_argument("--top-pathways", type=int, default=25); parser.add_argument("--top-cnv", type=int, default=25); parser.add_argument("--random-state", type=int, default=42); parser.add_argument("--force", action="store_true"); print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
