#!/usr/bin/env python3
"""Complete the frozen four-view state results with compact, reproducible summaries."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from four_view_state_common import DEFAULT_INPUT, DEFAULT_MEMBERSHIP, ROOT, load_membership, load_states
from tools import post_discovery_characterization as stats
from tools.ct_radiomics import build_ct_discovery_feature_matrix
from tools.pathway_enrichment import ssgsea_scores
from tools.subtype_review_common import clinical_table, read_gmt_gene_sets, tool_parameters

CHAR = ROOT / "output_kirc_v14/14_four_view_state_characterization"
MAP = ROOT / "output_kirc_v14/15_four_view_state_known_ccrcc_mapping"
AGENT = ROOT / "output_kirc_v14/11_four_view_no_cnv/agent_review"
AUDIT = ROOT / "output_kirc_v14/12_four_view_core_to_macro_state_audit"
STATE_ORDER = ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]


def write_csv(frame, path):
    frame.to_csv(path, index=False)


def patient_state_features(input_root, membership, output_root):
    ids = membership.case_id.tolist()
    rows = []
    wsi_dir = ROOT / "output_kirc_raw/wsi_tumor_seg"
    for case_id in ids:
        path = wsi_dir / case_id / "patch_probabilities.csv"
        if not path.exists():
            continue
        patches = pd.read_csv(path)
        selected = patches[patches["selected_tumor"].astype(bool)]
        if selected.empty:
            selected = patches[patches["tumor_probability"] >= 0.9]
        if selected.empty:
            continue
        values = {column.removeprefix("prob_"): float(selected[column].mean())
                  for column in selected.columns if column.startswith("prob_")}
        values.update({"case_id": case_id, "state_id": membership.set_index("case_id").loc[case_id, "state_id"],
                       "tumor_patch_n": len(selected), "all_patch_n": len(patches)})
        rows.append(values)
    patient = pd.DataFrame(rows)
    if patient.empty:
        return patient, pd.DataFrame(), pd.DataFrame()
    feature_columns = [x for x in patient.columns if x not in {"case_id", "state_id", "tumor_patch_n", "all_patch_n"}]
    means = patient.groupby("state_id")[feature_columns].mean().reindex(STATE_ORDER)
    means.insert(0, "state_n_with_wsi_phenotype", patient.groupby("state_id").size().reindex(STATE_ORDER))
    omnibus = stats.continuous_omnibus(
        {row.case_id: row[feature_columns].to_dict() for _, row in patient.iterrows()},
        feature_columns,
        {state: patient.loc[patient.state_id == state, "case_id"].tolist() for state in STATE_ORDER},
    )
    write_csv(patient, output_root / "wsi_patient_phenotype_features.csv")
    write_csv(means.reset_index(), output_root / "wsi_state_phenotype_means.csv")
    write_csv(pd.DataFrame(omnibus), output_root / "wsi_state_phenotype_omnibus.csv")
    return patient, means, pd.DataFrame(omnibus)


def ct_summary(input_root, membership, states, output_root):
    payload = build_ct_discovery_feature_matrix(
        [states[x] for x in sorted(states)], config_dir=str(ROOT / "configs"), output_root=str(input_root)
    )
    frame = pd.DataFrame(payload["matrix"], index=payload["patient_ids"], columns=payload["feature_names"]).reindex(membership.case_id)
    omnibus = pd.read_csv(CHAR / "ct_radiomics_omnibus.csv")
    selected = omnibus[omnibus.q_value < 0.05].sort_values("q_value").head(12).feature.tolist()
    selected = [feature for feature in selected if feature in frame.columns]
    means = frame.assign(state_id=membership.set_index("case_id").loc[frame.index, "state_id"]).groupby("state_id")[selected].mean().reindex(STATE_ORDER)
    rows = omnibus[omnibus.feature.isin(selected)].copy()
    write_csv(rows, output_root / "ct_representative_feature_summary.csv")
    write_csv(means.reset_index(), output_root / "ct_state_feature_means.csv")
    return frame, means, rows


def state_profiles(input_root, membership, states, clinical, ct, ct_summary_rows, wsi, rna, wxs, output_root):
    groups = {state: membership.loc[membership.state_id == state, "case_id"].tolist() for state in STATE_ORDER}
    clinical_rows = []
    for state, members in groups.items():
        records = [clinical[x] for x in members]
        ages = pd.to_numeric([record.get("age") for record in records], errors="coerce")
        ages = ages[np.isfinite(ages)]
        row = {"state_id": state, "n": len(members),
               "age_median": float(np.median(ages)) if len(ages) else np.nan,
               "age_q1": float(np.quantile(ages, .25)) if len(ages) else np.nan,
               "age_q3": float(np.quantile(ages, .75)) if len(ages) else np.nan,
               "os_events": int(sum(record.get("os_event") == 1 for record in records))}
        for variable in ("gender", "stage_group", "t_stage", "m_stage", "grade"):
            values = pd.Series([record.get(variable) or "Unknown" for record in records])
            for level, count in values.value_counts().items():
                row[f"{variable}_{level}"] = float(count / len(records))
        stage = pd.Series([record.get("stage_group") or "Unknown" for record in records])
        row["advanced_stage_fraction"] = float(stage.isin(["III", "IV"]).mean())
        row["m1_fraction"] = float(pd.Series([record.get("m_stage") for record in records]).eq("M1").mean())
        row["high_grade_fraction"] = float(pd.Series([record.get("grade") for record in records]).isin(["G3", "G4", "3", "4"]).mean())
        clinical_rows.append(row)
    clinical_profile = pd.DataFrame(clinical_rows)
    write_csv(clinical_profile, output_root / "clinical_state_profile.csv")

    selected_ct = [row["feature"] for row in ct_summary_rows.to_dict("records") if row.get("q_value") is not None and row["q_value"] < .05]
    selected_ct = [feature for feature in selected_ct if feature in ct.columns][:8]
    ct_rows = []
    for state, members in groups.items():
        for feature in selected_ct:
            values = pd.to_numeric(ct.loc[members, feature], errors="coerce").dropna()
            ct_rows.append({"state_id": state, "feature": feature, "n": len(values),
                            "mean": float(values.mean()), "median": float(values.median()),
                            "q1": float(values.quantile(.25)), "q3": float(values.quantile(.75)),
                            "global_q_value": next((row["q_value"] for row in ct_summary_rows.to_dict("records") if row["feature"] == feature), np.nan)})
    write_csv(pd.DataFrame(ct_rows), output_root / "ct_state_profile.csv")

    wsi_features = [column for column in wsi.columns if column not in {"case_id", "state_id", "tumor_patch_n", "all_patch_n"}]
    wsi_rows = []
    for state, members in groups.items():
        for feature in wsi_features:
            values = pd.to_numeric(wsi.loc[wsi.case_id.isin(members), feature], errors="coerce").dropna()
            wsi_rows.append({"state_id": state, "feature": feature, "n": len(values),
                             "mean": float(values.mean()), "median": float(values.median()),
                             "q1": float(values.quantile(.25)), "q3": float(values.quantile(.75))})
    wsi_order = json.loads((Path(input_root) / "candidate_subtype/affinity_patient_order.json").read_text())
    wsi_affinity = np.load(Path(input_root) / "candidate_subtype/wsi_affinity.npy")
    wsi_affinity = pd.DataFrame(wsi_affinity, index=wsi_order, columns=wsi_order)
    for state, members in groups.items():
        values = wsi_affinity.loc[members, members].to_numpy(float)
        np.fill_diagonal(values, np.nan)
        wsi_rows.append({"state_id": state, "feature": "within_state_wsi_affinity_mean", "n": len(members),
                         "mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)),
                         "q1": float(np.nanquantile(values, .25)), "q3": float(np.nanquantile(values, .75))})
    write_csv(pd.DataFrame(wsi_rows), output_root / "wsi_state_profile.csv")

    rna_rows, wxs_rows = [], []
    for state, members in groups.items():
        for feature in rna.columns:
            values = pd.to_numeric(rna.loc[members, feature], errors="coerce").dropna()
            rna_rows.append({"state_id": state, "feature": feature, "n": len(values),
                             "mean": float(values.mean()), "median": float(values.median()),
                             "q1": float(values.quantile(.25)), "q3": float(values.quantile(.75))})
        for feature in wxs.columns:
            values = pd.to_numeric(wxs.loc[members, feature], errors="coerce").fillna(0)
            wxs_rows.append({"state_id": state, "feature": feature.removeprefix("mutation::"), "n": len(values),
                             "mutation_frequency": float((values > 0).mean())})
    write_csv(pd.DataFrame(rna_rows), output_root / "rna_state_profile.csv")
    write_csv(pd.DataFrame(wxs_rows), output_root / "wxs_state_profile.csv")

    subtype_rows = []
    for filename, label in (("clearcode34_state_by_label.csv", "clearcode34"), ("mrna_m1_m4_state_by_label.csv", "mrna_m1_m4")):
        table = pd.read_csv(MAP / filename).set_index("state_id")
        for state in STATE_ORDER:
            values = table.loc[state].astype(float)
            total = values.sum()
            row = {"state_id": state, "reference": label}
            row.update({column: float(value / total) if total else np.nan for column, value in values.items()})
            subtype_rows.append(row)
    write_csv(pd.DataFrame(subtype_rows), output_root / "known_subtype_state_profile.csv")

    identity = pd.read_csv(output_root / "state_identity_card.csv").set_index("state_id")
    profile_rows = []
    for state in STATE_ORDER:
        clinical_row = clinical_profile.set_index("state_id").loc[state]
        row = {"state_id": state, "n": int(clinical_row["n"]),
               "age_median": clinical_row["age_median"], "advanced_stage_fraction": clinical_row["advanced_stage_fraction"],
               "m1_fraction": clinical_row["m1_fraction"], "high_grade_fraction": clinical_row["high_grade_fraction"],
               "os_events": clinical_row["os_events"],
               "top_rna_pathways": identity.loc[state, "top_state_enriched_rna_features"],
               "top_wxs_features": identity.loc[state, "top_state_enriched_wxs_features"],
               "top_ct_features": identity.loc[state, "top_state_enriched_ct_features"],
               "clearcode_ccA_fraction": identity.loc[state, "clearcode_ccA_fraction"],
               "clearcode_ccB_fraction": identity.loc[state, "clearcode_ccB_fraction"]}
        for feature in selected_ct:
            value = ct.loc[groups[state], feature].mean()
            row[f"ct_mean__{feature}"] = float(value)
        for feature in ("mutation::PBRM1", "mutation::BAP1", "mutation::VHL", "mutation::SETD2", "mutation::KDM5C"):
            if feature in wxs.columns:
                row[f"{feature.removeprefix('mutation::')}_frequency"] = float((wxs.loc[groups[state], feature].fillna(0) > 0).mean())
        wsi_state = wsi[wsi.state_id == state]
        row["wsi_within_state_affinity_mean"] = next((item["mean"] for item in wsi_rows if item["state_id"] == state and item["feature"] == "within_state_wsi_affinity_mean"), np.nan)
        profile_rows.append(row)
    write_csv(pd.DataFrame(profile_rows), output_root / "multimodal_state_profile.csv")


def micro_macro_summary(output_root):
    mapping = pd.read_csv(AUDIT / "core_to_macro_state.csv", dtype=str)
    counts = pd.crosstab(mapping.macro_state, mapping.core_id).reindex(index=STATE_ORDER, fill_value=0).fillna(0).astype(int)
    write_csv(counts.reset_index(), output_root / "micro_core_to_macro_state_counts.csv")
    mapping.to_csv(output_root / "micro_core_to_macro_state_membership.csv", index=False)
    return mapping, counts


def noncore_summary(input_root, membership, output_root):
    order = json.loads((Path(input_root) / "candidate_subtype/affinity_patient_order.json").read_text())
    fused = np.load(Path(input_root) / "candidate_subtype/fused_similarity.npy")
    coassign = pd.read_csv(AGENT / "joint_accepted_coassignment_matrix.csv", index_col=0).reindex(index=order, columns=order).to_numpy(float)
    assigned = set(membership.case_id)
    state_map = membership.set_index("case_id")["state_id"].to_dict()
    state_members = {state: membership.loc[membership.state_id == state, "case_id"].tolist() for state in STATE_ORDER}
    acceptance = pd.read_csv(AGENT / "patient_acceptance_frequency.csv").set_index("patient_id")["acceptance_frequency"]
    rows = []
    for i, case_id in enumerate(order):
        accepted_scores, fused_scores = {}, {}
        for state, members in state_members.items():
            indices = [order.index(member) for member in members if member != case_id]
            accepted_scores[state] = float(coassign[i, indices].mean()) if indices else np.nan
            fused_scores[state] = float(fused[i, indices].mean()) if indices else np.nan
        finite = {state: score for state, score in accepted_scores.items() if np.isfinite(score)}
        scores = np.array(list(finite.values()), float)
        probabilities = np.clip(scores - scores.min() + 1e-6, 1e-6, None)
        probabilities /= probabilities.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        ranked = sorted(finite.items(), key=lambda item: item[1], reverse=True)
        fused_ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        fused_values = np.array(list(fused_scores.values()), float)
        fused_prob = np.clip(fused_values - fused_values.min() + 1e-6, 1e-6, None)
        fused_prob /= fused_prob.sum()
        max_partner = float(np.max(np.delete(coassign[i], i)))
        rows.append({"case_id": case_id, "state_id": state_map.get(case_id, "NON_CORE"),
                     "is_non_core": case_id not in assigned, "acceptance_frequency": float(acceptance.get(case_id, np.nan)),
                     "max_coassignment": max_partner, "nearest_state": ranked[0][0], "nearest_state_affinity": ranked[0][1],
                     "second_state_affinity": ranked[1][1], "state_affinity_margin": ranked[0][1] - ranked[1][1],
                     "state_affinity_entropy": entropy, "fused_nearest_state": fused_ranked[0][0],
                     "fused_state_affinity_margin": fused_ranked[0][1] - fused_ranked[1][1],
                     "fused_state_affinity_entropy": float(-(fused_prob * np.log(fused_prob)).sum())})
    result = pd.DataFrame(rows)
    write_csv(result, output_root / "noncore_uncertainty_scores.csv")
    metrics = ["max_coassignment", "state_affinity_margin", "state_affinity_entropy"]
    tests = []
    for metric in metrics:
        assigned_values = result.loc[~result.is_non_core, metric].dropna().to_numpy()
        noncore_values = result.loc[result.is_non_core, metric].dropna().to_numpy()
        u = mannwhitneyu(assigned_values, noncore_values, alternative="two-sided")
        tests.append({"metric": metric, "assigned_n": len(assigned_values), "noncore_n": len(noncore_values),
                      "assigned_mean": float(assigned_values.mean()), "noncore_mean": float(noncore_values.mean()),
                      "mann_whitney_u": float(u.statistic), "p_value": float(u.pvalue),
                      "rank_biserial_core_minus_noncore": float(2 * u.statistic / (len(assigned_values) * len(noncore_values)) - 1)})
    test_frame = pd.DataFrame(tests)
    test_frame["q_value"] = stats.bh_adjust(test_frame.p_value.tolist())
    write_csv(result.groupby("is_non_core")[['acceptance_frequency', 'max_coassignment', 'state_affinity_margin', 'state_affinity_entropy']].mean().reset_index(), output_root / "core_noncore_stability_summary.csv")
    write_csv(test_frame, output_root / "core_noncore_stability_tests.csv")
    return result


def representative_cases(input_root, membership, states, wxs, output_root):
    order = json.loads((Path(input_root) / "candidate_subtype/affinity_patient_order.json").read_text())
    coassign = pd.read_csv(AGENT / "joint_accepted_coassignment_matrix.csv", index_col=0).reindex(index=order, columns=order).to_numpy(float)
    clinical = clinical_table(states)
    wxs_cols = [x for x in wxs.columns if x.startswith("mutation::")]
    rows = []
    for state in STATE_ORDER:
        members = membership.loc[membership.state_id == state, "case_id"].tolist()
        indices = [order.index(x) for x in members]
        local = coassign[np.ix_(indices, indices)].copy()
        np.fill_diagonal(local, np.nan)
        case_id = members[int(np.nanargmax(np.nanmean(local, axis=1)))]
        mutation = [x.removeprefix("mutation::") for x in wxs.loc[case_id, wxs_cols][wxs.loc[case_id, wxs_cols] > 0].index]
        record = clinical.get(case_id, {})
        rows.append({"state_id": state, "case_id": case_id, "state_n": len(members), "age": record.get("age"),
                     "stage": record.get("stage_group"), "m_stage": record.get("m_stage"), "major_mutations": ";".join(mutation),
                     "selection_rule": "accepted-coassignment medoid within state"})
    result = pd.DataFrame(rows)
    write_csv(result, output_root / "representative_state_patients.csv")
    return result


def one_vs_rest_features(frame, membership, state, binary=False, limit=3):
    rows = []
    rest = membership.loc[membership.state_id != state, "case_id"].tolist()
    target = membership.loc[membership.state_id == state, "case_id"].tolist()
    for feature in frame.columns:
        left = pd.to_numeric(frame.reindex(target)[feature], errors="coerce").dropna().to_numpy()
        right = pd.to_numeric(frame.reindex(rest)[feature], errors="coerce").dropna().to_numpy()
        if not len(left) or not len(right):
            continue
        if binary:
            table = [[int((left > 0).sum()), int((left <= 0).sum())], [int((right > 0).sum()), int((right <= 0).sum())]]
            p = fisher_exact(table)[1]
            effect = left.mean() - right.mean()
        else:
            p = mannwhitneyu(left, right, alternative="two-sided").pvalue
            effect = float(np.median(left) - np.median(right))
        rows.append({"feature": feature, "p_value": float(p), "effect": effect})
    if not rows:
        return []
    result = pd.DataFrame(rows)
    result["q_value"] = stats.bh_adjust(result.p_value.tolist())
    result = result[(result.q_value < 0.05) & (result.effect > 0)].sort_values("q_value")
    return result.feature.astype(str).str.replace("mutation::", "", regex=False).head(limit).tolist()


def identity_card(membership, mapping, output_root, rna, ct, wxs):

    cc = pd.read_csv(MAP / "clearcode34_state_by_label.csv").set_index("state_id")
    rows = []
    for state in STATE_ORDER:
        rna_names = one_vs_rest_features(rna, membership, state, limit=3)
        wxs_names = one_vs_rest_features(wxs, membership, state, binary=True, limit=3)
        ct_names = one_vs_rest_features(ct, membership, state, limit=2)
        cc_row = cc.loc[state] if state in cc.index else pd.Series(dtype=float)
        cc_total = float(cc_row.sum()) if len(cc_row) else 0
        rows.append({"state_id": state, "state_n": int((membership.state_id == state).sum()),
                     "micro_core_ids": ";".join(mapping.loc[mapping.macro_state == state, "core_id"]),
                     "top_state_enriched_rna_features": ";".join(rna_names) or "none detected",
                     "top_state_enriched_wxs_features": ";".join(wxs_names) or "none detected",
                     "top_state_enriched_ct_features": ";".join(ct_names) or "none detected",
                     "clearcode_ccA_fraction": float(cc_row.get("ccA", np.nan) / cc_total) if cc_total else np.nan,
                     "clearcode_ccB_fraction": float(cc_row.get("ccB", np.nan) / cc_total) if cc_total else np.nan})
    result = pd.DataFrame(rows)
    write_csv(result, output_root / "state_identity_card.csv")
    return result


def run(input_root=DEFAULT_INPUT, membership_path=DEFAULT_MEMBERSHIP, output_root=ROOT / "output_kirc_v14/18_four_view_state_result_completion", force=False):
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(input_root)
    membership = load_membership(membership_path)
    states = load_states(input_root)
    if not set(membership.case_id).issubset(states):
        raise ValueError("State membership contains patients absent from patient_states")
    wsi, _, _ = patient_state_features(input_root, membership, output_root)
    ct, _, ct_rows = ct_summary(input_root, membership, states, output_root)
    mapping, _ = micro_macro_summary(output_root)
    noncore = noncore_summary(input_root, membership, output_root)
    wxs = pd.read_csv(input_root / "wxs/wxs_discovery_features.csv", index_col=0).reindex(membership.case_id)
    reps = representative_cases(input_root, membership, states, wxs, output_root)
    rna_genes = pd.read_csv(input_root / "rna/case_pathway_features.csv", index_col=0).reindex(membership.case_id)
    gene_sets, _ = read_gmt_gene_sets(tool_parameters(str(ROOT / "configs"), "rna")["pathway_gene_sets_path"])
    gene_sets = {name: [gene for gene in genes if gene in rna_genes.columns] for name, genes in gene_sets.items()}
    rna_sets = {name: genes for name, genes in gene_sets.items() if len(genes) >= 15}
    rna = ssgsea_scores(rna_genes, rna_sets, 15).reindex(membership.case_id)
    card = identity_card(membership, mapping, output_root, rna, ct, wxs)
    state_profiles(input_root, membership, states, clinical_table(states), ct, ct_rows, wsi, rna, wxs, output_root)
    manifest = {"experiment": "four_view_state_result_completion", "input_root": str(input_root.resolve()),
                "membership_file": str(Path(membership_path).resolve()), "patient_count": int(len(membership)),
                "state_sizes": membership.groupby("state_id").size().to_dict(), "active_modalities": ["ct", "wsi", "rna", "wxs"],
                "wsi_summary": "patient-level means of tumor-selected patch class probabilities; no patch-level pseudoreplication",
                "ct_summary": "production-consistent 308-dimensional CT representation; top omnibus-FDR features exported",
                "noncore_summary": "post hoc stability explanation; non-core patients are not assigned to states",
                "profile_outputs": ["clinical_state_profile.csv", "ct_state_profile.csv", "wsi_state_profile.csv",
                                    "rna_state_profile.csv", "wxs_state_profile.csv", "known_subtype_state_profile.csv",
                                    "multimodal_state_profile.csv"],
                "outputs": {"wsi_patient_n": int(len(wsi)), "noncore_n": int(noncore.is_non_core.sum()), "representative_n": int(len(reps)), "identity_card_n": int(len(card))}}
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--membership-path", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/18_four_view_state_result_completion")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
