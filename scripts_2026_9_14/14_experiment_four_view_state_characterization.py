#!/usr/bin/env python3
"""Characterize the frozen four-view macro-states without CNV."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from four_view_state_common import DEFAULT_INPUT, DEFAULT_MEMBERSHIP, ROOT, groups, load_membership, load_states, load_table, write_manifest
from tools import post_discovery_characterization as stats
from tools.subtype_review_common import clinical_table, read_gmt_gene_sets, tool_parameters
from tools.pathway_enrichment import ssgsea_scores
from tools.ct_radiomics import build_ct_discovery_feature_matrix
from tools.multimodal_consistency_check import normalized_affinity_with_audit, permanova_metrics, permdisp_metrics


def continuous_table(frame, ids):
    return {case_id: frame.loc[case_id].to_dict() for case_id in ids if case_id in frame.index}

def stage_binary_cox(records, groups):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    rows=[]; data=[]
    stage={"I":0,"II":0,"III":1,"IV":1}
    for state,members in groups.items():
        for case_id in members:
            record=records[case_id]
            if record.get("os_time") is not None and record.get("stage_group") in stage:
                data.append({"time":record["os_time"],"event":int(record.get("os_event") or 0),"state":state,"advanced_stage":stage[record["stage_group"]]})
    frame=pd.DataFrame(data)
    if frame.empty or frame.event.sum()<3 or frame.state.nunique()<2: return [{"model":"state_plus_binary_stage","status":"not_estimable"}]
    frame=pd.get_dummies(frame,columns=["state"],dtype=float); state_cols=sorted(x for x in frame if x.startswith("state_")); frame=frame.drop(columns=state_cols[0])
    model=CoxPHFitter(penalizer=.1).fit(frame,duration_col="time",event_col="event")
    for covariate in ["advanced_stage",*state_cols[1:]]:
        interval=model.confidence_intervals_.loc[covariate].to_numpy(); rows.append({"model":"state_plus_binary_stage","covariate":covariate,"status":"estimable","hazard_ratio":float(np.exp(model.params_[covariate])),"ci_low":float(np.exp(interval[0])),"ci_high":float(np.exp(interval[1])),"p_value":float(model.summary.loc[covariate,"p"]),"reference_state":state_cols[0].removeprefix("state_") if covariate.startswith("state_") else None,"adjustment":"stage I-II vs III-IV","available_n":len(frame),"events":int(frame.event.sum()),"analysis_role":"secondary_exploratory"})
    ph=proportional_hazard_test(model,frame,time_transform="rank").summary
    for covariate,row in ph.iterrows(): rows.append({"model":"cox_ph_assumption","covariate":covariate,"status":"diagnostic","p_value":float(row["p"]),"analysis_role":"diagnostic_not_hypothesis_test"})
    return rows


def run(input_root=DEFAULT_INPUT, membership_path=DEFAULT_MEMBERSHIP, output_root=ROOT / "output_kirc_v14/14_four_view_state_characterization", permutations=9999, bootstrap_iterations=2000, force=False):
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        import shutil
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_root, membership = Path(input_root), load_membership(membership_path)
    state_groups, states = groups(membership), load_states(input_root)
    ids = membership.case_id.tolist()
    if not set(ids).issubset(states):
        raise ValueError("State membership contains patients absent from patient_states")

    rna_genes = load_table(input_root / "rna/case_pathway_features.csv").reindex(ids)
    gmt = tool_parameters(str(ROOT / "configs"), "rna")["pathway_gene_sets_path"]
    gene_sets, _ = read_gmt_gene_sets(gmt)
    gene_sets = {name: [gene for gene in genes if gene in rna_genes.columns] for name, genes in gene_sets.items()}
    gene_sets = {name: genes for name, genes in gene_sets.items() if len(genes) >= 15}
    rna = ssgsea_scores(rna_genes, gene_sets, 15).reindex(ids)
    wxs = load_table(input_root / "wxs/wxs_discovery_features.csv").reindex(ids)
    wxs_features = [x for x in wxs.columns if x.startswith("mutation::")]
    clinical = clinical_table({case_id: states[case_id] for case_id in ids})
    event_time_audit = []
    for case_id in ids:
        demographic = dict((states[case_id].get("clinical", {}) or {}).get("demographic", {}) or {})
        if str(demographic.get("vital_status", "")).strip().lower() in {"dead", "deceased", "1", "true", "yes"} and demographic.get("days_to_death") in (None, ""):
            event_time_audit.append(case_id)
    (output_root / "survival_event_time_audit.json").write_text(json.dumps({"dead_without_days_to_death": event_time_audit, "handled_as": "excluded_from_OS_analysis"}, indent=2), encoding="utf-8")

    ct_payload = build_ct_discovery_feature_matrix(
        [states[case_id] for case_id in sorted(states)],
        config_dir=str(ROOT / "configs"), output_root=str(input_root)
    )
    if list(ct_payload["patient_ids"]) != sorted(states):
        raise ValueError("CT production feature order does not match patient_states")
    ct = pd.DataFrame(ct_payload["matrix"], index=ct_payload["patient_ids"], columns=ct_payload["feature_names"]).reindex(ids)
    ct_features = list(ct.columns)
    ct_table = continuous_table(ct, ids)
    ct_rows = stats.continuous_omnibus(ct_table, ct_features, state_groups)
    ct_selected = [r["feature"] for r in ct_rows if r["q_value"] is not None and r["q_value"] < .05]
    pd.DataFrame(ct_rows).to_csv(output_root / "ct_radiomics_omnibus.csv", index=False)
    pd.DataFrame(stats.continuous_posthoc(ct_table, ct_selected, state_groups, bootstrap_iterations)).to_csv(output_root / "ct_radiomics_posthoc.csv", index=False)
    (output_root / "ct_radiomics_preprocessing_audit.json").write_text(json.dumps(ct_payload.get("audit", {}), indent=2), encoding="utf-8")

    for name, table, features in (("rna_hallmark", rna, list(rna.columns)), ("wxs_mutation", wxs, wxs_features)):
        if name.startswith("wxs"):
            rows = stats.binary_permutation_omnibus({k: v.to_dict() for k, v in table.iterrows()}, features, state_groups, permutations, 20260914)
            posthoc = stats.binary_posthoc({k: v.to_dict() for k, v in table.iterrows()}, [r["feature"] for r in rows if r["q_value"] is not None and r["q_value"] < .05], state_groups)
        else:
            rows = stats.continuous_omnibus(continuous_table(table, ids), features, state_groups)
            posthoc = stats.continuous_posthoc(continuous_table(table, ids), [r["feature"] for r in rows if r["q_value"] is not None and r["q_value"] < .05], state_groups, bootstrap_iterations)
        pd.DataFrame(rows).to_csv(output_root / f"{name}_omnibus.csv", index=False)
        pd.DataFrame(posthoc).to_csv(output_root / f"{name}_posthoc.csv", index=False)

    clinical_rows = []
    age = {case_id: {"age": record.get("age")} for case_id, record in clinical.items()}
    clinical_rows.extend(stats.continuous_omnibus(age, ["age"], state_groups))
    for variable in ("stage_group", "t_stage", "m_stage", "grade", "gender"):
        clinical_rows.append(stats.categorical_omnibus(clinical, variable, state_groups, permutations, 20260914))
    for row, q in zip(clinical_rows, stats.bh_adjust([r["p_value"] for r in clinical_rows])): row["q_value"] = q
    pd.DataFrame(clinical_rows).to_csv(output_root / "clinical_omnibus.csv", index=False)
    survival, survival_pairs = stats.survival_analysis(clinical, state_groups)
    pd.DataFrame([survival]).to_csv(output_root / "survival_global.csv", index=False)
    pd.DataFrame(survival_pairs).to_csv(output_root / "survival_posthoc.csv", index=False)
    pd.DataFrame(stats.stage_adjusted_survival(clinical, state_groups)).to_csv(output_root / "stage_adjusted_survival.csv", index=False)
    pd.DataFrame(stage_binary_cox(clinical, state_groups)).to_csv(output_root / "stage_binary_adjusted_survival.csv", index=False)

    affinity_paths = {"ct": input_root / "candidate_subtype/ct_affinity.npy", "wsi": input_root / "candidate_subtype/wsi_affinity.npy", "rna": input_root / "candidate_subtype/rna_affinity.npy", "wxs": input_root / "wxs/wxs_affinity.npy"}
    affinity_rows, audit = [], {}
    labels = np.array([next(state for state, members in state_groups.items() if case_id in members) for case_id in ids])
    order = json.loads((input_root / "candidate_subtype/affinity_patient_order.json").read_text())
    positions = [order.index(case_id) for case_id in ids]
    for modality, path in affinity_paths.items():
        matrix = np.load(path)[np.ix_(positions, positions)]
        normalized, audit[modality] = normalized_affinity_with_audit(matrix)
        p = permanova_metrics(normalized, labels, permutations=permutations, seed=20260914)
        d = permdisp_metrics(normalized, labels, permutations=permutations, seed=20260915)
        affinity_rows.append({"modality": modality, **p, **{"permdisp_" + k: v for k, v in d.items()}})
    pd.DataFrame(affinity_rows).to_csv(output_root / "affinity_permanova_permdisp.csv", index=False)
    write_manifest(output_root / "manifest.json", {"experiment": "four_view_state_characterization", "active_modalities": ["ct", "wsi", "rna", "wxs"], "membership_file": str(Path(membership_path).resolve()), "patient_count": len(ids), "state_sizes": {k: len(v) for k, v in state_groups.items()}, "cnv_included": False, "ct_feature_source": "production_consistent_103_patient_preprocessing_then_86_patient_subset", "ct_feature_count": len(ct_features), "rna_gene_count": int(rna_genes.shape[1]), "rna_pathway_count": len(rna.columns), "rna_min_pathway_overlap": 15, "rna_pathway_source": str((input_root / "rna/case_pathway_features.csv").resolve()), "rna_source_measurement": "tpm_unstranded", "rna_transform": "log2(TPM + 1)", "rna_filtering": "protein_coding_only; TPM >= 1 in at least 20% of the 103-patient cohort", "rna_pathway_matrix_is_independent_of_discovery_top_mad": True, "affinity_audit": audit})
    return {"output_root": str(output_root), "patient_count": len(ids), "state_sizes": {k: len(v) for k, v in state_groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--membership-path", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/14_four_view_state_characterization")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
