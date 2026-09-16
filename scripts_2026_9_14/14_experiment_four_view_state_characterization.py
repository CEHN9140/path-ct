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
from tools.multimodal_consistency_check import normalized_affinity_with_audit, permanova_metrics, permdisp_metrics


def continuous_table(frame, ids):
    return {case_id: frame.loc[case_id].to_dict() for case_id in ids if case_id in frame.index}


def run(input_root=DEFAULT_INPUT, membership_path=DEFAULT_MEMBERSHIP, output_root=ROOT / "output_kirc_v14/14_four_view_state_characterization", permutations=999, bootstrap_iterations=1000, force=False):
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    input_root, membership = Path(input_root), load_membership(membership_path)
    state_groups, states = groups(membership), load_states(input_root)
    ids = membership.case_id.tolist()
    if not set(ids).issubset(states):
        raise ValueError("State membership contains patients absent from patient_states")

    rna_genes = load_table(input_root / "rna/case_features.csv").reindex(ids)
    gmt = tool_parameters(str(ROOT / "configs"), "rna")["pathway_gene_sets_path"]
    gene_sets, _ = read_gmt_gene_sets(gmt)
    gene_sets = {name: [gene for gene in genes if gene in rna_genes.columns] for name, genes in gene_sets.items()}
    gene_sets = {name: genes for name, genes in gene_sets.items() if len(genes) >= 15}
    rna = ssgsea_scores(rna_genes, gene_sets, 15).reindex(ids)
    wxs = load_table(input_root / "wxs/wxs_discovery_features.csv").reindex(ids)
    wxs_features = [x for x in wxs.columns if x.startswith("mutation::")]
    clinical = clinical_table({case_id: states[case_id] for case_id in ids})

    radiomics = {}
    for case_id in ids:
        path = input_root / "ct_radiomics" / case_id / "radiomics_features.json"
        if path.is_file(): radiomics[case_id] = json.loads(path.read_text(encoding="utf-8"))
    if radiomics:
        ct = pd.DataFrame.from_dict(radiomics, orient="index").reindex(ids)
        ct = ct.loc[:, ct.notna().all()]
        ct_features = list(ct.columns)
        ct_table = continuous_table(ct, ids)
        ct_rows = stats.continuous_omnibus(ct_table, ct_features, state_groups)
        ct_selected = [r["feature"] for r in ct_rows if r["q_value"] is not None and r["q_value"] < .05]
        pd.DataFrame(ct_rows).to_csv(output_root / "ct_radiomics_omnibus.csv", index=False)
        pd.DataFrame(stats.continuous_posthoc(ct_table, ct_selected, state_groups, bootstrap_iterations)).to_csv(output_root / "ct_radiomics_posthoc.csv", index=False)
    else:
        raise FileNotFoundError("No CT radiomics feature cache found for frozen state patients")

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
    write_manifest(output_root / "manifest.json", {"experiment": "four_view_state_characterization", "active_modalities": ["ct", "wsi", "rna", "wxs"], "membership_file": str(Path(membership_path).resolve()), "patient_count": len(ids), "state_sizes": {k: len(v) for k, v in state_groups.items()}, "cnv_included": False, "affinity_audit": audit})
    return {"output_root": str(output_root), "patient_count": len(ids), "state_sizes": {k: len(v) for k, v in state_groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--membership-path", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/14_four_view_state_characterization")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
