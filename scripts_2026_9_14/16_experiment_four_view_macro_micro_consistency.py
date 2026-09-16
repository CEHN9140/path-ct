#!/usr/bin/env python3
"""Quantify agreement between frozen macro-states and their micro-cores."""
from __future__ import annotations
import argparse, json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from four_view_state_common import DEFAULT_INPUT, DEFAULT_MEMBERSHIP, ROOT, groups, load_membership, load_states, write_manifest
from tools.subtype_review_common import clinical_table
from tools.multimodal_consistency_check import normalized_affinity_with_audit

def run(input_root=DEFAULT_INPUT,membership=DEFAULT_MEMBERSHIP,output_root=ROOT/"output_kirc_v14/16_four_view_macro_micro_consistency",force=False):
    output_root=Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force: raise FileExistsError(f"Output exists: {output_root}")
    output_root.mkdir(parents=True,exist_ok=True); frame=load_membership(membership); gs=groups(frame); states=load_states(input_root)
    rows=[]
    for state,members in gs.items():
        sub=frame[frame.state_id.eq(state)]; rows.append({"state_id":state,"state_n":len(members),"micro_core_count":sub.core_id.nunique(),"micro_core_sizes":json.dumps({k:int(v) for k,v in sub.core_id.value_counts().sort_index().items()})})
    pd.DataFrame(rows).to_csv(output_root/"state_micro_core_composition.csv",index=False)
    paths={"ct":Path(input_root)/"candidate_subtype/ct_affinity.npy","wsi":Path(input_root)/"candidate_subtype/wsi_affinity.npy","rna":Path(input_root)/"candidate_subtype/rna_affinity.npy","wxs":Path(input_root)/"wxs/wxs_affinity.npy"}
    order=json.loads((Path(input_root)/"candidate_subtype/affinity_patient_order.json").read_text())
    index={x:i for i,x in enumerate(order)}
    pair=[]
    for modality,path in paths.items():
        matrix,_=normalized_affinity_with_audit(np.load(path))
        core_ids=sorted(frame.core_id.unique())
        for a,b in combinations(core_ids,2):
            left=[index[x] for x in frame.loc[frame.core_id.eq(a),"case_id"]]; right=[index[x] for x in frame.loc[frame.core_id.eq(b),"case_id"]]
            state_a=frame.loc[frame.core_id.eq(a),"state_id"].iloc[0]; state_b=frame.loc[frame.core_id.eq(b),"state_id"].iloc[0]
            pair.append({"modality":modality,"state_id":state_a if state_a==state_b else "between_states","state_a":state_a,"state_b":state_b,"core_a":a,"core_b":b,"n_a":len(left),"n_b":len(right),"mean_similarity":float(matrix[np.ix_(left,right)].mean())})
    pd.DataFrame(pair).to_csv(output_root/"within_state_micro_core_similarity.csv",index=False)
    records=clinical_table({x:states[x] for x in order if x in states}); clinical_rows=[]
    for state,members in gs.items():
        for core in sorted(frame.loc[frame.state_id.eq(state),"core_id"].unique()):
            ids=frame.loc[frame.core_id.eq(core),"case_id"].tolist()
            clinical_rows.append({"state_id":state,"core_id":core,"core_n":len(ids),"age_median":float(np.nanmedian([records[x]["age"] for x in ids if records[x].get("age") is not None])) if any(records[x].get("age") is not None for x in ids) else None,"stage_counts":json.dumps(pd.Series([records[x].get("stage_group","") for x in ids]).value_counts().to_dict())})
    pd.DataFrame(clinical_rows).to_csv(output_root/"within_state_micro_core_clinical.csv",index=False)
    write_manifest(output_root/"manifest.json",{"experiment":"four_view_macro_micro_consistency","active_modalities":["ct","wsi","rna","wxs"],"membership_file":str(Path(membership).resolve()),"patient_count":len(order),"cnv_included":False})
    return {"output_root":str(output_root),"state_sizes":{k:len(v) for k,v in gs.items()}}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,default=DEFAULT_INPUT); p.add_argument("--membership",type=Path,default=DEFAULT_MEMBERSHIP); p.add_argument("--output-root",type=Path,default=ROOT/"output_kirc_v14/16_four_view_macro_micro_consistency"); p.add_argument("--force",action="store_true"); print(json.dumps(run(**vars(p.parse_args())),ensure_ascii=False,indent=2))
