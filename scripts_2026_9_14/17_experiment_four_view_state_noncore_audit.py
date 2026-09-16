#!/usr/bin/env python3
"""Audit state-versus-non-core selection bias on the 103-patient four-view cohort."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from four_view_state_common import DEFAULT_INPUT, DEFAULT_MEMBERSHIP, ROOT, load_membership, load_states, write_manifest
from tools.confound import confounder_values
from tools.post_discovery_characterization import bh_adjust
from tools.subtype_review_common import clinical_table

def run(input_root=DEFAULT_INPUT,membership=DEFAULT_MEMBERSHIP,output_root=ROOT/"output_kirc_v14/17_four_view_state_noncore_audit",force=False):
    output_root=Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force: raise FileExistsError(f"Output exists: {output_root}")
    output_root.mkdir(parents=True,exist_ok=True); frame=load_membership(membership); states=load_states(input_root)
    order=json.loads((Path(input_root)/"candidate_subtype/affinity_patient_order.json").read_text())
    core=set(frame.case_id); noncore=[str(x) for x in order if str(x) not in core]
    records=clinical_table(states); conf=confounder_values(states,str(input_root)); wxs=pd.read_csv(Path(input_root)/"wxs/wxs_discovery_features.csv").set_index("case_id")
    rows=[]
    categorical=("stage_group","t_stage","m_stage","grade","gender","race","ct_phase","ct_manufacturer","ct_scanner_model","ct_reconstruction_kernel")
    for variable in categorical:
        values={x:records.get(x,{}).get(variable) for x in core|set(noncore)} if variable in records[next(iter(records))] else {x:conf.get(x,{}).get(variable) for x in core|set(noncore)}
        levels=sorted({str(v) for v in values.values() if str(v).strip() and str(v).upper() not in {"NAN","NONE","UNKNOWN"}})
        for level in levels:
            a=sum(str(values[x])==level for x in core); b=len(core)-a; c=sum(str(values[x])==level for x in noncore); d=len(noncore)-c
            p=fisher_exact([[a,b],[c,d]])[1] if (a+b)*(c+d)>0 else None
            rows.append({"variable":variable,"level":level,"core_n":len(core),"noncore_n":len(noncore),"core_level_n":a,"noncore_level_n":c,"test":"fisher_exact","p_value":float(p) if p is not None else None})
    numeric={"age":{x:records.get(x,{}).get("age") for x in core|set(noncore)},"ct_slice_thickness":{x:conf.get(x,{}).get("ct_slice_thickness") for x in core|set(noncore)}}
    for variable in ("patch_count", "tumor_patch_count", "tumor_patch_fraction"):
        numeric[f"wsi_{variable}"] = {}
        for case_id in core | set(noncore):
            path = Path(input_root) / "wsi_tumor_seg" / case_id / "summary.json"
            summary = json.loads(path.read_text()) if path.is_file() else {}
            value = summary.get(variable)
            if variable == "tumor_patch_fraction" and value is None:
                patch_count, tumor_count = summary.get("patch_count"), summary.get("tumor_patch_count")
                value = tumor_count / patch_count if patch_count else None
            numeric[f"wsi_{variable}"][case_id] = value
    for variable,values in numeric.items():
        left=[float(values[x]) for x in core if values[x] not in (None,"") and np.isfinite(float(values[x]))]; right=[float(values[x]) for x in noncore if values[x] not in (None,"") and np.isfinite(float(values[x]))]
        rows.append({"variable":variable,"level":"__numeric__","core_n":len(left),"noncore_n":len(right),"core_median":float(np.median(left)) if left else None,"noncore_median":float(np.median(right)) if right else None,"test":"mann_whitney_u","p_value":float(mannwhitneyu(left,right).pvalue) if left and right else None})
    zero={x:int((wxs.loc[x].fillna(0)==0).all()) if x in wxs.index else None for x in core|set(noncore)}
    a=sum(zero[x]==1 for x in core); c=sum(zero[x]==1 for x in noncore); rows.append({"variable":"wxs_discovery_all_zero_proxy","level":"all_zero","core_n":len(core),"noncore_n":len(noncore),"core_level_n":a,"noncore_level_n":c,"test":"fisher_exact","p_value":float(fisher_exact([[a,len(core)-a],[c,len(noncore)-c]])[1])})
    result=pd.DataFrame(rows); result["q_value"]=bh_adjust(result.p_value.tolist())
    result.to_csv(output_root/"state_vs_noncore_audit.csv",index=False)
    pd.DataFrame([{"case_id":x,"selection":"state" if x in core else "non_core"} for x in order]).to_csv(output_root/"selection_membership.csv",index=False)
    write_manifest(output_root/"manifest.json",{"experiment":"four_view_state_noncore_audit","membership_file":str(Path(membership).resolve()),"patient_count":len(order),"state_patient_count":len(core),"non_core_patient_count":len(noncore),"active_modalities":["ct","wsi","rna","wxs"],"cnv_included":False})
    return {"output_root":str(output_root),"state_patient_count":len(core),"non_core_patient_count":len(noncore)}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,default=DEFAULT_INPUT); p.add_argument("--membership",type=Path,default=DEFAULT_MEMBERSHIP); p.add_argument("--output-root",type=Path,default=ROOT/"output_kirc_v14/17_four_view_state_noncore_audit"); p.add_argument("--force",action="store_true"); print(json.dumps(run(**vars(p.parse_args())),ensure_ascii=False,indent=2))
