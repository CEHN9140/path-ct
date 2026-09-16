#!/usr/bin/env python3
"""Map frozen four-view states to TCGA mRNA and ClearCode34 labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.post_discovery_characterization import bh_adjust
from four_view_state_common import DEFAULT_MEMBERSHIP, ROOT, groups, load_membership, write_manifest

def permutation_chi2(labels, values, permutations=9999, seed=20260915):
    labels, values = np.asarray(labels), np.asarray(values)
    levels, states = sorted(set(values)), sorted(set(labels))
    observed = np.asarray([[np.sum((labels == state) & (values == level)) for level in levels] for state in states])
    statistic = float(chi2_contingency(observed, correction=False)[0])
    rng = np.random.default_rng(seed); exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(values)
        table = np.asarray([[np.sum((labels == state) & (shuffled == level)) for level in levels] for state in states])
        if chi2_contingency(table, correction=False)[0] >= statistic: exceed += 1
    return statistic, (exceed + 1) / (permutations + 1), observed

def run(membership=DEFAULT_MEMBERSHIP, output_root=ROOT/"output_kirc_v14/15_four_view_state_known_ccrcc_mapping", permutations=9999, force=False):
    output_root=Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force: raise FileExistsError(f"Output exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True); frame=load_membership(membership); gs=groups(frame); ids=frame.case_id.tolist()
    refs={"mrna_m1_m4":ROOT/"data/tcga_kirc_mrna_m1_m4.csv", "clearcode34":ROOT/"data/tcga_kirc_clearcode34.csv"}
    rows=[]
    for name,path in refs.items():
        ref=pd.read_csv(path,dtype={"case_id":str})[["case_id","reference_subtype","reference_status"]]
        merged=frame.merge(ref,on="case_id",how="left"); merged.to_csv(output_root/f"{name}_patient_mapping.csv",index=False)
        for state,members in gs.items():
            current=merged[merged.state_id.eq(state)]; known=current[current.reference_status.eq("matched")]
            counts=known.reference_subtype.value_counts().to_dict()
            rows.append({"reference":name,"state_id":state,"state_n":len(members),"known_n":len(known),"unknown_n":len(current)-len(known),"label_counts":json.dumps({str(k):int(v) for k,v in counts.items()})})
        table=pd.crosstab(merged.state_id, merged.reference_subtype).reindex(index=sorted(gs),fill_value=0); table.to_csv(output_root/f"{name}_state_by_label.csv")
        known=merged[merged.reference_status.eq("matched")].copy()
        if not known.empty and known.reference_subtype.nunique()>1:
            statistic,p_value,_=permutation_chi2(known.state_id,known.reference_subtype,permutations,20260915)
            rows.append({"reference":name,"analysis":"state_label_association","known_n":len(known),"chi_square":statistic,"p_value":p_value,"cramers_v":float(np.sqrt(statistic/(len(known)*min(len(gs)-1,known.reference_subtype.nunique()-1))))})
            y_true=known.state_id.to_numpy(); y_pred=known.reference_subtype.to_numpy(); rows.append({"reference":name,"analysis":"label_information","known_n":len(known),"ari":float(adjusted_rand_score(y_true,y_pred)),"nmi":float(normalized_mutual_info_score(y_true,y_pred))})
        missing_labels = merged.reference_status.ne("matched").map({True:"missing",False:"known"})
        if missing_labels.nunique() > 1:
            statistic,p_value,_=permutation_chi2(merged.state_id, missing_labels, permutations, 20260916)
            rows.append({"reference":name,"analysis":"coverage_association","known_n":int((missing_labels=="known").sum()),"missing_n":int((missing_labels=="missing").sum()),"chi_square":statistic,"p_value":p_value})
    result=pd.DataFrame(rows)
    if "p_value" in result: result["q_value"]=bh_adjust(result.p_value.tolist())
    result.to_csv(output_root/"mapping_coverage.csv",index=False)
    write_manifest(output_root/"manifest.json",{"experiment":"four_view_state_known_ccrcc_mapping","membership_file":str(Path(membership).resolve()),"references":{k:str(v.resolve()) for k,v in refs.items()},"patient_count":len(ids),"state_sizes":{k:len(v) for k,v in gs.items()},"permutations":permutations})
    return {"output_root":str(output_root),"patient_count":len(ids),"state_sizes":{k:len(v) for k,v in gs.items()}}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--membership",type=Path,default=DEFAULT_MEMBERSHIP); p.add_argument("--output-root",type=Path,default=ROOT/"output_kirc_v14/15_four_view_state_known_ccrcc_mapping"); p.add_argument("--force",action="store_true"); print(json.dumps(run(**vars(p.parse_args())),ensure_ascii=False,indent=2))
