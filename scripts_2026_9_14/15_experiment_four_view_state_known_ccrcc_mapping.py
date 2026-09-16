#!/usr/bin/env python3
"""Map frozen four-view states to TCGA mRNA and ClearCode34 labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact
from four_view_state_common import DEFAULT_MEMBERSHIP, ROOT, groups, load_membership, write_manifest

def run(membership=DEFAULT_MEMBERSHIP, output_root=ROOT/"output_kirc_v14/15_four_view_state_known_ccrcc_mapping", force=False):
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
    pd.DataFrame(rows).to_csv(output_root/"mapping_coverage.csv",index=False)
    write_manifest(output_root/"manifest.json",{"experiment":"four_view_state_known_ccrcc_mapping","membership_file":str(Path(membership).resolve()),"references":{k:str(v.resolve()) for k,v in refs.items()},"patient_count":len(ids),"state_sizes":{k:len(v) for k,v in gs.items()}})
    return {"output_root":str(output_root),"patient_count":len(ids),"state_sizes":{k:len(v) for k,v in gs.items()}}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--membership",type=Path,default=DEFAULT_MEMBERSHIP); p.add_argument("--output-root",type=Path,default=ROOT/"output_kirc_v14/15_four_view_state_known_ccrcc_mapping"); p.add_argument("--force",action="store_true"); print(json.dumps(run(**vars(p.parse_args())),ensure_ascii=False,indent=2))
