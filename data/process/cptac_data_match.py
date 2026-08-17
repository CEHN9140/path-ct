import glob
import json
import os
import re

import nibabel as nib
import pandas as pd

SAVE_DIR = "/data/qijun/path-ct/data"


def process_wsi(wsi_root):
    # wsi_save_dir = os.path.join(SAVE_DIR, "WSI")
    # os.makedirs(wsi_save_dir, exist_ok=True)
    wsi_xlsx_path = os.path.join(wsi_root, "TCIA CPTAC Pathology Portal.csv")
    df = pd.read_csv(wsi_xlsx_path)

    filtered_df = df.dropna(
        subset=[
            "Case_ID",
            "Specimen_ID",
            "Slide_ID",
            "Radiology",
            "Genomics",
            "Proteomics",
        ]
    )

    filtered_df = filtered_df[filtered_df["Specimen_Type"] == "tumor_tissue"]
    valid_rows = []

    for _, row in filtered_df.iterrows():
        slide_path = os.path.join(wsi_root, f"{row['Slide_ID']}.svs")
        if os.path.exists(slide_path):
            row = row.to_dict()
            row["File Path"] = slide_path
            valid_rows.append(row)
    print(f"Found {len(valid_rows)} valid WSI files.")

    return pd.DataFrame(valid_rows)


def process_ct(ct_root):
    ct_xlsx_path = os.path.join(ct_root, "metadata.csv")
    df = pd.read_csv(ct_xlsx_path)

    filtered_df = df.dropna(subset=["Subject ID"])
    valid_rows = []

    for _, row in filtered_df.iterrows():
        ct_subject_path = os.path.join(ct_root, f"{row['Subject ID']}")
        if os.path.exists(ct_subject_path):
            row = row.to_dict()
            row["File Path"] = ct_subject_path
            valid_rows.append(row)
    print(f"Found {len(valid_rows)} valid CT subject.")
    return (
        pd.DataFrame(valid_rows)
        .drop_duplicates(subset=["File Path"])
        .reset_index(drop=True)
    )


def process_rna_seq(rna_seq_root):
    # rna_seq_save_dir = os.path.join(SAVE_DIR, "RNA-Seq")
    # os.makedirs(rna_seq_save_dir, exist_ok=True)
    rna_xlsx_path = os.path.join(rna_seq_root, "gdc_sample_sheet.tsv")
    df = pd.read_csv(rna_xlsx_path, sep="\t")

    multi_val_cols = [
        "Case ID",
        "Sample ID",
        "Tissue Type",
        "Tumor Descriptor",
        "Specimen Type",
        "Preservation Method",
    ]
    for col in multi_val_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(",")
            df[col] = df[col].apply(
                lambda x: [i.strip() for i in x] if isinstance(x, list) else x
            )

    def align_lengths(row):
        lengths = [len(row[col]) for col in multi_val_cols if col in row]
        max_l = max(lengths) if lengths else 1
        for col in multi_val_cols:
            if col in row:
                if len(row[col]) < max_l:
                    row[col] = row[col] * max_l
        return row

    df = df.apply(align_lengths, axis=1)
    df = df.explode(multi_val_cols)
    df = df.replace("nan", pd.NA)

    filtered_df = df.dropna(subset=["File ID", "File Name"])
    filtered_df = filtered_df[filtered_df["Tissue Type"] == "Tumor"]
    filtered_df = filtered_df[filtered_df["Tumor Descriptor"] == "Primary"]
    filtered_df = filtered_df[filtered_df["Specimen Type"] == "Solid Tissue"]

    valid_rows = []
    for _, row in filtered_df.iterrows():
        file_path = os.path.join(
            rna_seq_root, str(row["File ID"]), str(row["File Name"])
        )
        if os.path.exists(file_path):
            row_dict = row.to_dict()
            row_dict["File Path"] = file_path
            valid_rows.append(row_dict)

    print(f"Found {len(valid_rows)} valid RNA-Seq files")
    return pd.DataFrame(valid_rows)


def process_wxs_maf(wxs_root):
    # wxs_save_dir = os.path.join(SAVE_DIR, "WXS")
    # os.makedirs(wxs_save_dir, exist_ok=True)

    wxs_xlsx_path = os.path.join(wxs_root, "gdc_sample_sheet.tsv")
    df = pd.read_csv(wxs_xlsx_path, sep="\t")

    multi_val_cols = [
        "Case ID",
        "Sample ID",
        "Tissue Type",
        "Tumor Descriptor",
        "Specimen Type",
        "Preservation Method",
    ]
    for col in multi_val_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(",")
            df[col] = df[col].apply(
                lambda x: [i.strip() for i in x] if isinstance(x, list) else x
            )

    def align_lengths(row):
        lengths = [len(row[col]) for col in multi_val_cols if col in row]
        max_l = max(lengths) if lengths else 1
        for col in multi_val_cols:
            if col in row:
                if len(row[col]) < max_l:
                    row[col] = row[col] * max_l
        return row

    df = df.apply(align_lengths, axis=1)
    df = df.explode(multi_val_cols)
    df = df.replace("nan", pd.NA)
    filtered_df = df.dropna(subset=["File ID", "File Name"])
    filtered_df = filtered_df[filtered_df["Tissue Type"] == "Tumor"]
    filtered_df = filtered_df[filtered_df["Tumor Descriptor"] == "Primary"]
    filtered_df = filtered_df[filtered_df["Specimen Type"] == "Solid Tissue"]

    valid_rows = []

    for _, row in filtered_df.iterrows():
        file_path = os.path.join(wxs_root, row["File ID"], row["File Name"])
        if os.path.exists(file_path):
            row = row.to_dict()
            row["File Path"] = file_path
            valid_rows.append(row)
    print(f"Found {len(valid_rows)} valid WXS MAF files.")

    return pd.DataFrame(valid_rows)


### 待解决
def process_protein(protein_root):
    protein_save_dir = os.path.join(SAVE_DIR, "Protein")
    os.makedirs(protein_save_dir, exist_ok=True)
    pro_dir_list = os.listdir(protein_root)
    proteome_rows = []
    phosphoproteome_rows = []
    for dir in pro_dir_list:
        pro_ass_dir = os.path.join(protein_root, dir, "1", "Protein Assembly")
        if not os.path.exists(pro_ass_dir):
            continue
        pro_dir = os.path.join(pro_ass_dir, os.listdir(pro_ass_dir)[0], "Text")
        if dir.lower() == "pdc000153":  # 全蛋白组
            tmt10_tsv = os.path.join(
                pro_dir, "CPTAC3_Lung_Adeno_Carcinoma_Proteome.tmt10.tsv"
            )
            sample_txt = os.path.join(
                pro_dir, "CPTAC3_Lung_Adeno_Carcinoma_Proteome_sample.txt"
            )
            summary_txt = os.path.join(
                pro_dir, "CPTAC3_Lung_Adeno_Carcinoma_Proteome_summary.tsv"
            )
    return pd.DataFrame(proteome_rows)


def process_clinical(clinical_root):
    # clinical_save_dir = os.path.join(SAVE_DIR, "Clinical")
    # os.makedirs(clinical_save_dir, exist_ok=True)
    clinical_file = glob.glob(os.path.join(clinical_root, "**.json"), recursive=True)[0]
    if os.path.exists(clinical_file):
        with open(clinical_file, "r") as f:
            clinical_data = json.load(f)
    return clinical_data


def merge_all_data(wsi_df, ct_df, rna_seq_df, wxs_maf_df, clinical_json):
    """
    Merge all data sources into a single structure and save as data.json.
    """
    # Standardize column names
    ct_df = ct_df.rename(columns={"Subject ID": "Case_ID"})
    rna_seq_df = rna_seq_df.rename(columns={"Case ID": "Case_ID"})
    wxs_maf_df = wxs_maf_df.rename(columns={"Case ID": "Case_ID"})

    # Drop unnecessary columns
    wsi_df = wsi_df.drop(
        columns=[
            "Embedding_Medium",
            "Pathology",
            "HasRadiology",
            "Radiology",
            "Genomics_Available",
            "Genomics",
            "GDC_Link",
            "Proteomics_Available",
            "Proteomics",
            "PDC_Link",
        ],
        errors="ignore",
    )

    ct_df = ct_df.drop(
        columns=[
            "Series UID",
            "Collection",
            "3rd Party Analysis",
            "Data Description URI",
            "Study UID",
            "Study Description",
            "Study Date",
            "Series Description",
            "Manufacturer",
            "Modality",
            "SOP Class Name",
            "SOP Class UID",
            "Number of Images",
            "File Size",
            "File Location",
            "Download Timestamp",
        ],
        errors="ignore",
    )

    rna_seq_df = rna_seq_df.drop(
        columns=[
            "File ID",
            "File Name",
            "Data Category",
            "Data Type",
            "Project ID",
            "Tissue Type",
            "Tumor Descriptor",
            "Specimen Type",
            "Preservation Method",
        ],
        errors="ignore",
    )
    wxs_maf_df = wxs_maf_df.drop(
        columns=[
            "File ID",
            "File Name",
            "Data Category",
            "Data Type",
            "Project ID",
            "Tissue Type",
            "Tumor Descriptor",
            "Specimen Type",
            "Preservation Method",
        ],
        errors="ignore",
    )

    # Initialize merged data
    merged_data = []

    for case_id, wsi_group in wsi_df.groupby("Case_ID"):
        wsi_rows = wsi_group.to_dict(orient="records")
        ct_rows = ct_df[ct_df["Case_ID"] == case_id]
        rna_rows = rna_seq_df[rna_seq_df["Case_ID"] == case_id]
        wxs_rows = wxs_maf_df[wxs_maf_df["Case_ID"] == case_id]

        if not wsi_rows or ct_rows.empty or (rna_rows.empty and wxs_rows.empty):
            continue

        entry = {
            "Case_ID": case_id,
            "WSI": wsi_rows,
            "CT": ct_rows.to_dict(orient="records"),
            "RNA_Seq": rna_rows.to_dict(orient="records") if not rna_rows.empty else [],
            "WXS": wxs_rows.to_dict(orient="records") if not wxs_rows.empty else [],
        }

        # Add Clinical data
        matching_clinical = next(
            (entry for entry in clinical_json if entry.get("submitter_id") == case_id),
            None,
        )
        if matching_clinical:
            entry["Clinical"] = matching_clinical

        # Remove Case_ID from nested entries
        for key in ["WSI", "CT", "RNA_Seq", "WXS"]:
            entry[key] = [
                {k: v for k, v in record.items() if k != "Case_ID"}
                for record in entry[key]
            ]

        merged_data.append(entry)

    # Save the final data.json
    final_data_path = os.path.join(SAVE_DIR, "cptac_data.json")
    with open(final_data_path, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Final data saved to {final_data_path}, with {len(merged_data)} entries.")


if __name__ == "__main__":
    WSI_ROOT_DIR = "/data/qijun/data/CPTAC/LUAD/WSI"
    CT_ROOT_DIR = "/data/qijun/data/CPTAC/LUAD/CT"
    RNA_SEQ_ROOT_DIR = "/data/qijun/data/CPTAC/LUAD/RNA-seq"
    WXS_ROOT_DIR = "/data/qijun/data/CPTAC/LUAD/WXS"
    CLINICAL_ROOT_DIR = "/data/qijun/data/CPTAC/LUAD/Clinical"

    # Process all data sources
    wsi_df = process_wsi(WSI_ROOT_DIR)
    ct_df = process_ct(CT_ROOT_DIR)
    rna_seq_df = process_rna_seq(RNA_SEQ_ROOT_DIR)
    wxs_maf_df = process_wxs_maf(WXS_ROOT_DIR)
    clinical_json = process_clinical(CLINICAL_ROOT_DIR)

    # Merge all data and save
    merge_all_data(wsi_df, ct_df, rna_seq_df, wxs_maf_df, clinical_json)
