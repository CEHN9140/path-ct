import glob
import json
import os
import sys

import pandas as pd

SAVE_DIR = "/data/qijun/path-ct/data"
TCGA_ROOT_DIR = "/data/qijun/data/TCGA/KIRC"
OUTPUT_JSON = os.path.join(SAVE_DIR, "tcga_kirc_data.json")
OUTPUT_JSON_NEPHRO = os.path.join(SAVE_DIR, "tcga_kirc_data_nephrographic.json")

GDC_MULTI_COLUMNS = [
    "File ID",
    "File Name",
    "Case ID",
    "Sample ID",
    "Tissue Type",
    "Tumor Descriptor",
    "Specimen Type",
    "Preservation Method",
]
CT_KEEP_COLUMNS = [
    "File Path",
    "Series UID",
    "Study UID",
    "Study Description",
    "Study Date",
    "Series Description",
    "Manufacturer",
    "Number of Images",
    "Zenodo Contrast Phase",
    "Zenodo Phase Reviewer",
]

QA_RESULTS_PATH = "/data/qijun/data/TCGA/KIRC/CT/segmentation/kidney-ct/qa-results.csv"


def process_gdc_files(root, label, *, unique_case=False):
    df = pd.read_csv(os.path.join(root, "gdc_sample_sheet.tsv"), sep="\t")
    multi_columns = [col for col in GDC_MULTI_COLUMNS if col in df.columns]

    # GDC sample sheets may store paired values in one cell, such as
    # "Tumor, Normal". Split these columns first, then expand them row-wise.
    for col in multi_columns:
        df[col] = df[col].fillna("").astype(str).str.split(",")
        df[col] = df[col].apply(lambda values: [value.strip() for value in values])

    rows = []
    for _, row in df.iterrows():
        max_length = max([len(row[col]) for col in multi_columns] or [1])
        for index in range(max_length):
            item = row.to_dict()
            for col in multi_columns:
                values = row[col]
                item[col] = values[index] if index < len(values) else values[0]
            rows.append(item)

    df = pd.DataFrame(rows).replace("", pd.NA)
    df = df.dropna(subset=["File ID", "File Name", "Case ID", "Sample ID"])
    df = df[(df["Tissue Type"] == "Tumor") & (df["Tumor Descriptor"] == "Primary")]

    # Keep only rows whose referenced local file actually exists.
    df["File Path"] = df.apply(
        lambda row: os.path.join(root, str(row["File ID"]), str(row["File Name"])),
        axis=1,
    )
    df = df[df["File Path"].apply(os.path.exists)]

    if unique_case:
        df = df.sort_values(["Case ID", "Sample ID", "File Name"])
        df = df.drop_duplicates("Case ID", keep="first")

    print(f"Found {len(df)} valid {label} files.")
    return df.reset_index(drop=True)


def process_ct(ct_root):
    df = pd.read_csv(os.path.join(ct_root, "metadata.csv")).dropna(
        subset=[
            "Subject ID",
            "Study Description",
            "Study Date",
            "Manufacturer",
            "Number of Images",
            "File Location",
        ]
    )
    df["Series Description"] = df["Series Description"].fillna("")

    # Drop the cohort prefix and rebuild the path under the selected CT root.
    file_locations = df["File Location"].astype(str).str.replace("\\", "/", regex=False)
    file_locations = file_locations.str.replace(r"^\./?", "", regex=True)
    file_locations = file_locations.str.replace(r"^TCGA-[^/]+/", "", regex=True)
    df["File Path"] = file_locations.apply(lambda path: os.path.join(ct_root, path))
    df = df[df["File Path"].apply(os.path.exists)]

    df["Study Date Parsed"] = pd.to_datetime(
        df["Study Date"], format="%m-%d-%Y", errors="raise"
    )
    earliest_dates = df.groupby("Subject ID")["Study Date Parsed"].transform("min")
    df = df[df["Study Date Parsed"] == earliest_dates].drop(
        columns="Study Date Parsed"
    )

    ct_df = df.reset_index(drop=True)
    print(
        f"Found {len(ct_df)} valid CT series from {ct_df['Subject ID'].nunique()} subjects."
    )
    return ct_df


def process_clinical(clinical_root):
    clinical_files = glob.glob(
        os.path.join(clinical_root, "**", "*.json"), recursive=True
    )
    with open(clinical_files[0], "r") as f:
        return json.load(f)


def merge_all_data(
    wsi_df, ct_df, rna_seq_df, wxs_maf_df, clinical_json, cnv_df=None, *, output_path=None
):
    # Use each source's native case identifier, then keep only cases present in
    # all four modalities and clinical records.
    data_by_modality = {
        "WSI": dict(tuple(wsi_df.groupby("Case ID"))) if "Case ID" in wsi_df else {},
        "CT": dict(tuple(ct_df.groupby("Subject ID"))) if "Subject ID" in ct_df else {},
        "RNA_Seq": dict(tuple(rna_seq_df.groupby("Case ID")))
        if "Case ID" in rna_seq_df
        else {},
        "WXS": dict(tuple(wxs_maf_df.groupby("Case ID")))
        if "Case ID" in wxs_maf_df
        else {},
    }
    if cnv_df is not None:
        data_by_modality["CNV"] = dict(tuple(cnv_df.groupby("Case ID")))
    clinical_by_case = {item.get("submitter_id"): item for item in clinical_json}

    merged_data = []
    for case_id in sorted(
        set.intersection(*(set(data) for data in data_by_modality.values()))
    ):
        clinical = clinical_by_case.get(case_id)
        if not clinical:
            continue

        entry = {"Case_ID": case_id}
        for modality, case_groups in data_by_modality.items():
            records = case_groups[case_id]
            if modality == "CT":
                # CT keeps the series metadata needed to identify the scan context.
                keep_cols = [col for col in CT_KEEP_COLUMNS if col in records.columns]
                entry[modality] = records[keep_cols].to_dict(orient="records")
            else:
                entry[modality] = [
                    {"File Path": file_path}
                    for file_path in records["File Path"].tolist()
                ]
        entry["Clinical"] = clinical

        merged_data.append(entry)

    out_path = output_path if output_path else OUTPUT_JSON
    with open(out_path, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Final data saved to {out_path}, with {len(merged_data)} entries")


if __name__ == "__main__":
    wsi_df = process_gdc_files(os.path.join(TCGA_ROOT_DIR, "WSI"), "WSI")
    ct_df = process_ct(os.path.join(TCGA_ROOT_DIR, "CT"))
    rna_seq_df = process_gdc_files(os.path.join(TCGA_ROOT_DIR, "RNA-seq"), "RNA-Seq")
    wxs_maf_df = process_gdc_files(os.path.join(TCGA_ROOT_DIR, "WXS"), "WXS MAF")
    cnv_df = process_gdc_files(
        os.path.join(TCGA_ROOT_DIR, "CNV"), "CNV", unique_case=True
    )
    clinical_json = process_clinical(os.path.join(TCGA_ROOT_DIR, "Clinical"))

    merge_all_data(
        wsi_df,
        ct_df,
        rna_seq_df,
        wxs_maf_df,
        clinical_json,
        cnv_df,
        output_path=OUTPUT_JSON,
    )
