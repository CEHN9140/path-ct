from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.io import ensure_dir, write_json
from utils.llm_utils import load_yaml_file
from utils.omics_utils import (
    build_cohort_signature,
    collect_case_file_paths,
    load_manifest_if_valid,
)
from utils.tool_utils import make_tool_result


def load_rna_config(config_dir: str = "") -> dict[str, Any]:
    path = Path(config_dir).expanduser() / "rna.yaml" if config_dir else Path("")
    if path.exists():
        return load_yaml_file(path)
    return {}


def rna_top_gene_count(config_dir: str = "") -> int:
    return max(int(load_rna_config(config_dir).get("top_gene_count", 2000) or 2000), 0)


def read_rna_expression_series(file_path: str) -> pd.Series:
    rna_df = pd.read_csv(file_path, sep="\t", header=None, names=range(9))
    rna_df.columns = rna_df.iloc[1]
    rna_df = rna_df.iloc[6:].reset_index(drop=True)
    if "gene_name" not in rna_df.columns or "tpm_unstranded" not in rna_df.columns:
        raise ValueError(f"{file_path} lack 'gene_name' or 'tpm_unstranded' column")
    rna_df = rna_df.loc[:, ["gene_name", "tpm_unstranded"]].copy()
    rna_df["gene_name"] = rna_df["gene_name"].astype(str).str.strip()
    rna_df = rna_df[rna_df["gene_name"] != ""].copy()
    rna_df["tpm_unstranded"] = pd.to_numeric(
        rna_df["tpm_unstranded"], errors="coerce"
    ).fillna(0.0)
    series = rna_df.groupby("gene_name", sort=False)["tpm_unstranded"].max()
    series = np.log2(series + 1.0)
    series.name = Path(file_path).name
    return series


def build_rna_cohort_cache(
    cohort_cases: Sequence[Mapping[str, Any]],
    *,
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    case_file_rows = collect_case_file_paths(cohort_cases, "RNA_Seq")
    output_dir = ensure_dir(Path(output_root) / "rna")
    manifest_path = output_dir / "manifest.json"
    case_features_path = output_dir / "case_features.csv"
    pathway_features_path = output_dir / "case_pathway_features.csv"
    gene_mad_path = output_dir / "gene_mad.csv"
    top_genes_path = output_dir / "top_genes.csv"
    selected_gene_limit = rna_top_gene_count(config_dir)
    signature = build_cohort_signature(
        case_file_rows,
        extra={
            "top_gene_count": selected_gene_limit,
            "modality": "RNA_Seq",
            "feature_mode": "top_genes_only",
        },
    )
    manifest = load_manifest_if_valid(
        manifest_path,
        signature=signature,
        required_paths=[
            case_features_path,
            pathway_features_path,
            gene_mad_path,
            top_genes_path,
        ],
    )
    if manifest is None:
        case_series_map: dict[str, pd.Series] = {}
        for case_id, file_path in case_file_rows:
            case_series_map[case_id] = read_rna_expression_series(file_path)

        expression_df = pd.DataFrame(case_series_map).fillna(0.0)
        gene_mad_df = pd.DataFrame(columns=["gene_name", "mad"])
        top_genes_df = pd.DataFrame(columns=["gene_name", "mad"])
        case_features_df = pd.DataFrame(columns=["case_id"])
        pathway_features_df = pd.DataFrame(columns=["case_id"])
        if not expression_df.empty:
            pathway_features_df = (
                expression_df.T.reset_index().rename(columns={"index": "case_id"})
            )
            gene_medians = expression_df.median(axis=1)
            gene_mads = expression_df.sub(gene_medians, axis=0).abs().median(axis=1)
            gene_mad_df = (
                pd.DataFrame(
                    {
                        "gene_name": expression_df.index.astype(str),
                        "mad": gene_mads.astype(float).values,
                    }
                )
                .sort_values(["mad", "gene_name"], ascending=[False, True])
                .reset_index(drop=True)
            )
            top_genes_df = gene_mad_df.head(selected_gene_limit).copy()
            selected_genes = top_genes_df["gene_name"].tolist()
            case_features_df = (
                expression_df.reindex(selected_genes)
                .T.reset_index()
                .rename(columns={"index": "case_id"})
            )

        gene_mad_df.to_csv(gene_mad_path, index=False)
        top_genes_df.to_csv(top_genes_path, index=False)
        case_features_df.to_csv(case_features_path, index=False)
        pathway_features_df.to_csv(pathway_features_path, index=False)
        manifest = {
            "signature": signature,
            "modality": "RNA_Seq",
            "cohort_case_count": len(case_file_rows),
            "selected_gene_count": len(top_genes_df),
            "top_gene_count": selected_gene_limit,
            "input_cases": [
                {"case_id": case_id, "file_path": file_path}
                for case_id, file_path in case_file_rows
            ],
            "files": {
                "case_features": str(case_features_path),
                "pathway_features": str(pathway_features_path),
                "gene_mad": str(gene_mad_path),
                "top_genes": str(top_genes_path),
            },
        }
        write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "case_features_path": str(case_features_path),
        "pathway_features_path": str(pathway_features_path),
        "gene_mad_path": str(gene_mad_path),
        "top_genes_path": str(top_genes_path),
        "signature": signature,
    }


def run_case_rna_features(
    *,
    case_id: str,
    cohort_cases: Sequence[Mapping[str, Any]],
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    case_id = str(case_id or "unknown_case")
    cache = build_rna_cohort_cache(
        cohort_cases, output_root=output_root, config_dir=config_dir
    )
    case_features_df = pd.read_csv(cache["case_features_path"])
    top_genes_df = pd.read_csv(cache["top_genes_path"])
    top_genes = top_genes_df["gene_name"].astype(str).tolist()
    case_file_map = dict(collect_case_file_paths(cohort_cases, "RNA_Seq"))
    selected_gene_count = len(top_genes)
    feature_values: list[float] = []
    status = "failure"
    errors = ["RNA features are unavailable for this case."]

    feature_row = case_features_df.loc[case_features_df["case_id"] == case_id]
    if not feature_row.empty:
        feature_values = [
            round(float(value), 6)
            for value in feature_row.iloc[0].drop(labels=["case_id"]).tolist()
        ]
        status = "success"
        errors = []

    payload = {
        "case_id": case_id,
        "source_path": case_file_map.get(case_id, ""),
        "selected_gene_count": selected_gene_count,
        "selected_genes": top_genes,
        "feature_values": feature_values,
    }
    tool_result = make_tool_result(
        output_root=output_root,
        tool_name="rna",
        status=status,
        identifier=case_id,
        metrics={
            "selected_gene_count": selected_gene_count,
            "feature_value_count": len(feature_values),
        },
        artifacts={
            "case_features_path": cache["case_features_path"],
            "pathway_features_path": cache["pathway_features_path"],
            "top_genes_path": cache["top_genes_path"],
            "manifest_path": cache["manifest_path"],
        },
        provenance={
            "backend": "raw_rna_seq",
            "case_id": case_id,
            "signature": cache["signature"],
        },
        errors=errors,
        payload=payload,
    )
    return {"tool_result": tool_result, "payload": payload}
