from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from utils.io import ensure_dir, write_json
from utils.omics_utils import (
    build_cohort_signature,
    collect_case_file_paths,
    load_manifest_if_valid,
)
from utils.tool_utils import make_tool_result


def read_wxs_case_payload(
    file_path: str, nonsynonymous_variants: Sequence[str]
) -> dict[str, Any]:
    mutated_genes: set[str] = set()
    mutation_rows: list[dict[str, Any]] = []
    mutation_columns: list[str] = []
    nonsyn_count = 0
    indel_count = 0
    nonsynonymous_variant_set = {
        str(item).strip() for item in nonsynonymous_variants if str(item).strip()
    }
    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as handle:
        header: list[str] | None = None
        column_index: dict[str, int] = {}
        dedupe_rows: set[tuple[str, ...]] = set()
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                mutation_columns = list(header)
                for name in [
                    "Hugo_Symbol",
                    "Variant_Classification",
                ]:
                    if name not in header:
                        raise ValueError(f"{file_path} lack column {name}")
                    column_index[name] = header.index(name)
                continue
            gene_name = fields[column_index["Hugo_Symbol"]].strip()
            variant_classification = fields[
                column_index["Variant_Classification"]
            ].strip()
            if not gene_name or variant_classification not in nonsynonymous_variant_set:
                continue
            mutation_row = {
                column_name: str(fields[index]).strip() if index < len(fields) else ""
                for index, column_name in enumerate(header)
            }
            mutation_key = tuple(
                mutation_row.get(column_name, "") for column_name in header
            )
            if mutation_key in dedupe_rows:
                continue
            dedupe_rows.add(mutation_key)
            mutated_genes.add(gene_name)
            nonsyn_count += 1
            if variant_classification in {
                "Frame_Shift_Del",
                "Frame_Shift_Ins",
                "In_Frame_Del",
                "In_Frame_Ins",
            }:
                indel_count += 1
            mutation_rows.append(mutation_row)
    return {
        "mutated_genes": mutated_genes,
        "mutation_rows": mutation_rows,
        "mutation_columns": mutation_columns,
        "nonsyn_count": nonsyn_count,
        "indel_count": indel_count,
        "snv_count": nonsyn_count - indel_count,
    }


def build_wxs_cohort_cache(
    cohort_cases: Sequence[Mapping[str, Any]],
    *,
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    import yaml

    tool_config = (
        yaml.safe_load(
            (Path(config_dir).expanduser() / "wxs.yaml").read_text(encoding="utf-8")
        )
        if config_dir
        else {}
    ) or {}
    nonsynonymous_variants = list(tool_config.get("nonsynonymous_variants"))
    selected_gene_limit = max(int(tool_config.get("top_gene_count", 2000) or 2000), 0)
    case_file_rows = collect_case_file_paths(cohort_cases, "WXS")
    output_dir = ensure_dir(Path(output_root) / "wxs")
    manifest_path = output_dir / "manifest.json"
    case_features_path = output_dir / "case_gene_binary.csv"
    all_features_path = output_dir / "case_all_gene_binary.csv"
    gene_frequency_path = output_dir / "gene_frequency.csv"
    case_summary_path = output_dir / "case_summary.csv"
    filtered_mutations_path = output_dir / "filtered_mutations.csv"
    capture_size_value = float(tool_config.get("capture_size"))
    signature = build_cohort_signature(
        case_file_rows,
        extra={
            "nonsynonymous_variants": nonsynonymous_variants,
            "filtered_mutation_columns": "all",
            "feature_mode": "top_frequency_gene_binary",
            "top_gene_count": selected_gene_limit,
            "capture_size": capture_size_value,
            "modality": "WXS",
        },
    )
    manifest = load_manifest_if_valid(
        manifest_path,
        signature=signature,
        required_paths=[
            case_features_path,
            all_features_path,
            gene_frequency_path,
            case_summary_path,
            filtered_mutations_path,
        ],
    )
    if manifest is None:
        case_payloads: dict[str, dict[str, Any]] = {}
        filtered_mutation_columns: list[str] = []
        filtered_mutation_rows: list[dict[str, Any]] = []
        for case_id, file_path in case_file_rows:
            payload = read_wxs_case_payload(file_path, nonsynonymous_variants)
            case_payloads[case_id] = payload
            for column_name in payload["mutation_columns"]:
                if column_name not in filtered_mutation_columns:
                    filtered_mutation_columns.append(column_name)
            filtered_mutation_rows.extend(
                [{"case_id": case_id, **row} for row in payload["mutation_rows"]]
            )

        sorted_genes = sorted(
            {
                gene_name
                for payload in case_payloads.values()
                for gene_name in payload["mutated_genes"]
            }
        )
        all_feature_rows: list[dict[str, Any]] = []
        for case_id, payload in case_payloads.items():
            row = {"case_id": case_id}
            mutated_genes = payload["mutated_genes"]
            for gene_name in sorted_genes:
                row[gene_name] = 1 if gene_name in mutated_genes else 0
            all_feature_rows.append(row)
        all_features_df = pd.DataFrame(all_feature_rows)
        if all_features_df.empty:
            all_features_df = pd.DataFrame(columns=["case_id"])
            case_features_df = pd.DataFrame(columns=["case_id"])
            gene_frequency_df = pd.DataFrame(columns=["gene_name", "num_cases", "freq"])
            selected_genes: list[str] = []
        else:
            numeric_df = all_features_df.set_index("case_id")
            gene_frequency_df = (
                pd.DataFrame(
                    {
                        "gene_name": numeric_df.columns.astype(str),
                        "num_cases": numeric_df.sum(axis=0).astype(int).values,
                        "freq": numeric_df.mean(axis=0).astype(float).values,
                    }
                )
                .sort_values(["num_cases", "gene_name"], ascending=[False, True])
                .reset_index(drop=True)
            )
            selected_genes = gene_frequency_df["gene_name"].head(selected_gene_limit).astype(str).tolist()
            all_features_df = numeric_df.reindex(
                columns=sorted_genes, fill_value=0
            ).reset_index()
            case_features_df = numeric_df.reindex(
                columns=selected_genes, fill_value=0
            ).reset_index()

        summary_rows: list[dict[str, Any]] = []
        for case_id, payload in case_payloads.items():
            summary_rows.append(
                {
                    "case_id": case_id,
                    "nonsyn_count": int(payload["nonsyn_count"]),
                    "INDEL_count": int(payload["indel_count"]),
                    "SNV_count": int(payload["snv_count"]),
                    "TMB": round(
                        float(payload["nonsyn_count"]) / capture_size_value, 6
                    ),
                }
            )
        case_summary_df = pd.DataFrame(summary_rows)

        case_features_df.to_csv(case_features_path, index=False)
        all_features_df.to_csv(all_features_path, index=False)
        gene_frequency_df.to_csv(gene_frequency_path, index=False)
        case_summary_df.to_csv(case_summary_path, index=False)
        filtered_mutations_df = pd.DataFrame(filtered_mutation_rows)
        if filtered_mutations_df.empty:
            filtered_mutations_df = pd.DataFrame(
                columns=["case_id", *filtered_mutation_columns]
            )
        else:
            filtered_mutations_df = filtered_mutations_df.reindex(
                columns=[
                    "case_id",
                    *filtered_mutation_columns,
                    *[
                        column_name
                        for column_name in filtered_mutations_df.columns
                        if column_name not in {"case_id", *filtered_mutation_columns}
                    ],
                ]
            )
        filtered_mutations_df.to_csv(filtered_mutations_path, index=False)
        manifest = {
            "signature": signature,
            "modality": "WXS",
            "cohort_case_count": len(case_file_rows),
            "nonsynonymous_variants": nonsynonymous_variants,
            "filtered_mutation_columns": "all",
            "selected_gene_count": len(selected_genes),
            "all_gene_count": len(gene_frequency_df),
            "top_gene_count": selected_gene_limit,
            "feature_mode": "top_frequency_gene_binary",
            "capture_size": capture_size_value,
            "input_cases": [
                {"case_id": case_id, "file_path": file_path}
                for case_id, file_path in case_file_rows
            ],
            "files": {
                "case_features": str(case_features_path),
                "all_features": str(all_features_path),
                "gene_frequency": str(gene_frequency_path),
                "case_summary": str(case_summary_path),
                "filtered_mutations": str(filtered_mutations_path),
            },
        }
        write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "case_features_path": str(case_features_path),
        "all_features_path": str(all_features_path),
        "gene_frequency_path": str(gene_frequency_path),
        "case_summary_path": str(case_summary_path),
        "filtered_mutations_path": str(filtered_mutations_path),
        "signature": signature,
    }


def run_case_wxs_features(
    *,
    case_id: str,
    cohort_cases: Sequence[Mapping[str, Any]],
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    case_id = str(case_id or "unknown_case")
    cache = build_wxs_cohort_cache(
        cohort_cases, output_root=output_root, config_dir=config_dir
    )
    case_features_df = pd.read_csv(cache["case_features_path"])
    case_summary_df = pd.read_csv(cache["case_summary_path"])
    selected_genes = case_features_df.columns.drop("case_id").astype(str).tolist()
    case_file_map = dict(collect_case_file_paths(cohort_cases, "WXS"))
    selected_gene_count = len(selected_genes)
    feature_values: list[int] = []
    mutated_genes: list[str] = []
    nonsyn_count = 0
    indel_count = 0
    snv_count = 0
    tmb = 0.0
    status = "failure"
    errors = ["WXS features are unavailable for this case."]
    artifacts = {
        "case_features_path": cache["case_features_path"],
        "all_features_path": cache["all_features_path"],
        "gene_frequency_path": cache["gene_frequency_path"],
        "case_summary_path": cache["case_summary_path"],
        "filtered_mutations_path": cache["filtered_mutations_path"],
        "manifest_path": cache["manifest_path"],
    }
    provenance = {
        "backend": "raw_wxs_maf",
        "case_id": case_id,
        "signature": cache["signature"],
    }

    feature_row = case_features_df.loc[case_features_df["case_id"] == case_id]
    summary_row = case_summary_df.loc[case_summary_df["case_id"] == case_id]
    if not feature_row.empty and not summary_row.empty:
        summary_values = pd.to_numeric(summary_row.iloc[0], errors="coerce").fillna(0)
        feature_values = [
            int(value)
            for value in feature_row.iloc[0].drop(labels=["case_id"]).tolist()
        ]
        mutated_genes = [
            gene_name
            for gene_name, value in zip(selected_genes, feature_values)
            if int(value) == 1
        ]
        nonsyn_count = int(summary_values.get("nonsyn_count", 0))
        indel_count = int(summary_values.get("INDEL_count", 0))
        snv_count = int(summary_values.get("SNV_count", 0))
        tmb = round(float(summary_values.get("TMB", 0.0)), 6)
        status = "success"
        errors = []

    payload = {
        "case_id": case_id,
        "source_path": case_file_map.get(case_id, ""),
        "selected_gene_count": selected_gene_count,
        "selected_genes": selected_genes,
        "feature_values": feature_values,
        "mutated_genes": mutated_genes,
        "nonsyn_count": nonsyn_count,
        "INDEL_count": indel_count,
        "SNV_count": snv_count,
        "TMB": tmb,
    }
    tool_result = make_tool_result(
        output_root=output_root,
        tool_name="wxs",
        status=status,
        identifier=case_id,
        metrics={
            "selected_gene_count": selected_gene_count,
            "feature_value_count": len(feature_values),
            "nonsyn_count": nonsyn_count,
        },
        artifacts=artifacts,
        provenance=provenance,
        errors=errors,
        payload=payload,
    )
    return {"tool_result": tool_result, "payload": payload}
