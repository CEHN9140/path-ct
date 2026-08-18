from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_cases(inventory_path: Path) -> list[tuple[str, str]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = []
    for case in inventory:
        case_id = str(case.get("Case_ID", "") or "")
        records = list(case.get("RNA_Seq", []) or [])
        if case_id and records:
            rows.append((case_id, str(records[0].get("File Path", "") or "")))
    return sorted(rows)


def read_rna_file(path: str, protein_coding_only: bool) -> tuple[pd.Series, int]:
    table = pd.read_csv(path, sep="\t", header=None, names=range(9))
    table.columns = table.iloc[1]
    table = table.iloc[6:].reset_index(drop=True)
    required = {"gene_name", "tpm_unstranded"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} lacks RNA columns: {sorted(missing)}")
    table["gene_name"] = table["gene_name"].astype(str).str.strip()
    table = table.loc[table["gene_name"] != ""].copy()
    if protein_coding_only:
        table["gene_type"] = table["gene_type"].astype(str).str.strip()
        table = table.loc[table["gene_type"] == "protein_coding"].copy()
    duplicate_count = int(table["gene_name"].duplicated(keep=False).sum())
    table["tpm_unstranded"] = pd.to_numeric(
        table["tpm_unstranded"], errors="coerce"
    ).fillna(0.0)
    return table.groupby("gene_name", sort=False)["tpm_unstranded"].max(), duplicate_count


def run(
    inventory_path: Path,
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cases = load_cases(inventory_path)
    rows = []
    series_by_case = {}
    for case_id, file_path in cases:
        if not Path(file_path).is_file():
            rows.append({"case_id": case_id, "file_path": file_path, "error": "file_missing"})
            continue
        try:
            series, duplicate_count = read_rna_file(
                file_path, bool(config["protein_coding_only"])
            )
            series_by_case[case_id] = series
            rows.append(
                {
                    "case_id": case_id,
                    "file_path": file_path,
                    "protein_coding_duplicate_row_count": duplicate_count,
                    "unique_gene_count": int(series.size),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "case_id": case_id,
                    "file_path": file_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    expression = pd.DataFrame(series_by_case).fillna(0.0)
    if not expression.empty:
        expressed_fraction = (
            expression >= float(config["min_tpm"])
        ).mean(axis=1)
        expression = expression.loc[
            expressed_fraction >= float(config["min_expressed_fraction"])
        ]
    expression = np.log2(expression + 1.0)
    mad = expression.sub(expression.median(axis=1), axis=0).abs().median(axis=1)
    ranked_genes = mad.sort_values(ascending=False, kind="stable").index.tolist()
    top_sets = {
        str(limit): set(ranked_genes[:limit]) for limit in (1000, 2000, 5000)
    }
    baseline = top_sets["2000"]
    sensitivity = []
    for limit, genes in top_sets.items():
        union = len(genes | baseline)
        sensitivity.append(
            {
                "top_gene_count": int(limit),
                "available_gene_count": int(len(ranked_genes)),
                "overlap_with_2000": int(len(genes & baseline)),
                "jaccard_with_2000": float(len(genes & baseline) / union) if union else 1.0,
            }
        )

    summary = {
        "experiment": "rna_feature_sensitivity_audit_v1",
        "inputs": {"inventory": str(inventory_path), "config": str(config_path)},
        "case_count": len(cases),
        "loaded_case_count": len(series_by_case),
        "duplicate_gene_case_count": sum(
            int(row.get("protein_coding_duplicate_row_count", 0) or 0) > 0
            for row in rows
        ),
        "duplicate_gene_row_count": int(
            sum(int(row.get("protein_coding_duplicate_row_count", 0) or 0) for row in rows)
        ),
        "filtered_gene_count": int(len(ranked_genes)),
        "top_gene_sensitivity": sensitivity,
        "interpretation_policy": {
            "primary_top_gene_count": 2000,
            "duplicate_handling_in_main_pipeline": "groupby_gene_name_max",
            "no_main_pipeline_change": True,
        },
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_csv(experiment_root / "rna_duplicate_gene_audit.csv", rows)
    write_csv(experiment_root / "rna_top_gene_sensitivity.csv", sensitivity)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RNA duplicate genes and top-MAD sensitivity.")
    parser.add_argument("--inventory", type=Path, default=Path("data/tcga_kirc_data.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/rna.yaml"))
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_rna_feature_sensitivity_audit"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.inventory, args.config, args.experiment_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
