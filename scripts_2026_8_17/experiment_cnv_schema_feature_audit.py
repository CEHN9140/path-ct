from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"Chromosome", "Start", "End", "Segment_Mean"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_cnv_records(inventory_path: Path) -> list[tuple[str, str]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = []
    for case in inventory:
        case_id = str(case.get("Case_ID", "") or "")
        records = list(case.get("CNV", []) or [])
        if case_id and records:
            rows.append((case_id, str(records[0].get("File Path", "") or "")))
    return sorted(rows)


def audit_schema(records: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for case_id, file_path in records:
        path = Path(file_path)
        row: dict[str, Any] = {
            "case_id": case_id,
            "file_path": file_path,
            "exists": path.is_file(),
            "required_columns_present": False,
            "columns": "",
            "segment_count": 0,
            "segment_mean_min": "",
            "segment_mean_median": "",
            "segment_mean_max": "",
            "invalid_segment_mean_count": "",
            "error": "",
        }
        if not path.is_file():
            row["error"] = "file_missing"
            rows.append(row)
            continue
        try:
            table = pd.read_csv(path, sep="\t", usecols=lambda column: True)
            columns = [str(column) for column in table.columns]
            row["columns"] = "|".join(columns)
            row["required_columns_present"] = REQUIRED_COLUMNS.issubset(columns)
            if not row["required_columns_present"]:
                row["error"] = "missing_required_columns"
                rows.append(row)
                continue
            values = pd.to_numeric(table["Segment_Mean"], errors="coerce")
            valid = values.dropna().to_numpy(float)
            row.update(
                {
                    "segment_count": int(len(table)),
                    "segment_mean_min": float(np.min(valid)) if len(valid) else "",
                    "segment_mean_median": float(np.median(valid)) if len(valid) else "",
                    "segment_mean_max": float(np.max(valid)) if len(valid) else "",
                    "invalid_segment_mean_count": int(values.isna().sum()),
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def audit_feature_correlations(feature_csv: Path, threshold: float) -> dict[str, Any]:
    if not feature_csv.is_file():
        return {
            "available": False,
            "feature_csv": str(feature_csv),
            "error": "feature_csv_missing",
            "high_correlation_pairs": [],
        }
    table = pd.read_csv(feature_csv)
    values = table.drop(columns=["case_id"], errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    correlation = values.corr().abs()
    pairs = []
    for index, left in enumerate(correlation.columns):
        for right in correlation.columns[index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and float(value) >= threshold:
                pairs.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "absolute_correlation": float(value),
                    }
                )
    pairs.sort(key=lambda item: item["absolute_correlation"], reverse=True)
    return {
        "available": True,
        "feature_csv": str(feature_csv),
        "case_count": int(len(table)),
        "feature_count": int(values.shape[1]),
        "threshold": float(threshold),
        "high_correlation_pair_count": len(pairs),
        "high_correlation_pairs": pairs,
    }


def run(
    inventory_path: Path,
    feature_csv: Path,
    experiment_root: Path,
    correlation_threshold: float,
) -> dict[str, Any]:
    records = load_cnv_records(inventory_path)
    schema_rows = audit_schema(records)
    correlation = audit_feature_correlations(feature_csv, correlation_threshold)
    summary = {
        "experiment": "cnv_schema_feature_audit_v2",
        "inputs": {
            "inventory": str(inventory_path),
            "feature_csv": str(feature_csv),
        },
        "provenance_review": {
            "observed_schema": sorted(REQUIRED_COLUMNS),
            "observed_value": "Segment_Mean",
            "expected_backend_label": "gdc_copy_number_segment",
            "current_algorithm": "length_weighted_arm_mean_plus_driver_loci_and_fga",
        },
        "schema": {
            "inventory_case_count": len(records),
            "existing_file_count": sum(bool(row["exists"]) for row in schema_rows),
            "valid_schema_count": sum(bool(row["required_columns_present"]) for row in schema_rows),
            "invalid_file_count": sum(bool(row["error"]) for row in schema_rows),
        },
        "feature_correlation": correlation,
        "interpretation_policy": {
            "schema_failure_is_data_audit_not_automatic_case_exclusion": True,
            "high_correlation_is_reported_not_pruned": True,
            "no_main_pipeline_change": True,
        },
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_csv(experiment_root / "cnv_schema_audit.csv", schema_rows)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CNV schema and feature redundancy.")
    parser.add_argument("--inventory", type=Path, default=Path("data/tcga_kirc_data.json"))
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("output_kirc/cnv/case_features.csv"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_cnv_schema_feature_audit"),
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.inventory, args.feature_csv, args.experiment_root, args.correlation_threshold),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
