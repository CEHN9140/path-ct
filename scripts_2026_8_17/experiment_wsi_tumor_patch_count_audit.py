from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


THRESHOLDS = (50, 100, 250)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tumor_summaries(output_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    summary_root = output_root / "wsi_tumor_seg"
    for path in sorted(summary_root.glob("*/summary.json")):
        summary = read_json(path)
        case_id = str(summary.get("case_id", path.parent.name) or path.parent.name)
        result[case_id] = summary
    for path in sorted(summary_root.glob("*.json")):
        snapshot = read_json(path)
        payload = dict(snapshot.get("payload", {}) or {})
        metrics = dict(snapshot.get("tool_result", {}).get("metrics", {}) or {})
        record = {**metrics, **payload}
        case_id = str(record.get("case_id", "") or path.stem)
        if case_id and case_id not in result and "tumor_patch_count" in record:
            result[case_id] = record
    if not result:
        raise FileNotFoundError(
            f"No WSI tumor-segmentation summaries found under {summary_root}"
        )
    return result


def load_grandqc_status(output_root: Path) -> dict[str, bool]:
    result = {}
    for path in sorted((output_root / "wsi_qc").glob("*/selection_summary.json")):
        summary = read_json(path)
        case_id = str(summary.get("case_id", path.parent.name) or path.parent.name)
        result[case_id] = bool(summary.get("case_qc_passes_threshold", False))
    return result


def build_audit_rows(
    tumor_summaries: Mapping[str, Mapping[str, Any]],
    grandqc_status: Mapping[str, bool],
) -> list[dict[str, Any]]:
    rows = []
    for case_id, summary in sorted(tumor_summaries.items()):
        tumor_count = int(summary.get("tumor_patch_count", 0) or 0)
        row = {
            "case_id": case_id,
            "slide_path": str(summary.get("slide_path", "") or ""),
            "patch_count": int(summary.get("patch_count", 0) or 0),
            "tumor_patch_count": tumor_count,
            "tumor_probability_threshold": summary.get("tumor_probability_threshold", ""),
            "grandqc_pass": grandqc_status.get(case_id, ""),
        }
        for threshold in THRESHOLDS:
            row[f"tumor_patches_ge_{threshold}"] = tumor_count >= threshold
        rows.append(row)
    return rows


def summarize_patch_counts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = np.asarray(
        [int(row.get("tumor_patch_count", 0) or 0) for row in rows], dtype=float
    )
    if not len(counts):
        return {"n": 0, "thresholds": {str(value): {"pass_count": 0} for value in THRESHOLDS}}
    summary = {
        "n": int(len(counts)),
        "min": int(np.min(counts)),
        "p10": float(np.percentile(counts, 10)),
        "p25": float(np.percentile(counts, 25)),
        "median": float(np.median(counts)),
        "p75": float(np.percentile(counts, 75)),
        "p90": float(np.percentile(counts, 90)),
        "max": int(np.max(counts)),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        pass_count = int(np.count_nonzero(counts >= threshold))
        summary["thresholds"][str(threshold)] = {
            "pass_count": pass_count,
            "retention_vs_tumor_seg": pass_count / len(counts),
        }
    return summary


def run(output_root: Path, experiment_root: Path) -> dict[str, Any]:
    tumor_summaries = load_tumor_summaries(output_root)
    grandqc_status = load_grandqc_status(output_root)
    rows = build_audit_rows(tumor_summaries, grandqc_status)
    summary = summarize_patch_counts(rows)
    grandqc_pass_count = sum(value is True for value in grandqc_status.values())
    tumor_cases_with_grandqc_pass = sum(row["grandqc_pass"] is True for row in rows)
    for threshold in THRESHOLDS:
        entry = summary["thresholds"][str(threshold)]
        grandqc_threshold_count = sum(
            row["grandqc_pass"] is True
            and int(row["tumor_patch_count"]) >= threshold
            for row in rows
        )
        entry["grandqc_pass_count"] = grandqc_threshold_count
        entry["retention_vs_grandqc_pass"] = (
            grandqc_threshold_count / grandqc_pass_count
            if grandqc_pass_count
            else None
        )
    summary.update(
        {
            "experiment": "wsi_tumor_patch_count_audit",
            "output_root": str(output_root),
            "tumor_seg_summary_case_count": len(rows),
            "grandqc_pass_case_count": grandqc_pass_count,
            "tumor_cases_with_grandqc_pass": tumor_cases_with_grandqc_pass,
            "grandqc_status_missing_for_tumor_cases": sum(
                row["grandqc_pass"] == "" for row in rows
            ),
        }
    )
    experiment_root.mkdir(parents=True, exist_ok=True)
    with (experiment_root / "tumor_patch_count_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit WSI tumor patch-count retention.")
    parser.add_argument("--output-root", type=Path, default=Path("output_kirc"))
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc/experiment_wsi_tumor_patch_count_audit"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.experiment_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
