from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PHASE_PRIORITY = {
    "MAIN_CE_HIGH": 0,
    "NEPH": 1,
    "MAIN_CE_MEDIUM": 2,
    "CE_UNSPECIFIED": 3,
    "ART": 4,
    "NC": 5,
    "DEL": 6,
    "UNKNOWN": 7,
}


def selection_rank(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def select_phase_aware(series_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in series_rows:
        if row.get("eligible_candidate"):
            grouped[str(row.get("case_id", ""))].append(row)
    selected = {}
    for case_id, rows in grouped.items():
        selected[case_id] = min(
            rows,
            key=lambda row: (
                PHASE_PRIORITY.get(str(row.get("inferred_phase", "UNKNOWN")), PHASE_PRIORITY["UNKNOWN"]),
                selection_rank(row.get("selection_rank")),
                str(row.get("series_uid", "")),
            ),
        )
    return selected


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_radqy_metrics(qc_root: Path, case_id: str) -> tuple[dict[str, dict[str, Any]], str]:
    path = qc_root / case_id / "selection_summary.json"
    if not path.is_file():
        return {}, ""
    payload = read_json(path)
    selected_id = str(payload.get("selected_series", {}).get("ct_id", "") or "")
    metrics = {}
    for item in payload.get("series_summaries", []):
        ct_id = str(item.get("ct_id", "") or "")
        if ct_id:
            metrics[ct_id] = item
    return metrics, selected_id


def metric_for_series(metrics: dict[str, dict[str, Any]], series_uid: str) -> dict[str, Any]:
    for ct_id, item in metrics.items():
        if ct_id.startswith(f"{series_uid}_"):
            return {
                "selected_ct_id": ct_id,
                "ct_path": item.get("ct_path", ""),
                "selection_rank": item.get("selection_rank", ""),
                "radqy_iqm_pass": item.get("radqy_iqm_pass", ""),
                "iqm_bad_count": item.get("iqm_bad_count", ""),
                "CV": item.get("CV", ""),
                "CJV": item.get("CJV", ""),
                "EFC": item.get("EFC", ""),
                "PSNR": item.get("PSNR", ""),
            }
    return {}


def read_candidates(phase_csv: Path, qc_root: Path) -> list[dict[str, Any]]:
    rows = []
    metrics_cache = {}
    selected_cache = {}
    with phase_csv.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("eligible_candidate", "").lower() != "true":
                continue
            case_id = str(raw.get("case_id", "") or "")
            if case_id not in metrics_cache:
                metrics_cache[case_id], selected_cache[case_id] = read_radqy_metrics(qc_root, case_id)
            metric = metric_for_series(metrics_cache[case_id], str(raw.get("series_uid", "") or ""))
            row = dict(raw)
            row.update(metric)
            row["current_qc_selected"] = str(raw.get("selected_by_current_qc", "")).lower() == "true"
            row["phase_priority"] = PHASE_PRIORITY.get(
                str(row.get("inferred_phase", "UNKNOWN")), PHASE_PRIORITY["UNKNOWN"]
            )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(phase_csv: Path, qc_root: Path, experiment_root: Path) -> dict[str, Any]:
    candidates = read_candidates(phase_csv, qc_root)
    selected = select_phase_aware(candidates)
    current_selected = {
        row["case_id"]: row
        for row in candidates
        if row["current_qc_selected"]
    }
    case_ids = sorted({row["case_id"] for row in candidates})
    selected_rows = []
    case_rows = []
    for case_id in case_ids:
        phase_row = selected[case_id]
        current = current_selected.get(case_id, {})
        selected_rows.append({**phase_row, "phase_aware_selected": True})
        case_rows.append(
            {
                "case_id": case_id,
                "candidate_series_count": sum(row["case_id"] == case_id for row in candidates),
                "current_phase": current.get("inferred_phase", "UNKNOWN"),
                "phase_aware_phase": phase_row.get("inferred_phase", "UNKNOWN"),
                "current_series_uid": current.get("series_uid", ""),
                "phase_aware_series_uid": phase_row.get("series_uid", ""),
                "phase_aware_selection_rank": phase_row.get("selection_rank", ""),
                "phase_aware_priority": phase_row.get("phase_priority", ""),
                "changed_selection": bool(current and current.get("series_uid") != phase_row.get("series_uid")),
            }
        )

    summary = {
        "experiment": "ct_phase_aware_reselection_v1",
        "inputs": {
            "phase_audit_csv": str(phase_csv),
            "qc_root": str(qc_root),
            "base_qc_candidate_definition": ["prefilter_pass", "nifti_qc_pass", "totalseg_pass"],
            "rerun_tumor_segmentation": False,
            "run_radiomics": False,
        },
        "candidate_series_count": len(candidates),
        "base_qc_case_count": len(case_ids),
        "phase_aware_retained_case_count": len(selected_rows),
        "current_selected_phase_counts": dict(Counter(row["current_phase"] for row in case_rows)),
        "phase_aware_selected_phase_counts": dict(Counter(row["phase_aware_phase"] for row in case_rows)),
        "changed_selection_case_count": sum(row["changed_selection"] for row in case_rows),
        "phase_priority": PHASE_PRIORITY,
        "selection_policy": {
            "phase_priority_before_radqy": True,
            "radqy_rank_used_within_phase": True,
            "unknown_is_fallback_not_excluded": True,
            "selection_rank_lower_is_better": True,
        },
        "output_files": {
            "candidates": str(experiment_root / "phase_aware_candidates.csv"),
            "selected": str(experiment_root / "phase_aware_selected_series.csv"),
            "cases": str(experiment_root / "case_reselection_summary.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    write_csv(experiment_root / "phase_aware_candidates.csv", [
        {**row, "phase_aware_selected": row is selected.get(row["case_id"])} for row in candidates
    ])
    write_csv(experiment_root / "phase_aware_selected_series.csv", selected_rows)
    write_csv(experiment_root / "case_reselection_summary.csv", case_rows)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test phase-aware CT series reselection without rerunning downstream tools.")
    parser.add_argument(
        "--phase-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_audit/series_phase_audit.csv"),
    )
    parser.add_argument("--qc-root", type=Path, default=Path("output_kirc/ct_qc"))
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_aware_reselection"),
    )
    args = parser.parse_args()
    summary = run(args.phase_csv, args.qc_root, args.experiment_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
