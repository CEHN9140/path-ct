from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from experiment_ct_phase_aware_reselection import read_candidates


PREFERRED_ENHANCED_PRIORITY = {
    "NEPH": 0,
    "MAIN_CE_HIGH": 1,
    "MAIN_CE_MEDIUM": 2,
    "CE_UNSPECIFIED": 3,
    "ART": 4,
}


def number(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def technical_key(row: dict[str, Any]) -> tuple[float, ... | str]:
    spacing = number(row.get("pixel_spacing_row")) * number(row.get("pixel_spacing_col"))
    return (
        number(row.get("slice_thickness_median")),
        spacing,
        number(row.get("z_gap_max")),
        -number(row.get("n_slices"), 0.0),
        -number(row.get("totalseg_total_roi_volume_ml"), 0.0),
        str(row.get("series_uid", "")),
    )


def select_without_radqy(series_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in series_rows:
        if row.get("eligible_candidate"):
            grouped[str(row.get("case_id", ""))].append(row)

    selected = {}
    for case_id, rows in grouped.items():
        enhanced = [
            row for row in rows
            if str(row.get("inferred_phase", "UNKNOWN")) in PREFERRED_ENHANCED_PRIORITY
        ]
        if enhanced:
            selected[case_id] = min(
                enhanced,
                key=lambda row: (
                    PREFERRED_ENHANCED_PRIORITY[row["inferred_phase"]],
                    technical_key(row),
                ),
            )
        else:
            selected[case_id] = min(rows, key=technical_key)
    return selected


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_previous(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle) if row.get("case_id")}


def read_technical_fields(qc_root: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = {}
    for case_id in sorted({row["case_id"] for row in candidates}):
        report_path = qc_root / case_id / "dicom_prefilter_report.csv"
        if not report_path.is_file():
            continue
        with report_path.open(encoding="utf-8", newline="") as handle:
            reports.update({row.get("source_path", ""): row for row in csv.DictReader(handle)})
    fields = (
        "n_slices",
        "slice_thickness_median",
        "pixel_spacing_row",
        "pixel_spacing_col",
        "z_spacing_median",
        "z_gap_max",
        "duplicate_z_count",
        "totalseg_total_roi_volume_ml",
    )
    output = []
    for row in candidates:
        report = reports.get(row.get("source_path", ""), {})
        output.append({**row, **{field: report.get(field, "") for field in fields}})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(
    phase_csv: Path,
    qc_root: Path,
    previous_case_csv: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    candidates = read_technical_fields(qc_root, read_candidates(phase_csv, qc_root))
    selected = select_without_radqy(candidates)
    previous = read_previous(previous_case_csv)
    current = {row["case_id"]: row for row in candidates if row["current_qc_selected"]}
    case_ids = sorted({row["case_id"] for row in candidates})

    candidate_rows = [
        {
            **row,
            "no_radqy_selected": row is selected.get(row["case_id"]),
            "preferred_enhanced_candidate": row.get("inferred_phase") in PREFERRED_ENHANCED_PRIORITY,
            "technical_sort_key": repr(technical_key(row)),
        }
        for row in candidates
    ]
    selected_rows = [{**row, "selection_arm": "no_radqy"} for row in selected.values()]
    case_rows = []
    for case_id in case_ids:
        new_row = selected.get(case_id, {})
        current_row = current.get(case_id, {})
        previous_row = previous.get(case_id, {})
        case_rows.append(
            {
                "case_id": case_id,
                "candidate_series_count": sum(row["case_id"] == case_id for row in candidates),
                "current_phase": current_row.get("inferred_phase", "UNKNOWN"),
                "previous_phase_aware_phase": previous_row.get("new_phase", previous_row.get("phase_aware_phase", "UNKNOWN")),
                "new_phase": new_row.get("inferred_phase", ""),
                "new_retained": bool(new_row),
                "new_series_uid": new_row.get("series_uid", ""),
                "current_to_new": f"{current_row.get('inferred_phase', 'UNKNOWN')}->{new_row.get('inferred_phase', 'NOT_RETAINED')}" if current_row else f"UNKNOWN->{new_row.get('inferred_phase', 'NOT_RETAINED')}",
                "previous_to_new": f"{previous_row.get('new_phase', previous_row.get('phase_aware_phase', 'UNKNOWN'))}->{new_row.get('inferred_phase', 'NOT_RETAINED')}",
                "current_selection_changed": bool(current_row and current_row.get("series_uid") != new_row.get("series_uid")),
                "previous_phase_changed": bool(
                    previous_row
                    and previous_row.get("new_phase", previous_row.get("phase_aware_phase", "UNKNOWN"))
                    != new_row.get("inferred_phase", "NOT_RETAINED")
                ),
            }
        )

    summary = {
        "experiment": "ct_no_radqy_selection_v1",
        "inputs": {
            "phase_audit_csv": str(phase_csv),
            "qc_root": str(qc_root),
            "previous_fallback_case_csv": str(previous_case_csv),
            "base_qc_candidate_definition": ["prefilter_pass", "nifti_qc_pass", "totalseg_pass"],
            "radqy_used_for_selection": False,
            "rerun_tumor_segmentation": False,
            "run_radiomics": False,
        },
        "candidate_series_count": len(candidates),
        "base_qc_case_count": len(case_ids),
        "new_retained_case_count": len(selected),
        "new_case_loss_count": len(set(case_ids) - set(selected)),
        "current_phase_counts": dict(Counter(row["current_phase"] for row in case_rows)),
        "previous_phase_aware_phase_counts": dict(Counter(row["previous_phase_aware_phase"] for row in case_rows)),
        "new_phase_counts": dict(Counter(row["new_phase"] for row in case_rows if row["new_retained"])),
        "current_selection_changed_case_count": sum(row["current_selection_changed"] for row in case_rows),
        "previous_phase_changed_case_count": sum(row["previous_phase_changed"] for row in case_rows),
        "selection_policy": {
            "hard_qc": "structural QC only",
            "preferred_enhanced_priority": PREFERRED_ENHANCED_PRIORITY,
            "same_phase_tie_break": "slice_thickness, pixel_spacing_area, z_gap_max, n_slices, kidney_volume",
            "without_preferred_enhanced": "technical_quality_best_without_phase_preference",
            "unknown_is_not_excluded": True,
        },
        "output_files": {
            "candidates": str(experiment_root / "candidate_reselection_audit.csv"),
            "selected": str(experiment_root / "selected_series.csv"),
            "cases": str(experiment_root / "case_reselection_summary.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    write_csv(experiment_root / "candidate_reselection_audit.csv", candidate_rows)
    write_csv(experiment_root / "selected_series.csv", selected_rows)
    write_csv(experiment_root / "case_reselection_summary.csv", case_rows)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CT selection without RadQy using phase and technical quality.")
    parser.add_argument(
        "--phase-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_audit/series_phase_audit.csv"),
    )
    parser.add_argument("--qc-root", type=Path, default=Path("output_kirc/ct_qc"))
    parser.add_argument(
        "--previous-case-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_aware_fallback_reselection/case_reselection_summary.csv"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_no_radqy_selection"),
    )
    args = parser.parse_args()
    summary = run(args.phase_csv, args.qc_root, args.previous_case_csv, args.experiment_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
