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


RELIABLE_ENHANCED_PRIORITY = {
    "NEPH": 0,
    "MAIN_CE_HIGH": 1,
    "MAIN_CE_MEDIUM": 2,
    "CE_UNSPECIFIED": 3,
    "ART": 4,
}


def is_radqy_pass(row: dict[str, Any]) -> bool:
    return str(row.get("radqy_iqm_pass", "")).strip().lower() == "true"


def rank_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def select_enhanced_or_radqy_best(series_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in series_rows:
        if row.get("eligible_candidate") and is_radqy_pass(row):
            grouped[str(row.get("case_id", ""))].append(row)

    selected = {}
    for case_id, rows in grouped.items():
        enhanced = [
            row for row in rows
            if str(row.get("inferred_phase", "UNKNOWN")) in RELIABLE_ENHANCED_PRIORITY
        ]
        if enhanced:
            selected[case_id] = min(
                enhanced,
                key=lambda row: (
                    RELIABLE_ENHANCED_PRIORITY[row["inferred_phase"]],
                    rank_value(row.get("selection_rank")),
                    str(row.get("series_uid", "")),
                ),
            )
        else:
            selected[case_id] = min(
                rows,
                key=lambda row: (rank_value(row.get("selection_rank")), str(row.get("series_uid", ""))),
            )
    return selected


def read_previous_selection(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle) if row.get("case_id")}


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
    candidates = read_candidates(phase_csv, qc_root)
    selected = select_enhanced_or_radqy_best(candidates)
    previous = read_previous_selection(previous_case_csv)
    current = {row["case_id"]: row for row in candidates if row["current_qc_selected"]}
    case_ids = sorted({row["case_id"] for row in candidates})

    candidate_rows = []
    for row in candidates:
        candidate_rows.append(
            {
                **row,
                "reselection_selected": row is selected.get(row["case_id"]),
                "reliable_enhanced_candidate": row.get("inferred_phase") in RELIABLE_ENHANCED_PRIORITY,
            }
        )

    selected_rows = [{**row, "selection_arm": "enhanced_or_radqy_fallback"} for row in selected.values()]
    case_rows = []
    for case_id in case_ids:
        new_row = selected.get(case_id, {})
        current_row = current.get(case_id, {})
        previous_row = previous.get(case_id, {})
        case_rows.append(
            {
                "case_id": case_id,
                "candidate_series_count": sum(row["case_id"] == case_id for row in candidates),
                "radqy_pass_candidate_count": sum(
                    row["case_id"] == case_id and is_radqy_pass(row) for row in candidates
                ),
                "reliable_enhanced_candidate_count": sum(
                    row["case_id"] == case_id and row.get("inferred_phase") in RELIABLE_ENHANCED_PRIORITY
                    and is_radqy_pass(row)
                    for row in candidates
                ),
                "current_phase": current_row.get("inferred_phase", "UNKNOWN"),
                "previous_phase_aware_phase": previous_row.get("phase_aware_phase", "UNKNOWN"),
                "new_phase": new_row.get("inferred_phase", ""),
                "new_retained": bool(new_row),
                "new_radqy_pass": new_row.get("radqy_iqm_pass", "") if new_row else "",
                "current_to_new": f"{current_row.get('inferred_phase', 'UNKNOWN')}->{new_row.get('inferred_phase', 'NOT_RETAINED')}" if current_row else f"UNKNOWN->{new_row.get('inferred_phase', 'NOT_RETAINED')}",
                "previous_to_new": f"{previous_row.get('phase_aware_phase', 'UNKNOWN')}->{new_row.get('inferred_phase', 'NOT_RETAINED')}",
                "current_selection_changed": bool(current_row and current_row.get("series_uid") != new_row.get("series_uid")),
                "previous_selection_changed": bool(previous_row and previous_row.get("phase_aware_series_uid") != new_row.get("series_uid")),
            }
        )

    summary = {
        "experiment": "ct_phase_aware_fallback_reselection_v1",
        "inputs": {
            "phase_audit_csv": str(phase_csv),
            "qc_root": str(qc_root),
            "previous_phase_aware_case_csv": str(previous_case_csv),
            "base_qc_candidate_definition": ["prefilter_pass", "nifti_qc_pass", "totalseg_pass"],
            "radqy_pass_is_hard_for_this_test": True,
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
        "previous_selection_changed_case_count": sum(row["previous_selection_changed"] for row in case_rows),
        "new_radqy_counts": dict(Counter("pass" if is_radqy_pass(row) else "fail" for row in selected.values())),
        "selection_policy": {
            "hard_qc": "structural QC plus RadQy pass",
            "reliable_enhancement_priority": RELIABLE_ENHANCED_PRIORITY,
            "fallback_without_reliable_enhancement": "best RadQy rank across all RadQy-pass candidates",
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
    parser = argparse.ArgumentParser(description="Test phase-aware selection with RadQy-best fallback.")
    parser.add_argument(
        "--phase-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_audit/series_phase_audit.csv"),
    )
    parser.add_argument("--qc-root", type=Path, default=Path("output_kirc/ct_qc"))
    parser.add_argument(
        "--previous-case-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_aware_reselection/case_reselection_summary.csv"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_aware_fallback_reselection"),
    )
    args = parser.parse_args()
    summary = run(args.phase_csv, args.qc_root, args.previous_case_csv, args.experiment_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
