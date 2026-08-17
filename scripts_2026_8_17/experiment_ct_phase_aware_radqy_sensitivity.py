from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from experiment_ct_phase_aware_reselection import read_candidates, select_phase_aware


def is_radqy_pass(row: dict[str, Any]) -> bool:
    return str(row.get("radqy_iqm_pass", "")).strip().lower() == "true"


def select_sensitivity_arms(
    series_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    soft = select_phase_aware(series_rows)
    strict = select_phase_aware([row for row in series_rows if is_radqy_pass(row)])
    return soft, strict


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(phase_csv: Path, qc_root: Path, experiment_root: Path) -> dict[str, Any]:
    candidates = read_candidates(phase_csv, qc_root)
    soft, strict = select_sensitivity_arms(candidates)
    case_ids = sorted({row["case_id"] for row in candidates})
    candidate_rows = []
    for row in candidates:
        case_id = row["case_id"]
        selected_soft = soft.get(case_id, {})
        selected_strict = strict.get(case_id, {})
        candidate_rows.append(
            {
                **row,
                "soft_selected": row is selected_soft,
                "strict_selected": row is selected_strict,
            }
        )

    selected_rows = []
    for arm, selection in (("soft", soft), ("strict", strict)):
        for row in selection.values():
            selected_rows.append({**row, "selection_arm": arm})

    case_rows = []
    for case_id in case_ids:
        soft_row = soft.get(case_id, {})
        strict_row = strict.get(case_id, {})
        case_rows.append(
            {
                "case_id": case_id,
                "candidate_series_count": sum(row["case_id"] == case_id for row in candidates),
                "soft_retained": bool(soft_row),
                "strict_retained": bool(strict_row),
                "soft_phase": soft_row.get("inferred_phase", ""),
                "strict_phase": strict_row.get("inferred_phase", ""),
                "soft_series_uid": soft_row.get("series_uid", ""),
                "strict_series_uid": strict_row.get("series_uid", ""),
                "soft_radqy_pass": soft_row.get("radqy_iqm_pass", "") if soft_row else "",
                "strict_radqy_pass": strict_row.get("radqy_iqm_pass", "") if strict_row else "",
                "arm_selection_differs": bool(soft_row and strict_row)
                and soft_row.get("series_uid") != strict_row.get("series_uid"),
            }
        )

    def phase_counts(selection: dict[str, dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(row.get("inferred_phase", "UNKNOWN") for row in selection.values()))

    def radqy_counts(selection: dict[str, dict[str, Any]]) -> dict[str, int]:
        return dict(Counter("pass" if is_radqy_pass(row) else "fail" for row in selection.values()))

    summary = {
        "experiment": "ct_phase_aware_radqy_sensitivity_v1",
        "inputs": {
            "phase_audit_csv": str(phase_csv),
            "qc_root": str(qc_root),
            "base_qc_candidate_definition": ["prefilter_pass", "nifti_qc_pass", "totalseg_pass"],
            "rerun_tumor_segmentation": False,
            "run_radiomics": False,
        },
        "candidate_series_count": len(candidates),
        "base_qc_case_count": len(case_ids),
        "soft_retained_case_count": len(soft),
        "strict_retained_case_count": len(strict),
        "strict_case_loss_count": len(set(case_ids) - set(strict)),
        "soft_phase_counts": phase_counts(soft),
        "strict_phase_counts": phase_counts(strict),
        "soft_radqy_counts": radqy_counts(soft),
        "strict_radqy_counts": radqy_counts(strict),
        "selection_arm_difference_count": sum(row["arm_selection_differs"] for row in case_rows),
        "policy": {
            "soft": "phase priority first, RadQy selection rank within phase, RadQy fail retained with warning",
            "strict": "RadQy pass filter first, then phase priority and RadQy rank",
            "phase_priority_is_same_between_arms": True,
            "structural_qc_remains_hard": True,
        },
        "output_files": {
            "candidates": str(experiment_root / "candidate_arm_comparison.csv"),
            "selected": str(experiment_root / "selected_series_by_arm.csv"),
            "cases": str(experiment_root / "case_arm_comparison.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    write_csv(experiment_root / "candidate_arm_comparison.csv", candidate_rows)
    write_csv(experiment_root / "selected_series_by_arm.csv", selected_rows)
    write_csv(experiment_root / "case_arm_comparison.csv", case_rows)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare soft versus strict RadQy in phase-aware CT selection.")
    parser.add_argument(
        "--phase-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_audit/series_phase_audit.csv"),
    )
    parser.add_argument("--qc-root", type=Path, default=Path("output_kirc/ct_qc"))
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_aware_radqy_sensitivity"),
    )
    args = parser.parse_args()
    summary = run(args.phase_csv, args.qc_root, args.experiment_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
