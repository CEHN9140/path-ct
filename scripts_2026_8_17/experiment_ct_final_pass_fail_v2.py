from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PHASE_PRIORITY = {
    "NEPH": 0,
    "MAIN_CE_HIGH": 1,
    "MAIN_CE_MEDIUM": 2,
    "CE_UNSPECIFIED": 3,
    "ART": 4,
    "DEL": 5,
    "NC": 6,
    "UNKNOWN": 7,
}
POST_TREATMENT_MARKER = re.compile(
    r"(?:\b(?:status\s+post|s/p|post[- ]?(?:operative|op))\b[^\n]{0,80}\b(?:nephrectomy|renal\s+ablation)\b|\bpost[- ]?nephrectomy\b)",
    re.IGNORECASE,
)


def as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any, default: float = math.inf) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def phase_priority(phase_or_row: Any) -> int:
    phase = phase_or_row.get("inferred_phase") if isinstance(phase_or_row, dict) else phase_or_row
    return PHASE_PRIORITY.get(str(phase or "UNKNOWN").strip().upper(), PHASE_PRIORITY["UNKNOWN"])


def technical_key(row: dict[str, Any]) -> tuple[Any, ...]:
    thickness = number(row.get("slice_thickness_median"))
    row_spacing = number(row.get("pixel_spacing_row"))
    col_spacing = number(row.get("pixel_spacing_col"))
    spacing = row_spacing * col_spacing if math.isfinite(row_spacing * col_spacing) else math.inf
    z_spacing = number(row.get("z_spacing_median"))
    z_gap = number(row.get("z_gap_max"))
    z_gap_ratio = z_gap / z_spacing if z_spacing > 0 and math.isfinite(z_gap) else math.inf
    n_slices = number(row.get("n_slices"), default=0.0)
    physical_coverage = z_spacing * n_slices if z_spacing > 0 and n_slices >= 0 else -math.inf
    return (
        thickness,
        spacing,
        z_gap_ratio,
        -physical_coverage,
        str(row.get("series_uid", "")),
    )


def candidate_pass(row: dict[str, Any]) -> bool:
    return all(
        as_bool(row.get(field))
        for field in ("eligible_candidate", "prefilter_pass", "nifti_qc_pass", "totalseg_pass")
    )


def select_best_series(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if candidate_pass(row):
            grouped[str(row.get("case_id", ""))].append(row)
    selected = {
        case_id: min(case_rows, key=lambda row: (phase_priority(row), technical_key(row)))
        for case_id, case_rows in grouped.items()
    }
    assert len(selected) == len(grouped)
    assert len({row.get("case_id") for row in selected.values()}) == len(selected)
    return selected


def pretreatment_pass(row: dict[str, Any]) -> bool:
    metadata = " ".join(
        str(row.get(field, "") or "")
        for field in ("series_description", "study_description", "protocol_name")
    )
    return POST_TREATMENT_MARKER.search(metadata) is None


def geometry_matches(reference: Any, image: Any) -> bool:
    import numpy as np

    return (
        reference.GetSize() == image.GetSize()
        and np.allclose(reference.GetSpacing(), image.GetSpacing(), atol=1e-4)
        and np.allclose(reference.GetOrigin(), image.GetOrigin(), atol=1e-4)
        and np.allclose(reference.GetDirection(), image.GetDirection(), atol=1e-4)
    )


def tumor_mask_qc(segmentation_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    from utils.tool_utils import safe_identifier

    result = {
        "segmentation_success": False,
        "mask_exists": False,
        "series_uid_match": False,
        "geometry_match": False,
        "mask_nonempty": False,
        "mask_positive_voxel_count": 0,
        "mask_reason": "tumor_segmentation_missing",
    }
    snapshot_path = segmentation_root / "ct_tumor_seg" / f"{safe_identifier(row['case_id'])}.json"
    if not snapshot_path.is_file():
        return result

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tool_result = snapshot.get("tool_result", {})
    payload = snapshot.get("payload", {})
    result["segmentation_success"] = tool_result.get("status") == "success"
    if not result["segmentation_success"]:
        result["mask_reason"] = "tumor_segmentation_failed"
        return result

    result["series_uid_match"] = payload.get("ct_identity", {}).get("series_uid") == row.get("series_uid", "")
    mask_path = Path(
        str(
            tool_result.get("artifacts", {}).get("segmentation_path", "")
            or payload.get("segmentation_path", "")
        )
    )
    result["mask_exists"] = mask_path.is_file()
    if not result["series_uid_match"]:
        result["mask_reason"] = "tumor_segmentation_series_uid_mismatch"
        return result
    if not result["mask_exists"]:
        result["mask_reason"] = "tumor_mask_missing"
        return result

    import SimpleITK as sitk

    ct = sitk.ReadImage(str(row["ct_path"]))
    mask = sitk.ReadImage(str(mask_path))
    result["geometry_match"] = geometry_matches(ct, mask)
    if not result["geometry_match"]:
        result["mask_reason"] = "tumor_mask_geometry_mismatch"
        return result
    positive_voxels = int((sitk.GetArrayFromImage(mask) > 0).sum())
    result["mask_positive_voxel_count"] = positive_voxels
    result["mask_nonempty"] = positive_voxels > 0
    result["mask_reason"] = "pass" if result["mask_nonempty"] else "tumor_mask_empty"
    return result


def final_status(checks: dict[str, Any]) -> str:
    required = (
        "segmentation_success",
        "mask_exists",
        "series_uid_match",
        "geometry_match",
        "mask_nonempty",
    )
    return "PASS" if all(bool(checks.get(field)) for field in required) else "FAIL"


def run_tumor_segmentation(rows: list[dict[str, Any]], output_root: Path, config_dir: Path) -> None:
    if not rows:
        return
    from tools.ct_tumor_seg import run_ct_tumor_seg_cohort

    requests = [
        {
            "case_id": row["case_id"],
            "ct_path": row["ct_path"],
            "ct_identity": {
                "series_uid": row.get("series_uid", ""),
                "study_uid": row.get("study_uid", ""),
            },
        }
        for row in rows
    ]
    run_ct_tumor_seg_cohort(requests, str(output_root), str(config_dir))


def audit_case(row: dict[str, Any], experiment_root: Path) -> dict[str, Any]:
    candidate_ok = candidate_pass(row)
    pretreatment_ok = pretreatment_pass(row)
    mask = tumor_mask_qc(experiment_root, row) if candidate_ok and pretreatment_ok else {
        "segmentation_success": False,
        "mask_exists": False,
        "series_uid_match": False,
        "geometry_match": False,
        "mask_nonempty": False,
        "mask_positive_voxel_count": 0,
        "mask_reason": "not_run_after_preselection_failure",
    }
    status = final_status(mask) if candidate_ok and pretreatment_ok else "FAIL"
    reasons = []
    if not candidate_ok:
        reasons.append("candidate_qc_failed")
    if not pretreatment_ok:
        reasons.append("post_treatment_marker_detected")
    if candidate_ok and pretreatment_ok and status == "FAIL":
        reasons.append(mask["mask_reason"])
    return {
        **row,
        **mask,
        "selected_phase": str(row.get("inferred_phase", "UNKNOWN") or "UNKNOWN").upper(),
        "phase_priority": phase_priority(row),
        "final_status": status,
        "final_fail_reasons": ";".join(dict.fromkeys(reasons)),
    }


def run(
    candidate_csv: Path = Path("output_kirc_v9/experiment_ct_no_radqy_selection/candidate_reselection_audit.csv"),
    experiment_root: Path = Path("output_kirc_v9/experiment_ct_final_pass_fail_v2"),
    config_dir: Path = Path("configs"),
    run_segmentation: bool = False,
) -> dict[str, Any]:
    rows = read_csv(candidate_csv)
    experiment_root.mkdir(parents=True, exist_ok=True)
    selected = select_best_series(rows)
    selected_rows = list(selected.values())
    assert len(selected_rows) == len({row["case_id"] for row in selected_rows})

    candidate_audit = [
        {**row, "candidate_status": "PASS" if candidate_pass(row) else "FAIL"}
        for row in rows
    ]
    segmentation_rows = [row for row in selected_rows if pretreatment_pass(row)]
    if run_segmentation:
        run_tumor_segmentation(segmentation_rows, experiment_root, config_dir)

    audited = [audit_case(row, experiment_root) for row in selected_rows]
    write_csv(experiment_root / "candidate_series_audit.csv", candidate_audit)
    write_csv(experiment_root / "selected_series.csv", audited)
    write_csv(experiment_root / "case_pass_fail_audit.csv", audited)

    summary = {
        "experiment": "ct_final_pass_fail_v2",
        "inputs": {
            "candidate_csv": str(candidate_csv),
            "config_dir": str(config_dir),
            "run_segmentation": run_segmentation,
            "radqy_used": False,
            "aorta_segmentation_used": False,
            "manual_review_used": False,
        },
        "candidate_series_count": len(rows),
        "candidate_pass_series_count": sum(row["candidate_status"] == "PASS" for row in candidate_audit),
        "candidate_case_count": len({row["case_id"] for row in candidate_audit if row["candidate_status"] == "PASS"}),
        "selected_case_count": len(selected_rows),
        "selected_ct_count": len(selected_rows),
        "assert_one_selected_per_case": len(selected_rows) == len({row["case_id"] for row in selected_rows}),
        "segmentation_request_count": len(segmentation_rows) if run_segmentation else 0,
        "selected_phase_counts": dict(Counter(row["selected_phase"] for row in audited)),
        "final_status_counts": dict(Counter(row["final_status"] for row in audited)),
        "failure_reason_counts": dict(
            Counter(
                reason
                for row in audited
                for reason in row["final_fail_reasons"].split(";")
                if reason
            )
        ),
        "policy": {
            "series_candidate_gate": "structural_and_existing_kidney_qc_only",
            "series_selection": "phase_priority_then_thickness_spacing_z_continuity_coverage_uid",
            "one_series_per_case": True,
            "unknown_phase_allowed": True,
            "tumor_qc": "segmentation_success_uid_geometry_nonempty",
            "final_status": ["PASS", "FAIL"],
        },
        "output_files": {
            "candidate_series_audit": str(experiment_root / "candidate_series_audit.csv"),
            "selected_series": str(experiment_root / "selected_series.csv"),
            "case_pass_fail_audit": str(experiment_root / "case_pass_fail_audit.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen automatic CT PASS/FAIL experiment.")
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_no_radqy_selection/candidate_reselection_audit.csv"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_final_pass_fail_v2"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--run-segmentation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
