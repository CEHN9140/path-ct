from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ENHANCED_PHASES = {
    "NEPH",
    "MAIN_CE_HIGH",
    "MAIN_CE_MEDIUM",
    "CE_UNSPECIFIED",
    "ART",
    "DEL",
}
CHECK_NAMES = (
    "candidate_pass",
    "phase_pass",
    "artifact_proxy_pass",
    "renal_coverage_pass",
    "pretreatment_pass",
    "tumor_mask_pass",
)
PRETREATMENT_MARKER = re.compile(
    r"\b(?:post[- ]?nephrectomy|s/p\s+nephrectomy|post[- ]?operative)\b",
    re.IGNORECASE,
)


def as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def phase_check(phase: Any) -> tuple[bool, str]:
    value = str(phase or "UNKNOWN").strip().upper() or "UNKNOWN"
    if value in ENHANCED_PHASES:
        return True, "enhanced_phase"
    if value == "NC":
        return False, "non_contrast"
    return False, "phase_unknown_requires_confirmation"


def all_checks_pass(checks: dict[str, bool]) -> bool:
    return all(checks.get(name, False) for name in CHECK_NAMES)


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


def read_qc_reports(qc_root: Path, rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    reports = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        report_path = qc_root / case_id / "dicom_prefilter_report.csv"
        if report_path.is_file():
            reports.update({row.get("source_path", ""): row for row in read_csv(report_path)})
    return reports


def audit_tumor_mask(tumor_root: Path, row: dict[str, str]) -> tuple[bool, list[str]]:
    case_id = row["case_id"]
    json_path = tumor_root / f"{case_id}.json"
    if not json_path.is_file():
        return False, ["tumor_segmentation_json_missing"]

    payload = json.loads(json_path.read_text(encoding="utf-8")).get("payload", {})
    qc = dict(payload.get("tumor_mask_qc", {}) or {})
    reasons = []
    if str(payload.get("ct_identity", {}).get("series_uid", "")) != str(row.get("series_uid", "")):
        reasons.append("tumor_mask_series_uid_mismatch")
    mask_path = str(payload.get("segmentation_path", "") or "")
    if not mask_path or not Path(mask_path).is_file():
        reasons.append("tumor_mask_file_missing")
    if not as_bool(qc.get("segmentation_exists")):
        reasons.append("segmentation_not_confirmed")
    if not as_bool(qc.get("has_positive_voxels")):
        reasons.append("tumor_mask_empty")
    if not as_bool(qc.get("geometry_matches_selected_ct")):
        reasons.append("tumor_mask_geometry_mismatch")
    if not as_bool(qc.get("passes_threshold")):
        reasons.append("tumor_mask_threshold_failed")
    return not reasons, reasons


def audit_case(row: dict[str, str], tumor_root: Path) -> dict[str, Any]:
    candidate_checks = {
        "prefilter_pass": as_bool(row.get("prefilter_pass")),
        "nifti_qc_pass": as_bool(row.get("nifti_qc_pass")),
        "totalseg_pass": as_bool(row.get("totalseg_pass")),
    }
    candidate_pass = all(candidate_checks.values())
    candidate_reasons = [name for name, passed in candidate_checks.items() if not passed]

    phase_pass, phase_reason = phase_check(row.get("inferred_phase"))

    artifact_proxy_pass = as_bool(row.get("nifti_qc_pass"))
    artifact_reasons = [] if artifact_proxy_pass else ["nifti_hu_geometry_qc_failed"]

    renal_coverage_pass = as_bool(row.get("totalseg_pass"))
    renal_reasons = [] if renal_coverage_pass else ["kidney_coverage_qc_failed"]

    metadata = " ".join(
        str(row.get(field, "") or "")
        for field in ("series_description", "study_description", "protocol_name")
    )
    pretreatment_marker = PRETREATMENT_MARKER.search(metadata)
    pretreatment_pass = not pretreatment_marker
    pretreatment_reasons = ["post_treatment_marker_detected"] if pretreatment_marker else []

    tumor_mask_pass, tumor_mask_reasons = audit_tumor_mask(tumor_root, row)
    checks = {
        "candidate_pass": candidate_pass,
        "phase_pass": phase_pass,
        "artifact_proxy_pass": artifact_proxy_pass,
        "renal_coverage_pass": renal_coverage_pass,
        "pretreatment_pass": pretreatment_pass,
        "tumor_mask_pass": tumor_mask_pass,
    }
    reasons = candidate_reasons + ([] if phase_pass else [phase_reason]) + artifact_reasons + renal_reasons
    reasons += pretreatment_reasons + tumor_mask_reasons
    return {
        **row,
        "qc_candidate_status": "PASS" if candidate_pass else "FAIL",
        "phase_status_final": "PASS" if phase_pass else "FAIL",
        "artifact_proxy_status": "PASS" if artifact_proxy_pass else "FAIL",
        "renal_coverage_status": "PASS" if renal_coverage_pass else "FAIL",
        "pretreatment_status": "PASS" if pretreatment_pass else "FAIL",
        "tumor_mask_status": "PASS" if tumor_mask_pass else "FAIL",
        "final_status": "PASS" if all_checks_pass(checks) else "FAIL",
        "final_fail_reasons": ";".join(dict.fromkeys(reasons)),
        "artifact_review_status": "NOT_PERFORMED",
        "pretreatment_check_scope": "metadata_marker_only",
        "tumor_segmentation_rerun": False,
        "radiomics_rerun": False,
    }


def run(
    selected_csv: Path,
    qc_root: Path,
    tumor_root: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    rows = read_csv(selected_csv)
    reports = read_qc_reports(qc_root, rows)
    audited = []
    for row in rows:
        report = reports.get(row.get("source_path", ""), {})
        merged = {**report, **row}
        audited.append(audit_case(merged, tumor_root))

    fail_reasons = Counter(
        reason
        for row in audited
        if row["final_status"] == "FAIL"
        for reason in row["final_fail_reasons"].split(";")
        if reason
    )
    summary = {
        "experiment": "ct_final_pass_fail_audit_v1",
        "scope": "standalone audit of existing no-RadQy phase-aware selected CT series",
        "inputs": {
            "selected_series_csv": str(selected_csv),
            "qc_root": str(qc_root),
            "tumor_segmentation_root": str(tumor_root),
            "radqy_used": False,
            "rerun_tumor_segmentation": False,
            "rerun_radiomics": False,
        },
        "selected_case_count": len(audited),
        "final_pass_case_count": sum(row["final_status"] == "PASS" for row in audited),
        "final_fail_case_count": sum(row["final_status"] == "FAIL" for row in audited),
        "stage_pass_counts": {
            name: sum(row[{
                "candidate_pass": "qc_candidate_status",
                "phase_pass": "phase_status_final",
                "artifact_proxy_pass": "artifact_proxy_status",
                "renal_coverage_pass": "renal_coverage_status",
                "pretreatment_pass": "pretreatment_status",
                "tumor_mask_pass": "tumor_mask_status",
            }[name]] == "PASS" for row in audited)
            for name in CHECK_NAMES
        },
        "final_phase_counts": dict(Counter(row["inferred_phase"] for row in audited if row["final_status"] == "PASS")),
        "final_fail_reason_counts": dict(fail_reasons),
        "limitations": {
            "artifact_review": "automated NIfTI/HU QC proxy only; visual severe-artifact review was not performed",
            "pretreatment_review": "metadata marker audit only; treatment timeline was not available",
            "tumor_mask_review": "existing JSON/mask artifacts were audited; segmentation was not rerun",
            "interpretation": "final PASS is an experiment-level automated audit result, not a frozen main-flow QC decision",
        },
        "output_files": {
            "case_audit": str(experiment_root / "case_pass_fail_audit.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_csv(experiment_root / "case_pass_fail_audit.csv", audited)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a compact CT QC pipeline with PASS/FAIL output.")
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_no_radqy_selection/selected_series.csv"),
    )
    parser.add_argument("--qc-root", type=Path, default=Path("output_kirc/ct_qc"))
    parser.add_argument("--tumor-root", type=Path, default=Path("output_kirc/ct_tumor_seg"))
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_final_pass_fail_audit"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.selected_csv, args.qc_root, args.tumor_root, args.experiment_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
