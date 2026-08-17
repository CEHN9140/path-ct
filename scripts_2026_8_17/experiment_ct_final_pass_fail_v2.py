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
PRETREATMENT_MARKER = re.compile(
    r"\b(?:post[- ]?nephrectomy|s/p\s+nephrectomy|post[- ]?operative)\b",
    re.IGNORECASE,
)


def normalize_review(value: Any) -> str:
    value = str(value or "").strip().upper().replace("-", "_")
    return {
        "ENHANCE": "ENHANCED",
        "ENHANCED_CT": "ENHANCED",
        "CE": "ENHANCED",
        "NONCONTRAST": "NON_CONTRAST",
        "NON_CONTRAST_CT": "NON_CONTRAST",
        "NC": "NON_CONTRAST",
        "YES": "PASS",
        "NO": "FAIL",
    }.get(value, value)


def phase_review_pass(phase: Any, review: Any) -> tuple[bool, str]:
    phase = str(phase or "UNKNOWN").strip().upper() or "UNKNOWN"
    review = normalize_review(review)
    if phase in ENHANCED_PHASES:
        if review == "NON_CONTRAST":
            return False, "phase_review_conflicts_with_metadata"
        return True, "explicit_enhanced"
    if phase == "NC":
        if review == "ENHANCED":
            return False, "phase_review_conflicts_with_metadata"
        return False, "non_contrast"
    if review == "ENHANCED":
        return True, "manual_enhanced"
    if review == "NON_CONTRAST":
        return False, "manual_non_contrast"
    return False, "phase_review_required"


def final_status(
    candidate_pass: bool,
    phase_pass: bool,
    artifact_pass: bool,
    segmentation_available: bool,
    mask_auto_pass: bool,
    mask_review: Any,
    review_complete: bool,
) -> str:
    if not review_complete:
        return "REVIEW_REQUIRED"
    if not candidate_pass or not phase_pass or not artifact_pass:
        return "FAIL"
    if not segmentation_available:
        return "SEGMENTATION_REQUIRED"
    if not mask_auto_pass:
        return "FAIL"
    mask_review = normalize_review(mask_review)
    if not mask_review:
        return "MASK_REVIEW_REQUIRED"
    return "PASS" if mask_review == "PASS" else "FAIL"


def as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


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


def save_contact_sheet(ct_path: Path, output_path: Path, title: str) -> None:
    import numpy as np
    import SimpleITK as sitk
    from PIL import Image, ImageDraw

    image = sitk.ReadImage(str(ct_path))
    values = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D CT, got shape={values.shape}: {ct_path}")

    def render(array: Any) -> Image.Image:
        array = np.clip((array + 100.0) / 600.0 * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB").resize((420, 300))

    z, y, x = values.shape
    z_indices = [int(index) for index in np.linspace(z * 0.25, z * 0.75, 3)]
    panels = [
        (f"Axial z={index}", render(np.rot90(values[index])))
        for index in z_indices
    ]
    panels.extend(
        [
            ("Coronal", render(np.rot90(values[:, y // 2, :]))),
            ("Sagittal", render(np.rot90(values[:, :, x // 2]))),
        ]
    )
    canvas = Image.new("RGB", (840, 690), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="black")
    for index, (label, panel) in enumerate(panels):
        left = (index % 2) * 420
        top = 30 + (index // 2) * 220
        canvas.paste(panel.resize((410, 190)), (left + 5, top + 20))
        draw.text((left + 8, top + 3), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_mask_overlay(ct_path: Path, mask_path: Path, output_path: Path) -> tuple[bool, int, str]:
    import numpy as np
    import SimpleITK as sitk
    from PIL import Image

    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.ReadImage(str(mask_path))
    ct_values = sitk.GetArrayFromImage(ct).astype(np.float32, copy=False)
    mask_values = sitk.GetArrayFromImage(mask) > 0
    geometry_match = (
        ct.GetSize() == mask.GetSize()
        and np.allclose(ct.GetSpacing(), mask.GetSpacing(), atol=1e-4)
        and np.allclose(ct.GetOrigin(), mask.GetOrigin(), atol=1e-4)
        and np.allclose(ct.GetDirection(), mask.GetDirection(), atol=1e-4)
    )
    positive = int(mask_values.sum())
    if positive == 0:
        return geometry_match, 0, "tumor_mask_empty"

    z_index = int(np.median(np.where(mask_values)[0]))
    base = np.clip((ct_values[z_index] + 100.0) / 600.0 * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[mask_values[z_index]] = [220, 40, 40]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rot90(rgb)).resize((700, 700)).save(output_path)
    return geometry_match, positive, ""


def mask_qc(segmentation_root: Path, row: dict[str, str], overlay_root: Path) -> dict[str, Any]:
    import numpy as np
    import SimpleITK as sitk

    case_id = row["case_id"]
    snapshot_path = segmentation_root / "ct_tumor_seg" / f"{case_id}.json"
    result = {
        "segmentation_available": snapshot_path.is_file(),
        "mask_auto_pass": False,
        "mask_auto_reason": "segmentation_json_missing",
        "mask_positive_voxel_count": 0,
        "mask_geometry_match": False,
        "mask_overlay_path": "",
    }
    if not snapshot_path.is_file():
        return result

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tool_result = snapshot.get("tool_result", {})
    payload = snapshot.get("payload", {})
    if tool_result.get("status") != "success":
        result["mask_auto_reason"] = "segmentation_tool_failed"
        return result
    expected_uid = str(row.get("series_uid", ""))
    actual_uid = str(payload.get("ct_identity", {}).get("series_uid", ""))
    if actual_uid != expected_uid:
        result["mask_auto_reason"] = "segmentation_series_uid_mismatch"
        return result

    mask_path = str(
        tool_result.get("artifacts", {}).get("segmentation_path", "")
        or payload.get("segmentation_path", "")
    )
    if not mask_path or not Path(mask_path).is_file():
        result["mask_auto_reason"] = "mask_file_missing"
        return result

    ct = sitk.ReadImage(str(row["ct_path"]))
    mask = sitk.ReadImage(mask_path)
    mask_values = sitk.GetArrayFromImage(mask) > 0
    geometry_match = (
        ct.GetSize() == mask.GetSize()
        and np.allclose(ct.GetSpacing(), mask.GetSpacing(), atol=1e-4)
        and np.allclose(ct.GetOrigin(), mask.GetOrigin(), atol=1e-4)
        and np.allclose(ct.GetDirection(), mask.GetDirection(), atol=1e-4)
    )
    positive = int(mask_values.sum())
    result.update(
        {
            "mask_positive_voxel_count": positive,
            "mask_geometry_match": geometry_match,
            "mask_auto_pass": geometry_match and positive > 0,
            "mask_auto_reason": "pass" if geometry_match and positive > 0 else "mask_geometry_or_content_failed",
        }
    )
    overlay_path = overlay_root / f"{case_id}.png"
    save_mask_overlay(Path(row["ct_path"]), Path(mask_path), overlay_path)
    result["mask_overlay_path"] = str(overlay_path)
    return result


def read_reviews(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    reviews = {}
    for row in read_csv(path):
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            reviews[case_id] = row
    return reviews


def make_review_template(rows: list[dict[str, str]], output_path: Path) -> None:
    template = []
    for row in rows:
        phase = str(row.get("inferred_phase", "UNKNOWN") or "UNKNOWN").upper()
        template.append(
            {
                "case_id": row["case_id"],
                "series_uid": row.get("series_uid", ""),
                "inferred_phase": phase,
                "ct_path": row.get("ct_path", ""),
                "contact_sheet_path": str(output_path.parent / "contact_sheets" / f"{row['case_id']}.png"),
                "phase_review": "ENHANCED" if phase in ENHANCED_PHASES else ("NON_CONTRAST" if phase == "NC" else ""),
                "artifact_review": "",
                "mask_review": "",
                "review_note": "",
            }
        )
    write_csv(output_path, template)


def audit_case(
    row: dict[str, str],
    review: dict[str, str],
    segmentation_root: Path,
    overlay_root: Path,
) -> dict[str, Any]:
    candidate_checks = {
        "prefilter_pass": as_bool(row.get("prefilter_pass")),
        "nifti_qc_pass": as_bool(row.get("nifti_qc_pass")),
        "totalseg_pass": as_bool(row.get("totalseg_pass")),
    }
    candidate_pass = all(candidate_checks.values())
    candidate_reasons = [name for name, passed in candidate_checks.items() if not passed]

    metadata = " ".join(str(row.get(field, "") or "") for field in ("series_description", "study_description", "protocol_name"))
    pretreatment_pass = not PRETREATMENT_MARKER.search(metadata)
    phase_pass, phase_reason = phase_review_pass(row.get("inferred_phase"), review.get("phase_review", ""))
    artifact_review = normalize_review(review.get("artifact_review", ""))
    artifact_pass = artifact_review == "PASS"
    review_complete = bool(review) and artifact_review in {"PASS", "FAIL"}
    if str(row.get("inferred_phase", "UNKNOWN")).upper() == "UNKNOWN":
        review_complete = review_complete and normalize_review(review.get("phase_review", "")) in {"ENHANCED", "NON_CONTRAST"}
    if not pretreatment_pass:
        candidate_reasons.append("post_treatment_marker_detected")
    if not phase_pass and phase_reason != "phase_review_required":
        candidate_reasons.append(phase_reason)
    if review_complete and not artifact_pass:
        candidate_reasons.append("manual_artifact_review_failed")

    mask = mask_qc(segmentation_root, row, overlay_root)
    status = final_status(
        candidate_pass and pretreatment_pass,
        phase_pass,
        artifact_pass,
        bool(mask["segmentation_available"]),
        bool(mask["mask_auto_pass"]),
        review.get("mask_review", ""),
        review_complete,
    )
    if status == "MASK_REVIEW_REQUIRED":
        review_complete = False
    return {
        **row,
        "phase_review": review.get("phase_review", ""),
        "artifact_review": review.get("artifact_review", ""),
        "mask_review": review.get("mask_review", ""),
        "candidate_status": "PASS" if candidate_pass and pretreatment_pass else "FAIL",
        "phase_status": "PASS" if phase_pass else "FAIL",
        "artifact_status": "PASS" if artifact_pass else ("FAIL" if review_complete else "REVIEW_REQUIRED"),
        "pretreatment_status": "PASS" if pretreatment_pass else "FAIL",
        "segmentation_status": "PASS" if mask["segmentation_available"] else "REQUIRED",
        "mask_auto_status": "PASS" if mask["mask_auto_pass"] else "FAIL",
        "mask_auto_reason": mask["mask_auto_reason"],
        "mask_geometry_match": mask["mask_geometry_match"],
        "mask_positive_voxel_count": mask["mask_positive_voxel_count"],
        "mask_overlay_path": mask["mask_overlay_path"],
        "final_status": status,
        "final_fail_reasons": ";".join(dict.fromkeys(candidate_reasons)),
        "review_complete": review_complete,
        "segmentation_rerun": mask["segmentation_available"],
    }


def run(
    selected_csv: Path,
    experiment_root: Path,
    review_csv: Path | None = None,
    config_dir: Path = Path("configs"),
    run_segmentation: bool = False,
    skip_contact_sheets: bool = False,
) -> dict[str, Any]:
    rows = read_csv(selected_csv)
    experiment_root.mkdir(parents=True, exist_ok=True)
    review_template = experiment_root / "review_template.csv"
    if not skip_contact_sheets:
        for row in rows:
            save_contact_sheet(
                Path(row["ct_path"]),
                experiment_root / "contact_sheets" / f"{row['case_id']}.png",
                f"{row['case_id']} | inferred_phase={row.get('inferred_phase', 'UNKNOWN')}",
            )
    if not review_template.is_file():
        make_review_template(rows, review_template)
    (experiment_root / "review_guide.md").write_text(
        "# CT v2 review guide\n\n"
        "Fill `phase_review` for UNKNOWN cases with `ENHANCED` or `NON_CONTRAST`.\n"
        "Fill `artifact_review` for every case with `PASS` or `FAIL`; FAIL means a severe artifact affects tumor assessment or segmentation.\n"
        "After fresh segmentation, inspect `mask_overlays/` and fill `mask_review` with `PASS` or `FAIL`.\n",
        encoding="utf-8",
    )
    reviews = read_reviews(review_csv)

    if run_segmentation:
        requests = []
        for row in rows:
            review = reviews.get(row["case_id"], {})
            phase_pass, _ = phase_review_pass(row.get("inferred_phase"), review.get("phase_review", ""))
            artifact_pass = normalize_review(review.get("artifact_review", "")) == "PASS"
            candidate_pass = all(as_bool(row.get(field)) for field in ("prefilter_pass", "nifti_qc_pass", "totalseg_pass"))
            if phase_pass and artifact_pass and candidate_pass:
                requests.append(
                    {
                        "case_id": row["case_id"],
                        "ct_path": row["ct_path"],
                        "ct_identity": {
                            "series_uid": row.get("series_uid", ""),
                            "study_uid": row.get("study_uid", ""),
                        },
                    }
                )
        if not requests:
            raise ValueError("No cases passed phase/artifact review for tumor segmentation.")
        from tools.ct_tumor_seg import run_ct_tumor_seg_cohort

        run_ct_tumor_seg_cohort(requests, str(experiment_root), str(config_dir))

    audited = [
        audit_case(
            row,
            reviews.get(row["case_id"], {}),
            experiment_root,
            experiment_root / "mask_overlays",
        )
        for row in rows
    ]
    statuses = Counter(row["final_status"] for row in audited)
    summary = {
        "experiment": "ct_final_pass_fail_audit_v2",
        "scope": "standalone phase/artifact review plus fresh tumor segmentation audit",
        "inputs": {
            "selected_series_csv": str(selected_csv),
            "review_csv": str(review_csv) if review_csv else "",
            "config_dir": str(config_dir),
            "radqy_used": False,
            "contact_sheets_generated": not skip_contact_sheets,
            "tumor_segmentation_rerun": run_segmentation,
        },
        "selected_case_count": len(rows),
        "reviewed_case_count": sum(bool(reviews.get(row["case_id"])) for row in rows),
        "segmentation_case_count": sum(row["segmentation_status"] == "PASS" for row in audited),
        "status_counts": dict(statuses),
        "phase_review_counts": dict(Counter(normalize_review(row["phase_review"]) or "MISSING" for row in audited)),
        "artifact_review_counts": dict(Counter(normalize_review(row["artifact_review"]) or "MISSING" for row in audited)),
        "mask_auto_pass_count": sum(row["mask_auto_status"] == "PASS" for row in audited),
        "limitations": {
            "phase": "UNKNOWN is not auto-rejected; it requires manual enhanced/non-contrast review.",
            "artifact": "PASS/FAIL comes from the supplied visual review CSV, not the old NIfTI proxy.",
            "pretreatment": "Only explicit postoperative metadata markers are excluded.",
            "tumor_mask": "Segmentation is rerun for the current selected series; mask review still requires overlay inspection.",
        },
        "output_files": {
            "review_template": str(review_template),
            "review_guide": str(experiment_root / "review_guide.md"),
            "contact_sheet_dir": str(experiment_root / "contact_sheets"),
            "case_audit": str(experiment_root / "case_pass_fail_audit.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    write_csv(experiment_root / "case_pass_fail_audit.csv", audited)
    (experiment_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit the v2 CT PASS/FAIL experiment.")
    parser.add_argument("--selected-csv", type=Path, default=Path("output_kirc_v9/experiment_ct_no_radqy_selection/selected_series.csv"))
    parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v9/experiment_ct_final_pass_fail_v2"))
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--run-segmentation", action="store_true")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
