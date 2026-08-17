from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
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
FEATURE_NAMES = (
    "aorta_median_hu",
    "aorta_p75_hu",
    "aorta_p90_hu",
    "kidney_median_hu",
    "kidney_p75_hu",
    "kidney_p90_hu",
)
PRETREATMENT_MARKER = re.compile(
    r"\b(?:status\s+post|s/p|post[- ]?(?:operative|op))\b[^\n]{0,80}\b(?:nephrectomy|renal\s+ablation)\b",
    re.IGNORECASE,
)


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


def phase_from_metadata(phase: Any) -> tuple[bool | None, float | None, str]:
    phase = str(phase or "UNKNOWN").strip().upper() or "UNKNOWN"
    if phase in ENHANCED_PHASES:
        return True, 1.0, "explicit_enhanced"
    if phase == "NC":
        return False, 0.0, "explicit_non_contrast"
    return None, None, "image_classifier_required"


def pretreatment_pass(row: dict[str, Any]) -> bool:
    metadata = " ".join(
        str(row.get(field, "") or "")
        for field in ("series_description", "study_description", "protocol_name")
    )
    return PRETREATMENT_MARKER.search(metadata) is None


def read_qc_reports(qc_root: Path, rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    reports = {}
    for case_id in sorted({row.get("case_id", "") for row in rows}):
        path = qc_root / case_id / "dicom_prefilter_report.csv"
        if path.is_file():
            reports.update({row.get("source_path", ""): row for row in read_csv(path)})
    return reports


def kidney_mask_paths(report: dict[str, str]) -> list[Path]:
    output_dir = Path(str(report.get("totalseg_output_dir", "") or ""))
    return [output_dir / "kidney_left.nii.gz", output_dir / "kidney_right.nii.gz"]


def geometry_matches(reference: Any, image: Any) -> bool:
    import numpy as np

    return (
        reference.GetSize() == image.GetSize()
        and np.allclose(reference.GetSpacing(), image.GetSpacing(), atol=1e-4)
        and np.allclose(reference.GetOrigin(), image.GetOrigin(), atol=1e-4)
        and np.allclose(reference.GetDirection(), image.GetDirection(), atol=1e-4)
    )


def load_union_mask(paths: list[Path], reference: Any) -> Any:
    import numpy as np
    import SimpleITK as sitk

    union = np.zeros(tuple(reversed(reference.GetSize())), dtype=bool)
    for path in paths:
        if not path.is_file():
            continue
        image = sitk.ReadImage(str(path))
        if geometry_matches(reference, image):
            union |= sitk.GetArrayFromImage(image) > 0
    return union


def hu_features(ct_path: Path, aorta_path: Path, kidney_paths: list[Path]) -> tuple[dict[str, float] | None, str]:
    import numpy as np
    import SimpleITK as sitk

    if not aorta_path.is_file():
        return None, "aorta_mask_missing"
    ct = sitk.ReadImage(str(ct_path))
    aorta = sitk.ReadImage(str(aorta_path))
    if not geometry_matches(ct, aorta):
        return None, "aorta_geometry_mismatch"
    kidney = load_union_mask(kidney_paths, ct)
    if not kidney.any():
        return None, "kidney_mask_missing"
    values = sitk.GetArrayFromImage(ct).astype(np.float32, copy=False)
    aorta_mask = sitk.GetArrayFromImage(aorta) > 0
    aorta_values = values[aorta_mask & np.isfinite(values)]
    kidney_values = values[kidney & np.isfinite(values)]
    if not aorta_values.size or not kidney_values.size:
        return None, "roi_has_no_finite_hu"
    return {
        "aorta_median_hu": float(np.median(aorta_values)),
        "aorta_p75_hu": float(np.percentile(aorta_values, 75)),
        "aorta_p90_hu": float(np.percentile(aorta_values, 90)),
        "kidney_median_hu": float(np.median(kidney_values)),
        "kidney_p75_hu": float(np.percentile(kidney_values, 75)),
        "kidney_p90_hu": float(np.percentile(kidney_values, 90)),
    }, "pass"


def run_aorta_segmentation(
    rows: list[dict[str, str]],
    output_root: Path,
    aorta_device: str,
    totalseg_home: Path,
) -> dict[str, dict[str, Any]]:
    from totalsegmentator.python_api import totalsegmentator

    os.environ["TOTALSEG_HOME_DIR"] = str(totalseg_home.resolve())
    results = {}
    for row in rows:
        case_id = row["case_id"]
        case_root = output_root / "aorta_segmentation" / case_id
        snapshot_path = case_root / "result.json"
        aorta_path = case_root / "aorta.nii.gz"
        if snapshot_path.is_file() and aorta_path.is_file():
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if snapshot.get("ct_identity", {}).get("series_uid") == row.get("series_uid", ""):
                results[case_id] = snapshot
                continue
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)
        result = {
            "case_id": case_id,
            "ct_identity": {
                "series_uid": row.get("series_uid", ""),
                "study_uid": row.get("study_uid", ""),
            },
            "aorta_path": str(aorta_path),
            "success": False,
            "reason": "aorta_segmentation_failed",
        }
        try:
            totalsegmentator(
                input=Path(row["ct_path"]),
                output=case_root,
                task="total",
                roi_subset=["aorta"],
                quiet=True,
                device=aorta_device.replace("cuda:", "gpu:"),
            )
            result["success"] = aorta_path.is_file()
            result["reason"] = "pass" if result["success"] else "aorta_mask_missing"
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
        snapshot_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results[case_id] = result
    return results


def fit_enhancement_model(
    rows: list[dict[str, str]], features: dict[str, dict[str, float]]
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_values = []
    labels = []
    for row in rows:
        label = 1 if str(row.get("inferred_phase", "")).upper() in ENHANCED_PHASES else 0 if str(row.get("inferred_phase", "")).upper() == "NC" else None
        feature = features.get(row["case_id"])
        if label is None or feature is None:
            continue
        values = [feature[name] for name in FEATURE_NAMES]
        if all(np.isfinite(values)):
            x_values.append(values)
            labels.append(label)
    counts = Counter(labels)
    metadata = {
        "training_case_count": len(labels),
        "training_label_counts": {"NC": counts.get(0, 0), "ENHANCED": counts.get(1, 0)},
        "features": list(FEATURE_NAMES),
        "threshold": 0.5,
    }
    if len(counts) < 2 or min(counts.values()) < 2:
        return None, {**metadata, "available": False, "reason": "insufficient_labeled_roi_features"}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )
    model.fit(np.asarray(x_values), np.asarray(labels))
    return model, {**metadata, "available": True}


def classify_phase(
    row: dict[str, str], feature: dict[str, float] | None, model: Any
) -> tuple[bool, float | None, str]:
    explicit_pass, explicit_probability, reason = phase_from_metadata(row.get("inferred_phase"))
    if explicit_pass is not None:
        return explicit_pass, explicit_probability, reason
    if model is None or feature is None:
        return False, None, "unknown_phase_classifier_unavailable"
    import numpy as np

    probability = float(model.predict_proba([[feature[name] for name in FEATURE_NAMES]])[0, 1])
    return probability >= 0.5, probability, "image_classifier_enhanced" if probability >= 0.5 else "image_classifier_non_contrast"


def largest_tumor_component_plausible(
    tumor_mask: Any, kidney_mask: Any, spacing: tuple[float, float, float]
) -> dict[str, Any]:
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(tumor_mask, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return {"pass": False, "reason": "tumor_mask_empty", "largest_component_voxels": 0}
    sizes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(sizes[1:]) + 1)
    component = labels == largest_label
    z_coordinates = np.where(component)[0]
    if z_coordinates.min() == 0 or z_coordinates.max() == tumor_mask.shape[0] - 1:
        reason = "largest_component_touches_ct_boundary"
        return {"pass": False, "reason": reason, "largest_component_voxels": int(component.sum())}
    if not kidney_mask.any():
        return {"pass": False, "reason": "kidney_mask_missing", "largest_component_voxels": int(component.sum())}
    distance = ndimage.distance_transform_edt(~kidney_mask, sampling=spacing)
    minimum_distance_mm = float(distance[component].min())
    intersects_kidney = bool(np.logical_and(component, kidney_mask).any())
    plausible = intersects_kidney or minimum_distance_mm <= 30.0
    return {
        "pass": plausible,
        "reason": "pass" if plausible else "largest_component_far_from_kidney",
        "largest_component_voxels": int(component.sum()),
        "largest_component_distance_to_kidney_mm": minimum_distance_mm,
        "largest_component_intersects_kidney": intersects_kidney,
        "largest_component_mask": component,
    }


def save_debug_overlay(ct_path: Path, component: Any, output_path: Path) -> None:
    import numpy as np
    import SimpleITK as sitk
    from PIL import Image

    values = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32, copy=False)
    z_coordinates = np.where(component)[0]
    z_start, z_end = int(z_coordinates.min()), int(z_coordinates.max())
    area_by_slice = component.reshape(component.shape[0], -1).sum(axis=1)
    z_indices = [int(index) for index in np.linspace(z_start, z_end, 3)]
    z_indices[1] = int(np.argmax(area_by_slice))
    panels = []
    for z_index in z_indices:
        base = np.clip((values[z_index] + 100.0) / 600.0 * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([base, base, base], axis=-1)
        rgb[component[z_index]] = [220, 40, 40]
        panels.append(Image.fromarray(np.rot90(rgb)).resize((300, 300)))
    canvas = Image.new("RGB", (900, 320), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 300, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def tumor_mask_qc(
    segmentation_root: Path,
    row: dict[str, str],
    kidney_paths: list[Path],
    debug_root: Path,
) -> dict[str, Any]:
    import SimpleITK as sitk

    case_id = row["case_id"]
    snapshot_path = segmentation_root / "ct_tumor_seg" / f"{case_id}.json"
    result = {
        "segmentation_success": False,
        "mask_auto_pass": False,
        "mask_reason": "tumor_segmentation_missing",
        "mask_positive_voxel_count": 0,
        "mask_overlay_path": "",
    }
    if not snapshot_path.is_file():
        return result
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tool_result = snapshot.get("tool_result", {})
    payload = snapshot.get("payload", {})
    if tool_result.get("status") != "success":
        result["mask_reason"] = "tumor_segmentation_failed"
        return result
    if payload.get("ct_identity", {}).get("series_uid") != row.get("series_uid", ""):
        result["mask_reason"] = "tumor_segmentation_series_uid_mismatch"
        return result
    mask_path = Path(str(tool_result.get("artifacts", {}).get("segmentation_path", "") or payload.get("segmentation_path", "")))
    if not mask_path.is_file():
        result["mask_reason"] = "tumor_mask_missing"
        return result
    result["segmentation_success"] = True
    ct = sitk.ReadImage(str(row["ct_path"]))
    mask = sitk.ReadImage(str(mask_path))
    if not geometry_matches(ct, mask):
        result["mask_reason"] = "tumor_mask_geometry_mismatch"
        return result
    tumor = sitk.GetArrayFromImage(mask) > 0
    kidney = load_union_mask(kidney_paths, ct)
    plausibility = largest_tumor_component_plausible(tumor, kidney, tuple(reversed(ct.GetSpacing())))
    result.update(
        {
            "mask_positive_voxel_count": int(tumor.sum()),
            "mask_auto_pass": bool(plausibility["pass"]),
            "mask_reason": plausibility["reason"],
            "largest_component_voxels": plausibility.get("largest_component_voxels", 0),
            "largest_component_distance_to_kidney_mm": plausibility.get("largest_component_distance_to_kidney_mm"),
            "largest_component_intersects_kidney": plausibility.get("largest_component_intersects_kidney", False),
        }
    )
    component = plausibility.get("largest_component_mask")
    if component is not None:
        overlay_path = debug_root / f"{case_id}.png"
        save_debug_overlay(Path(row["ct_path"]), component, overlay_path)
        result["mask_overlay_path"] = str(overlay_path)
    return result


def run_tumor_segmentation(rows: list[dict[str, str]], output_root: Path, config_dir: Path) -> None:
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


def run(
    selected_csv: Path,
    experiment_root: Path,
    qc_root: Path = Path("output_kirc/ct_qc"),
    config_dir: Path = Path("configs"),
    run_segmentation: bool = False,
    aorta_device: str = "cuda:0",
    totalseg_home: Path = Path("tools/totalsegmentator"),
) -> dict[str, Any]:
    rows = read_csv(selected_csv)
    experiment_root.mkdir(parents=True, exist_ok=True)
    reports = read_qc_reports(qc_root, rows)
    aorta_results = {}
    if run_segmentation:
        aorta_results = run_aorta_segmentation(rows, experiment_root, aorta_device, totalseg_home)
    else:
        for row in rows:
            snapshot_path = experiment_root / "aorta_segmentation" / row["case_id"] / "result.json"
            if snapshot_path.is_file():
                aorta_results[row["case_id"]] = json.loads(snapshot_path.read_text(encoding="utf-8"))

    features = {}
    feature_reasons = {}
    for row in rows:
        report = reports.get(row.get("source_path", ""), {})
        aorta_path = Path(str(aorta_results.get(row["case_id"], {}).get("aorta_path", "")))
        feature, reason = hu_features(Path(row["ct_path"]), aorta_path, kidney_mask_paths(report))
        if feature is not None:
            features[row["case_id"]] = feature
        else:
            feature_reasons[row["case_id"]] = reason

    model, model_summary = fit_enhancement_model(rows, features)
    phase_rows = []
    segmentation_candidates = []
    for row in rows:
        feature = features.get(row["case_id"])
        enhanced, probability, phase_reason = classify_phase(row, feature, model)
        candidate_pass = all(as_bool(row.get(field)) for field in ("prefilter_pass", "nifti_qc_pass", "totalseg_pass"))
        pre_pass = pretreatment_pass(row)
        if enhanced and candidate_pass and pre_pass:
            segmentation_candidates.append(row)
        phase_rows.append(
            {
                **row,
                "auto_enhanced": enhanced,
                "enhancement_probability": probability,
                "auto_phase_reason": phase_reason if phase_reason else feature_reasons.get(row["case_id"], ""),
                "aorta_feature_status": "PASS" if feature is not None else "FAIL",
            }
        )

    if run_segmentation:
        run_tumor_segmentation(segmentation_candidates, experiment_root, config_dir)

    audited = []
    for row in phase_rows:
        report = reports.get(row.get("source_path", ""), {})
        mask = tumor_mask_qc(
            experiment_root,
            row,
            kidney_mask_paths(report),
            experiment_root / "mask_overlays",
        )
        candidate_pass = as_bool(row.get("prefilter_pass")) and as_bool(row.get("nifti_qc_pass")) and as_bool(row.get("totalseg_pass"))
        pre_pass = pretreatment_pass(row)
        phase_pass = bool(row["auto_enhanced"])
        final_pass = candidate_pass and pre_pass and phase_pass and mask["segmentation_success"] and mask["mask_auto_pass"]
        reasons = []
        if not candidate_pass:
            reasons.append("candidate_qc_failed")
        if not pre_pass:
            reasons.append("post_treatment_marker_detected")
        if not phase_pass:
            reasons.append(str(row.get("auto_phase_reason", "phase_not_enhanced")))
        if not mask["segmentation_success"] or not mask["mask_auto_pass"]:
            reasons.append(mask["mask_reason"])
        audited.append(
            {
                **row,
                **mask,
                "artifact_qc": "not_a_separate_hard_gate",
                "final_status": "PASS" if final_pass else "FAIL",
                "final_fail_reasons": ";".join(dict.fromkeys(reasons)),
                "segmentation_success": mask["segmentation_success"],
            }
        )

    summary = {
        "experiment": "ct_final_pass_fail_audit_v2",
        "mode": "fully_automatic",
        "inputs": {
            "selected_series_csv": str(selected_csv),
            "qc_root": str(qc_root),
            "config_dir": str(config_dir),
            "run_segmentation": run_segmentation,
            "aorta_device": aorta_device,
            "totalseg_home": str(totalseg_home),
            "radqy_used": False,
            "manual_review_used": False,
        },
        "selected_case_count": len(rows),
        "aorta_success_count": sum(bool(item.get("success")) for item in aorta_results.values()),
        "feature_case_count": len(features),
        "segmentation_candidate_count": len(segmentation_candidates),
        "final_status_counts": dict(Counter(row["final_status"] for row in audited)),
        "auto_phase_counts": dict(Counter("ENHANCED" if row["auto_enhanced"] else "NON_CONTRAST" for row in audited)),
        "model": model_summary,
        "failure_reason_counts": dict(Counter(reason for row in audited for reason in row["final_fail_reasons"].split(";") if reason)),
        "policy": {
            "artifact_hard_gate": False,
            "unknown_phase": "logistic_regression_from_explicit_NC_and_enhanced_cases",
            "tumor_mask": "success + UID + geometry + nonempty + largest_component_near_kidney + no_z_boundary_touch",
            "debug_overlays_do_not_affect_decision": True,
        },
        "output_files": {
            "case_audit": str(experiment_root / "case_pass_fail_audit.csv"),
            "summary": str(experiment_root / "summary.json"),
            "aorta_root": str(experiment_root / "aorta_segmentation"),
            "tumor_segmentation_root": str(experiment_root / "ct_tumor_seg"),
            "mask_debug_overlay_root": str(experiment_root / "mask_overlays"),
        },
    }
    write_csv(experiment_root / "case_pass_fail_audit.csv", audited)
    (experiment_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully automatic CT PASS/FAIL audit with aorta and tumor segmentation.")
    parser.add_argument("--selected-csv", type=Path, default=Path("output_kirc_v9/experiment_ct_no_radqy_selection/selected_series.csv"))
    parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v9/experiment_ct_final_pass_fail_v2"))
    parser.add_argument("--qc-root", type=Path, default=Path("output_kirc/ct_qc"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--run-segmentation", action="store_true")
    parser.add_argument("--aorta-device", default="cuda:0")
    parser.add_argument("--totalseg-home", type=Path, default=Path("tools/totalsegmentator"))
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
