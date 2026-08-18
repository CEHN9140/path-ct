from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import yaml
from tqdm.auto import tqdm

from utils.tool_utils import safe_identifier, run_command, split_device_requests, to_jsonable


PHASE_PRIORITY = {
    "NEPH": 0, "MAIN_CE_HIGH": 1, "MAIN_CE_MEDIUM": 2,
    "CE_UNSPECIFIED": 3, "ART": 4, "DEL": 5, "NC": 6, "UNKNOWN": 6,
}
CT_QC_CACHE_VERSION = "2026-08-18-final"
CT_QC_RUNTIME_KEYS = {
    "device",
    "devices",
    "workers",
    "workers_per_gpu",
    "num_workers",
    "batch_size",
    "pin_memory",
    "prefetch_factor",
}
PHASE_PATTERNS = {
    "NC": (
        r"\bPRE[- ]?CONTRAST\b", r"\bUNENHANCED\b", r"\bNON[- ]?CONTRAST\b",
        r"\bNO CONTRAST\b", r"\bWITHOUT CONTRAST\b", r"(?<!W)\bW/O CONTRAST\b",
        r"(?<!W)\bWO CONTRAST\b", r"\bPRE LIVER\b", r"\bKIDNEYS PRE\b",
    ),
    "ART": (r"\bARTERIAL(?: PHASE)?\b", r"\bCORTICOMEDULLARY\b"),
    "NEPH": (r"\bNEPH(?:RO(?:GRAPHIC)?)?\b", r"\bPARENCHYMAL\b", r"\bPARANCHYMAL\b"),
    "DEL": (
        r"\bDELAY(?:ED)?\b", r"\bEXCRET(?:ION|ORY)?\b",
        r"\b(?:3|5|10|12|15)\s*MIN(?:UTE)?S?\b", r"\bDELAY BLADDER\b",
        r"\bKIDNEYS & DELAY\b",
    ),
}
CE_PATTERNS = (r"\bPOST[- ]?CONTRAST\b", r"\bWITH CONTRAST\b", r"\bW CONTRAST\b", r"\bVENOUS\b")
POST_TREATMENT_MARKER = re.compile(
    r"(?:\b(?:STATUS\s+POST|S/P|POST[- ]?(?:OPERATIVE|OP))\b[^\n]{0,80}\b(?:NEPHRECTOMY|RENAL\s+ABLATION)\b|\bPOST[- ]?NEPHRECTOMY\b)",
    re.IGNORECASE,
)


def semantic_ct_qc_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): semantic_ct_qc_config(item)
            for key, item in value.items()
            if str(key) not in CT_QC_RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [semantic_ct_qc_config(item) for item in value]
    return value


def ct_qc_config_signature(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"version": CT_QC_CACHE_VERSION, "config": to_jsonable(semantic_ct_qc_config(config))},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def ct_case_signature(case: Mapping[str, Any]) -> str:
    records = [to_jsonable(dict(record)) for record in list(case.get("CT", []) or [])]
    records.sort(key=lambda record: json.dumps(record, sort_keys=True))
    payload = {
        "case_id": str(case.get("Case_ID", case.get("case_id", "")) or ""),
        "ct_records": records,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_case_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None
    selected_series = summary.get("selected_series", {})
    selected_path = str(
        selected_series.get("ct_path", "") if isinstance(selected_series, Mapping) else ""
    )
    if summary.get("case_qc_passes_threshold") and not Path(selected_path).is_file():
        return None
    return summary


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def normalize_anatomy_text(value: Any) -> str:
    text = str(value or "").upper()
    text = re.sub(r"CTCHESTABDPEL|CHESTABDPEL", " CHEST ABD PEL ", text)
    text = re.sub(r"CTCHEST", " CT CHEST ", text)
    text = re.sub(r"CTCAP", " CT CAP ", text)
    text = re.sub(r"C\s*[/\-]\s*A\s*[/\-]\s*P", " CAP ", text)
    return re.sub(r"[^A-Z0-9]+", " ", text)


def non_abdominal_series(row: Mapping[str, Any]) -> bool:
    series_text = normalize_anatomy_text(row.get("Series Description", row.get("series_description", "")))
    if not series_text:
        series_text = normalize_anatomy_text(
            " ".join(str(row.get(key, "") or "") for key in (
                "Study Description", "study_description", "Protocol Name", "protocol_name"
            ))
        )
    thoracic = re.search(r"\b(?:CHEST|THORAX|LUNG)\b", series_text)
    abdominal = re.search(r"\b(?:ABD|ABDOMEN|PELVIS|RENAL|KIDNEY|CAP)\b", series_text)
    return bool(thoracic and not abdominal)


def candidate_fail_reasons(row: Mapping[str, Any]) -> list[str]:
    return ["non_abdominal_chest_series"] if non_abdominal_series(row) else []


def candidate_pass(row: Mapping[str, Any]) -> bool:
    return all(
        str(row.get(field, "")).strip().lower() == "true"
        for field in ("prefilter_pass", "nifti_qc_pass", "totalseg_pass")
    ) and not candidate_fail_reasons(row)


def pretreatment_pass(row: Mapping[str, Any]) -> bool:
    metadata = " ".join(
        str(row.get(field, "") or "")
        for field in (
            "Series Description", "series_description",
            "Study Description", "study_description",
            "Protocol Name", "protocol_name",
        )
    )
    return POST_TREATMENT_MARKER.search(metadata) is None


def phase_priority(phase_or_row: Any) -> int:
    phase = phase_or_row.get("inferred_phase") if isinstance(phase_or_row, Mapping) else phase_or_row
    return PHASE_PRIORITY.get(str(phase or "UNKNOWN").strip().upper(), 6)


def numeric(value: Any, default: float = float("inf")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def technical_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    row_spacing, col_spacing = numeric(row.get("pixel_spacing_row")), numeric(row.get("pixel_spacing_col"))
    spacing = row_spacing * col_spacing
    z_spacing, z_gap = numeric(row.get("z_spacing_median")), numeric(row.get("z_gap_max"))
    z_gap_ratio = z_gap / z_spacing if z_spacing > 0 and np.isfinite(z_gap) else float("inf")
    n_slices = numeric(row.get("n_slices"), 0.0)
    coverage = z_spacing * n_slices if z_spacing > 0 and n_slices >= 0 else -float("inf")
    return (numeric(row.get("slice_thickness_median")), spacing, z_gap_ratio, -coverage, str(row.get("series_uid", "")))


def select_best_series(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if candidate_pass(row):
            grouped.setdefault(str(row.get("case_id", "")), []).append(row)
    selected = {case_id: min(items, key=lambda row: (phase_priority(row), technical_key(row))) for case_id, items in grouped.items()}
    assert len(selected) == len(grouped)
    assert len({row.get("case_id") for row in selected.values()}) == len(selected)
    return selected


def read_dicom_headers(series_path: Path) -> list[Any]:
    groups: dict[str, list[Any]] = {}
    for path in sorted(series_path.rglob("*")):
        if not path.is_file():
            continue
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            header.source_path = str(path)
            uid = str(getattr(header, "SeriesInstanceUID", "") or "missing_series_uid")
            groups.setdefault(uid, []).append(header)
        except Exception:
            pass
    return max(groups.values(), key=len) if groups else []


def dicom_prefilter_reasons(series_summary: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    image_types = set(series_summary.get("image_type_union", []) or [])
    excluded = {str(item).upper() for item in config["excluded_image_types"]}
    required = {str(item).upper() for item in config["required_image_types"]}
    reasons = []
    if series_summary.get("modality") != "CT": reasons.append("not_ct")
    if image_types & excluded: reasons.append("derived_or_non_primary_image_type")
    if required and not required.issubset(image_types): reasons.append("not_original_primary")
    if series_summary.get("missing_spatial_tags"): reasons.append("missing_spatial_tags")
    thickness = series_summary.get("slice_thickness_median")
    if not isinstance(thickness, (int, float)) or thickness <= 0 or thickness > float(config["max_slice_thickness"]): reasons.append("slice_thickness_out_of_range")
    spacing = [series_summary.get("pixel_spacing_row"), series_summary.get("pixel_spacing_col")]
    if any(not isinstance(value, (int, float)) or value <= 0 or value > float(config["max_pixel_spacing"]) for value in spacing): reasons.append("pixel_spacing_out_of_range")
    normal_z = series_summary.get("normal_z_median")
    if not isinstance(normal_z, (int, float)) or abs(normal_z) < float(config["min_abs_normal_z"]): reasons.append("non_axial_orientation")
    if int(series_summary.get("rows") or 0) < int(config["min_matrix_size"]) or int(series_summary.get("columns") or 0) < int(config["min_matrix_size"]): reasons.append("matrix_too_small")
    z_gap, z_spacing = series_summary.get("z_gap_max"), series_summary.get("z_spacing_median")
    if isinstance(z_gap, (int, float)) and isinstance(z_spacing, (int, float)) and z_spacing > 0 and z_gap > float(config["max_z_gap_factor"]) * z_spacing: reasons.append("large_slice_gap")
    if int(series_summary.get("duplicate_z_count") or 0) > int(config["max_duplicate_slices"]): reasons.append("duplicate_slice_positions")
    if non_abdominal_series(series_summary): reasons.append("non_abdominal_chest_series")
    return reasons


def summarize_dicom_series(headers: list[Any], ct_record: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    first = headers[0] if headers else None
    image_type_union = sorted({str(item).upper() for header in headers for item in ([getattr(header, "ImageType", [])] if isinstance(getattr(header, "ImageType", []), str) else list(getattr(header, "ImageType", []) or []))})
    rows = pd.to_numeric([getattr(header, "Rows", np.nan) for header in headers], errors="coerce")
    columns = pd.to_numeric([getattr(header, "Columns", np.nan) for header in headers], errors="coerce")
    thickness = pd.to_numeric([getattr(header, "SliceThickness", np.nan) for header in headers], errors="coerce")
    spacing_rows, spacing_cols, normals, positions = [], [], [], []
    missing_spatial_tags = False
    for header in headers:
        pixel_spacing = pd.to_numeric(list(getattr(header, "PixelSpacing", []) or []), errors="coerce")
        if len(pixel_spacing) == 2 and not pd.isna(pixel_spacing).any(): spacing_rows.append(float(pixel_spacing[0])); spacing_cols.append(float(pixel_spacing[1]))
        orientation = pd.to_numeric(list(getattr(header, "ImageOrientationPatient", []) or []), errors="coerce")
        position = pd.to_numeric(list(getattr(header, "ImagePositionPatient", []) or []), errors="coerce")
        if len(orientation) != 6 or len(position) != 3 or pd.isna(orientation).any() or pd.isna(position).any():
            missing_spatial_tags = True
            continue
        normal = np.cross(np.asarray(orientation[:3], dtype=float), np.asarray(orientation[3:], dtype=float))
        normals.append(float(normal[2])); positions.append(float(np.dot(np.asarray(position, dtype=float), normal)))
    rounded_z = np.asarray([round(value, 3) for value in positions], dtype=float)
    unique_z = np.unique(rounded_z) if len(rounded_z) else np.asarray([])
    dz = np.diff(np.sort(unique_z)) if len(unique_z) > 1 else np.asarray([])
    summary = {
        "modality": str(getattr(first, "Modality", "") if first is not None else "").upper(),
        "n_slices": len(headers),
        "rows": None if pd.isna(rows).all() else int(pd.Series(rows).dropna().mode().iloc[0]),
        "columns": None if pd.isna(columns).all() else int(pd.Series(columns).dropna().mode().iloc[0]),
        "image_type_union": image_type_union,
        "slice_thickness_median": None if pd.isna(thickness).all() else float(np.nanmedian(thickness)),
        "pixel_spacing_row": float(np.median(spacing_rows)) if spacing_rows else None,
        "pixel_spacing_col": float(np.median(spacing_cols)) if spacing_cols else None,
        "normal_z_median": float(np.median(normals)) if normals else None,
        "z_spacing_median": float(np.median(dz)) if len(dz) else None,
        "z_gap_max": float(np.max(dz)) if len(dz) else None,
        "duplicate_z_count": int(len(rounded_z) - len(unique_z)) if len(rounded_z) else 0,
        "missing_spatial_tags": missing_spatial_tags or not len(positions),
    }
    reasons = dicom_prefilter_reasons({**summary, **ct_record}, config)
    summary.update({"prefilter_reasons": reasons, "prefilter_pass": not reasons})
    return summary


def convert_dicom_series(series_path: Path, output_dir: Path, config: Mapping[str, Any]) -> tuple[str, list[str], list[str], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_nifti = {path.resolve() for path in output_dir.glob("*.nii.gz")}
    existing_json = {path.resolve() for path in output_dir.glob("*.json")}
    command = [str(config["bin"]), "-z", str(config["gzip_output"]), "-b", str(config["bids_sidecar"]), "-ba", str(config["anonymize_bids"]), "-f", str(config["filename"]), "-o", str(output_dir), str(series_path)]
    completed = run_command(command)
    if completed.returncode != 0:
        return "", [completed.stderr.strip() or completed.stdout.strip() or "dcm2niix conversion failed"], [], []
    generated_nifti = [path for path in output_dir.glob("*.nii.gz") if path.resolve() not in existing_nifti]
    generated_json = [path for path in output_dir.glob("*.json") if path.resolve() not in existing_json]
    usable = [path for path in generated_nifti if not any(token in path.name.lower() for token in ("_eq_", "mask", "seg", "label", "derived"))]
    if not usable:
        return "", ["dcm2niix did not create a usable .nii.gz file"], [str(path) for path in generated_nifti], [str(path) for path in generated_json]
    selected = max(usable, key=lambda path: path.stat().st_size)
    return str(selected), [], [str(path) for path in generated_nifti], [str(path) for path in generated_json]


def check_nifti_volume(nifti_path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result = {"nifti_qc_pass": False, "nifti_qc_reasons": [], "hu_min": None, "hu_max": None, "hu_p1": None, "hu_p99": None, "hu_range_p99_p1": None}
    try:
        image = sitk.ReadImage(str(nifti_path))
        if image.GetDimension() != 3:
            result["nifti_qc_reasons"].append("not_3d")
            return result
        values = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    except Exception:
        result["nifti_qc_reasons"].append("nifti_unreadable")
        return result
    values = values[np.isfinite(values)]
    if values.size == 0:
        result["nifti_qc_reasons"].append("no_finite_voxels")
        return result
    p1, p99 = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    result.update({"hu_min": float(values.min()), "hu_max": float(values.max()), "hu_p1": p1, "hu_p99": p99, "hu_range_p99_p1": p99 - p1})
    if result["hu_min"] < float(config["hu_min_allowed"]) or result["hu_max"] > float(config["hu_max_allowed"]): result["nifti_qc_reasons"].append("hu_out_of_range")
    if result["hu_range_p99_p1"] < float(config["hu_range_p99_p1_min"]): result["nifti_qc_reasons"].append("hu_range_too_small")
    result["nifti_qc_pass"] = not result["nifti_qc_reasons"]
    return result


def check_totalsegmentator_rois(nifti_path: str | Path, output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rois = [str(item) for item in config["roi_subset"]]
    result = {"totalseg_pass": False, "totalseg_execution_success": False, "totalseg_reasons": [], "totalseg_output_dir": str(output_dir), "totalseg_detected_rois": [], "totalseg_missing_rois": [], "totalseg_roi_voxels": {}, "totalseg_roi_volumes_ml": {}, "totalseg_total_roi_volume_ml": 0.0}
    remove_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOTALSEG_HOME_DIR"] = str(Path(config["home_dir"]).resolve())
    try:
        from totalsegmentator.python_api import totalsegmentator
        totalsegmentator(input=Path(nifti_path), output=output_dir, task=str(config["task"]), roi_subset=rois if bool(config["use_roi_subset"]) else None, quiet=True, device=str(config["device"]).replace("cuda:", "gpu:"))
    except Exception as exc:
        result["totalseg_reasons"].append(f"totalsegmentator_failed:{type(exc).__name__}:{exc}")
        return result
    result["totalseg_execution_success"] = True
    image = sitk.ReadImage(str(nifti_path))
    voxel_volume_ml = float(np.prod(image.GetSpacing()) / 1000.0)
    for roi in rois:
        mask_path = output_dir / f"{roi}.nii.gz"
        voxels = 0
        if mask_path.is_file():
            try:
                voxels = int(np.count_nonzero(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0))
            except Exception:
                voxels = 0
        volume = float(voxels * voxel_volume_ml)
        result["totalseg_roi_voxels"][roi] = voxels
        result["totalseg_roi_volumes_ml"][roi] = volume
        result["totalseg_total_roi_volume_ml"] += volume
        (result["totalseg_detected_rois"] if volume > float(config["min_single_roi_volume_ml"]) else result["totalseg_missing_rois"]).append(roi)
    result["totalseg_pass"] = bool(result["totalseg_detected_rois"])
    if not result["totalseg_pass"]: result["totalseg_reasons"].append("kidney_coverage_not_detected")
    return result


def prepare_totalsegmentator_weights(config: Mapping[str, Any]) -> set[int]:
    if str(config["task"]) != "total":
        return set()
    os.environ["TOTALSEG_HOME_DIR"] = str(Path(config["home_dir"]).resolve())
    from totalsegmentator.libs import download_pretrained_weights
    kidney_only = bool(config["use_roi_subset"]) and set(config["roi_subset"]) <= {"kidney_left", "kidney_right"}
    for task_id in ([291, 298] if kidney_only else [291, 292, 293, 294, 295]): download_pretrained_weights(task_id)
    return {292, 293, 294, 295} if kidney_only else set()


def read_sidecar_metadata(paths: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("SeriesDescription", "StudyDescription", "ProtocolName", "ContrastBolusAgent", "AcquisitionTime", "SeriesTime", "ContentTime", "SeriesNumber", "AcquisitionNumber"):
            if str(payload.get(key, "") or "").strip(): values.setdefault(key, payload[key])
    return values


def series_metadata(record: Mapping[str, Any], sidecar: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "series_uid": str(record.get("Series UID", record.get("series_uid", "")) or ""),
        "study_uid": str(record.get("Study UID", record.get("study_uid", "")) or ""),
        "series_description": str(record.get("Series Description", record.get("series_description", "")) or ""),
        "study_description": str(record.get("Study Description", record.get("study_description", "")) or ""),
        "protocol_name": str(sidecar.get("ProtocolName", record.get("Protocol Name", "")) or ""),
        **{key: str(sidecar.get(key, "") or "") for key in ("SeriesDescription", "StudyDescription", "ProtocolName", "ContrastBolusAgent", "AcquisitionTime", "SeriesTime", "ContentTime", "SeriesNumber", "AcquisitionNumber")},
    }


def phase_from_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    series_text = " ".join(str(row.get(key, "") or "") for key in ("series_description", "SeriesDescription")).upper()
    protocol_text = str(row.get("ProtocolName", row.get("protocol_name", "")) or "").upper()

    def matches(text: str) -> dict[str, list[str]]:
        return {phase: [pattern for pattern in patterns if re.search(pattern, text)] for phase, patterns in PHASE_PATTERNS.items() if any(re.search(pattern, text) for pattern in patterns)}

    series_matches, protocol_matches = matches(series_text), matches(protocol_text)
    protocol_multiphase = bool(re.search(r"\bNC\b", protocol_text) and re.search(r"\b(?:WC|WITH|CONTRAST|DELAY|ARTERIAL|NEPHRO)\b", protocol_text))
    if len(series_matches) == 1:
        phase = next(iter(series_matches))
        return {"explicit_phase": phase, "inferred_phase": phase, "phase_source": "series_description"}
    if len(protocol_matches) == 1 and not protocol_multiphase:
        phase = next(iter(protocol_matches))
        return {"explicit_phase": phase, "inferred_phase": phase, "phase_source": "protocol_name"}
    contrast = str(row.get("ContrastBolusAgent", row.get("contrast_agent", "")) or "").strip()
    if not protocol_multiphase and (contrast or any(re.search(pattern, f"{series_text} {protocol_text}") for pattern in CE_PATTERNS)):
        return {"explicit_phase": "CE_UNSPECIFIED", "inferred_phase": "CE_UNSPECIFIED", "phase_source": "contrast_metadata"}
    return {"explicit_phase": "UNKNOWN", "inferred_phase": "UNKNOWN", "phase_source": "none"}


def order_value(value: Any) -> tuple[int, float | str]:
    text = str(value or "").strip()
    if not text:
        return (1, "")
    try:
        return (0, float(text.replace(":", "")))
    except ValueError:
        return (1, text)


def annotate_phases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = [{**row, **phase_from_metadata(row)} for row in rows]
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(annotated):
        study_uid = str(row.get("study_uid", "") or "")
        if study_uid:
            grouped.setdefault((str(row.get("case_id", "")), study_uid), []).append((index, row))
    for study_rows in grouped.values():
        ordered = sorted(study_rows, key=lambda item: (order_value(item[1].get("AcquisitionTime")), order_value(item[1].get("SeriesNumber")), order_value(item[1].get("AcquisitionNumber")), item[0]))
        nc = [pos for pos, (_, row) in enumerate(ordered) if row["explicit_phase"] == "NC"]
        delays = [pos for pos, (_, row) in enumerate(ordered) if row["explicit_phase"] == "DEL"]
        valid_nc = [pos for pos in nc if not delays or pos < min(delays)]
        if not valid_nc:
            continue
        last_nc = max(valid_nc)
        if delays:
            for pos in range(last_nc + 1, min(delays)):
                index, row = ordered[pos]
                if row["explicit_phase"] == "UNKNOWN" and "RECON" not in str(row.get("series_description", "")).upper(): annotated[index].update(inferred_phase="MAIN_CE_HIGH", phase_source="study_order")
        else:
            for pos in range(last_nc + 1, len(ordered)):
                index, row = ordered[pos]
                if row["explicit_phase"] == "CE_UNSPECIFIED": annotated[index].update(inferred_phase="MAIN_CE_MEDIUM", phase_source="study_order")
    return annotated


def prepare_ct_series(case_id: str, ct_records: list[Mapping[str, Any]], dcm2nii_dir: Path, dcm2niix_config: Mapping[str, Any], prefilter_config: Mapping[str, Any], nifti_qc_config: Mapping[str, Any], totalseg_config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    report_path = dcm2nii_dir.parent / "dicom_prefilter_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_rows, records = [], []
    for index, ct_record in tqdm(list(enumerate(ct_records, start=1)), desc=f"{case_id} CT series", unit="series", leave=False, disable=not sys.stderr.isatty()):
        source_path = Path(str(ct_record.get("File Path", "") or ""))
        headers = read_dicom_headers(source_path) if source_path.is_dir() else []
        summary = summarize_dicom_series(headers, ct_record, prefilter_config) if headers else {"modality": "", "n_slices": 0, "rows": None, "columns": None, "image_type_union": [], "slice_thickness_median": None, "pixel_spacing_row": None, "pixel_spacing_col": None, "normal_z_median": None, "z_spacing_median": None, "z_gap_max": None, "duplicate_z_count": 0, "missing_spatial_tags": True, "prefilter_reasons": ["no_readable_dicom_header"], "prefilter_pass": False}
        converted_path, convert_errors, generated_nifti, generated_json = "", [], [], []
        nifti_qc = {"nifti_qc_pass": False, "nifti_qc_reasons": [], "hu_min": None, "hu_max": None, "hu_p1": None, "hu_p99": None, "hu_range_p99_p1": None}
        totalseg_qc = {"totalseg_pass": False, "totalseg_execution_success": False, "totalseg_reasons": [], "totalseg_output_dir": "", "totalseg_detected_rois": [], "totalseg_missing_rois": [], "totalseg_roi_voxels": {}, "totalseg_roi_volumes_ml": {}, "totalseg_total_roi_volume_ml": 0.0}
        if summary["prefilter_pass"]:
            converted_path, convert_errors, generated_nifti, generated_json = convert_dicom_series(source_path, dcm2nii_dir, dcm2niix_config)
            if converted_path and not convert_errors:
                nifti_qc = check_nifti_volume(converted_path, nifti_qc_config)
                if nifti_qc["nifti_qc_pass"]:
                    ct_id = Path(converted_path).name.removesuffix(".nii.gz")
                    totalseg_qc = check_totalsegmentator_rois(converted_path, dcm2nii_dir.parent / "totalsegmentator" / ct_id, totalseg_config)
        metadata = series_metadata(ct_record, read_sidecar_metadata([str(path) for path in generated_json]))
        report_row = {"case_id": case_id, "record_index": index, "source_path": str(source_path), "cacheable": True, **metadata, **summary, "prefilter_reasons": ";".join(summary.get("prefilter_reasons", [])), "converted_path": converted_path, "convert_errors": ";".join(convert_errors), "generated_nifti_files": ";".join(str(path) for path in generated_nifti), "generated_json_files": ";".join(str(path) for path in generated_json), **nifti_qc, "nifti_qc_reasons": ";".join(nifti_qc.get("nifti_qc_reasons", [])), **totalseg_qc, "totalseg_reasons": ";".join(totalseg_qc.get("totalseg_reasons", [])), "totalseg_detected_rois": ";".join(totalseg_qc.get("totalseg_detected_rois", [])), "totalseg_missing_rois": ";".join(totalseg_qc.get("totalseg_missing_rois", [])), "totalseg_roi_voxels": json.dumps(totalseg_qc.get("totalseg_roi_voxels", {})), "totalseg_roi_volumes_ml": json.dumps(totalseg_qc.get("totalseg_roi_volumes_ml", {}))}
        report_rows.append(report_row)
        pd.DataFrame(report_rows).to_csv(report_path, index=False)
        if not converted_path or convert_errors or not nifti_qc["nifti_qc_pass"] or not totalseg_qc["totalseg_pass"]:
            continue
        records.append({"case_id": case_id, **metadata, "ct_id": Path(converted_path).name.removesuffix(".nii.gz"), "ct_path": converted_path, "ct_source_path": str(source_path), "qa_source_path": str(source_path), "input_case_path": str(source_path), "file_exists": Path(converted_path).is_file(), **summary, **nifti_qc, **totalseg_qc, "generated_nifti_files": [str(path) for path in generated_nifti], "generated_json_files": [str(path) for path in generated_json]})
    return records, str(report_path)


def prepare_ct_cases(cases: list[Mapping[str, Any]], ct_qc_dir: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    total_seg_config = dict(config["totalsegmentator"])
    skipped_task_ids = prepare_totalsegmentator_weights(total_seg_config) if cases else set()

    def prepare(case: Mapping[str, Any], device: str) -> tuple[str, list[dict[str, Any]], str]:
        case_id = str(case.get("Case_ID", case.get("case_id", "unknown_case")) or "unknown_case")
        records, report_path = prepare_ct_series(case_id, list(case.get("CT", []) or []), ct_qc_dir / case_id / "dcm2nii", config["dcm2niix"], config["dicom_prefilter"], config["nifti_qc"], {**config["totalsegmentator"], "device": device})
        return case_id, records, report_path

    assignments, _ = split_device_requests(cases, list(total_seg_config["devices"]), workers_per_device=int(total_seg_config["workers_per_gpu"]))
    from totalsegmentator import python_api
    original_download = python_api.download_pretrained_weights
    python_api.download_pretrained_weights = lambda task_id: original_download(task_id) if task_id not in skipped_task_ids else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(assignments))) as executor:
            prepared = list(executor.map(lambda assignment: [prepare(case, str(assignment["physical_device"])) for case in assignment["requests"]], assignments))
    finally:
        python_api.download_pretrained_weights = original_download
    return {case_id: {"series_records": records, "prefilter_report_path": report_path} for group in prepared for case_id, records, report_path in group}


def build_case_summary(case_id: str, records: list[dict[str, Any]], prefilter_report_path: str) -> dict[str, Any]:
    records = annotate_phases(records)
    selected = min(records, key=lambda row: (phase_priority(row), technical_key(row))) if records else None
    selected_ok = bool(selected and pretreatment_pass(selected))
    case_output_dir = Path(prefilter_report_path).parent
    selected_file = ""
    selected_series = {}
    if selected:
        selected_file = str(case_output_dir / f"{safe_identifier(case_id)}.nii.gz")
        shutil.copy2(str(selected["ct_path"]), selected_file)
        selected_series = {
            **selected,
            "ct_path": selected_file,
            "passes_threshold": selected_ok,
            "quality_label": "selected_for_case_qc" if selected_ok else "post_treatment_excluded",
        }
    selected_entry = {
        "case_id": case_id,
        "selected_file": selected_file,
        "selected_source_file": str(selected.get("ct_source_path", "") if selected else ""),
        "selected_input_path": str(selected.get("input_case_path", "") if selected else ""),
        "selected_ct_id": str(selected.get("ct_id", "") if selected else ""),
        "passes_threshold": selected_ok,
    }
    return {
        "case_id": case_id,
        "cacheable": True,
        "series_count": len(records),
        "case_qc_passes_threshold": selected_ok,
        "quality_label": (
            "selected_for_case_qc"
            if selected_ok
            else "post_treatment_excluded"
            if selected
            else "no_valid_series"
        ),
        "errors": (
            []
            if selected_ok
            else ["post_treatment_marker_detected"]
            if selected
            else ["no_valid_candidate_series"]
        ),
        "dicom_prefilter_report_path": prefilter_report_path,
        "series_summaries": [{**row, "selected_by_current_qc": bool(selected and row.get("series_uid") == selected.get("series_uid"))} for row in records],
        "selected_series": selected_series,
        "selected_files": [selected_entry] if selected else [],
        "passed_case_ids": [case_id] if selected_ok else [],
        "filtered_cases": [] if selected_ok else [selected_entry],
    }


def run_ct_qc_cohort(cases: list[Mapping[str, Any]], output_root: str = "", config_dir: str = "") -> dict[str, Any]:
    config = yaml.safe_load((Path(config_dir).expanduser() / "ct_qc.yaml").read_text(encoding="utf-8"))
    ct_qc_dir = Path(output_root) / "ct_qc"
    ct_qc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ct_qc_dir / "cohort_cache.json"
    cases_by_id = {
        str(case.get("Case_ID", case.get("case_id", "unknown_case")) or "unknown_case"): dict(case)
        for case in cases
    }
    case_signatures = {case_id: ct_case_signature(case) for case_id, case in cases_by_id.items()}
    semantic_config_signature = ct_qc_config_signature(config)
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    previous_case_signatures = dict(previous.get("case_signatures", {}) or {})
    config_matches = previous.get("semantic_config_signature") == semantic_config_signature
    summaries: dict[str, dict[str, Any]] = {}
    cases_to_prepare = []
    for case_id, case in cases_by_id.items():
        summary_path = ct_qc_dir / safe_identifier(case_id) / "selection_summary.json"
        if config_matches and previous_case_signatures.get(case_id) == case_signatures[case_id]:
            summary = load_case_summary(summary_path)
            if summary is not None:
                summaries[case_id] = summary
                continue
        remove_path(summary_path.parent)
        cases_to_prepare.append(case)

    for case_id in set(previous_case_signatures) - set(cases_by_id):
        remove_path(ct_qc_dir / safe_identifier(case_id))

    prepared = prepare_ct_cases(cases_to_prepare, ct_qc_dir, config) if cases_to_prepare else {}
    for case_id, item in tqdm(prepared.items(), desc="CT case selection", unit="case", disable=not sys.stderr.isatty()):
        summary = build_case_summary(case_id, item["series_records"], item["prefilter_report_path"])
        path = ct_qc_dir / safe_identifier(case_id) / "selection_summary.json"
        path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[case_id] = summary
    manifest_path.write_text(
        json.dumps(
            {
                "semantic_config_signature": semantic_config_signature,
                "case_signatures": case_signatures,
                "case_ids": sorted(summaries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "case_count": len(cases_by_id),
        "passed_case_ids": [case_id for case_id, summary in summaries.items() if summary.get("case_qc_passes_threshold")],
        "filtered_case_ids": [case_id for case_id, summary in summaries.items() if not summary.get("case_qc_passes_threshold")],
        "selection_summaries": summaries,
    }


def run_ct_qc(case_id: str, item: list[Mapping[str, Any]], output_root: str = "", config_dir: str = "") -> dict[str, Any]:
    result = run_ct_qc_cohort([{"Case_ID": str(case_id), "CT": list(item)}], output_root=output_root, config_dir=config_dir)
    return dict(result.get("selection_summaries", {}).get(str(case_id), {}))
