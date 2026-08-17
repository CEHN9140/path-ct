import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import yaml
from tqdm.auto import tqdm

from tools.radqy.backend.radqy import run_radqy_dataset
from utils.tool_utils import (
    safe_identifier,
    run_command,
    semantic_execution_config,
    split_device_requests,
    to_jsonable,
)


def ct_qc_config_signature(
    config: Mapping[str, Any], *, include_device_indices: bool = False
) -> str:
    signature_config = (
        to_jsonable(dict(config))
        if include_device_indices
        else semantic_execution_config(dict(config))
    )
    return hashlib.sha256(
        json.dumps(signature_config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


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


def dicom_prefilter_reasons(
    series_summary: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    image_types = set(series_summary.get("image_type_union", []) or [])
    excluded_image_types = {
        str(item).upper() for item in config["excluded_image_types"]
    }
    required_image_types = {
        str(item).upper() for item in config["required_image_types"]
    }
    reasons = []

    if series_summary.get("modality") != "CT":
        reasons.append("The series is not a CT acquisition.")
    if image_types & excluded_image_types:
        reasons.append("The image type indicates a non-primary or derived series.")
    if required_image_types and not required_image_types.issubset(image_types):
        reasons.append("The image type is not original primary.")
    if bool(series_summary.get("missing_spatial_tags", False)):
        reasons.append("The series is missing required spatial DICOM tags.")

    thickness = series_summary.get("slice_thickness_median")
    if (
        not isinstance(thickness, (int, float))
        or thickness <= 0
        or thickness > float(config["max_slice_thickness"])
    ):
        reasons.append("The slice thickness is missing or out of range.")

    pixel_spacing = [
        series_summary.get("pixel_spacing_row"),
        series_summary.get("pixel_spacing_col"),
    ]
    if any(
        not isinstance(value, (int, float))
        or value <= 0
        or value > float(config["max_pixel_spacing"])
        for value in pixel_spacing
    ):
        reasons.append("The pixel spacing is missing or out of range.")

    normal_z = series_summary.get("normal_z_median")
    if not isinstance(normal_z, (int, float)) or abs(normal_z) < float(
        config["min_abs_normal_z"]
    ):
        reasons.append("The series orientation is not axial enough.")
    if int(series_summary.get("rows") or 0) < int(config["min_matrix_size"]) or int(
        series_summary.get("columns") or 0
    ) < int(config["min_matrix_size"]):
        reasons.append("The image matrix is too small.")

    z_gap_max = series_summary.get("z_gap_max")
    z_spacing_median = series_summary.get("z_spacing_median")
    if (
        isinstance(z_gap_max, (int, float))
        and isinstance(z_spacing_median, (int, float))
        and z_spacing_median > 0
    ):
        if z_gap_max > float(config["max_z_gap_factor"]) * z_spacing_median:
            reasons.append("The series has a large slice gap.")

    if int(series_summary.get("duplicate_z_count") or 0) > int(
        config["max_duplicate_slices"]
    ):
        reasons.append("The series has too many duplicate slice positions.")

    return reasons


def summarize_dicom_series(
    headers: list[Any], ct_record: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    first = headers[0] if headers else None
    image_type_union = sorted(
        {
            str(item).upper()
            for header in headers
            for item in (
                [getattr(header, "ImageType", [])]
                if isinstance(getattr(header, "ImageType", []), str)
                else list(getattr(header, "ImageType", []) or [])
            )
        }
    )

    rows = pd.to_numeric(
        [getattr(header, "Rows", np.nan) for header in headers], errors="coerce"
    )
    columns = pd.to_numeric(
        [getattr(header, "Columns", np.nan) for header in headers], errors="coerce"
    )
    thickness = pd.to_numeric(
        [getattr(header, "SliceThickness", np.nan) for header in headers],
        errors="coerce",
    )
    spacing_rows, spacing_cols, normal_z_values, z_positions = [], [], [], []
    missing_spatial_tags = False

    for header in headers:
        pixel_spacing = pd.to_numeric(
            list(getattr(header, "PixelSpacing", []) or []), errors="coerce"
        )
        if len(pixel_spacing) == 2 and not pd.isna(pixel_spacing).any():
            spacing_rows.append(float(pixel_spacing[0]))
            spacing_cols.append(float(pixel_spacing[1]))

        orientation = pd.to_numeric(
            list(getattr(header, "ImageOrientationPatient", []) or []), errors="coerce"
        )
        position = pd.to_numeric(
            list(getattr(header, "ImagePositionPatient", []) or []), errors="coerce"
        )
        if (
            len(orientation) != 6
            or len(position) != 3
            or pd.isna(orientation).any()
            or pd.isna(position).any()
        ):
            missing_spatial_tags = True
            continue
        normal = np.cross(
            np.asarray(orientation[:3], dtype=float),
            np.asarray(orientation[3:], dtype=float),
        )
        normal_z_values.append(float(normal[2]))
        z_positions.append(float(np.dot(np.asarray(position, dtype=float), normal)))

    rounded_z = np.asarray([round(value, 3) for value in z_positions], dtype=float)
    unique_z = np.unique(rounded_z) if len(rounded_z) else np.asarray([])
    dz = np.diff(np.sort(unique_z)) if len(unique_z) > 1 else np.asarray([])
    z_spacing_median = float(np.median(dz)) if len(dz) else None
    z_gap_max = float(np.max(dz)) if len(dz) else None
    duplicate_z_count = int(len(rounded_z) - len(unique_z)) if len(rounded_z) else 0
    normal_z_median = float(np.median(normal_z_values)) if normal_z_values else None

    summary = {
        "modality": str(
            getattr(first, "Modality", "") if first is not None else ""
        ).upper(),
        "n_slices": len(headers),
        "rows": None
        if pd.isna(rows).all()
        else int(pd.Series(rows).dropna().mode().iloc[0]),
        "columns": None
        if pd.isna(columns).all()
        else int(pd.Series(columns).dropna().mode().iloc[0]),
        "image_type_union": image_type_union,
        "slice_thickness_median": None
        if pd.isna(thickness).all()
        else float(np.nanmedian(thickness)),
        "pixel_spacing_row": float(np.median(spacing_rows)) if spacing_rows else None,
        "pixel_spacing_col": float(np.median(spacing_cols)) if spacing_cols else None,
        "normal_z_median": normal_z_median,
        "z_spacing_median": z_spacing_median,
        "z_gap_max": z_gap_max,
        "duplicate_z_count": duplicate_z_count,
        "missing_spatial_tags": missing_spatial_tags or not len(z_positions),
    }
    reasons = dicom_prefilter_reasons(summary, config)
    summary.update(
        {
            "prefilter_reasons": reasons,
            "prefilter_pass": not reasons,
        }
    )
    return summary


def convert_dicom_series(
    series_path: Path, output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, list[str], list[str], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_nifti = {path.resolve() for path in output_dir.glob("*.nii.gz")}
    existing_json = {path.resolve() for path in output_dir.glob("*.json")}
    command = [
        str(config["bin"]),
        "-z",
        str(config["gzip_output"]),
        "-b",
        str(config["bids_sidecar"]),
        "-ba",
        str(config["anonymize_bids"]),
        "-f",
        str(config["filename"]),
        "-o",
        str(output_dir),
        str(series_path),
    ]
    completed = run_command(command)
    if completed.returncode != 0:
        return (
            "",
            [
                completed.stderr.strip()
                or completed.stdout.strip()
                or "dcm2niix conversion failed."
            ],
            [],
            [],
        )
    generated_nifti = sorted(
        path
        for path in output_dir.glob("*.nii.gz")
        if path.resolve() not in existing_nifti
    )
    generated_json = sorted(
        path
        for path in output_dir.glob("*.json")
        if path.resolve() not in existing_json
    )
    usable_nifti = [
        path
        for path in generated_nifti
        if not any(
            token in path.name.lower()
            for token in ["_eq_", "mask", "seg", "label", "derived"]
        )
    ]
    if not usable_nifti:
        return (
            "",
            ["dcm2niix did not create a usable .nii.gz file."],
            [str(path) for path in generated_nifti],
            [str(path) for path in generated_json],
        )
    selected = max(usable_nifti, key=lambda path: path.stat().st_size)
    return (
        str(selected),
        [],
        [str(path) for path in generated_nifti],
        [str(path) for path in generated_json],
    )


def check_nifti_volume(
    nifti_path: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "nifti_qc_pass": False,
        "nifti_qc_reasons": [],
        "hu_min": None,
        "hu_max": None,
        "hu_p1": None,
        "hu_p99": None,
        "hu_range_p99_p1": None,
    }
    try:
        image = sitk.ReadImage(str(nifti_path))
        if image.GetDimension() != 3:
            result["nifti_qc_reasons"].append("The NIfTI image is not a 3D volume.")
            return result
        values = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    except Exception:
        result["nifti_qc_reasons"].append(
            "The NIfTI image cannot be read by SimpleITK."
        )
        return result

    values = values[np.isfinite(values)]
    if values.size == 0:
        result["nifti_qc_reasons"].append("The NIfTI image has no finite voxel values.")
        return result

    hu_p1 = float(np.percentile(values, 1))
    hu_p99 = float(np.percentile(values, 99))
    result.update(
        {
            "hu_min": float(np.min(values)),
            "hu_max": float(np.max(values)),
            "hu_p1": hu_p1,
            "hu_p99": hu_p99,
            "hu_range_p99_p1": hu_p99 - hu_p1,
        }
    )
    if result["hu_min"] < float(config["hu_min_allowed"]) or result["hu_max"] > float(
        config["hu_max_allowed"]
    ):
        result["nifti_qc_reasons"].append(
            "The HU value range is outside the allowed range."
        )
    if result["hu_range_p99_p1"] < float(config["hu_range_p99_p1_min"]):
        result["nifti_qc_reasons"].append("The HU p99-p1 range is too small.")

    result["nifti_qc_pass"] = not result["nifti_qc_reasons"]
    return result


def check_totalsegmentator_rois(
    nifti_path: str | Path, output_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rois = [str(item) for item in config["roi_subset"]]
    result = {
        "totalseg_pass": False,
        "totalseg_execution_success": False,
        "totalseg_reasons": [],
        "totalseg_output_dir": str(output_dir),
        "totalseg_overlay_png": "",
        "totalseg_detected_rois": [],
        "totalseg_missing_rois": [],
        "totalseg_roi_voxels": {},
        "totalseg_roi_volumes_ml": {},
        "totalseg_total_roi_volume_ml": 0.0,
    }
    remove_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TOTALSEG_HOME_DIR"] = str(Path(config["home_dir"]).resolve())
    try:
        from totalsegmentator.python_api import totalsegmentator

        totalsegmentator(
            input=Path(nifti_path),
            output=output_dir,
            task=str(config["task"]),
            roi_subset=rois if bool(config["use_roi_subset"]) else None,
            quiet=True,
            device=str(config["device"]).replace("cuda:", "gpu:"),
        )
    except Exception as exc:
        result["totalseg_reasons"].append(
            f"TotalSegmentator failed to run: {type(exc).__name__}: {exc}"
        )
        return result
    result["totalseg_execution_success"] = True

    ct_image = sitk.ReadImage(str(nifti_path))
    ct_array = sitk.GetArrayFromImage(ct_image)
    voxel_volume_ml = float(np.prod(ct_image.GetSpacing()) / 1000.0)
    roi_masks = {}
    for roi in rois:
        mask_path = output_dir / f"{roi}.nii.gz"
        voxels = 0
        if mask_path.is_file():
            try:
                mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
                voxels = int(np.count_nonzero(mask_array))
                roi_masks[roi] = mask_array
            except Exception:
                voxels = 0
        result["totalseg_roi_voxels"][roi] = voxels
        result["totalseg_roi_volumes_ml"][roi] = float(voxels * voxel_volume_ml)
        result["totalseg_total_roi_volume_ml"] += result["totalseg_roi_volumes_ml"][roi]
        if voxels > 0:
            result["totalseg_detected_rois"].append(roi)
        else:
            result["totalseg_missing_rois"].append(roi)

    if roi_masks:
        combined_mask = np.logical_or.reduce(list(roi_masks.values()))
        slice_index = int(np.argmax(combined_mask.sum(axis=(1, 2))))
        image_slice = ct_array[slice_index].astype(np.float32)
        low, high = np.percentile(image_slice[np.isfinite(image_slice)], [1, 99])
        gray = np.clip((image_slice - low) / max(high - low, 1), 0, 1)
        rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        colors = [(255, 64, 64), (64, 160, 255), (64, 255, 128), (255, 220, 64)]
        for color, roi in zip(colors, rois):
            if roi not in roi_masks:
                continue
            mask_slice = roi_masks[roi][slice_index]
            rgb[mask_slice] = (
                0.55 * rgb[mask_slice] + 0.45 * np.asarray(color)
            ).astype(np.uint8)
        overlay_path = output_dir / "roi_overlay.png"
        from PIL import Image

        Image.fromarray(rgb).save(overlay_path)
        result["totalseg_overlay_png"] = str(overlay_path)

    min_single_roi_volume_ml = float(config["min_single_roi_volume_ml"])
    if any(
        float(value or 0) > min_single_roi_volume_ml
        for value in result["totalseg_roi_volumes_ml"].values()
    ):
        result["totalseg_pass"] = True
    else:
        result["totalseg_reasons"].append("No target ROI is large enough.")
    return result


def prepare_totalsegmentator_weights(config: Mapping[str, Any]) -> set[int]:
    if str(config["task"]) != "total":
        return set()
    os.environ["TOTALSEG_HOME_DIR"] = str(Path(config["home_dir"]).resolve())
    from totalsegmentator.libs import download_pretrained_weights

    kidney_only = bool(config["use_roi_subset"]) and set(config["roi_subset"]) <= {
        "kidney_left",
        "kidney_right",
    }
    task_ids = [291, 298] if kidney_only else [291, 292, 293, 294, 295]
    for task_id in task_ids:
        download_pretrained_weights(task_id)
    return {292, 293, 294, 295} if kidney_only else set()


def prepare_ct_series(
    case_id: str,
    ct_records: list[Mapping[str, Any]],
    dcm2nii_dir: Path,
    dcm2niix_config: Mapping[str, Any],
    prefilter_config: Mapping[str, Any],
    nifti_qc_config: Mapping[str, Any],
    totalseg_config: Mapping[str, Any],
    reuse_cached_rows: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    report_path = dcm2nii_dir.parent / "dicom_prefilter_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = {}
    if report_path.is_file():
        try:
            existing_table = pd.read_csv(report_path).fillna("")
            existing_rows = {
                str(row["source_path"]): {
                    str(key): value for key, value in row.to_dict().items()
                }
                for _, row in existing_table.iterrows()
                if str(row.get("source_path", "")).strip()
            }
        except Exception:
            existing_rows = {}
    report_rows = []
    series_records = []

    show_progress = sys.stderr.isatty()
    for index, ct_record in tqdm(
        list(enumerate(ct_records, start=1)),
        desc=f"{case_id} CT series",
        unit="series",
        leave=False,
        disable=not show_progress,
    ):
        source_path = Path(str(ct_record.get("File Path", "") or ""))
        cached_row = existing_rows.get(str(source_path))
        if cached_row:
            cached_full_pass = all(
                str(cached_row.get(key, "")).strip().lower() == "true"
                for key in ["prefilter_pass", "nifti_qc_pass", "totalseg_pass"]
            )
            converted_path = str(cached_row.get("converted_path", "") or "")
            totalseg_output_dir = str(cached_row.get("totalseg_output_dir", "") or "")

            cached_files_exist = (
                converted_path
                and Path(converted_path).is_file()
                and totalseg_output_dir
                and Path(totalseg_output_dir).is_dir()
            )

            if reuse_cached_rows and cached_full_pass and cached_files_exist:
                cached_row["record_index"] = index
                cached_row["cacheable"] = True
                report_rows.append(cached_row)
                pd.DataFrame(report_rows).to_csv(report_path, index=False)
                converted_file = Path(converted_path)
                converted_name = converted_file.name
                ct_id = (
                    converted_name[:-7]
                    if converted_name.endswith(".nii.gz")
                    else converted_file.stem
                )
                subject_id = safe_identifier(f"{case_id}__{ct_id.replace('.', '_')}")
                try:
                    roi_voxels = json.loads(
                        str(cached_row.get("totalseg_roi_voxels", "") or "{}")
                    )
                except json.JSONDecodeError:
                    roi_voxels = {}
                try:
                    roi_volumes = json.loads(
                        str(cached_row.get("totalseg_roi_volumes_ml", "") or "{}")
                    )
                except json.JSONDecodeError:
                    roi_volumes = {}
                n_slices = pd.to_numeric(cached_row.get("n_slices"), errors="coerce")
                slice_thickness = pd.to_numeric(
                    cached_row.get("slice_thickness_median"), errors="coerce"
                )
                numeric_fields = {
                    key: pd.to_numeric(cached_row.get(key), errors="coerce")
                    for key in [
                        "pixel_spacing_row",
                        "pixel_spacing_col",
                        "normal_z_median",
                        "z_spacing_median",
                        "z_gap_max",
                        "hu_min",
                        "hu_max",
                        "hu_p1",
                        "hu_p99",
                        "hu_range_p99_p1",
                        "totalseg_total_roi_volume_ml",
                    ]
                }
                series_records.append(
                    {
                        "case_id": case_id,
                        "ct_id": ct_id,
                        "ct_path": converted_path,
                        "ct_source_path": str(source_path),
                        "qa_source_path": str(source_path),
                        "input_case_path": str(source_path),
                        "file_exists": converted_file.exists(),
                        "file_size_mb": float(converted_file.stat().st_size / 1024**2),
                        "radqy_subject_id": subject_id,
                        "radqy_link_name": f"{subject_id}{''.join(converted_file.suffixes)}",
                        "n_slices": int(n_slices) if pd.notna(n_slices) else None,
                        "slice_thickness_median": float(slice_thickness)
                        if pd.notna(slice_thickness)
                        else None,
                        **{
                            key: float(value)
                            for key, value in numeric_fields.items()
                            if pd.notna(value)
                        },
                        "prefilter_pass": True,
                        "nifti_qc_pass": True,
                        "totalseg_pass": True,
                        "image_type_union": str(
                            cached_row.get("image_type_union", "") or ""
                        ).split(";")
                        if cached_row.get("image_type_union")
                        else [],
                        "prefilter_reasons": str(
                            cached_row.get("prefilter_reasons", "") or ""
                        ).split(";")
                        if cached_row.get("prefilter_reasons")
                        else [],
                        "nifti_qc_reasons": str(
                            cached_row.get("nifti_qc_reasons", "") or ""
                        ).split(";")
                        if cached_row.get("nifti_qc_reasons")
                        else [],
                        "totalseg_reasons": str(
                            cached_row.get("totalseg_reasons", "") or ""
                        ).split(";")
                        if cached_row.get("totalseg_reasons")
                        else [],
                        "totalseg_output_dir": totalseg_output_dir,
                        "totalseg_overlay_png": str(
                            cached_row.get("totalseg_overlay_png", "") or ""
                        ),
                        "totalseg_detected_rois": str(
                            cached_row.get("totalseg_detected_rois", "") or ""
                        ).split(";")
                        if cached_row.get("totalseg_detected_rois")
                        else [],
                        "totalseg_missing_rois": str(
                            cached_row.get("totalseg_missing_rois", "") or ""
                        ).split(";")
                        if cached_row.get("totalseg_missing_rois")
                        else [],
                        "totalseg_roi_voxels": roi_voxels,
                        "totalseg_roi_volumes_ml": roi_volumes,
                        "generated_nifti_files": str(
                            cached_row.get("generated_nifti_files", "") or ""
                        ).split(";")
                        if cached_row.get("generated_nifti_files")
                        else [],
                        "generated_json_files": str(
                            cached_row.get("generated_json_files", "") or ""
                        ).split(";")
                        if cached_row.get("generated_json_files")
                        else [],
                    }
                )
                continue
            cached_cacheable = (
                str(cached_row.get("cacheable", "")).strip().lower() == "true"
            )
            if reuse_cached_rows and not cached_full_pass and cached_cacheable:
                cached_row["record_index"] = index
                report_rows.append(cached_row)
                pd.DataFrame(report_rows).to_csv(report_path, index=False)
                continue

        headers = read_dicom_headers(source_path) if source_path.is_dir() else []
        series_summary = (
            summarize_dicom_series(headers, ct_record, prefilter_config)
            if headers
            else {
                "modality": "",
                "n_slices": 0,
                "rows": None,
                "columns": None,
                "image_type_union": [],
                "slice_thickness_median": None,
                "pixel_spacing_row": None,
                "pixel_spacing_col": None,
                "normal_z_median": None,
                "z_spacing_median": None,
                "z_gap_max": None,
                "duplicate_z_count": 0,
                "missing_spatial_tags": True,
                "prefilter_reasons": ["No readable DICOM header was found."],
                "prefilter_pass": False,
            }
        )
        converted_path, convert_errors, generated_nifti, generated_json = (
            "",
            [],
            [],
            [],
        )
        nifti_qc = {
            "nifti_qc_pass": None,
            "nifti_qc_reasons": [],
            "hu_min": None,
            "hu_max": None,
            "hu_p1": None,
            "hu_p99": None,
            "hu_range_p99_p1": None,
        }
        totalseg_qc = {
            "totalseg_pass": None,
            "totalseg_execution_success": False,
            "totalseg_reasons": [],
            "totalseg_output_dir": "",
            "totalseg_overlay_png": "",
            "totalseg_detected_rois": [],
            "totalseg_missing_rois": [],
            "totalseg_roi_voxels": {},
            "totalseg_roi_volumes_ml": {},
            "totalseg_total_roi_volume_ml": None,
        }
        if series_summary["prefilter_pass"]:
            converted_path, convert_errors, generated_nifti, generated_json = (
                convert_dicom_series(source_path, dcm2nii_dir, dcm2niix_config)
            )
            if converted_path and not convert_errors:
                nifti_qc = check_nifti_volume(converted_path, nifti_qc_config)
                if nifti_qc["nifti_qc_pass"]:
                    converted_name = Path(converted_path).name
                    ct_id = (
                        converted_name[:-7]
                        if converted_name.endswith(".nii.gz")
                        else Path(converted_name).stem
                    )
                    totalseg_qc = check_totalsegmentator_rois(
                        converted_path,
                        dcm2nii_dir.parent / "totalsegmentator" / ct_id,
                        totalseg_config,
                    )

        cacheable = bool(
            not series_summary["prefilter_pass"]
            or (
                converted_path
                and not convert_errors
                and (
                    not nifti_qc["nifti_qc_pass"]
                    or totalseg_qc["totalseg_execution_success"]
                )
            )
        )
        report_row = {
            "case_id": case_id,
            "record_index": index,
            "source_path": str(source_path),
            "cacheable": cacheable,
            **series_summary,
            "image_type_union": ";".join(
                series_summary.get("image_type_union", []) or []
            ),
            "prefilter_reasons": ";".join(
                series_summary.get("prefilter_reasons", []) or []
            ),
            "converted_path": converted_path,
            "convert_errors": ";".join(convert_errors),
            "generated_nifti_files": ";".join(generated_nifti),
            "generated_json_files": ";".join(generated_json),
            **nifti_qc,
            "nifti_qc_reasons": ";".join(nifti_qc.get("nifti_qc_reasons", []) or []),
            **totalseg_qc,
            "totalseg_reasons": ";".join(totalseg_qc.get("totalseg_reasons", []) or []),
            "totalseg_detected_rois": ";".join(
                totalseg_qc.get("totalseg_detected_rois", []) or []
            ),
            "totalseg_missing_rois": ";".join(
                totalseg_qc.get("totalseg_missing_rois", []) or []
            ),
            "totalseg_roi_voxels": json.dumps(
                totalseg_qc.get("totalseg_roi_voxels", {}) or {}, ensure_ascii=False
            ),
            "totalseg_roi_volumes_ml": json.dumps(
                totalseg_qc.get("totalseg_roi_volumes_ml", {}) or {}, ensure_ascii=False
            ),
        }
        report_rows.append(report_row)
        pd.DataFrame(report_rows).to_csv(report_path, index=False)
        if (
            not converted_path
            or convert_errors
            or not nifti_qc["nifti_qc_pass"]
            or not totalseg_qc["totalseg_pass"]
        ):
            continue

        converted_name = Path(converted_path).name
        ct_id = (
            converted_name[:-7]
            if converted_name.endswith(".nii.gz")
            else Path(converted_name).stem
        )
        subject_id = safe_identifier(f"{case_id}__{ct_id.replace('.', '_')}")
        converted_file = Path(converted_path)
        file_size_mb = (
            converted_file.stat().st_size / 1024**2 if converted_file.exists() else 0.0
        )
        series_records.append(
            {
                "case_id": case_id,
                "ct_id": ct_id,
                "ct_path": converted_path,
                "ct_source_path": str(source_path),
                "qa_source_path": str(source_path),
                "input_case_path": str(source_path),
                "file_exists": converted_file.exists(),
                "file_size_mb": float(file_size_mb),
                "radqy_subject_id": subject_id,
                "radqy_link_name": f"{subject_id}{''.join(Path(converted_path).suffixes)}",
                **series_summary,
                **nifti_qc,
                **totalseg_qc,
                "image_type_union": list(
                    series_summary.get("image_type_union", []) or []
                ),
                "prefilter_reasons": list(
                    series_summary.get("prefilter_reasons", []) or []
                ),
                "nifti_qc_reasons": list(nifti_qc.get("nifti_qc_reasons", []) or []),
                "totalseg_reasons": list(totalseg_qc.get("totalseg_reasons", []) or []),
                "generated_nifti_files": generated_nifti,
                "generated_json_files": generated_json,
            }
        )

    return series_records, str(report_path)


def prepare_ct_cases(
    cases: list[Mapping[str, Any]],
    ct_qc_dir: Path,
    config: Mapping[str, Any],
    *,
    reuse_cached_rows: bool,
) -> dict[str, dict[str, Any]]:
    total_seg_config = dict(config["totalsegmentator"])
    skipped_task_ids = set()
    if cases and "home_dir" in total_seg_config:
        skipped_task_ids = prepare_totalsegmentator_weights(total_seg_config)

    def prepare(
        case: Mapping[str, Any], device: str
    ) -> tuple[str, list[dict[str, Any]], str]:
        case_id = str(
            case.get("Case_ID", "") or case.get("case_id", "") or "unknown_case"
        )
        case_output_dir = ct_qc_dir / case_id
        dcm2nii_dir = case_output_dir / "dcm2nii"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        series_records, report_path = prepare_ct_series(
            case_id,
            list(case.get("CT", []) or []),
            dcm2nii_dir,
            dict(config["dcm2niix"]),
            dict(config["dicom_prefilter"]),
            dict(config["nifti_qc"]),
            {**dict(config["totalsegmentator"]), "device": device},
            reuse_cached_rows=reuse_cached_rows,
        )
        return case_id, series_records, report_path

    devices = list(total_seg_config["devices"])
    assignments, _ = split_device_requests(
        cases,
        devices,
        workers_per_device=int(total_seg_config["workers_per_gpu"]),
    )

    def prepare_group(assignment: Mapping[str, Any]):
        return [
            prepare(case, str(assignment["physical_device"]))
            for case in assignment["requests"]
        ]

    from totalsegmentator import python_api

    original_download = python_api.download_pretrained_weights

    def download_required_weights(task_id):
        if task_id not in skipped_task_ids:
            original_download(task_id)

    python_api.download_pretrained_weights = download_required_weights
    try:
        if len(assignments) == 1:
            prepared_groups = [prepare_group(assignments[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
                prepared_groups = list(executor.map(prepare_group, assignments))
    finally:
        python_api.download_pretrained_weights = original_download
    prepared = [item for group in prepared_groups for item in group]
    return {
        case_id: {"series_records": records, "prefilter_report_path": report_path}
        for case_id, records, report_path in prepared
    }


def radqy_robust_z_flags(
    table: pd.DataFrame,
    *,
    high_bad: list[str],
    low_bad: list[str],
    cutoff: float,
    fail_threshold: int,
) -> pd.DataFrame:
    result = table.copy()
    bad_metrics_by_row = [[] for _ in range(len(result))]
    robust_values = [{} for _ in range(len(result))]
    for metric in list(high_bad) + list(low_bad):
        if metric not in result.columns:
            continue
        values = pd.to_numeric(result[metric], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        if mad == 0:
            continue
        robust_z = (values - median) / (1.4826 * mad)
        for index, value in robust_z.items():
            if pd.isna(value):
                continue
            robust_values[index][metric] = float(value)
            is_bad = value > cutoff if metric in high_bad else value < -cutoff
            if is_bad:
                bad_metrics_by_row[index].append(metric)
    bad_counts = [len(items) for items in bad_metrics_by_row]
    result["iqm_bad_metrics"] = pd.Series(bad_metrics_by_row, dtype=object)
    result["iqm_bad_count"] = bad_counts
    result["iqm_robust_z"] = pd.Series(robust_values, dtype=object)
    result["radqy_iqm_pass"] = pd.Series(
        [count < int(fail_threshold) for count in bad_counts], dtype=object
    )
    return result


def link_dicom_series_for_radqy(source_dir: Path, target_dir: Path) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for index, source_file in enumerate(
        sorted(path for path in source_dir.rglob("*") if path.is_file()), start=1
    ):
        target_file = target_dir / f"{index:05d}_{safe_identifier(source_file.name)}"
        try:
            target_file.symlink_to(source_file)
        except OSError:
            shutil.copy2(source_file, target_file)
        linked += 1
    return linked > 0


def read_radqy_results(results_tsv: Path) -> pd.DataFrame:
    if results_tsv.is_file():
        try:
            skiprows = 0
            with results_tsv.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        break
                    skiprows += 1
            table = pd.read_csv(results_tsv, sep="\t", skiprows=skiprows).fillna("")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
    else:
        table = pd.DataFrame()
    return table


def match_radqy_rows(
    series_records: list[dict[str, Any]], results_tsv: Path
) -> list[dict[str, Any]]:
    table = read_radqy_results(results_tsv)
    rows = []
    participant_column = "Participant (topfolder--subfolder--patient ID)"
    for record in series_records:
        subject_id = str(record.get("radqy_subject_id", "") or "")
        row = {}
        if not table.empty and participant_column in table.columns:
            matches = table[
                table[participant_column]
                .astype(str)
                .str.contains(subject_id, regex=False)
            ]
            if not matches.empty:
                row = {
                    str(key): value for key, value in matches.iloc[0].to_dict().items()
                }
        rows.append({**record, **row, "series_id": subject_id})
    return rows


def radqy_results_cover_series(
    series_records: list[dict[str, Any]], results_tsv: Path, metrics: Sequence[str]
) -> bool:
    if not series_records or not results_tsv.is_file():
        return False
    matched = match_radqy_rows(series_records, results_tsv)
    for row in matched:
        if any(
            pd.isna(pd.to_numeric(row.get(metric), errors="coerce"))
            for metric in metrics
        ):
            return False
    return True


def build_case_summary(
    case_id: str,
    series_records: list[dict[str, Any]],
    prefilter_report_path: str,
    results_tsv_path: str,
) -> dict[str, Any]:
    ranked_series = []
    for series in series_records:
        errors = []
        required = ["CV", "CJV", "EFC", "PSNR"]
        metrics_available = all(
            pd.notna(pd.to_numeric(series.get(metric), errors="coerce"))
            for metric in required
        )
        if not series.get("file_exists"):
            errors.append("ct_file_missing")
        if not metrics_available:
            errors.append("radqy_metrics_incomplete")
        if not bool(series.get("radqy_iqm_pass", False)):
            errors.append("radqy_iqm_outlier")
        ranked_series.append(
            {**series, "metrics_available": metrics_available, "series_errors": errors}
        )

    ranked_series.sort(
        key=lambda item: (
            int(item.get("iqm_bad_count") or 999),
            float(pd.to_numeric(item.get("CV"), errors="coerce"))
            if pd.notna(pd.to_numeric(item.get("CV"), errors="coerce"))
            else float("inf"),
            float(pd.to_numeric(item.get("CJV"), errors="coerce"))
            if pd.notna(pd.to_numeric(item.get("CJV"), errors="coerce"))
            else float("inf"),
            float(pd.to_numeric(item.get("EFC"), errors="coerce"))
            if pd.notna(pd.to_numeric(item.get("EFC"), errors="coerce"))
            else float("inf"),
            -float(pd.to_numeric(item.get("PSNR"), errors="coerce"))
            if pd.notna(pd.to_numeric(item.get("PSNR"), errors="coerce"))
            else float("inf"),
            item.get("slice_thickness_median")
            if isinstance(item.get("slice_thickness_median"), (int, float))
            else float("inf"),
            -int(item.get("n_slices") or 0),
        )
    )
    for index, series in enumerate(ranked_series, start=1):
        series["selection_rank"] = index
    selected = next(
        (
            dict(series)
            for series in ranked_series
            if series.get("metrics_available") and series.get("radqy_iqm_pass")
        ),
        {},
    )
    case_output_dir = Path(prefilter_report_path).parent
    report = pd.DataFrame()
    if Path(prefilter_report_path).is_file():
        try:
            report = pd.read_csv(prefilter_report_path).fillna("")
        except pd.errors.EmptyDataError:
            pass
    cacheable = bool(
        report.empty
        or (
            "cacheable" in report.columns
            and report["cacheable"]
            .map(lambda value: str(value).strip().lower() == "true")
            .all()
        )
    )
    selected_file = ""
    if selected:
        selected_candidate_file = str(selected.get("ct_path", "") or "")
        if selected_candidate_file and Path(selected_candidate_file).exists():
            selected_file = str(case_output_dir / f"{safe_identifier(case_id)}.nii.gz")
            shutil.copy2(selected_candidate_file, selected_file)
        selected_series = {
            "case_id": case_id,
            "ct_id": selected.get("ct_id"),
            "ct_path": selected_file or selected_candidate_file,
            "CV": selected.get("CV"),
            "CJV": selected.get("CJV"),
            "EFC": selected.get("EFC"),
            "PSNR": selected.get("PSNR"),
            "iqm_robust_z": selected.get("iqm_robust_z", {}),
            "iqm_bad_metrics": selected.get("iqm_bad_metrics", []),
            "iqm_bad_count": selected.get("iqm_bad_count"),
            "radqy_iqm_pass": selected.get("radqy_iqm_pass"),
            "selection_rank": selected.get("selection_rank"),
            "n_slices": selected.get("n_slices"),
            "slice_thickness_median": selected.get("slice_thickness_median"),
            "prefilter_reasons": selected.get("prefilter_reasons", []),
            "hu_min": selected.get("hu_min"),
            "hu_max": selected.get("hu_max"),
            "hu_p1": selected.get("hu_p1"),
            "hu_p99": selected.get("hu_p99"),
            "hu_range_p99_p1": selected.get("hu_range_p99_p1"),
            "nifti_qc_reasons": selected.get("nifti_qc_reasons", []),
            "totalseg_pass": selected.get("totalseg_pass"),
            "totalseg_detected_rois": selected.get("totalseg_detected_rois", []),
            "totalseg_missing_rois": selected.get("totalseg_missing_rois", []),
            "totalseg_roi_voxels": selected.get("totalseg_roi_voxels", {}),
            "totalseg_roi_volumes_ml": selected.get("totalseg_roi_volumes_ml", {}),
            "totalseg_total_roi_volume_ml": selected.get(
                "totalseg_total_roi_volume_ml"
            ),
            "totalseg_overlay_png": selected.get("totalseg_overlay_png", ""),
            "totalseg_reasons": selected.get("totalseg_reasons", []),
            "passes_threshold": True,
            "quality_label": "selected_for_case_qc",
        }
    else:
        selected_series = {}

    selected_entry = {
        "case_id": case_id,
        "selected_file": selected_file,
        "selected_source_file": str(
            selected.get("qa_source_path") or selected.get("ct_source_path") or ""
        ),
        "selected_input_path": str(selected.get("input_case_path") or ""),
        "selected_ct_id": str(selected.get("ct_id", "") or ""),
        "passes_threshold": bool(selected),
    }
    if selected:
        errors = []
    elif series_records:
        errors = ["no_valid_radqy_series"]
    elif not report.empty:
        if (
            "totalseg_pass" in report.columns
            and (report["totalseg_pass"] == False).any()
        ):
            errors = ["no_valid_totalsegmentator_series"]
        elif (
            "nifti_qc_pass" in report.columns
            and (report["nifti_qc_pass"] == False).any()
        ):
            errors = ["no_valid_nifti_series"]
        else:
            errors = ["no_valid_dicom_series"]
    else:
        errors = ["no_valid_dicom_series"]

    selection_summary = {
        "case_id": case_id,
        "cacheable": cacheable,
        "series_count": len(ranked_series),
        "case_qc_passes_threshold": bool(selected),
        "failed_metrics": [],
        "quality_label": "selected_for_case_qc" if selected else "no_valid_series",
        "errors": errors,
        "results_tsv_path": results_tsv_path,
        "dicom_prefilter_report_path": prefilter_report_path,
        "series_summaries": [
            {
                "ct_id": series.get("ct_id"),
                "ct_path": series.get("ct_path"),
                "selection_rank": series.get("selection_rank"),
                "prefilter_pass": series.get("prefilter_pass"),
                "prefilter_reasons": series.get("prefilter_reasons", []),
                "nifti_qc_pass": series.get("nifti_qc_pass"),
                "nifti_qc_reasons": series.get("nifti_qc_reasons", []),
                "totalseg_pass": series.get("totalseg_pass"),
                "totalseg_detected_rois": series.get("totalseg_detected_rois", []),
                "totalseg_missing_rois": series.get("totalseg_missing_rois", []),
                "totalseg_total_roi_volume_ml": series.get(
                    "totalseg_total_roi_volume_ml"
                ),
                "totalseg_overlay_png": series.get("totalseg_overlay_png", ""),
                "CV": series.get("CV"),
                "CJV": series.get("CJV"),
                "EFC": series.get("EFC"),
                "PSNR": series.get("PSNR"),
                "iqm_robust_z": series.get("iqm_robust_z", {}),
                "iqm_bad_metrics": series.get("iqm_bad_metrics", []),
                "iqm_bad_count": series.get("iqm_bad_count"),
                "radqy_iqm_pass": series.get("radqy_iqm_pass"),
                "series_errors": series.get("series_errors", []),
            }
            for series in ranked_series
        ],
        "selected_series": selected_series,
        "case_rankings": [
            {
                "case_id": case_id,
                "ranked_files": [
                    {
                        "ct_id": str(series.get("ct_id", "") or ""),
                        "ct_path": str(series.get("ct_path", "") or ""),
                        "CV": series.get("CV"),
                        "CJV": series.get("CJV"),
                        "EFC": series.get("EFC"),
                        "PSNR": series.get("PSNR"),
                        "iqm_bad_count": series.get("iqm_bad_count"),
                    }
                    for series in ranked_series
                ],
            }
        ],
        "selected_files": [selected_entry] if selected else [],
        "passed_case_ids": [case_id] if selected else [],
        "filtered_cases": [] if selected else [selected_entry],
    }
    summary_path = case_output_dir / "selection_summary.json"
    summary_path.write_text(
        json.dumps(to_jsonable(selection_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selection_summary


def run_ct_qc_cohort(
    cases: list[Mapping[str, Any]], output_root: str = "", config_dir: str = ""
) -> dict[str, Any]:
    config = yaml.safe_load(
        (Path(config_dir).expanduser() / "ct_qc.yaml").read_text(encoding="utf-8")
    )
    ct_qc_dir = Path(output_root) / "ct_qc"
    ct_qc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ct_qc_dir / "cohort_cache.json"
    results_tsv = ct_qc_dir / "radqy_results.tsv"
    config_signature = ct_qc_config_signature(config)
    legacy_config_signature = ct_qc_config_signature(
        config, include_device_indices=True
    )
    cohort_records = []
    for case in cases:
        case_id = str(
            case.get("Case_ID", "") or case.get("case_id", "") or "unknown_case"
        )
        ct_records = list(case.get("CT", []) or [])
        ct_records.sort(key=lambda item: json.dumps(to_jsonable(item), sort_keys=True))
        cohort_records.append({"case_id": case_id, "ct_records": ct_records})
    cohort_records.sort(key=lambda item: item["case_id"])
    cohort_signature = hashlib.sha256(
        json.dumps(to_jsonable(cohort_records), sort_keys=True).encode("utf-8")
    ).hexdigest()

    previous_manifest = {}
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_paths = {
        record["case_id"]: ct_qc_dir / record["case_id"] / "selection_summary.json"
        for record in cohort_records
    }
    saved_config_signature = previous_manifest.get("config_signature")
    config_matches = saved_config_signature in {
        config_signature,
        legacy_config_signature,
    }
    exact_cache_match = (
        config_matches
        and previous_manifest.get("cohort_signature") == cohort_signature
        and results_tsv.exists()
        and all(path.exists() for path in summary_paths.values())
    )
    if exact_cache_match:
        summaries = {
            case_id: json.loads(path.read_text(encoding="utf-8"))
            for case_id, path in summary_paths.items()
        }
        summaries_cacheable = all(
            bool(summary.get("cacheable", False)) for summary in summaries.values()
        )
        selected_files_exist = all(
            not summary.get("case_qc_passes_threshold")
            or Path(
                str(
                    dict(summary.get("selected_series", {}) or {}).get(
                        "ct_path", ""
                    )
                )
            ).is_file()
            for summary in summaries.values()
        )
        if summaries_cacheable and selected_files_exist:
            if saved_config_signature != config_signature:
                previous_manifest["config_signature"] = config_signature
                manifest_path.write_text(
                    json.dumps(
                        to_jsonable(previous_manifest),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            tqdm.write(
                f"[ct_qc] Reuse complete CT QC cohort cache: {manifest_path}"
            )
            return {
                "case_count": len(cases),
                "passed_case_ids": [
                    case_id
                    for case_id, summary in summaries.items()
                    if summary.get("case_qc_passes_threshold")
                ],
                "filtered_case_ids": [
                    case_id
                    for case_id, summary in summaries.items()
                    if not summary.get("case_qc_passes_threshold")
                ],
                "selection_summaries": summaries,
                "radqy_results_tsv_path": str(results_tsv),
            }

    reuse_cached_rows = not previous_manifest or config_matches
    radqy_input_dir = ct_qc_dir / "radqy_input"
    remove_path(radqy_input_dir)
    radqy_input_dir.mkdir(parents=True)
    try:
        case_series: dict[str, list[dict[str, Any]]] = {}
        prefilter_paths: dict[str, str] = {}
        all_series = []
        show_progress = sys.stderr.isatty()
        prepared_cases = prepare_ct_cases(
            list(cases),
            ct_qc_dir,
            config,
            reuse_cached_rows=reuse_cached_rows,
        )
        for case in tqdm(
            cases, desc="CT QC cases", unit="case", disable=not show_progress
        ):
            case_id = str(
                case.get("Case_ID", "") or case.get("case_id", "") or "unknown_case"
            )
            prepared = prepared_cases[case_id]
            series_records = list(prepared["series_records"])
            prefilter_report_path = str(prepared["prefilter_report_path"])
            case_series[case_id] = series_records
            prefilter_paths[case_id] = prefilter_report_path
            for record in series_records:
                source_dir = Path(str(record.get("ct_source_path", "") or ""))
                subject_id = str(record.get("radqy_subject_id", "") or "")
                if (
                    source_dir.is_dir()
                    and subject_id
                    and link_dicom_series_for_radqy(
                        source_dir, radqy_input_dir / subject_id
                    )
                ):
                    all_series.append(record)

        radqy_config = dict(config["radqy"])
        metrics_config = dict(radqy_config["metrics"])
        radqy_metrics = [
            str(item)
            for item in list(metrics_config["high_bad"])
            + list(metrics_config["low_bad"])
        ]
        if all_series:
            if radqy_results_cover_series(all_series, results_tsv, radqy_metrics):
                tqdm.write(f"[ct_qc] Reuse existing RadQy results: {results_tsv}")
            else:
                tqdm.write(f"[ct_qc] Run RadQy on {len(all_series)} CT series.")
                run_radqy_dataset(
                    radqy_input_dir,
                    ct_qc_dir,
                    scantype=str(radqy_config["modality"]),
                    segmenter=str(radqy_config["segmenter"]),
                    num_samples=int(radqy_config["sample_stride"]),
                    middle_percent=int(radqy_config["middle_percent"]),
                    save_images=bool(radqy_config["save_images"]),
                    save_fgbg=False,
                    output_name="radqy_results.tsv",
                )
                tqdm.write(f"[ct_qc] RadQy finished: {results_tsv}")
        else:
            pd.DataFrame().to_csv(results_tsv, sep="\t", index=False)

        enriched = match_radqy_rows(all_series, results_tsv)
        flagged = radqy_robust_z_flags(
            pd.DataFrame(enriched),
            high_bad=[str(item) for item in metrics_config["high_bad"]],
            low_bad=[str(item) for item in metrics_config["low_bad"]],
            cutoff=float(radqy_config["robust_z_cutoff"]),
            fail_threshold=int(radqy_config["bad_count_fail_threshold"]),
        )
        flagged_by_series_id = (
            {str(row["series_id"]): dict(row) for _, row in flagged.iterrows()}
            if not flagged.empty
            else {}
        )

        summaries = {}
        for case_id, records in tqdm(
            list(case_series.items()),
            desc="CT case selection",
            unit="case",
            disable=not show_progress,
        ):
            records = [
                flagged_by_series_id.get(
                    str(record.get("radqy_subject_id", "") or ""), record
                )
                for record in records
            ]
            summaries[case_id] = build_case_summary(
                case_id, records, prefilter_paths[case_id], str(results_tsv)
            )
    finally:
        remove_path(radqy_input_dir)
    manifest_path.write_text(
        json.dumps(
            {
                "config_signature": config_signature,
                "cohort_signature": cohort_signature,
                "case_ids": [record["case_id"] for record in cohort_records],
                "radqy_results_tsv_path": str(results_tsv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "case_count": len(cases),
        "passed_case_ids": [
            case_id
            for case_id, summary in summaries.items()
            if summary.get("case_qc_passes_threshold")
        ],
        "filtered_case_ids": [
            case_id
            for case_id, summary in summaries.items()
            if not summary.get("case_qc_passes_threshold")
        ],
        "selection_summaries": summaries,
        "radqy_results_tsv_path": str(results_tsv),
    }


def run_ct_qc(
    case_id: str,
    item: list[Mapping[str, Any]],
    output_root: str = "",
    config_dir: str = "",
) -> dict[str, Any]:
    result = run_ct_qc_cohort(
        [{"Case_ID": str(case_id), "CT": list(item)}],
        output_root=output_root,
        config_dir=config_dir,
    )
    return dict(result.get("selection_summaries", {}).get(str(case_id), {}))
