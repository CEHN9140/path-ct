from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tool_utils import quiet_tool_logs


def ct_feature_filter_audit(
    *,
    original_names: Sequence[str],
    ccc_names: Sequence[str],
    correlation_names: Sequence[str],
    corrected_names: Sequence[str],
    confound_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "original_feature_count": len(original_names),
        "original_features": list(original_names),
        "ccc_retained_feature_count": len(ccc_names),
        "ccc_retained_features": list(ccc_names),
        "correlation_retained_feature_count": len(correlation_names),
        "correlation_retained_features": list(correlation_names),
        "corrected_feature_count": len(corrected_names),
        "corrected_features": list(corrected_names),
        "confound_metrics": dict(confound_metrics),
    }


def build_ct_affinity(
    patient_states: Sequence[Mapping[str, Any]],
    *,
    config_dir: str = "",
    output_root: str = "",
) -> dict[str, Any]:
    """Build the CT network from cached radiomics and record every filter step."""
    import json
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist
    from snf.compute import affinity_matrix
    from tools.confound import confounder_values
    from utils.llm_utils import load_candidate_proposer_config, load_yaml_file

    config_path = Path(config_dir or "configs")
    ct_config = load_yaml_file(config_path / "ct_radiomics.yaml")
    snf_config = load_candidate_proposer_config(config_path)["snf"]
    ccc_threshold = float(ct_config["ccc_threshold"])
    widths = [float(value) for value in ct_config["ccc_comparison_bin_widths"]]
    labels = [str(int(value)) if value.is_integer() else str(value).replace(".", "_") for value in widths]
    low_variance = float(snf_config["ct_low_variance_threshold"])
    high_correlation = float(snf_config["ct_high_correlation_threshold"])
    states = [dict(state) for state in patient_states if state.get("qc") == "success"]
    case_ids = [str(state.get("case_id", "")) for state in states]
    vectors, names = [], []
    comparison = {label: [] for label in labels}
    for state in states:
        evidence = dict(state.get("ct_evidence", {}) or {})
        feature_path = Path(str(evidence.get("feature_path", "") or ""))
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
        feature_names = [str(name) for name in payload]
        vectors.append([float(payload[name]) for name in feature_names])
        names.append(feature_names)
        for label in labels:
            path = Path(str(dict(evidence.get("ccc_feature_paths", {}) or {}).get(label, "") or ""))
            values = json.loads(path.read_text(encoding="utf-8"))
            if list(values) != feature_names:
                raise ValueError(f"CT CCC feature names mismatch for {state.get('case_id')}, binWidth={label}")
            comparison[label].append([float(values[name]) for name in feature_names])
    if not vectors:
        return {"affinity": np.eye(0), "patient_ids": [], "audit": {}}
    if any(current != names[0] for current in names[1:]):
        raise ValueError("CT feature names differ across patients")
    matrix = np.asarray(vectors, dtype=float)
    feature_names = names[0]
    keep = np.var(matrix, axis=0) > low_variance
    matrix = matrix[:, keep]
    feature_names = [name for name, value in zip(feature_names, keep) if value]
    comparison = {label: np.asarray(values, dtype=float)[:, keep] for label, values in comparison.items()}
    ccc_keep = np.ones(matrix.shape[1], dtype=bool)
    ccc_scores = {}
    for index, name in enumerate(feature_names):
        scores = {}
        reference = matrix[:, index]
        for label, values in comparison.items():
            current = values[:, index]
            x_mean, y_mean = reference.mean(), current.mean()
            covariance = np.mean((reference - x_mean) * (current - y_mean))
            denominator = np.var(reference) + np.var(current) + (x_mean - y_mean) ** 2
            scores[label] = float(2 * covariance / denominator) if denominator else 0.0
        ccc_scores[name] = scores
        ccc_keep[index] = all(score >= ccc_threshold for score in scores.values())
    matrix = matrix[:, ccc_keep]
    feature_names = [name for name, value in zip(feature_names, ccc_keep) if value]
    ccc_feature_names = list(feature_names)
    corr_keep = np.ones(matrix.shape[1], dtype=bool)
    if matrix.shape[1] > 1:
        corr = np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0)
        for index in range(matrix.shape[1]):
            if np.any(np.abs(corr[index, :index][corr_keep[:index]]) > high_correlation):
                corr_keep[index] = False
    matrix = matrix[:, corr_keep]
    feature_names = [name for name, value in zip(feature_names, corr_keep) if value]
    correlation_feature_names = list(feature_names)

    values = confounder_values({case_id: state for case_id, state in zip(case_ids, states)}, output_root)
    correction_config = dict(ct_config.get("confound_correction", {}) or {})
    fields = list(correction_config.get("fields") or [
        "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel", "ct_slice_thickness"
    ]) if bool(correction_config.get("enabled", True)) else []
    design_parts = []
    design_audit = {}
    for field in fields:
        raw = [dict(values.get(case_id, {}) or {}).get(field, "") for case_id in case_ids]
        if field == "ct_slice_thickness":
            numeric = pd.to_numeric(pd.Series(raw), errors="coerce")
            missing_count = int(numeric.isna().sum())
            numeric = numeric.fillna(float(numeric.median()) if numeric.notna().any() else 0.0)
            design_parts.append(numeric.to_numpy(float)[:, None])
            design_audit[field] = {"type": "numeric", "missing_count": missing_count}
        else:
            missing_count = sum(not str(item or "").strip() for item in raw)
            series = pd.Series([str(item or "missing") for item in raw])
            encoded = pd.get_dummies(series, dtype=float)
            design_parts.append(encoded.to_numpy(float))
            design_audit[field] = {"type": "categorical", "levels": sorted(series.unique().tolist()), "missing_count": missing_count}
    design = np.column_stack([np.ones(len(case_ids)), *design_parts]) if design_parts else np.ones((len(case_ids), 1))
    original_matrix = matrix.copy()
    coefficients = np.linalg.lstsq(design, matrix, rcond=None)[0]
    corrected = matrix - design @ coefficients
    corrected_std = corrected.std(axis=0)
    corrected_keep = corrected_std > low_variance
    corrected = corrected[:, corrected_keep]
    corrected_names = [name for name, value in zip(feature_names, corrected_keep) if value]
    def explained_variance(data: np.ndarray) -> dict[str, float | None]:
        if data.shape[1] == 0:
            return {"median_r2": None, "q90_r2": None, "max_r2": None}
        total = np.var(data, axis=0)
        fitted = design @ np.linalg.lstsq(design, data, rcond=None)[0]
        r2 = np.divide(np.var(fitted, axis=0), total, out=np.zeros_like(total), where=total > 0)
        return {"median_r2": float(np.median(r2)), "q90_r2": float(np.quantile(r2, .9)), "max_r2": float(np.max(r2))}
    confound_metrics = {
        "fields": design_audit,
        "technical_r2_before": explained_variance(original_matrix),
        "technical_r2_after": explained_variance(corrected),
        "correction": "least_squares_residualization_with_intercept",
    }
    means = corrected.mean(axis=0, keepdims=True)
    stds = corrected.std(axis=0, keepdims=True)
    corrected = np.divide(corrected - means, stds, out=np.zeros_like(corrected), where=stds > 0)
    distance = cdist(corrected, corrected, metric="euclidean")
    affinity = (
        np.ones((1, 1), dtype=float)
        if len(case_ids) == 1
        else affinity_matrix(distance, K=min(max(int(snf_config["neighbor_count"]), 1), len(case_ids) - 1), mu=float(snf_config["mu"]))
    )
    audit = ct_feature_filter_audit(
        original_names=names[0],
        ccc_names=ccc_feature_names,
        correlation_names=correlation_feature_names,
        corrected_names=corrected_names,
        confound_metrics=confound_metrics,
    )
    audit.update({
        "case_count": len(case_ids),
        "ccc_threshold": ccc_threshold,
        "high_correlation_threshold": high_correlation,
        "ccc_scores": ccc_scores,
        "confound_correction_enabled": bool(correction_config.get("enabled", True)),
        "confound_fields": fields,
    })
    return {"affinity": np.asarray(affinity, dtype=float), "patient_ids": case_ids, "feature_names": corrected_names, "audit": audit}


def run_ct_radiomics(
    *,
    case_id: str,
    ct_path: str = "",
    mask_path: str = "",
    ct_identity: Mapping[str, Any] | None = None,
    mask_identity: Mapping[str, Any] | None = None,
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    import numpy as np
    import SimpleITK as sitk
    import yaml
    from radiomics import featureextractor

    from utils.tool_utils import make_tool_result, safe_identifier

    tool_config = yaml.safe_load(
        (Path(config_dir).expanduser() / "ct_radiomics.yaml").read_text(
            encoding="utf-8"
        )
    )
    case_name = safe_identifier(case_id)
    case_output_dir = Path(output_root) / "ct_radiomics" / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    label = int(tool_config["label"])
    remove_diagnostics = bool(tool_config["remove_diagnostics"])
    extractor = dict(tool_config["extractor"])
    reference_bin_width = float(extractor["setting"]["binWidth"])
    comparison_bin_widths = [
        float(value) for value in tool_config["ccc_comparison_bin_widths"]
    ]

    hu_clip_range = [float(value) for value in tool_config["hu_clip_range"]]
    if len(hu_clip_range) != 2:
        raise ValueError("hu_clip_range must contain [lower, upper].")
    hu_lower, hu_upper = hu_clip_range
    if hu_lower >= hu_upper:
        raise ValueError("hu_clip_range lower bound must be smaller than upper bound.")

    ct_file_path = (
        Path(str(ct_path)).expanduser()
        if str(ct_path).strip()
        else Path(output_root) / "ct_qc" / case_name / f"{case_name}.nii.gz"
    )
    mask_file_path = (
        Path(str(mask_path)).expanduser()
        if str(mask_path).strip()
        else Path(output_root) / "ct_tumor_seg" / case_name / f"{case_id}_mask.nii.gz"
    )
    artifacts = {
        "features_json_path": str(case_output_dir / "radiomics_features.json"),
    }
    for bin_width in comparison_bin_widths:
        width_label = (
            str(int(bin_width))
            if bin_width.is_integer()
            else str(bin_width).replace(".", "_")
        )
        artifacts[f"features_bin_width_{width_label}_json_path"] = str(
            case_output_dir / f"radiomics_features_bin_width_{width_label}.json"
        )
    provenance = {"backend": "pyradiomics", "case_id": case_id}
    input_payload = {
        "case_id": case_id,
        "ct_path": str(ct_file_path),
        "mask_path": str(mask_file_path),
        "ct_identity": dict(ct_identity or {}),
        "mask_identity": dict(mask_identity or {}),
        "radiomics_config": tool_config,
        "hu_clip_range": hu_clip_range,
    }

    missing_inputs: list[str] = []
    if not ct_file_path.exists():
        missing_inputs.append(f"CT image does not exist: {ct_file_path}")
    if not mask_file_path.exists():
        missing_inputs.append(f"Tumor mask does not exist: {mask_file_path}")
    if missing_inputs:
        return make_tool_result(
            output_root=output_root,
            tool_name="ct_radiomics",
            status="failure",
            identifier=case_id,
            metrics={},
            artifacts={},
            provenance=provenance,
            errors=missing_inputs,
            payload=input_payload,
        )

    features_json_path = case_output_dir / "radiomics_features.json"
    diagnostics_json_path = case_output_dir / "diagnostics_features.json"
    started_at = time.time()

    try:
        ct_image = sitk.ReadImage(str(ct_file_path))
        mask_image = sitk.ReadImage(str(mask_file_path))
        mask_array = sitk.GetArrayFromImage(mask_image)
        mask_voxel_count = int(np.count_nonzero(mask_array == label))
        if mask_voxel_count <= 0:
            raise ValueError(f"Mask does not contain label {label}.")

        clamp_filter = sitk.ClampImageFilter()
        clamp_filter.SetLowerBound(hu_lower)
        clamp_filter.SetUpperBound(hu_upper)
        clipped_ct_image = clamp_filter.Execute(ct_image)

        raw_features_by_width: dict[float, dict[str, Any]] = {}
        for bin_width in [reference_bin_width, *comparison_bin_widths]:
            width_extractor = copy.deepcopy(extractor)
            width_extractor["setting"]["binWidth"] = bin_width
            with quiet_tool_logs():
                radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(
                    width_extractor
                )
                raw_features_by_width[bin_width] = dict(
                    radiomics_extractor.execute(
                        clipped_ct_image, mask_image, label=label
                    )
                )

        raw_features = raw_features_by_width[reference_bin_width]

        def json_value(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    pass
            if isinstance(value, (list, tuple)):
                return [json_value(item) for item in value]
            if hasattr(value, "tolist"):
                try:
                    return value.tolist()
                except Exception:
                    pass
            return str(value)

        diagnostics = {
            str(key): json_value(value)
            for key, value in raw_features.items()
            if str(key).startswith("diagnostics_")
        }
        selected_features = {
            str(key): json_value(value)
            for key, value in raw_features.items()
            if (not remove_diagnostics) or not str(key).startswith("diagnostics_")
        }
        numeric_features = {
            key: float(value)
            for key, value in selected_features.items()
            if isinstance(value, (int, float))
        }

        diagnostics_json_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        features_json_path.write_text(
            json.dumps(selected_features, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for bin_width in comparison_bin_widths:
            width_label = (
                str(int(bin_width))
                if bin_width.is_integer()
                else str(bin_width).replace(".", "_")
            )
            comparison_features = {
                str(key): json_value(value)
                for key, value in raw_features_by_width[bin_width].items()
                if (not remove_diagnostics) or not str(key).startswith("diagnostics_")
            }
            Path(artifacts[f"features_bin_width_{width_label}_json_path"]).write_text(
                json.dumps(comparison_features, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        metrics_payload = {
            "case_id": case_id,
            "ct_path": str(ct_file_path),
            "mask_path": str(mask_file_path),
            "ct_identity": dict(ct_identity or {}),
            "mask_identity": dict(mask_identity or {}),
            "label": label,
            "mask_voxel_count": mask_voxel_count,
            "feature_count": len(selected_features),
            "numeric_feature_count": len(numeric_features),
            "diagnostic_feature_count": len(diagnostics),
            "enabled_image_types": sorted(
                list(dict(extractor.get("imageType") or {}).keys())
            ),
            "enabled_feature_classes": sorted(
                list(dict(extractor.get("featureClass") or {}).keys())
            ),
            "remove_diagnostics": remove_diagnostics,
            "elapsed_seconds": round(time.time() - started_at, 3),
            "radiomics_config": tool_config,
            "hu_clip_range": hu_clip_range,
            "image_normalization": bool(
                extractor.get("setting", {}).get("normalize", False)
            ),
            "resampled_pixel_spacing": list(
                extractor.get("setting", {}).get("resampledPixelSpacing", [])
            ),
            "bin_width": float(extractor.get("setting", {}).get("binWidth")),
            "ccc_comparison_bin_widths": comparison_bin_widths,
            "ccc_threshold": float(tool_config["ccc_threshold"]),
            "issues": [],
        }
    except Exception as exc:
        return make_tool_result(
            output_root=output_root,
            tool_name="ct_radiomics",
            status="failure",
            identifier=case_id,
            metrics={},
            artifacts={},
            provenance=provenance,
            errors=[f"PyRadiomics extraction failed: {type(exc).__name__}: {exc}"],
            payload=input_payload,
        )

    issues = [str(item) for item in list(metrics_payload.get("issues") or [])]
    if issues:
        return make_tool_result(
            output_root=output_root,
            tool_name="ct_radiomics",
            status="failure",
            identifier=case_id,
            metrics={
                "mask_voxel_count": int(metrics_payload.get("mask_voxel_count") or 0),
                "feature_count": int(metrics_payload.get("feature_count") or 0),
                "elapsed_seconds": float(metrics_payload.get("elapsed_seconds") or 0.0),
            },
            artifacts=artifacts,
            provenance=provenance,
            errors=issues,
            payload=metrics_payload,
        )

    return make_tool_result(
        output_root=output_root,
        tool_name="ct_radiomics",
        status="success",
        identifier=case_id,
        metrics={
            "label": int(metrics_payload.get("label") or label),
            "mask_voxel_count": int(metrics_payload.get("mask_voxel_count") or 0),
            "feature_count": int(metrics_payload.get("feature_count") or 0),
            "elapsed_seconds": float(metrics_payload.get("elapsed_seconds") or 0.0),
        },
        artifacts=artifacts,
        provenance=provenance,
        errors=[],
        payload=metrics_payload,
    )
