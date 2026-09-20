from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tool_utils import quiet_tool_logs


def build_ct_discovery_feature_matrix(
    patient_states: Sequence[Mapping[str, Any]],
    *,
    config_dir: str = "",
    output_root: str = "",
) -> dict[str, Any]:
    import json
    import numpy as np
    from utils.llm_utils import load_candidate_proposer_config

    snf_config = load_candidate_proposer_config(config_dir or "configs")["snf"]
    correlation_threshold = float(snf_config.get("ct_high_correlation_threshold", 0.95))
    states = [dict(state) for state in patient_states if state.get("qc") == "success"]
    patient_ids = [str(state.get("case_id", "")) for state in states]
    vectors = []
    feature_names = []
    for state in states:
        evidence = dict(state.get("ct_evidence", {}) or {})
        values = json.loads(Path(str(evidence["feature_path"])).read_text(encoding="utf-8"))
        names = list(values)
        if feature_names and names != feature_names:
            raise ValueError(f"CT feature names differ for {state['case_id']}")
        feature_names = names
        vectors.append([float(values[name]) for name in names])

    matrix = np.asarray(vectors, dtype=float)
    if not patient_ids or not feature_names or not np.isfinite(matrix).all():
        raise ValueError("CT feature matrix must contain finite features for every eligible patient")

    scale = np.maximum(1.0, np.max(np.abs(matrix), axis=0))
    constant = matrix.std(axis=0) <= np.finfo(float).eps * scale * 16
    constant_removed = [name for name, drop in zip(feature_names, constant) if drop]
    matrix = matrix[:, ~constant]
    names = [name for name, drop in zip(feature_names, constant) if not drop]
    if not names:
        raise ValueError("CT radiomics has no non-constant features")

    correlation_pruned = []
    if len(names) > 1:
        absolute = np.nan_to_num(np.abs(np.corrcoef(matrix, rowvar=False)), nan=0.0)
        np.fill_diagonal(absolute, 0.0)
        edges = sorted(
            (
                (absolute[left, right], names[left], names[right], left, right)
                for left in range(len(names))
                for right in range(left + 1, len(names))
                if absolute[left, right] > correlation_threshold
            ),
            key=lambda edge: (-edge[0], edge[1], edge[2]),
        )
        active = np.ones(len(names), dtype=bool)
        row_sums = absolute.sum(axis=1)
        active_count = len(names)
        for _, _, _, left, right in edges:
            if active[left] and active[right]:
                drop = max(
                    (left, right),
                    key=lambda index: (row_sums[index] / max(active_count - 1, 1), names[index]),
                )
                active[drop] = False
                row_sums -= absolute[:, drop]
                active_count -= 1
                correlation_pruned.append(names[drop])
        matrix = matrix[:, active]
        names = [name for name, keep in zip(names, active) if keep]

    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    matrix = (matrix - mean) / std
    return {
        "matrix": matrix,
        "patient_ids": patient_ids,
        "feature_names": names,
        "audit": {
            "case_count": len(patient_ids),
            "raw_feature_count": len(feature_names),
            "constant_removed": constant_removed,
            "correlation_threshold": correlation_threshold,
            "correlation_pruned": correlation_pruned,
            "retained_features": names,
            "technical_residualization": False,
            "ccc_filter": False,
            "steps": [
                "cached PyRadiomics extraction",
                "constant and near-constant feature removal",
                "order-independent absolute Pearson correlation pruning",
                "feature-wise z-score",
                "Euclidean distance",
            ],
        },
    }


def build_ct_affinity(
    patient_states: Sequence[Mapping[str, Any]],
    *,
    config_dir: str = "",
    output_root: str = "",
) -> dict[str, Any]:
    from scipy.spatial.distance import cdist
    from tools.evidence_features import distance_to_affinity
    from utils.llm_utils import load_candidate_proposer_config

    payload = build_ct_discovery_feature_matrix(
        patient_states, config_dir=config_dir, output_root=output_root
    )
    config = load_candidate_proposer_config(config_dir or "configs")["snf"]
    distance = cdist(payload["matrix"], payload["matrix"], metric="euclidean")
    return {**payload, "affinity": distance_to_affinity(distance, config)}


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

        with quiet_tool_logs():
            radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(extractor)
            raw_features = dict(
                radiomics_extractor.execute(clipped_ct_image, mask_image, label=label)
            )

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
