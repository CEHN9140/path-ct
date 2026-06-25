from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tool_utils import quiet_tool_logs

def run_ct_radiomics(
    *,
    case_id: str,
    ct_path: str = "",
    mask_path: str = "",
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    import numpy as np
    import SimpleITK as sitk
    import yaml
    from radiomics import featureextractor

    from utils.tool_utils import make_tool_result, safe_identifier

    tool_config = yaml.safe_load((Path(config_dir).expanduser() / "ct_radiomics.yaml").read_text(encoding="utf-8")) or {}
    case_name = safe_identifier(case_id)
    case_output_dir = Path(output_root) / "ct_radiomics" / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    label = int(tool_config.get("label"))
    remove_diagnostics = bool(tool_config.get("remove_diagnostics"))
    extractor = dict(tool_config.get("extractor") or {})
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
    metrics_path = case_output_dir / "metrics.json"
    artifacts = {
        "features_json_path": str(case_output_dir / "radiomics_features.json"),
        "features_csv_path": str(case_output_dir / "radiomics_features.csv"),
        "metrics_path": str(metrics_path),
    }
    provenance = {"backend": "pyradiomics", "case_id": case_id}
    input_payload = {
        "case_id": case_id,
        "ct_path": str(ct_file_path),
        "mask_path": str(mask_file_path),
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
    features_csv_path = case_output_dir / "radiomics_features.csv"
    diagnostics_json_path = case_output_dir / "diagnostics_features.json"
    started_at = time.time()

    try:
        mask_image = sitk.ReadImage(str(mask_file_path))
        mask_array = sitk.GetArrayFromImage(mask_image)
        mask_voxel_count = int(np.count_nonzero(mask_array == label))
        if mask_voxel_count <= 0:
            raise ValueError(f"Mask does not contain label {label}.")

        with tempfile.TemporaryDirectory(prefix=f"ct_radiomics_{case_name}_") as temp_dir:
            extractor_config_path = Path(temp_dir) / "extractor_config.yaml"
            extractor_config_path.write_text(
                json.dumps(extractor, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with quiet_tool_logs():
                radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(
                    str(extractor_config_path)
                )
                raw_features = dict(
                    radiomics_extractor.execute(
                        str(ct_file_path), str(mask_file_path), label=label
                    )
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
            json.dumps(selected_features, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with features_csv_path.open("w", encoding="utf-8", newline="") as handle:
            if numeric_features:
                writer = csv.DictWriter(handle, fieldnames=list(numeric_features.keys()))
                writer.writeheader()
                writer.writerow(numeric_features)
            else:
                handle.write("")

        metrics_payload = {
            "case_id": case_id,
            "ct_path": str(ct_file_path),
            "mask_path": str(mask_file_path),
            "label": label,
            "mask_voxel_count": mask_voxel_count,
            "feature_count": len(selected_features),
            "numeric_feature_count": len(numeric_features),
            "diagnostic_feature_count": len(diagnostics),
            "enabled_image_types": sorted(list(dict(extractor.get("imageType") or {}).keys())),
            "enabled_feature_classes": sorted(
                list(dict(extractor.get("featureClass") or {}).keys())
            ),
            "remove_diagnostics": remove_diagnostics,
            "elapsed_seconds": round(time.time() - started_at, 3),
            "issues": [],
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
