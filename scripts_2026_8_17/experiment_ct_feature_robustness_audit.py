from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ccc(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = float(left.mean())
    right_mean = float(right.mean())
    denominator = float(left.var() + right.var() + (left_mean - right_mean) ** 2)
    return float(2 * np.mean((left - left_mean) * (right - right_mean)) / denominator) if denominator else 0.0


def run_mask_perturbation(
    rows: list[dict[str, Any]],
    config_path: Path,
    case_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import SimpleITK as sitk
    import yaml
    from radiomics import featureextractor

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    extractor_config = dict(config["extractor"])
    hu_lower, hu_upper = [float(value) for value in config["hu_clip_range"]]
    label = int(config["label"])
    extractor = featureextractor.RadiomicsFeatureExtractor(extractor_config)
    case_results = []
    selected_rows = rows[:case_limit] if case_limit else rows
    for row in selected_rows:
        case_id = str(row.get("case_id", "") or "")
        ct_path = Path(str(row.get("ct_path", "") or ""))
        mask_path = Path(str(row.get("mask_path", "") or ""))
        record = {
            "case_id": case_id,
            "ct_path": str(ct_path),
            "mask_path": str(mask_path),
            "status": "failure",
            "feature_count": 0,
            "error": "",
        }
        if not ct_path.is_file() or not mask_path.is_file():
            record["error"] = "ct_or_mask_missing"
            case_results.append(record)
            continue
        try:
            image = sitk.ReadImage(str(ct_path))
            mask = sitk.ReadImage(str(mask_path))
            clipped = sitk.Clamp(
                image,
                sitk.sitkFloat32,
                lowerBound=hu_lower,
                upperBound=hu_upper,
            )
            binary = sitk.Cast(mask == label, sitk.sitkUInt8)
            eroded = sitk.BinaryErode(binary, [1, 1, 1])
            dilated = sitk.BinaryDilate(binary, [1, 1, 1])
            original_features = {
                str(key): float(value)
                for key, value in extractor.execute(clipped, mask, label=label).items()
                if isinstance(value, (int, float)) and not str(key).startswith("diagnostics_")
            }
            erode_features = {
                str(key): float(value)
                for key, value in extractor.execute(clipped, eroded, label=1).items()
                if isinstance(value, (int, float)) and not str(key).startswith("diagnostics_")
            }
            dilate_features = {
                str(key): float(value)
                for key, value in extractor.execute(clipped, dilated, label=1).items()
                if isinstance(value, (int, float)) and not str(key).startswith("diagnostics_")
            }
            record.update(
                {
                    "status": "success",
                    "feature_count": len(original_features),
                }
            )
            record["_original"] = original_features
            record["_erode"] = erode_features
            record["_dilate"] = dilate_features
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        case_results.append(record)

    successful = [row for row in case_results if row["status"] == "success"]
    common = sorted(
        set.intersection(
            *[
                set(row["_original"]) & set(row["_erode"]) & set(row["_dilate"])
                for row in successful
            ]
        )
    ) if successful else []
    feature_rows = []
    for name in common:
        original = np.asarray([row["_original"][name] for row in successful], dtype=float)
        erode = np.asarray([row["_erode"][name] for row in successful], dtype=float)
        dilate = np.asarray([row["_dilate"][name] for row in successful], dtype=float)
        feature_rows.append(
            {
                "feature": name,
                "ccc_erode": ccc(original, erode),
                "ccc_dilate": ccc(original, dilate),
            }
        )
    for row in case_results:
        row.pop("_original", None)
        row.pop("_erode", None)
        row.pop("_dilate", None)
    return case_results, feature_rows


def load_snapshot_rows(snapshot_root: Path) -> list[dict[str, Any]]:
    rows = []
    for snapshot_path in sorted(snapshot_root.glob("*.json")):
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        tool_result = dict(snapshot.get("tool_result", {}) or {})
        if str(tool_result.get("status", "") or "").lower() != "success":
            continue
        payload = dict(snapshot.get("payload", {}) or {})
        artifacts = dict(tool_result.get("artifacts", {}) or {})
        case_id = str(payload.get("case_id", "") or snapshot_path.stem)
        rows.append(
            {
                "case_id": case_id,
                "ct_path": str(payload.get("input_ct_path", "") or ""),
                "mask_path": str(
                    artifacts.get("segmentation_path", "")
                    or payload.get("segmentation_path", "")
                    or ""
                ),
            }
        )
    return rows


def run(
    snapshot_root: Path,
    config_path: Path,
    experiment_root: Path,
    run_mask_audit: bool,
    case_limit: int,
) -> dict[str, Any]:
    selected = load_snapshot_rows(snapshot_root)
    selected = selected[:case_limit] if case_limit else selected
    hu_rows = []
    if selected:
        import SimpleITK as sitk
        import yaml

        radiomics_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        lower, upper = [float(value) for value in radiomics_config["hu_clip_range"]]
        label = int(radiomics_config["label"])
        for row in selected:
            case_id = str(row.get("case_id", "") or "")
            ct_path = Path(str(row.get("ct_path", "") or ""))
            mask_path = Path(str(row.get("mask_path", "") or ""))
            record = {"case_id": case_id, "ct_path": str(ct_path), "mask_path": str(mask_path), "status": "failure", "clipped_voxel_fraction": "", "error": ""}
            if ct_path.is_file() and mask_path.is_file():
                try:
                    image = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(float)
                    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) == label
                    values = image[mask]
                    record.update({"status": "success", "tumor_voxel_count": int(values.size), "clipped_voxel_fraction": float(np.mean((values < lower) | (values > upper))) if values.size else ""})
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
            else:
                record["error"] = "ct_or_mask_missing"
            hu_rows.append(record)

    perturbation_rows: list[dict[str, Any]] = []
    perturbation_feature_rows: list[dict[str, Any]] = []
    if run_mask_audit:
        perturbation_rows, perturbation_feature_rows = run_mask_perturbation(
            selected,
            config_path,
            case_limit,
        )
    summary = {
        "experiment": "ct_feature_robustness_audit_v3",
        "inputs": {"ct_tumor_seg_snapshot_root": str(snapshot_root), "config": str(config_path)},
        "selected_case_count": int(len(selected)),
        "hu_clipping": {
            "case_count": len(hu_rows),
            "successful_case_count": sum(row["status"] == "success" for row in hu_rows),
            "median_clipped_voxel_fraction": float(np.median([row["clipped_voxel_fraction"] for row in hu_rows if row["clipped_voxel_fraction"] != ""])) if any(row["clipped_voxel_fraction"] != "" for row in hu_rows) else None,
        },
        "mask_perturbation": {
            "executed": bool(run_mask_audit),
            "case_count": len(perturbation_rows),
            "successful_case_count": sum(row["status"] == "success" for row in perturbation_rows),
            "feature_count": len(perturbation_feature_rows),
            "median_ccc_erode": float(np.median([row["ccc_erode"] for row in perturbation_feature_rows])) if perturbation_feature_rows else None,
            "median_ccc_dilate": float(np.median([row["ccc_dilate"] for row in perturbation_feature_rows])) if perturbation_feature_rows else None,
            "ccc_ge_0_85_fraction_erode": float(np.mean([row["ccc_erode"] >= 0.85 for row in perturbation_feature_rows])) if perturbation_feature_rows else None,
            "ccc_ge_0_85_fraction_dilate": float(np.mean([row["ccc_dilate"] >= 0.85 for row in perturbation_feature_rows])) if perturbation_feature_rows else None,
        },
        "interpretation_policy": {
            "phase_group_is_audit_only": True,
            "hu_clip_range_is_not_changed": True,
            "mask_perturbation_does_not_change_main_features": True,
            "no_main_pipeline_change": True,
        },
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_csv(experiment_root / "ct_hu_clipping_audit.csv", hu_rows)
    if run_mask_audit:
        write_csv(experiment_root / "ct_mask_perturbation_audit.csv", perturbation_rows)
        write_csv(experiment_root / "ct_mask_perturbation_feature_audit.csv", perturbation_feature_rows)
    (experiment_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CT phase groups, HU clipping and mask robustness.")
    parser.add_argument("--snapshot-root", type=Path, default=Path("output_kirc/ct_tumor_seg"))
    parser.add_argument("--config", type=Path, default=Path("configs/ct_radiomics.yaml"))
    parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v9/experiment_ct_feature_robustness_audit"))
    parser.add_argument("--run-mask-perturbation", action="store_true")
    parser.add_argument("--case-limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot_root, args.config, args.experiment_root, args.run_mask_perturbation, args.case_limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
