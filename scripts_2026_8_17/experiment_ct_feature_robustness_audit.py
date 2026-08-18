from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def phase_group(value: str) -> str:
    phase = str(value or "UNKNOWN").strip().upper()
    if phase in {"NC", "ART", "NEPH", "DEL", "UNKNOWN"}:
        return phase
    if phase in {"MAIN_CE_HIGH", "MAIN_CE_MEDIUM", "CE_UNSPECIFIED"}:
        return "OTHER_CE"
    return "UNKNOWN"


def ccc(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = float(left.mean())
    right_mean = float(right.mean())
    denominator = float(left.var() + right.var() + (left_mean - right_mean) ** 2)
    return float(2 * np.mean((left - left_mean) * (right - right_mean)) / denominator) if denominator else 0.0


def run_mask_perturbation(
    rows: list[dict[str, Any]],
    mask_root: Path,
    config_path: Path,
    experiment_root: Path,
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
        mask_path = mask_root / case_id / f"{case_id}_mask.nii.gz"
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


def run(
    selected_csv: Path,
    mask_root: Path,
    config_path: Path,
    experiment_root: Path,
    run_mask_audit: bool,
    case_limit: int,
) -> dict[str, Any]:
    table = pd.read_csv(selected_csv)
    selected = table.loc[
        table["selected_by_current_qc"].astype(str).str.lower().eq("true")
    ].copy()
    phase_counts = dict(
        Counter(phase_group(value) for value in selected.get("selected_phase", []))
    )
    hu_rows = []
    if not selected.empty:
        import SimpleITK as sitk
        import yaml

        radiomics_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        lower, upper = [float(value) for value in radiomics_config["hu_clip_range"]]
        label = int(radiomics_config["label"])
        for row in selected.to_dict(orient="records"):
            case_id = str(row.get("case_id", "") or "")
            ct_path = Path(str(row.get("ct_path", "") or ""))
            mask_path = mask_root / case_id / f"{case_id}_mask.nii.gz"
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
            selected.to_dict(orient="records"),
            mask_root,
            config_path,
            experiment_root,
            case_limit,
        )
    summary = {
        "experiment": "ct_feature_robustness_audit_v1",
        "inputs": {"selected_csv": str(selected_csv), "mask_root": str(mask_root), "config": str(config_path)},
        "selected_case_count": int(len(selected)),
        "phase_group_counts": phase_counts,
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
    parser.add_argument("--selected-csv", type=Path, default=Path("output_kirc_v9/experiment_ct_final_pass_fail_v2/selected_series.csv"))
    parser.add_argument("--mask-root", type=Path, default=Path("output_kirc_v9/experiment_ct_final_pass_fail_v2/ct_tumor_seg"))
    parser.add_argument("--config", type=Path, default=Path("configs/ct_radiomics.yaml"))
    parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v9/experiment_ct_feature_robustness_audit"))
    parser.add_argument("--run-mask-perturbation", action="store_true")
    parser.add_argument("--case-limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.selected_csv, args.mask_root, args.config, args.experiment_root, args.run_mask_perturbation, args.case_limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
