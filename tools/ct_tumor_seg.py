from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from utils.tool_utils import (
    make_tool_result,
    run_function_workers,
    safe_identifier,
    split_device_requests,
)


def load_nnunet_predictor(config: Mapping[str, Any]):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not bool(config["disable_tta"]),
        perform_everything_on_device=True,
        device=torch.device(str(config["device"])),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    folds = [
        value if value == "all" else int(value)
        for value in str(config["fold"]).split()
    ]
    predictor.initialize_from_trained_model_folder(
        str(Path(str(config["model_folder"])).expanduser().resolve()),
        folds,
        checkpoint_name=str(config["checkpoint"]),
    )
    return predictor


def save_label_mask(
    prediction_path: Path, output_path: Path, output_label: int | None
) -> None:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(prediction_path))
    if output_label is not None:
        image = sitk.Cast(sitk.Equal(image, output_label), sitk.sitkUInt8)
    sitk.WriteImage(image, str(output_path))


def run_nnunet_cohort_worker(payload: Mapping[str, Any]) -> None:
    config = dict(payload["shared"])
    physical_device = str(config.pop("physical_device"))
    if physical_device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_device.split(":", 1)[1]
        config["device"] = "cuda:0"

    output_root = str(config["output_root"])
    output_label = (
        None if config["output_label"] is None else int(config["output_label"])
    )
    predictor = load_nnunet_predictor(config)
    provenance_base = {
        "backend": str(config["backend"]),
        "model_folder": str(Path(str(config["model_folder"])).expanduser().resolve()),
        "output_label": output_label,
    }

    for request in payload["requests"]:
        case_id = str(request["case_id"])
        ct_path = str(request["ct_path"])
        case_dir = Path(output_root) / "ct_tumor_seg" / safe_identifier(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        raw_prefix = case_dir / f"{case_id}_raw"
        raw_path = Path(f"{raw_prefix}.nii.gz")
        final_path = case_dir / f"{case_id}_mask.nii.gz"
        provenance = {**provenance_base, "case_id": case_id}
        try:
            predictor.predict_from_files(
                [[ct_path]],
                [str(raw_prefix)],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=int(
                    config["num_processes_preprocessing"]
                ),
                num_processes_segmentation_export=int(
                    config["num_processes_segmentation_export"]
                ),
            )
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"nnUNet did not produce a segmentation for {case_id}."
                )
            save_label_mask(raw_path, final_path, output_label)
            make_tool_result(
                output_root=output_root,
                tool_name="ct_tumor_seg",
                status="success",
                identifier=case_id,
                metrics={"returncode": 0},
                artifacts={"segmentation_path": str(final_path)},
                provenance=provenance,
                payload={
                    "case_id": case_id,
                    "ct_identity": dict(request["ct_identity"]),
                    "input_ct_path": ct_path,
                },
            )
        except Exception as exc:
            final_path.unlink(missing_ok=True)
            make_tool_result(
                output_root=output_root,
                tool_name="ct_tumor_seg",
                status="failure",
                identifier=case_id,
                metrics={"returncode": 1},
                artifacts={},
                provenance=provenance,
                errors=[f"{type(exc).__name__}: {exc}"],
                payload={
                    "case_id": case_id,
                    "ct_identity": dict(request["ct_identity"]),
                    "input_ct_path": ct_path,
                },
            )
        finally:
            raw_path.unlink(missing_ok=True)


def run_ct_tumor_seg_cohort(
    requests: list[Mapping[str, Any]], output_root: str, config_dir: str = ""
) -> dict[str, dict[str, Any]]:
    if not requests:
        return {}
    config = yaml.safe_load(
        (Path(config_dir).expanduser() / "ct_tumor_seg.yaml").read_text(
            encoding="utf-8"
        )
    )
    if str(config["backend"]).lower() not in {"nnunetv2_modelfolder", "kits23"}:
        raise ValueError("CT segmentation requires the nnUNetv2 model-folder backend.")

    assignments, _ = split_device_requests(
        list(requests),
        list(config["devices"]),
        workers_per_device=int(config["workers_per_gpu"]),
    )
    payloads = [
        {
            "shared": {
                **config,
                "device": assignment["device"],
                "physical_device": assignment["physical_device"],
                "output_root": output_root,
            },
            "requests": assignment["requests"],
        }
        for assignment in assignments
    ]
    exit_codes = run_function_workers(run_nnunet_cohort_worker, payloads)
    if any(exit_codes):
        raise RuntimeError(f"nnUNet cohort workers failed: exit_codes={exit_codes}")

    results = {}
    for request in requests:
        case_id = str(request["case_id"])
        snapshot = Path(output_root) / "ct_tumor_seg" / f"{safe_identifier(case_id)}.json"
        results[case_id] = json.loads(snapshot.read_text(encoding="utf-8"))[
            "tool_result"
        ]
    return results


def run_ct_tumor_seg(
    case_id: str,
    ct_path: str,
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    return run_ct_tumor_seg_cohort(
        [{"case_id": case_id, "ct_path": ct_path, "ct_identity": {}}],
        output_root,
        config_dir,
    )[case_id]
