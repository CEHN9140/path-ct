from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from utils.tool_utils import make_tool_result, safe_identifier


def run_nnunet_prediction(
    *,
    input_dir: Path,
    prediction_dir: Path,
    task_name: str,
    model: str,
    trainer_class_name: str,
    cascade_trainer_class_name: str,
    num_threads_preprocessing: int,
    num_threads_nifti_save: int,
    disable_tta: bool,
    python_executable: Path,
    nnunet_code_root: Path,
    nnunet_results_folder: Path,
    device: str,
    mpl_config_dir: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(python_executable),
        "-m",
        "nnunet.inference.predict_simple",
        "-i",
        str(input_dir),
        "-o",
        str(prediction_dir),
        "-t",
        task_name,
        "-m",
        model,
        "-tr",
        trainer_class_name,
        "-ctr",
        cascade_trainer_class_name,
        "--overwrite_existing",
        "--num_threads_preprocessing",
        str(num_threads_preprocessing),
        "--num_threads_nifti_save",
        str(num_threads_nifti_save),
    ]
    if disable_tta:
        command.append("--disable_tta")

    env = dict(os.environ)
    python_path_items = [str(nnunet_code_root)]
    if env.get("PYTHONPATH"):
        python_path_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_items)
    env["RESULTS_FOLDER"] = str(nnunet_results_folder)
    env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    device = str(device or "").strip().lower()
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1].strip()
    elif device.isdigit():
        env["CUDA_VISIBLE_DEVICES"] = device

    prediction_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        command,
        env=env,
        cwd=str(nnunet_code_root),
        capture_output=True,
        text=True,
        check=False,
    )


def run_nnunetv2_modelfolder_prediction(
    *,
    input_dir: Path,
    prediction_dir: Path,
    model_folder: Path,
    checkpoint: str,
    fold: str,
    num_processes_preprocessing: int,
    num_processes_segmentation_export: int,
    disable_tta: bool,
    predict_bin: Path,
    device: str,
    mpl_config_dir: Path,
) -> subprocess.CompletedProcess[str]:
    device_value = str(device or "cuda").strip().lower()
    command_device = "cuda" if device_value.startswith("cuda") else device_value
    command = [
        str(predict_bin),
        "-i",
        str(input_dir),
        "-o",
        str(prediction_dir),
        "-m",
        str(model_folder),
        "-f",
        fold,
        "-chk",
        checkpoint,
        "-device",
        command_device,
        "-npp",
        str(num_processes_preprocessing),
        "-nps",
        str(num_processes_segmentation_export),
    ]
    if disable_tta:
        command.append("--disable_tta")

    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    if device_value.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device_value.split(":", 1)[1].strip()
    elif device_value.isdigit():
        env["CUDA_VISIBLE_DEVICES"] = device_value

    prediction_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def copy_or_extract_label(prediction_path: Path, output_path: Path, output_label: int | None) -> None:
    if output_label is None:
        shutil.copy2(prediction_path, output_path)
        return

    import SimpleITK as sitk

    image = sitk.ReadImage(str(prediction_path))
    mask = sitk.Equal(image, int(output_label))
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    sitk.WriteImage(mask, str(output_path))


def run_ct_tumor_seg(
    case_id: str,
    ct_path: str,
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    import yaml

    tool_config = yaml.safe_load((Path(config_dir).expanduser() / "ct_tumor_seg.yaml").read_text(encoding="utf-8")) or {}
    backend = str(tool_config["backend"]).strip().lower()
    mpl_config_dir = Path(str(tool_config["mpl_config_dir"])).expanduser().resolve()
    disable_tta = bool(tool_config["disable_tta"])
    device = str(tool_config["device"])

    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    case_name = safe_identifier(case_id)
    raw_ct_path = str(ct_path or "").strip()
    source_path = Path(raw_ct_path).expanduser().resolve() if raw_ct_path else Path("")
    case_output_dir = Path(output_root) / "ct_tumor_seg" / case_name
    final_segmentation_path = case_output_dir / f"{case_id}_mask.nii.gz"
    provenance = {"backend": backend, "case_id": case_id}
    if backend in {"nnunetv2_modelfolder", "kits23"}:
        provenance.update(
            {
                "model_folder": str(Path(str(tool_config["model_folder"])).expanduser().resolve()),
                "output_label": int(tool_config["output_label"]) if tool_config["output_label"] is not None else None,
            }
        )
    else:
        provenance.update({"task_name": str(tool_config["task_name"])})

    if not raw_ct_path or not source_path.exists():
        return make_tool_result(
            output_root=output_root,
            tool_name="ct_tumor_seg",
            status="failure",
            identifier=case_id,
            metrics={},
            artifacts={},
            provenance=provenance,
            errors=[f"CT input path does not exist: {source_path}"],
            payload={"case_id": case_id, "input_ct_path": str(source_path)},
        )
    if not source_path.name.lower().endswith(".nii.gz"):
        return make_tool_result(
            output_root=output_root,
            tool_name="ct_tumor_seg",
            status="failure",
            identifier=case_id,
            metrics={},
            artifacts={},
            provenance=provenance,
            errors=[f"CT input path must be a .nii.gz file: {source_path}"],
            payload={"case_id": case_id, "input_ct_path": str(source_path)},
        )

    case_output_dir.mkdir(parents=True, exist_ok=True)
    for child in case_output_dir.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    with tempfile.TemporaryDirectory(prefix=f"ct_tumor_seg_{case_name}_") as temp_dir:
        staged_output_dir = Path(temp_dir) / "case_output"
        input_dir = staged_output_dir / "input"
        prediction_dir = staged_output_dir / "predictions"
        prepared_input_path = input_dir / f"{case_name}_0000.nii.gz"

        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, prepared_input_path)

        if backend in {"nnunetv2_modelfolder", "kits23"}:
            model_folder = Path(str(tool_config["model_folder"])).expanduser().resolve()
            run_info = run_nnunetv2_modelfolder_prediction(
                input_dir=input_dir,
                prediction_dir=prediction_dir,
                model_folder=model_folder,
                checkpoint=str(tool_config["checkpoint"]),
                fold=str(tool_config["fold"]),
                num_processes_preprocessing=int(tool_config["num_processes_preprocessing"]),
                num_processes_segmentation_export=int(tool_config["num_processes_segmentation_export"]),
                disable_tta=disable_tta,
                predict_bin=Path(str(tool_config["predict_bin"])).expanduser(),
                device=device,
                mpl_config_dir=mpl_config_dir,
            )
            output_label = int(tool_config["output_label"]) if tool_config["output_label"] is not None else None
        else:
            run_info = run_nnunet_prediction(
                input_dir=input_dir,
                prediction_dir=prediction_dir,
                task_name=str(tool_config["task_name"]),
                model=str(tool_config["model"]),
                trainer_class_name=str(tool_config["trainer_class_name"]),
                cascade_trainer_class_name=str(tool_config["cascade_trainer_class_name"]),
                num_threads_preprocessing=int(tool_config["num_threads_preprocessing"]),
                num_threads_nifti_save=int(tool_config["num_threads_nifti_save"]),
                disable_tta=disable_tta,
                python_executable=Path(sys.executable).expanduser(),
                nnunet_code_root=Path(str(tool_config["nnunet_code_root"])).expanduser().resolve(),
                nnunet_results_folder=Path(str(tool_config["nnunet_results_folder"])).expanduser().resolve(),
                device=device,
                mpl_config_dir=mpl_config_dir,
            )
            output_label = None
        returncode = int(run_info.returncode)

        if returncode != 0:
            return make_tool_result(
                output_root=output_root,
                tool_name="ct_tumor_seg",
                status="failure",
                identifier=case_id,
                metrics={"returncode": returncode},
                artifacts={},
                provenance=provenance,
                errors=[
                    run_info.stderr.strip()
                    or f"nnUNet prediction failed with return code {returncode}."
                ],
                payload={"case_id": case_id, "returncode": returncode},
            )

        prediction_path = prediction_dir / f"{case_name}.nii.gz"
        if not prediction_path.exists():
            candidates = sorted(prediction_dir.glob("*.nii.gz"))
            prediction_path = candidates[0] if candidates else None
        if prediction_path is None:
            return make_tool_result(
                output_root=output_root,
                tool_name="ct_tumor_seg",
                status="failure",
                identifier=case_id,
                metrics={"returncode": returncode},
                artifacts={},
                provenance=provenance,
                errors=[
                    f"nnUNet did not produce a segmentation file for case '{case_id}'.",
                    f"nnUNet return code: {returncode}",
                ],
                payload={"case_id": case_id, "returncode": returncode},
            )

        copy_or_extract_label(prediction_path, final_segmentation_path, output_label)

    return make_tool_result(
        output_root=output_root,
        tool_name="ct_tumor_seg",
        status="success",
        identifier=case_id,
        metrics={"returncode": returncode},
        artifacts={"segmentation_path": str(final_segmentation_path)},
        provenance=provenance,
        payload={
            "case_id": case_id,
            "segmentation_path": str(final_segmentation_path),
            "returncode": returncode,
        },
    )
