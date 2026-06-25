from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tool_utils import quiet_tool_logs, save_snapshot


def case_id_from_case(case: Mapping[str, Any]) -> str:
    return str(case.get("Case_ID", "") or case.get("case_id", "") or "unknown_case")


def summarize_qc_mask(
    mask_path: str,
    *,
    tissue_class_id: int,
    artifact_class_ids: Sequence[int],
    background_class_ids: Sequence[int],
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 1000000000
    mask_array = np.array(Image.open(mask_path))
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    tissue_pixels = int(np.count_nonzero(mask_array == tissue_class_id))
    artifact_pixels = int(
        sum(np.count_nonzero(mask_array == class_id) for class_id in artifact_class_ids)
    )
    background_pixels = int(
        sum(
            np.count_nonzero(mask_array == class_id)
            for class_id in background_class_ids
        )
    )
    informative_pixels = tissue_pixels + artifact_pixels
    tissue_rate = (
        round(tissue_pixels / informative_pixels, 6) if informative_pixels else 0.0
    )
    artifact_rate = (
        round(artifact_pixels / informative_pixels, 6) if informative_pixels else 0.0
    )
    return {
        "mask_path": mask_path,
        "tissue_pixels": tissue_pixels,
        "artifact_pixels": artifact_pixels,
        "background_pixels": background_pixels,
        "tissue_rate": tissue_rate,
        "artifact_rate": artifact_rate,
    }


def failure_summary(
    case_id: str,
    output_root: str,
    errors: Sequence[str],
) -> dict[str, Any]:
    summary = {
        "case_id": case_id,
        "errors": [str(item) for item in errors if str(item).strip()],
    }
    save_snapshot(output_root, f"wsi_qc/{case_id}", "selection_summary", summary)
    return summary


def build_wsi_case_summary(
    case_id: str,
    slide_summaries: Sequence[Mapping[str, Any]],
    *,
    output_root: str,
    max_artifact_rate: float,
    tissue_class_id: int,
    artifact_class_ids: Sequence[int],
    background_class_ids: Sequence[int],
    errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    slide_summaries = [dict(item) for item in slide_summaries]
    ranked_slides = sorted(
        slide_summaries,
        key=lambda slide: int(
            slide.get("usable_tissue_pixels", slide.get("tissue_pixels", 0)) or 0
        ),
        reverse=True,
    )
    selected_raw = ranked_slides[0] if ranked_slides else {}
    selected_usable_tissue_pixels = int(
        selected_raw.get("usable_tissue_pixels", selected_raw.get("tissue_pixels", 0))
        or 0
    )
    selected_artifact_rate = (
        float(selected_raw.get("artifact_rate"))
        if selected_raw.get("artifact_rate") is not None
        else 1.0
    )
    selected_passes = (
        bool(selected_raw)
        and selected_usable_tissue_pixels > 0
        and selected_artifact_rate <= max_artifact_rate
    )
    selected_slide = (
        {
            "case_id": case_id,
            "selected_slide_name": str(selected_raw.get("slide_name", "") or ""),
            "selected_slide_path": str(selected_raw.get("slide_path", "") or ""),
            "selected_output_dir": str(selected_raw.get("output_dir", "") or ""),
            "passes_threshold": bool(selected_passes),
            "selected_tissue_rate": float(selected_raw.get("tissue_rate", 0.0) or 0.0),
            "selected_artifact_rate": float(
                selected_raw.get("artifact_rate", 0.0) or 0.0
            ),
            "selected_usable_tissue_pixels": int(
                selected_raw.get(
                    "usable_tissue_pixels", selected_raw.get("tissue_pixels", 0)
                )
                or 0
            ),
            "selected_artifact_pixels": int(
                selected_raw.get("artifact_pixels", 0) or 0
            ),
        }
        if selected_raw
        else {}
    )
    summary = {
        "case_id": case_id,
        "case_qc_passes_threshold": bool(selected_passes),
        "thresholds": {
            "max_artifact_rate": max_artifact_rate,
            "tissue_class_id": tissue_class_id,
            "artifact_class_ids": [int(item) for item in artifact_class_ids],
            "background_class_ids": [int(item) for item in background_class_ids],
        },
        "selection_rule": "max_usable_tissue_pixels_then_artifact_threshold",
        "slide_summaries": slide_summaries,
        "selected_slide": selected_slide,
        "errors": [] if selected_passes else [str(item) for item in list(errors or [])],
    }
    save_snapshot(output_root, f"wsi_qc/{case_id}", "selection_summary", summary)
    return summary


def run_wsi_qc_cohort_worker(
    cases: list[Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    tool_config: Mapping[str, Any],
) -> dict[str, Any]:
    tool_config = dict(tool_config)
    device = str(tool_config["device"])
    max_artifact_rate = float(tool_config["max_artifact_rate"])
    tissue_class_id = int(tool_config["tissue_class_id"])
    artifact_class_ids = [
        int(class_id) for class_id in list(tool_config["artifact_class_ids"])
    ]
    background_class_ids = [
        int(class_id) for class_id in list(tool_config["background_class_ids"])
    ]

    from tools.pathology_qc.artifacts_seg import artifacts_seg
    from tools.pathology_qc.wsi_tis_detect import wsi_tis_detect

    case_slides: dict[str, list[dict[str, Any]]] = {}
    case_errors: dict[str, list[str]] = {}
    slide_tasks: list[dict[str, Any]] = []
    for case in cases:
        case_id = case_id_from_case(case)
        case_slides[case_id] = []
        case_errors[case_id] = []
        case_output_root = Path(output_root) / "wsi_qc" / case_id
        grandqc_root = case_output_root / "grandqc"
        grandqc_root.mkdir(parents=True, exist_ok=True)
        for wsi_record in list(case.get("WSI", []) or []):
            record = dict(wsi_record)
            slide_path = str(record.get("File Path", "") or "").strip()
            slide_name = os.path.basename(slide_path).replace(".svs", "")
            slide_output_dir = grandqc_root / slide_name
            task = {
                "case_id": case_id,
                "slide_name": slide_name,
                "slide_path": slide_path,
                "output_dir": str(slide_output_dir),
                "tissue_mask_path": str(slide_output_dir / "tis_det_mask" / f"{slide_name}_MASK.png"),
                "qc_mask_path": str(slide_output_dir / "mask_qc" / f"{slide_name}_mask.png"),
            }
            if not slide_path:
                task["error"] = "WSI record is missing File Path."
            elif not Path(slide_path).exists():
                task["error"] = f"WSI file was not found: {slide_path}"
            slide_tasks.append(task)
        if not list(case.get("WSI", []) or []):
            case_errors[case_id].append("No WSI records were provided.")

    show_progress = sys.stderr.isatty()
    for task in tqdm(slide_tasks, desc="WSI tissue detection", unit="slide", disable=not show_progress):
        if task.get("error"):
            continue
        tissue_mask_path = Path(str(task["tissue_mask_path"]))
        qc_mask_path = Path(str(task["qc_mask_path"]))
        if (
            tissue_mask_path.is_file()
            and tissue_mask_path.stat().st_size > 0
        ) or (
            qc_mask_path.is_file()
            and qc_mask_path.stat().st_size > 0
        ):
            task["tissue_detection_reused"] = True
            continue
        try:
            with quiet_tool_logs():
                wsi_tis_detect(
                    str(task["slide_path"]),
                    str(task["output_dir"]),
                    {
                        **dict(tool_config["tissue_detect"]),
                        "config_dir": config_dir,
                        "device": device,
                    },
                )
        except Exception as exc:
            task["error"] = f"{type(exc).__name__}: {exc}"

    for task in tqdm(slide_tasks, desc="WSI artifact segmentation", unit="slide", disable=not show_progress):
        if task.get("error"):
            continue
        qc_mask_path = Path(str(task["qc_mask_path"]))
        if qc_mask_path.is_file() and qc_mask_path.stat().st_size > 0:
            task["artifact_segmentation_reused"] = True
            continue
        try:
            with quiet_tool_logs():
                artifacts_seg(
                    str(task["slide_path"]),
                    str(task["output_dir"]),
                    {
                        **dict(tool_config["artifact_seg"]),
                        "config_dir": config_dir,
                        "device": device,
                    },
                )
        except Exception as exc:
            task["error"] = f"{type(exc).__name__}: {exc}"

    for task in tqdm(slide_tasks, desc="WSI mask summary", unit="slide", disable=not show_progress):
        case_id = str(task["case_id"])
        slide_name = str(task["slide_name"])
        slide_output_dir = Path(str(task["output_dir"]))
        if task.get("error"):
            error = f"{slide_name}: {task['error']}"
            case_errors.setdefault(case_id, []).append(error)
            case_slides.setdefault(case_id, []).append(
                {
                    "case_id": case_id,
                    "slide_name": slide_name,
                    "slide_path": str(task.get("slide_path", "") or ""),
                    "output_dir": str(slide_output_dir),
                    "passes_threshold": False,
                    "tissue_rate": 0.0,
                    "artifact_rate": 1.0,
                    "tissue_pixels": 0,
                    "usable_tissue_pixels": 0,
                    "artifact_pixels": 0,
                    "error": str(task["error"]),
                }
            )
            continue
        try:
            mask_path = Path(str(task["qc_mask_path"]))
            if not mask_path.exists():
                candidates = sorted((slide_output_dir / "mask_qc").glob("*_mask.png"))
                mask_path = candidates[0] if candidates else mask_path
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"GrandQC mask was not written for {slide_name}."
                )
            mask_summary = summarize_qc_mask(
                str(mask_path),
                tissue_class_id=tissue_class_id,
                artifact_class_ids=artifact_class_ids,
                background_class_ids=background_class_ids,
            )
            usable_tissue_pixels = int(mask_summary["tissue_pixels"])
            artifact_rate = float(mask_summary["artifact_rate"])
            case_slides.setdefault(case_id, []).append(
                {
                    "case_id": case_id,
                    "slide_name": slide_name,
                    "slide_path": str(task["slide_path"]),
                    "output_dir": str(slide_output_dir),
                    "passes_threshold": bool(
                        usable_tissue_pixels > 0 and artifact_rate <= max_artifact_rate
                    ),
                    "tissue_rate": float(mask_summary["tissue_rate"]),
                    "artifact_rate": artifact_rate,
                    "usable_tissue_pixels": usable_tissue_pixels,
                    **mask_summary,
                }
            )
        except Exception as exc:
            error = f"{slide_name}: {type(exc).__name__}: {exc}"
            case_errors.setdefault(case_id, []).append(error)
            case_slides.setdefault(case_id, []).append(
                {
                    "case_id": case_id,
                    "slide_name": slide_name,
                    "slide_path": str(task.get("slide_path", "") or ""),
                    "output_dir": str(slide_output_dir),
                    "passes_threshold": False,
                    "tissue_rate": 0.0,
                    "artifact_rate": 1.0,
                    "tissue_pixels": 0,
                    "usable_tissue_pixels": 0,
                    "artifact_pixels": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summaries = {}
    for case_id in tqdm(list(case_slides), desc="WSI case selection", unit="case", disable=not show_progress):
        summaries[case_id] = build_wsi_case_summary(
            case_id,
            case_slides.get(case_id, []),
            output_root=output_root,
            max_artifact_rate=max_artifact_rate,
            tissue_class_id=tissue_class_id,
            artifact_class_ids=artifact_class_ids,
            background_class_ids=background_class_ids,
            errors=case_errors.get(case_id, []),
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
    }


def run_wsi_qc(
    case_id: str,
    item: list[Mapping[str, Any]],
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    result = run_wsi_qc_cohort(
        [{"Case_ID": str(case_id), "WSI": list(item)}],
        output_root=output_root,
        config_dir=config_dir,
    )
    return dict(result.get("selection_summaries", {}).get(str(case_id), {}))


def run_wsi_qc_cohort(
    cases: list[Mapping[str, Any]],
    output_root: str = "",
    config_dir: str = "",
) -> dict[str, Any]:

    import yaml

    summaries: dict[str, Any] = {}
    pending_cases = []
    for case in cases:
        case_id = case_id_from_case(case)
        summary_path = Path(output_root) / "wsi_qc" / case_id / "selection_summary.json"
        if summary_path.exists() and summary_path.read_text(encoding="utf-8").strip():
            summaries[case_id] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            pending_cases.append(case)
    if not pending_cases:
        return {"case_count": len(cases), "selection_summaries": summaries}

    tool_config = yaml.safe_load((Path(config_dir).expanduser() / "wsi_qc.yaml").read_text(encoding="utf-8")) or {}
    tool_config["config_dir"] = config_dir
    python_executable = Path(str(tool_config["python_executable"])).expanduser()
    if not python_executable.exists():
        for case in pending_cases:
            case_id = case_id_from_case(case)
            summaries[case_id] = failure_summary(
                case_id,
                output_root,
                errors=[f"WSI QC python environment was not found: {python_executable}"],
            )
        return {"case_count": len(cases), "selection_summaries": summaries}

    run_config = json.dumps(
        {
            "cases": pending_cases,
            "output_root": output_root,
            "config_dir": config_dir,
            "tool_config": tool_config,
        },
        ensure_ascii=False,
    )
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "--grandqc-cohort-worker",
    ]
    env = dict(os.environ)
    project_dir = Path(config_dir).expanduser().resolve().parent
    python_path_items = [str(project_dir)]
    if env.get("PYTHONPATH"):
        python_path_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_items)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["PYTHONUNBUFFERED"] = "1"
    worker_stderr = None if sys.stderr.isatty() else subprocess.PIPE
    completed = subprocess.run(
        command,
        cwd=str(project_dir),
        env=env,
        input=run_config,
        stdout=subprocess.PIPE,
        stderr=worker_stderr,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        for case in pending_cases:
            case_id = case_id_from_case(case)
            summaries[case_id] = failure_summary(
                case_id,
                output_root,
                errors=[
                    f"WSI QC failed with return code {completed.returncode}.",
                    (completed.stderr or "").strip() or completed.stdout.strip(),
                ],
            )
        return {"case_count": len(cases), "selection_summaries": summaries}

    try:
        result = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        result = {}
    summaries.update(dict(result.get("selection_summaries", {}) or {}))
    for case in pending_cases:
        case_id = case_id_from_case(case)
        if case_id not in summaries:
            summaries[case_id] = failure_summary(
                case_id,
                output_root,
                errors=["WSI QC finished without writing selection_summary.json."],
            )
    result["selection_summaries"] = summaries
    result["case_count"] = len(cases)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WSI quality control.")
    parser.add_argument("--grandqc-cohort-worker", action="store_true")
    args = parser.parse_args()
    if args.grandqc_cohort_worker:
        payload = json.loads(sys.stdin.read() or "{}")
        result = run_wsi_qc_cohort_worker(
            list(payload.get("cases", []) or []),
            str(payload.get("output_root", "") or "output"),
            config_dir=str(payload.get("config_dir", "") or ""),
            tool_config=dict(payload.get("tool_config", {}) or {}),
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
