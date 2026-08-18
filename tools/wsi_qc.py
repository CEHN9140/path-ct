from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.cache_utils import file_identity, hash_payload, semantic_config
from utils.tool_utils import (
    quiet_tool_logs,
    run_json_workers,
    save_snapshot,
    split_device_requests,
    to_jsonable,
)

WSI_QC_CACHE_VERSION = 1
WSI_QC_RUNTIME_KEYS = {
    "device",
    "devices",
    "workers",
    "workers_per_gpu",
    "num_workers",
    "batch_size",
    "pin_memory",
    "prefetch_factor",
}


def wsi_qc_config_signature(
    config: Mapping[str, Any], *, include_device_indices: bool = False
) -> str:
    runtime_keys = set() if include_device_indices else WSI_QC_RUNTIME_KEYS
    return hash_payload(
        {
            "cache_version": WSI_QC_CACHE_VERSION,
            "semantic_config": semantic_config(config, runtime_keys),
        }
    )


def wsi_qc_stage_config_signature(config: Mapping[str, Any]) -> str:
    stage_config = {
        key: dict(config)[key]
        for key in ("python_executable", "tissue_detect", "artifact_seg")
    }
    return wsi_qc_config_signature(stage_config)


def wsi_case_cache_signature(
    case: Mapping[str, Any], config_signature: str
) -> str:
    records = []
    for raw_record in list(case.get("WSI", []) or []):
        record = dict(raw_record)
        source_path = str(record.get("File Path", "") or "").strip()
        input_identity = (
            file_identity(source_path) if source_path and Path(source_path).exists()
            else {"path": source_path}
        )
        records.append(
            {"record": to_jsonable(record), "file_identity": input_identity}
        )
    records.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return hash_payload({
        "cache_version": WSI_QC_CACHE_VERSION,
        "case_id": case_id_from_case(case),
        "config_signature": config_signature,
        "wsi_records": records,
    })


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
    cache_signature: str = "",
    grandqc_stage_signature: str = "",
) -> dict[str, Any]:
    summary = {
        "case_id": case_id,
        "cache_signature": cache_signature,
        "grandqc_stage_signature": grandqc_stage_signature,
        "cacheable": False,
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
    cache_signature: str = "",
    grandqc_stage_signature: str = "",
) -> dict[str, Any]:
    slide_summaries = [dict(item) for item in slide_summaries]
    for slide in slide_summaries:
        usable_tissue_pixels = int(
            slide.get("usable_tissue_pixels", slide.get("tissue_pixels", 0)) or 0
        )
        artifact_rate = float(slide.get("artifact_rate", 1.0) or 0.0)
        slide["passes_threshold"] = bool(
            usable_tissue_pixels > 0 and artifact_rate <= max_artifact_rate
        )
        slide["artifact_rate_exceeds_threshold"] = bool(
            artifact_rate > max_artifact_rate
        )
    ranked_slides = sorted(
        slide_summaries,
        key=lambda slide: int(
            slide.get("usable_tissue_pixels", slide.get("tissue_pixels", 0)) or 0
        ),
        reverse=True,
    )
    eligible_slides = [
        slide for slide in ranked_slides if bool(slide["passes_threshold"])
    ]
    selected_raw = (
        eligible_slides[0]
        if eligible_slides
        else (ranked_slides[0] if ranked_slides else {})
    )
    selected_usable_tissue_pixels = int(
        selected_raw.get("usable_tissue_pixels", selected_raw.get("tissue_pixels", 0))
        or 0
    )
    selected_artifact_rate = float(selected_raw.get("artifact_rate", 1.0) or 0.0)
    selected_passes = bool(selected_raw) and bool(
        selected_raw.get("passes_threshold")
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
            "artifact_rate_exceeds_threshold": bool(
                selected_artifact_rate > max_artifact_rate
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
    summary_errors = [str(item) for item in list(errors or []) if str(item).strip()]
    for slide in slide_summaries:
        slide_error = str(slide.get("error", "") or "").strip()
        if slide_error and slide_error not in summary_errors:
            summary_errors.append(slide_error)
    summary = {
        "case_id": case_id,
        "cache_signature": cache_signature,
        "grandqc_stage_signature": grandqc_stage_signature,
        "cacheable": not summary_errors,
        "case_qc_passes_threshold": bool(selected_passes),
        "thresholds": {
            "max_artifact_rate": max_artifact_rate,
            "tissue_class_id": tissue_class_id,
            "artifact_class_ids": [int(item) for item in artifact_class_ids],
            "background_class_ids": [int(item) for item in background_class_ids],
        },
        "selection_rule": "artifact_rate_threshold_then_max_usable_tissue_pixels",
        "slide_summaries": slide_summaries,
        "selected_slide": selected_slide,
        "errors": summary_errors,
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
    case_cache_signatures = dict(tool_config.get("case_cache_signatures", {}) or {})
    case_stage_signatures = dict(tool_config.get("case_stage_signatures", {}) or {})
    force_case_ids = set(tool_config.get("force_case_ids", []) or [])

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
                "force_recompute": case_id in force_case_ids,
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
        reusable_stage_output = (
            tissue_mask_path.is_file() and tissue_mask_path.stat().st_size > 0
        ) or (qc_mask_path.is_file() and qc_mask_path.stat().st_size > 0)
        if not task["force_recompute"] and reusable_stage_output:
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
        if (
            not task["force_recompute"]
            and qc_mask_path.is_file()
            and qc_mask_path.stat().st_size > 0
        ):
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
                        usable_tissue_pixels > 0
                        and artifact_rate <= max_artifact_rate
                    ),
                    "artifact_rate_exceeds_threshold": bool(
                        artifact_rate > max_artifact_rate
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
            cache_signature=str(case_cache_signatures.get(case_id, "") or ""),
            grandqc_stage_signature=str(
                case_stage_signatures.get(case_id, "") or ""
            ),
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

    tool_config = yaml.safe_load(
        (Path(config_dir).expanduser() / "wsi_qc.yaml").read_text(
            encoding="utf-8"
        )
    )
    config_signature = wsi_qc_config_signature(tool_config)
    summaries: dict[str, Any] = {}
    pending_cases = []
    case_cache_signatures = {}
    case_stage_signatures = {}
    force_case_ids = []
    for case in cases:
        case_id = case_id_from_case(case)
        cache_signature = wsi_case_cache_signature(case, config_signature)
        case_cache_signatures[case_id] = cache_signature
        case_stage_signatures[case_id] = cache_signature
        summary_path = Path(output_root) / "wsi_qc" / case_id / "selection_summary.json"
        summary = {}
        if summary_path.exists() and summary_path.read_text(encoding="utf-8").strip():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_cacheable = bool(
                summary.get("cacheable", not list(summary.get("errors", []) or []))
            )
            saved_cache_signature = str(summary.get("cache_signature", "") or "")
            selected_path = str(
                dict(summary.get("selected_slide") or {}).get("selected_slide_path", "")
                or ""
            )
            selected_mask_path = next(
                (
                    str(slide.get("mask_path", "") or "")
                    for slide in list(summary.get("slide_summaries", []) or [])
                    if str(slide.get("slide_path", "") or "") == selected_path
                ),
                "",
            )
            if (
                summary_cacheable
                and saved_cache_signature == cache_signature
                and selected_mask_path
                and Path(selected_mask_path).is_file()
            ):
                summaries[case_id] = summary
                continue
            if not summary_cacheable or saved_cache_signature != cache_signature:
                force_case_ids.append(case_id)
        pending_cases.append(case)
    if not pending_cases:
        return {"case_count": len(cases), "selection_summaries": summaries}

    tool_config["config_dir"] = config_dir
    tool_config["case_cache_signatures"] = {
        case_id_from_case(case): case_cache_signatures[case_id_from_case(case)]
        for case in pending_cases
    }
    tool_config["case_stage_signatures"] = {
        case_id_from_case(case): case_stage_signatures[case_id_from_case(case)]
        for case in pending_cases
    }
    tool_config["force_case_ids"] = force_case_ids
    python_executable = Path(str(tool_config["python_executable"])).expanduser()
    if not python_executable.exists():
        for case in pending_cases:
            case_id = case_id_from_case(case)
            summaries[case_id] = failure_summary(
                case_id,
                output_root,
                errors=[f"WSI QC python environment was not found: {python_executable}"],
                cache_signature=case_cache_signatures[case_id],
                grandqc_stage_signature=case_stage_signatures[case_id],
            )
        return {"case_count": len(cases), "selection_summaries": summaries}

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
    env["PYTHONUNBUFFERED"] = "1"
    devices = list(tool_config["devices"])
    assignments, visible_devices = split_device_requests(
        pending_cases,
        devices,
        workers_per_device=int(tool_config["workers_per_gpu"]),
    )
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    payloads = [
        {
            "cases": assignment["requests"],
            "output_root": output_root,
            "config_dir": config_dir,
            "tool_config": {**tool_config, "device": assignment["device"]},
            "print_result": False,
        }
        for assignment in assignments
    ]
    returncodes = run_json_workers(command, payloads, cwd=str(project_dir), env=env)
    for assignment, returncode in zip(assignments, returncodes):
        for case in assignment["requests"]:
            case_id = case_id_from_case(case)
            summary_path = (
                Path(output_root) / "wsi_qc" / case_id / "selection_summary.json"
            )
            if summary_path.exists():
                summaries[case_id] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
            else:
                summaries[case_id] = failure_summary(
                    case_id,
                    output_root,
                    errors=[f"WSI QC worker failed with return code {returncode}."],
                    cache_signature=case_cache_signatures[case_id],
                    grandqc_stage_signature=case_stage_signatures[case_id],
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
        if payload.get("print_result", True):
            print(json.dumps(result, ensure_ascii=False))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
