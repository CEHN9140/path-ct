from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from agents.common import (
    add_execution_errors,
    add_tool_result,
    announce_tool_action,
    case_from_state,
    ct_input_identity,
    load_selected_ct_record,
    load_selected_wsi_record,
    load_tool_snapshot,
    selected_ct_from_result,
)
from agents.inventory import build_patient_state
from utils.cache_utils import file_identity, hash_payload, semantic_config
from utils.llm_utils import load_yaml_file
from utils.tool_utils import (
    make_tool_result,
    safe_identifier,
    save_snapshot,
)

WSI_PATCH_RUNTIME_KEYS = {"batch_size", "num_workers"}
WSI_TUMOR_SEG_RUNTIME_KEYS = {
    "python_executable",
    "device",
    "devices",
    "batch_size",
    "workers_per_gpu",
}
CT_TUMOR_SEG_RUNTIME_KEYS = {
    "device",
    "devices",
    "workers_per_gpu",
    "num_processes_preprocessing",
    "num_processes_segmentation_export",
    "mpl_config_dir",
}


def wsi_patch(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> Mapping[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_wsi_record = load_selected_wsi_record(case, output_root, case_id)
    selected_slide_path = str(selected_wsi_record.get("File Path", "") or "")
    patch_config = load_yaml_file(Path(config_dir).expanduser() / "wsi_patch.yaml")
    patch_size = int(patch_config["patch_size"])
    target_mpp = float(patch_config["target_mpp"])
    patch_semantic_config = semantic_config(patch_config, WSI_PATCH_RUNTIME_KEYS)
    wsi_qc_summary_path = (
        Path(output_root) / "wsi_qc" / case_id / "selection_summary.json"
    )
    if not wsi_qc_summary_path.is_file():
        raise FileNotFoundError(f"WSI QC summary is missing for {case_id}.")
    wsi_qc_summary = json.loads(wsi_qc_summary_path.read_text(encoding="utf-8"))
    selected_slide_summary = dict(wsi_qc_summary.get("selected_slide") or {})
    qc_mask_path = next(
        (
            str(slide.get("mask_path", "") or "")
            for slide in list(wsi_qc_summary.get("slide_summaries", []) or [])
            if str(slide.get("slide_path", "") or "") == selected_slide_path
        ),
        "",
    )
    if not qc_mask_path:
        output_dir = str(selected_slide_summary.get("selected_output_dir", "") or "")
        slide_name = str(selected_slide_summary.get("selected_slide_name", "") or "")
        qc_mask_path = str(Path(output_dir) / "mask_qc" / f"{slide_name}_mask.png")
    patch_cache_signature = hash_payload(
        {
            "cache_version": 1,
            "semantic_config": patch_semantic_config,
            "inputs": {
                "slide": file_identity(selected_slide_path),
                "qc_mask": file_identity(qc_mask_path),
            },
        }
    )
    cached_bundle = load_tool_snapshot(
        output_root,
        "wsi_patch",
        case_id,
        required_artifact_keys=["patch_dir", "coordinates_h5_path"],
    )
    cached_payload = (
        dict(cached_bundle.get("payload", {}) or {})
        if cached_bundle is not None
        else {}
    )
    if (
        cached_bundle is not None
        and cached_payload.get("cache_signature") == patch_cache_signature
    ):
        announce_tool_action("wsi_patch", case_id, "Reuse cached tool `wsi_patch`.")
        return add_tool_result(
            state,
            bucket_name="wsi_evidence",
            evidence_key="patch_extraction",
            tool_result=dict(cached_bundle.get("tool_result", {}) or {}),
            node="wsi_patch",
        )

    announce_tool_action("wsi_patch", case_id, "Call tool `wsi_patch`.")
    from tools.wsi_patch import run_seg_wsi_patch

    patch_result = run_seg_wsi_patch(
        case_id=case_id,
        wsi_records=[selected_wsi_record],
        output_root=output_root,
        config_dir=config_dir,
    )
    if patch_result.get("errors"):
        raise RuntimeError(str(list(patch_result.get("errors") or [])[0]))
    patch_slide = dict(list(patch_result.get("slides", []) or [{}])[0] or {})
    patch_info = {
        "patch_dir": str(patch_slide.get("patch_dir", "") or ""),
        "patch_count": int(patch_slide.get("patch_count", 0) or 0),
        "patch_size": int(patch_result.get("patch_size", 0) or 0),
        "cache_signature": patch_cache_signature,
        "level": int(patch_slide.get("level", 0) or 0),
        "coordinate_space": str(
            patch_slide.get(
                "coordinate_space", patch_result.get("coordinate_space", "")
            )
            or ""
        ),
        "target_mpp": float(
            patch_slide.get("target_mpp", patch_result.get("target_mpp", 0.0))
            or 0.0
        ),
        "source_mpp_x": float(patch_slide.get("source_mpp_x", 0.0) or 0.0),
        "source_mpp_y": float(patch_slide.get("source_mpp_y", 0.0) or 0.0),
        "source_patch_width": int(
            patch_slide.get("source_patch_width", 0) or 0
        ),
        "source_patch_height": int(
            patch_slide.get("source_patch_height", 0) or 0
        ),
        "coordinates_h5_path": str(patch_slide.get("coordinates_h5_path", "") or ""),
        "qc_mask_path": str(patch_slide.get("qc_mask_path", "") or ""),
        "slide_name": str(patch_slide.get("slide_name", "") or ""),
        "slide_path": str(patch_slide.get("slide_path", "") or selected_slide_path),
        "qc_clean_tissue_value": int(
            patch_slide.get(
                "qc_clean_tissue_value", patch_result.get("qc_clean_tissue_value", 0)
            )
            or 0
        ),
    }
    patch_tool_result = make_tool_result(
        output_root=output_root,
        tool_name="wsi_patch",
        status="success",
        identifier=case_id,
        metrics={
            "patch_count": patch_info["patch_count"],
            "patch_size": patch_info["patch_size"],
            "level": patch_info["level"],
            "target_mpp": patch_info["target_mpp"],
        },
        artifacts={
            "patch_dir": patch_info["patch_dir"],
            "coordinates_h5_path": patch_info["coordinates_h5_path"],
            "qc_mask_path": patch_info["qc_mask_path"],
        },
        provenance={
            "backend": "openslide",
            "case_id": case_id,
            "slide_name": patch_info["slide_name"],
            "coordinate_space": patch_info["coordinate_space"],
            "source_mpp_x": patch_info["source_mpp_x"],
            "source_mpp_y": patch_info["source_mpp_y"],
        },
        errors=[],
        payload=patch_info,
    )
    return add_tool_result(
        state,
        bucket_name="wsi_evidence",
        evidence_key="patch_extraction",
        tool_result=patch_tool_result,
        node="wsi_patch",
    )


def wsi_tumor_seg_cache_signature(
    config: Mapping[str, Any], slide_path: str, patch_bundle: Mapping[str, Any]
) -> str:
    semantic_config_value = semantic_config(config, WSI_TUMOR_SEG_RUNTIME_KEYS)
    artifacts = dict(
        dict(patch_bundle.get("tool_result", {}) or {}).get("artifacts", {}) or {}
    )
    return hash_payload(
        {
            "cache_version": 1,
            "semantic_config": semantic_config_value,
            "upstream": {
                "wsi_patch": str(
                    dict(patch_bundle.get("payload", {}) or {}).get(
                        "cache_signature", ""
                    )
                ),
                "coordinates": file_identity(str(artifacts["coordinates_h5_path"])),
            },
            "model": {
                "model": file_identity(str(config["model_path"])),
                "inference_config": file_identity(str(config["inference_config_path"])),
                "code": file_identity(
                    str(Path(__file__).resolve().parent.parent / "tools" / "wsi_tumor_seg.py")
                ),
            },
        }
    )


def wsi_tumor_seg_context(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> dict[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    slide_path = str(
        load_selected_wsi_record(case, output_root, case_id).get("File Path", "") or ""
    )
    config = load_yaml_file(Path(config_dir).expanduser() / "wsi_tumor_seg.yaml")
    patch_bundle = load_tool_snapshot(
        output_root,
        "wsi_patch",
        case_id,
        required_artifact_keys=["patch_dir", "coordinates_h5_path"],
    )
    if patch_bundle is None:
        raise RuntimeError(f"WSI patch snapshot is missing for {case_id}.")
    signature = wsi_tumor_seg_cache_signature(config, slide_path, patch_bundle)
    cached = load_tool_snapshot(
        output_root,
        "wsi_tumor_seg",
        case_id,
        required_artifact_keys=[
            "tumor_coordinates_h5_path",
            "patch_probabilities_path",
            "summary_path",
            "tumor_probability_map_path",
            "overlay_path",
        ],
    )
    cached_payload = dict(cached.get("payload", {}) or {}) if cached else {}
    artifacts = dict(
        dict(patch_bundle.get("tool_result", {}) or {}).get("artifacts", {}) or {}
    )
    return {
        "case_id": case_id,
        "cached": cached,
        "reuse": bool(
            cached
            and cached_payload.get("cache_signature") == signature
        ),
        "request": {
            "case_id": case_id,
            "slide_path": slide_path,
            "coordinates_h5_path": str(artifacts["coordinates_h5_path"]),
            "source_patch_dir": str(artifacts["patch_dir"]),
            "cache_signature": signature,
        },
    }


def add_wsi_tumor_seg_result(
    state: Mapping[str, Any], result: Mapping[str, Any]
) -> Mapping[str, Any]:
    updated = add_tool_result(
        state,
        bucket_name="wsi_evidence",
        evidence_key="tumor_segmentation",
        tool_result=result,
        node="wsi_tumor_seg",
    )
    if str(result.get("status", "")) == "failure":
        failed = add_execution_errors(
            updated, "wsi_tumor_seg", list(result.get("errors", []) or [])
        )
        return {**failed, "qc": "fail"}
    tumor_count = int(dict(result.get("metrics", {}) or {}).get("tumor_patch_count", 0))
    return dict(updated) if tumor_count > 0 else {**dict(updated), "qc": "fail"}


def wsi_tumor_seg(
    states: list[dict[str, Any]], *, output_root: str, config_dir: str
) -> list[dict[str, Any]]:
    contexts = {
        state["case_id"]: wsi_tumor_seg_context(
            state, output_root=output_root, config_dir=config_dir
        )
        for state in states
        if state.get("qc") == "success"
    }
    requests = []
    reused_count = 0
    for context in contexts.values():
        if context["reuse"]:
            reused_count += 1
        else:
            requests.append(context["request"])
    print(
        f"[wsi_tumor_seg] queued={len(requests)}, cached={reused_count}",
        flush=True,
    )
    from tools.wsi_tumor_seg import run_wsi_tumor_seg_cohort

    new_results = run_wsi_tumor_seg_cohort(requests, output_root, config_dir)
    updated_states = []
    for state in states:
        if state.get("qc") != "success":
            updated_states.append(dict(state))
            continue
        context = contexts[state["case_id"]]
        result = (
            dict(context["cached"].get("tool_result", {}) or {})
            if context["reuse"]
            else new_results[state["case_id"]]
        )
        updated_states.append(dict(add_wsi_tumor_seg_result(state, result)))
    return updated_states


def nifti_geometry_matches(ct_path: str, mask_path: str) -> bool:
    import nibabel as nib
    import numpy as np

    if not Path(ct_path).is_file() or not Path(mask_path).is_file():
        return False
    try:
        ct_image = nib.load(ct_path)
        mask_image = nib.load(mask_path)
    except Exception:
        return False
    return (
        ct_image.shape == mask_image.shape
        and np.allclose(ct_image.affine, mask_image.affine, atol=1e-4)
    )


def ct_tumor_seg_config_signature(config: Mapping[str, Any]) -> str:
    return hash_payload(
        {
            "cache_version": 1,
            "semantic_config": semantic_config(config, CT_TUMOR_SEG_RUNTIME_KEYS),
        }
    )


def ct_tumor_seg_cache_signature(
    config: Mapping[str, Any], ct_path: str, ct_identity: Mapping[str, Any]
) -> str:
    return hash_payload(
        {
            "cache_version": 1,
            "semantic_config": semantic_config(config, CT_TUMOR_SEG_RUNTIME_KEYS),
            "inputs": {"ct_identity": dict(ct_identity), "ct_file": file_identity(ct_path)},
        }
    )


def ct_tumor_seg_context(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> dict[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_ct_record = load_selected_ct_record(case, output_root, case_id)
    ct_path = str(selected_ct_record.get("File Path", "") or "")
    current_ct_identity = ct_input_identity(selected_ct_record)
    cached_mask_path = (
        Path(output_root)
        / "ct_tumor_seg"
        / safe_identifier(case_id)
        / f"{case_id}_mask.nii.gz"
    )
    tool_config = load_yaml_file(Path(config_dir).expanduser() / "ct_tumor_seg.yaml")
    current_cache_signature = ct_tumor_seg_cache_signature(
        tool_config, ct_path, current_ct_identity
    )
    snapshot_path = (
        Path(output_root) / "ct_tumor_seg" / f"{safe_identifier(case_id)}.json"
    )
    cached_tool_result = {}
    cached_payload = {}
    if snapshot_path.exists():
        try:
            cached_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            cached_tool_result = dict(cached_snapshot.get("tool_result", {}) or {})
            cached_payload = dict(cached_snapshot.get("payload", {}) or {})
        except Exception:
            cached_tool_result = {}
            cached_payload = {}
    cached_provenance = dict(cached_tool_result.get("provenance", {}) or {})
    reuse_existing_mask = (
        cached_mask_path.exists()
        and cached_payload.get("cache_signature") == current_cache_signature
        and nifti_geometry_matches(ct_path, str(cached_mask_path))
    )
    if snapshot_path.exists() and not reuse_existing_mask:
        if cached_mask_path.parent.is_dir():
            shutil.rmtree(cached_mask_path.parent)
        snapshot_path.unlink()
    reused_tool_result = {}
    if reuse_existing_mask:
        reused_tool_result = make_tool_result(
            output_root=output_root,
            tool_name="ct_tumor_seg",
            status="success",
            identifier=case_id,
            metrics={"returncode": 0},
            artifacts={"segmentation_path": str(cached_mask_path)},
            provenance=cached_provenance
            or {"case_id": case_id},
            payload={
                "case_id": case_id,
                "segmentation_path": str(cached_mask_path),
                "input_ct_path": ct_path,
                "cache_signature": current_cache_signature,
                "reused_existing_mask": True,
            },
        )
    return {
        "case_id": case_id,
        "ct_path": ct_path,
        "ct_identity": current_ct_identity,
        "cache_signature": current_cache_signature,
        "mask_path": str(cached_mask_path),
        "reused_tool_result": reused_tool_result,
        "reuse_existing_mask": reuse_existing_mask,
    }


def finalize_ct_tumor_seg(
    state: Mapping[str, Any],
    tumor_seg_result: Mapping[str, Any],
    context: Mapping[str, Any],
    output_root: str,
) -> Mapping[str, Any]:
    import nibabel as nib
    import numpy as np

    case_id = str(context["case_id"])
    ct_path = str(context["ct_path"])
    current_ct_identity = dict(context["ct_identity"])
    tumor_seg_result = dict(tumor_seg_result)
    tumor_seg_payload = {
        "input_ct_path": ct_path,
        "ct_identity": current_ct_identity,
        "cache_signature": str(context["cache_signature"]),
        "reused_existing_mask": bool(context["reuse_existing_mask"]),
    }
    errors = [
        str(item).strip()
        for item in list(tumor_seg_result.get("errors", []) or [])
        if str(item).strip()
    ]
    if str(tumor_seg_result.get("status", "") or "") == "failure":
        artifacts = dict(tumor_seg_result.get("artifacts", {}) or {})
        artifacts["saved_output_path"] = str(
            Path(output_root) / "ct_tumor_seg" / f"{safe_identifier(case_id)}.json"
        )
        tumor_seg_result["artifacts"] = artifacts
        save_snapshot(
            output_root,
            "ct_tumor_seg",
            case_id,
            {
                "tool_result": tumor_seg_result,
                "payload": {
                    "case_id": case_id,
                    "ct_identity": current_ct_identity,
                    "cache_signature": str(context["cache_signature"]),
                    "input_ct_path": ct_path,
                },
            },
        )
        patient_state = add_tool_result(
            state,
            bucket_name="ct_evidence",
            evidence_key="tumor_segmentation",
            tool_result=tumor_seg_result,
            node="ct_tumor_seg",
        )
        failed_state = add_execution_errors(
            patient_state,
            "ct_tumor_seg",
            errors,
        )
        return {**failed_state, "qc": "fail"}

    metrics = dict(tumor_seg_result.get("metrics", {}) or {})
    artifacts = dict(tumor_seg_result.get("artifacts", {}) or {})
    segmentation_path = str(artifacts.get("segmentation_path", "") or "")
    segmentation_exists = bool(segmentation_path and Path(segmentation_path).exists())
    voxel_spacing, positive_voxel_count, positive_volume_mm3 = [], 0, 0.0
    if segmentation_exists:
        image = nib.load(segmentation_path)
        data = np.asarray(image.dataobj)
        voxel_spacing = [float(item) for item in image.header.get_zooms()[:3]]
        positive_voxel_count = int(np.count_nonzero(data > 0))
        if voxel_spacing:
            positive_volume_mm3 = positive_voxel_count * float(np.prod(voxel_spacing))
    geometry_matches = segmentation_exists and nifti_geometry_matches(
        ct_path, segmentation_path
    )
    passes_threshold = (
        segmentation_exists and geometry_matches and positive_voxel_count > 0
    )
    mask_qc_errors = [
        message
        for condition, message in (
            (not segmentation_exists, "Tumor segmentation file is missing."),
            (
                segmentation_exists and not geometry_matches,
                "Tumor mask geometry does not match the selected CT.",
            ),
            (
                segmentation_exists and positive_voxel_count <= 0,
                "Tumor mask QC failed: empty tumor mask.",
            ),
        )
        if condition
    ]
    metrics.update(
        {
            "positive_voxel_count": positive_voxel_count,
            "positive_volume_mm3": positive_volume_mm3,
        }
    )
    if mask_qc_errors:
        tumor_seg_result["status"] = "failure"
        existing_errors = [
            str(item).strip()
            for item in list(tumor_seg_result.get("errors", []) or [])
            if str(item).strip()
        ]
        tumor_seg_result["errors"] = existing_errors + [
            item for item in mask_qc_errors if item not in existing_errors
        ]
    tumor_seg_result["metrics"] = metrics
    artifacts["saved_output_path"] = str(
        Path(output_root) / "ct_tumor_seg" / f"{safe_identifier(case_id)}.json"
    )
    tumor_seg_result["artifacts"] = artifacts
    tumor_seg_payload.update(
        {
            "case_id": case_id,
            "segmentation_path": segmentation_path,
            "tumor_mask_qc": {
                "segmentation_exists": segmentation_exists,
                "has_positive_voxels": positive_voxel_count > 0,
                "geometry_matches_selected_ct": geometry_matches,
                "positive_voxel_count": positive_voxel_count,
                "positive_volume_mm3": positive_volume_mm3,
                "passes_threshold": passes_threshold,
            },
            "voxel_spacing": voxel_spacing,
        }
    )
    save_snapshot(
        output_root,
        "ct_tumor_seg",
        case_id,
        {"tool_result": tumor_seg_result, "payload": tumor_seg_payload},
    )
    patient_state = add_tool_result(
        state,
        bucket_name="ct_evidence",
        evidence_key="tumor_segmentation",
        tool_result=tumor_seg_result,
        node="ct_tumor_seg",
    )
    if not passes_threshold:
        return {**dict(patient_state), "qc": "fail"}
    return dict(patient_state)


def ct_tumor_seg(
    states: list[dict[str, Any]], *, output_root: str, config_dir: str
) -> list[dict[str, Any]]:
    contexts = {
        state["case_id"]: ct_tumor_seg_context(
            state, output_root=output_root, config_dir=config_dir
        )
        for state in states
        if state.get("qc") == "success"
    }
    requests = []
    for context in contexts.values():
        case_id = str(context["case_id"])
        if context["reuse_existing_mask"]:
            announce_tool_action(
                "ct_tumor_seg", case_id, "Reuse existing CT tumor mask."
            )
            continue
        announce_tool_action("ct_tumor_seg", case_id, "Queue tool `ct_tumor_seg`.")
        requests.append(
            {
                "case_id": case_id,
                "ct_path": context["ct_path"],
                "ct_identity": context["ct_identity"],
            }
        )

    from tools.ct_tumor_seg import run_ct_tumor_seg_cohort

    new_results = run_ct_tumor_seg_cohort(requests, output_root, config_dir)
    updated = []
    for state in states:
        if state.get("qc") != "success":
            updated.append(dict(state))
            continue
        context = contexts[state["case_id"]]
        result = (
            context["reused_tool_result"]
            if context["reuse_existing_mask"]
            else new_results[state["case_id"]]
        )
        updated.append(
            dict(finalize_ct_tumor_seg(state, result, context, output_root))
        )
    return updated


def failed_qc_state(
    case: Mapping[str, Any], summary: Mapping[str, Any], tool_name: str
) -> dict[str, Any]:
    state = {**build_patient_state(case, overall="fail"), "qc": "fail"}
    if summary.get("cacheable") is False:
        return add_execution_errors(
            state, tool_name, list(summary.get("errors", []) or [])
        )
    return state


def ct_qc(
    patient_states: list[Mapping[str, Any]], *, output_root: str, config_dir: str
) -> list[dict[str, Any]]:
    from tools.ct_qc import run_ct_qc

    eligible_states = [item for item in patient_states if item.get("qc") == "success"]
    if not eligible_states:
        return [dict(item) for item in patient_states]
    cases = [case_from_state(item) for item in eligible_states]
    ct_result = run_ct_qc(cases, output_root=output_root, config_dir=config_dir)
    summaries = dict(ct_result.get("selection_summaries", {}) or {})
    updated_states = []
    for patient_state in patient_states:
        case = case_from_state(patient_state)
        case_id = str(case.get("Case_ID", "") or "unknown_case")
        if patient_state.get("qc") != "success":
            updated_states.append(dict(patient_state))
            continue
        if not case.get("CT"):
            updated_states.append({**build_patient_state(case, overall="fail"), "qc": "fail"})
            continue
        summary = dict(summaries.get(case_id, {}) or {})
        selected_ct = selected_ct_from_result(summary, case_id)
        if not selected_ct.get("passes_threshold", False):
            updated_states.append(failed_qc_state(case, summary, "ct_qc"))
            continue
        updated_states.append(build_patient_state(case, overall="success"))
    return ct_tumor_seg(
        updated_states, output_root=output_root, config_dir=config_dir
    )


def wsi_qc(
    patient_states: list[Mapping[str, Any]], *, output_root: str, config_dir: str
) -> list[dict[str, Any]]:
    from tools.wsi_qc import run_wsi_qc_cohort as wsi_qc_cohort_tool

    cases = [
        case_from_state(item)
        for item in patient_states
        if dict(item).get("qc") == "success"
    ]
    if not cases:
        return [dict(item) for item in patient_states]
    wsi_result = wsi_qc_cohort_tool(cases, output_root=output_root, config_dir=config_dir)
    summaries = dict(wsi_result.get("selection_summaries", {}) or {})
    updated_states = []
    for patient_state in patient_states:
        case = case_from_state(patient_state)
        case_id = str(case.get("Case_ID", "") or "unknown_case")
        if patient_state.get("qc") != "success":
            updated_states.append(dict(patient_state))
            continue
        if not case.get("WSI"):
            updated_states.append(
                add_execution_errors(
                    {**dict(patient_state), "qc": "fail"},
                    "wsi_qc",
                    ["No WSI record is available."],
                )
            )
            continue
        summary = dict(summaries.get(case_id, {}) or {})
        selected_wsi = dict(summary.get("selected_slide") or {})
        if not selected_wsi.get("passes_threshold", False):
            updated_states.append(
                add_execution_errors(
                    {**dict(patient_state), "qc": "fail"},
                    "wsi_qc",
                    list(summary.get("errors", []) or []) or ["WSI QC failed."],
                )
            )
            continue
        updated_states.append({**dict(patient_state), "qc": "success"})
    patched_states = []
    for state in updated_states:
        if state.get("qc") != "success":
            patched_states.append(dict(state))
            continue
        try:
            patched_states.append(
                dict(wsi_patch(state, output_root=output_root, config_dir=config_dir))
            )
        except Exception as exc:
            patched_states.append(
                add_execution_errors(
                    {**dict(state), "qc": "fail"},
                    "wsi_patch",
                    [f"{type(exc).__name__}: {exc}"],
                )
            )
    return wsi_tumor_seg(
        patched_states, output_root=output_root, config_dir=config_dir
    )
