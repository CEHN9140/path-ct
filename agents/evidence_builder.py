from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from agents.common import (
    add_execution_errors,
    add_omics_result,
    add_tool_result,
    announce_tool_action,
    case_from_state,
    convert_slide_embedding_to_npy,
    load_selected_ct_record,
    load_selected_wsi_record,
    load_tool_snapshot,
)
from utils.llm_utils import load_yaml_file
from utils.omics_utils import build_cohort_signature, collect_case_file_paths
from utils.tool_utils import (
    make_tool_result,
    safe_identifier,
    save_snapshot,
    semantic_execution_config,
)


WSI_EMBEDDING_RUNTIME_KEYS = {
    "batch_size",
    "max_tiles_without_flash_attention",
    "num_workers",
    "pin_memory",
    "prefetch_factor",
    "workers_per_gpu",
    "devices",
}
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
    patch_semantic_config = {
        key: value
        for key, value in patch_config.items()
        if key not in WSI_PATCH_RUNTIME_KEYS
    }
    patch_config_signature = hashlib.sha256(
        json.dumps(patch_semantic_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
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
        and str(cached_payload.get("slide_path", "") or "") == selected_slide_path
        and int(cached_payload.get("patch_size", 0) or 0) == patch_size
        and float(cached_payload.get("target_mpp", 0.0) or 0.0) == target_mpp
        and cached_payload.get("config_signature") == patch_config_signature
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
        "config_signature": patch_config_signature,
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
    semantic_config = {
        key: value
        for key, value in config.items()
        if key not in WSI_TUMOR_SEG_RUNTIME_KEYS
    }
    artifacts = dict(
        dict(patch_bundle.get("tool_result", {}) or {}).get("artifacts", {}) or {}
    )
    identity = {
        "config": semantic_config,
        "slide_path": slide_path,
        "coordinates": file_identity(str(artifacts["coordinates_h5_path"])),
        "model": file_identity(str(config["model_path"])),
        "inference_config": file_identity(str(config["inference_config_path"])),
        "code": file_identity(
            str(Path(__file__).resolve().parent.parent / "tools" / "wsi_tumor_seg.py")
        ),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()


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


def build_wsi_tumor_seg_cohort(
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
    if not all(
        Path(str(state.get("wsi_evidence", {}).get("tile_embeddings_path", ""))).is_file()
        for state in updated_states
        if state.get("qc") == "success"
    ):
        return updated_states

    return updated_states


def wsi_embedding_cache_signature(
    *,
    embedding_config: Mapping[str, Any],
    selected_slide_path: str,
    patch_bundle: Mapping[str, Any],
) -> str:
    code_root = Path(str(embedding_config["prov_gigapath_code_root"])).expanduser()
    checkpoint_dir = code_root / "checkpoints"
    signature_config = {
        key: value
        for key, value in embedding_config.items()
        if key not in WSI_EMBEDDING_RUNTIME_KEYS
    }
    signature_config["device"] = "cuda"
    patch_artifacts = dict(
        dict(patch_bundle.get("tool_result", {}) or {}).get("artifacts", {}) or {}
    )
    identity = {
        "embedding_config": signature_config,
        "selected_slide_path": selected_slide_path,
        "patch_payload": dict(patch_bundle.get("payload", {}) or {}),
        "tumor_selection": file_identity(
            str(patch_artifacts["tumor_coordinates_h5_path"])
        ),
        "model_files": {
            name: file_identity(str(checkpoint_dir / name))
            for name in ("config.json", "pytorch_model.bin", "slide_encoder.pth")
        },
        "embedding_code": file_identity(
            str(Path(__file__).resolve().parent.parent / "tools" / "wsi_embeddings.py")
        ),
        "slide_encoder_code": file_identity(
            str(code_root / "gigapath" / "slide_encoder.py")
        ),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()


def wsi_embedding_context(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> dict[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_wsi_record = load_selected_wsi_record(case, output_root, case_id)
    selected_slide_path = str(selected_wsi_record.get("File Path", "") or "")
    embedding_config = load_yaml_file(
        Path(config_dir).expanduser() / "wsi_embeddings.yaml"
    )
    patch_bundle = load_tool_snapshot(
        output_root,
        "wsi_tumor_seg",
        case_id,
        required_artifact_keys=[
            "tumor_coordinates_h5_path",
            "patch_probabilities_path",
        ],
    )
    if patch_bundle is None:
        raise RuntimeError(f"WSI tumor segmentation snapshot is missing for {case_id}.")
    source_patch_bundle = load_tool_snapshot(
        output_root,
        "wsi_patch",
        case_id,
        required_artifact_keys=["patch_dir", "coordinates_h5_path"],
    )
    if source_patch_bundle is None:
        raise RuntimeError(f"WSI patch snapshot is missing for {case_id}.")
    current_cache_signature = wsi_embedding_cache_signature(
        embedding_config=embedding_config,
        selected_slide_path=selected_slide_path,
        patch_bundle=patch_bundle,
    )
    cached_bundle = load_tool_snapshot(
        output_root,
        "wsi_embeddings",
        case_id,
        required_artifact_keys=[
            "patch_dir",
            "tile_embeddings_path",
            "slide_embedding_path",
        ],
    )
    cached_payload = (
        dict(cached_bundle.get("payload", {}) or {}) if cached_bundle else {}
    )
    cached_tool_result = (
        dict(cached_bundle.get("tool_result", {}) or {}) if cached_bundle else {}
    )
    cached_artifacts = dict(cached_tool_result.get("artifacts", {}) or {})
    patch_artifacts = dict(
        dict(patch_bundle.get("tool_result", {}) or {}).get("artifacts", {}) or {}
    )
    source_patch_artifacts = dict(
        dict(source_patch_bundle.get("tool_result", {}) or {}).get("artifacts", {})
        or {}
    )
    patch_dir_path = Path(str(source_patch_artifacts["patch_dir"]))
    tumor_coordinates_h5_path = Path(
        str(patch_artifacts["tumor_coordinates_h5_path"])
    )
    tile_embeddings_path = Path(
        str(cached_artifacts.get("tile_embeddings_path", "") or "")
    )
    slide_embedding_path = Path(
        str(cached_artifacts.get("slide_embedding_path", "") or "")
    )
    slide_embedding_npy_value = str(
        cached_artifacts.get("slide_embedding_npy_path", "") or ""
    )
    if not slide_embedding_npy_value and str(slide_embedding_path) not in {"", "."}:
        slide_embedding_npy_value = str(slide_embedding_path.with_suffix(".npy"))
    slide_embedding_npy_path = Path(slide_embedding_npy_value or "")
    cached_embeddings_are_current = (
        tile_embeddings_path.exists()
        and slide_embedding_path.exists()
        and min(
            tile_embeddings_path.stat().st_mtime, slide_embedding_path.stat().st_mtime
        )
        >= max(
            (path.stat().st_mtime for path in patch_dir_path.glob("*.png")), default=0.0
        )
    )
    cached_inputs_match = (
        str(cached_artifacts.get("patch_dir", "") or "") == str(patch_dir_path)
        and str(cached_artifacts.get("tumor_coordinates_h5_path", "") or "")
        == str(tumor_coordinates_h5_path)
    )
    legacy_v1_cache = int(cached_payload.get("semantic_cache_version", 0) or 0) == 1
    reuse_cached_embeddings = (
        cached_bundle
        and str(cached_payload.get("slide_path", "") or "") == selected_slide_path
        and cached_inputs_match
        and (
            cached_payload.get("cache_signature") == current_cache_signature
            or legacy_v1_cache
        )
        and cached_embeddings_are_current
    )
    if reuse_cached_embeddings:
        convert_slide_embedding_to_npy(slide_embedding_path, slide_embedding_npy_path)
        if slide_embedding_npy_path.exists():
            cached_artifacts["slide_embedding_npy_path"] = str(slide_embedding_npy_path)
            cached_tool_result["artifacts"] = cached_artifacts
    return {
        "case_id": case_id,
        "selected_slide_path": selected_slide_path,
        "current_cache_signature": current_cache_signature,
        "cached_tool_result": cached_tool_result,
        "reuse_cached_embeddings": reuse_cached_embeddings,
        "request": {
            "case_id": case_id,
            "slide_name": Path(selected_slide_path).stem or patch_dir_path.name,
            "slide_path": selected_slide_path,
            "patch_dir": str(patch_dir_path),
            "tumor_coordinates_h5_path": str(tumor_coordinates_h5_path),
            "cache_signature": current_cache_signature,
        },
    }


def add_wsi_embedding_result(
    state: Mapping[str, Any], result: Mapping[str, Any]
) -> Mapping[str, Any]:
    next_state = add_tool_result(
        state,
        bucket_name="wsi_evidence",
        evidence_key="embeddings",
        tool_result=result,
        node="wsi_embeddings",
    )
    if str(result.get("status", "") or "") == "failure":
        failed_state = add_execution_errors(
            next_state,
            "wsi_embeddings",
            list(result.get("errors", []) or []),
        )
        return {**failed_state, "qc": "fail"}
    return dict(next_state)


def wsi_embeddings(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> Mapping[str, Any]:
    context = wsi_embedding_context(
        state, output_root=output_root, config_dir=config_dir
    )
    case_id = str(context["case_id"])
    if context["reuse_cached_embeddings"]:
        announce_tool_action(
            "wsi_embeddings", case_id, "Reuse cached tool `wsi_embeddings`."
        )
        return add_wsi_embedding_result(state, context["cached_tool_result"])

    announce_tool_action("wsi_embeddings", case_id, "Call tool `wsi_embeddings`.")
    from tools.wsi_embeddings import run_wsi_embeddings

    embedding_result = run_wsi_embeddings(
        case_id=case_id,
        wsi_record={"File Path": context["selected_slide_path"]},
        output_root=output_root,
        patch_dir=str(context["request"]["patch_dir"]),
        tumor_coordinates_h5_path=str(
            context["request"]["tumor_coordinates_h5_path"]
        ),
        config_dir=config_dir,
    )
    errors = [
        str(item).strip()
        for item in list(embedding_result.get("errors", []) or [])
        if str(item).strip()
    ]
    if str(embedding_result.get("status", "") or "") == "failure":
        raise RuntimeError(errors[0] if errors else "WSI embeddings failed.")
    snapshot_path = (
        Path(output_root) / "wsi_embeddings" / f"{safe_identifier(case_id)}.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    embedding_payload = dict(snapshot.get("payload", {}) or {})
    embedding_payload.update(
        {
            "slide_path": context["selected_slide_path"],
            "cache_signature": context["current_cache_signature"],
            "semantic_cache_version": 1,
        }
    )
    save_snapshot(
        output_root,
        "wsi_embeddings",
        case_id,
        {"tool_result": embedding_result, "payload": embedding_payload},
    )
    return add_wsi_embedding_result(state, embedding_result)


def build_wsi_embeddings_cohort(
    states: list[dict[str, Any]], *, output_root: str, config_dir: str
) -> list[dict[str, Any]]:
    contexts = {
        state["case_id"]: wsi_embedding_context(
            state, output_root=output_root, config_dir=config_dir
        )
        for state in states
        if state.get("qc") == "success"
    }
    requests = []
    for context in contexts.values():
        case_id = str(context["case_id"])
        if context["reuse_cached_embeddings"]:
            announce_tool_action(
                "wsi_embeddings", case_id, "Reuse cached tool `wsi_embeddings`."
            )
            continue
        announce_tool_action("wsi_embeddings", case_id, "Queue tool `wsi_embeddings`.")
        requests.append(context["request"])

    from tools.wsi_embeddings import run_wsi_embeddings_cohort

    new_results = run_wsi_embeddings_cohort(requests, output_root, config_dir)
    updated = []
    for state in states:
        if state.get("qc") != "success":
            updated.append(dict(state))
            continue
        context = contexts[state["case_id"]]
        result = (
            context["cached_tool_result"]
            if context["reuse_cached_embeddings"]
            else new_results[state["case_id"]]
        )
        updated.append(dict(add_wsi_embedding_result(state, result)))
    return updated


def ct_input_identity(selected_ct_record: Mapping[str, Any]) -> dict[str, str]:
    series_uid = str(selected_ct_record.get("Series UID", "") or "")
    study_uid = str(selected_ct_record.get("Study UID", "") or "")
    if series_uid:
        return {"series_uid": series_uid, "study_uid": study_uid}
    source_path = str(selected_ct_record.get("File Path", "") or "")
    return {"source_path": str(Path(source_path).expanduser().resolve())}


def file_identity(path: str) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def nifti_geometry_matches(ct_path: str, mask_path: str) -> bool:
    import numpy as np
    import SimpleITK as sitk

    if not Path(ct_path).is_file() or not Path(mask_path).is_file():
        return False
    try:
        ct_image = sitk.ReadImage(ct_path)
        mask_image = sitk.ReadImage(mask_path)
    except Exception:
        return False
    return (
        ct_image.GetSize() == mask_image.GetSize()
        and np.allclose(ct_image.GetSpacing(), mask_image.GetSpacing(), atol=1e-4)
        and np.allclose(ct_image.GetOrigin(), mask_image.GetOrigin(), atol=1e-4)
        and np.allclose(ct_image.GetDirection(), mask_image.GetDirection(), atol=1e-4)
    )


def ct_tumor_seg_config_signature(config: Mapping[str, Any]) -> str:
    semantic_config = {
        key: value
        for key, value in config.items()
        if key not in CT_TUMOR_SEG_RUNTIME_KEYS
    }
    return hashlib.sha256(
        json.dumps(
            semantic_execution_config(semantic_config), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


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
    current_config_signature = ct_tumor_seg_config_signature(tool_config)
    expected_backend = str(tool_config["backend"]).strip().lower()
    expected_model = (
        str(Path(str(tool_config["model_folder"])).expanduser().resolve())
        if "model_folder" in tool_config
        else str(tool_config["task_name"])
    )
    expected_label = tool_config["output_label"]
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
    cached_config_signature = str(cached_payload.get("config_signature", "") or "")
    cached_config_matches = (
        str(cached_provenance.get("backend", "")).lower() == expected_backend
        and str(
            cached_provenance.get("model_folder")
            or cached_provenance.get("task_name")
            or ""
        )
        == expected_model
        and str(cached_provenance.get("output_label", ""))
        == str(expected_label if expected_label is not None else "")
        and bool(cached_config_signature)
        and cached_config_signature == current_config_signature
    )
    cached_ct_identity = dict(cached_payload.get("ct_identity", {}) or {})
    cached_input_matches = (
        bool(cached_ct_identity)
        and cached_ct_identity == current_ct_identity
    )
    reuse_existing_mask = (
        cached_mask_path.exists()
        and cached_config_matches
        and cached_input_matches
        and nifti_geometry_matches(ct_path, str(cached_mask_path))
    )
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
            or {"backend": expected_backend, "case_id": case_id},
            payload={
                "case_id": case_id,
                "segmentation_path": str(cached_mask_path),
                "reused_existing_mask": True,
            },
        )
    return {
        "case_id": case_id,
        "ct_path": ct_path,
        "ct_identity": current_ct_identity,
        "config_signature": current_config_signature,
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
        "ct_identity": current_ct_identity,
        "config_signature": str(context["config_signature"]),
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
                    "config_signature": str(context["config_signature"]),
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
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> Mapping[str, Any]:
    context = ct_tumor_seg_context(
        state, output_root=output_root, config_dir=config_dir
    )
    case_id = str(context["case_id"])
    if context["reuse_existing_mask"]:
        announce_tool_action("ct_tumor_seg", case_id, "Reuse existing CT tumor mask.")
        result = context["reused_tool_result"]
    else:
        announce_tool_action("ct_tumor_seg", case_id, "Call tool `ct_tumor_seg`.")
        from tools.ct_tumor_seg import run_ct_tumor_seg

        result = run_ct_tumor_seg(
            case_id=case_id,
            ct_path=str(context["ct_path"]),
            output_root=output_root,
            config_dir=config_dir,
        )
    return finalize_ct_tumor_seg(state, result, context, output_root)


def build_ct_tumor_seg_cohort(
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


def ct_radiomics(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> Mapping[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_ct_record = load_selected_ct_record(case, output_root, case_id)
    ct_path = str(selected_ct_record.get("File Path", "") or "")
    current_ct_identity = ct_input_identity(selected_ct_record)
    mask_path = str(
        Path(output_root)
        / "ct_tumor_seg"
        / safe_identifier(case_id)
        / f"{case_id}_mask.nii.gz"
    )
    current_mask_identity = (
        file_identity(mask_path) if Path(mask_path).is_file() else {}
    )
    current_radiomics_config = load_yaml_file(
        Path(config_dir).expanduser() / "ct_radiomics.yaml"
    )
    extraction_cache_config = dict(current_radiomics_config)
    extraction_cache_config.pop("confound_correction", None)
    comparison_artifact_keys = []
    for value in current_radiomics_config["ccc_comparison_bin_widths"]:
        bin_width = float(value)
        width_label = (
            str(int(bin_width))
            if bin_width.is_integer()
            else str(bin_width).replace(".", "_")
        )
        comparison_artifact_keys.append(
            f"features_bin_width_{width_label}_json_path"
        )
    cached_bundle = load_tool_snapshot(
        output_root,
        "ct_radiomics",
        case_id,
        required_artifact_keys=[
            "features_json_path",
            *comparison_artifact_keys,
        ],
    )
    cached_payload = (
        dict(cached_bundle.get("payload", {}) or {}) if cached_bundle else {}
    )
    cached_radiomics_config = dict(cached_payload.get("radiomics_config", {}) or {})
    cached_radiomics_config.pop("confound_correction", None)
    if (
        cached_bundle
        and dict(cached_payload.get("ct_identity", {}) or {}) == current_ct_identity
        and dict(cached_payload.get("mask_identity", {}) or {}) == current_mask_identity
        and cached_radiomics_config == extraction_cache_config
    ):
        announce_tool_action(
            "ct_radiomics", case_id, "Reuse cached tool `ct_radiomics`."
        )
        return add_tool_result(
            state,
            bucket_name="ct_evidence",
            evidence_key="radiomics",
            tool_result=dict(cached_bundle.get("tool_result", {}) or {}),
            node="ct_radiomics",
        )

    announce_tool_action("ct_radiomics", case_id, "Call tool `ct_radiomics`.")
    from tools.ct_radiomics import run_ct_radiomics

    radiomics_result = run_ct_radiomics(
        case_id=case_id,
        ct_path=ct_path,
        mask_path=mask_path,
        ct_identity=current_ct_identity,
        mask_identity=current_mask_identity,
        output_root=output_root,
        config_dir=config_dir,
    )
    errors = [
        str(item).strip()
        for item in list(radiomics_result.get("errors", []) or [])
        if str(item).strip()
    ]
    if str(radiomics_result.get("status", "") or "") == "failure":
        raise RuntimeError(errors[0] if errors else "CT radiomics failed.")
    return add_tool_result(
        state,
        bucket_name="ct_evidence",
        evidence_key="radiomics",
        tool_result=radiomics_result,
        node="ct_radiomics",
    )


def evidence_builder(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> Mapping[str, Any]:
    patient_state = dict(state)
    for build_step in (ct_radiomics, wsi_patch):
        if patient_state.get("qc") != "success":
            break
        patient_state = build_step(
            patient_state, output_root=output_root, config_dir=config_dir
        )
    return dict(patient_state)


def save_modality_affinity_artifacts(
    *,
    output_root: str,
    patient_ids: list[str],
    modality_affinities: Mapping[str, Any],
    genomic_discovery: Mapping[str, str],
    audit: Mapping[str, Any],
) -> dict[str, str]:
    import numpy as np

    affinity_dir = Path(output_root) / "candidate_subtype"
    affinity_dir.mkdir(parents=True, exist_ok=True)
    (affinity_dir / "genomic_affinity.npy").unlink(missing_ok=True)
    paths = {}
    for modality in ("ct", "wsi", "rna"):
        path = affinity_dir / f"{modality}_affinity.npy"
        np.save(path, modality_affinities[modality])
        paths[modality] = str(path)
    genomic_path = Path(genomic_discovery["genomic_affinity_path"])
    order_path = Path(genomic_discovery["wxs_patient_order_path"])
    if not genomic_path.is_file() or json.loads(order_path.read_text(encoding="utf-8")) != patient_ids:
        raise ValueError("WXS genomic affinity artifacts do not match the evidence cohort.")
    paths["genomic"] = str(genomic_path)
    (affinity_dir / "feature_engineering_audit.json").write_text(
        json.dumps(dict(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def build_evidence_states(
    patient_states: list[dict[str, Any]], *, output_root: str, config_dir: str = ""
) -> list[dict[str, Any]]:
    from tools.rna import (
        build_rna_cohort_cache,
        rna_signature_extra,
        run_case_rna_features,
    )
    from tools.wxs import (
        build_cnv_cohort_cache,
        build_genomic_discovery_artifacts,
        build_wxs_cohort_cache,
        run_case_cnv_features,
    )

    cohort_cases = []
    for patient_state in patient_states:
        if patient_state.get("qc") != "success":
            continue
        inventory = dict(patient_state.get("inventory", {}) or {})
        cohort_cases.append(
            {
                "Case_ID": str(
                    patient_state.get("case_id", "") or inventory.get("Case_ID", "")
                ),
                "RNA_Seq": list(inventory.get("RNA_Seq") or []),
                "WXS": list(inventory.get("WXS") or []),
                "CNV": list(inventory.get("CNV") or []),
                "Clinical": dict(inventory.get("Clinical") or {}),
            }
        )
    if not cohort_cases:
        return [dict(patient_state) for patient_state in patient_states]

    build_rna_cohort_cache(cohort_cases, output_root=output_root, config_dir=config_dir)
    wxs_cache = build_wxs_cohort_cache(
        cohort_cases, output_root=output_root
    )
    cnv_cache = build_cnv_cohort_cache(
        cohort_cases, output_root=output_root, config_dir=config_dir
    )
    genomic_discovery = build_genomic_discovery_artifacts(
        wxs_cache,
        cnv_cache,
        cohort_cases,
        output_root=output_root,
        config_dir=config_dir,
    )
    rna_signature = build_cohort_signature(
        collect_case_file_paths(cohort_cases, "RNA_Seq"),
        extra=rna_signature_extra(config_dir),
    )
    cnv_signature = str(cnv_cache.get("signature", "") or "")
    updated_states = []
    for patient_state in patient_states:
        updated = dict(patient_state)
        if updated.get("qc") != "success":
            updated_states.append(updated)
            continue
        case_id = str(updated.get("case_id", "") or "unknown_case")
        cached_rna = load_tool_snapshot(
            output_root,
            "rna",
            case_id,
            required_artifact_keys=[
                "case_features_path",
                "pathway_features_path",
                "top_genes_path",
                "manifest_path",
            ],
        )
        rna_provenance = (
            dict(
                dict(cached_rna.get("tool_result", {}) or {}).get("provenance", {})
                or {}
            )
            if cached_rna is not None
            else {}
        )
        if str(rna_provenance.get("signature", "") or "") == rna_signature:
            announce_tool_action(
                "evidence_builder", case_id, "Reuse cached tool `rna`."
            )
            rna_bundle = cached_rna
        else:
            announce_tool_action("evidence_builder", case_id, "Call tool `rna`.")
            rna_bundle = run_case_rna_features(
                case_id=case_id,
                cohort_cases=cohort_cases,
                output_root=output_root,
                config_dir=config_dir,
            )
        updated = add_omics_result(
            updated,
            evidence_key="rna_seq",
            result_bundle=rna_bundle,
            node="evidence_builder",
        )

        updated.setdefault("omics_evidence", {})
        updated["omics_evidence"].update(genomic_discovery)
        cached_cnv = load_tool_snapshot(
            output_root,
            "cnv",
            case_id,
            required_artifact_keys=["case_features_path", "manifest_path"],
        )
        cnv_provenance = (
            dict(
                dict(cached_cnv.get("tool_result", {}) or {}).get("provenance", {})
                or {}
            )
            if cached_cnv is not None
            else {}
        )
        if str(cnv_provenance.get("signature", "") or "") == cnv_signature:
            announce_tool_action(
                "evidence_builder", case_id, "Reuse cached tool `cnv`."
            )
            cnv_bundle = cached_cnv
        else:
            announce_tool_action("evidence_builder", case_id, "Call tool `cnv`.")
            cnv_bundle = run_case_cnv_features(
                case_id=case_id,
                cohort_cases=cohort_cases,
                output_root=output_root,
                config_dir=config_dir,
            )
        updated_states.append(
            add_omics_result(
                updated,
                evidence_key="cnv",
                result_bundle=cnv_bundle,
                node="evidence_builder",
            )
        )
    eligible_states = [state for state in updated_states if state.get("qc") == "success"]
    if not eligible_states:
        return updated_states
    if not all(
        Path(str(state.get("wsi_evidence", {}).get("tile_embeddings_path", ""))).is_file()
        for state in eligible_states
    ):
        return updated_states
    from tools.evidence_features import build_modality_affinity_artifacts

    feature_payload = build_modality_affinity_artifacts(
        eligible_states,
        config_dir=config_dir,
        output_root=output_root,
    )
    patient_ids = [str(state.get("case_id", "")) for state in eligible_states]
    affinity_paths = save_modality_affinity_artifacts(
        output_root=output_root,
        patient_ids=patient_ids,
        modality_affinities=feature_payload["modality_affinities"],
        genomic_discovery=genomic_discovery,
        audit=feature_payload.get("audit", {}),
    )
    affinity_dir = Path(output_root) / "candidate_subtype"
    audit_path = affinity_dir / "feature_engineering_audit.json"
    patient_order_path = Path(genomic_discovery["wxs_patient_order_path"])
    for state in updated_states:
        if state.get("qc") == "success":
            state.setdefault("omics_evidence", {}).update({
                "modality_affinity_paths": affinity_paths,
                "multimodal_audit_path": str(audit_path),
                "modality_affinity_patient_order_path": str(patient_order_path),
            })
    return updated_states
