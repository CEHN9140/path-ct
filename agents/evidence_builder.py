from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agents.common import (
    add_omics_result,
    add_tool_result,
    announce_tool_action,
    convert_slide_embedding_to_npy,
    case_from_state,
    load_selected_ct_record,
    load_selected_wsi_record,
    load_tool_snapshot,
)
from utils.llm_utils import load_yaml_file
from utils.omics_utils import build_cohort_signature, collect_case_file_paths
from utils.tool_utils import make_tool_result, safe_identifier, save_snapshot


def wsi_patch(state: Mapping[str, Any], *, output_root: str, config_dir: str) -> Mapping[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_wsi_record = load_selected_wsi_record(case, output_root, case_id)
    selected_slide_path = str(selected_wsi_record.get("File Path", "") or "")
    cached_bundle = load_tool_snapshot(
        output_root,
        "wsi_patch",
        case_id,
        required_artifact_keys=["patch_dir", "coordinates_h5_path"],
    )
    cached_payload = dict(cached_bundle.get("payload", {}) or {}) if cached_bundle is not None else {}
    if cached_bundle is not None and str(cached_payload.get("slide_path", "") or "") == selected_slide_path:
        announce_tool_action("wsi_patch", case_id, "Reuse cached tool `wsi_patch`.")
        return add_tool_result(state, bucket_name="wsi_evidence", evidence_key="patch_extraction", tool_result=dict(cached_bundle.get("tool_result", {}) or {}), node="wsi_patch")

    announce_tool_action("wsi_patch", case_id, "Call tool `wsi_patch`.")
    from tools.wsi_patch import run_seg_wsi_patch

    patch_result = run_seg_wsi_patch(case_id=case_id, wsi_records=list(case.get("WSI", []) or []), output_root=output_root, config_dir=config_dir)
    if patch_result.get("errors"):
        raise RuntimeError(str(list(patch_result.get("errors") or [])[0]))
    patch_slide = dict(list(patch_result.get("slides", []) or [{}])[0] or {})
    patch_info = {
        "patch_dir": str(patch_slide.get("patch_dir", "") or ""),
        "patch_count": int(patch_slide.get("patch_count", 0) or 0),
        "patch_size": int(patch_result.get("patch_size", 0) or 0),
        "level": int(patch_result.get("level", 0) or 0),
        "coordinate_level": int(patch_result.get("coordinate_level", 0) or 0),
        "level_downsample": float(patch_slide.get("level_downsample", 1.0) or 1.0),
        "coordinates_h5_path": str(patch_slide.get("coordinates_h5_path", "") or ""),
        "qc_mask_path": str(patch_slide.get("qc_mask_path", "") or ""),
        "slide_name": str(patch_slide.get("slide_name", "") or ""),
        "slide_path": str(patch_slide.get("slide_path", "") or selected_slide_path),
        "qc_clean_tissue_value": int(patch_slide.get("qc_clean_tissue_value", patch_result.get("qc_clean_tissue_value", 0)) or 0),
    }
    patch_tool_result = make_tool_result(
        output_root=output_root,
        tool_name="wsi_patch",
        status="success",
        identifier=case_id,
        metrics={"patch_count": patch_info["patch_count"], "patch_size": patch_info["patch_size"], "level": patch_info["level"]},
        artifacts={"patch_dir": patch_info["patch_dir"], "coordinates_h5_path": patch_info["coordinates_h5_path"], "qc_mask_path": patch_info["qc_mask_path"]},
        provenance={"backend": "openslide", "case_id": case_id, "slide_name": patch_info["slide_name"]},
        errors=[],
        payload=patch_info,
    )
    return add_tool_result(state, bucket_name="wsi_evidence", evidence_key="patch_extraction", tool_result=patch_tool_result, node="wsi_patch")


def wsi_embeddings(state: Mapping[str, Any], *, output_root: str, config_dir: str) -> Mapping[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_wsi_record = load_selected_wsi_record(case, output_root, case_id)
    selected_slide_path = str(selected_wsi_record.get("File Path", "") or "")
    cached_bundle = load_tool_snapshot(output_root, "wsi_embeddings", case_id, required_artifact_keys=["patch_dir", "tile_embeddings_path", "slide_embedding_path"])
    cached_payload = dict(cached_bundle.get("payload", {}) or {}) if cached_bundle else {}
    cached_tool_result = dict(cached_bundle.get("tool_result", {}) or {}) if cached_bundle else {}
    cached_artifacts = dict(cached_tool_result.get("artifacts", {}) or {})
    patch_dir_path = Path(str(cached_artifacts.get("patch_dir", "") or ""))
    tile_embeddings_path = Path(str(cached_artifacts.get("tile_embeddings_path", "") or ""))
    slide_embedding_path = Path(str(cached_artifacts.get("slide_embedding_path", "") or ""))
    slide_embedding_npy_value = str(cached_artifacts.get("slide_embedding_npy_path", "") or "")
    if not slide_embedding_npy_value and str(slide_embedding_path) not in {"", "."}:
        slide_embedding_npy_value = str(slide_embedding_path.with_suffix(".npy"))
    slide_embedding_npy_path = Path(slide_embedding_npy_value or "")
    cached_embeddings_are_current = (
        tile_embeddings_path.exists()
        and slide_embedding_path.exists()
        and min(tile_embeddings_path.stat().st_mtime, slide_embedding_path.stat().st_mtime)
        >= max((path.stat().st_mtime for path in patch_dir_path.glob("*.png")), default=0.0)
    )
    if cached_bundle and str(cached_payload.get("slide_path", "") or "") == selected_slide_path and cached_embeddings_are_current:
        convert_slide_embedding_to_npy(slide_embedding_path, slide_embedding_npy_path, config_dir)
        if slide_embedding_npy_path.exists():
            cached_artifacts["slide_embedding_npy_path"] = str(slide_embedding_npy_path)
            cached_tool_result["artifacts"] = cached_artifacts
        announce_tool_action("wsi_embeddings", case_id, "Reuse cached tool `wsi_embeddings`.")
        return add_tool_result(state, bucket_name="wsi_evidence", evidence_key="embeddings", tool_result=cached_tool_result, node="wsi_embeddings")

    announce_tool_action("wsi_embeddings", case_id, "Call tool `wsi_embeddings`.")
    from tools.wsi_embeddings import run_wsi_embeddings

    embedding_result = run_wsi_embeddings(case_id=case_id, wsi_record=selected_wsi_record, output_root=output_root, config_dir=config_dir)
    errors = [str(item).strip() for item in list(embedding_result.get("errors", []) or []) if str(item).strip()]
    if str(embedding_result.get("status", "") or "") == "failure":
        raise RuntimeError(errors[0] if errors else "WSI embeddings failed.")
    return add_tool_result(state, bucket_name="wsi_evidence", evidence_key="embeddings", tool_result=embedding_result, node="wsi_embeddings")


def ct_tumor_seg(state: Mapping[str, Any], *, output_root: str, config_dir: str) -> Mapping[str, Any]:
    import nibabel as nib
    import numpy as np

    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_ct_record = load_selected_ct_record(case, output_root, case_id)
    ct_path = str(selected_ct_record.get("File Path", "") or "")
    cached_mask_path = Path(output_root) / "ct_tumor_seg" / safe_identifier(case_id) / f"{case_id}_mask.nii.gz"
    tool_config = load_yaml_file(Path(config_dir).expanduser() / "ct_tumor_seg.yaml")
    expected_backend = str(tool_config["backend"]).strip().lower()
    expected_model = (
        str(Path(str(tool_config["model_folder"])).expanduser().resolve())
        if "model_folder" in tool_config
        else str(tool_config["task_name"])
    )
    expected_label = tool_config.get("output_label")
    snapshot_path = Path(output_root) / "ct_tumor_seg" / f"{safe_identifier(case_id)}.json"
    cached_tool_result = {}
    if snapshot_path.exists():
        try:
            cached_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            cached_tool_result = dict(cached_snapshot.get("tool_result", {}) or {})
        except Exception:
            cached_tool_result = {}
    cached_provenance = dict(cached_tool_result.get("provenance", {}) or {})
    cached_config_matches = (
        str(cached_provenance.get("backend", "")).lower() == expected_backend
        and str(cached_provenance.get("model_folder") or cached_provenance.get("task_name") or "") == expected_model
        and str(cached_provenance.get("output_label", "")) == str(expected_label if expected_label is not None else "")
    )
    reuse_existing_mask = cached_mask_path.exists() and cached_config_matches
    reuse_cached_failure = (
        str(cached_tool_result.get("status", "") or "").lower() == "failure"
        and cached_config_matches
    )
    cache_message = (
        "Reuse cached CT tumor segmentation failure."
        if reuse_cached_failure
        else "Reuse existing CT tumor mask."
        if reuse_existing_mask
        else "Call tool `ct_tumor_seg`."
    )
    announce_tool_action("ct_tumor_seg", case_id, cache_message)
    if reuse_cached_failure:
        patient_state = add_tool_result(state, bucket_name="ct_evidence", evidence_key="tumor_segmentation", tool_result=cached_tool_result, node="ct_tumor_seg")
        return {**dict(patient_state), "qc": "fail"}
    if reuse_existing_mask:
        tumor_seg_result = make_tool_result(
            output_root=output_root,
            tool_name="ct_tumor_seg",
            status="success",
            identifier=case_id,
            metrics={"returncode": 0},
            artifacts={"segmentation_path": str(cached_mask_path)},
            provenance=cached_provenance or {"backend": expected_backend, "case_id": case_id},
            payload={"case_id": case_id, "segmentation_path": str(cached_mask_path), "reused_existing_mask": True},
        )
        tumor_seg_payload = {"reused_existing_mask": True}
    else:
        from tools.ct_tumor_seg import run_ct_tumor_seg

        tumor_seg_result = run_ct_tumor_seg(case_id=case_id, ct_path=ct_path, output_root=output_root, config_dir=config_dir)
        errors = [str(item).strip() for item in list(tumor_seg_result.get("errors", []) or []) if str(item).strip()]
        if str(tumor_seg_result.get("status", "") or "") == "failure":
            artifacts = dict(tumor_seg_result.get("artifacts", {}) or {})
            artifacts["saved_output_path"] = str(Path(output_root) / "ct_tumor_seg" / f"{safe_identifier(case_id)}.json")
            tumor_seg_result["artifacts"] = artifacts
            save_snapshot(output_root, "ct_tumor_seg", case_id, {"tool_result": tumor_seg_result, "payload": {"case_id": case_id, "input_ct_path": ct_path}})
            patient_state = add_tool_result(state, bucket_name="ct_evidence", evidence_key="tumor_segmentation", tool_result=tumor_seg_result, node="ct_tumor_seg")
            return {**dict(patient_state), "qc": "fail"}
        tumor_seg_payload = {}

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
    passes_threshold = segmentation_exists and positive_voxel_count > 0
    mask_qc_errors = [
        message
        for condition, message in (
            (not segmentation_exists, "Tumor segmentation file is missing."),
            (segmentation_exists and not passes_threshold, "Tumor mask QC failed: empty tumor mask."),
        )
        if condition
    ]
    metrics.update({"positive_voxel_count": positive_voxel_count, "positive_volume_mm3": positive_volume_mm3})
    if mask_qc_errors:
        tumor_seg_result["status"] = "failure"
        existing_errors = [str(item).strip() for item in list(tumor_seg_result.get("errors", []) or []) if str(item).strip()]
        tumor_seg_result["errors"] = existing_errors + [item for item in mask_qc_errors if item not in existing_errors]
    tumor_seg_result["metrics"] = metrics
    artifacts["saved_output_path"] = str(Path(output_root) / "ct_tumor_seg" / f"{safe_identifier(case_id)}.json")
    tumor_seg_result["artifacts"] = artifacts
    tumor_seg_payload.update(
        {
            "case_id": case_id,
            "segmentation_path": segmentation_path,
            "tumor_mask_qc": {
                "segmentation_exists": segmentation_exists,
                "has_positive_voxels": positive_voxel_count > 0,
                "positive_voxel_count": positive_voxel_count,
                "positive_volume_mm3": positive_volume_mm3,
                "passes_threshold": passes_threshold,
            },
            "voxel_spacing": voxel_spacing,
        }
    )
    save_snapshot(output_root, "ct_tumor_seg", case_id, {"tool_result": tumor_seg_result, "payload": tumor_seg_payload})
    patient_state = add_tool_result(state, bucket_name="ct_evidence", evidence_key="tumor_segmentation", tool_result=tumor_seg_result, node="ct_tumor_seg")
    if not passes_threshold:
        return {**dict(patient_state), "qc": "fail"}
    return dict(patient_state)


def ct_radiomics(state: Mapping[str, Any], *, output_root: str, config_dir: str) -> Mapping[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    selected_ct_record = load_selected_ct_record(case, output_root, case_id)
    ct_path = str(selected_ct_record.get("File Path", "") or "")
    mask_path = str(Path(output_root) / "ct_tumor_seg" / safe_identifier(case_id) / f"{case_id}_mask.nii.gz")
    cached_bundle = load_tool_snapshot(output_root, "ct_radiomics", case_id, required_artifact_keys=["features_json_path", "metrics_path"])
    cached_payload = dict(cached_bundle.get("payload", {}) or {}) if cached_bundle else {}
    if cached_bundle and str(cached_payload.get("ct_path", "") or "") == ct_path and str(cached_payload.get("mask_path", "") or "") == mask_path:
        announce_tool_action("ct_radiomics", case_id, "Reuse cached tool `ct_radiomics`.")
        return add_tool_result(state, bucket_name="ct_evidence", evidence_key="radiomics", tool_result=dict(cached_bundle.get("tool_result", {}) or {}), node="ct_radiomics")

    announce_tool_action("ct_radiomics", case_id, "Call tool `ct_radiomics`.")
    from tools.ct_radiomics import run_ct_radiomics

    radiomics_result = run_ct_radiomics(case_id=case_id, ct_path=ct_path, mask_path=mask_path, output_root=output_root, config_dir=config_dir)
    errors = [str(item).strip() for item in list(radiomics_result.get("errors", []) or []) if str(item).strip()]
    if str(radiomics_result.get("status", "") or "") == "failure":
        raise RuntimeError(errors[0] if errors else "CT radiomics failed.")
    return add_tool_result(state, bucket_name="ct_evidence", evidence_key="radiomics", tool_result=radiomics_result, node="ct_radiomics")


def evidence_builder(state: Mapping[str, Any], *, output_root: str, config_dir: str) -> Mapping[str, Any]:
    patient_state = dict(state)
    for build_step in (ct_tumor_seg, ct_radiomics, wsi_patch, wsi_embeddings):
        if patient_state.get("qc") != "success":
            break
        patient_state = build_step(patient_state, output_root=output_root, config_dir=config_dir)
    return dict(patient_state)


def build_evidence_states(patient_states: list[dict[str, Any]], *, output_root: str, config_dir: str = "") -> list[dict[str, Any]]:
    from tools.rna import build_rna_cohort_cache, rna_top_gene_count, run_case_rna_features
    from tools.wxs import build_wxs_cohort_cache, run_case_wxs_features

    cohort_cases = []
    for patient_state in patient_states:
        if patient_state.get("qc") != "success":
            continue
        inventory = dict(patient_state.get("inventory", {}) or {})
        cohort_cases.append(
            {
                "Case_ID": str(patient_state.get("case_id", "") or inventory.get("Case_ID", "")),
                "RNA_Seq": list(inventory.get("RNA_Seq") or []),
                "WXS": list(inventory.get("WXS") or []),
                "Clinical": dict(inventory.get("Clinical") or {}),
            }
        )
    if not cohort_cases:
        return [dict(patient_state) for patient_state in patient_states]

    build_rna_cohort_cache(cohort_cases, output_root=output_root, config_dir=config_dir)
    wxs_cache = build_wxs_cohort_cache(cohort_cases, output_root=output_root, config_dir=config_dir)
    top_gene_count = rna_top_gene_count(config_dir)
    rna_signature = build_cohort_signature(
        collect_case_file_paths(cohort_cases, "RNA_Seq"),
        extra={"top_gene_count": top_gene_count, "modality": "RNA_Seq", "feature_mode": "top_genes_only"},
    )
    wxs_signature = str(wxs_cache.get("signature", "") or "")
    updated_states = []
    for patient_state in patient_states:
        updated = dict(patient_state)
        if updated.get("qc") != "success":
            updated_states.append(updated)
            continue
        case_id = str(updated.get("case_id", "") or "unknown_case")
        cached_rna = load_tool_snapshot(output_root, "rna", case_id, required_artifact_keys=["case_features_path", "pathway_features_path", "top_genes_path", "manifest_path"])
        rna_provenance = dict(dict(cached_rna.get("tool_result", {}) or {}).get("provenance", {}) or {}) if cached_rna is not None else {}
        if str(rna_provenance.get("signature", "") or "") == rna_signature:
            announce_tool_action("evidence_builder", case_id, "Reuse cached tool `rna`.")
            rna_bundle = cached_rna
        else:
            announce_tool_action("evidence_builder", case_id, "Call tool `rna`.")
            rna_bundle = run_case_rna_features(case_id=case_id, cohort_cases=cohort_cases, output_root=output_root, config_dir=config_dir)
        updated = add_omics_result(updated, evidence_key="rna_seq", result_bundle=rna_bundle, node="evidence_builder")

        cached_wxs = load_tool_snapshot(
            output_root,
            "wxs",
            case_id,
            required_artifact_keys=["case_features_path", "all_features_path", "gene_frequency_path", "case_summary_path", "filtered_mutations_path", "manifest_path"],
        )
        wxs_provenance = dict(dict(cached_wxs.get("tool_result", {}) or {}).get("provenance", {}) or {}) if cached_wxs is not None else {}
        if str(wxs_provenance.get("signature", "") or "") == wxs_signature:
            announce_tool_action("evidence_builder", case_id, "Reuse cached tool `wxs`.")
            wxs_bundle = cached_wxs
        else:
            announce_tool_action("evidence_builder", case_id, "Call tool `wxs`.")
            wxs_bundle = run_case_wxs_features(case_id=case_id, cohort_cases=cohort_cases, output_root=output_root, config_dir=config_dir)
        updated_states.append(add_omics_result(updated, evidence_key="wxs", result_bundle=wxs_bundle, node="evidence_builder"))
    return updated_states
