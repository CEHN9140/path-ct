from __future__ import annotations

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
    ct_input_identity,
    load_selected_ct_record,
    load_selected_wsi_record,
    load_tool_snapshot,
)
from agents.inventory import missing_view_reasons
from utils.cache_utils import file_identity, hash_payload, semantic_config
from utils.llm_utils import load_candidate_proposer_config, load_yaml_file
from utils.omics_utils import build_cohort_signature, collect_case_file_paths
from utils.tool_utils import (
    safe_identifier,
    save_snapshot,
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
CT_RADIOMICS_RUNTIME_KEYS = {"num_workers"}


def wsi_embedding_cache_signature(
    *,
    embedding_config: Mapping[str, Any],
    selected_slide_path: str,
    patch_bundle: Mapping[str, Any],
) -> str:
    code_root = Path(str(embedding_config["prov_gigapath_code_root"])).expanduser()
    checkpoint_dir = code_root / "checkpoints"
    signature_config = semantic_config(embedding_config, WSI_EMBEDDING_RUNTIME_KEYS)
    patch_artifacts = dict(
        dict(patch_bundle.get("tool_result", {}) or {}).get("artifacts", {}) or {}
    )
    return hash_payload(
        {
            "cache_version": 3,
            "semantic_config": signature_config,
            "upstream": {
                "wsi_tumor_seg": str(
                    dict(patch_bundle.get("payload", {}) or {}).get(
                        "cache_signature", ""
                    )
                ),
            },
            "inputs": {
                "tumor_coordinates": file_identity(
                    str(patch_artifacts["tumor_coordinates_h5_path"])
                ),
            },
            "models": {
                name: file_identity(str(checkpoint_dir / name))
                for name in (
                    "config.json",
                    "pytorch_model.bin",
                    "slide_encoder.pth",
                )
            },
            "code": {
                "embedding": file_identity(
                    str(
                        Path(__file__).resolve().parent.parent
                        / "tools"
                        / "wsi_embeddings.py"
                    )
                ),
                "slide_encoder": file_identity(
                    str(code_root / "gigapath" / "slide_encoder.py")
                ),
            },
        }
    )


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
    tumor_coordinates_h5_path = Path(str(patch_artifacts["tumor_coordinates_h5_path"]))
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
    cached_inputs_match = str(cached_artifacts.get("patch_dir", "") or "") == str(
        patch_dir_path
    ) and str(cached_artifacts.get("tumor_coordinates_h5_path", "") or "") == str(
        tumor_coordinates_h5_path
    )
    reuse_cached_embeddings = (
        cached_bundle
        and str(cached_payload.get("slide_path", "") or "") == selected_slide_path
        and cached_inputs_match
        and cached_payload.get("cache_signature") == current_cache_signature
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
        tumor_coordinates_h5_path=str(context["request"]["tumor_coordinates_h5_path"]),
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
            "semantic_cache_version": 3,
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
    tumor_seg_bundle = load_tool_snapshot(
        output_root,
        "ct_tumor_seg",
        case_id,
        required_artifact_keys=["segmentation_path"],
    )
    current_radiomics_cache_signature = hash_payload(
        {
            "cache_version": 1,
            "semantic_config": semantic_config(
                current_radiomics_config, CT_RADIOMICS_RUNTIME_KEYS
            ),
            "upstream": {
                "ct_tumor_seg": str(
                    dict(tumor_seg_bundle.get("payload", {}) or {}).get(
                        "cache_signature", ""
                    )
                    if tumor_seg_bundle
                    else ""
                )
            },
            "inputs": {
                "ct": file_identity(ct_path)
                if Path(ct_path).is_file()
                else {"path": ct_path},
                "mask": current_mask_identity,
            },
        }
    )
    cached_bundle = load_tool_snapshot(
        output_root,
        "ct_radiomics",
        case_id,
        required_artifact_keys=[
            "features_json_path",
        ],
    )
    cached_payload = (
        dict(cached_bundle.get("payload", {}) or {}) if cached_bundle else {}
    )
    if (
        cached_bundle
        and cached_payload.get("cache_signature") == current_radiomics_cache_signature
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
    snapshot_path = (
        Path(output_root) / "ct_radiomics" / f"{safe_identifier(case_id)}.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_payload = dict(snapshot.get("payload", {}) or {})
    snapshot_payload["cache_signature"] = current_radiomics_cache_signature
    snapshot_payload["cache_version"] = 1
    save_snapshot(
        output_root,
        "ct_radiomics",
        case_id,
        {"tool_result": radiomics_result, "payload": snapshot_payload},
    )
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
    for build_step in (ct_radiomics,):
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
    audit: Mapping[str, Any],
) -> dict[str, str]:
    import numpy as np

    candidate_dir = Path(output_root) / "candidate_subtype"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    if set(modality_affinities) != {"ct", "wsi", "rna", "wxs"}:
        raise ValueError(
            "Candidate generation requires exactly CT, WSI, RNA, and WXS affinities"
        )
    paths = {}
    for modality, matrix in modality_affinities.items():
        path = candidate_dir / f"{modality}_affinity.npy"
        np.save(path, np.asarray(matrix, dtype=float))
        paths[modality] = str(path)
    (candidate_dir / "affinity_patient_order.json").write_text(
        json.dumps(patient_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (candidate_dir / "feature_engineering_audit.json").write_text(
        json.dumps(dict(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def filter_four_view_states(
    patient_states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    updated_states = []
    complete_states = []
    audit = {
        "pre_qc_count": len(patient_states),
        "ct_wsi_pass_count": sum(
            state.get("qc") == "success" for state in patient_states
        ),
    }
    excluded = {}
    for state in patient_states:
        updated = dict(state)
        reasons = set(map(str, state.get("missing_view_reason", [])))
        if state.get("qc") != "success":
            reasons.add("ct_wsi_qc_failed")
        else:
            reasons.update(missing_view_reasons(dict(state.get("inventory", {}) or {})))
            for bucket, reason in (
                ("ct_evidence", "missing_ct_feature"),
                ("wsi_evidence", "missing_wsi_embedding"),
            ):
                if not Path(
                    str(dict(state.get(bucket, {}) or {}).get("feature_path", ""))
                ).is_file():
                    reasons.add(reason)
        if reasons:
            updated["qc"] = "fail"
            updated["missing_view_reason"] = sorted(reasons)
            case_id = str(updated.get("case_id", ""))
            for reason in reasons:
                excluded.setdefault(reason, []).append(case_id)
        else:
            updated["missing_view_reason"] = []
            complete_states.append(updated)
        updated_states.append(updated)
    audit.update(
        {
            "four_view_complete_count": len(complete_states),
            **{
                f"excluded_{reason}": sorted(case_ids)
                for reason, case_ids in excluded.items()
            },
        }
    )
    return updated_states, complete_states, audit


def build_evidence_states(
    patient_states: list[dict[str, Any]], *, output_root: str, config_dir: str = ""
) -> list[dict[str, Any]]:
    from tools.evidence_features import build_modality_affinity_artifacts
    from tools.rna import build_rna_cohort_cache, rna_signature_extra
    from tools.wxs import build_wxs_artifacts, build_wxs_cohort_cache
    from utils.cache_utils import file_identity, hash_payload
    from utils.llm_utils import load_candidate_proposer_config
    from utils.omics_utils import build_cohort_signature, collect_case_file_paths

    updated_states, eligible_states, audit = filter_four_view_states(patient_states)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "four_view_cohort_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not eligible_states:
        raise ValueError("No patients passed four-view QC and feature availability")
    cohort_cases = []
    for state in eligible_states:
        inventory = dict(state["inventory"])
        cohort_cases.append(
            {
                "Case_ID": str(state["case_id"]),
                "RNA_Seq": list(inventory["RNA_Seq"]),
                "WXS": list(inventory["WXS"]),
            }
        )

    rna_cache = build_rna_cohort_cache(
        cohort_cases, output_root=output_root, config_dir=config_dir
    )
    wxs_cache = build_wxs_cohort_cache(cohort_cases, output_root=output_root)
    wxs_artifacts = build_wxs_artifacts(
        wxs_cache, cohort_cases, output_root=output_root, config_dir=config_dir
    )
    rna_signature = build_cohort_signature(
        collect_case_file_paths(cohort_cases, "RNA_Seq"),
        extra=rna_signature_extra(config_dir),
    )
    for state in eligible_states:
        state.setdefault("omics_evidence", {}).update(
            {
                "rna_feature_path": rna_cache["case_features_path"],
                "rna_pathway_feature_path": rna_cache["pathway_features_path"],
                "wxs_discovery_feature_path": wxs_artifacts[
                    "wxs_discovery_feature_path"
                ],
                "wxs_validation_feature_path": wxs_artifacts[
                    "wxs_validation_feature_path"
                ],
                "wxs_discovery_audit_path": wxs_artifacts["wxs_discovery_audit_path"],
                "wxs_patient_order_path": wxs_artifacts["wxs_patient_order_path"],
            }
        )

    patient_ids = [str(state["case_id"]) for state in eligible_states]
    candidate_dir = Path(output_root) / "candidate_subtype"
    manifest_path = candidate_dir / "affinity_cache.json"
    config_path = Path(config_dir or "configs")
    proposer_config = load_candidate_proposer_config(config_path)
    input_files = []
    for state in eligible_states:
        for bucket in ("ct_evidence", "wsi_evidence"):
            input_files.append(file_identity(str(state[bucket]["feature_path"])))
    for _, path in collect_case_file_paths(cohort_cases, "RNA_Seq"):
        input_files.append(file_identity(path))
    input_files.extend(
        file_identity(row["file_path"]) for row in wxs_cache["manifest"]["input_cases"]
    )
    signature = hash_payload(
        {
            "cache_version": 7,
            "patient_ids": patient_ids,
            "input_files": input_files,
            "rna_signature": rna_signature,
            "wxs_signature": wxs_cache["signature"],
            "wxs_affinity": file_identity(wxs_artifacts["wxs_affinity_path"]),
            "snf": proposer_config["snf"],
            "candidate_views": ["ct", "wsi", "rna", "wxs"],
            "code": {
                "evidence_builder.py": file_identity(__file__),
                **{
                    name: file_identity(
                        str(Path(__file__).resolve().parent.parent / "tools" / name)
                    )
                    for name in (
                        "evidence_features.py",
                        "ct_radiomics.py",
                        "rna.py",
                        "wsi_affinity.py",
                        "wxs.py",
                    )
                },
            },
        }
    )
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    paths = dict(manifest.get("paths", {}))
    if (
        manifest.get("cache_signature") != signature
        or manifest.get("patient_ids") != patient_ids
        or set(paths) != {"ct", "wsi", "rna", "wxs"}
        or not all(Path(path).is_file() for path in paths.values())
        or not (candidate_dir / "feature_engineering_audit.json").is_file()
    ):
        features = build_modality_affinity_artifacts(
            eligible_states,
            output_root=output_root,
            config_dir=config_dir,
            discovery_artifacts=wxs_artifacts,
        )
        paths = save_modality_affinity_artifacts(
            output_root=output_root,
            patient_ids=patient_ids,
            modality_affinities=features["modality_affinities"],
            audit=features["audit"],
        )
        manifest = {
            "cache_version": 7,
            "cache_signature": signature,
            "patient_ids": patient_ids,
            "paths": paths,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for state in updated_states:
        if state.get("qc") == "success":
            state.setdefault("omics_evidence", {}).update(
                {
                    "modality_affinity_paths": paths,
                    "modality_affinity_cache_signature": signature,
                    "modality_affinity_patient_order_path": str(
                        candidate_dir / "affinity_patient_order.json"
                    ),
                    "multimodal_audit_path": str(
                        candidate_dir / "feature_engineering_audit.json"
                    ),
                }
            )
    return updated_states
