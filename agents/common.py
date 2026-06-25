from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from utils.tool_utils import safe_identifier, to_jsonable
from utils.llm_utils import load_yaml_file


def case_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    return {
        "Case_ID": str(state.get("case_id", "") or inventory.get("Case_ID", "") or inventory.get("case_id", "")),
        "WSI": list(inventory.get("WSI") or inventory.get("wsi_records", []) or []),
        "CT": list(inventory.get("CT") or inventory.get("ct_records", []) or []),
        "RNA_Seq": list(inventory.get("RNA_Seq") or inventory.get("rna_seq_records", []) or []),
        "WXS": list(inventory.get("WXS") or inventory.get("wxs_records", []) or []),
        "Clinical": dict(inventory.get("Clinical") or inventory.get("clinical", {}) or {}),
    }


def load_selected_wsi_record(case: Mapping[str, Any], output_root: str, case_id: str) -> dict[str, Any]:
    records = list(case.get("WSI", []) or [])
    selected = {}
    summary_path = Path(output_root) / "wsi_qc" / case_id / "selection_summary.json"
    if summary_path.exists():
        try:
            selected = dict(json.loads(summary_path.read_text(encoding="utf-8")).get("selected_slide") or {})
        except Exception:
            selected = {}
    selected_path = str(selected.get("selected_slide_path", "") or "")
    selected_name = str(selected.get("selected_slide_name", "") or "")
    for record in records:
        slide_path = str(dict(record).get("File Path", "") or "")
        slide_name = os.path.basename(slide_path).replace(".svs", "")
        if slide_path == selected_path or slide_name == selected_name:
            return dict(record)
    fallback = dict(records[0]) if records else {}
    if selected_path:
        fallback["File Path"] = selected_path
    return fallback


def selected_ct_from_result(ct_result: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    selected_ct = next(
        (dict(item) for item in list(ct_result.get("selected_files", [])) if str(item.get("case_id", "")) == case_id),
        {},
    )
    selected_ct.update(dict(ct_result.get("selected_series", {}) or {}))
    return selected_ct


def selected_ct_record(case: Mapping[str, Any], selected_ct: Mapping[str, Any]) -> dict[str, Any]:
    records = list(case.get("CT", []) or [])
    selected_record = dict(records[0]) if records else {}
    selected_file = str(selected_ct.get("selected_file", "") or selected_ct.get("ct_path", "") or "")
    selected_source_file = str(selected_ct.get("selected_source_file", "") or "")
    selected_input_path = str(selected_ct.get("selected_input_path", "") or "")
    for ct_record in records:
        record_path = str(dict(ct_record).get("File Path", "") or "")
        if not record_path:
            continue
        if record_path == selected_file or record_path == selected_input_path:
            selected_record = dict(ct_record)
            break
        if selected_source_file and record_path == selected_source_file:
            selected_record = dict(ct_record)
            break
        if selected_source_file:
            try:
                Path(selected_source_file).resolve().relative_to(Path(record_path).resolve())
                selected_record = dict(ct_record)
                break
            except ValueError:
                pass
    if selected_file:
        selected_record["File Path"] = selected_file
    return selected_record


def load_selected_ct_record(case: Mapping[str, Any], output_root: str, case_id: str) -> dict[str, Any]:
    summary_path = Path(output_root) / "ct_qc" / case_id / "selection_summary.json"
    if not summary_path.exists():
        return selected_ct_record(case, {})
    try:
        ct_result = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return selected_ct_record(case, {})
    return selected_ct_record(case, selected_ct_from_result(ct_result, case_id))


def announce_tool_action(node: str, case_id: str, message: str) -> None:
    print(f"[{node}] {case_id}: {message}", flush=True)


def announce_tool_result_path(node: str, case_id: str, tool_result: Mapping[str, Any]) -> None:
    saved_output_path = str(dict(tool_result.get("artifacts", {}) or {}).get("saved_output_path", "") or "").strip()
    if saved_output_path:
        print(f"[{node}] {case_id}: Result saved to {saved_output_path}", flush=True)


def load_tool_snapshot(
    output_root: str,
    tool_name: str,
    case_id: str,
    *,
    required_artifact_keys: list[str] | None = None,
) -> dict[str, Any] | None:
    snapshot_path = Path(output_root) / tool_name / f"{safe_identifier(case_id)}.json"
    if not snapshot_path.exists():
        return None
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    tool_result = dict(snapshot.get("tool_result", {}) or {})
    if str(tool_result.get("tool_name", "") or "") != tool_name:
        return None
    if str(tool_result.get("status", "") or "").lower() == "failure":
        return None

    artifacts = dict(tool_result.get("artifacts", {}) or {})
    for artifact_key in required_artifact_keys or []:
        artifact_path = str(artifacts.get(artifact_key, "") or "")
        if not artifact_path or not Path(artifact_path).exists():
            return None
    return {"tool_result": tool_result, "payload": dict(snapshot.get("payload", {}) or {})}


def convert_slide_embedding_to_npy(slide_embedding_path: Path, slide_embedding_npy_path: Path, config_dir: str) -> None:
    if slide_embedding_npy_path.exists() or not slide_embedding_path.exists():
        return
    try:
        import subprocess

        config = load_yaml_file(Path(config_dir).expanduser() / "wsi_embeddings.yaml")
        python_executable = Path(str(config["python_executable"])).expanduser()
        if not python_executable.exists():
            return
        slide_embedding_npy_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                str(python_executable),
                "-c",
                (
                    "import sys, torch, numpy as np; "
                    "value = torch.load(sys.argv[1], map_location='cpu'); "
                    "value = value.detach().cpu().numpy() if hasattr(value, 'detach') else value; "
                    "np.save(sys.argv[2], value)"
                ),
                str(slide_embedding_path),
                str(slide_embedding_npy_path),
            ],
            check=False,
        )
        if completed.returncode != 0 and slide_embedding_npy_path.exists():
            slide_embedding_npy_path.unlink()
    except Exception:
        return


def read_json_feature_map(path: str) -> dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(key): float(value) for key, value in dict(payload).items() if isinstance(value, (int, float))}


def read_npy_feature_vector(path: str) -> list[float]:
    if not path or not Path(path).exists():
        return []
    try:
        import numpy as np

        return np.asarray(np.load(path), dtype=float).reshape(-1).tolist()
    except Exception:
        return []


def add_tool_result(
    patient_state: Mapping[str, Any],
    *,
    bucket_name: str,
    evidence_key: str,
    tool_result: Mapping[str, Any],
    node: str,
) -> dict[str, Any]:
    updated = dict(patient_state)
    case_id = str(updated.get("case_id", "") or "")
    artifacts = dict(tool_result.get("artifacts", {}) or {})
    if bucket_name == "ct_evidence" and evidence_key == "radiomics":
        feature_path = str(artifacts.get("features_json_path", "") or "")
        evidence = {"features": to_jsonable(read_json_feature_map(feature_path)), "feature_path": feature_path}
    elif bucket_name == "wsi_evidence" and evidence_key == "embeddings":
        feature_path = str(artifacts.get("slide_embedding_npy_path", "") or artifacts.get("slide_embedding_path", "") or "")
        evidence = {"features": to_jsonable(read_npy_feature_vector(feature_path)), "feature_path": feature_path}
    else:
        evidence = dict(updated.get(bucket_name, {}) or {})
    updated[bucket_name] = evidence
    announce_tool_result_path(node, case_id, tool_result)
    return updated


def add_omics_result(
    patient_state: Mapping[str, Any],
    *,
    evidence_key: str,
    result_bundle: Mapping[str, Any],
    node: str,
) -> dict[str, Any]:
    updated = dict(patient_state)
    case_id = str(updated.get("case_id", "") or "")
    tool_result = dict(result_bundle.get("tool_result", {}) or {})
    payload = dict(result_bundle.get("payload", {}) or {})
    artifacts = dict(tool_result.get("artifacts", {}) or {})
    omics_evidence = dict(updated.get("omics_evidence", {}))
    normalized_key = "rna" if evidence_key in {"rna", "rna_seq"} else evidence_key
    feature_path = str(artifacts.get("case_features_path", "") or "")
    omics_evidence[f"{normalized_key}_features"] = to_jsonable(list(payload.get("feature_values", []) or []))
    omics_evidence[f"{normalized_key}_feature_path"] = feature_path
    if normalized_key == "rna" and artifacts.get("pathway_features_path"):
        omics_evidence["rna_pathway_feature_path"] = str(
            artifacts.get("pathway_features_path", "") or ""
        )
    if normalized_key == "wxs" and artifacts.get("all_features_path"):
        omics_evidence["wxs_full_feature_path"] = str(
            artifacts.get("all_features_path", "") or ""
        )
    updated["omics_evidence"] = omics_evidence
    announce_tool_result_path(node, case_id, tool_result)
    return updated
