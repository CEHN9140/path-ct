from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agents.common import announce_tool_action, case_from_state, selected_ct_from_result
from agents.inventory import build_patient_state


def wsi_qc_node(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> dict[str, Any]:
    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    if not case.get("WSI"):
        return {**build_patient_state(case, overall="fail"), "qc": "fail"}

    summary_path = Path(output_root) / "wsi_qc" / case_id / "selection_summary.json"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8").strip():
        announce_tool_action("wsi_qc_node", case_id, "Reuse cached tool `wsi_qc`.")
        wsi_result = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        announce_tool_action("wsi_qc_node", case_id, "Call tool `wsi_qc`.")
        from tools.wsi_qc import run_wsi_qc

        wsi_result = run_wsi_qc(
            case_id,
            list(case.get("WSI", []) or []),
            output_root=output_root,
            config_dir=config_dir,
        )

    selected_wsi = dict(wsi_result.get("selected_slide") or {})
    if not selected_wsi.get("passes_threshold", False):
        return {**build_patient_state(case, overall="fail"), "qc": "fail"}
    return build_patient_state(case, overall="success")


def ct_qc_node(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> dict[str, Any]:
    if state.get("qc") != "success":
        return state

    case = case_from_state(state)
    case_id = str(case.get("Case_ID", "") or "unknown_case")
    if not case.get("CT"):
        return {**build_patient_state(case, overall="fail"), "qc": "fail"}

    summary_path = Path(output_root) / "ct_qc" / case_id / "selection_summary.json"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8").strip():
        announce_tool_action("ct_qc_node", case_id, "Reuse cached tool `ct_qc`.")
        ct_result = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        announce_tool_action("ct_qc_node", case_id, "Call tool `ct_qc`.")
        from tools.ct_qc import run_ct_qc

        ct_result = run_ct_qc(
            case_id,
            list(case.get("CT", []) or []),
            output_root=output_root,
            config_dir=config_dir,
        )

    selected_ct = selected_ct_from_result(ct_result, case_id)
    if not selected_ct.get("passes_threshold", False):
        return {**build_patient_state(case, overall="fail"), "qc": "fail"}
    return build_patient_state(case, overall="success")


def quality_control(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> dict[str, Any]:
    patient_state = ct_qc_node(state, output_root=output_root, config_dir=config_dir)
    if patient_state.get("qc") == "success":
        patient_state = wsi_qc_node(
            patient_state, output_root=output_root, config_dir=config_dir
        )
    return dict(patient_state)


def run_ct_qc_cohort(
    patient_states: list[Mapping[str, Any]], *, output_root: str, config_dir: str
) -> list[dict[str, Any]]:
    from tools.ct_qc import run_ct_qc_cohort as run_ct_qc_cohort_tool

    cases = [case_from_state(item) for item in patient_states]
    ct_result = run_ct_qc_cohort_tool(cases, output_root=output_root, config_dir=config_dir)
    summaries = dict(ct_result.get("selection_summaries", {}) or {})
    updated_states = []
    for patient_state in patient_states:
        case = case_from_state(patient_state)
        case_id = str(case.get("Case_ID", "") or "unknown_case")
        if not case.get("CT"):
            updated_states.append({**build_patient_state(case, overall="fail"), "qc": "fail"})
            continue
        summary = dict(summaries.get(case_id, {}) or {})
        selected_ct = selected_ct_from_result(summary, case_id)
        if not selected_ct.get("passes_threshold", False):
            updated_states.append({**build_patient_state(case, overall="fail"), "qc": "fail"})
            continue
        updated_states.append(build_patient_state(case, overall="success"))
    return updated_states


def run_wsi_qc_cohort(
    patient_states: list[Mapping[str, Any]], *, output_root: str, config_dir: str
) -> list[dict[str, Any]]:
    from tools.wsi_qc import run_wsi_qc_cohort as run_wsi_qc_cohort_tool

    cases = [
        case_from_state(item)
        for item in patient_states
        if dict(item).get("qc") == "success"
    ]
    if not cases:
        return [dict(item) for item in patient_states]
    wsi_result = run_wsi_qc_cohort_tool(cases, output_root=output_root, config_dir=config_dir)
    summaries = dict(wsi_result.get("selection_summaries", {}) or {})
    updated_states = []
    for patient_state in patient_states:
        case = case_from_state(patient_state)
        case_id = str(case.get("Case_ID", "") or "unknown_case")
        if patient_state.get("qc") != "success":
            updated_states.append(dict(patient_state))
            continue
        if not case.get("WSI"):
            updated_states.append({**build_patient_state(case, overall="fail"), "qc": "fail"})
            continue
        summary = dict(summaries.get(case_id, {}) or {})
        selected_wsi = dict(summary.get("selected_slide") or {})
        if not selected_wsi.get("passes_threshold", False):
            updated_states.append({**build_patient_state(case, overall="fail"), "qc": "fail"})
            continue
        updated_states.append(build_patient_state(case, overall="success"))
    return updated_states
