from __future__ import annotations

from typing import Any, Mapping

from agents.common import (
    add_execution_errors,
    case_from_state,
    selected_ct_from_result,
)
from agents.inventory import build_patient_state


def failed_qc_state(
    case: Mapping[str, Any], summary: Mapping[str, Any], tool_name: str
) -> dict[str, Any]:
    state = {**build_patient_state(case, overall="fail"), "qc": "fail"}
    if summary.get("cacheable") is False:
        return add_execution_errors(
            state, tool_name, list(summary.get("errors", []) or [])
        )
    return state


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
            updated_states.append(failed_qc_state(case, summary, "ct_qc"))
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
    return updated_states
