from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from tools.subtype_review_common import clinical_table, member_case_ids, tool_result


def round_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return round(number, 6)


def candidate_set_members(cluster_state: Mapping[str, Any], all_cluster_states: Any) -> dict[str, list[str]]:
    cluster_states = list(all_cluster_states or []) or [cluster_state]
    memberships: dict[str, list[str]] = {}
    assigned = set()
    for index, state in enumerate(cluster_states):
        row = dict(state or {})
        set_id = str(row.get("cluster_id") or row.get("candidate_set_id") or f"C{index + 1}")
        members = []
        for case_id in sorted(member_case_ids(row)):
            if case_id not in assigned:
                members.append(case_id)
                assigned.add(case_id)
        if members:
            memberships[set_id] = members
    return memberships


def survival_rows(
    memberships: Mapping[str, list[str]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    clinical = clinical_table(patient_states_by_id)
    total = sum(len(members) for members in memberships.values())
    rows = []
    for set_id, members in memberships.items():
        for case_id in members:
            record = dict(clinical.get(case_id, {}) or {})
            time = record.get("os_time")
            event = record.get("os_event")
            if time is None or event is None:
                continue
            try:
                time_value = float(time)
                event_value = int(event)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(time_value):
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_set_id": set_id,
                    "time": time_value,
                    "event": event_value,
                    "age": record.get("age"),
                    "stage_group": record.get("stage_group"),
                    "grade": record.get("grade"),
                }
            )
    return rows, total - len(rows)


def median_os_days(records: list[dict[str, Any]]) -> Any:
    if not records:
        return None
    try:
        kmf = KaplanMeierFitter().fit(
            [row["time"] for row in records],
            [row["event"] for row in records],
        )
        return round_value(kmf.median_survival_time_)
    except Exception:
        return None


def global_logrank(rows: list[dict[str, Any]]) -> Any:
    if len({row["candidate_set_id"] for row in rows}) < 2:
        return None
    try:
        return round_value(
            multivariate_logrank_test(
                [row["time"] for row in rows],
                [row["candidate_set_id"] for row in rows],
                [row["event"] for row in rows],
            ).p_value
        )
    except Exception:
        return None


def set_logrank(set_records: list[dict[str, Any]], rest_records: list[dict[str, Any]]) -> Any:
    if not set_records or not rest_records:
        return None
    try:
        return round_value(
            logrank_test(
                [row["time"] for row in set_records],
                [row["time"] for row in rest_records],
                [row["event"] for row in set_records],
                [row["event"] for row in rest_records],
            ).p_value
        )
    except Exception:
        return None


def cox_set_vs_rest(set_records: list[dict[str, Any]], rest_records: list[dict[str, Any]]) -> dict[str, Any]:
    null_result = {
        "hazard_ratio": None,
        "ci_95_lower": None,
        "ci_95_upper": None,
        "cox_p_value": None,
        "direction": "not_estimable",
    }
    if not set_records or not rest_records:
        return null_result
    try:
        frame = pd.DataFrame(
            [
                {"time": row["time"], "event": row["event"], "candidate_set": 1}
                for row in set_records
            ]
            + [
                {"time": row["time"], "event": row["event"], "candidate_set": 0}
                for row in rest_records
            ]
        )
        cph = CoxPHFitter().fit(
            frame,
            duration_col="time",
            event_col="event",
        )
        summary = cph.summary.loc["candidate_set"]
        hazard_ratio = float(summary.get("exp(coef)", np.exp(cph.params_["candidate_set"])))
        lower = float(summary.get("exp(coef) lower 95%"))
        upper = float(summary.get("exp(coef) upper 95%"))
        p_value = float(summary.get("p"))
        if not all(math.isfinite(value) for value in [hazard_ratio, lower, upper, p_value]):
            return null_result
        direction = (
            "worse_survival_in_set"
            if hazard_ratio > 1
            else "better_survival_in_set"
            if hazard_ratio < 1
            else "not_estimable"
        )
        return {
            "hazard_ratio": round_value(hazard_ratio),
            "ci_95_lower": round_value(lower),
            "ci_95_upper": round_value(upper),
            "cox_p_value": round_value(p_value),
            "direction": direction,
        }
    except Exception:
        return null_result


def adjusted_cox_set_vs_rest(
    set_records: list[dict[str, Any]],
    rest_records: list[dict[str, Any]],
    covariates: list[str],
) -> dict[str, Any]:
    null_result = {
        "hazard_ratio": None,
        "ci_95_lower": None,
        "ci_95_upper": None,
        "cox_p_value": None,
        "direction": "not_estimable",
        "covariates": list(covariates),
        "used_covariates": [],
        "available_n": 0,
        "event_n": 0,
        "clinical_c_index": None,
        "clinical_plus_set_c_index": None,
        "delta_c_index": None,
        "not_estimable_reason": "",
    }
    records = [
        {**row, "candidate_set": 1} for row in set_records
    ] + [
        {**row, "candidate_set": 0} for row in rest_records
    ]
    if not records:
        null_result["not_estimable_reason"] = "no_survival_records"
        return null_result
    frame = pd.DataFrame(records)
    keep_columns = ["time", "event", "candidate_set"] + list(covariates)
    frame = frame[keep_columns].copy()
    for column in ["time", "event", "candidate_set"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for covariate in covariates:
        if covariate == "age":
            frame[covariate] = pd.to_numeric(frame[covariate], errors="coerce")
        else:
            frame[covariate] = frame[covariate].replace("", np.nan)
    frame = frame.dropna()
    null_result["available_n"] = int(len(frame))
    null_result["event_n"] = int(frame["event"].sum()) if not frame.empty else 0
    if len(frame) < 6 or int(frame["event"].sum()) < 3:
        null_result["not_estimable_reason"] = "insufficient_events_or_cases"
        return null_result
    used_covariates = []
    model_frame = frame[["time", "event", "candidate_set"]].copy()
    clinical_frame = frame[["time", "event"]].copy()
    for covariate in covariates:
        if covariate == "age":
            if frame[covariate].nunique(dropna=True) > 1:
                model_frame[covariate] = frame[covariate].astype(float)
                clinical_frame[covariate] = frame[covariate].astype(float)
                used_covariates.append(covariate)
            continue
        if frame[covariate].nunique(dropna=True) > 1:
            dummies = pd.get_dummies(frame[covariate].astype(str), prefix=covariate, drop_first=True)
            if not dummies.empty:
                model_frame = pd.concat([model_frame, dummies], axis=1)
                clinical_frame = pd.concat([clinical_frame, dummies], axis=1)
                used_covariates.append(covariate)
    null_result["used_covariates"] = list(used_covariates)
    if not used_covariates:
        null_result["not_estimable_reason"] = "no_informative_covariates"
        return null_result
    if model_frame["candidate_set"].nunique(dropna=True) < 2:
        null_result["not_estimable_reason"] = "no_candidate_set_variation"
        return null_result
    try:
        clinical_c_index = None
        if clinical_frame.shape[1] > 2:
            clinical_model = CoxPHFitter(penalizer=0.1).fit(
                clinical_frame,
                duration_col="time",
                event_col="event",
            )
            clinical_c_index = float(clinical_model.concordance_index_)
        cph = CoxPHFitter(penalizer=0.1).fit(
            model_frame,
            duration_col="time",
            event_col="event",
        )
        summary = cph.summary.loc["candidate_set"]
        hazard_ratio = float(summary.get("exp(coef)", np.exp(cph.params_["candidate_set"])))
        lower = float(summary.get("exp(coef) lower 95%"))
        upper = float(summary.get("exp(coef) upper 95%"))
        p_value = float(summary.get("p"))
        values = [hazard_ratio, lower, upper, p_value]
        if not all(math.isfinite(value) for value in values):
            null_result["not_estimable_reason"] = "non_finite_adjusted_cox"
            return null_result
        plus_c_index = float(cph.concordance_index_)
        direction = (
            "worse_survival_in_set"
            if hazard_ratio > 1
            else "better_survival_in_set"
            if hazard_ratio < 1
            else "not_estimable"
        )
        return {
            **null_result,
            "hazard_ratio": round_value(hazard_ratio),
            "ci_95_lower": round_value(lower),
            "ci_95_upper": round_value(upper),
            "cox_p_value": round_value(p_value),
            "direction": direction,
            "clinical_c_index": round_value(clinical_c_index),
            "clinical_plus_set_c_index": round_value(plus_c_index),
            "delta_c_index": round_value(
                plus_c_index - clinical_c_index
                if clinical_c_index is not None
                else None
            ),
            "not_estimable_reason": "",
        }
    except Exception as exc:
        null_result["not_estimable_reason"] = f"cox_fit_failed:{type(exc).__name__}"
        return null_result


def tool_survival_analysis(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    memberships = candidate_set_members(cluster_state, all_cluster_states)
    rows, missing_n = survival_rows(memberships, patient_states_by_id)
    if not rows:
        return tool_result(
            tool_name="tool_survival_analysis",
            status="missing",
            cluster_id=cluster_id,
            output_root=output_root,
            summary="No OS survival time/event data are available for the reviewed candidate sets.",
            metrics={
                "survival_global_association": {
                    "endpoint": "OS",
                    "available_n": 0,
                    "missing_n": missing_n,
                    "event_n": 0,
                    "censored_n": 0,
                    "per_set_n": {},
                    "per_set_event_n": {},
                    "per_set_censored_n": {},
                    "per_set_median_os_days": {},
                    "logrank_p_value": None,
                },
                "survival_set_association": {},
            },
            missing_reason="no OS time/event data",
            support_level="none",
            concern_level="high",
            figures={},
        )

    records_by_set = {
        set_id: [row for row in rows if row["candidate_set_id"] == set_id]
        for set_id in memberships
    }
    per_set_n = {set_id: len(records) for set_id, records in records_by_set.items()}
    per_set_event_n = {
        set_id: int(sum(row["event"] for row in records))
        for set_id, records in records_by_set.items()
    }
    per_set_censored_n = {
        set_id: int(len(records) - per_set_event_n[set_id])
        for set_id, records in records_by_set.items()
    }
    per_set_median = {
        set_id: median_os_days(records) for set_id, records in records_by_set.items()
    }
    global_metrics = {
        "endpoint": "OS",
        "available_n": len(rows),
        "missing_n": missing_n,
        "event_n": int(sum(row["event"] for row in rows)),
        "censored_n": int(len(rows) - sum(row["event"] for row in rows)),
        "per_set_n": per_set_n,
        "per_set_event_n": per_set_event_n,
        "per_set_censored_n": per_set_censored_n,
        "per_set_median_os_days": per_set_median,
        "logrank_p_value": global_logrank(rows),
    }

    set_metrics = {}
    for set_id, set_records in records_by_set.items():
        rest_records = [row for row in rows if row["candidate_set_id"] != set_id]
        cox = cox_set_vs_rest(set_records, rest_records)
        rest_event_n = int(sum(row["event"] for row in rest_records))
        set_event_n = int(sum(row["event"] for row in set_records))
        set_metrics[set_id] = {
            "candidate_set_id": set_id,
            "endpoint": "OS",
            "available_n": len(rows),
            "missing_n": missing_n,
            "set_n": len(set_records),
            "rest_n": len(rest_records),
            "set_event_n": set_event_n,
            "rest_event_n": rest_event_n,
            "set_censored_n": int(len(set_records) - set_event_n),
            "rest_censored_n": int(len(rest_records) - rest_event_n),
            "set_median_os_days": median_os_days(set_records),
            "rest_median_os_days": median_os_days(rest_records),
            **cox,
            "logrank_p_value": set_logrank(set_records, rest_records),
            "adjusted_cox_age_stage": adjusted_cox_set_vs_rest(
                set_records,
                rest_records,
                ["age", "stage_group"],
            ),
            "adjusted_cox_age_grade": adjusted_cox_set_vs_rest(
                set_records,
                rest_records,
                ["age", "grade"],
            ),
        }
        set_metrics[set_id]["delta_c_index_age_stage"] = set_metrics[set_id][
            "adjusted_cox_age_stage"
        ].get("delta_c_index")
        set_metrics[set_id]["delta_c_index_age_grade"] = set_metrics[set_id][
            "adjusted_cox_age_grade"
        ].get("delta_c_index")

    return tool_result(
        tool_name="tool_survival_analysis",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary=(
            f"OS survival metrics were computed for {len(memberships)} candidate sets: "
            f"{global_metrics['available_n']} available cases and {global_metrics['event_n']} events."
        ),
        metrics={
            "survival_global_association": global_metrics,
            "survival_set_association": set_metrics,
        },
        evidence_hints=[
            {
                "evidence_type": "survival",
                "summary": (
                    f"OS global log-rank p={global_metrics['logrank_p_value']}; "
                    f"{global_metrics['event_n']} events among {global_metrics['available_n']} available cases."
                ),
            }
        ],
        support_level="informational",
        concern_level="none",
        figures={},
    )
