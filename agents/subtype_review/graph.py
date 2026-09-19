from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.subtype_review.llm import parse_json_content, parse_revision_plan, parse_router_plan
from agents.subtype_review.llm_summary import summarize_reports
from agents.subtype_review.schemas import EVIDENCE_DIMENSIONS, EvidenceRequest, EvidenceReportBatch, ReviewContext, ReviewState, RouterPlan, set_id
from utils.tool_utils import to_jsonable

# Keep the existing graph import surface for experiment runners and contract tests.
from agents.subtype_review.state import (
    partition_signature,
    current_sets,
    compact_partition_for_llm,
    context_values,
    append_trace,
    is_length_finish_error,
    mark_failure,
    mark_success,
    initial_review_state,
    history_entry,
    reset_verifier_round_state,
)
from agents.subtype_review.evidence import (
    required_reports_for_round,
    expected_tool_refs_for_round,
    eligible_tools_for_requests,
    completed_tool_keys,
    available_evidence_requests,
    validate_selected_tool_coverage,
    execute_tool_calls,
    available_revision_metric_refs,
    validate_reports,
    evidence_coverage,
    validate_decision_state_evidence,
    validate_router_plan,
    revision_metrics,
    compact_structural_index,
)
from agents.subtype_review.revision import (
    revision_plan_signature,
    validate_revision_plan,
    execute_revision_plan,
)


def verifier_node(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
    values = context_values(runtime)
    control = dict(state["control"])
    try:
        if control.get("next") == "verifier_acquire":
            requests = [
                EvidenceRequest.model_validate(item).model_dump()
                for item in control.get("pending_evidence_requests", [])
            ]
            if not requests:
                raise ValueError("Verifier requires Router EvidenceRequests")
            eligible = eligible_tools_for_requests(state, values, requests)
            if not eligible:
                raise ValueError("No eligible tool can answer Router EvidenceRequests")
            control["eligible_tools"] = eligible
            state["control"] = control
            model = values["verifier_model"]
            result = model.invoke({
                "mode": "acquire",
                "partition": state["partition"],
                "evidence_requests": requests,
                "current_evidence": state.get("reports", []),
                "eligible_tools": eligible,
                "round": control["round"],
            })
            execute_tool_calls(state, result, values)
            mark_success(state)
            return state
        if control.get("next") != "verifier_audit":
            raise ValueError(f"Unexpected Verifier state: {control.get('next')}")
        audit_payload = {
            "mode": "audit",
            "partition": compact_partition_for_llm(state),
            "required_reports": [
                {key: value for key, value in item.items() if key != "tool_names"}
                for item in required_reports_for_round(state, values)
            ],
            "prior_reports": state.get("reports", []),
            "round_evidence": state.get("round_evidence", []),
            "tool_messages": list(state.get("messages", [])),
            "round": control["round"],
        }
        append_trace(state, {
            "node": "verifier",
            "event": "audit_payload_size",
            "request_chars": len(json.dumps(
                {key: value for key, value in audit_payload.items() if key != "tool_messages"},
                ensure_ascii=False,
            )),
            "history_chars": sum(
                len(str(getattr(message, "content", message)))
                for message in audit_payload["tool_messages"]
            ),
            "tool_message_chars": [
                len(str(getattr(message, "content", message)))
                for message in audit_payload["tool_messages"]
            ],
        })
        result = values["verifier_model"].invoke(audit_payload)
        data = result if isinstance(result, Mapping) else parse_json_content(getattr(result, "content", result))
        batch = EvidenceReportBatch.model_validate(data)
        expected_refs = expected_tool_refs_for_round(state, values)
        for report in batch.reports:
            key = (
                report.dimension,
                report.scope,
                next(iter(report.target_ids), ""),
            )
            report.tool_refs = expected_refs.get(key, [])
        try:
            validate_reports(batch, state, values)
        except Exception as exc:
            mark_failure(state, "verifier", exc, immediate=True)
            return state
        merged = {
            (
                str(report["dimension"]),
                str(report["scope"]),
                next(iter(report.get("target_ids", [])), ""),
            ): dict(report)
            for report in state.get("reports", []) or []
        }
        for report in batch.reports:
            key = (report.dimension, report.scope, next(iter(report.target_ids), ""))
            merged[key] = report.model_dump()
        state["reports"] = list(merged.values())
        signature = control.get("partition_signature", partition_signature(current_sets(state)))
        state.setdefault("evidence_memory", {})[signature] = copy.deepcopy(state["reports"])
        mark_success(state)
        append_trace(state, {"node": "verifier", "event": "reports", "count": len(batch.reports)})
        control = dict(state["control"])
        control["pending_evidence_requests"] = []
        control["eligible_tools"] = {}
        control["next"] = "router"
        state["control"] = control
    except Exception as exc:
        mark_failure(state, "verifier", exc, is_length_finish_error(exc))
    return state


def router_node(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
    values = context_values(runtime)
    control = dict(state["control"])
    if control.get("status") != "reviewing":
        return state
    terminal_only = control.get("round", 0) >= control.get("max_rounds", 10)
    plan = None
    try:
        payload = {
            "partition": state["partition"],
            "evidence_reports": summarize_reports(state.get("reports", [])),
            "evidence_coverage": evidence_coverage(state, values),
            "structural_index": compact_structural_index(state),
            "available_evidence_requests": available_evidence_requests(state, values),
            "round": (
                int(control.get("round", 0))
                if terminal_only
                else int(control.get("round", 0)) + 1
            ),
        }
        if terminal_only:
            payload["terminal_only"] = True
        if control.get("router_validation_error"):
            payload["validation_error"] = control["router_validation_error"]
            payload["instruction"] = "return a corrected RouterPlan only"
            payload["correction_rules"] = [
                "Any dimension marked unassessed in evidence_coverage must remain unassessed in decision_state.",
                "Do not convert an unassessed dimension to a favorable state.",
                "If an unassessed dimension is decision-critical, request its available evidence; otherwise a terminal action may retain unassessed and explain why.",
            ]
            if control.get("previous_invalid_plan") is not None:
                payload["previous_invalid_plan"] = control["previous_invalid_plan"]
        plan = parse_router_plan(
            values["router_model"].invoke(copy.deepcopy(payload))
        )
        validate_router_plan(plan, state, values)
        if terminal_only and (
            plan.evidence_requests
            or any(action.action not in {"accept", "drop"} for action in plan.actions)
        ):
            control["status"] = "review_incomplete_due_to_round_budget"
            control["next"] = "end"
            control["error"] = "Router requested a non-terminal action after round budget"
            state["control"] = control
            append_trace(state, {
                "node": "router",
                "event": "round_budget_exhausted",
                "plan": plan.model_dump(),
            })
            return state
    except Exception as exc:
        if is_length_finish_error(exc):
            mark_failure(state, "router", exc, immediate=True)
        elif isinstance(exc, ValueError):
            attempts = int(control.get("router_correction_attempts", 0)) + 1
            if attempts > 2:
                mark_failure(state, "router", exc, immediate=True)
                return state
            control["router_correction_attempts"] = attempts
            control["router_validation_error"] = f"{type(exc).__name__}: {exc}"
            control["previous_invalid_plan"] = plan.model_dump() if plan else None
            control["error"] = control["router_validation_error"]
            control["next"] = "router"
            state["control"] = control
            append_trace(state, {
                "node": "router",
                "event": "validation_retry",
                "attempt": attempts,
                "error": control["router_validation_error"],
            })
        else:
            mark_failure(state, "router", exc, immediate=False)
        return state
    if not terminal_only:
        control["round"] = int(control.get("round", 0)) + 1
    control["error"] = None
    control["router_validation_error"] = None
    control["router_correction_attempts"] = 0
    control["previous_invalid_plan"] = None
    state["router_plan"] = plan.model_dump()
    state["history"].append(history_entry(state, plan, terminal_only))
    control["history_index"] = len(state["history"]) - 1
    append_trace(state, {
        "node": "router",
        "event": "decision",
        "plan": plan.model_dump(),
    })
    control["pending_evidence_requests"] = []
    control["eligible_tools"] = {}
    if plan.evidence_requests:
        control["pending_evidence_requests"] = [request.model_dump() for request in plan.evidence_requests]
        reset_verifier_round_state(state)
        control["next"] = "verifier_acquire"
    elif any(action.action in {"split", "merge"} for action in plan.actions):
        control["next"] = "reviser"
    else:
        control["status"] = "complete"
        control["next"] = "end"
    state["control"] = control
    mark_success(state)
    return state


def reviser_node(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
    values = context_values(runtime)
    router_plan = RouterPlan.model_validate(state["router_plan"])
    control = dict(state["control"])
    payload = {
        "partition": state["partition"],
        "router_plan": router_plan.model_dump(),
        "raw_structural_metrics": revision_metrics(state),
        "available_metric_refs": available_revision_metric_refs(state, router_plan),
    }
    if control.get("revision_validation_error"):
        payload["previous_revision_plan"] = state.get("revision_plan")
        payload["revision_validation_error"] = control["revision_validation_error"]
    try:
        result = values["reviser_model"].invoke(payload)
        plan = parse_revision_plan(result)
        signature = revision_plan_signature(plan)
        failed = set(control.get("failed_revision_plan_signatures", []) or [])
        if signature in failed:
            raise ValueError("Reviser returned a previously failed RevisionPlan")
        state["revision_plan"] = plan.model_dump()
        try:
            validate_revision_plan(plan, router_plan, state)
            revision_result = execute_revision_plan(state, plan, values)
        except Exception as exc:
            control = dict(state["control"])
            control["failed_revision_plan_signatures"] = sorted(
                set(control.get("failed_revision_plan_signatures", []) or []) | {signature}
            )
            control["revision_validation_error"] = f"{type(exc).__name__}: {exc}"
            state["control"] = control
            raise
        state["revision_result"] = revision_result
        index = int(state["control"].get("history_index", len(state["history"]) - 1))
        state["history"][index]["revision_plan"] = copy.deepcopy(state["revision_plan"])
        state["history"][index]["revision_result"] = copy.deepcopy(revision_result)
        append_trace(state, {"node": "reviser", "event": "revision_applied", "result": revision_result})
        mark_success(state)
        control = dict(state["control"])
        control["revision_validation_error"] = None
        reset_verifier_round_state(state)
        control["next"] = "router"
        state["control"] = control
    except Exception as exc:
        mark_failure(state, "reviser", exc, is_length_finish_error(exc))
    return state


def route_node(state: Mapping[str, Any], *, node: str, destinations: Mapping[str, str], default: str) -> str:
    control = state["control"]
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return node
    return destinations.get(control.get("next", ""), default)


def invoke_node(state: ReviewState, runtime: Runtime[ReviewContext], *, node: Callable, writes: tuple[str, ...]) -> dict[str, Any]:
    """Isolate the mutable transition helpers, then emit explicit graph updates.

    Evidence and memberships are read-only inside a node. Only these containers
    are mutated in place; copying their owners preserves earlier graph snapshots.
    """
    local = {
        **state,
        "control": {**state["control"], "trace": list(state["control"].get("trace", []))},
        "history": [dict(entry) for entry in state.get("history", [])],
        "evidence_memory": dict(state.get("evidence_memory", {})),
    }
    result = node(local, runtime)
    return {key: result[key] for key in writes if key in result}


def build_review_graph() -> Any:
    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    nodes = (
        ("verifier", verifier_node, ("control", "round_evidence", "messages", "reports", "evidence_memory"),
         {"router": "router"}, "verifier"),
        ("router", router_node, ("control", "router_plan", "history", "round_evidence", "messages"),
         {"verifier_acquire": "verifier", "reviser": "reviser"}, "end"),
        ("reviser", reviser_node, ("control", "partition", "reports", "revision_plan", "revision_result", "history", "round_evidence", "messages"),
         {"router": "router"}, "end"),
    )
    for name, node, writes, destinations, default in nodes:
        graph.add_node(name, partial(invoke_node, node=node, writes=writes))
        graph.add_conditional_edges(
            name, partial(route_node, node=name, destinations=destinations, default=default),
            {target: END if target == "end" else target for target in {name, "end", *destinations.values()}},
        )
    graph.add_edge(START, "router")
    return graph.compile()


def evidence_for_set(state: Mapping[str, Any], target: str) -> list[dict[str, Any]]:
    return [
        report for report in state.get("reports", []) or []
        if report.get("scope") == "partition" or target in report.get("target_ids", [])
    ]


def metric_refs_for_reports(reports: list[Mapping[str, Any]]) -> list[str]:
    return sorted({
        ref
        for report in reports
        for ref in report.get("metric_refs", []) or []
    })


def final_actions(state: Mapping[str, Any]) -> dict[str, str]:
    plan = state.get("router_plan") or {}
    return {
        target: action.get("action", "")
        for action in plan.get("actions", []) or []
        for target in action.get("target_ids", [])
        if action.get("action") in {"accept", "drop"}
    }


def save_review_outputs(state: Mapping[str, Any], output_root: str, *, direct: bool = False) -> dict[str, Any]:
    root = Path(output_root) if direct else Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    actions = final_actions(state)
    sets = current_sets(state)
    accepted = [item for item in sets if actions.get(set_id(item)) == "accept"]
    dropped = [item for item in sets if actions.get(set_id(item)) == "drop"]
    accepted_reports = []
    for item in accepted:
        reports = evidence_for_set(state, set_id(item))
        accepted_reports.append({
            "set_id": set_id(item),
            "membership": item["member_ids"],
            "revision_lineage": item.get("revision_lineage", []),
            "evidence_by_dimension": {
                dimension: [report for report in reports if report.get("dimension") == dimension]
                for dimension in EVIDENCE_DIMENSIONS
            },
            "metric_refs": metric_refs_for_reports(reports),
            "reports": reports,
            "sections": [
                "CT", "WSI", "RNA/pathway", "WXS", "CNV", "clinical",
                "cross-modal", "confounder", "known-label", "statistics",
                "medical_interpretation", "limitations",
            ],
        })
    dropped_reports = [
        {
            "set_id": set_id(item),
            "membership": item["member_ids"],
            "drop_reason": next(
                action.get("reason", "")
                for action in state.get("router_plan", {}).get("actions", []) or []
                if set_id(item) in action.get("target_ids", [])
            ),
            "key_evidence": evidence_for_set(state, set_id(item)),
            "decision_trace": state["control"].get("trace", []),
        }
        for item in dropped
    ]
    status = str(state["control"].get("status", "review_unavailable"))
    summary = {
        "stage": "subtype_review",
        "status": "review_complete" if status == "complete" else status,
        "raw_control_status": status,
        "rounds_used": state["control"].get("round", 0),
        "llm_usage": to_jsonable(state["control"].get("llm_usage", {})),
        "partition": to_jsonable(state["partition"]),
        "partition_sets": to_jsonable(sets),
        "accepted_subtype_sets": to_jsonable(accepted),
        "accepted_subtype_reports": to_jsonable(accepted_reports),
        "dropped_set_registry": to_jsonable(dropped_reports),
        "accepted_patient_count": len({member for item in accepted for member in item["member_ids"]}),
        "history": to_jsonable(state.get("history", [])),
        "decision_trace": to_jsonable(state["control"].get("trace", [])),
    }
    for filename, payload in {
        "final_partition_sets.json": sets,
        "final_subtype_sets.json": accepted,
        "accepted_subtype_reports.json": accepted_reports,
        "dropped_set_reports.json": dropped_reports,
        "review_history.json": state.get("history", []),
        "final_review_summary.json": summary,
    }.items():
        (root / filename).write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "decision_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in state["control"].get("trace", []):
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    return summary
