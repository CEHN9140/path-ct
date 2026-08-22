from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.subtype_review.llm import parse_json_content, parse_reviser_output, parse_router_output
from agents.subtype_review.llm_summary import summarize_evidence, summarize_reports
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS,
    EvidenceReportBatch,
    ReviewContext,
    ReviewState,
    RouterAction,
    RouterOutput,
    active_sets,
    set_id,
)
from agents.subtype_review.tools import VALIDATION_FUNCTIONS
from utils.tool_utils import to_jsonable


def partition_signature(sets: list[dict[str, Any]]) -> str:
    return json.dumps(
        sorted(sorted(str(member) for member in item.get("member_ids", [])) for item in sets),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def partition_artifact_id(signature: str) -> str:
    return "p_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def current_sets(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(active_sets(list(state.get("sets", []) or [])), key=set_id)


def subject_signature(
    dimension: str,
    scope: str,
    sets: list[dict[str, Any]],
    target_ids: list[str] | None = None,
) -> str:
    targets = sorted(str(target) for target in target_ids or [])
    members = {
        set_id(item): sorted(str(member) for member in item.get("member_ids", []))
        for item in sets
        if set_id(item) in targets or scope == "partition"
    }
    payload = {
        "dimension": dimension,
        "scope": scope,
        "partition": partition_signature(sets),
        "targets": targets,
        "members": members,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evidence_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(key, "") or "")
        for key in ("dimension", "scope", "subject_signature")
    )


def raw_results(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(dict(state.get("evidence", {}) or {}).get("raw", []) or [])


def matching_raw(state: Mapping[str, Any], dimension: str, scope: str, signature: str) -> list[dict[str, Any]]:
    return [
        row for row in raw_results(state)
        if row.get("dimension") == dimension
        and row.get("scope") == scope
        and str(row.get("subject_signature", "")) == signature
    ]


def append_trace(state: dict[str, Any], row: dict[str, Any]) -> None:
    control = dict(state.get("control", {}) or {})
    control.setdefault("trace", []).append({"round": control.get("round", 0), **row})
    state["control"] = control


def mark_failure(state: dict[str, Any], node: str, exc: Exception) -> None:
    control = dict(state.get("control", {}) or {})
    control["failures"] = int(control.get("failures", 0) or 0) + 1
    control["error"] = f"{type(exc).__name__}: {exc}"
    if control["failures"] >= int(control.get("max_failures", 3) or 3):
        control["status"] = "review_unavailable"
    state["control"] = control
    append_trace(state, {"node": node, "event": "failure", "error": control["error"]})


def mark_success(state: dict[str, Any]) -> None:
    control = dict(state.get("control", {}) or {})
    control["failures"] = 0
    control["error"] = None
    state["control"] = control


def initial_review_state(candidate_sets: list[dict[str, Any]]) -> ReviewState:
    sets = []
    for item in candidate_sets:
        identifier = set_id(dict(item))
        if identifier:
            sets.append({
                "set_id": identifier,
                "cluster_id": identifier,
                "member_ids": sorted(str(member) for member in item.get("member_ids", [])),
                "status": "active",
                "parent_ids": [],
            })
    identifiers = [set_id(item) for item in sets]
    members = [member for item in sets for member in item["member_ids"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Initial candidate sets contain duplicate identifiers")
    if len(members) != len(set(members)):
        raise ValueError("Initial candidate sets contain overlapping patients")
    signature = partition_signature(sets)
    return {
        "sets": sets,
        "evidence": {"raw": [], "history": []},
        "reports": [],
        "messages": [],
        "action": None,
        "revision": None,
        "control": {
            "round": 0,
            "failures": 0,
            "status": "reviewing",
            "next": "router",
            "error": None,
            "max_rounds": 10,
            "max_failures": 3,
            "visited_partitions": [signature],
            "trace": [],
        },
    }


def merge_runtime(base: Mapping[str, Any], context: Mapping[str, Any] | None) -> dict[str, Any]:
    runtime = dict(base)
    if hasattr(context, "context"):
        context = context.context
    runtime.update(dict(context or {}))
    return runtime


def report_summaries(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return summarize_reports(state.get("reports", []))


def all_metric_refs(raw: Mapping[str, Any]) -> set[str]:
    refs = set()
    for result in raw.get("results", []) or []:
        for child in result.get("results", []) or []:
            refs.update(str(ref) for ref in child.get("metric_refs", []) or [])
    return refs


def execute_tool_calls(state: dict[str, Any], message: Any, runtime: Mapping[str, Any]) -> bool:
    calls = list(getattr(message, "tool_calls", []) or [])
    if isinstance(message, Mapping):
        calls = list(message.get("tool_calls", []) or [])
    names = sorted({str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")) for call in calls if call})
    if not names:
        return False
    action = dict(state.get("action", {}) or {})
    if action.get("action") != "need_more_evidence":
        raise ValueError("Verifier tool calls require a pending evidence action")
    requests = [dict(row) for row in action.get("requests", []) or []]
    requested = {str(row["dimension"]) for row in requests}
    if set(names) != requested:
        raise ValueError(f"Verifier tools do not match requests: requested={sorted(requested)}, called={names}")
    sets = current_sets(state)
    partition = partition_signature(sets)
    patient_states = dict(runtime.get("patient_states_by_id", {}) or {})
    cluster_state = {
        "cluster_id": partition_artifact_id(partition),
        "member_ids": sorted(member for item in sets for member in item.get("member_ids", [])),
    }
    evidence = dict(state.get("evidence", {}) or {})
    raw = list(evidence.get("raw", []) or [])
    existing = {evidence_key(row) for row in raw}
    messages = list(state.get("messages", []) or [])
    messages.append(message)
    executed = []
    for request in requests:
        dimension = str(request["dimension"])
        scope = str(request["scope"])
        targets = list(request.get("target_ids", []) or [])
        signature = subject_signature(dimension, scope, sets, targets)
        key = evidence_key({"dimension": dimension, "scope": scope, "subject_signature": signature})
        if key in existing:
            raise ValueError(f"Evidence already exists for {key}")
        artifact_root = Path(str(runtime.get("review_output_root", runtime.get("output_root", "")))) / "evidence" / scope
        payload = VALIDATION_FUNCTIONS[dimension](
            cluster_state,
            patient_states,
            str(runtime.get("data_root", runtime.get("output_root", ""))),
            str(runtime.get("config_dir", "")),
            sets,
            scope=scope,
            target_ids=targets,
            artifact_root=str(artifact_root),
        )
        payload.update({
            "dimension": dimension,
            "scope": scope,
            "target_ids": targets,
            "partition_signature": partition,
            "subject_signature": signature,
        })
        raw.append(payload)
        executed.append(payload)
        existing.add(key)
        try:
            from langchain_core.messages import ToolMessage

            call = next(call for call in calls if str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")) == dimension)
            call_id = str(call.get("id", "") if isinstance(call, Mapping) else getattr(call, "id", "") or dimension)
            messages.append(ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=call_id))
        except Exception:
            messages.append({"role": "tool", "name": dimension, "content": payload})
    evidence["raw"] = raw
    state["evidence"] = evidence
    state["messages"] = messages[-12:]
    control = dict(state.get("control", {}) or {})
    control["next"] = "interpret"
    state["control"] = control
    append_trace(state, {
        "node": "verifier",
        "event": "tools",
        "dimensions": names,
        "evidence_keys": [evidence_key(row) for row in executed],
    })
    if any(str(row.get("status", "")) != "success" for row in executed):
        control["status"] = "review_unavailable"
        state["control"] = control
    return True


def validate_reports(batch: EvidenceReportBatch, state: Mapping[str, Any]) -> None:
    known = {set_id(item) for item in current_sets(state)}
    for report in batch.reports:
        if set(report.target_ids) - known:
            raise ValueError("Evidence report references an inactive set")
        expected = subject_signature(report.dimension, report.scope, current_sets(state), report.target_ids)
        if report.subject_signature and report.subject_signature != expected:
            raise ValueError("Evidence report has an invalid subject_signature")
        exact = matching_raw(state, report.dimension, report.scope, expected)
        if not exact:
            raise ValueError("Evidence report has no exact raw evidence")
        available = all_metric_refs({"results": exact})
        refs = set(report.metric_refs)
        refs.update(ref for observation in report.observations for ref in observation.metric_refs)
        if refs - available:
            raise ValueError(f"Evidence report references unavailable metrics: {sorted(refs - available)}")


def verifier_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    control = dict(state.get("control", {}) or {})
    mode = "acquire" if control.get("next") == "acquire" else "audit"
    requests = list(dict(state.get("action", {}) or {}).get("requests", []) or [])
    payload = {
        "mode": mode,
        "partition": current_sets(state),
        "evidence_inventory": summarize_evidence(raw_results(state)),
        "reports": report_summaries(state),
        "requests": requests,
        "message_history": list(state.get("messages", []) or []),
        "round": control.get("round", 0),
    }
    try:
        result = model.invoke(payload)
        if mode == "acquire":
            if not execute_tool_calls(state, result, runtime):
                raise ValueError("Verifier acquisition returned no tool call")
            if state.get("control", {}).get("status") != "review_unavailable":
                mark_success(state)
            return state
        data = result if isinstance(result, Mapping) else parse_json_content(getattr(result, "content", result))
        batch = EvidenceReportBatch.model_validate(data)
        validate_reports(batch, state)
        if mode == "audit" and requests:
            expected = {
                evidence_key({
                    "dimension": request["dimension"],
                    "scope": request["scope"],
                    "subject_signature": subject_signature(
                        request["dimension"], request["scope"], current_sets(state), request.get("target_ids", []),
                    ),
                })
                for request in requests
            }
            reported = {
                evidence_key({
                    "dimension": report.dimension,
                    "scope": report.scope,
                    "subject_signature": subject_signature(
                        report.dimension, report.scope, current_sets(state), report.target_ids,
                    ),
                })
                for report in batch.reports
            }
            if not expected.issubset(reported):
                raise ValueError("Verifier omitted an Evidence Report for an acquired request")
        reports = list(state.get("reports", []) or [])
        for report in batch.reports:
            item = report.model_dump()
            item["subject_signature"] = subject_signature(
                report.dimension, report.scope, current_sets(state), report.target_ids
            )
            reports = [
                old for old in reports
                if not (
                    old.get("dimension") == item["dimension"]
                    and old.get("scope") == item["scope"]
                    and old.get("subject_signature") == item["subject_signature"]
                )
            ]
            reports.append(item)
        state["reports"] = reports
        state["messages"] = []
        control["next"] = "router"
        state["control"] = control
        mark_success(state)
        append_trace(state, {"node": "verifier", "event": "report", "reports": batch.model_dump()})
    except Exception as exc:
        mark_failure(state, "verifier", exc)
    return state


def structure_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    for row in reversed(raw_results(state)):
        if row.get("dimension") != "cross_modal_consistency" or row.get("scope") != "set_identity":
            continue
        for child in row.get("results", []) or []:
            if child.get("tool_name") == "multimodal_consistency_check":
                return dict(child.get("metrics", {}) or {})
    return {}


def target_evidence_closed(state: Mapping[str, Any], target: str) -> bool:
    relevant_raw = [
        row for row in raw_results(state)
        if row.get("scope") == "partition"
        or target in {str(value) for value in row.get("target_ids", []) or []}
    ]
    if not relevant_raw:
        return False
    report_keys = set()
    for report in state.get("reports", []) or []:
        if report.get("scope") != "partition" and target not in {
            str(value) for value in report.get("target_ids", []) or []
        }:
            continue
        signature = str(report.get("subject_signature", "") or "")
        if not signature:
            signature = subject_signature(
                str(report.get("dimension")),
                str(report.get("scope")),
                current_sets(state),
                list(report.get("target_ids", []) or []),
            )
        report_keys.add(evidence_key({
            "dimension": report.get("dimension"),
            "scope": report.get("scope"),
            "subject_signature": signature,
        }))
    return all(evidence_key(row) in report_keys for row in relevant_raw)


def confounder_invalidates(state: Mapping[str, Any], target: str) -> bool:
    for row in raw_results(state):
        if row.get("dimension") != "confounder_exclusion":
            continue
        for child in row.get("results", []) or []:
            metrics = dict(child.get("metrics", {}) or {})
            flags = dict(metrics.get("deterministic_flags", {}) or {})
            if target in {str(value) for value in flags.get("invalidated_set_ids", []) or []}:
                return True
    return False


def positive_split(state: Mapping[str, Any], target: str) -> bool:
    rows = dict(structure_metrics(state).get("internal_structure_by_set", {}) or {})
    return bool(dict(rows.get(target, {}) or {}).get("positive_internal_heterogeneity"))


def positive_merge(state: Mapping[str, Any], targets: list[str]) -> bool:
    key = "+".join(sorted(targets))
    return key in dict(structure_metrics(state).get("positive_weak_boundary_pairs", {}) or {})


def positive_merge_for_target(state: Mapping[str, Any], target: str) -> bool:
    return any(
        target in key.split("+")
        for key in dict(structure_metrics(state).get("positive_weak_boundary_pairs", {}) or {})
    )


def validate_router_action(action: RouterAction, state: Mapping[str, Any]) -> None:
    known = {set_id(item) for item in current_sets(state)}
    targets = set(action.target_ids)
    if not targets.issubset(known):
        raise ValueError(f"Router referenced inactive or unknown sets: {sorted(targets - known)}")
    if action.action == "need_more_evidence":
        for request in action.requests:
            if request.scope not in {"set_identity", "partition"}:
                raise ValueError("Evidence requests must use set_identity or partition scope")
            if not set(request.target_ids).issubset(known):
                raise ValueError("Evidence request references an inactive set")
            signature = subject_signature(request.dimension, request.scope, current_sets(state), request.target_ids)
            if evidence_key({"dimension": request.dimension, "scope": request.scope, "subject_signature": signature}) in {
                evidence_key(row) for row in raw_results(state)
            }:
                raise ValueError("Evidence request already exists")
        return
    if action.action == "split" and not positive_split(state, action.target_ids[0]):
        raise ValueError("Split requires positive internal heterogeneity")
    if action.action == "merge" and not positive_merge(state, action.target_ids):
        raise ValueError("Merge requires positive weak-boundary evidence")
    if action.action == "accept":
        target = action.target_ids[0]
        if not target_evidence_closed(state, target):
            raise ValueError("Accept requires closed relevant evidence")
        if confounder_invalidates(state, target):
            raise ValueError("Accept is blocked by technical invalidation")
        if positive_split(state, target) or positive_merge_for_target(state, target):
            raise ValueError("Accept is blocked by a positive structural signal")
    if action.action == "drop":
        target = action.target_ids[0]
        if confounder_invalidates(state, target):
            return
        if positive_split(state, target) or positive_merge_for_target(state, target):
            raise ValueError("Drop is blocked by a positive structural signal")
        if not target_evidence_closed(state, target):
            raise ValueError("Drop requires closed evidence or technical invalidation")


def validate_router_output(output: RouterOutput, state: Mapping[str, Any]) -> None:
    actions = output.actions
    if any(action.action == "need_more_evidence" for action in actions) and len(actions) != 1:
        raise ValueError("need_more_evidence must be selected alone")
    occupied = set()
    for action in actions:
        targets = set(action.target_ids)
        if occupied.intersection(targets):
            raise ValueError("A current set may belong to only one action")
        occupied.update(targets)
    for action in actions:
        validate_router_action(action, state)


def set_terminal_status(state: dict[str, Any], action: RouterAction) -> None:
    target = action.target_ids[0]
    item = next(item for item in state["sets"] if set_id(item) == target)
    item["status"] = "provisionally_accepted" if action.action == "accept" else "provisionally_dropped"
    if action.action == "drop":
        item["drop_reason"] = "technical_invalidation" if confounder_invalidates(state, target) else "insufficient_evidence_for_acceptance"


def router_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "review_incomplete_due_to_round_budget"}:
        return state
    if control.get("round", 0) >= control.get("max_rounds", 10):
        if all(item.get("status") in {"provisionally_accepted", "provisionally_dropped", "accept", "drop"} for item in current_sets(state)):
            control["status"] = "complete"
        else:
            control["status"] = "review_incomplete_due_to_round_budget"
        control["next"] = "end"
        state["control"] = control
        append_trace(state, {"node": "router", "event": "round_budget_exhausted"})
        return state
    payload = {
        "partition": [
            {"set_id": set_id(item), "member_count": len(item.get("member_ids", [])), "status": item.get("status", "active")}
            for item in current_sets(state)
        ],
        "evidence_reports": report_summaries(state),
        "raw_evidence_inventory": summarize_evidence(raw_results(state)),
        "available_dimensions": list(EVIDENCE_DIMENSIONS),
        "rules": "Accept needs sufficient independent evidence; Split needs positive internal heterogeneity; Merge needs positive weak boundary; Drop needs technical invalidation or closed evidence that fails Accept.",
        "round": control.get("round", 0),
    }
    try:
        result = model.invoke(payload)
        output = parse_router_output(result)
        validate_router_output(output, state)
    except Exception as exc:
        mark_failure(state, "router", exc)
        return state
    control["round"] = int(control.get("round", 0) or 0) + 1
    control["status"] = "reviewing"
    actions = output.actions
    if actions[0].action == "need_more_evidence":
        state["action"] = actions[0].model_dump()
        control["next"] = "acquire"
    else:
        terminal = [action for action in actions if action.action in {"accept", "drop"}]
        structural = [action for action in actions if action.action in {"split", "merge"}]
        for action in terminal:
            set_terminal_status(state, action)
        if structural:
            state["revision"] = {
                "pending_actions": [action.model_dump() for action in structural],
                "plans": [],
                "partition_signature": partition_signature(current_sets(state)),
            }
            state["action"] = structural[0].model_dump()
            control["next"] = "revise"
        else:
            state["action"] = None
            control["next"] = "router"
            if all(item.get("status") in {"provisionally_accepted", "provisionally_dropped"} for item in current_sets(state)):
                for item in current_sets(state):
                    item["status"] = "accept" if item["status"] == "provisionally_accepted" else "drop"
                control["status"] = "complete"
    state["control"] = control
    mark_success(state)
    append_trace(state, {"node": "router", "event": "actions", "actions": [action.model_dump() for action in actions], "reason": output.actions[0].reason})
    return state


def apply_split(state: dict[str, Any], target: str, groups: list[list[str]]) -> None:
    source = next(item for item in current_sets(state) if set_id(item) == target)
    source_members = set(source.get("member_ids", []))
    flat = [member for group in groups for member in group]
    if len(flat) != len(set(flat)) or set(flat) != source_members or len(groups) < 2:
        raise ValueError("Split plan must partition the source set exactly")
    source["status"] = "superseded_by_split"
    for index, group in enumerate(groups, 1):
        state["sets"].append({
            "set_id": f"{target}_S{index}",
            "cluster_id": f"{target}_S{index}",
            "member_ids": sorted(group),
            "status": "active",
            "parent_ids": [target],
        })


def apply_merge(state: dict[str, Any], targets: list[str]) -> None:
    selected = [item for item in current_sets(state) if set_id(item) in targets]
    if len(selected) != len(targets):
        raise ValueError("Merge plan contains an inactive set")
    members = sorted(member for item in selected for member in item.get("member_ids", []))
    if len(members) != len(set(members)):
        raise ValueError("Merge plan contains overlapping members")
    for item in selected:
        item["status"] = "superseded_by_merge"
    merged_id = "_M_".join(sorted(targets))
    state["sets"].append({
        "set_id": merged_id,
        "cluster_id": merged_id,
        "member_ids": members,
        "status": "active",
        "parent_ids": sorted(targets),
    })


def revision_metrics(state: Mapping[str, Any], target_ids: list[str], action: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    metrics = structure_metrics(state)
    if action == "split":
        return dict(dict(metrics.get("internal_structure_by_set", {}) or {}).get(target_ids[0], {}) or {})
    key = "+".join(sorted(target_ids))
    return dict(dict(metrics.get("boundary_by_pair", {}) or {}).get(key, {}) or {})


def reset_after_structural_change(state: dict[str, Any], signature: str) -> None:
    evidence = dict(state.get("evidence", {}) or {})
    old_signature = str(
        dict(state.get("revision", {}) or {}).get("partition_signature")
        or partition_signature(current_sets(state))
    )
    evidence.setdefault("history", []).append({
        "partition_signature": old_signature,
        "raw": evidence.get("raw", []),
        "reports": state.get("reports", []),
    })
    evidence["raw"] = []
    state["evidence"] = evidence
    state["reports"] = []
    state["revision"] = None
    state["action"] = None
    state["messages"] = []
    for item in current_sets(state):
        item["status"] = "active"
        item.pop("drop_reason", None)
    control = dict(state.get("control", {}) or {})
    control["visited_partitions"] = [*control.get("visited_partitions", []), signature]
    control["next"] = "router"
    state["control"] = control


def reviser_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    revision = dict(state.get("revision", {}) or {})
    pending = list(revision.get("pending_actions", []) or [])
    if not pending:
        return state
    action = RouterAction.model_validate(pending[0])
    targets = sorted(action.target_ids)
    payload = {
        "action": action.model_dump(),
        "raw_structure_metrics": revision_metrics(state, targets, action.action, runtime),
        "evidence_reports": report_summaries(state),
        "legal_execution_strategies": ["multimodal_consensus", "fused_similarity_spectral"],
    }
    try:
        result = model.invoke(payload)
        plan = parse_reviser_output(result)
        if plan.action != action.action or sorted(plan.target_ids) != targets:
            raise ValueError("Reviser plan does not match Router action")
        if action.action == "split":
            if plan.n_children is None or plan.execution_strategy not in {"multimodal_consensus", "fused_similarity_spectral"}:
                raise ValueError("Split plan requires n_children and a legal execution strategy")
            from tools.structural_adequacy import execute_split_membership

            source = next(item for item in current_sets(state) if set_id(item) == targets[0])
            groups = execute_split_membership(
                str(runtime.get("data_root", runtime.get("output_root", ""))),
                list(source.get("member_ids", [])),
                int(plan.n_children),
                str(plan.execution_strategy),
            )
            apply_split(state, targets[0], groups)
        else:
            apply_merge(state, targets)
        revision.setdefault("plans", []).append(plan.model_dump())
        revision["pending_actions"] = pending[1:]
        state["revision"] = revision
        append_trace(state, {"node": "reviser", "event": "plan_applied", "plan": plan.model_dump()})
        mark_success(state)
        if revision["pending_actions"]:
            state["action"] = revision["pending_actions"][0]
            state["control"]["next"] = "revise"
        else:
            signature = partition_signature(current_sets(state))
            reset_after_structural_change(state, signature)
    except Exception as exc:
        mark_failure(state, "reviser", exc)
    return state


def verifier_route(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "review_incomplete_due_to_round_budget", "complete"}:
        return "end"
    if control.get("error"):
        return "retry"
    return "router" if control.get("next") == "router" else "audit"


def route_after_router(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "review_incomplete_due_to_round_budget", "complete"}:
        return "end"
    if control.get("error"):
        return "retry"
    if control.get("next") == "acquire":
        return "verify"
    if control.get("next") == "revise":
        return "revise"
    return "router"


def route_after_reviser(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "review_incomplete_due_to_round_budget"}:
        return "end"
    if control.get("error"):
        return "retry"
    return "revise" if control.get("next") == "revise" else "router"


def build_review_graph(*, verifier_model: Any, router_model: Any, reviser_model: Any, runtime: Mapping[str, Any] | None = None) -> Any:
    base_runtime = dict(runtime or {})

    def verifier(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        return verifier_node(state, merge_runtime(base_runtime, runtime), verifier_model)

    def router(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        return router_node(state, merge_runtime(base_runtime, runtime), router_model)

    def reviser(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        return reviser_node(state, merge_runtime(base_runtime, runtime), reviser_model)

    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    graph.add_node("verifier", verifier)
    graph.add_node("router", router)
    graph.add_node("reviser", reviser)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {"verify": "verifier", "revise": "reviser", "retry": "router", "router": "router", "end": END})
    graph.add_conditional_edges("verifier", verifier_route, {"audit": "verifier", "router": "router", "retry": "verifier", "end": END})
    graph.add_conditional_edges("reviser", route_after_reviser, {"revise": "reviser", "router": "router", "retry": "reviser", "end": END})
    return graph.compile()


def save_review_outputs(state: Mapping[str, Any], output_root: str, *, direct: bool = False) -> dict[str, Any]:
    root = Path(output_root) if direct else Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    control = dict(state.get("control", {}) or {})
    sets = current_sets(state)
    accepted = [item for item in sets if item.get("status") == "accept"]
    dropped = [item for item in sets if item.get("status") == "drop"]
    status = str(control.get("status", "review_unavailable"))
    final_status = {
        "complete": "review_complete",
        "review_unavailable": "review_unavailable",
        "review_incomplete_due_to_round_budget": "review_incomplete_due_to_round_budget",
    }.get(status, "review_failed_runtime")
    summary = {
        "stage": "subtype_review",
        "status": final_status,
        "raw_control_status": status,
        "rounds_used": control.get("round", 0),
        "llm_usage": to_jsonable(control.get("llm_usage", {})),
        "partition_sets": to_jsonable(sets),
        "accepted_subtype_sets": to_jsonable(accepted),
        "dropped_set_registry": to_jsonable(dropped),
        "partition_patient_count": len({member for item in sets for member in item.get("member_ids", [])}),
        "accepted_patient_count": len({member for item in accepted for member in item.get("member_ids", [])}),
        "evidence_reports": to_jsonable(state.get("reports", [])),
        "raw_evidence": to_jsonable(raw_results(state)),
        "evidence_history": to_jsonable(dict(state.get("evidence", {}) or {}).get("history", [])),
        "decision_trace": to_jsonable(control.get("trace", [])),
    }
    (root / "final_partition_sets.json").write_text(json.dumps(to_jsonable(sets), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "final_subtype_sets.json").write_text(json.dumps(to_jsonable(accepted), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "evidence_audit.json").write_text(json.dumps(to_jsonable(state.get("reports", [])), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "raw_evidence.json").write_text(json.dumps(to_jsonable(raw_results(state)), ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "decision_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in control.get("trace", []):
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    (root / "final_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
