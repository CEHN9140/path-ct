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
    EvidenceRequest,
    ReviewContext,
    ReviewState,
    RouterAction,
    RouterOutput,
    active_sets,
    set_id,
)
from agents.subtype_review.tools import VALIDATION_FUNCTIONS
from utils.tool_utils import to_jsonable


FULL_ANALYSIS = "round_validation"


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
    analysis: str = FULL_ANALYSIS,
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
        "analysis": analysis,
        "partition": partition_signature(sets),
        "targets": targets,
        "members": members,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def evidence_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(key, "") or "")
        for key in ("dimension", "scope", "analysis", "subject_signature")
    )


def round_evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(state.get("round_evidence", []) or [])


def append_trace(state: dict[str, Any], event: dict[str, Any]) -> None:
    control = dict(state.get("control", {}) or {})
    control.setdefault("trace", []).append({"round": control.get("round", 0), **event})
    state["control"] = control


def mark_failure(state: dict[str, Any], node: str, exc: Exception) -> None:
    control = dict(state.get("control", {}) or {})
    control["failures"] = int(control.get("failures", 0) or 0) + 1
    control["error"] = f"{type(exc).__name__}: {exc}"
    if control["failures"] >= int(control.get("max_failures", 3) or 3):
        control["status"] = "review_unavailable"
        control["next"] = "end"
    state["control"] = control
    append_trace(state, {"node": node, "event": "failure", "error": control["error"]})


def mark_success(state: dict[str, Any]) -> None:
    control = dict(state.get("control", {}) or {})
    control["failures"] = 0
    control["error"] = None
    state["control"] = control


def full_requests(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = [set_id(item) for item in current_sets(state)]
    requests = []
    for dimension in EVIDENCE_DIMENSIONS:
        requests.append(
            EvidenceRequest(
                dimension=dimension,
                scope="partition" if dimension == "known_label_echo" else "set_identity",
                analysis=FULL_ANALYSIS,
                target_ids=[] if dimension == "known_label_echo" else targets,
            ).model_dump()
        )
    return requests


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
                "revision_lineage": [],
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
        "round_evidence": [],
        "evidence_history": [],
        "reports": [],
        "messages": [],
        "revision": None,
        "control": {
            "round": 0,
            "failures": 0,
            "status": "reviewing",
            "next": "full_acquire",
            "error": None,
            "max_rounds": 10,
            "max_failures": 3,
            "pending_requests": full_requests({"sets": sets}),
            "acquire_kind": "full",
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


def historical_raw(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in state.get("evidence_history", []) or []:
        rows.extend(entry.get("raw", []) or [])
    return rows


def successful_keys(state: Mapping[str, Any]) -> set[str]:
    return {
        evidence_key(row)
        for row in [*round_evidence(state), *historical_raw(state)]
        if str(row.get("status", "")).lower() == "success"
    }


def execute_tool_calls(state: dict[str, Any], message: Any, runtime: Mapping[str, Any]) -> bool:
    calls = list(getattr(message, "tool_calls", []) or [])
    if isinstance(message, Mapping):
        calls = list(message.get("tool_calls", []) or [])
    names = [
        str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", ""))
        for call in calls
        if call
    ]
    requests = [dict(row) for row in dict(state["control"]).get("pending_requests", [])]
    requested_names = [str(row["dimension"]) for row in requests]
    if not calls or len(names) != len(set(names)) or set(names) != set(requested_names):
        raise ValueError(
            f"Verifier tools do not match requests: requested={requested_names}, called={names}"
        )
    sets = current_sets(state)
    partition = partition_signature(sets)
    patient_states = dict(runtime.get("patient_states_by_id", {}) or {})
    cluster_state = {
        "cluster_id": partition_artifact_id(partition),
        "member_ids": sorted(member for item in sets for member in item.get("member_ids", [])),
    }
    raw = list(round_evidence(state))
    existing = {evidence_key(row) for row in raw}
    messages = [*state.get("messages", []), message]
    executed = []
    for request in requests:
        dimension = str(request["dimension"])
        scope = str(request["scope"])
        analysis = str(request["analysis"])
        targets = list(request.get("target_ids", []) or [])
        signature = subject_signature(dimension, scope, sets, targets, analysis)
        key = evidence_key({
            "dimension": dimension,
            "scope": scope,
            "analysis": analysis,
            "subject_signature": signature,
        })
        if key in existing:
            raise ValueError(f"Evidence already exists for {key}")
        payload = VALIDATION_FUNCTIONS[dimension](
            cluster_state,
            patient_states,
            str(runtime.get("data_root", runtime.get("output_root", ""))),
            str(runtime.get("config_dir", "")),
            sets,
            scope=scope,
            target_ids=targets,
            artifact_root=str(
                Path(str(runtime.get("review_output_root", runtime.get("output_root", ""))))
                / "evidence"
                / scope
            ),
        )
        payload.update({
            "dimension": dimension,
            "scope": scope,
            "analysis": analysis,
            "target_ids": targets,
            "partition_signature": partition,
            "subject_signature": signature,
        })
        raw.append(payload)
        executed.append(payload)
        existing.add(key)
        call = next(
            call for call in calls
            if str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")) == dimension
        )
        call_id = str(call.get("id", "") if isinstance(call, Mapping) else getattr(call, "id", "") or dimension)
        try:
            from langchain_core.messages import ToolMessage

            messages.append(ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                tool_call_id=call_id,
            ))
        except Exception:
            messages.append({"role": "tool", "name": dimension, "content": payload})
    state["round_evidence"] = raw
    state["messages"] = messages[-16:]
    control = dict(state["control"])
    control["next"] = "audit"
    state["control"] = control
    append_trace(state, {
        "node": "verifier",
        "event": "tools",
        "analysis": control.get("acquire_kind"),
        "dimensions": names,
        "evidence_keys": [evidence_key(row) for row in executed],
    })
    if any(str(row.get("status", "")) != "success" for row in executed):
        control["status"] = "review_unavailable"
        control["next"] = "end"
        state["control"] = control
    return True


def validate_reports(batch: EvidenceReportBatch, state: Mapping[str, Any]) -> None:
    known = {set_id(item) for item in current_sets(state)}
    requests = [dict(row) for row in dict(state["control"]).get("pending_requests", [])]
    expected = {
        evidence_key({
            "dimension": request["dimension"],
            "scope": request["scope"],
            "analysis": request["analysis"],
            "subject_signature": subject_signature(
                request["dimension"],
                request["scope"],
                current_sets(state),
                request.get("target_ids", []),
                request["analysis"],
            ),
        })
        for request in requests
    }
    reported = set()
    for report in batch.reports:
        if set(report.target_ids) - known:
            raise ValueError("Evidence report references an inactive set")
        signature = subject_signature(
            report.dimension,
            report.scope,
            current_sets(state),
            report.target_ids,
            report.analysis,
        )
        if report.subject_signature and report.subject_signature != signature:
            raise ValueError("Evidence report has an invalid subject_signature")
        exact = [
            row for row in round_evidence(state)
            if row.get("dimension") == report.dimension
            and row.get("scope") == report.scope
            and row.get("analysis") == report.analysis
            and str(row.get("subject_signature", "")) == signature
        ]
        if not exact:
            raise ValueError("Evidence report has no exact round evidence")
        refs = set(report.metric_refs)
        refs.update(ref for observation in report.observations for ref in observation.metric_refs)
        if refs - set().union(*(all_metric_refs(row) for row in exact)):
            raise ValueError("Evidence report references unavailable metrics")
        reported.add(evidence_key({
            "dimension": report.dimension,
            "scope": report.scope,
            "analysis": report.analysis,
            "subject_signature": signature,
        }))
    if not expected.issubset(reported):
        raise ValueError("Verifier omitted an Evidence Report for an acquired request")


def archive_round(state: dict[str, Any], signature: str | None = None) -> None:
    if not round_evidence(state) and not state.get("reports"):
        return
    history = list(state.get("evidence_history", []) or [])
    history.append({
        "round": state["control"].get("round", 0),
        "partition_signature": signature or partition_signature(current_sets(state)),
        "raw": round_evidence(state),
        "reports": list(state.get("reports", []) or []),
    })
    state["evidence_history"] = history
    state["round_evidence"] = []
    state["reports"] = []
    state["messages"] = []


def start_full_round(state: dict[str, Any]) -> None:
    control = dict(state["control"])
    control["pending_requests"] = full_requests(state)
    control["acquire_kind"] = "full"
    control["next"] = "full_acquire"
    state["control"] = control


def verifier_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    control = dict(state["control"])
    try:
        if control.get("next") in {"full_acquire", "supplement_acquire"}:
            result = model.invoke({
                "mode": "acquire",
                "partition": current_sets(state),
                "requests": control.get("pending_requests", []),
                "round": control.get("round", 0),
            })
            if not execute_tool_calls(state, result, runtime):
                raise ValueError("Verifier acquisition returned no tool call")
            if state["control"].get("status") != "review_unavailable":
                mark_success(state)
            return state

        if control.get("next") != "audit":
            raise ValueError(f"Unexpected verifier state: {control.get('next')}")
        result = model.invoke({
            "mode": "audit",
            "partition": current_sets(state),
            "requests": control.get("pending_requests", []),
            "round_evidence": summarize_evidence(round_evidence(state)),
            "reports": report_summaries(state),
            "message_history": list(state.get("messages", []) or []),
            "round": control.get("round", 0),
        })
        data = result if isinstance(result, Mapping) else parse_json_content(getattr(result, "content", result))
        batch = EvidenceReportBatch.model_validate(data)
        validate_reports(batch, state)
        reports = list(state.get("reports", []) or [])
        for report in batch.reports:
            item = report.model_dump()
            item["subject_signature"] = subject_signature(
                report.dimension,
                report.scope,
                current_sets(state),
                report.target_ids,
                report.analysis,
            )
            key = evidence_key(item)
            reports = [old for old in reports if evidence_key(old) != key]
            reports.append(item)
        state["reports"] = reports
        state["messages"] = []
        mark_success(state)
        append_trace(state, {"node": "verifier", "event": "report", "count": len(batch.reports)})
        if control.get("acquire_kind") == "supplement":
            if control.get("round", 0) >= control.get("max_rounds", 10):
                control["status"] = "review_incomplete_due_to_round_budget"
                control["next"] = "end"
            else:
                archive_round(state)
                start_full_round(state)
                control = dict(state["control"])
        else:
            control["next"] = "router"
        state["control"] = control
    except Exception as exc:
        mark_failure(state, "verifier", exc)
    return state


def structure_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    for row in reversed(round_evidence(state)):
        if row.get("dimension") != "cross_modal_consistency":
            continue
        for child in row.get("results", []) or []:
            if child.get("tool_name") == "multimodal_consistency_check":
                return dict(child.get("metrics", {}) or {})
    return {}


def positive_split(state: Mapping[str, Any], target: str) -> bool:
    row = dict(dict(structure_metrics(state).get("internal_structure_by_set", {})).get(target, {}) or {})
    return bool(row.get("positive_internal_heterogeneity"))


def positive_merge(state: Mapping[str, Any], targets: list[str]) -> bool:
    key = "+".join(sorted(targets))
    return key in dict(structure_metrics(state).get("positive_weak_boundary_pairs", {}) or {})


def positive_merge_for_target(state: Mapping[str, Any], target: str) -> bool:
    return any(
        target in key.split("+")
        for key in dict(structure_metrics(state).get("positive_weak_boundary_pairs", {}) or {})
    )


def technical_invalidates(state: Mapping[str, Any], target: str) -> bool:
    for row in round_evidence(state):
        if row.get("dimension") != "confounder_exclusion":
            continue
        for child in row.get("results", []) or []:
            flags = dict(dict(child.get("metrics", {}) or {}).get("deterministic_flags", {}) or {})
            if target in {str(value) for value in flags.get("invalidated_set_ids", []) or []}:
                return True
    return False


def report_for_target(state: Mapping[str, Any], target: str) -> bool:
    return any(
        report.get("scope") == "partition"
        or target in {str(value) for value in report.get("target_ids", []) or []}
        for report in state.get("reports", []) or []
    )


def acceptance_support(state: Mapping[str, Any], target: str) -> bool:
    metrics = structure_metrics(state)
    level = dict(metrics.get("identity_evidence_level_by_set", {}) or {}).get(target)
    return level in {"concordant", "complementary"}


def validate_router_action(action: RouterAction, state: Mapping[str, Any], scientific: bool = True) -> None:
    known = {set_id(item) for item in current_sets(state)}
    if not set(action.target_ids).issubset(known):
        raise ValueError("Router referenced an inactive or unknown set")
    if action.action == "need_more_evidence":
        keys = set()
        for request in action.requests:
            if not set(request.target_ids).issubset(known):
                raise ValueError("Evidence request references an inactive set")
            signature = subject_signature(
                request.dimension,
                request.scope,
                current_sets(state),
                request.target_ids,
                request.analysis,
            )
            key = evidence_key({
                "dimension": request.dimension,
                "scope": request.scope,
                "analysis": request.analysis,
                "subject_signature": signature,
            })
            if key in keys:
                raise ValueError("Duplicate evidence request")
            keys.add(key)
            if request.analysis == FULL_ANALYSIS:
                raise ValueError("Router cannot request the internal full-round analysis")
            if key in successful_keys(state):
                raise ValueError("Evidence request already succeeded for this partition and analysis")
        return
    if not scientific:
        return
    target = action.target_ids[0]
    if action.action == "accept":
        if not report_for_target(state, target):
            raise ValueError("Accept requires current Evidence Reports")
        if not acceptance_support(state, target):
            raise ValueError("Accept requires reliable cross-modal or complementary support")
        if technical_invalidates(state, target):
            raise ValueError("Accept is blocked by technical invalidation")
        if positive_split(state, target) or positive_merge_for_target(state, target):
            raise ValueError("Accept is blocked by a positive structural signal")
    elif action.action == "drop":
        if technical_invalidates(state, target):
            return
        if positive_split(state, target) or positive_merge_for_target(state, target):
            raise ValueError("Drop is blocked by a positive structural signal")
        if not report_for_target(state, target):
            raise ValueError("Drop requires current Evidence Reports or technical invalidation")
    elif action.action == "split" and not positive_split(state, target):
        raise ValueError("Split requires positive internal heterogeneity")
    elif action.action == "merge" and not positive_merge(state, action.target_ids):
        raise ValueError("Merge requires positive weak-boundary evidence")


def validate_router_output(output: RouterOutput, state: Mapping[str, Any]) -> None:
    known = {set_id(item) for item in current_sets(state)}
    occupied = set()
    for action in output.actions:
        targets = set(action.target_ids)
        if occupied.intersection(targets):
            raise ValueError("A current set may belong to only one action")
        occupied.update(targets)
    if occupied != known:
        raise ValueError("Router actions must cover every current set exactly once")
    has_need = any(action.action == "need_more_evidence" for action in output.actions)
    for action in output.actions:
        validate_router_action(action, state, scientific=not has_need)


def set_status(state: dict[str, Any], action: RouterAction) -> None:
    item = next(item for item in state["sets"] if set_id(item) == action.target_ids[0])
    item["status"] = action.action
    if action.action == "drop":
        item["drop_reason"] = (
            "technical_invalidation"
            if technical_invalidates(state, action.target_ids[0])
            else "insufficient_evidence_for_acceptance"
        )


def router_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    control = dict(state["control"])
    if control.get("status") != "reviewing":
        return state
    if control.get("round", 0) >= control.get("max_rounds", 10):
        control["status"] = "review_incomplete_due_to_round_budget"
        control["next"] = "end"
        state["control"] = control
        return state
    payload = {
        "partition": [
            {
                "set_id": set_id(item),
                "member_count": len(item.get("member_ids", [])),
                "status": item.get("status", "active"),
            }
            for item in current_sets(state)
        ],
        "evidence_reports": report_summaries(state),
        "round_evidence_inventory": summarize_evidence(round_evidence(state)),
        "available_dimensions": list(EVIDENCE_DIMENSIONS),
        "rules": "Every current set must occur exactly once. Need Evidence has priority and makes all other actions tentative. Accept requires cross-modal or complementary support, biology, clean confounding and no structural signal. Split requires positive internal heterogeneity. Merge requires positive weak boundary. Drop requires technical invalidation or closed evidence failing Accept.",
        "round": control.get("round", 0) + 1,
    }
    try:
        output = parse_router_output(model.invoke(payload))
        validate_router_output(output, state)
    except Exception as exc:
        mark_failure(state, "router", exc)
        return state

    control["round"] = int(control.get("round", 0)) + 1
    append_trace(state, {
        "node": "router",
        "event": "decision",
        "actions": [action.model_dump() for action in output.actions],
    })
    if any(action.action == "need_more_evidence" for action in output.actions):
        control["pending_requests"] = [
            request.model_dump()
            for action in output.actions
            if action.action == "need_more_evidence"
            for request in action.requests
        ]
        control["acquire_kind"] = "supplement"
        control["next"] = "supplement_acquire"
        state["control"] = control
        mark_success(state)
        return state

    structural = [action for action in output.actions if action.action in {"split", "merge"}]
    for action in output.actions:
        if action.action in {"accept", "drop"}:
            set_status(state, action)
    if structural:
        state["revision"] = {
            "pending_actions": [action.model_dump() for action in structural],
            "plans": [],
            "partition_signature": partition_signature(current_sets(state)),
        }
        control["next"] = "revise"
    else:
        control["next"] = "end"
        control["status"] = "complete"
    state["control"] = control
    mark_success(state)
    return state


def apply_split(state: dict[str, Any], target: str, groups: list[list[str]], plan: Mapping[str, Any]) -> None:
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
            "revision_lineage": [*source.get("revision_lineage", []), dict(plan)],
        })


def apply_merge(state: dict[str, Any], targets: list[str], plan: Mapping[str, Any]) -> None:
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
        "revision_lineage": [
            *sum((item.get("revision_lineage", []) for item in selected), []),
            dict(plan),
        ],
    })


def revision_metrics(state: Mapping[str, Any], target_ids: list[str], action: str) -> dict[str, Any]:
    metrics = structure_metrics(state)
    if action == "split":
        return dict(dict(metrics.get("internal_structure_by_set", {})).get(target_ids[0], {}) or {})
    return dict(dict(metrics.get("boundary_by_pair", {})).get("+".join(sorted(target_ids)), {}) or {})


def reset_after_structural_change(state: dict[str, Any], signature: str) -> None:
    old_signature = str(dict(state.get("revision") or {}).get("partition_signature", ""))
    archive_round(state, old_signature or None)
    for item in current_sets(state):
        item["status"] = "active"
        item.pop("drop_reason", None)
    state["revision"] = None
    control = dict(state["control"])
    control["visited_partitions"] = [*control.get("visited_partitions", []), signature]
    if control.get("round", 0) >= control.get("max_rounds", 10):
        control["status"] = "review_incomplete_due_to_round_budget"
        control["next"] = "end"
    else:
        start_full_round(state)
    state["control"] = control


def reviser_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    revision = dict(state.get("revision") or {})
    pending = list(revision.get("pending_actions", []) or [])
    if not pending:
        return state
    action = RouterAction.model_validate(pending[0])
    targets = sorted(action.target_ids)
    try:
        result = model.invoke({
            "action": action.model_dump(),
            "raw_structure_metrics": revision_metrics(state, targets, action.action),
            "evidence_reports": report_summaries(state),
            "legal_execution_strategies": ["multimodal_consensus", "fused_similarity_spectral"],
        })
        plan = parse_reviser_output(result)
        if plan.action != action.action or sorted(plan.target_ids) != targets:
            raise ValueError("Reviser plan does not match Router action")
        available_refs = all_metric_refs({"results": round_evidence(state)})
        if set(plan.metric_refs) - available_refs:
            raise ValueError("Reviser plan references unavailable metrics")
        if action.action == "split":
            if plan.n_children is None or plan.execution_strategy is None:
                raise ValueError("Split plan requires n_children and execution_strategy")
            from tools.structural_adequacy import execute_split_membership

            source = next(item for item in current_sets(state) if set_id(item) == targets[0])
            groups = execute_split_membership(
                str(runtime.get("data_root", runtime.get("output_root", ""))),
                list(source.get("member_ids", [])),
                int(plan.n_children),
                plan.execution_strategy,
                list(plan.structural_basis),
            )
            apply_split(state, targets[0], groups, plan.model_dump())
        else:
            apply_merge(state, targets, plan.model_dump())
        revision["plans"] = [*revision.get("plans", []), plan.model_dump()]
        revision["pending_actions"] = pending[1:]
        state["revision"] = revision
        append_trace(state, {"node": "reviser", "event": "plan_applied", "plan": plan.model_dump()})
        mark_success(state)
        if revision["pending_actions"]:
            state["control"]["next"] = "revise"
        else:
            reset_after_structural_change(state, partition_signature(current_sets(state)))
    except Exception as exc:
        mark_failure(state, "reviser", exc)
    return state


def verifier_route(state: Mapping[str, Any]) -> str:
    control = dict(state["control"])
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return "retry"
    if control.get("next") in {"full_acquire", "supplement_acquire", "audit"}:
        return "verifier"
    return "router"


def route_after_router(state: Mapping[str, Any]) -> str:
    control = dict(state["control"])
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return "retry"
    if control.get("next") in {"full_acquire", "supplement_acquire"}:
        return "verify"
    if control.get("next") == "revise":
        return "revise"
    return "end"


def route_after_reviser(state: Mapping[str, Any]) -> str:
    control = dict(state["control"])
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return "retry"
    if control.get("next") == "revise":
        return "revise"
    return "verify"


def build_review_graph(
    *,
    verifier_model: Any,
    router_model: Any,
    reviser_model: Any,
    runtime: Mapping[str, Any] | None = None,
) -> Any:
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
    graph.add_edge(START, "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {"verifier": "verifier", "router": "router", "retry": "verifier", "end": END},
    )
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"verify": "verifier", "revise": "reviser", "retry": "router", "end": END},
    )
    graph.add_conditional_edges(
        "reviser",
        route_after_reviser,
        {"verify": "verifier", "revise": "reviser", "retry": "reviser", "end": END},
    )
    return graph.compile()


def evidence_for_set(state: Mapping[str, Any], target: str) -> list[dict[str, Any]]:
    return [
        report for report in state.get("reports", []) or []
        if report.get("scope") == "partition"
        or target in {str(value) for value in report.get("target_ids", []) or []}
    ]


def metric_refs_for_reports(reports: list[Mapping[str, Any]]) -> list[str]:
    refs = set()
    for report in reports:
        refs.update(str(ref) for ref in report.get("metric_refs", []) or [])
        for observation in report.get("observations", []) or []:
            refs.update(str(ref) for ref in observation.get("metric_refs", []) or [])
    return sorted(refs)


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
    accepted_reports = [
        {
            "set_id": set_id(item),
            "membership": item.get("member_ids", []),
            "revision_lineage": item.get("revision_lineage", []),
            "reports": evidence_for_set(state, set_id(item)),
            "evidence_by_dimension": {
                dimension: [
                    report for report in evidence_for_set(state, set_id(item))
                    if report.get("dimension") == dimension
                ]
                for dimension in EVIDENCE_DIMENSIONS
            },
            "metric_refs": metric_refs_for_reports(evidence_for_set(state, set_id(item))),
            "raw_evidence": round_evidence(state),
            "evidence_history": state.get("evidence_history", []),
            "sections": [
                "CT", "WSI", "RNA/pathway", "WXS", "CNV", "clinical",
                "cross-modal", "confounder", "known-label", "statistics", "medical_interpretation", "limitations",
            ],
        }
        for item in accepted
    ]
    dropped_reports = [
        {
            "set_id": set_id(item),
            "membership": item.get("member_ids", []),
            "drop_reason": item.get("drop_reason", "insufficient_evidence_for_acceptance"),
            "key_evidence": evidence_for_set(state, set_id(item)),
            "decision_trace": control.get("trace", []),
        }
        for item in dropped
    ]
    summary = {
        "stage": "subtype_review",
        "status": final_status,
        "raw_control_status": status,
        "rounds_used": control.get("round", 0),
        "llm_usage": to_jsonable(control.get("llm_usage", {})),
        "partition_sets": to_jsonable(sets),
        "accepted_subtype_sets": to_jsonable(accepted),
        "accepted_subtype_reports": to_jsonable(accepted_reports),
        "dropped_set_registry": to_jsonable(dropped_reports),
        "partition_patient_count": len({member for item in sets for member in item.get("member_ids", [])}),
        "accepted_patient_count": len({member for item in accepted for member in item.get("member_ids", [])}),
        "reports": to_jsonable(state.get("reports", [])),
        "round_evidence": to_jsonable(round_evidence(state)),
        "evidence_history": to_jsonable(state.get("evidence_history", [])),
        "decision_trace": to_jsonable(control.get("trace", [])),
    }
    (root / "final_partition_sets.json").write_text(json.dumps(to_jsonable(sets), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "final_subtype_sets.json").write_text(json.dumps(to_jsonable(accepted), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "accepted_subtype_reports.json").write_text(json.dumps(to_jsonable(accepted_reports), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "dropped_set_reports.json").write_text(json.dumps(to_jsonable(dropped_reports), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "evidence_history.json").write_text(json.dumps(to_jsonable(state.get("evidence_history", [])), ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "decision_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in control.get("trace", []):
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    (root / "final_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
