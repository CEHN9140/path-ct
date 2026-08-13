from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.subtype_review.llm import parse_json_content, parse_router_action
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS,
    ReviserOutput,
    ReviewState,
    RouterAction,
    VerifierOutput,
    active_sets,
    set_id,
)
from agents.subtype_review.tools import execute_capability
from utils.tool_utils import to_jsonable


def partition_artifact_id(signature: str) -> str:
    return "p_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


class ReviewContext(TypedDict, total=False):
    patient_states_by_id: dict[str, dict[str, Any]]
    output_root: str
    config_dir: str
    tool_functions: dict[str, Any]


CAPABILITY_TO_TOOLS = {
    "biological_support": {"tool_mutation_enrichment", "tool_pathway_enrichment"},
    "cross_modal_consistency": {"tool_multimodal_consistency_check"},
    "confounder_exclusion": {"tool_confound_test"},
    "known_label_echo": {"tool_known_label_echo_test"},
    "structural_adequacy": {"tool_structural_adequacy"},
}


def initial_review_state(candidate_sets: list[dict[str, Any]]) -> ReviewState:
    sets = []
    for item in candidate_sets:
        cluster_id = set_id(dict(item))
        if not cluster_id:
            continue
        sets.append(
            {
                "set_id": cluster_id,
                "cluster_id": cluster_id,
                "member_ids": sorted(str(x) for x in item.get("member_ids", [])),
                "status": "active",
                "parent_ids": [],
            }
        )
    set_ids = [set_id(item) for item in sets]
    if len(set_ids) != len(set(set_ids)):
        raise ValueError("Initial candidate sets contain duplicate identifiers")
    members = [member for item in sets for member in item["member_ids"]]
    if len(members) != len(set(members)):
        raise ValueError("Initial candidate sets contain overlapping patients")
    state: ReviewState = {
        "sets": sets,
        "evidence": {"results": []},
        "audit": {"findings": [], "gaps": []},
        "action": None,
        "messages": [],
        "control": {
            "round": 0,
            "failures": 0,
            "status": "reviewing",
            "next": "audit",
            "error": None,
            "blocked_actions": [],
            "visited_partitions": [partition_signature(sets)],
            "trace": [],
            "max_rounds": 12,
            "max_failures": 3,
        },
    }
    return state


def merge_runtime(base: Mapping[str, Any], context: Mapping[str, Any] | None) -> dict[str, Any]:
    runtime = dict(base)
    if hasattr(context, "context"):
        context = context.context
    runtime.update(dict(context or {}))
    return runtime


def current_sets(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(active_sets(list(state.get("sets", []) or [])), key=set_id)


def partition_signature(sets: list[dict[str, Any]]) -> str:
    groups = [sorted(str(x) for x in item.get("member_ids", [])) for item in sets]
    return json.dumps(sorted(groups), ensure_ascii=False)


def append_trace(state: dict[str, Any], row: dict[str, Any]) -> None:
    control = dict(state.get("control", {}) or {})
    trace = list(control.get("trace", []) or [])
    trace.append({"round": int(control.get("round", 0) or 0), **row})
    control["trace"] = trace
    state["control"] = control


def mark_failure(state: dict[str, Any], node: str, exc: Exception) -> None:
    control = dict(state.get("control", {}) or {})
    failures = int(control.get("failures", 0) or 0) + 1
    control["failures"] = failures
    control["error"] = f"{type(exc).__name__}: {exc}"
    if failures >= int(control.get("max_failures", 3) or 3):
        control["status"] = "review_unavailable"
    state["control"] = control
    append_trace(state, {"node": node, "status": control.get("status", "reviewing"), "error": control["error"]})


def mark_success(state: dict[str, Any]) -> None:
    control = dict(state.get("control", {}) or {})
    control["failures"] = 0
    control["error"] = None
    state["control"] = control


def invoke_with_recovery(model: Any, payload: dict[str, Any], state: dict[str, Any], node: str) -> Any:
    try:
        return model.invoke(payload)
    except Exception as exc:
        mark_failure(state, node, exc)
        return None


def all_metric_refs(evidence: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()

    def visit(prefix: str, value: Any) -> None:
        refs.add(prefix)
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{prefix}.{key}", child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{prefix}.{index}", child)

    for result in list(evidence.get("results", []) or []):
        capability = str(result.get("capability", "") or "")
        visit(f"evidence.{capability}", result)
        for child in list(result.get("results", []) or []):
            tool_name = str(child.get("tool_name", "") or "")
            visit(f"tool_results.{tool_name}.metrics", child.get("metrics", {}))
    return refs


def missing_metric_refs(metric_refs: list[str], evidence: Mapping[str, Any]) -> list[str]:
    available = all_metric_refs(evidence)
    return [ref for ref in metric_refs if re.sub(r"\[(\d+)\]", r".\1", ref) not in available]


def metric_refs_for_tools(evidence: Mapping[str, Any], tool_names: set[str]) -> set[str]:
    return {
        ref for ref in all_metric_refs(evidence)
        if ref.startswith("tool_results.")
        and ref.split(".", 2)[1] in tool_names
    }


def metric_refs_from_findings(audit: Mapping[str, Any]) -> set[str]:
    return {
        str(ref)
        for finding in list(audit.get("findings", []) or [])
        for ref in list(finding.get("metric_refs", []) or [])
    }


def require_metric_refs(metric_refs: list[str], context: str) -> None:
    if not metric_refs:
        raise ValueError(f"{context} requires metric_refs")


def serializable_messages(messages: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for message in list(messages or []):
        if isinstance(message, Mapping):
            rows.append({
                "role": str(message.get("role", "tool")),
                "content": to_jsonable(message.get("content", "")),
                "tool_call_id": str(message.get("tool_call_id", "") or ""),
            })
        else:
            rows.append({
                "role": str(getattr(message, "type", "tool")),
                "content": to_jsonable(getattr(message, "content", "")),
                "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
            })
    return rows


def tool_call_name(call: Any) -> str:
    return str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")).strip()


def execute_tool_calls(state: dict[str, Any], ai_message: Any, runtime: Mapping[str, Any]) -> bool:
    calls = list(getattr(ai_message, "tool_calls", []) or [])
    if not calls and isinstance(ai_message, Mapping):
        calls = list(ai_message.get("tool_calls", []) or [])
    calls = [call for call in calls if tool_call_name(call)]
    if not calls:
        return False

    action = dict(state.get("action", {}) or {})
    if action.get("action") != "need_more_evidence":
        raise ValueError("Verifier tool calls require a pending need_more_evidence action")
    dimension = str(action.get("dimension", "") or "")
    names = [tool_call_name(call) for call in calls]
    if any(name != dimension for name in names):
        raise ValueError(f"Verifier tool call does not match requested dimension: {names} != {dimension}")
    if dimension not in EVIDENCE_DIMENSIONS:
        raise ValueError(f"Unknown validation dimension: {dimension}")
    if len(names) != len(set(names)):
        raise ValueError("Verifier requested the same validation capability more than once")

    all_sets = current_sets(state)
    signature = partition_signature(all_sets)
    evidence = dict(state.get("evidence", {}) or {})
    results = list(evidence.get("results", []) or [])
    cached = {
        str(item.get("capability", ""))
        for item in results
        if item.get("partition_signature") == signature
    }
    repeated = [name for name in names if name in cached]
    if repeated:
        raise ValueError(f"Validation evidence is already available for this partition: {repeated}")

    patient_states = dict(runtime.get("patient_states_by_id", {}) or {})
    cluster_state = {
        "cluster_id": partition_artifact_id(signature),
        "member_ids": sorted(
            str(member)
            for item in all_sets
            for member in list(item.get("member_ids", []) or [])
        ),
    }
    functions = dict(runtime.get("tool_functions", {}) or {})
    messages = list(state.get("messages", []) or [])
    messages.append(ai_message)
    executed = []
    for call in calls:
        name = tool_call_name(call)
        payload = execute_capability(
            name,
            functions,
            cluster_state,
            patient_states,
            str(runtime.get("output_root", "")),
            str(runtime.get("config_dir", "")),
            all_sets,
        )
        payload["partition_signature"] = signature
        results.append(payload)
        executed.append(payload)
        try:
            from langchain_core.messages import ToolMessage

            call_id = str(call.get("id", "") if isinstance(call, Mapping) else getattr(call, "id", ""))
            messages.append(ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=call_id or name))
        except Exception:
            messages.append({"role": "tool", "name": name, "content": payload})
    evidence["results"] = results
    state["evidence"] = evidence
    for item in current_sets(state):
        if str(item.get("status", "")) in {"provisionally_accepted", "provisionally_dropped"}:
            item["status"] = "active"
    state["messages"] = messages[-8:]
    control = dict(state.get("control", {}) or {})
    control["blocked_actions"] = []
    control["next"] = "audit"
    state["control"] = control
    append_trace(state, {
        "node": "verifier",
        "event": "tools",
        "partition_signature": signature,
        "capabilities": names,
        "evidence_refs": [
            {
                "capability": item["capability"],
                "status": item["status"],
                "partition_signature": item["partition_signature"],
                "tool_results": [
                    {
                        key: child.get(key)
                        for key in ("tool_name", "status", "metric_refs", "artifact_paths")
                    }
                    for child in item["results"]
                ],
            }
            for item in executed
        ],
    })
    return True


def current_partition_evidence(state: Mapping[str, Any]) -> dict[str, Any]:
    signature = partition_signature(current_sets(state))
    evidence = dict(state.get("evidence", {}) or {})
    return {
        **evidence,
        "results": [
            item for item in list(evidence.get("results", []) or [])
            if item.get("partition_signature") in {None, "", signature}
        ],
    }


def attempted_dimensions(state: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("capability", ""))
        for item in current_partition_evidence(state).get("results", [])
    }


def evidence_inventory(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for item in sorted(list(evidence.get("results", []) or []), key=lambda row: str(row.get("capability", ""))):
        children = list(item.get("results", []) or [])
        inventory.append({
            "dimension": str(item.get("capability", "")),
            "status": str(item.get("status", "")),
            "has_metrics": any(bool(dict(child.get("metrics", {}) or {})) for child in children),
            "metric_refs": sorted({
                str(ref)
                for child in children
                for ref in list(child.get("metric_refs", []) or [])
            }),
            "missing_reasons": sorted({
                str(child.get("missing_reason", ""))
                for child in children
                if str(child.get("missing_reason", ""))
            }),
        })
    return inventory


def verifier_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    action = dict(state.get("action", {}) or {})
    acquire = action.get("action") == "need_more_evidence" and dict(state.get("control", {}) or {}).get("next") == "acquire"
    current_evidence = current_partition_evidence(state)
    payload = {
        "mode": "acquire" if acquire else "audit",
        "sets": current_sets(state),
        "evidence_inventory": evidence_inventory(current_evidence),
        "evidence": current_evidence,
        "tool_messages": serializable_messages(list(state.get("messages", []) or [])),
        "previous_audit": state.get("audit", {}),
        "request": action if acquire else None,
        "round": dict(state.get("control", {}) or {}).get("round", 0),
        "validation_error": dict(state.get("control", {}) or {}).get("error"),
    }
    result = invoke_with_recovery(model, payload, state, "verifier")
    if result is None:
        return state
    if acquire:
        try:
            if not execute_tool_calls(state, result, runtime):
                raise ValueError("Verifier acquisition returned no tool call")
        except Exception as exc:
            mark_failure(state, "verifier", exc)
        else:
            mark_success(state)
        return state
    try:
        audit_payload = result if isinstance(result, Mapping) else parse_json_content(getattr(result, "content", result))
        parsed = VerifierOutput.model_validate(audit_payload)
        validate_verifier_audit(parsed, state)
    except Exception as exc:
        mark_failure(state, "verifier", exc)
        return state
    reactivate_provisional_sets(state, parsed)
    state["audit"] = parsed.model_dump()
    state["messages"] = []
    state["action"] = None
    mark_success(state)
    append_trace(state, {
        "node": "verifier",
        "event": "audit",
        "partition_signature": partition_signature(current_sets(state)),
        "audit": parsed.model_dump(),
        "evidence_inventory": evidence_inventory(current_evidence),
    })
    control = dict(state.get("control", {}) or {})
    control["next"] = "router"
    structural_attempted = "structural_adequacy" in attempted_dimensions(state)
    if has_known_label_conflict(state) and structural_attempted and not any(
        finding.dimension == "structural_adequacy"
        and finding.status == "conflicting"
        for finding in parsed.findings
    ):
        control["status"] = "final_validation_failed"
        control["error"] = "known_label_echo_conflict"
    elif has_known_label_conflict(state) and not structural_attempted and not any(
        gap.dimension == "structural_adequacy" for gap in parsed.gaps
    ):
        control["status"] = "final_validation_failed"
        control["error"] = "known_label_echo_conflict_without_structural_gap"
    elif complete_audit(state):
        control["status"] = "complete"
    elif current_sets(state) and all(
        str(item.get("status", "")) in {"provisionally_accepted", "provisionally_dropped"}
        for item in current_sets(state)
    ) and any(str(item.get("status", "")) == "provisionally_dropped" for item in current_sets(state)):
        control["status"] = "final_validation_failed"
    elif int(control.get("round", 0) or 0) >= int(control.get("max_rounds", 12) or 12):
        control["status"] = "final_validation_failed"
    state["control"] = control
    return state


def validate_verifier_audit(audit: VerifierOutput, state: Mapping[str, Any]) -> None:
    sets = current_sets(state)
    if sets and not audit.findings and not audit.gaps:
        raise ValueError("Verifier returned an empty audit for a nonempty partition")
    known = {set_id(item) for item in sets}
    referenced = {str(target) for row in [*audit.findings, *audit.gaps] for target in row.target_ids}
    unknown = sorted(referenced - known)
    if unknown:
        raise ValueError(f"Verifier referenced inactive or unknown sets: {unknown}")
    evidence = current_partition_evidence(state)
    attempted = attempted_dimensions(state)
    repeated_gaps = sorted({
        str(gap.dimension)
        for gap in audit.gaps
        if str(gap.dimension) in attempted
    })
    if repeated_gaps:
        raise ValueError(f"Verifier gap references an already attempted dimension: {repeated_gaps}")
    supported_targets = supported_structural_split_targets(state)
    reported_structural_conflicts = {
        str(target)
        for finding in audit.findings
        if finding.dimension == "structural_adequacy" and finding.status == "conflicting"
        for target in finding.target_ids
    }
    missing_structural_conflicts = sorted(supported_targets - reported_structural_conflicts)
    if missing_structural_conflicts:
        raise ValueError(
            "Verifier must mark supported structural split targets as conflicting: "
            + ", ".join(missing_structural_conflicts)
        )
    for capability in failed_capabilities(state):
        if not any(
            finding.dimension == capability and finding.status in {"unavailable", "inconclusive"}
            for finding in audit.findings
        ):
            raise ValueError(
                f"Failed evidence capability requires an unavailable or inconclusive finding: {capability}"
            )
    for finding in audit.findings:
        if finding.status != "unavailable":
            require_metric_refs(finding.metric_refs, "Evidence finding")
        missing = missing_metric_refs(finding.metric_refs, evidence)
        if missing:
            raise ValueError(f"Verifier referenced unavailable metrics: {missing}")
        allowed = metric_refs_for_tools(evidence, CAPABILITY_TO_TOOLS[finding.dimension])
        if not set(finding.metric_refs).issubset(allowed):
            raise ValueError(f"Verifier metric_refs are outside {finding.dimension} evidence")


def reactivate_provisional_sets(state: dict[str, Any], audit: VerifierOutput) -> None:
    targets = {
        str(target)
        for finding in audit.findings
        if finding.status == "conflicting"
        for target in finding.target_ids
    }
    global_conflict = any(
        finding.status == "conflicting" and not finding.target_ids
        and finding.dimension != "known_label_echo"
        for finding in audit.findings
    )
    for item in state["sets"]:
        if str(item.get("status", "")) in {"provisionally_accepted", "provisionally_dropped"} and (global_conflict or set_id(item) in targets):
            item["status"] = "active"


def structural_candidates(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    splits: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    signature = partition_signature(current_sets(state))
    for result in list(dict(state.get("evidence", {}) or {}).get("results", []) or []):
        if result.get("capability") != "structural_adequacy" or result.get("partition_signature") not in {None, "", signature}:
            continue
        for child in list(result.get("results", []) or []):
            metrics = dict(child.get("metrics", {}) or {})
            splits.extend(dict(x) for x in list(metrics.get("split_candidates", []) or []))
            merges.extend(dict(x) for x in list(metrics.get("merge_candidates", []) or []))
    return sorted(splits, key=lambda item: str(item.get("plan_id", ""))), sorted(merges, key=lambda item: str(item.get("plan_id", "")))


def supported_structural_split_targets(state: Mapping[str, Any]) -> set[str]:
    targets = set()
    for row in structural_candidates(state)[0]:
        selection = dict(row.get("selection_adjusted_null", {}) or {})
        if float(selection.get("separation_gain_over_null", 0) or 0) <= 0 or float(selection.get("q_value", 1) or 1) > 0.05:
            continue
        confirmed = 0
        for modality in ("ct", "wsi", "rna", "genomic"):
            metrics = dict(row.get("modality_community", {}).get(modality, {}) or {})
            if (
                float(metrics.get("separation_gain_over_null", 0) or 0) > 0
                and float(metrics.get("q_value", 1) or 1) <= 0.05
                and float(metrics.get("minimum_child_separation", 0) or 0) > 0
            ):
                confirmed += 1
        if confirmed >= 2 and str(row.get("source_set_id", "")):
            targets.add(str(row["source_set_id"]))
    return targets


def failed_capabilities(state: Mapping[str, Any]) -> set[str]:
    failed = set()
    for item in current_partition_evidence(state).get("results", []):
        children = list(item.get("results", []) or [])
        statuses = {str(child.get("status", "")) for child in children}
        if children and statuses and statuses.issubset({"failure", "unavailable"}):
            failed.add(str(item.get("capability", "")))
    return failed


def structural_conflict_targets(state: Mapping[str, Any]) -> set[str]:
    return {
        str(target_id)
        for finding in list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
        if finding.get("dimension") == "structural_adequacy" and finding.get("status") == "conflicting"
        for target_id in list(finding.get("target_ids", []) or [])
    }


def complete_audit(state: Mapping[str, Any]) -> bool:
    sets = current_sets(state)
    if not sets or any(str(item.get("status")) != "provisionally_accepted" for item in sets):
        return False
    audit = dict(state.get("audit", {}) or {})
    if list(audit.get("gaps", []) or []):
        return False
    failed = failed_capabilities(state)
    if any(
        finding.get("dimension") in failed
        and finding.get("status") in {"unavailable", "inconclusive"}
        for finding in list(audit.get("findings", []) or [])
    ):
        return False
    conflicts = structural_conflict_targets(state)
    splits, merges = structural_candidates(state)
    if any(str(row.get("source_set_id", "")) in conflicts for row in splits):
        return False
    if any(conflicts.intersection(map(str, row.get("set_ids", []) or [])) for row in merges):
        return False
    return not any(
        finding.get("dimension") == "known_label_echo" and finding.get("status") == "conflicting"
        for finding in list(audit.get("findings", []) or [])
    )


def has_known_label_conflict(state: Mapping[str, Any]) -> bool:
    return any(
        finding.get("dimension") == "known_label_echo"
        and finding.get("status") == "conflicting"
        for finding in list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
    )


def validate_router_action(action: RouterAction, state: Mapping[str, Any]) -> None:
    sets = current_sets(state)
    known = {set_id(item) for item in sets}
    targets = set(action.target_ids)
    if not targets.issubset(known):
        raise ValueError(f"Router referenced inactive or unknown sets: {sorted(targets - known)}")
    if action.action == "need_more_evidence":
        gaps = list(dict(state.get("audit", {}) or {}).get("gaps", []) or [])
        matching = [gap for gap in gaps if gap.get("dimension") == action.dimension]
        if not matching:
            raise ValueError(f"Router requested evidence for a dimension without a current gap: {action.dimension}")
        gap_targets = set(str(item) for gap in matching for item in gap.get("target_ids", []) or [])
        if not gap_targets and targets:
            raise ValueError("Whole-partition evidence requests must use an empty target_ids list")
        if targets and not targets.issubset(gap_targets):
            raise ValueError("Router evidence target is outside the matching Verifier gap")
        signature = partition_signature(current_sets(state))
        completed = {
            str(item.get("capability", ""))
            for item in list(dict(state.get("evidence", {}) or {}).get("results", []) or [])
            if item.get("partition_signature") == signature
        }
        if action.dimension in completed:
            raise ValueError(f"Evidence dimension already attempted for this partition: {action.dimension}")
        return
    if len(action.target_ids) != 1:
        raise ValueError("Scientific actions require exactly one target")
    target = next(item for item in sets if set_id(item) == action.target_ids[0])
    if str(target.get("status", "active")) != "active":
        raise ValueError(f"Router target is already provisionally decided: {action.target_ids[0]}")
    gaps = list(dict(state.get("audit", {}) or {}).get("gaps", []) or [])
    attempted = attempted_dimensions(state)
    blocking_gaps = [
        gap for gap in gaps
        if (
            not gap.get("target_ids")
            or action.target_ids[0] in {str(item) for item in gap.get("target_ids", []) or []}
        )
        and str(gap.get("dimension", "")) not in attempted
    ]
    if blocking_gaps:
        raise ValueError("Scientific action is blocked by a decision-relevant gap")
    findings = list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
    if action.action in {"split", "merge"} and not any(
        finding.get("dimension") == "structural_adequacy"
        and finding.get("status") == "conflicting"
        and action.target_ids[0] in {str(item) for item in finding.get("target_ids", []) or []}
        for finding in findings
    ):
        raise ValueError("Structural action requires a conflicting structural finding")
    blocked = set(dict(state.get("control", {}) or {}).get("blocked_actions", []) or [])
    key = f"{action.action}:{action.target_ids[0]}"
    if key in blocked:
        raise ValueError(f"Action is blocked for this partition: {key}")
    if missing_metric_refs(action.metric_refs, current_partition_evidence(state)):
        raise ValueError("Router referenced unavailable metrics")
    require_metric_refs(action.metric_refs, "Scientific action")
    audit_refs = metric_refs_from_findings(dict(state.get("audit", {}) or {}))
    if not set(action.metric_refs).issubset(audit_refs):
        raise ValueError("Router metric_refs were not reported by the current Verifier audit")


def router_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    control = dict(state.get("control", {}) or {})
    if int(control.get("round", 0) or 0) >= int(control.get("max_rounds", 12) or 12):
        control["status"] = "final_validation_failed"
        control["error"] = "max_rounds_exhausted"
        control["next"] = "end"
        state["control"] = control
        append_trace(state, {"node": "router", "event": "budget_exhausted"})
        return state
    signature = partition_signature(current_sets(state))
    sets = current_sets(state)
    audit = dict(state.get("audit", {}) or {})
    gaps = list(audit.get("gaps", []) or [])
    attempted = {
        str(item.get("capability", ""))
        for item in list(dict(state.get("evidence", {}) or {}).get("results", []) or [])
        if item.get("partition_signature") == signature
    }
    requestable = sorted({
        str(gap.get("dimension", ""))
        for gap in gaps
        if str(gap.get("dimension", "")) and str(gap.get("dimension", "")) not in attempted
    })
    available = [
        {"capability": str(item.get("capability", "")), "status": str(item.get("status", ""))}
        for item in list(dict(state.get("evidence", {}) or {}).get("results", []) or [])
        if item.get("partition_signature") == signature
    ]
    payload = {
        "sets": sets,
        "eligible_action_target_ids": [
            set_id(item) for item in sets if str(item.get("status", "active")) == "active"
        ],
        "provisional_decisions": {
            set_id(item): str(item.get("status", ""))
            for item in sets
            if str(item.get("status", "")) in {"provisionally_accepted", "provisionally_dropped"}
        },
        "validation_error": dict(state.get("control", {}) or {}).get("error"),
        "audit": audit,
        "available_evidence": available,
        "requestable_evidence_dimensions": requestable,
        "attempted_evidence_dimensions": sorted(attempted),
        "blocked_actions": list(dict(state.get("control", {}) or {}).get("blocked_actions", []) or []),
        "control": {
            key: dict(state.get("control", {}) or {}).get(key)
            for key in ("round", "max_rounds", "failures", "error")
        },
    }
    action = None
    for attempt in range(3):
        result = invoke_with_recovery(model, payload, state, "router")
        if result is None:
            return state
        try:
            action = parse_router_action(result)
            validate_router_action(action, state)
            break
        except Exception as exc:
            if attempt == 2:
                mark_failure(state, "router", exc)
                return state
            payload["validation_error"] = f"{type(exc).__name__}: {exc}"
            payload["rejected_action"] = result
    if action is None:
        return state
    control = dict(state.get("control", {}) or {})
    control["round"] = int(control.get("round", 0) or 0) + 1
    control["status"] = "reviewing"
    state["action"] = action.model_dump()
    if action.action == "need_more_evidence":
        control["next"] = "acquire"
    elif action.action in {"split", "merge"}:
        control["next"] = "revise"
    else:
        control["next"] = "router"
        target = action.target_ids[0]
        for item in state["sets"]:
            if set_id(item) == target:
                item["status"] = "provisionally_accepted" if action.action == "accept" else "provisionally_dropped"
        if all(
            str(item.get("status", "")) in {"provisionally_accepted", "provisionally_dropped"}
            for item in current_sets(state)
        ):
            control["status"] = "complete" if complete_audit(state) else "final_validation_failed"
    state["control"] = control
    mark_success(state)
    append_trace(state, {
        "node": "router",
        "partition_signature": signature,
        "action": action.model_dump(),
    })
    return state


def merge_options(target: str, merges: list[dict[str, Any]], sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_map = {
        frozenset(map(str, row.get("set_ids", []))): row
        for row in merges
        if len(list(row.get("set_ids", []) or [])) == 2
    }
    ids = sorted({set_id(item) for item in sets})
    options = []
    for size in range(2, len(ids) + 1):
        for group in combinations(ids, size):
            if target not in group or not all(frozenset(pair) in pair_map for pair in combinations(group, 2)):
                continue
            options.append({
                "plan_id": "merge:" + "+".join(group),
                "set_ids": list(group),
                "pairwise_evidence": [pair_map[frozenset(pair)] for pair in combinations(group, 2)],
            })
    return sorted(options, key=lambda item: str(item["plan_id"]))


def candidate_partition_signature(state: Mapping[str, Any], action: str, plan: Mapping[str, Any]) -> str:
    groups = {set_id(item): sorted(str(x) for x in item.get("member_ids", [])) for item in current_sets(state)}
    if action == "split":
        source_id = str(plan.get("source_set_id", ""))
        groups.pop(source_id)
        for index, members in enumerate(list(plan.get("groups", []) or []), start=1):
            groups[f"{source_id}:candidate:{index}"] = sorted(str(x) for x in members)
    else:
        ids = [str(x) for x in list(plan.get("set_ids", []) or [])]
        merged = sorted({member for item in ids for member in groups.pop(item)})
        groups["merge:candidate"] = merged
    return partition_signature([{"set_id": key, "member_ids": members} for key, members in groups.items()])


def apply_split(state: dict[str, Any], plan: Mapping[str, Any]) -> None:
    source_id = str(plan.get("source_set_id", ""))
    source = next(item for item in current_sets(state) if set_id(item) == source_id)
    source_members = set(map(str, source.get("member_ids", [])))
    raw_groups = [list(group) for group in list(plan.get("groups", []) or [])]
    raw_members = [str(member) for group in raw_groups for member in group]
    if len(raw_members) != len(set(raw_members)):
        raise ValueError("Split plan contains duplicate members")
    groups = [set(map(str, group)) for group in raw_groups]
    if len(groups) < 2 or set().union(*groups) != source_members or sum(map(len, groups)) != len(source_members):
        raise ValueError("Split plan does not partition the source set exactly")
    source["status"] = "retired"
    for index, group in enumerate(groups, start=1):
        state["sets"].append({
            "set_id": f"{source_id}_S{index}",
            "cluster_id": f"{source_id}_S{index}",
            "member_ids": sorted(group),
            "status": "active",
            "parent_ids": [source_id],
        })


def apply_merge(state: dict[str, Any], plan: Mapping[str, Any]) -> None:
    ids = [str(x) for x in list(plan.get("set_ids", []) or [])]
    if len(ids) < 2:
        raise ValueError("Merge plan must contain at least two sets")
    selected = [item for item in current_sets(state) if set_id(item) in ids]
    if len(selected) != len(ids):
        raise ValueError("Merge plan contains an inactive set")
    members = sorted({str(x) for item in selected for x in item.get("member_ids", [])})
    if len(members) != sum(len(item.get("member_ids", [])) for item in selected):
        raise ValueError("Merge plan contains overlapping members")
    for item in selected:
        item["status"] = "retired"
    merged_id = "_M_".join(ids)
    state["sets"].append({
        "set_id": merged_id,
        "cluster_id": merged_id,
        "member_ids": members,
        "status": "active",
        "parent_ids": ids,
    })


def reset_after_structural_change(state: dict[str, Any], signature: str) -> None:
    for item in current_sets(state):
        item["status"] = "active"
    state["audit"] = {"findings": [], "gaps": []}
    state["action"] = None
    state["messages"] = []
    control = dict(state.get("control", {}) or {})
    control["visited_partitions"] = list(control.get("visited_partitions", []) or []) + [signature]
    control["blocked_actions"] = []
    control["next"] = "audit"
    state["control"] = control


def reviser_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    action = dict(state.get("action", {}) or {})
    target = str((action.get("target_ids") or [""])[0])
    partition_before = partition_signature(current_sets(state))
    splits, merges = structural_candidates(state)
    candidates = [row for row in splits if str(row.get("source_set_id", "")) == target] if action.get("action") == "split" else merge_options(target, merges, current_sets(state))
    visited = set(dict(state.get("control", {}) or {}).get("visited_partitions", []) or [])
    candidates = [row for row in candidates if candidate_partition_signature(state, str(action.get("action", "")), row) not in visited]
    if not candidates:
        blocked = list(dict(state.get("control", {}) or {}).get("blocked_actions", []) or [])
        blocked.append(f"{action.get('action')}:{target}")
        state["control"]["blocked_actions"] = sorted(set(blocked))
        state["action"] = None
        state["control"]["next"] = "router"
        append_trace(state, {
            "node": "reviser",
            "partition_before": partition_before,
            "partition_after": partition_before,
            "candidates": [],
            "selection": {"plan_id": None, "reason": "no_valid_plan", "metric_refs": []},
        })
        return state
    payload = {
        "action": action,
        "candidates": candidates,
        "audit": state.get("audit", {}),
        "validation_error": dict(state.get("control", {}) or {}).get("error"),
    }
    result = invoke_with_recovery(model, payload, state, "reviser")
    if result is None:
        return state
    try:
        parsed = ReviserOutput.model_validate(result)
    except Exception as exc:
        mark_failure(state, "reviser", exc)
        return state
    if not parsed.plan_id:
        blocked = list(dict(state.get("control", {}) or {}).get("blocked_actions", []) or [])
        blocked.append(f"{action.get('action')}:{target}")
        state["control"]["blocked_actions"] = sorted(set(blocked))
        state["action"] = None
        state["control"]["next"] = "router"
        mark_success(state)
        append_trace(state, {
            "node": "reviser",
            "partition_before": partition_before,
            "partition_after": partition_before,
            "candidates": to_jsonable(candidates),
            "selection": parsed.model_dump(),
        })
        return state
    plan = next((row for row in candidates if str(row.get("plan_id", "")) == parsed.plan_id), None)
    if plan is None:
        mark_failure(state, "reviser", ValueError("Reviser selected an unknown plan"))
        return state
    try:
        require_metric_refs(parsed.metric_refs, "Reviser plan")
    except ValueError as exc:
        mark_failure(state, "reviser", exc)
        return state
    if missing_metric_refs(parsed.metric_refs, current_partition_evidence(state)):
        mark_failure(state, "reviser", ValueError("Reviser referenced unavailable metrics"))
        return state
    structural_refs = metric_refs_for_tools(
        current_partition_evidence(state), {"tool_structural_adequacy"}
    )
    required_prefix = "tool_results.tool_structural_adequacy.metrics."
    if not set(parsed.metric_refs).issubset(structural_refs) or not any(
        ref.startswith(required_prefix + ("split_candidates" if action.get("action") == "split" else "merge_candidates"))
        for ref in parsed.metric_refs
    ):
        mark_failure(state, "reviser", ValueError("Reviser plan metric_refs are not structural plan evidence"))
        return state
    next_signature = candidate_partition_signature(state, str(action.get("action", "")), plan)
    if next_signature in set(dict(state.get("control", {}) or {}).get("visited_partitions", []) or []):
        raise ValueError("Structural plan recreates a visited partition")
    if action.get("action") == "split":
        apply_split(state, plan)
    else:
        set_ids = tuple(sorted(str(x) for x in list(plan.get("set_ids", []) or [])))
        for pair in combinations(set_ids, 2):
            if not any(frozenset(map(str, row.get("set_ids", []))) == frozenset(pair) for row in merges):
                raise ValueError("Merge plan lacks complete pairwise evidence")
        apply_merge(state, plan)
    reset_after_structural_change(state, next_signature)
    mark_success(state)
    append_trace(state, {
        "node": "reviser",
        "partition_before": partition_before,
        "partition_after": partition_signature(current_sets(state)),
        "candidates": to_jsonable(candidates),
        "selection": parsed.model_dump(),
    })
    return state


def verifier_route(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "complete", "final_validation_failed"}:
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    return "router" if control.get("next") == "router" else "audit"


def route_after_router(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "complete", "final_validation_failed"}:
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    action = dict(state.get("action", {}) or {})
    if action.get("action") in {"split", "merge"}:
        return "revise"
    if action.get("action") in {"accept", "drop"}:
        return "router"
    return "verify"


def route_after_reviser(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") == "review_unavailable":
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    return "router" if control.get("next") == "router" else "verify"


def build_review_graph(*, verifier_model: Any, router_model: Any, reviser_model: Any, tool_functions: Mapping[str, Any], runtime: Mapping[str, Any] | None = None) -> Any:
    base_runtime = dict(runtime or {})

    def verifier(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        merged = merge_runtime(base_runtime, runtime)
        merged.setdefault("tool_functions", dict(tool_functions))
        return verifier_node(state, merged, verifier_model)

    def router(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        return router_node(state, merge_runtime(base_runtime, runtime), router_model)

    def reviser(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        return reviser_node(state, merge_runtime(base_runtime, runtime), reviser_model)

    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    graph.add_node("verifier", verifier)
    graph.add_node("router", router)
    graph.add_node("reviser", reviser)
    graph.add_edge(START, "verifier")
    graph.add_conditional_edges("verifier", verifier_route, {"router": "router", "audit": "verifier", "retry": "verifier", "end": END})
    graph.add_conditional_edges("router", route_after_router, {"router": "router", "verify": "verifier", "revise": "reviser", "retry": "router", "end": END})
    graph.add_conditional_edges("reviser", route_after_reviser, {"router": "router", "verify": "verifier", "retry": "reviser", "end": END})
    return graph.compile()


def save_review_outputs(state: Mapping[str, Any], output_root: str) -> dict[str, Any]:
    root = Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    control = dict(state.get("control", {}) or {})
    sets = current_sets(state)
    accepted_sets = [item for item in sets if item.get("status") == "provisionally_accepted"]
    summary = {
        "stage": "subtype_review",
        "status": control.get("status", "review_unavailable"),
        "rounds_used": control.get("round", 0),
        "partition_sets": to_jsonable(sets),
        "accepted_subtype_sets": to_jsonable(accepted_sets),
        "partition_patient_count": len({member for item in sets for member in item.get("member_ids", [])}),
        "accepted_patient_count": len({member for item in accepted_sets for member in item.get("member_ids", [])}),
        "audit": to_jsonable(state.get("audit", {})),
        "decision_trace": to_jsonable(control.get("trace", [])),
    }
    (root / "final_partition_sets.json").write_text(json.dumps(to_jsonable(sets), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "final_subtype_sets.json").write_text(json.dumps(to_jsonable(accepted_sets), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "evidence_audit.json").write_text(json.dumps(to_jsonable(state.get("audit", {})), ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "decision_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in control.get("trace", []):
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    (root / "final_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
