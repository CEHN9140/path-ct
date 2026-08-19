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
    "biological_support": {"tool_mutation_enrichment", "tool_pathway_enrichment", "tool_cnv_characterization"},
    "cross_modal_consistency": {"tool_multimodal_consistency_check"},
    "confounder_exclusion": {"tool_confound_test"},
    "known_label_echo": {"tool_known_label_echo_test"},
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
        "structure_proposals": {"split_proposals": [], "merge_proposals": []},
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
            "max_rounds": 20,
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


def subject_signature(
    scope: str,
    sets: list[dict[str, Any]],
    target_ids: list[str] | None = None,
    proposal: Mapping[str, Any] | None = None,
) -> str:
    proposal = dict(proposal or {})
    if scope == "partition":
        value = {"scope": scope, "partition": partition_signature(sets)}
    elif scope == "set_identity":
        selected = [item for item in sets if set_id(item) in set(target_ids or [])]
        value = {
            "scope": scope,
            "members": sorted(str(member) for item in selected for member in item.get("member_ids", [])),
        }
    elif scope == "split_proposal":
        value = {
            "scope": scope,
            "parent_members": sorted(str(member) for member in proposal.get("parent_members", []) or []),
            "groups": sorted(sorted(str(member) for member in group) for group in proposal.get("groups", []) or []),
        }
    elif scope == "merge_proposal":
        value = {
            "scope": scope,
            "groups": sorted(sorted(str(member) for member in group) for group in proposal.get("memberships", []) or []),
        }
    else:
        raise ValueError(f"Unknown evidence scope: {scope}")
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evidence_request_key(item: Mapping[str, Any]) -> str:
    return "|".join(
        str(item.get(key, "") or "")
        for key in ("capability", "scope", "subject_signature", "proposal_id")
    )


def proposal_by_id(state: Mapping[str, Any], proposal_id: str) -> dict[str, Any]:
    for key in ("split_proposals", "merge_proposals"):
        for proposal in list(dict(state.get("structure_proposals", {}) or {}).get(key, []) or []):
            if str(proposal.get("proposal_id") or proposal.get("plan_id") or "") == proposal_id:
                return dict(proposal)
    return {}


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
    if "structure_proposals" in evidence:
        visit("structure_proposals", evidence.get("structure_proposals", {}))
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
    scope = str(action.get("scope", "") or "")
    proposal_id = str(action.get("proposal_id", "") or "")
    names = [tool_call_name(call) for call in calls]
    if any(name != dimension for name in names):
        raise ValueError(f"Verifier tool call does not match requested dimension: {names} != {dimension}")
    if dimension not in EVIDENCE_DIMENSIONS:
        raise ValueError(f"Unknown validation dimension: {dimension}")
    if len(names) != len(set(names)):
        raise ValueError("Verifier requested the same validation capability more than once")

    all_sets = current_sets(state)
    proposal = proposal_by_id(state, proposal_id) if proposal_id else {}
    partition = partition_signature(all_sets)
    signature = subject_signature(scope, all_sets, list(action.get("target_ids", []) or []), proposal)
    evidence = dict(state.get("evidence", {}) or {})
    results = list(evidence.get("results", []) or [])
    cached = {
        evidence_request_key(item)
        for item in results
    }
    requested_keys = [
        "|".join((name, scope, signature, proposal_id)) for name in names
    ]
    repeated = [key for key in requested_keys if key in cached]
    if repeated:
        raise ValueError(f"Validation evidence is already available for this partition: {repeated}")

    patient_states = dict(runtime.get("patient_states_by_id", {}) or {})
    cluster_state = {
        "cluster_id": partition_artifact_id(partition_signature(all_sets)),
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
            scope=scope,
            target_ids=list(action.get("target_ids", []) or []),
            proposal=proposal,
        )
        payload["partition_signature"] = partition
        payload["scope"] = scope
        payload["target_ids"] = list(action.get("target_ids", []) or [])
        payload["proposal_id"] = proposal_id or None
        payload["subject_signature"] = signature
        payload["subject_signatures"] = {
            set_id(item): subject_signature("set_identity", [item], [set_id(item)])
            for item in all_sets
        } if scope == "set_identity" else {}
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
        "partition_signature": partition,
        "capabilities": names,
        "evidence_refs": [
            {
                "capability": item["capability"],
                "status": item["status"],
                "scope": item["scope"],
                "subject_signature": item["subject_signature"],
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
    sets = current_sets(state)
    signatures = {
        partition_signature(sets),
        *{
            subject_signature("set_identity", [item], [set_id(item)])
            for item in sets
        },
    }
    for key in ("split_proposals", "merge_proposals"):
        for proposal in list(dict(state.get("structure_proposals", {}) or {}).get(key, []) or []):
            scope = "split_proposal" if key == "split_proposals" else "merge_proposal"
            targets = [str(proposal.get("source_set_id", ""))] if scope == "split_proposal" else list(proposal.get("set_ids", []) or [])
            signatures.add(subject_signature(scope, sets, targets, proposal))
    evidence = dict(state.get("evidence", {}) or {})
    return {
        **evidence,
        "structure_proposals": dict(state.get("structure_proposals", {}) or {}),
        "results": [
            item for item in list(evidence.get("results", []) or [])
            if item.get("subject_signature") in signatures
            or signatures.intersection(set(dict(item.get("subject_signatures", {}) or {}).values()))
        ],
    }


def attempted_dimensions(state: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("capability", ""))
        for item in current_partition_evidence(state).get("results", [])
    }


def attempted_evidence_keys(state: Mapping[str, Any]) -> set[str]:
    return {evidence_request_key(item) for item in current_partition_evidence(state).get("results", [])}


def evidence_inventory(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for item in sorted(list(evidence.get("results", []) or []), key=lambda row: str(row.get("capability", ""))):
        children = list(item.get("results", []) or [])
        inventory.append({
            "dimension": str(item.get("capability", "")),
            "scope": str(item.get("scope", "")),
            "proposal_id": item.get("proposal_id"),
            "subject_signature": str(item.get("subject_signature", "")),
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


def proposal_generator_node(state: dict[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    signature = partition_signature(current_sets(state))
    control = dict(state.get("control", {}) or {})
    if control.get("proposal_partition_signature") == signature and dict(state.get("structure_proposals", {}) or {}).get("split_proposals") is not None:
        return state
    sets = current_sets(state)
    patient_states = dict(runtime.get("patient_states_by_id", {}) or {})
    if not runtime.get("output_root") or not patient_states:
        control["proposal_partition_signature"] = signature
        state["control"] = control
        return state
    from tools.structure_proposal_generator import generate_structure_proposals
    cluster_state = {
        "cluster_id": partition_artifact_id(signature),
        "member_ids": sorted({member for item in sets for member in item.get("member_ids", [])}),
    }
    try:
        result = generate_structure_proposals(
            cluster_state,
            patient_states,
            str(runtime.get("output_root", "")),
            config_dir=str(runtime.get("config_dir", "")),
            all_cluster_states=sets,
        )
        metrics = dict(dict(result.get("results", {}) or {}).get("metrics", {}) or {})
        state["structure_proposals"] = metrics
        control["proposal_partition_signature"] = signature
        state["control"] = control
        append_trace(state, {
            "node": "structure_proposals",
            "partition_signature": signature,
            "split_count": len(metrics.get("split_proposals", []) or []),
            "merge_count": len(metrics.get("merge_proposals", []) or []),
        })
    except Exception as exc:
        mark_failure(state, "structure_proposals", exc)
        state["structure_proposals"] = {"split_proposals": [], "merge_proposals": []}
    return state


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
    if complete_audit(state):
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
    proposal_ids = {
        str(item.get("proposal_id") or item.get("plan_id") or "")
        for key in ("split_proposals", "merge_proposals")
        for item in list(dict(state.get("structure_proposals", {}) or {}).get(key, []) or [])
    }
    for row in [*audit.findings, *audit.gaps]:
        if row.scope in {"split_proposal", "merge_proposal"}:
            if not row.proposal_id or row.proposal_id not in proposal_ids:
                raise ValueError(f"Evidence references an unknown proposal: {row.proposal_id}")
            proposal = proposal_by_id(state, row.proposal_id)
        else:
            if row.proposal_id:
                raise ValueError(f"{row.scope} evidence cannot set proposal_id")
            proposal = {}
        expected_signature = subject_signature(
            row.scope,
            sets,
            row.target_ids,
            proposal,
        )
        if row.subject_signature != expected_signature:
            raise ValueError(f"Evidence subject_signature does not match its scope: {row.scope}")
    evidence = current_partition_evidence(state)
    repeated_gaps = sorted({
        evidence_request_key({
            "capability": gap.dimension,
            "scope": gap.scope,
            "subject_signature": gap.subject_signature,
            "proposal_id": gap.proposal_id,
        })
        for gap in audit.gaps
        if evidence_request_key({
            "capability": gap.dimension,
            "scope": gap.scope,
            "subject_signature": gap.subject_signature,
            "proposal_id": gap.proposal_id,
        }) in attempted_evidence_keys(state)
    })
    if repeated_gaps:
        raise ValueError(f"Verifier gap references an already attempted evidence request: {repeated_gaps}")
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
    proposals = dict(state.get("structure_proposals", {}) or {})
    return (
        sorted([dict(item) for item in proposals.get("split_proposals", []) or []], key=lambda item: str(item.get("proposal_id", item.get("plan_id", "")))),
        sorted([dict(item) for item in proposals.get("merge_proposals", []) or []], key=lambda item: str(item.get("proposal_id", item.get("plan_id", "")))),
    )


def failed_capabilities(state: Mapping[str, Any]) -> set[str]:
    failed = set()
    for item in current_partition_evidence(state).get("results", []):
        children = list(item.get("results", []) or [])
        statuses = {str(child.get("status", "")) for child in children}
        if children and statuses and statuses.issubset({"failure", "unavailable"}):
            failed.add(str(item.get("capability", "")))
    return failed


def complete_audit(state: Mapping[str, Any]) -> bool:
    sets = current_sets(state)
    if not sets or any(
        str(item.get("status")) not in {"provisionally_accepted", "provisionally_dropped"}
        for item in sets
    ):
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
    return not any(
        finding.get("dimension") in {"known_label_echo", "confounder_exclusion"}
        and finding.get("status") == "conflicting"
        for finding in list(audit.get("findings", []) or [])
    )


def validate_router_action(action: RouterAction, state: Mapping[str, Any]) -> None:
    sets = current_sets(state)
    known = {set_id(item) for item in sets}
    targets = set(action.target_ids)
    if not targets.issubset(known):
        raise ValueError(f"Router referenced inactive or unknown sets: {sorted(targets - known)}")
    if action.action == "need_more_evidence":
        proposal = proposal_by_id(state, str(action.proposal_id or "")) if action.proposal_id else {}
        signature = subject_signature(action.scope, sets, action.target_ids, proposal)
        gaps = list(dict(state.get("audit", {}) or {}).get("gaps", []) or [])
        matching = [
            gap for gap in gaps
            if gap.get("dimension") == action.dimension
            and gap.get("scope") == action.scope
            and str(gap.get("proposal_id", "") or "") == str(action.proposal_id or "")
            and str(gap.get("subject_signature", "")) == signature
        ]
        if not matching:
            raise ValueError("Router requested evidence without a matching scoped gap")
        gap_targets = set(str(item) for gap in matching for item in gap.get("target_ids", []) or [])
        if not gap_targets and targets:
            raise ValueError("Whole-partition evidence requests must use an empty target_ids list")
        if targets and not targets.issubset(gap_targets):
            raise ValueError("Router evidence target is outside the matching Verifier gap")
        key = "|".join((action.dimension, action.scope, signature, str(action.proposal_id or "")))
        if key in attempted_evidence_keys(state):
            raise ValueError("Evidence request already attempted for this subject")
        return
    if len(action.target_ids) != 1:
        raise ValueError("Scientific actions require exactly one target")
    target = next(item for item in sets if set_id(item) == action.target_ids[0])
    if str(target.get("status", "active")) != "active":
        raise ValueError(f"Router target is already provisionally decided: {action.target_ids[0]}")
    gaps = list(dict(state.get("audit", {}) or {}).get("gaps", []) or [])
    blocking_gaps = [
        gap for gap in gaps
        if (
            not gap.get("target_ids")
            or action.target_ids[0] in {str(item) for item in gap.get("target_ids", []) or []}
        )
        and evidence_request_key({
            "capability": gap.get("dimension"),
            "scope": gap.get("scope"),
            "subject_signature": gap.get("subject_signature"),
            "proposal_id": gap.get("proposal_id"),
        }) not in attempted_evidence_keys(state)
    ]
    if blocking_gaps:
        raise ValueError("Scientific action is blocked by a decision-relevant gap")
    findings = list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
    if action.action in {"split", "merge"}:
        candidates = structural_candidates(state)[0 if action.action == "split" else 1]
        matching = [
            row for row in candidates
            if (
                (action.action == "split" and str(row.get("source_set_id", "")) == action.target_ids[0])
                or (action.action == "merge" and action.target_ids[0] in {str(item) for item in row.get("set_ids", [])})
            )
            and bool(row.get("eligible_for_review", True))
        ]
        if not matching:
            raise ValueError("Structural action requires an eligible generated proposal")
        proposal_ids = {str(row.get("proposal_id") or row.get("plan_id") or "") for row in matching}
        if not any(
            str(finding.get("proposal_id", "")) in proposal_ids
            and finding.get("scope") == ("split_proposal" if action.action == "split" else "merge_proposal")
            and finding.get("dimension") == "cross_modal_consistency"
            and finding.get("status") == "supporting"
            for finding in findings
        ):
            raise ValueError("Structural action lacks proposal-specific cross-modal support")
        if action.action == "split" and not any(
            str(finding.get("proposal_id", "")) in proposal_ids
            and finding.get("scope") == "split_proposal"
            and finding.get("dimension") == "biological_support"
            and finding.get("status") == "supporting"
            for finding in findings
        ):
            raise ValueError("Split requires proposal-specific molecular or biological support")
        proposal_findings = [
            finding for finding in findings
            if str(finding.get("proposal_id", "")) in proposal_ids
            and finding.get("scope") == ("split_proposal" if action.action == "split" else "merge_proposal")
        ]
        if any(
            finding.get("dimension") in {"confounder_exclusion", "known_label_echo"}
            and finding.get("status") == "conflicting"
            for finding in proposal_findings
        ):
            raise ValueError("Structural action is vetoed by confounder or known-label evidence")
        if action.action == "merge":
            if any(
                finding.get("dimension") == "biological_support"
                and finding.get("status") == "conflicting"
                for finding in proposal_findings
            ):
                raise ValueError("Merge is vetoed by proposal-specific biological distinction")
            strong_boundary_modalities = set()
            for result in current_partition_evidence(state).get("results", []):
                if result.get("proposal_id") not in proposal_ids:
                    continue
                for child in result.get("results", []) or []:
                    strong_boundary_modalities.update(
                        dict(dict(child.get("metrics", {}) or {}).get("decision_metrics", {}) or {})
                        .get("merge_strong_boundary_modalities", [])
                    )
                    strong_boundary_modalities.update(
                        dict(child.get("metrics", {}) or {}).get("merge_strong_boundary_modalities", []) or []
                    )
            if len(strong_boundary_modalities) >= 2:
                raise ValueError("Merge is blocked by strong boundaries in at least two modalities")
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
    if action.action == "accept":
        identity = [
            finding for finding in findings
            if finding.get("dimension") == "cross_modal_consistency"
            and finding.get("scope") == "set_identity"
            and finding.get("status") == "supporting"
            and action.target_ids[0] in {str(item) for item in finding.get("target_ids", [])}
        ]
        if not identity:
            raise ValueError("Accept requires supporting cross-modal set identity evidence")
        modality_count = 0
        for result in current_partition_evidence(state).get("results", []):
            for child in result.get("results", []) or []:
                if child.get("tool_name") != "tool_multimodal_consistency_check":
                    continue
                decision = dict(child.get("metrics", {}) or {})
                by_set = dict(decision.get("identity_supporting_modalities_by_set", {}) or {})
                modality_count = max(
                    modality_count,
                    len(list(by_set.get(action.target_ids[0], []) or [])),
                )
                modality_count = max(
                    modality_count,
                    len(list(decision.get("identity_supporting_modalities", []) or [])),
                )
        if modality_count < 2:
            raise ValueError("Accept requires at least two supporting original modalities")
    if action.action == "drop":
        positive = any(
            finding.get("dimension") in {"confounder_exclusion", "known_label_echo"}
            and finding.get("status") == "conflicting"
            and action.target_ids[0] in {str(item) for item in finding.get("target_ids", [])}
            for finding in findings
        )
        if not positive:
            raise ValueError("Drop requires positive confounder or known-label invalidating evidence")


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
    attempted = attempted_evidence_keys(state)
    requestable = [
        {
            "dimension": str(gap.get("dimension", "")),
            "scope": str(gap.get("scope", "")),
            "proposal_id": gap.get("proposal_id"),
            "target_ids": list(gap.get("target_ids", []) or []),
        }
        for gap in gaps
        if evidence_request_key({
            "capability": gap.get("dimension"),
            "scope": gap.get("scope"),
            "subject_signature": gap.get("subject_signature"),
            "proposal_id": gap.get("proposal_id"),
        }) not in attempted
    ]
    available = [
        {"capability": str(item.get("capability", "")), "status": str(item.get("status", ""))}
        for item in list(dict(state.get("evidence", {}) or {}).get("results", []) or [])
        if item in current_partition_evidence(state).get("results", [])
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
        "structure_proposals": dict(state.get("structure_proposals", {}) or {}),
        "available_evidence": available,
        "requestable_evidence": requestable,
        "attempted_evidence_keys": sorted(attempted),
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
                control["status"] = "unresolved"
                control["error"] = f"{type(exc).__name__}: {exc}"
                control["next"] = "end"
                state["control"] = control
                append_trace(state, {"node": "router", "event": "unresolved", "error": control["error"]})
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
    state["structure_proposals"] = {"split_proposals": [], "merge_proposals": []}
    state["action"] = None
    state["messages"] = []
    control = dict(state.get("control", {}) or {})
    control["visited_partitions"] = list(control.get("visited_partitions", []) or []) + [signature]
    control.pop("proposal_partition_signature", None)
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
    structural_refs = {
        ref for ref in all_metric_refs(current_partition_evidence(state))
        if ref.startswith("structure_proposals")
    }
    required_prefix = "structure_proposals."
    if not set(parsed.metric_refs).issubset(structural_refs) or not any(
        ref.startswith(required_prefix + ("split_proposals" if action.get("action") == "split" else "merge_proposals"))
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
    if control.get("status") in {"review_unavailable", "complete", "final_validation_failed", "unresolved"}:
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    return "router" if control.get("next") == "router" else "audit"


def route_after_router(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "complete", "final_validation_failed", "unresolved"}:
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
    if control.get("status") in {"review_unavailable", "unresolved"}:
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    if control.get("next") == "router":
        return "router"
    if control.get("proposal_partition_signature") != partition_signature(current_sets(state)):
        return "proposals"
    return "verify"


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
    graph.add_node("proposals", lambda state, runtime: proposal_generator_node(state, merge_runtime(base_runtime, runtime)))
    graph.add_node("verifier", verifier)
    graph.add_node("router", router)
    graph.add_node("reviser", reviser)
    graph.add_edge(START, "proposals")
    graph.add_edge("proposals", "verifier")
    graph.add_conditional_edges("verifier", verifier_route, {"router": "router", "audit": "verifier", "retry": "verifier", "end": END})
    graph.add_conditional_edges("router", route_after_router, {"router": "router", "verify": "verifier", "revise": "reviser", "retry": "router", "end": END})
    graph.add_conditional_edges("reviser", route_after_reviser, {"router": "router", "verify": "verifier", "proposals": "proposals", "retry": "reviser", "end": END})
    return graph.compile()


def save_review_outputs(state: Mapping[str, Any], output_root: str) -> dict[str, Any]:
    root = Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    control = dict(state.get("control", {}) or {})
    sets = current_sets(state)
    accepted_sets = [item for item in sets if item.get("status") == "provisionally_accepted"]
    dropped_sets = [item for item in sets if item.get("status") == "provisionally_dropped"]
    active_unresolved = [item for item in sets if item.get("status") == "active"]
    raw_status = str(control.get("status", "review_unavailable"))
    if raw_status == "complete" and dropped_sets:
        final_status = "review_complete_with_dropped_sets"
    elif raw_status == "complete":
        final_status = "review_complete_all_accepted"
    elif raw_status in {"final_validation_failed", "unresolved"} and active_unresolved:
        final_status = "review_complete_with_unresolved_sets"
    elif raw_status == "review_unavailable":
        final_status = "review_unavailable"
    else:
        final_status = "review_failed_runtime" if control.get("error") else raw_status
    summary = {
        "stage": "subtype_review",
        "status": final_status,
        "raw_control_status": raw_status,
        "rounds_used": control.get("round", 0),
        "partition_sets": to_jsonable(sets),
        "accepted_subtype_sets": to_jsonable(accepted_sets),
        "partition_patient_count": len({member for item in sets for member in item.get("member_ids", [])}),
        "accepted_patient_count": len({member for item in accepted_sets for member in item.get("member_ids", [])}),
        "dropped_set_registry": to_jsonable(dropped_sets),
        "structure_proposals": to_jsonable(state.get("structure_proposals", {})),
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
