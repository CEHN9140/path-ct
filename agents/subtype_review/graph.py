from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.subtype_review.llm import LLMCallBudgetExceeded, parse_json_content, parse_router_action
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS,
    ReviserOutput,
    ReviewState,
    RouterAction,
    VerifierOutput,
    active_sets,
    set_id,
)
from agents.subtype_review.tools import VALIDATION_FUNCTIONS
from utils.tool_utils import to_jsonable


def partition_artifact_id(signature: str) -> str:
    return "p_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


class ReviewContext(TypedDict, total=False):
    patient_states_by_id: dict[str, dict[str, Any]]
    output_root: str
    data_root: str
    review_output_root: str
    config_dir: str


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
        "revision": None,
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
            "max_rounds": 60,
            "max_failures": 3,
            "policy": {
                "accept_min_supporting_modalities": 2,
                "split_min_supporting_modalities": 2,
                "merge_min_supporting_modalities": 2,
                "split_require_molecular_or_biology": True,
                "min_split_size": 10,
            },
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
    dimension: str,
    scope: str,
    sets: list[dict[str, Any]],
    target_ids: list[str] | None = None,
    proposal: Mapping[str, Any] | None = None,
) -> str:
    proposal = dict(proposal or {})
    if scope == "partition":
        subject = {"partition": partition_signature(sets)}
    elif scope == "set_identity":
        subject = {"partition": partition_signature(sets)}
    elif scope == "split_proposal":
        groups = sorted(
            sorted(str(member) for member in group)
            for group in proposal.get("groups", []) or []
        )
        parent_members = proposal.get("parent_members", []) or [
            member for group in groups for member in group
        ]
        subject = {
            "parent_members": sorted(str(member) for member in parent_members),
            "groups": groups,
        }
        if dimension == "known_label_echo":
            subject["partition"] = partition_signature(sets)
    elif scope == "merge_proposal":
        subject = {
            "groups": sorted(sorted(str(member) for member in group) for group in proposal.get("memberships", []) or []),
        }
        if dimension == "known_label_echo":
            subject["partition"] = partition_signature(sets)
    else:
        raise ValueError(f"Unknown evidence scope: {scope}")
    value = {"dimension": dimension, "scope": scope, "subject": subject}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evidence_request_key(item: Mapping[str, Any]) -> str:
    return "|".join(
        str(item.get(key, "") or "")
        for key in ("dimension", "scope", "subject_signature", "proposal_id")
    )


def proposal_by_id(state: Mapping[str, Any], proposal_id: str) -> dict[str, Any]:
    for proposal in list(dict(state.get("revision", {}) or {}).get("candidates", []) or []):
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
    except LLMCallBudgetExceeded as exc:
        control = dict(state.get("control", {}) or {})
        control["status"] = "unresolved_due_to_budget"
        control["error"] = str(exc)
        control["next"] = "end"
        state["control"] = control
        append_trace(state, {"node": node, "event": "llm_budget_exhausted", "error": str(exc)})
        return None
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
        dimension = str(result.get("dimension", "") or "")
        visit(f"evidence.{dimension}", result)
        for child in list(result.get("results", []) or []):
            tool_name = str(child.get("tool_name", "") or "")
            visit(f"tool_results.{tool_name}.metrics", child.get("metrics", {}))
    return refs


def missing_metric_refs(metric_refs: list[str], evidence: Mapping[str, Any]) -> list[str]:
    available = all_metric_refs(evidence)
    return [ref for ref in metric_refs if re.sub(r"\[(\d+)\]", r".\1", ref) not in available]


def matching_evidence_results(
    evidence: Mapping[str, Any],
    dimension: str,
    scope: str,
    signature: str,
    proposal_id: str | None,
) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(evidence.get("results", []) or [])
        if item.get("dimension") == dimension
        and item.get("scope") == scope
        and str(item.get("proposal_id", "") or "") == str(proposal_id or "")
        and signature == str(item.get("subject_signature", ""))
    ]


def metric_refs_for_action(action: RouterAction, state: Mapping[str, Any]) -> list[str]:
    if action.action == "need_more_evidence":
        return []
    targets = set(action.target_ids)
    if action.action in {"accept", "drop"}:
        scopes = {"set_identity", "partition"}
    else:
        scopes = {f"{action.action}_proposal"}
    revision = dict(state.get("revision", {}) or {})
    proposal_ids = {
        str(item.get("proposal_id") or item.get("plan_id") or "")
        for item in revision.get("candidates", []) or []
    }
    return sorted({
        str(ref)
        for finding in list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
        if finding.get("scope") in scopes
        and (action.action in {"accept", "drop"} or str(finding.get("proposal_id", "")) in proposal_ids)
        and (
            not finding.get("target_ids")
            or targets.intersection(str(item) for item in finding.get("target_ids", []) or [])
        )
        for ref in list(finding.get("metric_refs", []) or [])
    })


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
        raise ValueError("Verifier requested the same validation dimension more than once")

    all_sets = current_sets(state)
    proposal = proposal_by_id(state, proposal_id) if proposal_id else {}
    partition = partition_signature(all_sets)
    signature = subject_signature(dimension, scope, all_sets, list(action.get("target_ids", []) or []), proposal)
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
    messages = list(state.get("messages", []) or [])
    messages.append(ai_message)
    executed = []
    for call in calls:
        name = tool_call_name(call)
        payload = VALIDATION_FUNCTIONS[name](
            cluster_state,
            patient_states,
            str(runtime.get("data_root", runtime.get("output_root", ""))),
            str(runtime.get("config_dir", "")),
            all_sets,
            scope=scope,
            target_ids=list(action.get("target_ids", []) or []),
            proposal=proposal,
            artifact_root=str(runtime.get("review_output_root", runtime.get("output_root", ""))),
        )
        payload["partition_signature"] = partition
        payload["scope"] = scope
        payload["target_ids"] = list(action.get("target_ids", []) or [])
        payload["proposal_id"] = proposal_id or None
        payload["subject_signature"] = signature
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
    control["next"] = "audit"
    state["control"] = control
    append_trace(state, {
        "node": "verifier",
        "event": "tools",
        "partition_signature": partition,
        "dimensions": names,
        "evidence_refs": [
            {
                "dimension": item["dimension"],
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
    evidence = dict(state.get("evidence", {}) or {})
    current_results = []
    for item in list(evidence.get("results", []) or []):
        dimension = str(item.get("dimension", ""))
        signatures = {
            subject_signature(dimension, "partition", sets),
            *{
                subject_signature(dimension, "set_identity", sets, [set_id(row)])
                for row in sets
            },
        }
        revision = dict(state.get("revision", {}) or {})
        scope = f"{revision.get('action')}_proposal"
        for proposal in revision.get("candidates", []) or []:
            signatures.add(subject_signature(
                dimension,
                scope,
                sets,
                list(revision.get("target_ids", []) or []),
                proposal,
            ))
        if item.get("subject_signature") in signatures:
            current_results.append(item)
    return {
        **evidence,
        "results": current_results,
    }


def attempted_evidence_keys(state: Mapping[str, Any]) -> set[str]:
    keys = set()
    for item in current_partition_evidence(state).get("results", []):
        keys.add(evidence_request_key(item))
    return keys


def evidence_inventory(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for item in sorted(list(evidence.get("results", []) or []), key=lambda row: str(row.get("dimension", ""))):
        children = list(item.get("results", []) or [])
        inventory.append({
            "dimension": str(item.get("dimension", "")),
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
        audit_payload = dict(audit_payload)
        sets = current_sets(state)
        for key in ("findings", "gaps"):
            rows = []
            for raw_row in list(audit_payload.get(key, []) or []):
                row = dict(raw_row)
                proposal = proposal_by_id(state, str(row.get("proposal_id", "") or ""))
                row["subject_signature"] = subject_signature(
                    str(row.get("dimension", "")),
                    str(row.get("scope", "")),
                    sets,
                    list(row.get("target_ids", []) or []),
                    proposal,
                )
                rows.append(row)
            audit_payload[key] = rows
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
    elif int(control.get("round", 0) or 0) >= int(control.get("max_rounds", 12) or 12):
        control["status"] = "unresolved_due_to_budget"
    state["control"] = control
    return state


def validate_verifier_audit(audit: VerifierOutput, state: Mapping[str, Any]) -> None:
    sets = current_sets(state)
    if sets and not audit.findings and not audit.gaps and not required_evidence_requests(state):
        raise ValueError("Verifier returned an empty audit for a nonempty partition")
    known = {set_id(item) for item in sets}
    referenced = {str(target) for row in [*audit.findings, *audit.gaps] for target in row.target_ids}
    unknown = sorted(referenced - known)
    if unknown:
        raise ValueError(f"Verifier referenced inactive or unknown sets: {unknown}")
    proposal_ids = {
        str(item.get("proposal_id") or item.get("plan_id") or "")
        for item in list(dict(state.get("revision", {}) or {}).get("candidates", []) or [])
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
            row.dimension,
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
            "dimension": gap.dimension,
            "scope": gap.scope,
            "subject_signature": gap.subject_signature,
            "proposal_id": gap.proposal_id,
        })
        for gap in audit.gaps
        if evidence_request_key({
            "dimension": gap.dimension,
            "scope": gap.scope,
            "subject_signature": gap.subject_signature,
            "proposal_id": gap.proposal_id,
        }) in attempted_evidence_keys(state)
    })
    if repeated_gaps:
        raise ValueError(f"Verifier gap references an already attempted evidence request: {repeated_gaps}")
    for request in required_evidence_requests(state, include_available=True):
        exact = matching_evidence_results(
            evidence,
            request["dimension"],
            request["scope"],
            request["subject_signature"],
            request["proposal_id"],
        )
        if not exact:
            continue
        corresponding = [
            finding
            for finding in audit.findings
            if finding.dimension == request["dimension"]
            and finding.scope == request["scope"]
            and finding.subject_signature == request["subject_signature"]
            and str(finding.proposal_id or "") == str(request["proposal_id"] or "")
        ]
        covered_targets = {
            target for finding in corresponding for target in finding.target_ids
        }
        if not corresponding or not set(request["target_ids"]).issubset(covered_targets):
            raise ValueError("Acquired mandatory evidence has no corresponding Finding")
        fully_failed = all(
            (
                result.get("results")
                and {
                    str(child.get("status", ""))
                    for child in result.get("results", []) or []
                }.issubset({"failure", "unavailable"})
            )
            or (
                not result.get("results")
                and str(result.get("status", "")) == "failure"
            )
            for result in exact
        )
        if fully_failed and not any(
            finding.status in {"unavailable", "inconclusive"}
            for finding in corresponding
        ):
            raise ValueError(
                "Failed mandatory evidence requires an unavailable or inconclusive Finding"
            )
    for finding in audit.findings:
        exact = matching_evidence_results(
            evidence,
            finding.dimension,
            finding.scope,
            finding.subject_signature,
            finding.proposal_id,
        )
        if not exact:
            raise ValueError("Evidence finding has no exact evidence instance")
        if finding.status != "unavailable":
            require_metric_refs(finding.metric_refs, "Evidence finding")
        exact_evidence = {"results": exact}
        missing = missing_metric_refs(finding.metric_refs, exact_evidence)
        if missing:
            raise ValueError(f"Verifier referenced metrics outside exact evidence: {missing}")
        allowed = {
            re.sub(r"\[(\d+)\]", r".\1", str(ref))
            for result in exact
            for child in result.get("results", []) or []
            for ref in child.get("metric_refs", []) or []
        }
        if not all(
            any(
                normalized == root or normalized.startswith(root + ".")
                for root in allowed
            )
            for normalized in (
                re.sub(r"\[(\d+)\]", r".\1", ref)
                for ref in finding.metric_refs
            )
        ):
            raise ValueError("Verifier metric_refs are outside exact evidence")


def blocks_set_acceptance(finding: Mapping[str, Any], target: str) -> bool:
    if finding.get("status") != "conflicting":
        return False
    if finding.get("scope") == "partition":
        return finding.get("dimension") == "known_label_echo"
    return (
        finding.get("scope") == "set_identity"
        and finding.get("dimension") in {
            "biological_support",
            "cross_modal_consistency",
            "confounder_exclusion",
        }
        and target in {str(value) for value in finding.get("target_ids", []) or []}
    )


def invalidates_set(finding: Mapping[str, Any], target: str) -> bool:
    return (
        finding.get("dimension") == "confounder_exclusion"
        and finding.get("scope") == "set_identity"
        and finding.get("status") == "conflicting"
        and target in {str(value) for value in finding.get("target_ids", []) or []}
    )


def reactivate_provisional_sets(state: dict[str, Any], audit: VerifierOutput) -> None:
    findings = [finding.model_dump() for finding in audit.findings]
    for item in state["sets"]:
        status = str(item.get("status", ""))
        target = set_id(item)
        if status == "provisionally_accepted" and any(
            blocks_set_acceptance(finding, target) for finding in findings
        ):
            item["status"] = "active"
        elif status == "provisionally_dropped" and not any(
            invalidates_set(finding, target) for finding in findings
        ):
            item["status"] = "active"


def revision_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(item) for item in dict(state.get("revision", {}) or {}).get("candidates", []) or []],
        key=lambda item: str(item.get("proposal_id") or item.get("plan_id") or ""),
    )


def required_evidence_requests(
    state: Mapping[str, Any], include_available: bool = False
) -> list[dict[str, Any]]:
    sets = current_sets(state)
    set_ids = [set_id(item) for item in sets]
    evidence = current_partition_evidence(state)
    policy = dict(dict(state.get("control", {}) or {}).get("policy", {}) or {})
    requests = []

    def add(dimension, scope, targets, proposal=None):
        proposal = dict(proposal or {})
        proposal_id = str(proposal.get("proposal_id") or proposal.get("plan_id") or "")
        signature = subject_signature(dimension, scope, sets, targets, proposal)
        if include_available or not matching_evidence_results(
            evidence, dimension, scope, signature, proposal_id or None
        ):
            requests.append({
                "dimension": dimension,
                "scope": scope,
                "target_ids": list(targets),
                "proposal_id": proposal_id or None,
                "subject_signature": signature,
            })

    add("cross_modal_consistency", "set_identity", set_ids)
    add("confounder_exclusion", "set_identity", set_ids)
    add("known_label_echo", "partition", [])

    revision = dict(state.get("revision", {}) or {})
    action = str(revision.get("action", ""))
    if action in {"split", "merge"}:
        scope = f"{action}_proposal"
        minimum = int(policy.get(f"{action}_min_supporting_modalities", 2) or 2)
        for proposal in revision_candidates(state):
            if not bool(proposal.get("eligible_for_review", True)):
                continue
            targets = (
                [str(proposal.get("source_set_id", ""))]
                if action == "split"
                else [str(value) for value in proposal.get("set_ids", []) or []]
            )
            signature = subject_signature(
                "cross_modal_consistency", scope, sets, targets, proposal
            )
            cross_modal = matching_evidence_results(
                evidence,
                "cross_modal_consistency",
                scope,
                signature,
                str(proposal.get("proposal_id") or proposal.get("plan_id") or ""),
            )
            if not cross_modal:
                add("cross_modal_consistency", scope, targets, proposal)
                continue
            modalities = {
                str(modality)
                for result in cross_modal
                for child in result.get("results", []) or []
                for modality in dict(child.get("metrics", {}) or {}).get(
                    f"{action}_supporting_modalities", []
                )
            }
            if len(modalities) < minimum:
                continue
            add("confounder_exclusion", scope, targets, proposal)
            add("known_label_echo", scope, targets, proposal)
            if action == "merge" or not modalities.intersection({"rna", "genomic"}):
                add("biological_support", scope, targets, proposal)
    unique = {evidence_request_key({"dimension": row["dimension"], **row}): row for row in requests}
    return list(unique.values())


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
    findings = list(audit.get("findings", []) or [])
    for item in sets:
        target = set_id(item)
        accept_blocker = any(blocks_set_acceptance(finding, target) for finding in findings)
        drop_evidence = any(invalidates_set(finding, target) for finding in findings)
        if item.get("status") == "provisionally_accepted" and accept_blocker:
            return False
        if item.get("status") == "provisionally_dropped" and not drop_evidence:
            return False
    return True


def supported_revision_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    revision = dict(state.get("revision", {}) or {})
    action = str(revision.get("action", ""))
    targets = sorted(str(item) for item in revision.get("target_ids", []) or [])
    if action not in {"split", "merge"}:
        return []
    scope = "split_proposal" if action == "split" else "merge_proposal"
    candidates = revision_candidates(state)
    findings = list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
    if any(invalidates_set(finding, target) for finding in findings for target in targets):
        return []
    evidence = current_partition_evidence(state)
    missing_required = required_evidence_requests(state)
    policy = dict(dict(state.get("control", {}) or {}).get("policy", {}) or {})
    minimum = int(policy.get(f"{action}_min_supporting_modalities", 2) or 2)
    supported = []
    for proposal in candidates:
        proposal_id = str(proposal.get("proposal_id") or proposal.get("plan_id") or "")
        if not bool(proposal.get("eligible_for_review", True)):
            continue
        if action == "split" and str(proposal.get("source_set_id", "")) != targets[0]:
            continue
        if action == "merge" and sorted(str(item) for item in proposal.get("set_ids", []) or []) != targets:
            continue
        proposal_findings = [
            finding for finding in findings
            if finding.get("scope") == scope
            and str(finding.get("proposal_id", "")) == proposal_id
        ]
        if not any(
            finding.get("dimension") == "cross_modal_consistency"
            and finding.get("status") == "supporting"
            for finding in proposal_findings
        ):
            continue
        decision = {}
        for result in evidence.get("results", []):
            if (
                result.get("dimension") == "cross_modal_consistency"
                and result.get("scope") == scope
                and str(result.get("proposal_id", "")) == proposal_id
            ):
                for child in result.get("results", []) or []:
                    if child.get("tool_name") == "multimodal_consistency_check":
                        decision = dict(child.get("metrics", {}) or {})
        supporting_modalities = set(decision.get(f"{action}_supporting_modalities", []) or [])
        if len(supporting_modalities) < minimum:
            continue
        if any(
            request.get("scope") == scope
            and str(request.get("proposal_id", "") or "") == proposal_id
            for request in missing_required
        ):
            continue
        required_dimensions = {
            "cross_modal_consistency",
            "confounder_exclusion",
            "known_label_echo",
        }
        if action == "merge" or not supporting_modalities.intersection({"rna", "genomic"}):
            required_dimensions.add("biological_support")
        if any(
            not any(
                finding.get("dimension") == dimension
                and finding.get("status") != "unavailable"
                for finding in proposal_findings
            )
            for dimension in required_dimensions
        ):
            continue
        if any(
            finding.get("dimension") in {"confounder_exclusion", "known_label_echo"}
            and finding.get("status") == "conflicting"
            for finding in proposal_findings
        ):
            continue
        if action == "split":
            biology_support = any(
                finding.get("dimension") == "biological_support"
                and finding.get("status") == "supporting"
                for finding in proposal_findings
            )
            if policy.get("split_require_molecular_or_biology", True) and not (
                biology_support or supporting_modalities.intersection({"rna", "genomic"})
            ):
                continue
        else:
            if len(set(decision.get("merge_strong_boundary_modalities", []) or [])) >= minimum:
                continue
            if any(
                finding.get("dimension") == "biological_support"
                and finding.get("status") == "conflicting"
                for finding in proposal_findings
            ):
                continue
        supported.append(dict(proposal))
    return sorted(
        supported,
        key=lambda row: str(row.get("proposal_id") or row.get("plan_id") or ""),
    )


def initial_revision_intents(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    if state.get("revision") is not None:
        return []
    sets = [item for item in current_sets(state) if str(item.get("status", "active")) == "active"]
    if any(request["scope"] in {"set_identity", "partition"} for request in required_evidence_requests(state)):
        return []
    findings = list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
    policy = dict(dict(state.get("control", {}) or {}).get("policy", {}) or {})
    blocked = set(dict(state.get("control", {}) or {}).get("blocked_actions", []) or [])
    min_split_size = int(policy.get("min_split_size", 10) or 10)
    invalid = {
        set_id(item) for item in sets
        if any(invalidates_set(row, set_id(item)) for row in findings)
    }
    intents = [
        {"action": "split", "target_ids": [set_id(item)]}
        for item in sets
        if len(item.get("member_ids", []) or []) >= 2 * min_split_size
        and set_id(item) not in invalid
        and f"split:{set_id(item)}" not in blocked
    ]
    for index, left in enumerate(sets):
        for right in sets[index + 1:]:
            targets = sorted([set_id(left), set_id(right)])
            if (
                not invalid.intersection(targets)
                and f"merge:{'+'.join(targets)}" not in blocked
            ):
                intents.append({"action": "merge", "target_ids": targets})
    return intents


def abandon_revision(state: dict[str, Any], intent: str) -> None:
    control = dict(state.get("control", {}) or {})
    control["blocked_actions"] = sorted(set([*control.get("blocked_actions", []), intent]))
    control["next"] = "router"
    state["control"] = control
    state["revision"] = None
    state["action"] = None
    audit = dict(state.get("audit", {}) or {})
    state["audit"] = {
        "findings": [row for row in audit.get("findings", []) or [] if row.get("scope") in {"set_identity", "partition"}],
        "gaps": [row for row in audit.get("gaps", []) or [] if row.get("scope") in {"set_identity", "partition"}],
    }


def validate_router_action(action: RouterAction, state: Mapping[str, Any]) -> None:
    sets = current_sets(state)
    known = {set_id(item) for item in sets}
    targets = set(action.target_ids)
    if not targets.issubset(known):
        raise ValueError(f"Router referenced inactive or unknown sets: {sorted(targets - known)}")
    if action.action == "need_more_evidence":
        proposal = proposal_by_id(state, str(action.proposal_id or "")) if action.proposal_id else {}
        signature = subject_signature(action.dimension, action.scope, sets, action.target_ids, proposal)
        gaps = [
            *list(dict(state.get("audit", {}) or {}).get("gaps", []) or []),
            *required_evidence_requests(state),
        ]
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
    selected = [item for item in sets if set_id(item) in targets]
    if any(str(item.get("status", "active")) != "active" for item in selected):
        raise ValueError("Router target is already provisionally decided")
    if any(request["scope"] in {"set_identity", "partition"} for request in required_evidence_requests(state)):
        raise ValueError("Scientific action is blocked by mandatory missing evidence")
    blocking_gaps = [
        gap for gap in list(dict(state.get("audit", {}) or {}).get("gaps", []) or [])
        if (
            not gap.get("target_ids")
            or targets.intersection(str(item) for item in gap.get("target_ids", []) or [])
        )
        and evidence_request_key({
            "dimension": gap.get("dimension"),
            "scope": gap.get("scope"),
            "subject_signature": gap.get("subject_signature"),
            "proposal_id": gap.get("proposal_id"),
        }) not in attempted_evidence_keys(state)
    ]
    if blocking_gaps:
        raise ValueError("Scientific action is blocked by a decision-relevant gap")
    findings = list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
    blocked = set(dict(state.get("control", {}) or {}).get("blocked_actions", []) or [])
    key = f"{action.action}:{'+'.join(action.target_ids)}"
    if key in blocked:
        raise ValueError(f"Action is blocked for this partition: {key}")
    if action.action in {"split", "merge"}:
        revision = dict(state.get("revision", {}) or {})
        if revision:
            if revision.get("status") != "ready_for_revision" or not supported_revision_candidates(state):
                raise ValueError("Structural action has no fully supported exact candidate")
            if action.action != revision.get("action") or action.target_ids != sorted(revision.get("target_ids", []) or []):
                raise ValueError("Structural action does not match the active revision intent")
        elif (action.action, action.target_ids) not in [
            (intent["action"], intent["target_ids"]) for intent in initial_revision_intents(state)
        ]:
            raise ValueError("Structural action lacks a deterministic revision motive")
        return
    if state.get("revision") is not None:
        raise ValueError("Accept and Drop are blocked while a revision is active")
    target = action.target_ids[0]
    if action.action == "accept":
        if any(target in intent["target_ids"] for intent in initial_revision_intents(state)):
            raise ValueError("Accept is blocked by pending structural review")
        identity = [
            finding for finding in findings
            if finding.get("dimension") == "cross_modal_consistency"
            and finding.get("scope") == "set_identity"
            and finding.get("status") == "supporting"
            and target in {str(item) for item in finding.get("target_ids", [])}
        ]
        if not identity:
            raise ValueError("Accept requires supporting cross-modal set identity evidence")
        modality_count = 0
        signature = subject_signature(
            "cross_modal_consistency", "set_identity", sets, [target]
        )
        for result in current_partition_evidence(state).get("results", []):
            if (
                result.get("dimension") != "cross_modal_consistency"
                or result.get("scope") != "set_identity"
                or result.get("proposal_id")
                or signature != str(result.get("subject_signature", ""))
            ):
                continue
            for child in result.get("results", []) or []:
                if child.get("tool_name") != "multimodal_consistency_check":
                    continue
                decision = dict(child.get("metrics", {}) or {})
                by_set = dict(decision.get("identity_supporting_modalities_by_set", {}) or {})
                modality_count = max(
                    modality_count,
                    len(list(by_set.get(target, []) or [])),
                )
        policy = dict(dict(state.get("control", {}) or {}).get("policy", {}) or {})
        minimum = int(policy.get("accept_min_supporting_modalities", 2) or 2)
        if modality_count < minimum:
            raise ValueError("Accept requires at least two supporting original modalities")
        confound = [
            finding
            for finding in findings
            if finding.get("dimension") == "confounder_exclusion"
            and finding.get("scope") == "set_identity"
            and target
            in {str(item) for item in finding.get("target_ids", []) or []}
        ]
        known_label = [
            finding
            for finding in findings
            if finding.get("dimension") == "known_label_echo"
            and finding.get("scope") == "partition"
            and (
                not finding.get("target_ids")
                or target
                in {str(item) for item in finding.get("target_ids", []) or []}
            )
        ]
        if not any(row.get("status") != "unavailable" for row in confound) or not any(
            row.get("status") != "unavailable" for row in known_label
        ):
            raise ValueError("Accept requires available confounder and known-label evidence")
        if any(blocks_set_acceptance(finding, target) for finding in findings):
            raise ValueError("Accept is vetoed by conflicting set-identity or partition evidence")
    if action.action == "drop":
        positive = any(invalidates_set(finding, target) for finding in findings)
        if not positive:
            raise ValueError("Drop requires positive confounder set-identity invalidating evidence")


def router_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    control = dict(state.get("control", {}) or {})
    if int(control.get("round", 0) or 0) >= int(control.get("max_rounds", 12) or 12):
        control["status"] = "unresolved_due_to_budget"
        control["error"] = None
        control["next"] = "end"
        state["control"] = control
        append_trace(state, {"node": "router", "event": "budget_exhausted"})
        return state
    signature = partition_signature(current_sets(state))
    sets = current_sets(state)
    audit = dict(state.get("audit", {}) or {})
    gaps = list(audit.get("gaps", []) or [])
    attempted = attempted_evidence_keys(state)
    missing = {
        evidence_request_key({
            "dimension": row.get("dimension"),
            "scope": row.get("scope"),
            "subject_signature": row.get("subject_signature"),
            "proposal_id": row.get("proposal_id"),
        }): row
        for row in [*gaps, *required_evidence_requests(state)]
    }
    requestable = [
        {
            "dimension": str(row.get("dimension", "")),
            "scope": str(row.get("scope", "")),
            "proposal_id": row.get("proposal_id"),
            "target_ids": list(row.get("target_ids", []) or []),
        }
        for key, row in missing.items()
        if key not in attempted
    ]
    revision = dict(state.get("revision", {}) or {})
    if revision.get("status") == "evidence_collection" and not any(
        row["scope"] in {"split_proposal", "merge_proposal"} for row in requestable
    ):
        if supported_revision_candidates(state):
            revision["status"] = "ready_for_revision"
            state["revision"] = revision
        else:
            targets = sorted(str(item) for item in revision.get("target_ids", []) or [])
            key = f"{revision.get('action')}:{'+'.join(targets)}"
            abandon_revision(state, key)
            audit = dict(state["audit"])
            requestable = [
                row for row in requestable
                if row["scope"] in {"set_identity", "partition"}
            ]
            append_trace(state, {"node": "router", "event": "revision_unsupported", "intent": key})
            control = dict(state["control"])
    available = [
        {"dimension": str(item.get("dimension", "")), "status": str(item.get("status", ""))}
        for item in list(dict(state.get("evidence", {}) or {}).get("results", []) or [])
        if item in current_partition_evidence(state).get("results", [])
    ]
    revision = dict(state.get("revision", {}) or {})
    if revision.get("status") == "ready_for_revision":
        legal_action_candidates = [RouterAction(
            action=revision["action"],
            target_ids=revision["target_ids"],
        ).model_dump()]
        requestable = []
    elif revision:
        legal_action_candidates = []
    else:
        legal_action_candidates = []
        for item in sets:
            if str(item.get("status", "active")) != "active":
                continue
            for action_name in ("accept", "drop"):
                candidate = RouterAction(action=action_name, target_ids=[set_id(item)])
                try:
                    validate_router_action(candidate, state)
                except ValueError:
                    continue
                legal_action_candidates.append(candidate.model_dump())
        legal_action_candidates.extend(
            RouterAction(**intent).model_dump() for intent in initial_revision_intents(state)
        )
    if not requestable and not legal_action_candidates:
        control["status"] = "unresolved"
        control["error"] = None
        control["next"] = "end"
        state["control"] = control
        append_trace(state, {"node": "router", "event": "scientific_evidence_exhausted"})
        return state
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
        "revision": to_jsonable(state.get("revision")),
        "available_evidence": available,
        "requestable_evidence": requestable,
        "legal_action_candidates": legal_action_candidates,
        "attempted_evidence_keys": sorted(attempted),
        "blocked_actions": list(dict(state.get("control", {}) or {}).get("blocked_actions", []) or []),
        "control": {
            key: dict(state.get("control", {}) or {}).get(key)
            for key in ("round", "max_rounds", "failures", "error")
        },
    }
    legal_action_keys = {
        (
            str(row["action"]),
            tuple(str(item) for item in row["target_ids"]),
        )
        for row in legal_action_candidates
    }
    action = None
    for attempt in range(3):
        result = invoke_with_recovery(model, payload, state, "router")
        if result is None:
            return state
        parsed_action = None
        try:
            parsed_action = parse_router_action(result)
            if parsed_action.action != "need_more_evidence" and (
                parsed_action.action,
                tuple(parsed_action.target_ids),
            ) not in legal_action_keys:
                raise ValueError("Router scientific action is outside legal_action_candidates")
            validate_router_action(parsed_action, state)
            action = parsed_action
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rejected_action = (
                parsed_action.model_dump() if parsed_action is not None else to_jsonable(result)
            )
            append_trace(state, {
                "node": "router",
                "event": "contract_rejection",
                "attempt": attempt + 1,
                "rejected_action": rejected_action,
                "error": error,
            })
            if attempt == 2:
                control = dict(state.get("control", {}) or {})
                control["status"] = "review_failed_runtime"
                control["error"] = error
                control["next"] = "end"
                state["control"] = control
                append_trace(state, {"node": "router", "event": "contract_failure", "error": control["error"]})
                return state
            payload["validation_error"] = error
            payload["rejected_action"] = rejected_action
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
            control["status"] = "complete" if complete_audit(state) else "unresolved"
    state["control"] = control
    mark_success(state)
    append_trace(state, {
        "node": "router",
        "partition_signature": signature,
        "action": action.model_dump(),
        "metric_refs": metric_refs_for_action(action, state),
    })
    return state


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
    state["revision"] = None
    state["action"] = None
    state["messages"] = []
    control = dict(state.get("control", {}) or {})
    control["visited_partitions"] = list(control.get("visited_partitions", []) or []) + [signature]
    control["blocked_actions"] = []
    control["next"] = "audit"
    state["control"] = control


def reviser_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    action = dict(state.get("action", {}) or {})
    targets = sorted(str(item) for item in action.get("target_ids", []) or [])
    partition_before = partition_signature(current_sets(state))
    revision = dict(state.get("revision", {}) or {})
    if not revision:
        from tools.structural_adequacy import generate_revision_candidates

        cluster_state = {
            "cluster_id": partition_artifact_id(partition_before),
            "member_ids": sorted({member for item in current_sets(state) for member in item.get("member_ids", [])}),
        }
        candidates = generate_revision_candidates(
            action["action"],
            targets,
            cluster_state,
            dict(runtime.get("patient_states_by_id", {}) or {}),
            str(runtime.get("output_root", "")),
            config_dir=str(runtime.get("config_dir", "")),
            all_cluster_states=current_sets(state),
        )
        key = f"{action['action']}:{'+'.join(targets)}"
        if not candidates:
            abandon_revision(state, key)
            append_trace(state, {"node": "reviser", "event": "no_candidates", "intent": key})
            return state
        state["revision"] = {
            "action": action["action"],
            "target_ids": targets,
            "partition_signature": partition_before,
            "status": "evidence_collection",
            "candidates": candidates,
        }
        state["action"] = None
        state["control"]["next"] = "router"
        append_trace(state, {"node": "reviser", "event": "candidates_generated", "intent": key, "candidate_ids": [item["proposal_id"] for item in candidates]})
        return state
    if revision.get("status") != "ready_for_revision" or action.get("action") != revision.get("action") or targets != sorted(revision.get("target_ids", []) or []):
        raise ValueError("Reviser received an action outside the ready revision context")
    candidates = supported_revision_candidates(state)
    visited = set(dict(state.get("control", {}) or {}).get("visited_partitions", []) or [])
    candidates = [row for row in candidates if candidate_partition_signature(state, str(action.get("action", "")), row) not in visited]
    key = f"{action['action']}:{'+'.join(targets)}"
    if not candidates:
        abandon_revision(state, key)
        append_trace(state, {
            "node": "reviser",
            "partition_before": partition_before,
            "partition_after": partition_before,
            "candidates": [],
            "selection": {"plan_id": None, "reason": "no_valid_plan"},
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
        abandon_revision(state, key)
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
    if str(plan.get("partition_signature", "")) != partition_before:
        raise ValueError("Revision candidate belongs to a stale partition")
    proposal_id = str(plan.get("proposal_id") or plan.get("plan_id") or "")
    audit_refs = sorted({
        str(ref)
        for finding in list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
        if str(finding.get("proposal_id", "")) == proposal_id
        for ref in finding.get("metric_refs", []) or []
    })
    candidate_index = next(
        index
        for index, item in enumerate(revision_candidates(state))
        if str(item.get("proposal_id") or item.get("plan_id") or "") == proposal_id
    )
    structural_refs = [f"revision.candidates.{candidate_index}.{key}" for key in sorted(plan)]
    next_signature = candidate_partition_signature(state, str(action.get("action", "")), plan)
    if next_signature in set(dict(state.get("control", {}) or {}).get("visited_partitions", []) or []):
        raise ValueError("Structural plan recreates a visited partition")
    if action.get("action") == "split":
        apply_split(state, plan)
    else:
        apply_merge(state, plan)
    reset_after_structural_change(state, next_signature)
    mark_success(state)
    append_trace(state, {
        "node": "reviser",
        "partition_before": partition_before,
        "partition_after": partition_signature(current_sets(state)),
        "candidates": to_jsonable(candidates),
        "selection": parsed.model_dump(),
        "structural_metric_refs": structural_refs,
        "audit_metric_refs": audit_refs,
    })
    return state


def verifier_route(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "review_failed_runtime", "complete", "unresolved", "unresolved_due_to_budget"}:
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    return "router" if control.get("next") == "router" else "audit"


def route_after_router(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") in {"review_unavailable", "review_failed_runtime", "complete", "unresolved", "unresolved_due_to_budget"}:
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
    if control.get("status") in {"review_unavailable", "review_failed_runtime", "unresolved", "unresolved_due_to_budget"}:
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    if control.get("next") == "router":
        return "router"
    return "verify"


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
    graph.add_edge(START, "verifier")
    graph.add_conditional_edges("verifier", verifier_route, {"router": "router", "audit": "verifier", "retry": "verifier", "end": END})
    graph.add_conditional_edges("router", route_after_router, {"router": "router", "verify": "verifier", "revise": "reviser", "retry": "router", "end": END})
    graph.add_conditional_edges("reviser", route_after_reviser, {"router": "router", "verify": "verifier", "retry": "reviser", "end": END})
    return graph.compile()


def save_review_outputs(state: Mapping[str, Any], output_root: str, *, direct: bool = False) -> dict[str, Any]:
    root = Path(output_root) if direct else Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    control = dict(state.get("control", {}) or {})
    sets = current_sets(state)
    accepted_sets = [item for item in sets if item.get("status") == "provisionally_accepted"]
    dropped_sets = [item for item in sets if item.get("status") == "provisionally_dropped"]
    raw_status = str(control.get("status", "review_unavailable"))
    if raw_status == "complete" and dropped_sets:
        final_status = "review_complete_with_dropped_sets"
    elif raw_status == "complete":
        final_status = "review_complete_all_accepted"
    elif raw_status in {"unresolved", "unresolved_due_to_budget"}:
        final_status = "review_complete_with_unresolved_sets"
    elif raw_status == "review_unavailable":
        final_status = "review_unavailable"
    elif raw_status == "review_failed_runtime":
        final_status = "review_failed_runtime"
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
        "revision": to_jsonable(state.get("revision")),
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
