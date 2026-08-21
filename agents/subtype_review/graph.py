from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.subtype_review.llm import LLMCallBudgetExceeded, parse_json_content, parse_router_selection
from agents.subtype_review.llm_summary import summarize_audit, summarize_evidence
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS,
    EvidenceRequest,
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
    requests = list(action.get("requests", []) or [])
    if not requests:
        requests = [{
            "dimension": action.get("dimension"),
            "scope": action.get("scope"),
            "target_ids": list(action.get("target_ids", []) or []),
            "proposal_id": action.get("proposal_id"),
        }]
    names = sorted({tool_call_name(call) for call in calls})
    requested_dimensions = {str(request.get("dimension", "")) for request in requests}
    if any(name not in requested_dimensions for name in names):
        raise ValueError(f"Verifier called an unrequested evidence tool: {names}")
    if requested_dimensions - set(names):
        raise ValueError(f"Verifier omitted requested evidence tools: {sorted(requested_dimensions - set(names))}")
    if any(name not in EVIDENCE_DIMENSIONS for name in names):
        raise ValueError(f"Unknown validation tool: {names}")

    all_sets = current_sets(state)
    partition = partition_signature(all_sets)
    evidence = dict(state.get("evidence", {}) or {})
    results = list(evidence.get("results", []) or [])
    cached = {
        evidence_request_key(item)
        for item in results
    }
    requested_keys = []
    for request in requests:
        proposal_id = str(request.get("proposal_id", "") or "")
        request_signature = subject_signature(
            str(request["dimension"]),
            str(request["scope"]),
            all_sets,
            list(request.get("target_ids", []) or []),
            proposal_by_id(state, proposal_id) if proposal_id else {},
        )
        requested_keys.append("|".join((str(request["dimension"]), str(request["scope"]), request_signature, proposal_id)))
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
    for name in names:
        for request in [row for row in requests if str(row.get("dimension")) == name]:
            proposal_id = str(request.get("proposal_id", "") or "")
            proposal = proposal_by_id(state, proposal_id) if proposal_id else {}
            request_signature = subject_signature(
                name,
                str(request["scope"]),
                all_sets,
                list(request.get("target_ids", []) or []),
                proposal,
            )
            artifact_root = (
                Path(str(runtime.get("review_output_root", runtime.get("output_root", ""))))
                / "evidence"
                / str(request["scope"])
            )
            if proposal_id:
                artifact_root /= re.sub(r"[^A-Za-z0-9._-]+", "_", proposal_id).strip("_")
            payload = VALIDATION_FUNCTIONS[name](
                cluster_state,
                patient_states,
                str(runtime.get("data_root", runtime.get("output_root", ""))),
                str(runtime.get("config_dir", "")),
                all_sets,
                scope=str(request["scope"]),
                target_ids=list(request.get("target_ids", []) or []),
                proposal=proposal,
                artifact_root=str(artifact_root),
            )
            payload["partition_signature"] = partition
            payload["scope"] = str(request["scope"])
            payload["target_ids"] = list(request.get("target_ids", []) or [])
            payload["proposal_id"] = proposal_id or None
            payload["subject_signature"] = request_signature
            results.append(payload)
            executed.append(payload)
            if name == "confounder_exclusion" and request["scope"] == "set_identity":
                partition_payload = dict(payload)
                partition_payload["scope"] = "partition"
                partition_payload["target_ids"] = []
                partition_payload["subject_signature"] = subject_signature(
                    name, "partition", all_sets
                )
                results.append(partition_payload)
            try:
                from langchain_core.messages import ToolMessage

                call_id = str(next(
                    call.get("id", "") if isinstance(call, Mapping) else getattr(call, "id", "")
                    for call in calls
                    if tool_call_name(call) == name
                ) or name)
                messages.append(ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=call_id))
            except Exception:
                messages.append({"role": "tool", "name": name, "content": payload})
    evidence["results"] = results
    state["evidence"] = evidence
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
    mandatory_finding_requirements = []
    if not acquire:
        for request in required_evidence_requests(state, include_available=True):
            exact = matching_evidence_results(
                current_evidence,
                request["dimension"],
                request["scope"],
                request["subject_signature"],
                request["proposal_id"],
            )
            targets = request["target_ids"] if request["scope"] == "set_identity" else [None]
            for target in targets:
                target_ids = [target] if target is not None else request["target_ids"]
                existing = [
                    finding
                    for finding in list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
                    if finding.get("dimension") == request["dimension"]
                    and finding.get("scope") == request["scope"]
                    and str(finding.get("subject_signature", "")) == request["subject_signature"]
                    and str(finding.get("proposal_id", "") or "") == str(request["proposal_id"] or "")
                    and set(target_ids).issubset({str(value) for value in finding.get("target_ids", []) or []})
                ]
                if exact and not existing:
                    mandatory_finding_requirements.append({
                        "dimension": request["dimension"],
                        "scope": request["scope"],
                        "target_ids": target_ids,
                        "proposal_id": request["proposal_id"],
                        "python_derived": (
                            request["dimension"] == "cross_modal_consistency"
                            and request["scope"] == "set_identity"
                        ),
                        "metric_refs": sorted({
                            str(ref)
                            for result in exact
                            for child in result.get("results", []) or []
                            for ref in child.get("metric_refs", []) or []
                        }),
                    })
    payload = {
        "mode": "acquire" if acquire else "audit",
        "sets": current_sets(state),
        "evidence_inventory": evidence_inventory(current_evidence),
        "evidence_summary": summarize_evidence(current_evidence),
        "audit_summary": summarize_audit(state.get("audit", {})),
        "message_history": list(state.get("messages", []) or []),
        "request": action if acquire else None,
        "round": dict(state.get("control", {}) or {}).get("round", 0),
        "validation_error": dict(state.get("control", {}) or {}).get("error"),
        "mandatory_finding_requirements": mandatory_finding_requirements,
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
        identity_modalities_by_set = {}
        identity_evidence_available = False
        identity_evidence_attempted = False
        for evidence_row in current_evidence.get("results", []) or []:
            if (
                evidence_row.get("dimension") != "cross_modal_consistency"
                or evidence_row.get("scope") != "set_identity"
            ):
                continue
            identity_evidence_attempted = True
            for child in evidence_row.get("results", []) or []:
                if child.get("tool_name") == "multimodal_consistency_check":
                    identity_evidence_available = (
                        identity_evidence_available or child.get("status") == "success"
                    )
                    identity_modalities_by_set.update(
                        dict(child.get("metrics", {}) or {}).get(
                            "identity_supporting_modalities_by_set", {}
                        )
                    )
        minimum = int(
            dict(dict(state.get("control", {}) or {}).get("policy", {}) or {}).get(
                "accept_min_supporting_modalities", 2
            )
            or 2
        )
        confound_metrics = {}
        confound_evidence = None
        for evidence_row in current_evidence.get("results", []) or []:
            if (
                evidence_row.get("dimension") == "confounder_exclusion"
                and evidence_row.get("scope") == "set_identity"
            ):
                for child in evidence_row.get("results", []) or []:
                    if child.get("tool_name") == "confound_test":
                        metrics = dict(child.get("metrics", {}) or {})
                        if metrics.get("deterministic_flags"):
                            confound_metrics = metrics
                            confound_evidence = evidence_row
        if confound_evidence and not matching_evidence_results(
            current_evidence,
            "confounder_exclusion",
            "partition",
            subject_signature("confounder_exclusion", "partition", sets),
            None,
        ):
            partition_evidence = dict(confound_evidence)
            partition_evidence["scope"] = "partition"
            partition_evidence["target_ids"] = []
            partition_evidence["subject_signature"] = subject_signature(
                "confounder_exclusion", "partition", sets
            )
            current_evidence["results"].append(partition_evidence)
            state["evidence"]["results"].append(partition_evidence)

        for key in ("findings", "gaps"):
            rows = []
            for raw_row in list(audit_payload.get(key, []) or []):
                row = dict(raw_row)
                if (
                    confound_metrics
                    and row.get("dimension") == "confounder_exclusion"
                    and row.get("scope") in {"partition", "set_identity"}
                ):
                    continue
                if (
                    row.get("dimension") == "cross_modal_consistency"
                    and row.get("scope") == "set_identity"
                ):
                    continue
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
        if identity_evidence_attempted:
            ref_root = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
            for target in sorted(set_id(item) for item in sets):
                modalities = sorted(identity_modalities_by_set.get(target, []) or [])
                audit_payload["findings"].append({
                    "target_ids": [target],
                    "dimension": "cross_modal_consistency",
                    "scope": "set_identity",
                    "subject_signature": subject_signature(
                        "cross_modal_consistency", "set_identity", sets, [target]
                    ),
                    "status": (
                        "unavailable" if not identity_evidence_available
                        else "supporting" if len(modalities) >= minimum
                        else "mixed" if modalities
                        else "inconclusive"
                    ),
                    "summary": (
                        f"Python-derived identity support from {len(modalities)} original modalities: "
                        + (", ".join(modalities) if modalities else "none")
                    ),
                    "metric_refs": [f"{ref_root}.{target}"] if identity_evidence_available else [],
                })
        if confound_metrics:
            flags = dict(confound_metrics.get("deterministic_flags", {}) or {})
            root = "tool_results.confound_test.metrics.deterministic_flags"
            global_fields = list(flags.get("global_significant_fields", []) or [])
            audit_payload["findings"].append({
                "target_ids": [],
                "dimension": "confounder_exclusion",
                "scope": "partition",
                "subject_signature": subject_signature(
                    "confounder_exclusion", "partition", sets
                ),
                "status": (
                    "conflicting" if flags.get("strong_technical_conflict")
                    else "mixed" if global_fields
                    else "supporting"
                ),
                "summary": "Python-derived partition-level technical association status.",
                "metric_refs": [
                    f"{root}.strong_technical_conflict",
                    f"{root}.global_significant_fields",
                ],
            })
            significant_by_set = dict(flags.get("set_significant_fields_by_set", {}) or {})
            invalidated = {str(value) for value in flags.get("invalidated_set_ids", []) or []}
            for target in sorted(set_id(item) for item in sets):
                fields = list(significant_by_set.get(target, []) or [])
                audit_payload["findings"].append({
                    "target_ids": [target],
                    "dimension": "confounder_exclusion",
                    "scope": "set_identity",
                    "subject_signature": subject_signature(
                        "confounder_exclusion", "set_identity", sets, [target]
                    ),
                    "status": (
                        "conflicting" if target in invalidated
                        else "mixed" if fields
                        else "supporting"
                    ),
                    "summary": "Python-derived set-level technical association status.",
                    "metric_refs": [
                        f"{root}.set_significant_fields_by_set.{target}",
                        f"{root}.invalidated_set_ids",
                    ],
                })
        attempted = attempted_evidence_keys(state)
        for key in ("findings", "gaps"):
            combined = {}
            for raw_row in [
                *list(dict(state.get("audit", {}) or {}).get(key, []) or []),
                *list(audit_payload.get(key, []) or []),
            ]:
                row = dict(raw_row)
                if (
                    row.get("dimension") == "cross_modal_consistency"
                    and row.get("scope") == "set_identity"
                    and len(row.get("target_ids", []) or []) != 1
                ):
                    continue
                proposal_id = str(row.get("proposal_id", "") or "")
                proposal = proposal_by_id(state, proposal_id) if proposal_id else {}
                row["subject_signature"] = subject_signature(
                    str(row.get("dimension", "")),
                    str(row.get("scope", "")),
                    sets,
                    list(row.get("target_ids", []) or []),
                    proposal,
                )
                evidence_key = evidence_request_key(row)
                if key == "gaps" and evidence_key in attempted:
                    continue
                combined[(
                    row.get("dimension"), row.get("scope"),
                    row.get("subject_signature"), proposal_id,
                    tuple(sorted(str(target) for target in row.get("target_ids", []) or [])),
                )] = row
            audit_payload[key] = list(combined.values())
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
        if row.scope == "set_identity" and len(row.target_ids) != 1:
            raise ValueError("set_identity evidence requires exactly one target")
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
            raise ValueError(
                "Acquired mandatory evidence has no corresponding Finding: "
                + json.dumps({
                    "dimension": request["dimension"],
                    "scope": request["scope"],
                    "target_ids": request["target_ids"],
                    "proposal_id": request["proposal_id"],
                }, ensure_ascii=False)
            )
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
    if finding.get("scope") == "partition":
        return False
    return (
        finding.get("status") == "conflicting"
        and finding.get("scope") == "set_identity"
        and finding.get("dimension") in {
            "biological_support",
            "cross_modal_consistency",
            "confounder_exclusion",
        }
        and target in {str(value) for value in finding.get("target_ids", []) or []}
    )


def invalidates_set(finding: Mapping[str, Any], target: str) -> bool:
    targets = {str(value) for value in finding.get("target_ids", []) or []}
    return (
        finding.get("dimension") == "confounder_exclusion"
        and finding.get("scope") == "set_identity"
        and finding.get("status") == "conflicting"
        and len(targets) == 1
        and target in targets
    )


def set_acceptance(state: Mapping[str, Any], target: str) -> tuple[bool, str]:
    sets = current_sets(state)
    findings = list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
    if not any(
        row.get("dimension") == "cross_modal_consistency"
        and row.get("scope") == "set_identity"
        and row.get("status") == "supporting"
        and target in {str(value) for value in row.get("target_ids", []) or []}
        for row in findings
    ):
        return False, "Accept requires supporting cross-modal set identity evidence"

    signature = subject_signature("cross_modal_consistency", "set_identity", sets, [target])
    modalities = set()
    moderate_modalities = set()
    evidence_level = None
    for result in current_partition_evidence(state).get("results", []):
        if (
            result.get("dimension") == "cross_modal_consistency"
            and result.get("scope") == "set_identity"
            and not result.get("proposal_id")
            and signature == str(result.get("subject_signature", ""))
        ):
            for child in result.get("results", []) or []:
                if child.get("tool_name") == "multimodal_consistency_check":
                    metrics = dict(child.get("metrics", {}) or {})
                    modalities.update(
                        metrics.get(
                            "identity_supporting_modalities_by_set", {}
                        ).get(target, []) or []
                    )
                    moderate_modalities.update(
                        metrics.get("identity_moderate_modalities_by_set", {}).get(target, []) or []
                    )
                    evidence_level = metrics.get("identity_evidence_level_by_set", {}).get(target)
    minimum = int(
        dict(dict(state.get("control", {}) or {}).get("policy", {}) or {}).get(
            "accept_min_supporting_modalities", 2
        ) or 2
    )
    if evidence_level and evidence_level not in {"concordant", "complementary"}:
        return False, f"Accept requires concordant or complementary identity evidence, got {evidence_level}"

    biology = [
        row for row in findings
        if row.get("dimension") == "biological_support"
        and row.get("scope") == "set_identity"
        and target in {str(value) for value in row.get("target_ids", []) or []}
    ]
    if not any(row.get("status") != "unavailable" for row in biology):
        return False, "Accept requires available set-level biological evidence"
    biology_supporting = any(row.get("status") == "supporting" for row in biology)
    if evidence_level == "concordant" and len(modalities) < minimum:
        return False, "Accept requires at least two supporting original modalities"
    if evidence_level == "complementary" and (
        not modalities
        or not moderate_modalities.difference(modalities)
        or not biology_supporting
    ):
        return False, "Complementary Accept requires one supporting modality, one distinct moderate modality, and supporting biology"
    if evidence_level is None and len(modalities) < minimum:
        return False, "Accept requires at least two supporting original modalities"

    confound = [
        row for row in findings
        if row.get("dimension") == "confounder_exclusion"
        and row.get("scope") == "set_identity"
        and target in {str(value) for value in row.get("target_ids", []) or []}
    ]
    known_label = [
        row for row in findings
        if row.get("dimension") == "known_label_echo"
        and row.get("scope") == "partition"
        and (
            not row.get("target_ids")
            or target in {str(value) for value in row.get("target_ids", []) or []}
        )
    ]
    if not any(row.get("status") != "unavailable" for row in confound) or not any(
        row.get("status") != "unavailable" for row in known_label
    ):
        return False, "Accept requires available confounder and known-label evidence"
    if any(row.get("status") == "conflicting" for row in known_label):
        return False, "Accept is blocked by conflicting partition known-label evidence"
    if any(blocks_set_acceptance(row, target) for row in findings):
        return False, "Accept is vetoed by conflicting set-identity or partition evidence"
    return True, ""


def reactivate_provisional_sets(state: dict[str, Any], audit: VerifierOutput) -> None:
    findings = [finding.model_dump() for finding in audit.findings]
    for item in state["sets"]:
        status = str(item.get("status", ""))
        target = set_id(item)
        if status == "provisionally_accepted" and any(
            blocks_set_acceptance(finding, target) for finding in findings
        ):
            item["status"] = "active"
            item.pop("drop_reason", None)
        elif (
            status == "provisionally_dropped"
            and (
                (
                    item.get("drop_reason") == "technical_invalidation"
                    and not any(invalidates_set(finding, target) for finding in findings)
                )
            )
        ):
            item["status"] = "active"
            item.pop("drop_reason", None)


def revision_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(item) for item in dict(state.get("revision", {}) or {}).get("candidates", []) or []],
        key=lambda item: str(item.get("proposal_id") or item.get("plan_id") or ""),
    )


def revision_candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "plan_id": str(candidate.get("plan_id") or candidate.get("proposal_id") or ""),
        "proposal_id": str(candidate.get("proposal_id") or candidate.get("plan_id") or ""),
        "source_set_id": candidate.get("source_set_id"),
        "set_ids": list(candidate.get("set_ids", []) or []),
        "source_networks": list(candidate.get("source_networks", []) or []),
        "eligible_for_review": bool(candidate.get("eligible_for_review", True)),
    }
    for key in ("child_count", "child_sizes", "set_sizes"):
        if key in candidate:
            summary[key] = candidate[key]
    return summary


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

    add("biological_support", "set_identity", set_ids)
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
            if include_available or not cross_modal:
                add("cross_modal_consistency", scope, targets, proposal)
            if not cross_modal:
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
        if (
            item.get("status") == "provisionally_dropped"
            and not drop_evidence
            and item.get("drop_reason") not in {
                "insufficient_validation_after_exhaustion",
            }
        ):
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
    sets = [
        item for item in current_sets(state)
        if str(item.get("status", "active")) != "provisionally_dropped"
    ]
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
    cross_metrics = {}
    for result in current_partition_evidence(state).get("results", []) or []:
        if (
            result.get("dimension") == "cross_modal_consistency"
            and result.get("scope") == "set_identity"
            and not result.get("proposal_id")
        ):
            for child in result.get("results", []) or []:
                cross_metrics.update(dict(child.get("metrics", {}) or {}))
    split_signals = {
        str(target)
        for target, modalities in dict(
            cross_metrics.get("split_candidate_sets_by_modality", {}) or {}
        ).items()
        if modalities
    }
    merge_signals = {
        tuple(sorted(str(value) for value in pair.split("+")))
        for pair, modalities in dict(
            cross_metrics.get("merge_candidate_pairs_by_modality", {}) or {}
        ).items()
        if modalities and len(pair.split("+")) == 2
    }
    motives = set()
    for item in sets:
        target = set_id(item)
        if str(item.get("status", "active")) != "active" or target in invalid:
            continue
        if any(
            target in {str(value) for value in row.get("target_ids", []) or []}
            and row.get("scope") == "set_identity"
            and (
                (
                    row.get("dimension") == "cross_modal_consistency"
                    and row.get("status") in {"mixed", "conflicting"}
                )
                or (
                    row.get("dimension") == "biological_support"
                    and row.get("status") == "conflicting"
                )
            )
            for row in findings
        ):
            motives.add(target)
        if target in split_signals:
            motives.add(target)
    intents = [
        {"action": "split", "target_ids": [set_id(item)]}
        for item in sets
        if set_id(item) in motives
        if len(item.get("member_ids", []) or []) >= 2 * min_split_size
        and f"split:{set_id(item)}" not in blocked
    ]
    merge_ids = {set_id(item) for item in sets if set_id(item) not in invalid}
    for targets in sorted(merge_signals):
        if set(targets).issubset(merge_ids) and f"merge:{'+'.join(targets)}" not in blocked:
            intents.append({"action": "merge", "target_ids": list(targets)})
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
    request_rows = action.requests or (
        [EvidenceRequest(
            dimension=str(action.dimension),
            scope=str(action.scope),
            target_ids=action.target_ids,
            proposal_id=action.proposal_id,
        )]
        if action.action == "need_more_evidence"
        else []
    )
    targets = set(action.target_ids)
    targets.update(target for request in request_rows for target in request.target_ids)
    if not targets.issubset(known):
        raise ValueError(f"Router referenced inactive or unknown sets: {sorted(targets - known)}")
    if action.action == "need_more_evidence":
        gaps = [*list(dict(state.get("audit", {}) or {}).get("gaps", []) or []), *required_evidence_requests(state)]
        attempted = attempted_evidence_keys(state)
        for request in request_rows:
            proposal = proposal_by_id(state, str(request.proposal_id or "")) if request.proposal_id else {}
            signature = subject_signature(
                request.dimension,
                request.scope,
                sets,
                request.target_ids,
                proposal,
            )
            matching = [
                gap for gap in gaps
                if gap.get("dimension") == request.dimension
                and gap.get("scope") == request.scope
                and str(gap.get("proposal_id", "") or "") == str(request.proposal_id or "")
                and str(gap.get("subject_signature", "")) == signature
            ]
            if not matching:
                raise ValueError("Router requested evidence without a matching scoped gap")
            gap_targets = set(str(item) for gap in matching for item in gap.get("target_ids", []) or [])
            if not gap_targets and request.target_ids:
                raise ValueError("Whole-partition evidence requests must use empty target_ids")
            if request.target_ids and not set(request.target_ids).issubset(gap_targets):
                raise ValueError("Router evidence target is outside the matching Verifier gap")
            key = "|".join((request.dimension, request.scope, signature, str(request.proposal_id or "")))
            if key in attempted:
                raise ValueError("Evidence request already attempted for this subject")
        return
    selected = [item for item in sets if set_id(item) in targets]
    allowed_statuses = {"active", "provisionally_accepted"} if action.action == "merge" else {"active"}
    if any(str(item.get("status", "active")) not in allowed_statuses for item in selected):
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
    if any(
        target in intent["target_ids"]
        for intent in initial_revision_intents(state)
    ):
        raise ValueError("Scientific action is blocked by pending structural review")
    if action.action == "accept":
        acceptable, reason = set_acceptance(state, target)
        if not acceptable:
            raise ValueError(reason)
    if action.action == "drop":
        positive = any(invalidates_set(finding, target) for finding in findings)
        exhausted = (
            not set_acceptance(state, target)[0]
            and not any(target in intent["target_ids"] for intent in initial_revision_intents(state))
        )
        if not positive and not exhausted:
            raise ValueError("Drop requires technical invalidation or exhausted unsupported identity")


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
            RouterAction(**intent).model_dump()
            for intent in initial_revision_intents(state)
        )
    if requestable:
        legal_action_candidates.append(
            RouterAction(
                action="need_more_evidence",
                requests=[EvidenceRequest(**{
                    key: row[key]
                    for key in ("dimension", "scope", "target_ids", "proposal_id")
                }) for row in requestable],
            ).model_dump()
        )
    if not requestable and not legal_action_candidates:
        control["status"] = "unresolved"
        control["error"] = None
        control["next"] = "end"
        state["control"] = control
        append_trace(state, {"node": "router", "event": "scientific_evidence_exhausted"})
        return state
    legal_actions = [
        {"action_id": f"A{index}", "action": row}
        for index, row in enumerate(legal_action_candidates)
    ]
    payload = {
        "sets": [
            {
                "id": set_id(item),
                "size": len(item.get("member_ids", []) or []),
                "status": str(item.get("status", "active")),
            }
            for item in sets
        ],
        "evidence_summary": [
            {
                "dimension": str(row.get("dimension", "")),
                "scope": str(row.get("scope", "")),
                "target_ids": list(row.get("target_ids", []) or []),
                "proposal_id": row.get("proposal_id"),
                "status": str(row.get("status", "")),
                "summary": str(row.get("summary", ""))[:240],
            }
            for row in list(audit.get("findings", []) or [])
        ],
        "legal_actions": legal_actions,
        "revision": {
            key: to_jsonable(revision[key])
            for key in ("action", "target_ids", "status")
            if key in revision
        },
        "validation_error": dict(state.get("control", {}) or {}).get("error"),
    }
    action = None
    legal_action_map = {row["action_id"]: row["action"] for row in legal_actions}
    for attempt in range(3):
        if len(legal_action_candidates) == 1:
            action = RouterAction(**legal_action_candidates[0])
            break
        result = invoke_with_recovery(model, payload, state, "router")
        if result is None:
            return state
        try:
            selection = parse_router_selection(result)
            if selection.action_id not in legal_action_map:
                raise ValueError("Router selected an unknown action_id")
            action = RouterAction(**{
                **legal_action_map[selection.action_id],
                "reason": selection.reason,
            })
            validate_router_action(action, state)
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rejected_action = to_jsonable(result)
            append_trace(state, {
                "node": "router",
                "event": "contract_rejection",
                "attempt": attempt + 1,
                "rejected_action": rejected_action,
                "error": error,
            })
            if attempt == 2:
                control = dict(state.get("control", {}) or {})
                control["status"] = "unresolved"
                control["error"] = None
                control["next"] = "end"
                control["router_contract_error"] = error
                state["control"] = control
                append_trace(state, {
                    "node": "router",
                    "event": "router_contract_exhausted",
                    "error": error,
                    "legal_actions": legal_actions,
                })
                return state
            payload["validation_error"] = {
                "error": error,
                "rejected_action": rejected_action,
                "legal_actions": legal_actions,
                "instruction": "Return one action_id exactly as provided.",
            }
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
                if action.action == "drop":
                    findings = list(audit.get("findings", []) or [])
                    if any(invalidates_set(row, target) for row in findings):
                        item["drop_reason"] = "technical_invalidation"
                    else:
                        item["drop_reason"] = "insufficient_validation_after_exhaustion"
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
        item.pop("drop_reason", None)
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
            str(runtime.get("data_root", runtime.get("output_root", ""))),
            config_dir=str(runtime.get("config_dir", "")),
            all_cluster_states=current_sets(state),
            artifact_root=str(
                Path(str(runtime.get("review_output_root", runtime.get("output_root", ""))))
                / "revision_candidates"
                / str(action["action"])
                / re.sub(r"[^A-Za-z0-9._-]+", "_", "+".join(targets)).strip("_")
            ),
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
    if len(candidates) == 1:
        parsed = ReviserOutput(plan_id=str(candidates[0].get("plan_id") or ""), reason="single valid plan")
    else:
        payload = {
            "action": {
                "action": action.get("action"),
                "target_ids": targets,
            },
            "candidates": [revision_candidate_summary(candidate) for candidate in candidates],
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
    partition_findings = [
        row for row in list(dict(state.get("audit", {}) or {}).get("findings", []) or [])
        if row.get("scope") == "partition"
    ]
    partition_statuses = {str(row.get("status", "")) for row in partition_findings}
    partition_assessment = (
        "conflicting" if "conflicting" in partition_statuses
        else "mixed" if partition_statuses.intersection({"mixed", "inconclusive"})
        else "supporting" if partition_statuses and partition_statuses == {"supporting"}
        else "unavailable"
    )
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
        "llm_usage": to_jsonable(control.get("llm_usage", {})),
        "partition_sets": to_jsonable(sets),
        "accepted_subtype_sets": to_jsonable(accepted_sets),
        "partition_patient_count": len({member for item in sets for member in item.get("member_ids", [])}),
        "accepted_patient_count": len({member for item in accepted_sets for member in item.get("member_ids", [])}),
        "partition_assessment": partition_assessment,
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
