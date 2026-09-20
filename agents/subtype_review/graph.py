from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.subtype_review.llm import parse_json_content, summarize_reports
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS, EvidenceRequest, EvidenceReportBatch, ReviewContext,
    ReviewState, RevisionPlan, RouterPlan, set_id,
)
from agents.subtype_review.tools import TOOL_REGISTRY, compact_tool_result
from utils.tool_utils import to_jsonable


def partition_signature(sets: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "set_id": set_id(item),
                "member_ids": sorted(str(member) for member in item.get("member_ids", [])),
            }
            for item in sorted(sets, key=set_id)
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def current_sets(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(state["partition"]["sets"], key=set_id)


def append_trace(state: dict[str, Any], event: dict[str, Any]) -> None:
    control = state["control"]
    control["trace"].append({"round": control["round"], **event})


def is_length_finish_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in {"LengthFinishReasonError", "LLMOutputLengthError"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def mark_failure(
    state: dict[str, Any], node: str, exc: Exception, immediate: bool = False
) -> None:
    control = dict(state["control"])
    control["failures"] += 1
    control["error"] = f"{type(exc).__name__}: {exc}"
    if node != "verifier":
        control["next"] = node
    if immediate or control["failures"] >= control["max_failures"]:
        control["status"] = "review_unavailable"
        control["next"] = "end"
    state["control"] = control
    append_trace(state, {"node": node, "event": "failure", "error": control["error"]})


def initial_review_state(candidate_sets: list[dict[str, Any]]) -> ReviewState:
    sets = []
    for item in candidate_sets:
        identifier = set_id(item)
        if not identifier:
            raise ValueError("Initial candidate set has no identifier")
        sets.append({
            "set_id": identifier,
            "member_ids": sorted(str(member) for member in item["member_ids"]),
            "revision_lineage": list(item.get("revision_lineage", [])),
        })
    identifiers = [set_id(item) for item in sets]
    members = [member for item in sets for member in item["member_ids"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Initial candidate sets contain duplicate identifiers")
    if len(members) != len(set(members)):
        raise ValueError("Initial candidate sets contain overlapping patients")
    return {
        "partition": {"sets": sets},
        "round_evidence": [],
        "reports": [],
        "evidence_memory": {},
        "messages": [],
        "router_plan": None,
        "revision_plan": None,
        "revision_result": None,
        "history": [],
        "control": {
            "round": 0,
            "failures": 0,
            "status": "reviewing",
            "next": "router",
            "error": None,
            "max_rounds": 10,
            "max_failures": 3,
            "pending_evidence_requests": [],
            "router_validation_error": None,
            "router_correction_attempts": 0,
            "previous_invalid_plan": None,
            "eligible_tools": {},
            "trace": [],
        },
    }


def current_partition_evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    signature = partition_signature(current_sets(state))
    rows = [
        row
        for entry in state["history"]
        for row in entry.get("round_evidence", []) or []
    ]
    rows.extend(state["round_evidence"])
    return [row for row in rows if row.get("partition_signature") == signature]


def report_tool_refs(rows: list[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[str]]:
    """Index tool provenance by dimension, scope and individual report target."""
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        targets = [""] if row.get("scope") == "partition" else row.get("target_ids", []) or []
        for target in targets:
            grouped.setdefault(
                (str(row["dimension"]), str(row["scope"]), str(target)), set()
            ).add(str(row["tool_name"]))
    return {key: sorted(value) for key, value in grouped.items()}


def completed_tool_keys(state: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (str(row.get("tool_name", "")), str(row["partition_signature"]), str(target))
        for row in current_partition_evidence(state)
        if row.get("status") in {"success", "scientific_unavailable"}
        for target in row.get("target_ids", []) or [""]
    }


def evidence_coverage(state: Mapping[str, Any], runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    scope_dimensions = {"set_identity": set(), "partition": set()}
    for metadata in (runtime or {}).get("tool_registry", TOOL_REGISTRY).values():
        if metadata.get("verifier_selectable"):
            scope_dimensions[metadata["scope"]].add(metadata["dimension"])
    coverage = {
        "set_identity": {
            set_id(item): {
                dimension: "unassessed"
                for dimension in sorted(scope_dimensions["set_identity"])
            }
            for item in current_sets(state)
        },
        "partition": {
            dimension: "unassessed"
            for dimension in sorted(scope_dimensions["partition"])
        },
    }
    for report in state["reports"]:
        dimension = str(report.get("dimension", ""))
        if report.get("scope") == "partition":
            if dimension in coverage["partition"]:
                coverage["partition"][dimension] = "assessed"
        else:
            for target in report.get("target_ids", []) or []:
                target = str(target)
                if dimension in coverage["set_identity"].get(target, {}):
                    coverage["set_identity"][target][dimension] = "assessed"
    return coverage


def structural_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    row = next(
        (
            item for item in state["round_evidence"]
            if item.get("tool_name") == "multimodal_consistency_check"
        ),
        {},
    )
    if not row:
        row = next(
            (item for item in reversed(current_partition_evidence(state))
             if item.get("tool_name") == "multimodal_consistency_check"),
            {},
        )
    return row.get("full_metrics", {}).get("structural_characterization", {})


def compact_structural_index(state: Mapping[str, Any]) -> dict[str, Any]:
    structural = structural_metrics(state)
    per_set = {}
    for item in current_sets(state):
        target = set_id(item)
        metrics = structural.get("internal_structure_by_set", {}).get(target, {})
        probe = metrics.get("fused_binary_probe", {})
        resampling = probe.get("resampling", {})
        modality_probe = metrics.get("probe_support_by_modality", {})
        per_set[target] = {
            "member_n": len(item["member_ids"]),
            "binary_probe": {
                "child_sizes": probe.get("child_sizes"),
                "fused_silhouette": probe.get("median_silhouette"),
                "normalized_cut": probe.get("normalized_cut"),
                "median_ari": resampling.get("median_resample_ari"),
                "consensus_separation": resampling.get("consensus_separation"),
                "pac": resampling.get("pac"),
                "degenerate_fraction": resampling.get("degenerate_resample_fraction"),
                "modality_probe_silhouettes": {
                    name: modality_probe.get(name, {}).get("median_silhouette")
                    for name in ("ct", "wsi", "rna", "wxs", "cnv")
                },
            },
        }
    boundaries = []
    pair_fields = (
        "pair_median_silhouette", "left_median_margin", "right_median_margin",
        "left_boundary_separation", "right_boundary_separation",
    )
    for _, pair in sorted(structural.get("boundary_by_pair", {}).items()):
        boundaries.append({
            "pair": sorted(pair["targets"]),
            "fused": {key: pair.get("fused", {}).get(key) for key in pair_fields},
            "modalities": {
                name: {key: pair.get("modalities", {}).get(name, {}).get(key) for key in pair_fields}
                for name in ("ct", "wsi", "rna", "wxs", "cnv")
            },
        })
    return {
        "per_set": per_set,
        "boundary_by_pair": boundaries,
        "limitations": list(structural.get("limitations", []) or []),
    }


def verifier_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    writes = ("control", "round_evidence", "messages", "reports", "evidence_memory")
    state = {
        **state,
        "control": {**state["control"], "trace": list(state["control"]["trace"])},
        "evidence_memory": dict(state["evidence_memory"]),
    }
    values = runtime.context if isinstance(runtime, Runtime) else runtime
    control = dict(state["control"])
    try:
        if control.get("next") == "verifier_acquire":
            requests = [
                EvidenceRequest.model_validate(item).model_dump()
                for item in control.get("pending_evidence_requests", [])
            ]
            if not requests:
                raise ValueError("Verifier requires Router EvidenceRequests")
            tools = values.get("tool_registry", TOOL_REGISTRY)
            sets = current_sets(state)
            signature = partition_signature(sets)
            completed = completed_tool_keys(state)
            eligible = {}
            targets = {set_id(item) for item in sets}
            for name, metadata in sorted(tools.items()):
                if not metadata.get("verifier_selectable"):
                    continue
                matching = [item for item in requests if item["dimension"] == metadata["dimension"]]
                if not matching:
                    continue
                if metadata["scope"] == "partition":
                    if not any(not item["target_ids"] for item in matching) or (name, signature, "") in completed:
                        continue
                    allowed = set()
                else:
                    allowed = {
                        target
                        for item in matching
                        for target in item["target_ids"]
                        if target in targets and (name, signature, target) not in completed
                    }
                    if not allowed:
                        continue
                eligible[name] = {
                    "dimension": metadata["dimension"],
                    "scope": metadata["scope"],
                    "target_ids": sorted(allowed),
                    "description": metadata["description"],
                }
            if not eligible:
                raise ValueError("No eligible tool can answer Router EvidenceRequests")
            control["eligible_tools"] = eligible
            state["control"] = control
            result = values["verifier_model"].invoke({
                "mode": "acquire",
                "partition": state["partition"],
                "evidence_requests": requests,
                "current_evidence": state["reports"],
                "eligible_tools": eligible,
                "round": control["round"],
            })
            calls = result["tool_calls"] if isinstance(result, Mapping) else result.tool_calls
            names = [call["name"] for call in calls]
            if len(names) != len(set(names)):
                raise ValueError("Verifier cannot call the same registered tool twice")
            cluster_state = {
                "cluster_id": "p_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16],
                "member_ids": sorted(member for item in sets for member in item["member_ids"]),
                "active_modalities": tuple(values.get("active_modalities") or ("ct", "wsi", "rna", "wxs", "cnv")),
            }
            results = []
            messages = [*state["messages"], result]
            for request in requests:
                targets = set(request["target_ids"])
                covered = set()
                partition_covered = False
                for call in calls:
                    name, args = call["name"], call["args"]
                    metadata = tools.get(name, {})
                    if metadata.get("dimension") != request["dimension"]:
                        continue
                    call_targets = {str(target) for target in args.get("target_ids", []) or []}
                    if not targets:
                        if metadata.get("scope") == "partition" and not call_targets:
                            partition_covered = True
                    elif metadata.get("scope") == "set_identity":
                        covered.update(args.get("target_ids", []) or [])
                if not targets:
                    if not partition_covered:
                        raise ValueError(
                            "Verifier tool calls do not cover partition EvidenceRequest "
                            f"{request['dimension']}"
                        )
                    continue
                if not targets.issubset(covered):
                    raise ValueError(
                        "Verifier tool calls do not cover EvidenceRequest "
                        f"{request['dimension']}:{sorted(targets)}"
                    )
            for call in calls:
                name, args = call["name"], call["args"]
                if name not in tools:
                    raise ValueError(f"Verifier called an unregistered tool: {name}")
                metadata = tools[name]
                if name not in eligible:
                    raise ValueError(f"Verifier called tool not listed in eligible_tools: {name}")
                target_ids = sorted({str(target) for target in args.get("target_ids", []) or [] if str(target)})
                if metadata["scope"] == "partition" and target_ids:
                    raise ValueError(f"Partition tool {name} cannot target sets")
                if metadata["scope"] == "set_identity" and not target_ids:
                    raise ValueError(f"Set tool {name} requires target_ids")
                available_targets = set(eligible[name].get("target_ids", []) or [])
                if metadata["scope"] == "set_identity" and not set(target_ids).issubset(available_targets):
                    raise ValueError(f"Verifier tool {name} targeted an ineligible set")
                raw = metadata["function"](
                    cluster_state,
                    values["patient_states_by_id"],
                    str(values["data_root"]),
                    config_dir=str(values["config_dir"]),
                    all_cluster_states=sets,
                    scope=str(metadata["scope"]),
                    target_ids=target_ids,
                    artifact_root=str(values.get("artifact_root", values["data_root"])),
                )
                compact = compact_tool_result(raw, name)
                if compact["status"] == "runtime_failure":
                    raise RuntimeError(
                        f"Scientific tool {name} failed at runtime: "
                        f"{'; '.join(compact.get('errors', []))}"
                    )
                compact.update({
                    "dimension": metadata["dimension"],
                    "scope": metadata["scope"],
                    "target_ids": target_ids,
                    "partition_signature": signature,
                })
                results.append(compact)
                messages.append(ToolMessage(
                    content=json.dumps({key: compact[key] for key in (
                        "tool_name", "dimension", "scope", "target_ids", "status",
                        "metrics", "warnings", "missing_reason", "errors",
                    )}, ensure_ascii=False),
                    tool_call_id=call["id"],
                ))
            state["round_evidence"] = results
            state["messages"] = messages
            control["next"] = "verifier_audit"
            append_trace(state, {"node": "verifier", "event": "tool_selection", "tools": names})
            state["control"].update(failures=0, error=None)
            return {key: state[key] for key in writes}
        if control.get("next") != "verifier_audit":
            raise ValueError(f"Unexpected Verifier state: {control.get('next')}")
        expected = set(report_tool_refs(state["round_evidence"]))
        audit_payload = {
            "mode": "audit",
            "partition": {"sets": [
                {"set_id": set_id(item), "member_n": len(item["member_ids"])}
                for item in current_sets(state)
            ]},
            "required_reports": [
                {"dimension": dimension, "scope": scope, "target_ids": [target] if target else []}
                for dimension, scope, target in sorted(expected)
            ],
            "prior_reports": state["reports"],
            "round_evidence": state["round_evidence"],
            "tool_messages": list(state["messages"]),
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
        evidence = current_partition_evidence(state)
        expected_refs = report_tool_refs([
            row for row in evidence if row["status"] in {"success", "scientific_unavailable"}
        ])
        try:
            sets = {set_id(item) for item in current_sets(state)}
            prior_keys = {
                (str(report["dimension"]), str(report["scope"]), next(iter(report.get("target_ids", [])), ""))
                for report in state["reports"]
            }
            allowed_keys = expected | prior_keys
            reported = set()
            for report in batch.reports:
                targets = set(report.target_ids)
                if targets - sets:
                    raise ValueError("Evidence Report references a non-current set")
                if report.scope == "set_identity" and len(targets) != 1:
                    raise ValueError("set Evidence Reports must target exactly one current set")
                key = (
                    report.dimension,
                    report.scope,
                    next(iter(targets)) if targets else "",
                )
                if key in reported:
                    raise ValueError("Duplicate Evidence Report")
                if key not in allowed_keys:
                    raise ValueError("Evidence Report does not match current evidence")
                report.tool_refs = expected_refs.get(key, [])
                if not report.tool_refs:
                    raise ValueError("Evidence Report must cite tool_refs")
                reported.add(key)
            if not expected.issubset(reported):
                raise ValueError(f"Evidence Reports must cover exactly current targets: {expected - reported}")
            rows = {row["tool_name"]: row for row in evidence}
            for report in batch.reports:
                target = next(iter(report.target_ids), "")
                refs = []
                for tool_name in report.tool_refs:
                    row = rows[tool_name]
                    metrics = row.get("metrics", {}) or {}
                    leaves = [str(ref) for ref in row.get("metric_refs", []) or []]
                    base = f"tool_results.{tool_name}.metrics"
                    if report.scope == "partition":
                        refs.extend(leaves)
                        continue
                    prefixes = []
                    for key, value in metrics.items():
                        if not isinstance(value, Mapping):
                            continue
                        if key == "cross_modal_consistency":
                            if target in value.get("per_set", {}):
                                prefixes.append(f"{base}.{key}.per_set.{target}")
                            if "partition" in value:
                                prefixes.append(f"{base}.{key}.partition")
                        elif target in value:
                            prefixes.append(f"{base}.{key}.{target}")
                        elif f"{target}_vs_rest" in value:
                            prefixes.append(f"{base}.{key}.{target}_vs_rest")
                    refs.extend(
                        ref for ref in leaves
                        if any(ref.startswith(f"{prefix}.") or ref.startswith(f"{prefix}[") for prefix in prefixes)
                    )
                report.metric_refs = sorted(set(refs))
        except Exception as exc:
            mark_failure(state, "verifier", exc, immediate=True)
            return {key: state[key] for key in writes}
        merged = {
            (
                str(report["dimension"]),
                str(report["scope"]),
                next(iter(report.get("target_ids", [])), ""),
            ): dict(report)
            for report in state["reports"]
        }
        for report in batch.reports:
            key = (report.dimension, report.scope, next(iter(report.target_ids), ""))
            merged[key] = report.model_dump()
        state["reports"] = list(merged.values())
        signature = partition_signature(current_sets(state))
        state["evidence_memory"][signature] = copy.deepcopy(state["reports"])
        state["control"].update(failures=0, error=None)
        append_trace(state, {"node": "verifier", "event": "reports", "count": len(batch.reports)})
        control = dict(state["control"])
        control["pending_evidence_requests"] = []
        control["eligible_tools"] = {}
        control["next"] = "router"
        state["control"] = control
    except Exception as exc:
        mark_failure(state, "verifier", exc, is_length_finish_error(exc))
    return {key: state[key] for key in writes}


def router_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    writes = ("control", "router_plan", "history", "round_evidence", "messages")
    state = {
        **state,
        "control": {**state["control"], "trace": list(state["control"]["trace"])},
        "history": list(state["history"]),
    }
    values = runtime.context if isinstance(runtime, Runtime) else runtime
    control = dict(state["control"])
    if control.get("status") != "reviewing":
        return {key: state[key] for key in writes}
    terminal_only = control["round"] >= control["max_rounds"]
    plan = None
    try:
        tools = values.get("tool_registry", TOOL_REGISTRY)
        signature = partition_signature(current_sets(state))
        completed = completed_tool_keys(state)
        targets = [set_id(item) for item in current_sets(state)]
        requests = {}
        for name, metadata in sorted(tools.items()):
            if not metadata.get("verifier_selectable"):
                continue
            for target in [""] if metadata["scope"] == "partition" else targets:
                if (name, signature, target) in completed:
                    continue
                question = f"Clarify the current {metadata['dimension']} evidence"
                requests[(metadata["dimension"], target)] = {
                    "dimension": metadata["dimension"],
                    "target_ids": [target] if target else [],
                    "question": f"{question} for {target}." if target else f"{question}.",
                }
        available = [requests[key] for key in sorted(requests)]
        payload = {
            "partition": state["partition"],
            "evidence_reports": summarize_reports(state["reports"]),
            "evidence_coverage": evidence_coverage(state, values),
            "structural_index": compact_structural_index(state),
            "available_evidence_requests": available,
            "round": control["round"] + (0 if terminal_only else 1),
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
        result = values["router_model"].invoke(copy.deepcopy(payload))
        plan = RouterPlan.model_validate(parse_json_content(result))
        coverage = payload["evidence_coverage"]
        structural_rows = [
            row.get("full_metrics", {}).get("structural_characterization", {})
            for row in current_partition_evidence(state)
            if row.get("tool_name") == "multimodal_consistency_check" and row.get("status") == "success"
        ]
        if plan.evidence_requests:
            seen = set()
            for request in plan.evidence_requests:
                same_dimension = [
                    item for item in available if item["dimension"] == request.dimension
                ]
                targets = set(request.target_ids)
                if not targets:
                    valid = any(not item["target_ids"] for item in same_dimension)
                else:
                    available_targets = {
                        target
                        for item in same_dimension
                        for target in item["target_ids"]
                    }
                    valid = targets.issubset(available_targets)
                if not valid:
                    raise ValueError(
                        "EvidenceRequest is not currently available "
                        "for the requested dimension/targets"
                    )
                targets = request.target_ids or ["__partition__"]
                for target in targets:
                    key = (request.dimension, target)
                    if key in seen:
                        raise ValueError("Duplicate or overlapping EvidenceRequest coverage")
                    seen.add(key)
        else:
            known = {set_id(item) for item in current_sets(state)}
            occupied = set()
            for action in plan.actions:
                targets = set(action.target_ids)
                if occupied.intersection(targets):
                    raise ValueError("Current sets cannot occur in multiple Router actions")
                occupied.update(targets)
            if occupied != known:
                raise ValueError("Router plan must cover every current set exactly once")
            for action in plan.actions:
                for target in action.target_ids:
                    current = coverage["set_identity"].get(target, {})
                    if action.decision_state.identity != "unassessed" and current.get("biological_support") != "assessed":
                        raise ValueError("identity assessment requires biological_support evidence")
                    if action.decision_state.structure != "unassessed" and current.get("cross_modal_consistency") != "assessed":
                        raise ValueError("structure assessment requires cross_modal_consistency evidence")
                    if (
                        action.decision_state.alternative_explanation != "unassessed"
                        and current.get("confounder_exclusion") != "assessed"
                    ):
                        raise ValueError("alternative-explanation assessment requires confounder evidence")
                if action.action == "split":
                    if action.decision_state.structure != "incompatible":
                        raise ValueError("split requires structure=incompatible")
                    if not any(action.target_ids[0] in row.get("internal_structure_by_set", {}) for row in structural_rows):
                        raise ValueError("split requires assessed structural evidence")
                if action.action == "merge":
                    if action.decision_state.structure != "incompatible":
                        raise ValueError("merge requires structure=incompatible")
                    if not any(set(pair["targets"]) == set(action.target_ids) for row in structural_rows for pair in row.get("boundary_by_pair", {}).values()):
                        raise ValueError("merge requires assessed pairwise structural evidence")
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
            return {key: state[key] for key in writes}
    except Exception as exc:
        if is_length_finish_error(exc):
            mark_failure(state, "router", exc, immediate=True)
        elif isinstance(exc, ValueError):
            attempts = int(control.get("router_correction_attempts", 0)) + 1
            if attempts > 2:
                mark_failure(state, "router", exc, immediate=True)
                return {key: state[key] for key in writes}
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
        return {key: state[key] for key in writes}
    if not terminal_only:
        control["round"] += 1
    control["error"] = None
    control["router_validation_error"] = None
    control["router_correction_attempts"] = 0
    control["previous_invalid_plan"] = None
    state["router_plan"] = plan.model_dump()
    state["history"].append({
        "round": control["round"],
        "terminal_only": terminal_only,
        "partition_signature": partition_signature(current_sets(state)),
        "partition": copy.deepcopy(state["partition"]),
        "round_evidence": copy.deepcopy(state["round_evidence"]),
        "evidence_reports": copy.deepcopy(state["reports"]),
        "router_plan": plan.model_dump(),
        "revision_plan": None,
        "revision_result": None,
    })
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
        state["round_evidence"] = []
        state["messages"] = []
        control["next"] = "verifier_acquire"
    elif any(action.action in {"split", "merge"} for action in plan.actions):
        control["next"] = "reviser"
    else:
        control["status"] = "complete"
        control["next"] = "end"
    state["control"] = control
    state["control"].update(failures=0, error=None)
    return {key: state[key] for key in writes}


def reviser_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    writes = ("control", "partition", "reports", "revision_plan", "revision_result",
              "history", "round_evidence", "messages")
    state = {
        **state,
        "control": {**state["control"], "trace": list(state["control"]["trace"])},
        "history": [dict(entry) for entry in state["history"]],
    }
    values = runtime.context if isinstance(runtime, Runtime) else runtime
    router_plan = RouterPlan.model_validate(state["router_plan"])
    control = dict(state["control"])
    refs = {str(ref) for row in current_partition_evidence(state) for ref in row.get("metric_refs", [])}
    prefix = "tool_results.multimodal_consistency_check.metrics.cross_modal_consistency.per_set."
    selected = set()
    for action in router_plan.actions:
        if action.action == "split":
            paths = (f"{prefix}{action.target_ids[0]}.internal_structure.",)
        elif action.action == "merge":
            left, right = action.target_ids
            paths = (
                f"{prefix}{left}.boundary_to_other_sets.{right}.",
                f"{prefix}{right}.boundary_to_other_sets.{left}.",
            )
        else:
            continue
        selected.update(ref for ref in refs if ref.startswith(paths))
    payload = {
        "partition": state["partition"],
        "router_plan": router_plan.model_dump(),
        "raw_structural_metrics": {"partition": state["partition"], "structural_characterization": structural_metrics(state)},
        "available_metric_refs": sorted(selected),
    }
    if control.get("revision_validation_error"):
        payload["previous_revision_plan"] = state.get("revision_plan")
        payload["revision_validation_error"] = control["revision_validation_error"]
    try:
        result = values["reviser_model"].invoke(payload)
        plan = RevisionPlan.model_validate(parse_json_content(result))
        revision_identity = {
            "split_plans": [item.model_dump(include={
                "target_id", "n_children", "structural_basis", "execution_strategy", "metric_refs",
            }) for item in plan.split_plans],
            "merge_plans": [item.model_dump(include={"target_ids", "metric_refs"}) for item in plan.merge_plans],
        }
        signature = hashlib.sha256(
            json.dumps(revision_identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        failed = set(control.get("failed_revision_plan_signatures", []) or [])
        if signature in failed:
            raise ValueError("Reviser returned a previously failed RevisionPlan")
        state["revision_plan"] = plan.model_dump()
        try:
            expected_splits = {action.target_ids[0] for action in router_plan.actions if action.action == "split"}
            expected_merges = {tuple(action.target_ids) for action in router_plan.actions if action.action == "merge"}
            if {item.target_id for item in plan.split_plans} != expected_splits:
                raise ValueError("RevisionPlan split targets do not match RouterPlan")
            if {tuple(item.target_ids) for item in plan.merge_plans} != expected_merges:
                raise ValueError("RevisionPlan merge targets do not match RouterPlan")
            old_sets = current_sets(state)
            by_id = {set_id(item): item for item in old_sets}
            occupied = set()
            for item in plan.split_plans:
                if item.target_id not in by_id or item.target_id in occupied:
                    raise ValueError("RevisionPlan split target is not current or is duplicated")
                if (
                    item.n_children != 2
                    or item.structural_basis != ["fused"]
                    or item.execution_strategy != "fused_similarity_spectral"
                ):
                    raise ValueError("Split requires a binary fused_similarity_spectral plan with structural_basis=['fused']")
                occupied.add(item.target_id)
            for item in plan.merge_plans:
                targets = set(item.target_ids)
                if not targets.issubset(by_id) or occupied.intersection(targets):
                    raise ValueError("RevisionPlan merge targets overlap or are not current")
                occupied.update(targets)
            for item in [*plan.split_plans, *plan.merge_plans]:
                if not set(item.metric_refs).issubset(refs):
                    raise ValueError("RevisionPlan references unavailable metrics")
            from tools.cross_modal_structure import execute_split_membership

            replacements: dict[str, list[dict[str, Any]]] = {}
            superseded = []
            operations = []
            for item in plan.split_plans:
                source = by_id[item.target_id]
                groups = execute_split_membership(
                    str(values["data_root"]),
                    list(source["member_ids"]),
                    item.n_children,
                    item.execution_strategy,
                    list(item.structural_basis),
                )
                if any(len(group) < 2 for group in groups):
                    raise ValueError("Split produced a non-estimable child set")
                children = []
                for index, members in enumerate(groups, 1):
                    child_id = f"{item.target_id}_S{index}"
                    children.append({
                        "set_id": child_id,
                        "member_ids": sorted(members),
                        "revision_lineage": [
                            *source.get("revision_lineage", []),
                            {"action": "split", "parent_set_id": item.target_id, "plan": item.model_dump()},
                        ],
                    })
                replacements[item.target_id] = children
                superseded.append(copy.deepcopy(source))
                operations.append({"action": "split", "target_ids": [item.target_id], "children": [child["set_id"] for child in children]})
            for item in plan.merge_plans:
                sources = [by_id[target] for target in item.target_ids]
                members = sorted(member for source in sources for member in source["member_ids"])
                if len(members) != len(set(members)):
                    raise ValueError("Merge target memberships overlap")
                merged_id = "_M_".join(sorted(item.target_ids))
                replacements.update({target: [] for target in item.target_ids})
                replacements[item.target_ids[0]] = [{
                    "set_id": merged_id,
                    "member_ids": members,
                    "revision_lineage": [
                        *sum((source.get("revision_lineage", []) for source in sources), []),
                        {"action": "merge", "parent_set_ids": sorted(item.target_ids), "plan": item.model_dump()},
                    ],
                }]
                superseded.extend(copy.deepcopy(source) for source in sources)
                operations.append({"action": "merge", "target_ids": sorted(item.target_ids), "new_set_id": merged_id})
            new_sets = [item for source in old_sets for item in replacements.get(set_id(source), [source])]
            new_partition = {"sets": sorted(new_sets, key=set_id)}
            revision_result = {
                "old_partition_signature": partition_signature(old_sets),
                "new_partition_signature": partition_signature(new_partition["sets"]),
                "operations": operations,
                "superseded_sets": superseded,
                "algorithm": "deterministic_spectral_clustering_and_set_union",
                "seed": 0,
                "metric_refs": sorted({ref for item in [*plan.split_plans, *plan.merge_plans] for ref in item.metric_refs}),
                "parameters": {"split_strategies": [item.execution_strategy for item in plan.split_plans]},
            }
            state["partition"] = new_partition
            state["reports"] = copy.deepcopy(
                state["evidence_memory"].get(revision_result["new_partition_signature"], [])
            )
        except Exception as exc:
            control = dict(state["control"])
            control["failed_revision_plan_signatures"] = sorted(
                set(control.get("failed_revision_plan_signatures", []) or []) | {signature}
            )
            control["revision_validation_error"] = f"{type(exc).__name__}: {exc}"
            state["control"] = control
            raise
        state["revision_result"] = revision_result
        index = state["control"]["history_index"]
        state["history"][index]["revision_plan"] = copy.deepcopy(state["revision_plan"])
        state["history"][index]["revision_result"] = copy.deepcopy(revision_result)
        append_trace(state, {"node": "reviser", "event": "revision_applied", "result": revision_result})
        state["control"].update(failures=0, error=None)
        control = dict(state["control"])
        control["revision_validation_error"] = None
        state["round_evidence"] = []
        state["messages"] = []
        control["next"] = "router"
        state["control"] = control
    except Exception as exc:
        mark_failure(state, "reviser", exc, is_length_finish_error(exc))
    return {key: state[key] for key in writes}


def route_node(state: Mapping[str, Any], *, node: str, destinations: Mapping[str, str], default: str) -> str:
    control = state["control"]
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return node
    return destinations.get(control.get("next", ""), default)


def build_review_graph() -> Any:
    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    nodes = (
        ("verifier", verifier_node,
         {"router": "router"}, "verifier"),
        ("router", router_node,
         {"verifier_acquire": "verifier", "reviser": "reviser"}, "end"),
        ("reviser", reviser_node,
         {"router": "router"}, "end"),
    )
    for name, node, destinations, default in nodes:
        graph.add_node(name, node)
        graph.add_conditional_edges(
            name, partial(route_node, node=name, destinations=destinations, default=default),
            {target: END if target == "end" else target for target in {name, "end", *destinations.values()}},
        )
    graph.add_edge(START, "router")
    return graph.compile()


def save_review_outputs(state: Mapping[str, Any], output_root: str, *, direct: bool = False) -> dict[str, Any]:
    root = Path(output_root) if direct else Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    plan = state.get("router_plan") or {}
    actions = {
        target: action.get("action", "")
        for action in plan.get("actions", []) or []
        for target in action.get("target_ids", [])
        if action.get("action") in {"accept", "drop"}
    }
    sets = current_sets(state)
    accepted = [item for item in sets if actions.get(set_id(item)) == "accept"]
    dropped = [item for item in sets if actions.get(set_id(item)) == "drop"]
    accepted_reports = []
    for item in accepted:
        reports = [report for report in state["reports"] if report["scope"] == "partition" or set_id(item) in report["target_ids"]]
        accepted_reports.append({
            "set_id": set_id(item),
            "membership": item["member_ids"],
            "revision_lineage": item.get("revision_lineage", []),
            "evidence_by_dimension": {
                dimension: [report for report in reports if report.get("dimension") == dimension]
                for dimension in EVIDENCE_DIMENSIONS
            },
            "metric_refs": sorted({ref for report in reports for ref in report.get("metric_refs", [])}),
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
            "key_evidence": [report for report in state["reports"] if report["scope"] == "partition" or set_id(item) in report["target_ids"]],
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
        "history": to_jsonable(state["history"]),
        "decision_trace": to_jsonable(state["control"].get("trace", [])),
    }
    for filename, payload in {
        "final_partition_sets.json": sets,
        "final_subtype_sets.json": accepted,
        "accepted_subtype_reports.json": accepted_reports,
        "dropped_set_reports.json": dropped_reports,
        "review_history.json": state["history"],
        "final_review_summary.json": summary,
    }.items():
        (root / filename).write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "decision_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in state["control"].get("trace", []):
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    return summary
