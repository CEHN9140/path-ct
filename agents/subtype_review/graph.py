from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.subtype_review.llm import parse_json_content, parse_revision_plan, parse_router_plan
from agents.subtype_review.llm_summary import summarize_reports
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS,
    EvidenceRequest,
    EvidenceReportBatch,
    MergePlan,
    ReviewContext,
    ReviewState,
    RevisionPlan,
    RouterAction,
    RouterPlan,
    SplitPlan,
    set_id,
)
from agents.subtype_review.tools import (
    TOOL_REGISTRY,
    compact_tool_result,
)
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


def partition_artifact_id(signature: str) -> str:
    return "p_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def current_sets(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(item) for item in dict(state.get("partition", {}) or {}).get("sets", []) or []],
        key=set_id,
    )


def compact_partition_for_llm(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sets": [
            {"set_id": set_id(item), "member_n": len(item.get("member_ids", []) or [])}
            for item in current_sets(state)
        ]
    }


def required_reports_for_round(
    state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in state.get("round_evidence", []) or []:
        target_ids = [str(target) for target in row.get("target_ids", []) or []]
        targets = [""] if row.get("scope") == "partition" else target_ids
        for target in targets:
            grouped.setdefault(
                (str(row["dimension"]), str(row["scope"]), target), set()
            ).add(str(row["tool_name"]))
    return [
        {
            "dimension": dimension,
            "scope": scope,
            "target_ids": [] if not target else [target],
            "tool_names": sorted(tool_names),
        }
        for (dimension, scope, target), tool_names in sorted(grouped.items())
    ]


def expected_tool_refs_for_round(
    state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[tuple[str, str, str], list[str]]:
    grouped: dict[tuple[str, str, str], set[str]] = {}
    signature = partition_signature(current_sets(state))
    rows = [
        *[row for row in state.get("history", []) for row in row.get("round_evidence", [])],
        *list(state.get("round_evidence", []) or []),
    ]
    for row in rows:
        if row.get("partition_signature") != signature or row.get("status") not in {"success", "scientific_unavailable"}:
            continue
        targets = [""] if row.get("scope") == "partition" else row.get("target_ids", [])
        for target in targets:
            grouped.setdefault((str(row["dimension"]), str(row["scope"]), str(target)), set()).add(str(row["tool_name"]))
    return {key: sorted(value) for key, value in grouped.items()}


def context_values(runtime: Any) -> dict[str, Any]:
    if hasattr(runtime, "context"):
        return dict(runtime.context or {})
    return dict(runtime or {})


def registry(runtime: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(runtime.get("tool_registry") or TOOL_REGISTRY)


def append_trace(state: dict[str, Any], event: dict[str, Any]) -> None:
    control = dict(state.get("control", {}) or {})
    control.setdefault("trace", []).append(
        {"round": control.get("round", 0), **event}
    )
    state["control"] = control


def is_length_finish_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in {"LengthFinishReasonError", "LLMOutputLengthError"}:
            return True
        current = current.__cause__ or current.__context__
    return any(name in str(exc) for name in ("LengthFinishReasonError", "LLMOutputLengthError"))


def mark_failure(
    state: dict[str, Any], node: str, exc: Exception, immediate: bool = False
) -> None:
    control = dict(state.get("control", {}) or {})
    control["failures"] = int(control.get("failures", 0)) + 1
    control["error"] = f"{type(exc).__name__}: {exc}"
    if node != "verifier":
        control["next"] = node
    if immediate or control["failures"] >= int(control.get("max_failures", 3)):
        control["status"] = "review_unavailable"
        control["next"] = "end"
    state["control"] = control
    append_trace(state, {"node": node, "event": "failure", "error": control["error"]})


def mark_success(state: dict[str, Any]) -> None:
    control = dict(state["control"])
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
                "member_ids": sorted(str(member) for member in item.get("member_ids", [])),
                "revision_lineage": list(item.get("revision_lineage", []) or []),
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
            "next": "prepare_round",
            "error": None,
            "max_rounds": 10,
            "max_failures": 3,
            "pending_evidence_requests": [],
            "router_validation_error": None,
            "router_correction_attempted": False,
            "eligible_tools": {},
            "trace": [],
        },
    }


def prepare_round_node(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
    if state["control"].get("status") != "reviewing":
        return state
    values = context_values(runtime)
    tools = registry(values)
    control = dict(state["control"])
    requests = (
        [EvidenceRequest.model_validate(item).model_dump() for item in control.get("pending_evidence_requests", [])]
        or [
            EvidenceRequest(
                dimension=dimension,
                target_ids=[] if dimension == "known_label_echo" else [set_id(item) for item in current_sets(state)],
                question=f"Interpret the current {dimension} evidence.",
            ).model_dump()
            for dimension in EVIDENCE_DIMENSIONS
        ]
    )
    signature = partition_signature(current_sets(state))
    completed = completed_tool_keys(state)
    eligible = {}
    for name, metadata in sorted(tools.items()):
        if not metadata.get("verifier_selectable"):
            continue
        matching = [request for request in requests if request["dimension"] == metadata["dimension"]]
        if not matching or metadata["scope"] == "partition" and any(request["target_ids"] for request in matching):
            continue
        if metadata["scope"] == "partition":
            if (name, signature, "") in completed:
                continue
            eligible[name] = {"dimension": metadata["dimension"], "scope": metadata["scope"], "target_ids": [], "description": metadata["description"]}
            continue
        targets = {
            target
            for request in matching
            for target in request["target_ids"]
            if (name, signature, target) not in completed
        }
        if targets:
            eligible[name] = {"dimension": metadata["dimension"], "scope": metadata["scope"], "target_ids": sorted(targets), "description": metadata["description"]}
    state["round_evidence"] = []
    state["messages"] = []
    state["router_plan"] = None
    state["revision_plan"] = None
    state["revision_result"] = None
    control["failed_revision_plan_signatures"] = []
    control["revision_validation_error"] = None
    state["reports"] = copy.deepcopy(state.get("evidence_memory", {}).get(signature, []))
    control["pending_evidence_requests"] = requests
    control["eligible_tools"] = eligible
    control["next"] = "verifier_acquire"
    control["partition_signature"] = signature
    state["control"] = control
    append_trace(state, {
        "node": "prepare_round",
        "event": "prepared",
        "evidence_requests": requests,
        "eligible_tools": sorted(eligible),
    })
    return state


def completed_tool_keys(state: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    signature = partition_signature(current_sets(state))
    keys = set()
    rows = list(state.get("round_evidence", []) or [])
    rows.extend(
        row
        for entry in state.get("history", []) or []
        for row in entry.get("round_evidence", []) or []
    )
    for row in rows:
        if row.get("partition_signature") != signature:
            continue
        if row.get("status") in {"success", "scientific_unavailable"}:
            base = (
                str(row.get("tool_name", "")),
                str(row.get("partition_signature", "")),
            )
            targets = [str(target) for target in row.get("target_ids", []) or []]
            keys.update((*base, target) for target in targets)
            if not targets:
                keys.add((*base, ""))
    return keys


def current_partition_evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    signature = partition_signature(current_sets(state))
    rows = [
        row
        for entry in state.get("history", []) or []
        for row in entry.get("round_evidence", []) or []
    ]
    rows.extend(state.get("round_evidence", []) or [])
    return [row for row in rows if row.get("partition_signature") == signature]


def available_evidence_requests(
    state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    tools = registry(runtime)
    eligible = dict(state["control"].get("eligible_tools", {}) or {})
    signature = partition_signature(current_sets(state))
    completed = completed_tool_keys(state)
    requests = []
    for name, metadata in sorted(tools.items()):
        if name not in eligible or not metadata.get("verifier_selectable"):
            continue
        if metadata["scope"] == "partition":
            if (name, signature, "") in completed:
                continue
            requests.append({
                "dimension": metadata["dimension"],
                "target_ids": [],
                "question": f"Clarify the current {metadata['dimension']} evidence.",
            })
            continue
        requests.extend({
            "dimension": metadata["dimension"],
            "target_ids": [target],
            "question": f"Clarify the current {metadata['dimension']} evidence for {target}.",
        } for target in eligible[name].get("target_ids", [])
          if (name, signature, target) not in completed)
    unique = {(item["dimension"], tuple(item["target_ids"])): item for item in requests}
    return [unique[key] for key in sorted(unique)]


def execute_tool_calls(
    state: dict[str, Any], message: Any, runtime: Mapping[str, Any]
) -> None:
    calls = list(getattr(message, "tool_calls", []) or [])
    if isinstance(message, Mapping):
        calls = list(message.get("tool_calls", []) or [])
    names = [str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")) for call in calls]
    if len(names) != len(set(names)):
        raise ValueError("Verifier cannot call the same registered tool twice")
    sets = current_sets(state)
    signature = partition_signature(sets)
    tools = registry(runtime)
    eligible = dict(state["control"].get("eligible_tools", {}) or {})
    requests = state["control"].get("pending_evidence_requests", []) or []
    cluster_state = {
        "cluster_id": partition_artifact_id(signature),
        "member_ids": sorted(member for item in sets for member in item["member_ids"]),
    }
    results = []
    messages = [*state.get("messages", []), message]
    for call in calls:
        name = str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", ""))
        if name not in tools:
            raise ValueError(f"Verifier called an unregistered tool: {name}")
        metadata = tools[name]
        if not metadata.get("verifier_selectable"):
            raise ValueError(f"Tool {name} is not selectable by Verifier")
        if name not in eligible:
            raise ValueError(f"Verifier called tool not listed in eligible_tools: {name}")
        args = call.get("args", {}) if isinstance(call, Mapping) else getattr(call, "args", {})
        if isinstance(args, str):
            args = parse_json_content(args)
        args = dict(args or {})
        target_ids = sorted({str(target) for target in args.get("target_ids", []) or [] if str(target)})
        known = {set_id(item) for item in sets}
        if not set(target_ids).issubset(known):
            raise ValueError(f"Verifier tool {name} targeted a non-current set")
        if metadata["scope"] == "partition" and target_ids:
            raise ValueError(f"Partition tool {name} cannot target sets")
        if metadata["scope"] == "set_identity" and not target_ids:
            raise ValueError(f"Set tool {name} requires target_ids")
        available_targets = set(eligible[name].get("target_ids", []) or [])
        if metadata["scope"] == "set_identity" and not set(target_ids).issubset(available_targets):
            raise ValueError(f"Verifier tool {name} targeted an ineligible set")
        matching = [request for request in requests if request["dimension"] == metadata["dimension"]]
        if metadata["scope"] == "partition":
            if not any(not request["target_ids"] for request in matching):
                raise ValueError(f"Verifier tool {name} does not answer a pending EvidenceRequest")
        elif not any(set(target_ids).issubset(set(request["target_ids"])) for request in matching):
            raise ValueError(f"Verifier tool {name} does not answer the selected EvidenceRequest")
        completed = completed_tool_keys(state)
        if metadata["scope"] == "partition":
            if (name, signature, "") in completed:
                raise ValueError(f"Verifier repeated completed tool {name}")
        elif any((name, signature, target) in completed for target in target_ids):
            raise ValueError(f"Verifier repeated completed tool {name} for target")
        raw = metadata["function"](
            cluster_state,
            dict(runtime.get("patient_states_by_id", {}) or {}),
            str(runtime.get("data_root", "")),
            config_dir=str(runtime.get("config_dir", "")),
            all_cluster_states=sets,
            scope=str(metadata["scope"]),
            target_ids=target_ids,
            artifact_root=str(runtime.get("artifact_root", runtime.get("data_root", ""))),
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
        call_id = str(call.get("id", name) if isinstance(call, Mapping) else getattr(call, "id", name))
        tool_message_payload = {
            "tool_name": name,
            "dimension": metadata["dimension"],
            "scope": metadata["scope"],
            "target_ids": target_ids,
            "status": compact["status"],
            "metrics": compact["metrics"],
            "warnings": list(compact.get("warnings", []) or []),
            "missing_reason": compact.get("missing_reason", ""),
            "errors": list(compact.get("errors", []) or []),
        }
        try:
            from langchain_core.messages import ToolMessage

            messages.append(ToolMessage(
                content=json.dumps(tool_message_payload, ensure_ascii=False),
                tool_call_id=call_id,
            ))
        except Exception:
            messages.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(tool_message_payload, ensure_ascii=False),
            })
    state["round_evidence"] = results
    state["messages"] = messages
    control = dict(state["control"])
    control["next"] = "verifier_audit"
    state["control"] = control
    append_trace(state, {"node": "verifier", "event": "tool_selection", "tools": names})


def raw_metric_refs(rows: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(ref)
        for row in rows
        for ref in row.get("metric_refs", []) or []
    }


def metric_blocks_for_report(
    report: Any, state: Mapping[str, Any]
) -> list[str]:
    rows = {
        str(row.get("tool_name", "")): row
        for row in current_partition_evidence(state)
    }
    target = next(iter(report.target_ids), "")
    refs = []
    for tool_name in report.tool_refs:
        metrics = rows.get(tool_name, {}).get("metrics", {}) or {}
        base = f"tool_results.{tool_name}.metrics"
        if report.scope == "partition":
            refs.extend(f"{base}.{key}" for key in sorted(metrics))
            continue
        for key, value in metrics.items():
            if not isinstance(value, Mapping):
                continue
            if key == "cross_modal_consistency":
                if target in value.get("per_set", {}):
                    refs.append(f"{base}.{key}.per_set.{target}")
                if "partition" in value:
                    refs.append(f"{base}.{key}.partition")
            elif key == "sets" and target in value:
                refs.append(f"{base}.{key}.{target}")
            elif target in value:
                refs.append(f"{base}.{key}.{target}")
            elif f"{target}_vs_rest" in value:
                refs.append(f"{base}.{key}.{target}_vs_rest")
    return sorted(set(refs))


def expected_report_keys(state: Mapping[str, Any], runtime: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    keys = set()
    for row in state.get("round_evidence", []) or []:
        targets = [""] if row.get("scope") == "partition" else row.get("target_ids", [])
        keys.update((str(row["dimension"]), str(row["scope"]), str(target)) for target in targets)
    return keys


def validate_reports(
    batch: EvidenceReportBatch, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    sets = {set_id(item) for item in current_sets(state)}
    expected = expected_report_keys(state, runtime)
    prior_keys = {
        (str(report["dimension"]), str(report["scope"]), next(iter(report.get("target_ids", [])), ""))
        for report in state.get("reports", []) or []
    }
    allowed_keys = expected | prior_keys
    expected_refs = expected_tool_refs_for_round(state, runtime)
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
        if not report.tool_refs:
            raise ValueError("Evidence Report must cite tool_refs")
        target = next(iter(targets), "")
        allowed = set(expected_refs.get(key, []))
        if set(report.tool_refs) != allowed:
            raise ValueError(
                f"Evidence Report tool_refs mismatch for {key}: "
                f"expected={sorted(allowed)} got={sorted(report.tool_refs)}"
            )
        reported.add(key)
    if not expected.issubset(reported):
        raise ValueError(f"Evidence Reports must cover exactly current targets: {expected - reported}")
    for report in batch.reports:
        report.metric_refs = metric_blocks_for_report(report, state)


def verifier_node(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
    values = context_values(runtime)
    control = dict(state["control"])
    try:
        if control.get("next") == "verifier_acquire":
            model = values["verifier_model"]
            result = model.invoke({
                "mode": "acquire",
                "partition": state["partition"],
                "evidence_requests": control["pending_evidence_requests"],
                "current_evidence": state.get("reports", []),
                "eligible_tools": control.get("eligible_tools", {}),
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
        if control.get("round", 0) >= control.get("max_rounds", 10):
            control["status"] = "review_incomplete_due_to_round_budget"
            control["next"] = "end"
        else:
            control["next"] = "router"
        state["control"] = control
    except Exception as exc:
        mark_failure(state, "verifier", exc, is_length_finish_error(exc))
    return state


def validate_evidence_request(
    request: EvidenceRequest, state: Mapping[str, Any], runtime: Mapping[str, Any], action_targets: set[str]
) -> None:
    known = {set_id(item) for item in current_sets(state)}
    targets = set(request.target_ids)
    if not targets.issubset(known) or not targets.issubset(action_targets):
        raise ValueError("Evidence request targets are outside its action")
    if request.dimension == "known_label_echo" and targets:
        raise ValueError("known_label_echo requests are partition-scoped")
    if request.dimension != "known_label_echo" and not targets:
        raise ValueError("set-level evidence requests require targets")
    available = available_evidence_requests(state, runtime)
    if request.dimension == "known_label_echo":
        valid = any(item["dimension"] == request.dimension and not item["target_ids"] for item in available)
    else:
        valid = set(request.target_ids).issubset({
            target
            for item in available
            if item["dimension"] == request.dimension
            for target in item["target_ids"]
        })
    if not valid:
        raise ValueError("Evidence request is not currently available to Verifier")


def validate_action_contract(
    action: RouterAction, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    known = {set_id(item) for item in current_sets(state)}
    if not set(action.target_ids).issubset(known):
        raise ValueError("Router referenced a non-current set")
    if action.action == "need_more_evidence":
        seen = set()
        for request in action.evidence_requests:
            validate_evidence_request(request, state, runtime, set(action.target_ids))
            key = (request.dimension, tuple(request.target_ids))
            if key in seen:
                raise ValueError("Duplicate extra tool request")
            seen.add(key)
        return


def validate_router_plan(
    plan: RouterPlan, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
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
        validate_action_contract(action, state, runtime)


def history_entry(state: Mapping[str, Any], plan: RouterPlan) -> dict[str, Any]:
    return {
        "round": int(state["control"].get("round", 0)) + 1,
        "partition_signature": partition_signature(current_sets(state)),
        "partition": copy.deepcopy(state["partition"]),
        "round_evidence": copy.deepcopy(state.get("round_evidence", [])),
        "evidence_reports": copy.deepcopy(state.get("reports", [])),
        "router_plan": plan.model_dump(),
        "revision_plan": None,
        "revision_result": None,
    }


def router_node(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
    values = context_values(runtime)
    control = dict(state["control"])
    if control.get("status") != "reviewing":
        return state
    if control.get("round", 0) >= control.get("max_rounds", 10):
        control["status"] = "review_incomplete_due_to_round_budget"
        control["next"] = "end"
        state["control"] = control
        return state
    try:
        payload = {
            "partition": state["partition"],
            "evidence_reports": summarize_reports(state.get("reports", [])),
            "structural_index": compact_structural_index(state),
            "available_evidence_requests": available_evidence_requests(state, values),
            "round": int(control.get("round", 0)) + 1,
        }
        if control.get("router_validation_error"):
            payload["validation_error"] = control["router_validation_error"]
            payload["instruction"] = "return a corrected RouterPlan only"
        plan = parse_router_plan(
            values["router_model"].invoke(copy.deepcopy(payload))
        )
        validate_router_plan(plan, state, values)
    except Exception as exc:
        if is_length_finish_error(exc) or control.get("router_correction_attempted"):
            mark_failure(state, "router", exc, immediate=True)
        elif isinstance(exc, ValueError):
            control["router_correction_attempted"] = True
            control["router_validation_error"] = f"{type(exc).__name__}: {exc}"
            control["error"] = control["router_validation_error"]
            control["next"] = "router"
            state["control"] = control
            append_trace(state, {
                "node": "router",
                "event": "validation_retry",
                "error": control["router_validation_error"],
            })
        else:
            mark_failure(state, "router", exc, immediate=False)
        return state
    control["round"] = int(control.get("round", 0)) + 1
    control["error"] = None
    control["router_validation_error"] = None
    control["router_correction_attempted"] = False
    state["router_plan"] = plan.model_dump()
    state["history"].append(history_entry(state, plan))
    control["history_index"] = len(state["history"]) - 1
    append_trace(state, {
        "node": "router",
        "event": "decision",
        "plan": plan.model_dump(),
    })
    if any(action.action == "need_more_evidence" for action in plan.actions):
        control["pending_evidence_requests"] = [
            request.model_dump()
            for action in plan.actions
            if action.action == "need_more_evidence"
            for request in action.evidence_requests
        ]
        control["next"] = "prepare_round"
    elif any(action.action in {"split", "merge"} for action in plan.actions):
        control["pending_evidence_requests"] = []
        control["next"] = "reviser"
    else:
        control["pending_evidence_requests"] = []
        control["status"] = "complete"
        control["next"] = "end"
    state["control"] = control
    mark_success(state)
    return state


def structural_actions(plan: RouterPlan) -> tuple[list[RouterAction], list[RouterAction]]:
    return (
        [action for action in plan.actions if action.action == "split"],
        [action for action in plan.actions if action.action == "merge"],
    )


def revision_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    row = next(
        (
            item for item in state.get("round_evidence", []) or []
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
    return {
        "partition": state["partition"],
        "structural_characterization": dict(
            dict(row.get("full_metrics", {}) or {}).get("structural_characterization", {}) or {}
        ),
    }


def compact_structural_index(state: Mapping[str, Any]) -> dict[str, Any]:
    row = next(
        (
            item for item in state.get("round_evidence", []) or []
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
    structural = dict(
        dict(row.get("full_metrics", {}) or {}).get("structural_characterization", {}) or {}
    )
    per_set = {}
    for item in current_sets(state):
        target = set_id(item)
        metrics = dict(structural.get("internal_structure_by_set", {}).get(target, {}) or {})
        probe = dict(metrics.get("fused_binary_probe", {}) or {})
        resampling = dict(probe.get("resampling", {}) or {})
        modality_probe = dict(metrics.get("probe_support_by_modality", {}) or {})
        per_set[target] = {
            "member_n": len(item.get("member_ids", []) or []),
            "binary_probe": {
                "child_sizes": probe.get("child_sizes"),
                "fused_silhouette": probe.get("median_silhouette"),
                "normalized_cut": probe.get("normalized_cut"),
                "median_ari": resampling.get("median_resample_ari"),
                "consensus_separation": resampling.get("consensus_separation"),
                "pac": resampling.get("pac"),
                "degenerate_fraction": resampling.get("degenerate_resample_fraction"),
                "modality_probe_silhouettes": {
                    name: dict(modality_probe.get(name, {}) or {}).get("median_silhouette")
                    for name in ("ct", "wsi", "rna", "wxs", "cnv")
                },
            },
        }
    boundaries = []
    for pair_key, pair_metrics in sorted(
        dict(structural.get("boundary_by_pair", {}) or {}).items()
    ):
        pair = dict(pair_metrics or {})
        targets = list(pair.get("targets", []) or [])
        if len(targets) != 2:
            targets = str(pair_key).split("+", 1)
        fused = dict(pair.get("fused", {}) or {})
        boundaries.append({
            "pair": sorted(str(target) for target in targets),
            "fused": {
                key: fused.get(key)
                for key in (
                    "pair_median_silhouette", "left_median_margin", "right_median_margin",
                    "left_boundary_separation", "right_boundary_separation",
                )
            },
            "modalities": {
                name: {
                    key: dict(pair.get("modalities", {}).get(name, {}) or {}).get(key)
                    for key in (
                        "pair_median_silhouette", "left_median_margin", "right_median_margin",
                        "left_boundary_separation", "right_boundary_separation",
                    )
                }
                for name in ("ct", "wsi", "rna", "wxs", "cnv")
            },
        })
    return {
        "per_set": per_set,
        "boundary_by_pair": boundaries,
        "limitations": list(structural.get("limitations", []) or []),
    }


def revision_plan_signature(plan: RevisionPlan) -> str:
    payload = {
        "split_plans": [
            {
                "target_id": item.target_id,
                "n_children": item.n_children,
                "structural_basis": item.structural_basis,
                "execution_strategy": item.execution_strategy,
            }
            for item in plan.split_plans
        ],
        "merge_plans": [{"target_ids": item.target_ids} for item in plan.merge_plans],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_revision_plan(
    plan: RevisionPlan, router_plan: RouterPlan, state: Mapping[str, Any]
) -> None:
    split_actions, merge_actions = structural_actions(router_plan)
    expected_splits = {action.target_ids[0] for action in split_actions}
    expected_merges = {tuple(action.target_ids) for action in merge_actions}
    if {item.target_id for item in plan.split_plans} != expected_splits:
        raise ValueError("RevisionPlan split targets do not match RouterPlan")
    if {tuple(item.target_ids) for item in plan.merge_plans} != expected_merges:
        raise ValueError("RevisionPlan merge targets do not match RouterPlan")
    current = {set_id(item): item for item in current_sets(state)}
    occupied = set()
    for item in plan.split_plans:
        if item.target_id not in current or item.target_id in occupied:
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
        if not targets.issubset(current) or occupied.intersection(targets):
            raise ValueError("RevisionPlan merge targets overlap or are not current")
        occupied.update(targets)
    refs = raw_metric_refs(current_partition_evidence(state))
    for item in [*plan.split_plans, *plan.merge_plans]:
        if not set(item.metric_refs).issubset(refs):
            raise ValueError("RevisionPlan references unavailable metrics")


def execute_revision_plan(
    state: dict[str, Any], plan: RevisionPlan, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    from tools.cross_modal_structure import execute_split_membership

    old_sets = current_sets(state)
    old_signature = partition_signature(old_sets)
    by_id = {set_id(item): item for item in old_sets}
    replacements: dict[str, list[dict[str, Any]]] = {}
    superseded = []
    operations = []
    for item in plan.split_plans:
        source = by_id[item.target_id]
        groups = execute_split_membership(
            str(runtime["data_root"]),
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
    new_sets = []
    for source in old_sets:
        if set_id(source) not in replacements:
            new_sets.append(source)
            continue
        if replacements[set_id(source)]:
            new_sets.extend(replacements[set_id(source)])
    new_partition = {"sets": sorted(new_sets, key=set_id)}
    result = {
        "old_partition_signature": old_signature,
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
        state.get("evidence_memory", {}).get(result["new_partition_signature"], [])
    )
    return result


def reviser_node(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
    values = context_values(runtime)
    router_plan = RouterPlan.model_validate(state["router_plan"])
    control = dict(state["control"])
    payload = {
        "partition": state["partition"],
        "router_plan": router_plan.model_dump(),
        "raw_structural_metrics": revision_metrics(state),
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
        if control["round"] >= control.get("max_rounds", 10):
            control["status"] = "review_incomplete_due_to_round_budget"
            control["next"] = "end"
        else:
            control["next"] = "prepare_round"
        state["control"] = control
    except Exception as exc:
        mark_failure(state, "reviser", exc, is_length_finish_error(exc))
    return state


def route_prepare(state: Mapping[str, Any]) -> str:
    return "end" if state["control"].get("status") != "reviewing" else "verifier"


def route_verifier(state: Mapping[str, Any]) -> str:
    control = state["control"]
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return "verifier"
    if control.get("next") == "router":
        return "router"
    return "verifier"


def route_router(state: Mapping[str, Any]) -> str:
    control = state["control"]
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return "router"
    if control.get("next") == "prepare_round":
        return "prepare_round"
    if control.get("next") == "reviser":
        return "reviser"
    return "end"


def route_reviser(state: Mapping[str, Any]) -> str:
    control = state["control"]
    if control.get("status") != "reviewing":
        return "end"
    if control.get("error"):
        return "reviser"
    return "prepare_round" if control.get("next") == "prepare_round" else "end"


def build_review_graph() -> Any:
    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    graph.add_node("prepare_round", lambda state, runtime: prepare_round_node(state, runtime))
    graph.add_node("verifier", lambda state, runtime: verifier_node(state, runtime))
    graph.add_node("router", lambda state, runtime: router_node(state, runtime))
    graph.add_node("reviser", lambda state, runtime: reviser_node(state, runtime))
    graph.add_edge(START, "prepare_round")
    graph.add_edge("prepare_round", "verifier")
    graph.add_conditional_edges(
        "verifier", route_verifier,
        {"verifier": "verifier", "router": "router", "end": END},
    )
    graph.add_conditional_edges(
        "router", route_router,
        {"prepare_round": "prepare_round", "reviser": "reviser", "router": "router", "end": END},
    )
    graph.add_conditional_edges(
        "reviser", route_reviser,
        {"prepare_round": "prepare_round", "reviser": "reviser", "end": END},
    )
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
