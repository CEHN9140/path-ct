from __future__ import annotations

import json
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


class ReviewContext(TypedDict, total=False):
    patient_states_by_id: dict[str, dict[str, Any]]
    output_root: str
    config_dir: str
    tool_functions: dict[str, Any]
    source_output_root: str


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
    return {
        "sets": sets,
        "evidence": {"results": []},
        "audit": {"findings": [], "gaps": []},
        "action": None,
        "messages": [],
        "control": {
            "round": 0,
            "failures": 0,
            "status": "reviewing",
            "error": None,
            "blocked_actions": [],
            "trace": [],
            "max_rounds": 12,
            "max_failures": 3,
        },
    }


def merge_runtime(base: Mapping[str, Any], context: Mapping[str, Any] | None) -> dict[str, Any]:
    runtime = dict(base)
    if hasattr(context, "context"):
        context = context.context
    runtime.update(dict(context or {}))
    return runtime


def current_sets(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return active_sets(list(state.get("sets", []) or []))


def partition_signature(sets: list[dict[str, Any]]) -> str:
    rows = [
        (set_id(item), tuple(sorted(str(x) for x in item.get("member_ids", []))))
        for item in sets
    ]
    return json.dumps(sorted(rows), ensure_ascii=False)


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
    append_trace(state, {"node": node, "status": control["status"], "error": control["error"]})


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
            for ref in list(child.get("metric_refs", []) or []):
                refs.add(str(ref))
            visit(f"tool_results.{tool_name}.metrics", child.get("metrics", {}))
    return refs


def serializable_messages(messages: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for message in list(messages or []):
        if isinstance(message, Mapping):
            rows.append({
                "role": str(message.get("role", "tool")),
                "content": to_jsonable(message.get("content", "")),
                "tool_call_id": str(message.get("tool_call_id", "") or ""),
            })
            continue
        rows.append({
            "role": str(getattr(message, "type", "tool")),
            "content": to_jsonable(getattr(message, "content", "")),
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
        })
    return rows


def execute_tool_calls(
    state: dict[str, Any],
    ai_message: Any,
    runtime: Mapping[str, Any],
) -> bool:
    calls = list(getattr(ai_message, "tool_calls", []) or [])
    if not calls and isinstance(ai_message, Mapping):
        calls = list(ai_message.get("tool_calls", []) or [])
    calls = [
        call
        for call in calls
        if str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")).strip()
    ]
    if not calls:
        return False
    patient_states = dict(runtime.get("patient_states_by_id", {}) or {})
    all_sets = current_sets(state)
    cluster_state = {
        "cluster_id": "GLOBAL",
        "member_ids": sorted(
            str(member)
            for item in all_sets
            for member in list(item.get("member_ids", []) or [])
        ),
    }
    functions = dict(runtime.get("tool_functions", {}) or {})
    evidence = dict(state.get("evidence", {}) or {})
    results = list(evidence.get("results", []) or [])
    signature = partition_signature(all_sets)
    names = [
        str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")).strip()
        for call in calls
    ]
    allowed = {
        "biological_support",
        "cross_modal_consistency",
        "confounder_exclusion",
        "known_label_echo",
        "structural_adequacy",
    }
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"Router requested unknown validation capability: {unknown[0]}")
    if len(names) != len(set(names)):
        raise ValueError("Router requested the same validation capability more than once")
    cached = {
        str(result.get("capability", ""))
        for result in results
        if result.get("partition_signature") == signature
    }
    repeated = [name for name in names if name in cached]
    if repeated:
        raise ValueError(f"Validation evidence is already available for this partition: {repeated}")
    messages = list(state.get("messages", []) or [])
    messages.append(ai_message)
    for call in calls:
        name = str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")).strip()
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
        try:
            from langchain_core.messages import ToolMessage

            call_id = str(call.get("id", "") if isinstance(call, Mapping) else getattr(call, "id", ""))
            messages.append(
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=call_id or name,
                )
            )
        except Exception:
            messages.append({"role": "tool", "name": name, "content": payload})
    evidence["results"] = results
    state["evidence"] = evidence
    state["messages"] = messages[-8:]
    control = dict(state.get("control", {}) or {})
    control["round"] = int(control.get("round", 0) or 0) + 1
    control["next"] = "verify"
    state["control"] = control
    capability_names = [
        str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", ""))
        for call in calls
    ]
    append_trace(state, {"node": "router", "event": "tools", "capabilities": capability_names})
    return True


def verifier_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    payload = {
        "sets": current_sets(state),
        "evidence": state.get("evidence", {}),
        "tool_messages": serializable_messages(list(state.get("messages", []) or [])),
        "previous_audit": state.get("audit", {}),
        "round": dict(state.get("control", {}) or {}).get("round", 0),
    }
    result = invoke_with_recovery(model, payload, state, "verifier")
    if result is None:
        return state
    try:
        parsed = VerifierOutput.model_validate(result)
    except Exception as exc:
        mark_failure(state, "verifier", exc)
        return state
    state["audit"] = parsed.model_dump()
    state["messages"] = []
    mark_success(state)
    append_trace(state, {"node": "verifier", "findings": len(parsed.findings), "gaps": len(parsed.gaps)})
    control = dict(state.get("control", {}) or {})
    if complete_audit(state):
        control["status"] = "complete"
    elif int(control.get("round", 0) or 0) >= int(control.get("max_rounds", 12) or 12):
        control["status"] = "final_validation_failed"
    state["control"] = control
    return state


def complete_audit(state: Mapping[str, Any]) -> bool:
    sets = current_sets(state)
    if not sets or any(str(item.get("status")) != "provisionally_accepted" for item in sets):
        return False
    audit = dict(state.get("audit", {}) or {})
    if list(audit.get("gaps", []) or []):
        return False
    findings = [dict(item or {}) for item in list(audit.get("findings", []) or [])]
    current_ids = {set_id(item) for item in sets}
    for dimension in EVIDENCE_DIMENSIONS:
        dimension_findings = [row for row in findings if str(row.get("dimension", "")) == dimension]
        if not dimension_findings:
            return False
        covered_ids = {
            str(target)
            for row in dimension_findings
            for target in list(row.get("target_ids", []) or [])
        }
        if dimension != "known_label_echo" and not current_ids.issubset(covered_ids):
            return False
    echo_findings = [
        dict(finding or {})
        for finding in findings
        if str(dict(finding or {}).get("dimension", "")) == "known_label_echo"
    ]
    if not echo_findings:
        return False
    for finding in findings:
        row = dict(finding or {})
        dimension = str(row.get("dimension", ""))
        status = str(row.get("status", ""))
        if dimension == "known_label_echo" and status != "supporting":
            return False
        if dimension == "structural_adequacy" and status in {"conflicting", "mixed", "unavailable"}:
            return False
    return True


def verifier_route(state: dict[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    status = str(control.get("status", "reviewing"))
    if status in {"review_unavailable", "complete", "final_validation_failed"}:
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    if status == "complete":
        return "end"
    if status == "final_validation_failed":
        return "end"
    return "router"


def validate_action(action: RouterAction, state: Mapping[str, Any]) -> None:
    sets = current_sets(state)
    known = {set_id(item) for item in sets}
    if action.target_id not in known:
        raise ValueError(f"Router target is not an active set: {action.target_id}")
    target = next(item for item in sets if set_id(item) == action.target_id)
    expected_status = {
        "accept": "provisionally_accepted",
        "drop": "provisionally_dropped",
    }.get(action.action)
    if expected_status and str(target.get("status", "")) == expected_status:
        label = "accepted" if action.action == "accept" else "dropped"
        raise ValueError(f"Router target is already provisionally {label}: {action.target_id}")
    refs = all_metric_refs(dict(state.get("evidence", {}) or {}))
    missing = [ref for ref in action.metric_refs if ref not in refs]
    if missing:
        raise ValueError(f"Router referenced unavailable metrics: {missing}")


def router_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    payload = {
        "sets": current_sets(state),
        "audit": state.get("audit", {}),
        "evidence": state.get("evidence", {}),
        "control": {
            key: dict(state.get("control", {}) or {}).get(key)
            for key in ("round", "max_rounds", "blocked_actions")
        },
    }
    result = invoke_with_recovery(model, payload, state, "router")
    if result is None:
        return state
    try:
        if execute_tool_calls(state, result, runtime):
            mark_success(state)
            return state
        if isinstance(result, Mapping):
            action = parse_router_action(result)
        else:
            action = parse_router_action(parse_json_content(getattr(result, "content", result)))
        validate_action(action, state)
        control = dict(state.get("control", {}) or {})
        control["round"] = int(control.get("round", 0) or 0) + 1
        control["next"] = "revise" if action.action in {"split", "merge"} else "verify"
        state["action"] = action.model_dump()
        if action.action in {"accept", "drop"}:
            for item in state["sets"]:
                if set_id(item) == action.target_id:
                    item["status"] = f"provisionally_{action.action}ed" if action.action == "accept" else "provisionally_dropped"
        control["status"] = "reviewing"
        state["control"] = control
        mark_success(state)
        append_trace(state, {"node": "router", "action": action.model_dump()})
        return state
    except Exception as exc:
        mark_failure(state, "router", exc)
        return state


def structural_candidates(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    splits: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    for result in list(dict(state.get("evidence", {}) or {}).get("results", []) or []):
        if result.get("capability") != "structural_adequacy":
            continue
        for child in list(result.get("results", []) or []):
            metrics = dict(child.get("metrics", {}) or {})
            splits.extend(dict(x) for x in list(metrics.get("split_candidates", []) or []))
            merges.extend(dict(x) for x in list(metrics.get("merge_candidates", []) or []))
    return splits, merges


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
            options.append(
                {
                    "plan_id": "merge:" + "+".join(group),
                    "set_ids": list(group),
                    "pair_plan_ids": [pair_map[frozenset(pair)].get("plan_id", "") for pair in combinations(group, 2)],
                }
            )
    return options


def apply_split(state: dict[str, Any], plan: Mapping[str, Any]) -> None:
    source_id = str(plan.get("source_set_id", ""))
    source = next(item for item in current_sets(state) if set_id(item) == source_id)
    source_members = set(map(str, source.get("member_ids", [])))
    groups = [set(map(str, group)) for group in list(plan.get("groups", []) or [])]
    if len(groups) < 2 or set().union(*groups) != source_members or sum(map(len, groups)) != len(source_members):
        raise ValueError("Split plan does not partition the source set exactly")
    source["status"] = "retired"
    for index, group in enumerate(groups, start=1):
        state["sets"].append(
            {
                "set_id": f"{source_id}_S{index}",
                "cluster_id": f"{source_id}_S{index}",
                "member_ids": sorted(group),
                "status": "active",
                "parent_ids": [source_id],
            }
        )


def apply_merge(state: dict[str, Any], plan: Mapping[str, Any]) -> None:
    ids = [str(x) for x in list(plan.get("set_ids", []) or [])]
    if len(ids) < 2:
        raise ValueError("Merge plan must contain at least two sets")
    selected = [item for item in current_sets(state) if set_id(item) in ids]
    if len(selected) != len(ids):
        raise ValueError("Merge plan contains an inactive set")
    members = sorted({str(x) for item in selected for x in item.get("member_ids", [])})
    for item in selected:
        item["status"] = "retired"
    merged_id = "_M_".join(ids)
    state["sets"].append(
        {
            "set_id": merged_id,
            "cluster_id": merged_id,
            "member_ids": members,
            "status": "active",
            "parent_ids": ids,
        }
    )


def reviser_node(state: dict[str, Any], runtime: Mapping[str, Any], model: Any) -> dict[str, Any]:
    action = dict(state.get("action", {}) or {})
    splits, merges = structural_candidates(state)
    target = str(action.get("target_id", ""))
    candidates = (
        [row for row in splits if str(row.get("source_set_id", "")) == target]
        if action.get("action") == "split"
        else merge_options(target, merges, current_sets(state))
    )
    payload = {"action": action, "candidates": candidates, "audit": state.get("audit", {})}
    result = invoke_with_recovery(model, payload, state, "reviser")
    if result is None:
        return state
    try:
        parsed = ReviserOutput.model_validate(result)
        plan_id = parsed.plan_id
        if not plan_id:
            blocked = list(dict(state["control"]).get("blocked_actions", []) or [])
            blocked.append(f"{action.get('action')}:{target}")
            state["control"]["blocked_actions"] = sorted(set(blocked))
            state["action"] = None
            state["control"]["next"] = "verify"
            mark_success(state)
            append_trace(state, {"node": "reviser", "plan_id": None, "reason": parsed.reason})
            return state
        plan = next((row for row in candidates if str(row.get("plan_id", "")) == plan_id), None)
        if plan is None:
            raise ValueError(f"Reviser selected unknown plan: {plan_id}")
        refs = all_metric_refs(dict(state.get("evidence", {}) or {}))
        if any(ref not in refs for ref in parsed.metric_refs):
            raise ValueError("Reviser referenced unavailable metrics")
        if action.get("action") == "split":
            apply_split(state, plan)
        else:
            set_ids = tuple(sorted(str(x) for x in list(plan.get("set_ids", []) or [])))
            if len(set_ids) < 2:
                raise ValueError("Invalid Merge candidate")
            for pair in combinations(set_ids, 2):
                if not any(
                    frozenset(map(str, row.get("set_ids", []))) == frozenset(pair)
                    for row in merges
                ):
                    raise ValueError("Merge plan lacks complete pairwise evidence")
            apply_merge(state, plan)
        state["action"] = None
        state["control"]["next"] = "verify"
        mark_success(state)
        append_trace(state, {"node": "reviser", "plan_id": plan_id, "reason": parsed.reason})
        return state
    except Exception as exc:
        mark_failure(state, "reviser", exc)
        return state


def route_after_router(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") == "review_unavailable":
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    return str(control.get("next", "verify"))


def route_after_reviser(state: Mapping[str, Any]) -> str:
    control = dict(state.get("control", {}) or {})
    if control.get("status") == "review_unavailable":
        return "end"
    if control.get("error") and int(control.get("failures", 0) or 0) > 0:
        return "retry"
    return "verify"


def build_review_graph(
    *,
    verifier_model: Any,
    router_model: Any,
    reviser_model: Any,
    tool_functions: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
) -> Any:
    base_runtime = dict(runtime or {})

    def verifier(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        return verifier_node(state, merge_runtime(base_runtime, runtime), verifier_model)

    def router(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        merged = merge_runtime(base_runtime, runtime)
        merged.setdefault("tool_functions", dict(tool_functions))
        return router_node(state, merged, router_model)

    def reviser(state: dict[str, Any], runtime: Runtime[ReviewContext]) -> dict[str, Any]:
        return reviser_node(state, merge_runtime(base_runtime, runtime), reviser_model)

    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    graph.add_node("verifier", verifier)
    graph.add_node("router", router)
    graph.add_node("reviser", reviser)
    graph.add_edge(START, "verifier")
    graph.add_conditional_edges("verifier", verifier_route, {"router": "router", "retry": "verifier", "end": END})
    graph.add_conditional_edges("router", route_after_router, {"verify": "verifier", "revise": "reviser", "retry": "router", "end": END})
    graph.add_conditional_edges("reviser", route_after_reviser, {"verify": "verifier", "retry": "reviser", "end": END})
    return graph.compile()


def save_review_outputs(state: Mapping[str, Any], output_root: str) -> dict[str, Any]:
    root = Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    control = dict(state.get("control", {}) or {})
    sets = current_sets(state)
    final_sets = [item for item in sets if item.get("status") == "provisionally_accepted"]
    summary = {
        "stage": "subtype_review",
        "status": control.get("status", "review_unavailable"),
        "rounds_used": control.get("round", 0),
        "sets": to_jsonable(sets),
        "final_sets": to_jsonable(final_sets),
        "final_patient_count": len({member for item in final_sets for member in item.get("member_ids", [])}),
        "audit": to_jsonable(state.get("audit", {})),
        "decision_trace": to_jsonable(control.get("trace", [])),
    }
    (root / "final_subtype_sets.json").write_text(json.dumps(final_sets, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "evidence_audit.json").write_text(json.dumps(to_jsonable(state.get("audit", {})), ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "decision_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in control.get("trace", []):
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    (root / "final_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
