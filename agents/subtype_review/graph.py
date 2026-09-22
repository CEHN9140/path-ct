from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from sklearn.cluster import SpectralClustering

from agents.subtype_review.llm import parse_json_content, summarize_reports
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS,
    EvidenceReportBatch,
    EvidenceRequest,
    ReviewContext,
    ReviewState,
    RevisionPlan,
    RouterAction,
    RouterPlan,
    set_id,
)
from agents.subtype_review.runtime_trace import append_runtime_trace, partition_snapshot
from agents.subtype_review.evidence_semantics import EVIDENCE_ROLE_CONTRACTS, guidance_for
from agents.subtype_review.tools import (
    TOOL_REGISTRY,
    candidate_consensus_geometry,
    compact_tool_result,
)
from utils.llm_utils import load_yaml_file
from utils.tool_utils import to_jsonable


def partition_signature(sets: list[dict[str, Any]]) -> str:
    payload = [
        {"set_id": set_id(item), "member_ids": sorted(map(str, item["member_ids"]))}
        for item in sorted(sets, key=set_id)
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def current_sets(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(state["partition"]["sets"], key=set_id)


def agent_partition_context(partition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sets": [
            {key: copy.deepcopy(value) for key, value in item.items() if key != "generator"}
            for item in partition.get("sets", [])
        ]
    }


def evidence_report_ref(signature: str, dimension: str, aspect: str, scope: str,
                        targets: tuple[str, ...]) -> str:
    partition_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    target_hash = hashlib.sha256("\0".join(targets).encode("utf-8")).hexdigest()[:10]
    return f"ER:{partition_hash}:{dimension}:{aspect}:{scope}:{target_hash}"


def evidence_request_ref(signature: str, request: EvidenceRequest) -> str:
    payload = {
        "dimension": request.dimension,
        "scope": request.scope,
        "target_ids": request.target_ids,
        "focus": request.focus,
        "question": request.question,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    partition_hash = hashlib.sha256(signature.encode()).hexdigest()[:10]
    return f"RQ:{partition_hash}:{digest}"


def latest_acquisition_context(
    state: Mapping[str, Any],
    signature: str,
) -> dict[str, Any]:
    if not state["history"]:
        return {"source_round": None, "evidence_requests": [], "new_reports": []}

    previous = state["history"][-1]
    if previous["partition_signature"] != signature:
        return {"source_round": None, "evidence_requests": [], "new_reports": []}

    plan = RouterPlan.model_validate(previous["router_plan"])
    if not plan.evidence_requests:
        return {"source_round": None, "evidence_requests": [], "new_reports": []}

    requested = {
        (item.dimension, item.scope, tuple(item.target_ids))
        for item in plan.evidence_requests
    }
    new_refs = {
        evidence_report_ref(
            signature, row["dimension"], row["aspect"], row["scope"], tuple(row["target_ids"]),
        )
        for row in state["round_evidence"]
        if row["partition_signature"] == signature
        and (row["dimension"], row["scope"], tuple(row["target_ids"])) in requested
    }
    return {
        "source_round": previous["round"],
        "evidence_requests": [item.model_dump() for item in plan.evidence_requests],
        "new_reports": [row for row in state["reports"] if row["report_ref"] in new_refs],
    }


def terminal_closure_refs(
    current: list[dict[str, Any]],
    new_reports: list[Mapping[str, Any]],
) -> dict[str, set[str]]:
    required = {set_id(item): set() for item in current}
    for report in new_reports:
        targets = required if report["scope"] == "partition" else report["target_ids"]
        for target in targets:
            if target in required:
                required[target].add(report["report_ref"])
    return required


def terminal_accountability_refs(
    current: list[dict[str, Any]],
    reports: list[Mapping[str, Any]],
) -> dict[str, set[str]]:
    required = {set_id(item): set() for item in current}
    for report in reports:
        scope = report.get("scope")
        targets = list(report.get("target_ids", []))
        ref = str(report.get("report_ref", ""))
        if not ref:
            continue
        if scope == "set" and len(targets) == 1:
            if targets[0] in required:
                required[targets[0]].add(ref)
        elif scope == "pair" and len(targets) == 2:
            for target in targets:
                if target in required:
                    required[target].add(ref)
    return required


def build_pair_review_status(
    reports: list[Mapping[str, Any]],
    available: set[tuple[str, str, tuple[str, ...], str]],
) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        if (
            report.get("dimension") != "cross_modal_consistency"
            or report.get("scope") != "pair"
        ):
            continue
        targets = tuple(sorted(map(str, report.get("target_ids", []))))
        if len(targets) != 2:
            continue
        item = pairs.setdefault(targets, {
            "target_ids": list(targets),
            "boundary_representation_report_ref": None,
            "boundary_structure_report_ref": None,
        })
        foci = report.get("request_foci", [])
        if "boundary_representation" in foci:
            item["boundary_representation_report_ref"] = report.get("report_ref")
        if "boundary_structure" in foci:
            item["boundary_structure_report_ref"] = report.get("report_ref")
    for targets, item in pairs.items():
        item["boundary_structure_available"] = (
            "cross_modal_consistency", "pair", targets, "boundary_structure"
        ) in available
    return [pairs[key] for key in sorted(pairs)]


def structural_pair_followup_targets(
    state: Mapping[str, Any],
    signature: str,
) -> set[tuple[str, str]]:
    screen = next((row for row in state["tool_evidence"]
                   if row["partition_signature"] == signature
                   and row["tool_name"] == "structural_diagnostics"
                   and row["scope"] == "partition"), None)
    pairs = (screen or {}).get("metrics", {}).get("partition", {}).get("nearest_pair_targets", [])
    return {tuple(sorted(map(str, pair))) for pair in pairs if len(pair) == 2}


def reports_for_request(
    reports: list[Mapping[str, Any]], request: EvidenceRequest,
    registry: Mapping[str, Mapping[str, Any]] = TOOL_REGISTRY,
) -> list[dict[str, Any]]:
    expected_aspects = {
        metadata.get("aspect", name)
        for name, metadata in registry.items()
        if request.focus in metadata.get("question_foci", {}).get(request.scope, ())
    }
    return [
        {key: copy.deepcopy(value) for key, value in report.items() if key != "metric_refs"}
        for report in reports
        if report["dimension"] == request.dimension
        and report["scope"] == request.scope
        and sorted(map(str, report["target_ids"])) == request.target_ids
        and report.get("aspect") in expected_aspects
    ]


def eligible_tools_for_request(
    request: EvidenceRequest,
    registry: Mapping[str, Mapping[str, Any]],
    completed: set[tuple[str, str, tuple[str, ...]]],
    attempted: set[str],
    *,
    partition_screen_done: bool,
) -> list[str]:
    targets = tuple(request.target_ids)
    return sorted(
        name for name, metadata in registry.items()
        if name not in attempted
        and (name, request.scope, targets) not in completed
        and metadata["dimension"] == request.dimension
        and request.focus in metadata.get("question_foci", {}).get(request.scope, ())
        and request.scope in metadata["scopes"]
        and not (name == "structural_diagnostics" and (
            (request.scope == "partition" and partition_screen_done)
            or (request.scope in {"set", "pair"} and not partition_screen_done)
        ))
    )


def initial_review_state(candidate_sets: list[dict[str, Any]]) -> ReviewState:
    sets = [
        {
            "set_id": set_id(item),
            "member_ids": sorted(map(str, item["member_ids"])),
            "revision_lineage": list(item.get("revision_lineage", [])),
            "generator": copy.deepcopy(item.get("generator", {})),
        }
        for item in candidate_sets
    ]
    identifiers = [set_id(item) for item in sets]
    members = [member for item in sets for member in item["member_ids"]]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Initial candidate sets require unique identifiers")
    if len(members) != len(set(members)):
        raise ValueError("Initial candidate sets cannot overlap")
    return {
        "partition": {"sets": sets},
        "round_evidence": [],
        "tool_evidence": [],
        "reports": [],
        "evidence_memory": {},
        "router_plan": None,
        "revision_plan": None,
        "revision_result": None,
        "history": [],
        "control": {
            "round": 0,
            "status": "reviewing",
            "next": "router",
            "max_rounds": 10,
            "pending_evidence_requests": [],
            "trace": [],
        },
    }


def validate_router_plan(
    plan: RouterPlan,
    state: Mapping[str, Any],
    available: set[tuple[str, str, tuple[str, ...], str]],
    required_terminal_refs: Mapping[str, set[str]] | None = None,
    terminal_accountability_refs_by_target: Mapping[str, set[str]] | None = None,
    merge_legal: bool = True,
) -> None:
    current = {set_id(item) for item in current_sets(state)}
    if plan.evidence_requests:
        seen = set()
        for request in plan.evidence_requests:
            key = (request.dimension, request.scope, tuple(request.target_ids), request.focus)
            if key not in available:
                raise ValueError(f"EvidenceRequest is unavailable: {key}")
            if key in seen:
                raise ValueError(f"EvidenceRequest is duplicated: {key}")
            seen.add(key)
        return

    reports = {row["report_ref"]: row for row in state["reports"]}
    structural = [action for action in plan.actions if action.action in {"split", "merge"}]
    for action in plan.actions:
        if not set(action.target_ids).issubset(current):
            raise ValueError("Action targets must be current sets")
        cited = [reports.get(ref) for ref in action.evidence_report_refs]
        if any(row is None for row in cited):
            raise ValueError("Action cites an unknown or non-current Evidence Report")
        if any(not (row["scope"] == "partition" or set(action.target_ids) & set(row["target_ids"])) for row in cited):
            raise ValueError("Action cites an Evidence Report unrelated to its targets")
        required = set().union(*(
            (required_terminal_refs or {}).get(target, set()) for target in action.target_ids
        ))
        missing = required - set(action.evidence_report_refs)
        if missing:
            raise ValueError(
                "Action does not close the latest Router-requested evidence for "
                f"{', '.join(action.target_ids)}: missing report refs {sorted(missing)}"
            )
    if structural:
        if len(plan.actions) != 1 or len(structural) != 1:
            actions = [item.action for item in plan.actions]
            raise ValueError(
                "A revision round must contain exactly one action, and that action must be one "
                f"split or one merge; received actions={actions}"
            )
        action = structural[0]
        if action.action == "split":
            target = action.target_ids[0]
            report = next((row for row in cited if row["dimension"] == "cross_modal_consistency"
                           and row["aspect"] == "structural_diagnostics" and row["scope"] == "set"
                           and row["target_ids"] == [target]), None)
            if report is None:
                raise ValueError("Split requires its exact-set structural Evidence Report")
            structural_result = next((row for row in state["tool_evidence"]
                if row["partition_signature"] == partition_signature(current_sets(state))
                and row["tool_name"] == "structural_diagnostics" and row["scope"] == "set"
                and row["target_ids"] == [target]), None)
            solutions = structural_result["metrics"].get("set", {}).get(target, {}).get("solutions", {}) if structural_result else {}
            if str(action.n_children) not in solutions:
                raise ValueError("Split child count must be a feasible structural solution")
        else:
            targets = sorted(action.target_ids)
            report = next((row for row in cited if row["dimension"] == "cross_modal_consistency"
                           and row["aspect"] == "structural_diagnostics" and row["scope"] == "pair"
                           and row["target_ids"] == targets), None)
            if report is None:
                raise ValueError("Merge requires its exact-pair structural Evidence Report")
            if len(current) - 1 < 2:
                raise ValueError("Merge cannot collapse the subtype partition to a single whole-cohort set")
        return

    targets = [target for action in plan.actions for target in action.target_ids]
    if any(action.action not in {"accept", "drop"} for action in plan.actions):
        raise ValueError("Terminal disposition may only accept or drop sets")
    if len(targets) != len(set(targets)) or set(targets) != current:
        raise ValueError("Terminal actions must cover each current set exactly once")
    if not any(
        row["dimension"] == "cross_modal_consistency"
        and row["aspect"] == "structural_diagnostics"
        and row["scope"] == "partition"
        and not row["target_ids"]
        for row in state["reports"]
    ):
        raise ValueError("Terminal disposition requires a partition structural screen")

    for action in plan.actions:
        if not action.evidence_report_refs:
            raise ValueError("Every terminal action must cite at least one Evidence Report")
        required_refs = set((terminal_accountability_refs_by_target or {}).get(
            action.target_ids[0], set()
        ))
        missing_refs = required_refs - set(action.evidence_report_refs)
        if missing_refs:
            raise ValueError(
                "Terminal action must cite all target-specific set/pair Evidence Reports "
                f"acquired for this candidate; missing {sorted(missing_refs)}"
            )
        if action.action == "accept":
            target = action.target_ids[0]
            if not any(
                is_set_membership_report(reports.get(ref), target)
                for ref in action.evidence_report_refs
            ):
                raise ValueError(
                    "Accept requires the target's exact-set membership_representation "
                    "Evidence Report; pair-boundary, structural, biological, confounder, "
                    "or partition evidence cannot substitute for it."
                )
            if not any(
                is_set_biological_support_report(reports.get(ref), target)
                for ref in action.evidence_report_refs
            ):
                raise ValueError(
                    "Accept requires a target-specific set-scope biological_support "
                    "Evidence Report establishing an interpretable candidate identity."
                )


def is_set_membership_report(report: Mapping[str, Any] | None, target: str) -> bool:
    return bool(
        report
        and report.get("dimension") == "cross_modal_consistency"
        and report.get("scope") == "set"
        and list(report.get("target_ids", [])) == [target]
        and "membership_representation" in report.get("request_foci", [])
    )


def is_set_biological_support_report(
    report: Mapping[str, Any] | None, target: str,
) -> bool:
    return bool(
        report
        and report.get("dimension") == "biological_support"
        and report.get("scope") == "set"
        and list(report.get("target_ids", [])) == [target]
    )


def has_pair_boundary_representation(
    reports: list[Mapping[str, Any]], targets: list[str],
) -> bool:
    targets = sorted(map(str, targets))
    return any(
        row.get("dimension") == "cross_modal_consistency"
        and row.get("scope") == "pair"
        and sorted(map(str, row.get("target_ids", []))) == targets
        and "boundary_representation" in row.get("request_foci", [])
        for row in reports
    )


def router_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    values = runtime.context if isinstance(runtime, Runtime) else runtime
    control = dict(state["control"])
    current = current_sets(state)
    signature = partition_signature(current)
    completed = {
        (row["tool_name"], row["scope"], tuple(row["target_ids"]))
        for row in state["tool_evidence"] if row["partition_signature"] == signature
    }
    partition_screen_done = ("structural_diagnostics", "partition", ()) in completed
    structural_pairs = structural_pair_followup_targets(state, signature)
    set_ids = [set_id(item) for item in current]
    available_aspects = {}
    for name, metadata in values.get("tool_registry", TOOL_REGISTRY).items():
        dimension = metadata["dimension"]
        for scope in metadata["scopes"]:
            if scope == "partition":
                targets = [()]
            elif scope == "set":
                targets = [(item,) for item in set_ids]
            else:
                targets = list(combinations(set_ids, 2))
                if dimension == "cross_modal_consistency" and partition_screen_done:
                    targets = [pair for pair in targets if pair in structural_pairs]
            for target_ids in targets:
                if name == "structural_diagnostics" and (
                    (scope == "partition" and partition_screen_done)
                    or (scope in {"set", "pair"} and not partition_screen_done)
                ):
                    continue
                if (
                    scope == "pair"
                    and "boundary_structure" in metadata.get("question_foci", {}).get(scope, ())
                    and not has_pair_boundary_representation(state["reports"], list(target_ids))
                ):
                    continue
                key = (name, scope, tuple(target_ids))
                if key not in completed:
                    for focus in metadata.get("question_foci", {}).get(scope, ()):
                        request_key = (dimension, scope, tuple(target_ids), focus)
                        available_aspects.setdefault(request_key, set()).add(metadata.get("aspect", name))

    available = set(available_aspects)
    option_groups: dict[tuple[str, str, tuple[str, ...]], dict[str, set[str]]] = {}
    for (dimension, scope, target_ids, focus), aspects in available_aspects.items():
        group = option_groups.setdefault(
            (dimension, scope, target_ids), {"aspects": set(), "foci": set()}
        )
        group["aspects"].update(aspects)
        group["foci"].add(focus)
    request_options = [
        {
            "dimension": dimension,
            "scope": scope,
            "target_ids": list(target_ids),
            "available_aspects": sorted(group["aspects"]),
            "available_question_foci": sorted(group["foci"]),
        }
        for (dimension, scope, target_ids), group in sorted(option_groups.items())
    ]
    router_request_options = [
        {key: item[key] for key in ("dimension", "scope", "target_ids", "available_question_foci")}
        for item in request_options
    ]
    coverage = {
        "set": {item: {dimension: "unassessed" for dimension in EVIDENCE_DIMENSIONS} for item in set_ids},
        "pair": {},
        "partition": {dimension: "unassessed" for dimension in EVIDENCE_DIMENSIONS},
    }
    for report in state["reports"]:
        scope = report["scope"]
        if scope == "partition":
            coverage["partition"][report["dimension"]] = "assessed"
            continue
        target = "|".join(report["target_ids"]) if scope == "pair" else (
            report["target_ids"][0]
        )
        coverage[scope].setdefault(target, {})[report["dimension"]] = "assessed"
    latest_acquisition = latest_acquisition_context(state, signature)
    required_closure = terminal_closure_refs(current, latest_acquisition["new_reports"])
    terminal_accountability = terminal_accountability_refs(current, state["reports"])
    closure_payload = {
        "source_round": latest_acquisition["source_round"],
        "previous_evidence_requests": latest_acquisition["evidence_requests"],
        "new_report_refs": sorted(row["report_ref"] for row in latest_acquisition["new_reports"]),
        "required_terminal_report_refs_by_target": {
            target: sorted(refs) for target, refs in required_closure.items()
        },
    }
    budget_exhausted = control["round"] >= control["max_rounds"]
    merge_legal = len(current) > 2
    workflow_constraints = {
        "allowed_structural_actions": ["split", "merge"] if merge_legal else ["split"],
        "forbidden_structural_actions": [] if merge_legal else [{
            "action": "merge",
            "reason": "This workflow does not permit a merge that leaves fewer than two current sets.",
        }],
    }
    if not budget_exhausted:
        control["round"] += 1
    payload = {
        "partition": agent_partition_context(state["partition"]),
        "evidence_reports": summarize_reports(state["reports"]),
        "evidence_coverage": coverage,
        "evidence_dimension_contracts": copy.deepcopy(EVIDENCE_ROLE_CONTRACTS),
        "available_evidence_requests": router_request_options,
        "latest_acquisition_closure": closure_payload,
        "terminal_accountability_report_refs_by_target": {
            target: sorted(refs) for target, refs in terminal_accountability.items()
        },
        "pair_review_status": build_pair_review_status(state["reports"], available),
        "workflow_constraints": workflow_constraints,
        "round": control["round"],
        "budget_exhausted": budget_exhausted,
    }
    append_runtime_trace(
        values.get("runtime_trace_path"),
        node="router",
        event="router_context",
        round_id=control["round"],
        payload={
            "partition": partition_snapshot(current),
            "evidence_reports": payload["evidence_reports"],
            "evidence_coverage": coverage,
            "evidence_dimension_contracts": payload["evidence_dimension_contracts"],
            "available_evidence_requests": router_request_options,
            "latest_acquisition_closure": closure_payload,
            "terminal_accountability_report_refs_by_target": {
                target: sorted(refs) for target, refs in terminal_accountability.items()
            },
            "pair_review_status": build_pair_review_status(state["reports"], available),
            "workflow_constraints": workflow_constraints,
            "budget_exhausted": budget_exhausted,
        },
    )
    plan_source = "llm_router" if partition_screen_done else "protocol_mandatory_partition_screen"
    if partition_screen_done:
        model = values["router_model"]
        retries = int(getattr(model, "config", {}).get("router_plan_validation_retries", 0))
        validation_feedback = None
        for attempt in range(retries + 1):
            request_payload = copy.deepcopy(payload)
            if validation_feedback is not None:
                request_payload["validation_feedback"] = validation_feedback
            plan = RouterPlan.model_validate(parse_json_content(model.invoke(request_payload)))
            try:
                validate_router_plan(
                    plan, state, available,
                    required_terminal_refs=required_closure,
                    terminal_accountability_refs_by_target=terminal_accountability,
                    merge_legal=merge_legal,
                )
                break
            except ValueError as exc:
                if attempt >= retries:
                    raise
                invalid_requests = [
                    {
                        "dimension": request.dimension,
                        "scope": request.scope,
                        "target_ids": list(request.target_ids),
                        "focus": request.focus,
                        "reason": "No currently eligible evidence tool remains for this request."
                        if (request.dimension, request.scope, tuple(request.target_ids), request.focus) not in available
                        else "This evidence request is duplicated within the RouterPlan.",
                    }
                    for request in plan.evidence_requests
                    if (request.dimension, request.scope, tuple(request.target_ids), request.focus) not in available
                    or sum(
                        1 for other in plan.evidence_requests
                        if (
                            other.dimension,
                            other.scope,
                            tuple(other.target_ids),
                            other.focus,
                        ) == (
                            request.dimension,
                            request.scope,
                            tuple(request.target_ids),
                            request.focus,
                        )
                    ) > 1
                ]
                feedback = {
                    "error": str(exc),
                    "invalid_evidence_requests": invalid_requests,
                    "available_evidence_requests": router_request_options,
                    "instruction": (
                        "Repair the RouterPlan according to the validation error. Use only "
                        "available_evidence_requests and current Evidence Reports. Do not preserve "
                        "an action when the validation error shows that its required evidential role "
                        "is missing. In that case, reconsider the action or request decision-relevant "
                        "evidence if an eligible request remains. Do not invent evidence. "
                        "Do not add new evidence requests merely to complete coverage. Prefer the "
                        "smallest evidence set sufficient to resolve the current scientific decision."
                    ),
                    "required_terminal_report_refs_by_target": closure_payload[
                        "required_terminal_report_refs_by_target"
                    ],
                }
                if "Terminal action must cite all target-specific" in str(exc):
                    feedback["instruction"] = (
                        "Terminal evidence_report_refs must include every target-specific report "
                        "listed in terminal_accountability_report_refs_by_target. Do not change "
                        "the scientific action solely because a required reference was omitted."
                    )
                append_runtime_trace(
                    values.get("runtime_trace_path"),
                    node="router",
                    event="router_plan_validation_retry",
                    round_id=control["round"],
                    payload={
                        "attempt": attempt + 1,
                        "error": str(exc),
                        "invalid_evidence_requests": invalid_requests,
                        "remaining_available_request_count": len(router_request_options),
                        "remaining_available_requests": router_request_options,
                    },
                )
                validation_feedback = feedback
    else:
        if not any(
            key[:3] == ("cross_modal_consistency", "partition", ())
            and key[3] == "partition_structural_screen"
            for key in available
        ):
            raise ValueError("Structural diagnostics must support a partition-level screen")
        plan = RouterPlan(evidence_requests=[EvidenceRequest(
            dimension="cross_modal_consistency",
            scope="partition",
            target_ids=[],
            focus="partition_structural_screen",
            question="Screen the current partition for unsupported internal splits and weak pair boundaries.",
        )])
        validate_router_plan(
            plan, state, available,
            required_terminal_refs=required_closure,
            terminal_accountability_refs_by_target=terminal_accountability,
            merge_legal=merge_legal,
        )
    append_runtime_trace(
        values.get("runtime_trace_path"),
        node="router",
        event="router_plan",
        round_id=control["round"],
        payload={
            "partition": partition_snapshot(current),
            "plan_source": plan_source,
            "plan": plan.model_dump(),
        },
    )
    state["router_plan"] = plan.model_dump()
    state["history"].append({
        "round": control["round"],
        "partition_signature": signature,
        "partition": copy.deepcopy(state["partition"]),
        "evidence_reports": copy.deepcopy(state["reports"]),
        "router_plan": plan.model_dump(),
    })
    control["trace"].append({
        "round": control["round"], "node": "router", "event": "decision",
        "plan": plan.model_dump(),
    })
    control["pending_evidence_requests"] = [item.model_dump() for item in plan.evidence_requests]
    needs_more_work = bool(plan.evidence_requests) or any(
        item.action in {"split", "merge"} for item in plan.actions
    )
    if budget_exhausted and needs_more_work:
        control["status"] = "incomplete_due_to_round_budget"
        control["next"] = "end"
        control["trace"].append({
            "round": control["round"],
            "node": "router",
            "event": "round_budget_exhausted",
            "pending_evidence_requests": control["pending_evidence_requests"],
            "proposed_actions": [item.model_dump() for item in plan.actions],
        })
    elif plan.evidence_requests:
        control["next"] = "verifier"
    elif any(item.action in {"split", "merge"} for item in plan.actions):
        control["next"] = "reviser"
    else:
        control["status"] = "complete"
        control["next"] = "end"
    state["control"] = control
    return {"control": control, "router_plan": state["router_plan"], "history": state["history"]}


def verifier_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    values = runtime.context if isinstance(runtime, Runtime) else runtime
    requests = [EvidenceRequest.model_validate(item) for item in state["control"]["pending_evidence_requests"]]
    current = current_sets(state)
    signature = partition_signature(current)
    registry = values.get("tool_registry", TOOL_REGISTRY)
    completed = {
        (row["tool_name"], row["scope"], tuple(row["target_ids"]))
        for row in state["tool_evidence"] if row["partition_signature"] == signature
    }
    screen = next((row for row in state["tool_evidence"]
                   if row["partition_signature"] == signature
                   and row["tool_name"] == "structural_diagnostics"
                   and row["scope"] == "partition"), None)
    partition_screen_done = screen is not None
    mandatory_partition_screen = (
        screen is None
        and len(requests) == 1
        and requests[0].dimension == "cross_modal_consistency"
        and requests[0].scope == "partition"
        and not requests[0].target_ids
    )
    if mandatory_partition_screen:
        if "structural_diagnostics" not in registry:
            raise ValueError("Mandatory partition structural screen requires structural_diagnostics to be eligible")
    request_states = {}
    for request in requests:
        ref = evidence_request_ref(signature, request)
        if ref in request_states:
            raise ValueError(f"Duplicate EvidenceRequest identity: {ref}")
        request_states[ref] = {
            "request": request,
            "attempted_tools": set(),
            "stopped": False,
            "stop_reason": "",
        }

    working_reports = copy.deepcopy(state["reports"])
    working_tool_evidence = copy.deepcopy(state["tool_evidence"])
    new_rows_all = []
    wave = 0
    while any(not item["stopped"] for item in request_states.values()):
        wave += 1
        active_refs = [ref for ref, item in sorted(request_states.items()) if not item["stopped"]]
        report_snapshot = copy.deepcopy(working_reports)
        selected = {}
        scheduled = set(completed)
        append_runtime_trace(
            values.get("runtime_trace_path"), node="verifier",
            event="verifier_acquisition_wave_start", round_id=state["control"]["round"],
            payload={"wave": wave, "request_refs": active_refs},
        )

        for ref in active_refs:
            request_state = request_states[ref]
            request = request_state["request"]
            remaining = eligible_tools_for_request(
                request, registry, scheduled, request_state["attempted_tools"],
                partition_screen_done=partition_screen_done,
            )
            if mandatory_partition_screen:
                remaining = [name for name in remaining if name == "structural_diagnostics"]
                if not remaining:
                    raise ValueError("Mandatory partition structural screen requires structural_diagnostics to be eligible")
                selected_tool = "structural_diagnostics"
                require_tool = True
            elif not remaining:
                if not request_state["attempted_tools"]:
                    raise ValueError(
                        f"No unrun scientific tool can answer EvidenceRequest {ref}"
                    )
                request_state.update(stopped=True, stop_reason="eligible_tools_exhausted")
                append_runtime_trace(
                    values.get("runtime_trace_path"), node="verifier",
                    event="verifier_request_stopped", round_id=state["control"]["round"],
                    payload={"wave": wave, "request_ref": ref, "remaining_tools": [],
                             "stop_reason": request_state["stop_reason"]},
                )
                continue
            else:
                require_tool = not request_state["attempted_tools"]
                selection_payload = {
                    "mode": "select",
                    "round": state["control"]["round"],
                    "wave": wave,
                    "partition": {"sets": [
                        {"set_id": set_id(item), "member_n": len(item["member_ids"])}
                        for item in current
                    ]},
                    "evidence_request": request.model_dump(),
                    "current_evidence": reports_for_request(report_snapshot, request, registry),
                    "attempted_tools": sorted(request_state["attempted_tools"]),
                    "remaining_tools": remaining,
                    "require_tool": require_tool,
                }
                decision = values["verifier_model"].invoke(selection_payload)
                selected_tool = decision.get("selected_tool") if isinstance(decision, Mapping) else None
                if selected_tool is None:
                    if require_tool:
                        raise RuntimeError(f"Verifier stopped before acquiring evidence for request {ref}")
                    request_state.update(
                        stopped=True,
                        stop_reason=str(decision.get("stop_reason", "Verifier stopped evidence acquisition.")),
                    )
                    append_runtime_trace(
                        values.get("runtime_trace_path"), node="verifier",
                        event="verifier_request_stopped", round_id=state["control"]["round"],
                        payload={"wave": wave, "request_ref": ref, "remaining_tools": remaining,
                                 "stop_reason": request_state["stop_reason"]},
                    )
                    continue
                if selected_tool not in remaining:
                    raise ValueError(f"Verifier selected an ineligible tool: {selected_tool}")

            key = (selected_tool, request.scope, tuple(request.target_ids))
            if key in scheduled:
                raise ValueError(f"Verifier selected a tool already scheduled this partition: {key}")
            scheduled.add(key)
            request_state["attempted_tools"].add(selected_tool)
            selected[ref] = selected_tool
            append_runtime_trace(
                values.get("runtime_trace_path"), node="verifier",
                event="verifier_tool_selection", round_id=state["control"]["round"],
                payload={"wave": wave, "request_ref": ref,
                         "dimension": request.dimension, "scope": request.scope,
                         "target_ids": request.target_ids, "focus": request.focus,
                         "remaining_tools": remaining,
                         "selected_tool": selected_tool},
            )

        if not selected:
            break

        new_rows_wave = []
        expected_reports: dict[tuple[str, str, str, tuple[str, ...]], set[str]] = {}
        report_foci: dict[tuple[str, str, str, tuple[str, ...]], set[str]] = {}
        for ref, name in selected.items():
            request = request_states[ref]["request"]
            metadata = registry[name]
            scope, targets = request.scope, tuple(request.target_ids)
            try:
                raw = metadata["function"](
                    patient_states_by_id=values["patient_states_by_id"],
                    output_root=str(values["data_root"]),
                    config_dir=str(values["config_dir"]),
                    all_cluster_states=current,
                    scope=scope,
                    target_ids=list(targets),
                )
            except Exception as exc:
                append_runtime_trace(
                    values.get("runtime_trace_path"), node="verifier", event="tool_failed",
                    round_id=state["control"]["round"],
                    payload={"request_ref": ref, "tool_name": name, "scope": scope,
                             "target_ids": list(targets), "partition": partition_snapshot(current),
                             "error_type": type(exc).__name__, "error_message": str(exc)},
                )
                raise
            completed.add((name, scope, targets))
            row = compact_tool_result(raw, name)
            if row["status"] == "runtime_failure":
                raise RuntimeError(f"Scientific tool {name} failed: {row['errors']}")
            aspect = metadata.get("aspect", name)
            row.update({
                "dimension": metadata["dimension"], "aspect": aspect, "scope": scope,
                "target_ids": list(targets), "partition_signature": signature,
                "request_ref": ref, "focus": request.focus,
            })
            new_rows_wave.append(row)
            report_targets = [()] if scope == "partition" else [(target,) for target in targets] if scope == "set" else [targets]
            for report_target in report_targets:
                expected_reports.setdefault(
                    (metadata["dimension"], aspect, scope, report_target), set()
                ).add(name)
                report_foci.setdefault(
                    (metadata["dimension"], aspect, scope, report_target), set()
                ).add(request.focus)
            append_runtime_trace(
                values.get("runtime_trace_path"), node="verifier", event="tool_result",
                round_id=state["control"]["round"],
                payload={"wave": wave, "request_ref": ref, "tool_name": name,
                         "scope": scope, "target_ids": list(targets), "result": row},
            )

        audit_payload = {
            "mode": "audit",
            "partition": {"sets": [{"set_id": set_id(item), "member_n": len(item["member_ids"])} for item in current]},
            "required_reports": [
                {"dimension": dimension, "aspect": aspect, "scope": scope,
                 "target_ids": list(targets), "tool_refs": sorted(names),
                 "evidence_guidance": guidance_for(dimension, aspect, scope)}
                for (dimension, aspect, scope, targets), names in sorted(expected_reports.items())
            ],
            "prior_reports": summarize_reports(report_snapshot),
            "round_evidence": new_rows_wave,
            "round": state["control"]["round"],
            "wave": wave,
        }
        audit_feedback = None
        reports_by_key = None
        max_coverage_retries = max(0, int(values.get("verifier_audit_coverage_retries", 2)))
        for audit_attempt in range(max_coverage_retries + 1):
            request_payload = copy.deepcopy(audit_payload)
            if audit_feedback is not None:
                request_payload["audit_validation_feedback"] = audit_feedback
            audit = values["verifier_model"].invoke(request_payload)
            data = audit if isinstance(audit, Mapping) else parse_json_content(getattr(audit, "content", audit))
            batch = EvidenceReportBatch.model_validate(data)
            actual_keys = [
                (report.dimension, report.aspect, report.scope, tuple(report.target_ids))
                for report in batch.reports
            ]
            expected_keys = set(expected_reports)
            actual_key_set = set(actual_keys)
            duplicates = sorted({key for key in actual_keys if actual_keys.count(key) > 1})
            if actual_key_set == expected_keys and len(actual_keys) == len(expected_keys) and not duplicates:
                reports_by_key = dict(zip(actual_keys, batch.reports))
                if audit_attempt:
                    append_runtime_trace(
                        values.get("runtime_trace_path"), node="verifier",
                        event="verifier_audit_coverage_repaired", round_id=state["control"]["round"],
                        payload={"wave": wave, "attempt": audit_attempt + 1},
                    )
                break

            missing = sorted(expected_keys - actual_key_set)
            extra = sorted(actual_key_set - expected_keys)
            append_runtime_trace(
                values.get("runtime_trace_path"), node="verifier",
                event="verifier_audit_coverage_invalid", round_id=state["control"]["round"],
                payload={"wave": wave, "attempt": audit_attempt + 1,
                         "expected": sorted(expected_keys), "actual": actual_keys,
                         "missing": missing, "extra": extra, "duplicates": duplicates},
            )
            if audit_attempt >= max_coverage_retries:
                raise ValueError(
                    "Verifier reports must exactly cover the tool evidence targets after audit retries"
                )
            audit_feedback = {
                "error": "Verifier report coverage does not exactly match required_reports.",
                "missing_reports": missing,
                "extra_reports": extra,
                "duplicate_reports": duplicates,
                "instruction": (
                    "Return exactly one Evidence Report for every required_reports item. "
                    "Copy dimension, aspect, scope, and target_ids exactly. Do not omit, "
                    "duplicate, add, merge, rename, or retarget reports."
                ),
            }

        for key, report in reports_by_key.items():
            dimension, aspect, scope, targets = key
            report.tool_refs = sorted(expected_reports[key])
            metric_refs = set()
            report_target = "|".join(targets) if scope == "pair" else (targets[0] if targets else "partition")
            for row in new_rows_wave:
                if row["tool_name"] not in report.tool_refs:
                    continue
                prefix = f"tool_results.{row['tool_name']}.metrics.{scope}.{report_target}"
                metric_refs.update(
                    ref for ref in row["metric_refs"]
                    if scope == "partition" or ref.startswith(prefix + ".") or ref.startswith(prefix + "[")
                )
            report.metric_refs = sorted(metric_refs)
            if aspect == "structural_diagnostics" and "structural_diagnostics" not in report.tool_refs:
                raise ValueError("Structural Evidence Reports require structural_diagnostics tool provenance")
            report_row = report.model_dump()
            report_row["request_foci"] = sorted(report_foci.get(key, set()))
            report_row["report_ref"] = evidence_report_ref(signature, dimension, aspect, scope, targets)
            reports_by_key[key] = report_row

        merged = {
            (row["dimension"], row["aspect"], row["scope"], tuple(row["target_ids"])): row
            for row in working_reports
        }
        for key, report in reports_by_key.items():
            previous = merged.get(key)
            row = report
            if previous:
                row["observations"] = previous["observations"] + row["observations"]
                row["limitations"] = sorted(set(previous["limitations"] + row["limitations"]))
                row["tool_refs"] = sorted(set(previous["tool_refs"] + row["tool_refs"]))
                row["metric_refs"] = sorted(set(previous["metric_refs"] + row["metric_refs"]))
                row["request_foci"] = sorted(set(
                    previous.get("request_foci", []) + row.get("request_foci", [])
                ))
                row["cross_evidence_context"] = row["cross_evidence_context"] or previous["cross_evidence_context"]
            merged[key] = row
        working_reports = list(merged.values())
        working_tool_evidence.extend(new_rows_wave)
        new_rows_all.extend(new_rows_wave)
        if mandatory_partition_screen:
            for item in request_states.values():
                item.update(stopped=True, stop_reason="mandatory_partition_screen_complete")
        append_runtime_trace(
            values.get("runtime_trace_path"), node="verifier", event="evidence_reports",
            round_id=state["control"]["round"],
            payload={"wave": wave, "reports": list(reports_by_key.values())},
        )
        append_runtime_trace(
            values.get("runtime_trace_path"), node="verifier",
            event="verifier_acquisition_wave_complete", round_id=state["control"]["round"],
            payload={"wave": wave, "selected_tools": [
                {"request_ref": ref, "tool_name": name} for ref, name in selected.items()
            ], "reports": list(reports_by_key.values())},
        )

    append_runtime_trace(
        values.get("runtime_trace_path"), node="verifier",
        event="verifier_acquisition_complete", round_id=state["control"]["round"],
        payload={"wave_count": wave, "request_states": [
            {"request_ref": ref, "attempted_tools": sorted(item["attempted_tools"]),
             "stop_reason": item["stop_reason"]}
            for ref, item in sorted(request_states.items())
        ]},
    )
    state["reports"] = working_reports
    state["evidence_memory"][signature] = copy.deepcopy(working_reports)
    state["tool_evidence"] = working_tool_evidence
    state["round_evidence"] = new_rows_all
    state["control"]["pending_evidence_requests"] = []
    state["control"]["next"] = "router"
    state["control"]["trace"].append({
        "round": state["control"]["round"], "node": "verifier", "event": "acquisition_complete",
        "waves": wave, "tools": [row["tool_name"] for row in new_rows_all],
        "report_count": len(working_reports),
    })
    return {
        "reports": working_reports, "evidence_memory": state["evidence_memory"],
        "tool_evidence": working_tool_evidence, "round_evidence": new_rows_all,
        "control": state["control"],
    }


def validate_revision_plan(
    plan: RevisionPlan,
    action: RouterAction,
    available_refs: list[str],
) -> None:
    if action.action == "split":
        if (len(plan.split_plans) != 1 or plan.merge_plans
                or plan.split_plans[0].target_id != action.target_ids[0]
                or plan.split_plans[0].n_children != action.n_children
                or plan.split_plans[0].structural_basis != ["candidate_consensus"]
                or plan.split_plans[0].execution_strategy != "candidate_consensus_spectral"):
            raise ValueError("RevisionPlan must preserve the exact Router split target and child count")
    elif len(plan.merge_plans) != 1 or plan.split_plans or sorted(plan.merge_plans[0].target_ids) != sorted(action.target_ids):
        raise ValueError("RevisionPlan must preserve the exact Router merge pair")
    plans = [*plan.split_plans, *plan.merge_plans]
    if any(not set(item.metric_refs).issubset(available_refs) for item in plans):
        raise ValueError("RevisionPlan references unavailable structural metrics")


def reviser_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    values = runtime.context if isinstance(runtime, Runtime) else runtime
    router_plan = RouterPlan.model_validate(state["router_plan"])
    action = router_plan.actions[0]
    current = current_sets(state)
    if action.action == "merge" and len(current) - 1 < 2:
        raise ValueError("Merge cannot collapse the subtype partition to a single whole-cohort set")
    signature = partition_signature(current)
    tool_rows = [row for row in state["tool_evidence"]
                 if row["partition_signature"] == signature and row["tool_name"] == "structural_diagnostics"]
    target_key = action.target_ids[0] if action.action == "split" else "|".join(action.target_ids)
    scope = "set" if action.action == "split" else "pair"
    prefix = f"tool_results.structural_diagnostics.metrics.{scope}.{target_key}"
    available_refs = sorted({
        ref for row in tool_rows for ref in row["metric_refs"]
        if ref.startswith(prefix + ".") or ref.startswith(prefix + "[")
    })
    append_runtime_trace(
        values.get("runtime_trace_path"),
        node="reviser",
        event="reviser_context",
        round_id=state["control"]["round"],
        payload={
            "partition": partition_snapshot(current),
            "router_action": action.model_dump(),
            "referenced_reports": [row for row in state["reports"]
                                   if row["report_ref"] in action.evidence_report_refs],
            "available_metric_refs": available_refs,
        },
    )
    model = values["reviser_model"]
    payload = {
        "partition": agent_partition_context(state["partition"]),
        "router_action": action.model_dump(),
        "referenced_reports": [row for row in state["reports"]
                               if row["report_ref"] in action.evidence_report_refs],
        "available_metric_refs": available_refs,
    }
    validation_feedback = None
    max_retries = int(getattr(model, "config", {}).get("reviser_plan_validation_retries", 1))
    for attempt in range(max_retries + 1):
        request = dict(payload)
        if validation_feedback is not None:
            request["validation_feedback"] = validation_feedback
        plan = RevisionPlan.model_validate(parse_json_content(model.invoke(request)))
        try:
            validate_revision_plan(plan, action, available_refs)
            break
        except ValueError as exc:
            if attempt >= max_retries:
                raise
            validation_feedback = {
                "error": str(exc),
                "instruction": (
                    "Repair only the execution-plan contract violation. Preserve the Router action, "
                    "target set or pair, and split child count exactly. Use only available_metric_refs."
                ),
            }
            append_runtime_trace(
                values.get("runtime_trace_path"),
                node="reviser",
                event="reviser_plan_validation_retry",
                round_id=state["control"]["round"],
                payload={"attempt": attempt + 1, "error": str(exc)},
            )
    append_runtime_trace(
        values.get("runtime_trace_path"),
        node="reviser",
        event="revision_plan",
        round_id=state["control"]["round"],
        payload={"plan": plan.model_dump()},
    )
    by_id = {set_id(item): item for item in current}
    replacements = {}
    if action.action == "split":
        split = plan.split_plans[0]
        source = by_id[split.target_id]
        patient_ids, consensus = candidate_consensus_geometry(
            values["data_root"], current,
        )
        indices = {str(patient_id): index for index, patient_id in enumerate(patient_ids)}
        members = source["member_ids"]
        local = consensus[np.ix_([indices[item] for item in members], [indices[item] for item in members])]
        settings = load_yaml_file(Path(values["config_dir"]) / "subtype_review.yaml")["cross_modal"]["structural"]
        labels = SpectralClustering(
            n_clusters=split.n_children, affinity="precomputed", assign_labels="cluster_qr", random_state=0
        ).fit_predict(local)
        groups = [
            sorted(members[index] for index, label in enumerate(labels) if label == child)
            for child in range(split.n_children)
        ]
        if min(map(len, groups)) < int(settings["min_child_size"]):
            raise ValueError("Spectral split produced a child smaller than min_child_size")
        groups.sort(key=lambda group: min(indices[case_id] for case_id in group))
        replacements[split.target_id] = [
            {
                "set_id": f"{split.target_id}_S{index + 1}",
                "member_ids": members,
                "generator": copy.deepcopy(source.get("generator", {})),
                "revision_lineage": [*source.get("revision_lineage", []), {
                    "action": "split", "parent_set_id": split.target_id, "plan": split.model_dump(),
                }],
            }
            for index, members in enumerate(groups)
        ]
    else:
        merge = plan.merge_plans[0]
        sources = [by_id[target] for target in merge.target_ids]
        generators = {
            json.dumps(source.get("generator", {}), sort_keys=True)
            for source in sources
        }
        if len(generators) != 1:
            raise ValueError("Merge sources must share candidate-generation geometry")
        members = sorted(member for source in sources for member in source["member_ids"])
        if len(members) != len(set(members)):
            raise ValueError("Merge target sets have overlapping patient membership")
        merged_id = "_M_".join(sorted(merge.target_ids))
        replacements.update({target: [] for target in merge.target_ids})
        replacements[sorted(merge.target_ids)[0]] = [{
            "set_id": merged_id,
            "member_ids": members,
            "generator": copy.deepcopy(sources[0].get("generator", {})),
            "revision_lineage": [
                *sum((source.get("revision_lineage", []) for source in sources), []),
                {"action": "merge", "parent_set_ids": sorted(merge.target_ids), "plan": merge.model_dump()},
            ],
        }]

    new_sets = [replacement for item in current for replacement in replacements.get(set_id(item), [item])]
    new_sets.sort(key=set_id)
    new_signature = partition_signature(new_sets)
    validated_plans = [*plan.split_plans, *plan.merge_plans]
    result = {
        "old_partition_signature": signature,
        "new_partition_signature": new_signature,
        "operation": action.action,
        "target_ids": action.target_ids,
        "new_set_ids": [set_id(item) for item in new_sets if set_id(item) not in by_id],
        "metric_refs": sorted({ref for item in validated_plans for ref in item.metric_refs}),
    }
    append_runtime_trace(
        values.get("runtime_trace_path"),
        node="reviser",
        event="revision_applied",
        round_id=state["control"]["round"],
        payload={
            "operation": action.action,
            "old_partition": partition_snapshot(current),
            "new_partition": partition_snapshot(new_sets),
            "revision_result": result,
        },
    )
    state["partition"] = {"sets": new_sets}
    state["reports"] = copy.deepcopy(state["evidence_memory"].get(new_signature, []))
    state["revision_plan"] = plan.model_dump()
    state["revision_result"] = result
    state["history"][-1]["revision_plan"] = plan.model_dump()
    state["history"][-1]["revision_result"] = result
    state["control"]["next"] = "router"
    state["control"]["pending_evidence_requests"] = []
    state["control"]["trace"].append({
        "round": state["control"]["round"], "node": "reviser", "event": "revision_applied",
        "result": result,
    })
    return {
        "partition": state["partition"], "reports": state["reports"],
        "revision_plan": state["revision_plan"], "revision_result": result,
        "history": state["history"], "control": state["control"],
    }


def build_review_graph() -> Any:
    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    graph.add_node("router", router_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("reviser", reviser_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda state: state["control"]["next"], {
        "verifier": "verifier", "reviser": "reviser", "end": END,
    })
    graph.add_edge("verifier", "router")
    graph.add_edge("reviser", "router")
    return graph.compile()


def save_review_outputs(state: Mapping[str, Any], output_root: str, *, direct: bool = False) -> dict[str, Any]:
    root = Path(output_root) if direct else Path(output_root) / "subtype_review"
    root.mkdir(parents=True, exist_ok=True)
    actions = {
        target: action["action"]
        for action in (state.get("router_plan") or {}).get("actions", [])
        for target in action["target_ids"]
    }
    sets = current_sets(state)
    accepted = [item for item in sets if actions.get(set_id(item)) == "accept"]
    dropped = [item for item in sets if actions.get(set_id(item)) == "drop"]
    accepted_reports = []
    for item in accepted:
        identifier = set_id(item)
        reports = [report for report in state["reports"]
                   if report["scope"] == "partition" or identifier in report["target_ids"]]
        accepted_reports.append({
            "set_id": identifier,
            "membership": item["member_ids"],
            "revision_lineage": item.get("revision_lineage", []),
            "evidence_by_dimension": {
                dimension: [report for report in reports if report["dimension"] == dimension]
                for dimension in EVIDENCE_DIMENSIONS
            },
            "metric_refs": sorted({ref for report in reports for ref in report["metric_refs"]}),
            "reports": reports,
        })
    summary = {
        "stage": "subtype_review",
        "status": "review_complete" if state["control"]["status"] == "complete" else state["control"]["status"],
        "raw_control_status": state["control"]["status"],
        "rounds_used": state["control"]["round"],
        "llm_usage": to_jsonable(state["control"].get("llm_usage", {})),
        "partition": to_jsonable(state["partition"]),
        "partition_sets": to_jsonable(sets),
        "accepted_subtype_sets": to_jsonable(accepted),
        "accepted_subtype_reports": to_jsonable(accepted_reports),
        "dropped_set_registry": to_jsonable(dropped),
        "pending_evidence_requests": to_jsonable(state["control"].get("pending_evidence_requests", [])),
        "accepted_patient_count": len({member for item in accepted for member in item["member_ids"]}),
        "history": to_jsonable(state["history"]),
        "decision_trace": to_jsonable(state["control"]["trace"]),
    }
    for filename, payload in {
        "final_partition_sets.json": sets,
        "final_subtype_sets.json": accepted,
        "accepted_subtype_reports.json": accepted_reports,
        "dropped_set_reports.json": dropped,
        "review_history.json": state["history"],
        "tool_evidence.json": state["tool_evidence"],
        "final_review_summary.json": summary,
    }.items():
        (root / filename).write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "decision_trace.jsonl").write_text(
        "".join(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n" for row in state["control"]["trace"]),
        encoding="utf-8",
    )
    return summary
