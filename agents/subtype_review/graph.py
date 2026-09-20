from __future__ import annotations

import copy
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
    RouterPlan,
    set_id,
)
from agents.subtype_review.tools import TOOL_REGISTRY, compact_tool_result
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


def initial_review_state(candidate_sets: list[dict[str, Any]]) -> ReviewState:
    sets = [
        {
            "set_id": set_id(item),
            "member_ids": sorted(map(str, item["member_ids"])),
            "revision_lineage": list(item.get("revision_lineage", [])),
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
    available: set[tuple[str, str, tuple[str, ...]]],
    terminal_only: bool,
) -> None:
    current = {set_id(item) for item in current_sets(state)}
    if plan.evidence_requests:
        if terminal_only:
            raise ValueError("Terminal round cannot request additional evidence")
        seen = set()
        for request in plan.evidence_requests:
            key = (request.dimension, request.scope, tuple(request.target_ids))
            if key not in available or key in seen:
                raise ValueError("EvidenceRequest is unavailable or duplicated")
            seen.add(key)
        return

    structural = [action for action in plan.actions if action.action in {"split", "merge"}]
    if structural:
        if terminal_only or len(plan.actions) != 1 or len(structural) != 1:
            raise ValueError("A revision round must contain exactly one split or merge")
        action = structural[0]
        if not set(action.target_ids).issubset(current):
            raise ValueError("Structural action targets must be current sets")
        if action.decision_state.structure != "incompatible":
            raise ValueError("Structural revision requires structure=incompatible")
        if action.action == "split":
            target = action.target_ids[0]
            report = next((row for row in reversed(state["reports"])
                           if row["dimension"] == "cross_modal_consistency"
                           and row["aspect"] == "structural_diagnostics"
                           and row["scope"] == "set" and row["target_ids"] == [target]), None)
            if (report is None
                    or report.get("internal_structure_assessment") != "supports_subdivision"
                    or report.get("suggested_k") != action.n_children):
                raise ValueError("Split child count must match the Verifier's supported suggested_k")
        else:
            targets = sorted(action.target_ids)
            report = next((row for row in reversed(state["reports"])
                           if row["dimension"] == "cross_modal_consistency"
                           and row["aspect"] == "structural_diagnostics"
                           and row["scope"] == "pair" and row["target_ids"] == targets), None)
            if report is None or report.get("pair_boundary_assessment") != "insufficiently_separated":
                raise ValueError("Merge requires Verifier evidence of an insufficiently separated pair")
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

    signature = partition_signature(current_sets(state))
    screen = next((row for row in state["tool_evidence"]
                   if row["tool_name"] == "structural_diagnostics"
                   and row["scope"] == "partition"
                   and row["partition_signature"] == signature), None)
    screened_sets = (screen or {}).get("metrics", {}).get("partition", {}).get("internal_structure", {})
    for action in plan.actions:
        target = action.target_ids[0]
        reports = state["reports"]
        decision = action.decision_state
        if decision.structure == "unassessed":
            raise ValueError("Terminal structure must be assessed after the partition structural screen")
        if int(screened_sets.get(target, {}).get("candidate_k") or 1) > 1 and not any(
            row["dimension"] == "cross_modal_consistency"
            and row["aspect"] == "structural_diagnostics"
            and row["scope"] == "set" and row["target_ids"] == [target]
            for row in reports
        ):
            raise ValueError(f"Terminal disposition requires a targeted set structural diagnostic for {target}")
        requirements = (
            (decision.identity, "biological_support", "set"),
            (decision.structure, "cross_modal_consistency", "set"),
            (decision.alternative_explanation, "confounder_exclusion", "set"),
        )
        for value, dimension, scope in requirements:
            if value == "unassessed":
                continue
            assessed = any(
                row["dimension"] == dimension
                and (
                    row["aspect"] == "structural_diagnostics"
                    and (row["scope"] == "partition" or (row["scope"] == scope and target in row["target_ids"]))
                    if dimension == "cross_modal_consistency"
                    else row["scope"] == scope and target in row["target_ids"]
                )
                for row in reports
            )
            if not assessed:
                raise ValueError(f"{dimension} cannot be asserted without a set-level report for {target}")


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
    set_ids = [set_id(item) for item in current]
    available = set()
    for name, metadata in values.get("tool_registry", TOOL_REGISTRY).items():
        dimension = metadata["dimension"]
        for scope in metadata["scopes"]:
            if scope == "partition":
                targets = [()]
            elif scope == "set":
                targets = [(item,) for item in set_ids]
            else:
                targets = list(combinations(set_ids, 2))
            for target_ids in targets:
                key = (name, scope, tuple(target_ids))
                if key not in completed:
                    available.add((dimension, scope, tuple(target_ids)))

    request_options = [
        {"dimension": dimension, "scope": scope, "target_ids": list(target_ids)}
        for dimension, scope, target_ids in sorted(available)
    ]
    coverage = {
        "set": {item: {dimension: "unassessed" for dimension in EVIDENCE_DIMENSIONS} for item in set_ids},
        "pair": {},
        "partition": {dimension: "unassessed" for dimension in EVIDENCE_DIMENSIONS},
    }
    for report in state["reports"]:
        scope = report["scope"]
        target = "|".join(report["target_ids"]) if scope == "pair" else (
            report["target_ids"][0] if scope == "set" else "partition"
        )
        coverage[scope].setdefault(target, {})[report["dimension"]] = "assessed"
    terminal_only = control["round"] >= control["max_rounds"] and partition_screen_done
    if not terminal_only:
        control["round"] += 1
    payload = {
        "partition": state["partition"],
        "evidence_reports": summarize_reports(state["reports"]),
        "evidence_coverage": coverage,
        "available_evidence_requests": request_options,
        "round": control["round"],
        "terminal_only": terminal_only,
    }
    if partition_screen_done:
        plan = RouterPlan.model_validate(parse_json_content(values["router_model"].invoke(payload)))
    else:
        if ("cross_modal_consistency", "partition", ()) not in available:
            raise ValueError("Structural diagnostics must support a partition-level screen")
        plan = RouterPlan(evidence_requests=[EvidenceRequest(
            dimension="cross_modal_consistency",
            scope="partition",
            target_ids=[],
            question="Screen the current partition for unsupported internal splits and weak pair boundaries.",
        )])
    validate_router_plan(plan, state, available, terminal_only)

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
    if plan.evidence_requests:
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
    set_ids = {set_id(item) for item in current}
    signature = partition_signature(current)
    registry = values.get("tool_registry", TOOL_REGISTRY)
    completed = {
        (row["tool_name"], row["scope"], tuple(row["target_ids"]))
        for row in state["tool_evidence"] if row["partition_signature"] == signature
    }
    eligible = {}
    for name, metadata in registry.items():
        allowed = []
        for request in requests:
            targets = tuple(request.target_ids)
            if (metadata["dimension"] == request.dimension
                    and request.scope in metadata["scopes"]
                    and (name, request.scope, targets) not in completed):
                allowed.append({"scope": request.scope, "target_ids": list(targets)})
        if allowed:
            eligible[name] = {"description": metadata["description"], "allowed_requests": allowed}
    if not eligible:
        raise ValueError("No unrun scientific tool can answer the pending EvidenceRequests")

    acquisition = values["verifier_model"].invoke({
        "mode": "acquire",
        "partition": state["partition"],
        "evidence_requests": [item.model_dump() for item in requests],
        "current_evidence": summarize_reports(state["reports"]),
        "eligible_tools": eligible,
        "round": state["control"]["round"],
    })
    calls = acquisition.get("tool_calls", []) if isinstance(acquisition, Mapping) else acquisition.tool_calls
    expected_reports: dict[tuple[str, str, str, tuple[str, ...]], set[str]] = {}
    new_rows = []
    called = set()
    for call in calls:
        name, args = call["name"], call["args"]
        if name not in eligible:
            raise ValueError(f"Verifier selected an ineligible tool: {name}")
        metadata = registry[name]
        scope = str(args["scope"])
        targets = tuple(sorted(set(map(str, args.get("target_ids", [])))))
        if {"set": 1, "pair": 2, "partition": 0}[scope] != len(targets):
            expected = {"set": "exactly one set target", "pair": "exactly two pair targets", "partition": "no partition targets"}[scope]
            raise ValueError(f"Verifier must provide {expected}")
        if scope == "pair" and not set(targets).issubset(set_ids):
            raise ValueError("Verifier pair must reference two current sets")
        if scope == "set" and not set(targets).issubset(set_ids):
            raise ValueError("Verifier set request includes a non-current target")
        allowed = {(item["scope"], tuple(item["target_ids"])) for item in eligible[name]["allowed_requests"]}
        if any((scope, (target,)) not in allowed for target in targets) if scope == "set" else (scope, targets) not in allowed:
            raise ValueError(f"Verifier tool call is outside Router EvidenceRequests: {name}")
        call_key = (name, scope, targets)
        if call_key in called or call_key in completed:
            raise ValueError(f"Verifier repeated a tool call: {name} {scope} {targets}")
        called.add(call_key)
        raw = metadata["function"](
            patient_states_by_id=values["patient_states_by_id"],
            output_root=str(values["data_root"]),
            config_dir=str(values["config_dir"]),
            data_root=str(values["data_root"]),
            all_cluster_states=current,
            scope=scope,
            target_ids=list(targets),
        )
        row = compact_tool_result(raw, name)
        if row["status"] == "runtime_failure":
            raise RuntimeError(f"Scientific tool {name} failed: {row['errors']}")
        aspect = metadata.get("aspect", name)
        row.update({
            "dimension": metadata["dimension"],
            "aspect": aspect,
            "scope": scope,
            "target_ids": list(targets),
            "partition_signature": signature,
        })
        new_rows.append(row)
        report_targets = [()] if scope == "partition" else [(target,) for target in targets] if scope == "set" else [targets]
        for report_target in report_targets:
            expected_reports.setdefault((metadata["dimension"], aspect, scope, report_target), set()).add(name)

    for request in requests:
        covered = any(
            dimension == request.dimension and scope == request.scope
            and targets == tuple(request.target_ids)
            for dimension, aspect, scope, targets in expected_reports
        )
        if not covered:
            raise ValueError(f"Verifier did not call a tool for request {(request.dimension, request.scope, tuple(request.target_ids))}")

    audit = values["verifier_model"].invoke({
        "mode": "audit",
        "partition": {"sets": [{"set_id": set_id(item), "member_n": len(item["member_ids"])} for item in current]},
        "required_reports": [
            {"dimension": dimension, "aspect": aspect, "scope": scope,
             "target_ids": list(targets), "tool_refs": sorted(names)}
            for (dimension, aspect, scope, targets), names in sorted(expected_reports.items())
        ],
        "prior_reports": summarize_reports(state["reports"]),
        "round_evidence": new_rows,
        "round": state["control"]["round"],
    })
    data = audit if isinstance(audit, Mapping) else parse_json_content(getattr(audit, "content", audit))
    batch = EvidenceReportBatch.model_validate(data)
    reports_by_key = {
        (report.dimension, report.aspect, report.scope, tuple(report.target_ids)): report
        for report in batch.reports
    }
    if set(reports_by_key) != set(expected_reports):
        raise ValueError("Verifier reports must exactly cover the tool evidence targets")

    for key, report in reports_by_key.items():
        dimension, aspect, scope, targets = key
        report.tool_refs = sorted(expected_reports[key])
        metric_refs = set()
        report_target = "|".join(targets) if scope == "pair" else (targets[0] if targets else "partition")
        for row in new_rows:
            if row["tool_name"] not in report.tool_refs:
                continue
            prefix = f"tool_results.{row['tool_name']}.metrics.{scope}.{report_target}"
            metric_refs.update(
                ref for ref in row["metric_refs"]
                if scope == "partition" or ref.startswith(prefix + ".") or ref.startswith(prefix + "[")
            )
            if (row["tool_name"] == "structural_diagnostics" and aspect == "structural_diagnostics" and scope == "set"
                    and report_target in row["metrics"].get("set", {})):
                report.suggested_k = row["metrics"]["set"][report_target]["suggested_k"]
        report.metric_refs = sorted(metric_refs)
        if report.internal_structure_assessment == "supports_subdivision" and report.suggested_k is None:
            raise ValueError("Verifier cannot support subdivision without an estimable suggested_k")
        if aspect != "structural_diagnostics" and (
            report.internal_structure_assessment is not None
            or report.pair_boundary_assessment is not None
            or report.suggested_k is not None
        ):
            raise ValueError("Structural assessments must be reported under the structural_diagnostics aspect")
        if aspect == "structural_diagnostics" and scope == "partition" and (
            report.internal_structure_assessment is not None
            or report.pair_boundary_assessment is not None
            or report.suggested_k is not None
        ):
            raise ValueError("Partition structural screens use observations, not set/pair assessment fields")
        if report.pair_boundary_assessment == "insufficiently_separated" and "structural_diagnostics" not in report.tool_refs:
            raise ValueError("Verifier cannot report a weak boundary without pair structural diagnostics")

    merged = {
        (row["dimension"], row["aspect"], row["scope"], tuple(row["target_ids"])): row
        for row in state["reports"]
    }
    for key, report in reports_by_key.items():
        previous = merged.get(key)
        row = report.model_dump()
        if previous:
            row["observations"] = previous["observations"] + row["observations"]
            row["limitations"] = sorted(set(previous["limitations"] + row["limitations"]))
            row["tool_refs"] = sorted(set(previous["tool_refs"] + row["tool_refs"]))
            row["metric_refs"] = sorted(set(previous["metric_refs"] + row["metric_refs"]))
        merged[key] = row
    state["reports"] = list(merged.values())
    state["evidence_memory"][signature] = copy.deepcopy(state["reports"])
    state["tool_evidence"].extend(new_rows)
    state["round_evidence"] = new_rows
    state["control"]["pending_evidence_requests"] = []
    state["control"]["next"] = "router"
    state["control"]["trace"].append({
        "round": state["control"]["round"], "node": "verifier", "event": "audit",
        "tools": sorted(called), "report_count": len(reports_by_key),
    })
    return {
        "reports": state["reports"], "evidence_memory": state["evidence_memory"],
        "tool_evidence": state["tool_evidence"], "round_evidence": new_rows,
        "control": state["control"],
    }


def reviser_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    values = runtime.context if isinstance(runtime, Runtime) else runtime
    router_plan = RouterPlan.model_validate(state["router_plan"])
    action = router_plan.actions[0]
    current = current_sets(state)
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
    plan = RevisionPlan.model_validate(parse_json_content(values["reviser_model"].invoke({
        "partition": state["partition"],
        "router_action": action.model_dump(),
        "available_metric_refs": available_refs,
    })))
    if action.action == "split":
        if (len(plan.split_plans) != 1 or plan.merge_plans
                or plan.split_plans[0].target_id != action.target_ids[0]
                or plan.split_plans[0].n_children != action.n_children
                or plan.split_plans[0].structural_basis != ["fused"]
                or plan.split_plans[0].execution_strategy != "fused_similarity_spectral"):
            raise ValueError("RevisionPlan must preserve the exact Router split target and child count")
    elif len(plan.merge_plans) != 1 or plan.split_plans or sorted(plan.merge_plans[0].target_ids) != sorted(action.target_ids):
        raise ValueError("RevisionPlan must preserve the exact Router merge pair")
    plans = [*plan.split_plans, *plan.merge_plans]
    if any(not set(item.metric_refs).issubset(available_refs) for item in plans):
        raise ValueError("RevisionPlan references unavailable structural metrics")

    by_id = {set_id(item): item for item in current}
    replacements = {}
    if action.action == "split":
        split = plan.split_plans[0]
        source = by_id[split.target_id]
        root = Path(values["data_root"]) / "candidate_subtype"
        patient_ids = json.loads((root / "affinity_patient_order.json").read_text())
        fused = np.load(root / "fused_similarity.npy")
        indices = {str(patient_id): index for index, patient_id in enumerate(patient_ids)}
        members = source["member_ids"]
        local = fused[np.ix_([indices[item] for item in members], [indices[item] for item in members])]
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
                "revision_lineage": [*source.get("revision_lineage", []), {
                    "action": "split", "parent_set_id": split.target_id, "plan": split.model_dump(),
                }],
            }
            for index, members in enumerate(groups)
        ]
    else:
        merge = plan.merge_plans[0]
        sources = [by_id[target] for target in merge.target_ids]
        members = sorted(member for source in sources for member in source["member_ids"])
        if len(members) != len(set(members)):
            raise ValueError("Merge target sets have overlapping patient membership")
        merged_id = "_M_".join(sorted(merge.target_ids))
        replacements.update({target: [] for target in merge.target_ids})
        replacements[sorted(merge.target_ids)[0]] = [{
            "set_id": merged_id,
            "member_ids": members,
            "revision_lineage": [
                *sum((source.get("revision_lineage", []) for source in sources), []),
                {"action": "merge", "parent_set_ids": sorted(merge.target_ids), "plan": merge.model_dump()},
            ],
        }]

    new_sets = [replacement for item in current for replacement in replacements.get(set_id(item), [item])]
    new_sets.sort(key=set_id)
    new_signature = partition_signature(new_sets)
    result = {
        "old_partition_signature": signature,
        "new_partition_signature": new_signature,
        "operation": action.action,
        "target_ids": action.target_ids,
        "new_set_ids": [set_id(item) for item in new_sets if set_id(item) not in by_id],
        "metric_refs": sorted({ref for item in plans for ref in item.metric_refs}),
    }
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
