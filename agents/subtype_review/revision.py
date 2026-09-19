from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agents.subtype_review.evidence import current_partition_evidence, raw_metric_refs
from agents.subtype_review.schemas import RevisionPlan, RouterPlan, set_id
from agents.subtype_review.state import current_sets, partition_signature


def revision_plan_signature(plan: RevisionPlan) -> str:
    payload = {
        "split_plans": [
            {
                "target_id": item.target_id,
                "n_children": item.n_children,
                "structural_basis": item.structural_basis,
                "execution_strategy": item.execution_strategy,
                "metric_refs": item.metric_refs,
            }
            for item in plan.split_plans
        ],
        "merge_plans": [
            {"target_ids": item.target_ids, "metric_refs": item.metric_refs}
            for item in plan.merge_plans
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_revision_plan(
    plan: RevisionPlan, router_plan: RouterPlan, state: Mapping[str, Any]
) -> None:
    expected_splits = {action.target_ids[0] for action in router_plan.actions if action.action == "split"}
    expected_merges = {tuple(action.target_ids) for action in router_plan.actions if action.action == "merge"}
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
        new_sets.extend(replacements.get(set_id(source), [source]))
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
