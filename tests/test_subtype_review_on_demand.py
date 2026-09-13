import pytest

from agents.subtype_review.graph import (
    available_evidence_requests,
    eligible_tools_for_requests,
    execute_tool_calls,
    initial_review_state,
    validate_router_plan,
)
from agents.subtype_review.schemas import EvidenceRequest, RouterAction, RouterPlan
from agents.subtype_review.tools import TOOL_REGISTRY, build_validation_tools


def state_for(*sets):
    return initial_review_state([
        {"set_id": name, "member_ids": members} for name, members in sets
    ])


def runtime(registry=None):
    return {
        "tool_registry": registry or TOOL_REGISTRY,
        "patient_states_by_id": {},
        "data_root": "/tmp",
        "artifact_root": "/tmp",
        "config_dir": "configs",
    }


def decision_state():
    return {
        "identity": "unassessed",
        "structure": "unassessed",
        "alternative_explanation": "unassessed",
        "uncertainty": "yes",
    }


def test_evidence_request_is_tool_free_and_normalizes_targets():
    request = EvidenceRequest(
        dimension="biological_support",
        target_ids=["C2", "C1", "C1"],
        question="Is the molecular identity coherent?",
    )
    assert request.target_ids == ["C1", "C2"]
    with pytest.raises(ValueError):
        RouterAction(action="accept", target_ids=["C1"], tool_name="pathway_enrichment")


def test_router_plan_supports_evidence_only_and_action_only_modes():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    evidence = RouterPlan(evidence_requests=[{
        "dimension": "biological_support", "target_ids": ["C1"], "question": "x",
    }])
    validate_router_plan(evidence, state, runtime())
    actions = RouterPlan(actions=[
        {"action": "drop", "target_ids": ["C1"], "decision_state": decision_state()},
        {"action": "drop", "target_ids": ["C2"], "decision_state": decision_state()},
    ])
    validate_router_plan(actions, state, runtime())
    with pytest.raises(ValueError, match="exactly one mode"):
        RouterPlan()
    with pytest.raises(ValueError, match="exactly one mode"):
        RouterPlan(actions=actions.actions, evidence_requests=[{
            "dimension": "biological_support", "target_ids": ["C1"], "question": "x",
        }])


def test_evidence_requests_may_be_partial_and_cross_dimension_overlapping():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]), ("C3", ["P5", "P6"]))
    plan = RouterPlan(evidence_requests=[
        {"dimension": "biological_support", "target_ids": ["C2"], "question": "x"},
        {"dimension": "confounder_exclusion", "target_ids": ["C2"], "question": "y"},
    ])
    validate_router_plan(plan, state, runtime())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_router_plan(RouterPlan(evidence_requests=[
            {"dimension": "biological_support", "target_ids": ["C1", "C2"], "question": "x"},
            {"dimension": "biological_support", "target_ids": ["C2"], "question": "y"},
        ]), state, runtime())


def test_partition_and_set_requests_can_share_a_plan():
    state = state_for(("C1", ["P1", "P2"]))
    plan = RouterPlan(evidence_requests=[
        {"dimension": "confounder_exclusion", "target_ids": ["C1"], "question": "x"},
        {"dimension": "known_label_echo", "target_ids": [], "question": "y"},
    ])
    validate_router_plan(plan, state, runtime())


def test_scope_is_registry_driven():
    registry = {name: {**metadata} for name, metadata in TOOL_REGISTRY.items()}
    registry["pathway_enrichment"]["scope"] = "partition"
    state = state_for(("C1", ["P1", "P2"]))
    plan = RouterPlan(evidence_requests=[{
        "dimension": "biological_support", "target_ids": [], "question": "x",
    }])
    validate_router_plan(plan, state, runtime(registry))


def test_verifier_tools_follow_router_targets():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    request = {"dimension": "biological_support", "target_ids": ["C1"], "question": "x"}
    state["control"].update({
        "pending_evidence_requests": [request],
        "eligible_tools": eligible_tools_for_requests(state, runtime(), [request]),
    })
    registry = {name: {**metadata} for name, metadata in TOOL_REGISTRY.items()}
    seen = []
    registry["pathway_enrichment"]["function"] = lambda *args, **kwargs: (
        seen.append(kwargs["target_ids"]) or {"status": "success", "results": {"decision_metrics": {}}}
    )
    execute_tool_calls(state, {"tool_calls": [{
        "name": "pathway_enrichment", "id": "1", "args": {"target_ids": ["C1"]},
    }]}, runtime(registry))
    assert seen == [["C1"]]
    assert state["round_evidence"][0]["target_ids"] == ["C1"]


def test_one_tool_call_may_cover_multiple_pending_targets():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    requests = [
        {"dimension": "biological_support", "target_ids": [target], "question": "x"}
        for target in ("C1", "C2")
    ]
    registry = {name: {**metadata} for name, metadata in TOOL_REGISTRY.items()}
    registry["pathway_enrichment"]["function"] = lambda *args, **kwargs: {
        "status": "success", "results": {"decision_metrics": {}},
    }
    state["control"].update({
        "pending_evidence_requests": requests,
        "eligible_tools": eligible_tools_for_requests(state, runtime(registry), requests),
    })
    execute_tool_calls(state, {"tool_calls": [{
        "name": "pathway_enrichment", "id": "call-1",
        "args": {"target_ids": ["C1", "C2"]},
    }]}, runtime(registry))
    assert state["round_evidence"][0]["target_ids"] == ["C1", "C2"]


def test_available_evidence_requests_skip_completed_tools():
    state = state_for(("C1", ["P1", "P2"]))
    request = {"dimension": "biological_support", "target_ids": ["C1"], "question": "x"}
    state["control"].update({
        "pending_evidence_requests": [request],
        "eligible_tools": eligible_tools_for_requests(state, runtime(), [request]),
    })
    registry = {name: {**metadata} for name, metadata in TOOL_REGISTRY.items()}
    for name in ("pathway_enrichment", "mutation_enrichment", "cnv_characterization"):
        registry[name]["function"] = lambda *args, **kwargs: {
            "status": "success", "results": {"decision_metrics": {}},
        }
    execute_tool_calls(state, {"tool_calls": [
        {"name": name, "id": name, "args": {"target_ids": ["C1"]}}
        for name in ("pathway_enrichment", "mutation_enrichment", "cnv_characterization")
    ]}, runtime(registry))
    assert not any(
        item["dimension"] == "biological_support" and item["target_ids"] == ["C1"]
        for item in available_evidence_requests(state, runtime(registry))
    )


def test_validation_tools_require_target_ids():
    tools = {item.name: item for item in build_validation_tools()}
    assert "target_ids" in tools["pathway_enrichment"].args_schema.model_fields
