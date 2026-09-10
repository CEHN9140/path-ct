import json

import pytest

from agents.subtype_review.graph import (
    available_evidence_requests,
    execute_tool_calls,
    initial_review_state,
    prepare_round_node,
    validate_router_plan,
)
from agents.subtype_review.schemas import EvidenceRequest, RouterAction, RouterPlan
from agents.subtype_review.tools import TOOL_REGISTRY, build_validation_tools


def state_for(*sets):
    return initial_review_state([
        {"set_id": name, "member_ids": members} for name, members in sets
    ])


def runtime(registry=None):
    registry = registry or TOOL_REGISTRY
    return {
        "tool_registry": registry,
        "patient_states_by_id": {},
        "data_root": "/tmp",
        "artifact_root": "/tmp",
        "config_dir": "configs",
    }


def test_evidence_request_has_no_tool_name_and_normalizes_targets():
    request = EvidenceRequest(
        dimension="biological_support",
        target_ids=["C2", "C1", "C1"],
        question="Is the molecular identity coherent?",
    )
    assert request.model_dump() == {
        "dimension": "biological_support",
        "target_ids": ["C1", "C2"],
        "question": "Is the molecular identity coherent?",
    }
    with pytest.raises(ValueError):
        RouterAction(
            action="accept", target_ids=["C1"], tool_name="pathway_enrichment"
        )


def test_prepare_round_creates_goals_and_eligible_tools_without_preselecting_tools():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    prepare_round_node(state, runtime())
    control = state["control"]
    assert {item["dimension"] for item in control["pending_evidence_requests"]} == {
        "biological_support", "cross_modal_consistency",
        "confounder_exclusion", "known_label_echo",
    }
    assert "pending_tools" not in control
    assert set(control["eligible_tools"]) == {
        "pathway_enrichment", "mutation_enrichment", "cnv_characterization",
        "multimodal_consistency_check", "confound_test", "known_label_echo_test",
    }


def test_verifier_tool_calls_may_select_a_subset_with_explicit_targets():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    prepare_round_node(state, runtime())
    registry = {name: {**metadata} for name, metadata in TOOL_REGISTRY.items()}
    calls = []

    def fake_tool(*args, **kwargs):
        calls.append(kwargs["target_ids"])
        return {"status": "success", "results": {"decision_metrics": {}}}

    registry["pathway_enrichment"]["function"] = fake_tool
    execute_tool_calls(
        state,
        {"tool_calls": [{"name": "pathway_enrichment", "id": "1", "args": {"target_ids": ["C1"]}}]},
        runtime(registry),
    )
    assert calls == [["C1"]]
    assert state["round_evidence"][0]["target_ids"] == ["C1"]
    assert state["control"]["next"] == "verifier_audit"


def test_router_requests_evidence_not_tools_and_must_cover_current_sets():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    prepare_round_node(state, runtime())
    plan = RouterPlan(actions=[
        RouterAction(
            action="need_more_evidence", target_ids=["C1"], evidence_requests=[
                EvidenceRequest(
                    dimension="biological_support",
                    target_ids=["C1"],
                    question="Clarify the molecular signal.",
                )
            ]
        ),
        RouterAction(action="drop", target_ids=["C2"]),
    ])
    validate_router_plan(plan, state, runtime())
    with pytest.raises(ValueError):
        RouterPlan(actions=[
            {"action": "need_more_evidence", "target_ids": ["C1"],
             "evidence_requests": [{
                 "dimension": "biological_support", "target_ids": ["C1"],
                 "question": "x", "tool_name": "pathway_enrichment"
             }]},
            {"action": "drop", "target_ids": ["C2"]},
        ])


def test_validation_tools_expose_target_ids():
    tools = {item.name: item for item in build_validation_tools()}
    assert "target_ids" in tools["pathway_enrichment"].args_schema.model_fields
