from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agents.subtype_review.graph import (
    available_evidence_requests,
    build_review_graph,
    completed_tool_keys,
    execute_tool_calls,
    initial_review_state,
    is_length_finish_error,
    partition_signature,
    prepare_round_node,
    router_node,
    save_review_outputs,
    validate_reports,
    validate_router_plan,
    verifier_node,
)
from agents.subtype_review.llm import LLMOutputLengthError, LLMUsageTracker
from agents.subtype_review.schemas import EvidenceReportBatch, EvidenceRequest, RouterAction, RouterPlan
from agents.subtype_review.tools import TOOL_REGISTRY, compact_tool_result


def state_for(*groups):
    return initial_review_state([
        {"set_id": name, "member_ids": members} for name, members in groups
    ])


def registry():
    result = {}
    for name, metadata in TOOL_REGISTRY.items():
        result[name] = {**metadata, "function": lambda *args, _name=name, **kwargs: {
            "tool_name": _name, "status": "success",
            "results": {"decision_metrics": {}}, "artifacts": {},
        }}
    return result


def runtime(tools=None):
    return {
        "tool_registry": tools or registry(), "patient_states_by_id": {},
        "data_root": "/tmp", "artifact_root": "/tmp", "config_dir": "configs",
    }


def test_state_has_evidence_memory_and_no_preselected_tools():
    state = state_for(("C1", ["P1"]), ("C2", ["P2"]))
    assert "evidence_memory" in state
    assert "pending_evidence_requests" in state["control"]
    assert "pending_tools" not in state["control"]


def test_evidence_request_is_tool_free_and_normalizes_targets():
    request = EvidenceRequest(
        dimension="biological_support", target_ids=["C2", "C1", "C1"], question="Clarify signal."
    )
    assert request.target_ids == ["C1", "C2"]
    with pytest.raises(ValueError):
        RouterAction(action="accept", target_ids=["C1"], tool_name="pathway_enrichment")


def test_registry_uses_verifier_selectable_only():
    assert all("verifier_selectable" in metadata for metadata in TOOL_REGISTRY.values())
    assert all("default_every_round" not in metadata for metadata in TOOL_REGISTRY.values())
    assert TOOL_REGISTRY["clinical_characterization"]["verifier_selectable"] is False


def test_prepare_round_exposes_tools_without_precreating_requests():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    prepare_round_node(state, runtime())
    assert state["control"]["pending_evidence_requests"] == []
    assert state["control"]["acquisition_mode"] == "initial"
    assert set(state["control"]["eligible_tools"]) == {
        "pathway_enrichment", "mutation_enrichment", "cnv_characterization",
        "multimodal_consistency_check", "confound_test", "known_label_echo_test",
    }


def test_verifier_can_select_subset_with_explicit_target_ids():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    prepare_round_node(state, runtime())
    tools = registry()
    seen = []

    def pathway(*args, **kwargs):
        seen.append(kwargs["target_ids"])
        return {"status": "success", "results": {"decision_metrics": {}}}

    tools["pathway_enrichment"]["function"] = pathway
    execute_tool_calls(
        state,
        {"tool_calls": [{"name": "pathway_enrichment", "id": "call-1", "args": {"target_ids": ["C1"]}}]},
        runtime(tools),
    )
    assert seen == [["C1"]]
    assert state["round_evidence"][0]["target_ids"] == ["C1"]


def test_available_evidence_requests_do_not_expose_tool_names():
    state = state_for(("C1", ["P1", "P2"]))
    prepare_round_node(state, runtime())
    available = available_evidence_requests(state, runtime())
    assert available and all("tool_name" not in item for item in available)


def test_selected_tool_is_removed_from_next_evidence_availability():
    state = state_for(("C1", ["P1", "P2"]))
    prepare_round_node(state, runtime())
    execute_tool_calls(
        state,
        {"tool_calls": [
            {"name": name, "id": name, "args": {"target_ids": ["C1"]}}
            for name in ("pathway_enrichment", "mutation_enrichment", "cnv_characterization")
        ]},
        runtime(),
    )
    assert not any(
        item["dimension"] == "biological_support" and item["target_ids"] == ["C1"]
        for item in available_evidence_requests(state, runtime())
    )


def test_available_evidence_requests_survives_multiple_targeted_rounds():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    signature = partition_signature(state["partition"]["sets"])
    state["history"] = [{
        "partition_signature": signature,
        "round_evidence": [{
            "tool_name": "pathway_enrichment",
            "status": "success",
            "target_ids": ["C1"],
            "partition_signature": signature,
        }],
    }]
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "confounder_exclusion",
        "target_ids": ["C1"],
        "question": "Clarify C1 technical evidence.",
    }]
    state["router_request"] = list(state["control"]["pending_evidence_requests"])
    prepare_round_node(state, runtime())
    state["round_evidence"] = [{
        "tool_name": "confound_test",
        "status": "success",
        "target_ids": ["C1"],
        "partition_signature": signature,
    }]
    available = available_evidence_requests(state, runtime())
    assert any(
        item["dimension"] == "cross_modal_consistency"
        and item["target_ids"] == ["C1"]
        for item in available
    )
    assert any(
        item["dimension"] == "biological_support"
        and item["target_ids"] == ["C1"]
        for item in available
    )
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "cross_modal_consistency",
        "target_ids": ["C1"],
        "question": "Clarify C1 cross-modal evidence.",
    }]
    prepare_round_node(state, runtime())
    assert set(state["control"]["eligible_tools"]) == {
        "multimodal_consistency_check",
    }


def test_router_plan_requires_complete_nonoverlapping_coverage():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    with pytest.raises(ValueError, match="cover every current set"):
        validate_router_plan(RouterPlan(actions=[{"action": "drop", "target_ids": ["C1"]}]), state, runtime())


def test_router_stores_evidence_requests_not_tools():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    prepare_round_node(state, runtime())

    class Router:
        def invoke(self, payload):
            return {"actions": [
                {"action": "need_more_evidence", "target_ids": ["C1"], "evidence_requests": [{
                    "dimension": "biological_support", "target_ids": ["C1"], "question": "Clarify C1."
                }]},
                {"action": "drop", "target_ids": ["C2"]},
            ]}

    router_node(state, {**runtime(), "router_model": Router()})
    assert state["control"]["pending_evidence_requests"][0]["target_ids"] == ["C1"]
    assert state["router_request"][0]["target_ids"] == ["C1"]
    assert "pending_tools" not in state["control"]


def test_completed_tool_keys_are_target_specific():
    state = state_for(("C1", ["P1"]), ("C2", ["P2"]))
    signature = partition_signature(state["partition"]["sets"])
    state["round_evidence"] = [{
        "tool_name": "pathway_enrichment", "status": "success", "target_ids": ["C1", "C2"],
        "partition_signature": signature,
    }]
    assert ("pathway_enrichment", signature, "C1") in completed_tool_keys(state)
    assert ("pathway_enrichment", signature, "C2") in completed_tool_keys(state)


def test_reports_are_checked_against_actual_current_calls():
    state = state_for(("C1", ["P1", "P2"]))
    signature = partition_signature(state["partition"]["sets"])
    state["round_evidence"] = [{
        "tool_name": "pathway_enrichment", "dimension": "biological_support", "scope": "set_identity",
        "target_ids": ["C1"], "status": "success", "metrics": {}, "metric_refs": [],
        "partition_signature": signature,
    }]
    batch = EvidenceReportBatch.model_validate({"reports": [{
        "dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1"],
        "observations": [], "tool_refs": ["pathway_enrichment"],
    }]})
    validate_reports(batch, state, runtime())


def test_audit_merges_reports_into_partition_memory():
    state = state_for(("C1", ["P1", "P2"]))
    signature = partition_signature(state["partition"]["sets"])
    state["round_evidence"] = [{
        "tool_name": "pathway_enrichment", "dimension": "biological_support", "scope": "set_identity",
        "target_ids": ["C1"], "status": "success", "metrics": {}, "metric_refs": [],
        "partition_signature": signature,
    }]
    state["control"].update({"next": "verifier_audit", "partition_signature": signature})

    class Verifier:
        def invoke(self, payload):
            return {"reports": [{
                "dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1"],
                "observations": [], "statistical_interpretation": "ok", "medical_interpretation": "ok",
                "limitations": [], "tool_refs": [],
            }]}

    verifier_node(state, {**runtime(), "verifier_model": Verifier()})
    assert state["reports"] and signature in state["evidence_memory"]


def test_verifier_receives_router_request_for_targeted_acquisition():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "confounder_exclusion",
        "target_ids": ["C1"],
        "question": "Could CT acquisition explain C1?",
    }]
    state["router_request"] = state["control"]["pending_evidence_requests"]
    prepare_round_node(state, runtime())
    captured = {}

    class Verifier:
        def invoke(self, payload):
            captured.update(payload)
            return {"tool_calls": []}

    verifier_node(state, {**runtime(), "verifier_model": Verifier()})
    assert captured["acquisition_mode"] == "targeted"
    assert captured["router_request"][0]["target_ids"] == ["C1"]


def test_save_review_outputs_groups_accept_reports_by_dimension(tmp_path):
    state = state_for(("C1", ["P1", "P2"]))
    state["router_plan"] = {"actions": [{"action": "accept", "target_ids": ["C1"], "reason": "retained"}]}
    state["reports"] = [{
        "dimension": "biological_support", "scope": "set_identity",
        "target_ids": ["C1"], "metric_refs": [],
    }]
    result = save_review_outputs(state, str(tmp_path), direct=True)
    assert [item["set_id"] for item in result["accepted_subtype_sets"]] == ["C1"]


def test_scientific_unavailable_is_completed_runtime_failure_is_not():
    assert compact_tool_result({"status": "failure", "results": {"missing_reason": "none"}, "errors": []}, "x")["status"] == "scientific_unavailable"
    assert compact_tool_result({"status": "failure", "results": {"missing_reason": ""}, "errors": ["I/O"]}, "x")["status"] == "runtime_failure"


def test_graph_has_three_agents_and_prepare_round():
    nodes = build_review_graph().get_graph().nodes
    assert {"prepare_round", "verifier", "router", "reviser"}.issubset(nodes)
    assert "init_agent" not in nodes


def test_router_payload_has_no_tool_registry_or_raw_metrics():
    state = state_for(("C1", ["P1", "P2"]))
    prepare_round_node(state, runtime())
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [{"action": "drop", "target_ids": ["C1"]}]}

    router_node(state, {**runtime(), "router_model": Router()})
    assert "tool_registry" not in captured and "raw_structural_metrics" not in captured
    assert "available_evidence_requests" in captured


def test_length_errors_are_recognized_and_usage_is_statistics_only():
    assert is_length_finish_error(LLMOutputLengthError("length"))
    tracker = LLMUsageTracker()
    tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}})
    assert tracker.snapshot() == {"api_calls": 1, "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
