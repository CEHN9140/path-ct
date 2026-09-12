from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agents.subtype_review.graph import (
    available_evidence_requests,
    build_review_graph,
    completed_tool_keys,
    eligible_tools_for_requests,
    evidence_coverage,
    execute_tool_calls,
    initial_review_state,
    is_length_finish_error,
    partition_signature,
    router_node,
    save_review_outputs,
    validate_reports,
    validate_decision_state_evidence,
    validate_selected_tool_coverage,
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


def decision_state(**values):
    return {
        "identity": values.get("identity", "unassessed"),
        "structure": values.get("structure", "unassessed"),
        "alternative_explanation": values.get("alternative_explanation", "unassessed"),
        "uncertainty": values.get("uncertainty", "yes"),
    }


def test_state_has_evidence_memory_and_no_preselected_tools():
    state = state_for(("C1", ["P1"]), ("C2", ["P2"]))
    assert "evidence_memory" in state
    assert "pending_evidence_requests" in state["control"]
    assert state["control"]["next"] == "router"
    assert "router_request" not in state
    assert "pending_tools" not in state["control"]


def test_evidence_request_is_tool_free_and_normalizes_targets():
    request = EvidenceRequest(
        dimension="biological_support", target_ids=["C2", "C1", "C1"], question="Clarify signal."
    )
    assert request.target_ids == ["C1", "C2"]
    with pytest.raises(ValueError):
        RouterAction(action="accept", target_ids=["C1"], tool_name="pathway_enrichment")
    with pytest.raises(ValueError):
        RouterAction(action="drop", target_ids=["C1"])


def test_registry_uses_verifier_selectable_only():
    assert all("verifier_selectable" in metadata for metadata in TOOL_REGISTRY.values())
    assert all("default_every_round" not in metadata for metadata in TOOL_REGISTRY.values())
    assert TOOL_REGISTRY["clinical_characterization"]["verifier_selectable"] is False


def test_verifier_can_select_subset_with_explicit_target_ids():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    request = {"dimension": "biological_support", "target_ids": ["C1"], "question": "x"}
    state["control"].update({
        "pending_evidence_requests": [request],
        "eligible_tools": eligible_tools_for_requests(state, runtime(), [request]),
    })
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
    available = available_evidence_requests(state, runtime())
    assert available and all("tool_name" not in item for item in available)


def test_selected_tool_is_removed_from_next_evidence_availability():
    state = state_for(("C1", ["P1", "P2"]))
    request = {"dimension": "biological_support", "target_ids": ["C1"], "question": "x"}
    state["control"].update({
        "pending_evidence_requests": [request],
        "eligible_tools": eligible_tools_for_requests(state, runtime(), [request]),
    })
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
    state["control"]["eligible_tools"] = eligible_tools_for_requests(
        state, runtime(), state["control"]["pending_evidence_requests"]
    )
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
    state["control"]["eligible_tools"] = eligible_tools_for_requests(
        state, runtime(), state["control"]["pending_evidence_requests"]
    )
    assert set(state["control"]["eligible_tools"]) == {
        "multimodal_consistency_check",
    }


def test_router_plan_requires_complete_nonoverlapping_coverage():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    with pytest.raises(ValueError, match="cover every current set"):
        validate_router_plan(RouterPlan(actions=[{
            "action": "drop", "target_ids": ["C1"], "decision_state": decision_state()
        }]), state, runtime())


def test_router_stores_evidence_requests_not_tools():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))

    class Router:
        def invoke(self, payload):
            return {"actions": [
                {"action": "need_more_evidence", "target_ids": ["C1"], "evidence_requests": [{
                    "dimension": "biological_support", "target_ids": ["C1"], "question": "Clarify C1."
                }], "decision_state": decision_state()},
                {"action": "drop", "target_ids": ["C2"], "decision_state": decision_state()},
            ]}

    router_node(state, {**runtime(), "router_model": Router()})
    assert state["control"]["pending_evidence_requests"][0]["target_ids"] == ["C1"]
    assert state["control"]["next"] == "verifier_acquire"
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


def test_verifier_clears_request_after_audit_and_returns_to_router():
    state = state_for(("C1", ["P1", "P2"]))
    request = {"dimension": "biological_support", "target_ids": ["C1"], "question": "x"}
    state["control"].update({
        "pending_evidence_requests": [request],
        "next": "verifier_acquire",
    })

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                return {"tool_calls": [{
                    "name": "pathway_enrichment", "id": "call-1",
                    "args": {"target_ids": ["C1"]},
                }]}
            return {"reports": [{
                "dimension": "biological_support", "scope": "set_identity",
                "target_ids": ["C1"], "observations": [],
                "statistical_interpretation": "ok", "medical_interpretation": "ok",
                "limitations": [], "tool_refs": [],
            }]}

    values = {**runtime(), "verifier_model": Verifier()}
    verifier_node(state, values)
    assert state["control"]["next"] == "verifier_audit"
    verifier_node(state, values)
    assert state["control"]["pending_evidence_requests"] == []
    assert state["control"]["eligible_tools"] == {}
    assert state["control"]["next"] == "router"


def test_router_first_graph_supports_multiple_evidence_rounds():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    calls = []
    decisions = [
        {"dimension": "biological_support", "question": "Clarify biology."},
        {"dimension": "cross_modal_consistency", "question": "Clarify structure."},
        {"dimension": "confounder_exclusion", "question": "Clarify confounding."},
    ]

    class Router:
        def invoke(self, payload):
            calls.append("router")
            step = sum(item == "router" for item in calls)
            if step == 1:
                assert payload["evidence_reports"] == []
                assert any(
                    item["dimension"] == "biological_support"
                    and item["target_ids"] == ["C1"]
                    for item in payload["available_evidence_requests"]
                )
            action_state = decision_state()
            if step == 2:
                action_state = decision_state(identity="supported")
            elif step >= 3:
                action_state = decision_state(identity="supported", structure="compatible")
            if step >= 4:
                action_state = decision_state(
                    identity="supported", structure="compatible",
                    alternative_explanation="not_supported", uncertainty="no",
                )
                return {"actions": [
                    {"action": "accept", "target_ids": ["C1"], "decision_state": action_state},
                    {"action": "drop", "target_ids": ["C2"], "decision_state": decision_state()},
                ]}
            request = {**decisions[step - 1], "target_ids": ["C1"]}
            return {"actions": [
                {"action": "need_more_evidence", "target_ids": ["C1"],
                 "decision_state": action_state, "evidence_requests": [request]},
                {"action": "drop", "target_ids": ["C2"], "decision_state": decision_state()},
            ]}

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                calls.append("verifier")
                name = {
                    "biological_support": "pathway_enrichment",
                    "cross_modal_consistency": "multimodal_consistency_check",
                    "confounder_exclusion": "confound_test",
                }[payload["evidence_requests"][0]["dimension"]]
                return {"tool_calls": [{
                    "name": name, "id": name, "args": {"target_ids": ["C1"]},
                }]}
            dimension = payload["required_reports"][0]["dimension"]
            return {"reports": [{
                "dimension": dimension, "scope": "set_identity", "target_ids": ["C1"],
                "observations": [], "limitations": [], "tool_refs": [],
            }]}

    graph = build_review_graph()
    result = graph.invoke(
        state,
        context={**runtime(), "router_model": Router(), "verifier_model": Verifier()},
    )
    assert calls == ["router", "verifier", "router", "verifier", "router", "verifier", "router"]
    assert result["control"]["status"] == "complete"
    assert result["control"]["round"] == 4


def test_verifier_receives_router_evidence_request():
    state = state_for(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "confounder_exclusion",
        "target_ids": ["C1"],
        "question": "Could CT acquisition explain C1?",
    }]
    state["control"]["next"] = "verifier_acquire"
    captured = {}

    class Verifier:
        def invoke(self, payload):
            captured.update(payload)
            return {"tool_calls": [{
                "name": "confound_test", "id": "call-1", "args": {"target_ids": ["C1"]}
            }]}

    verifier_node(state, {**runtime(), "verifier_model": Verifier()})
    assert captured["evidence_requests"][0]["target_ids"] == ["C1"]
    assert captured["eligible_tools"]["confound_test"]["target_ids"] == ["C1"]


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


def test_graph_has_three_agents_and_starts_with_router():
    nodes = build_review_graph().get_graph().nodes
    assert {"verifier", "router", "reviser"}.issubset(nodes)
    assert "prepare_round" not in nodes


def test_router_payload_reports_unassessed_dimensions():
    state = state_for(("C1", ["P1", "P2"]))
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [{
                "action": "drop", "target_ids": ["C1"], "decision_state": decision_state()
            }]}

    router_node(state, {**runtime(), "router_model": Router()})
    assert captured["evidence_coverage"]["C1"]["biological_support"] == "unassessed"


def test_decision_state_cannot_claim_unobtained_evidence():
    state = state_for(("C1", ["P1", "P2"]))
    action = RouterAction(
        action="accept", target_ids=["C1"],
        decision_state=decision_state(identity="supported"),
    )
    with pytest.raises(ValueError, match="biological_support"):
        validate_decision_state_evidence(action, state, runtime())


def test_verifier_tool_calls_must_cover_each_request():
    request = {"dimension": "biological_support", "target_ids": ["C1"], "question": "x"}
    with pytest.raises(ValueError, match="do not cover EvidenceRequest"):
        validate_selected_tool_coverage([], [request], registry())


def test_partition_request_requires_a_partition_tool_call():
    request = {
        "dimension": "known_label_echo", "target_ids": [],
        "question": "Clarify stage/grade echo.",
    }
    with pytest.raises(ValueError, match="partition EvidenceRequest"):
        validate_selected_tool_coverage([], [request], registry())
    validate_selected_tool_coverage([{
        "name": "known_label_echo_test", "args": {"target_ids": []},
    }], [request], registry())


def test_router_plan_cannot_mix_evidence_and_revision():
    with pytest.raises(ValueError, match="cannot mix evidence acquisition"):
        RouterPlan(actions=[
            {"action": "need_more_evidence", "target_ids": ["C1"],
             "decision_state": decision_state(), "evidence_requests": [{
                 "dimension": "biological_support", "target_ids": ["C1"], "question": "x"
             }]},
            {"action": "split", "target_ids": ["C2"], "decision_state": decision_state(
                structure="incompatible"
            )},
        ])


def test_round_budget_returns_after_evidence_for_terminal_router_decision():
    state = state_for(("C1", ["P1", "P2"]))
    state["control"].update({
        "round": 10,
        "pending_evidence_requests": [{
            "dimension": "biological_support", "target_ids": ["C1"], "question": "x"
        }],
        "next": "verifier_acquire",
    })

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                return {"tool_calls": [{
                    "name": "pathway_enrichment", "id": "call-1",
                    "args": {"target_ids": ["C1"]},
                }]}
            return {"reports": [{
                "dimension": "biological_support", "scope": "set_identity",
                "target_ids": ["C1"], "observations": [], "limitations": [], "tool_refs": [],
            }]}

    values = {**runtime(), "verifier_model": Verifier()}
    verifier_node(state, values)
    verifier_node(state, values)
    assert state["control"]["next"] == "router"
    assert state["control"]["status"] == "reviewing"

    class Router:
        def invoke(self, payload):
            assert payload["terminal_only"] is True
            return {"actions": [{
                "action": "accept", "target_ids": ["C1"], "decision_state": decision_state()
            }]}

    router_node(state, {**runtime(), "router_model": Router()})
    assert state["control"]["status"] == "complete"
    assert state["control"]["round"] == 10


def test_router_payload_has_no_tool_registry_or_raw_metrics():
    state = state_for(("C1", ["P1", "P2"]))
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [{
                "action": "drop", "target_ids": ["C1"], "decision_state": decision_state()
            }]}

    router_node(state, {**runtime(), "router_model": Router()})
    assert "tool_registry" not in captured and "raw_structural_metrics" not in captured
    assert "available_evidence_requests" in captured


def test_length_errors_are_recognized_and_usage_is_statistics_only():
    assert is_length_finish_error(LLMOutputLengthError("length"))
    tracker = LLMUsageTracker()
    tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}})
    assert tracker.snapshot() == {"api_calls": 1, "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
