from __future__ import annotations

import json

import numpy as np
import pytest

from agents.subtype_review.graph import (
    build_review_graph,
    initial_review_state,
    is_length_finish_error,
    partition_signature,
    prepare_round_node,
    reviser_node,
    router_node,
    save_review_outputs,
    completed_tool_keys,
    compact_structural_index,
    expected_tool_refs_for_round,
    available_extra_evidence,
    required_reports_for_round,
    validate_reports,
    validate_tool_request,
    validate_router_plan,
    verifier_node,
)
from agents.subtype_review.llm import LLMOutputLengthError, LLMUsageTracker
from agents.subtype_review.schemas import EvidenceReport, EvidenceReportBatch, MergePlan, RouterAction, RouterPlan, ToolRequest
from agents.subtype_review.tools import TOOL_REGISTRY, clinical_characterization, compact_tool_result
from tools.cross_modal_structure import compute_structural_characterization, execute_split_membership


def make_state(*groups):
    return initial_review_state([
        {"cluster_id": name, "member_ids": members}
        for name, members in groups
    ])


def decision_fields(corroboration=False):
    return {}


def fake_registry():
    registry = {}
    for name, metadata in TOOL_REGISTRY.items():
        registry[name] = {**metadata}
        registry[name]["router_requestable"] = not metadata["default_every_round"]

        def fake_tool(*args, _name=name, **kwargs):
            return {
                "tool_name": _name,
                "status": "success",
                "results": {"summary": "ok", "decision_metrics": {}},
                "artifacts": {},
            }

        registry[name]["function"] = fake_tool
    return registry


def reports_for_requests(requests):
    if isinstance(requests, dict):
        return {"reports": [
            {
                "dimension": item["dimension"],
                "scope": item["scope"],
                "target_ids": item["target_ids"],
                "observations": [],
                "statistical_interpretation": "computed",
                "medical_interpretation": "computed",
                "limitations": [],
                "tool_refs": item["tool_names"],
            }
            for item in requests["required_reports"]
        ]}
    reports = []
    dimensions = sorted({TOOL_REGISTRY[item["tool_name"]]["dimension"] for item in requests})
    for dimension in dimensions:
        dimension_requests = [
            item for item in requests
            if TOOL_REGISTRY[item["tool_name"]]["dimension"] == dimension
        ]
        if any(item.get("target_ids") for item in dimension_requests):
            set_ids = sorted({
                target for item in dimension_requests for target in item.get("target_ids", [])
            })
            reports.extend({
                "dimension": dimension,
                "scope": "set_identity",
                "target_ids": [set_id],
                "observations": [],
                "statistical_interpretation": "computed",
                "medical_interpretation": "computed",
                "limitations": [],
                "tool_refs": sorted(
                    item["tool_name"] for item in dimension_requests
                    if set_id in item.get("target_ids", [])
                ),
                "metric_refs": [],
            } for set_id in set_ids)
        else:
            tool_names = [item["tool_name"] for item in dimension_requests]
            reports.append({
                "dimension": dimension,
                "scope": "partition",
                "target_ids": [],
                "observations": [],
                "statistical_interpretation": "computed",
                "medical_interpretation": "computed",
                "limitations": [],
                "tool_refs": sorted(tool_names),
                "metric_refs": [],
            })
    return {"reports": reports}


def test_state_has_one_partition_object_and_no_scientific_set_status():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    assert set(state) == {
        "partition", "round_evidence", "reports", "messages", "router_plan",
        "revision_plan", "revision_result", "history", "control",
    }
    assert [set(item) for item in state["partition"]["sets"]] == [
        {"set_id", "member_ids", "revision_lineage"},
        {"set_id", "member_ids", "revision_lineage"},
    ]


def test_reports_have_detailed_fields_and_no_free_analysis_or_status():
    report = EvidenceReport.model_validate({
        "dimension": "biological_support",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "observations": [{"metric": "q", "finding": "signal", "metric_refs": []}],
        "statistical_interpretation": "controlled",
        "medical_interpretation": "coherent",
        "limitations": [],
        "tool_refs": ["pathway_enrichment"],
        "metric_refs": [],
    })
    assert not hasattr(report, "analysis")
    assert not hasattr(report, "status")


def test_tool_registry_uses_real_names_and_separates_default_from_extra():
    assert "multimodal_consistency_check" in TOOL_REGISTRY
    assert all("function" in metadata for metadata in TOOL_REGISTRY.values())
    assert all(
        metadata["default_every_round"]
        for name, metadata in TOOL_REGISTRY.items()
        if name != "clinical_characterization"
    )
    assert not TOOL_REGISTRY["clinical_characterization"]["default_every_round"]


def test_clinical_characterization_is_not_router_requestable():
    assert TOOL_REGISTRY["clinical_characterization"]["router_requestable"] is False


def test_available_extra_evidence_excludes_completed_targets_and_partition_tools():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    tools = fake_registry()
    tools["extra_partition"] = {
        "tool_name": "extra_partition", "dimension": "biological_support",
        "default_every_round": False, "router_requestable": True,
        "scope": "partition", "function": lambda *args, **kwargs: {},
    }
    signature = partition_signature(state["partition"]["sets"])
    state["round_evidence"] = [
        {
            "tool_name": "clinical_characterization", "status": "success",
            "target_ids": ["C1"], "partition_signature": signature,
        },
        {
            "tool_name": "extra_partition", "status": "scientific_unavailable",
            "target_ids": [], "partition_signature": signature,
        },
    ]
    available = available_extra_evidence(state, {"tool_registry": tools})
    assert {tuple(item["target_ids"]) for item in available if item["tool_name"] == "clinical_characterization"} == {("C2",)}
    assert not any(item["tool_name"] == "extra_partition" for item in available)


def test_need_evidence_for_empty_availability_is_rejected():
    state = make_state(("C1", ["P1"]))
    tools = fake_registry()
    signature = partition_signature(state["partition"]["sets"])
    state["round_evidence"] = [{
        "tool_name": "clinical_characterization", "status": "success",
        "target_ids": ["C1"], "partition_signature": signature,
    }]
    plan = RouterPlan.model_validate({
        "actions": [{
            "action": "need_more_evidence", "target_ids": ["C1"],
            **decision_fields(),
            "tool_requests": [{
                "tool_name": "clinical_characterization", "target_ids": ["C1"],
            }],
        }],
    })
    with pytest.raises(ValueError, match="already succeeded|available"):
        validate_router_plan(plan, state, tools)


def test_router_validation_uses_one_correction_retry_only():
    state = make_state(("C1", ["P1", "P2"]))
    state["round_evidence"] = [{
        "tool_name": "clinical_characterization", "status": "success",
        "target_ids": ["C1"],
        "partition_signature": partition_signature(state["partition"]["sets"]),
    }]

    class Router:
        def __init__(self):
            self.calls = []

        def invoke(self, payload):
            self.calls.append(payload)
            if len(self.calls) == 1:
                return {"actions": [{
                    "action": "need_more_evidence", "target_ids": ["C1"],
                    **decision_fields(),
                    "tool_requests": [{
                        "tool_name": "clinical_characterization", "target_ids": ["C1"],
                    }],
                }]}
            return {"actions": [{"action": "drop", "target_ids": ["C1"], **decision_fields()}]}

    router = Router()
    router_node(state, {"tool_registry": fake_registry(), "router_model": router})
    assert state["control"]["round"] == 0
    assert state["control"]["router_correction_attempted"] is True
    router_node(state, {"tool_registry": fake_registry(), "router_model": router})
    assert len(router.calls) == 2
    assert router.calls[1]["validation_error"]
    assert router.calls[1]["available_extra_evidence"] == []
    assert router.calls[1]["instruction"] == "return a corrected RouterPlan only"
    assert state["control"]["status"] == "complete"


def test_router_second_validation_failure_fails_fast_without_third_call():
    state = make_state(("C1", ["P1", "P2"]))
    state["round_evidence"] = [{
        "tool_name": "clinical_characterization", "status": "success",
        "target_ids": ["C1"],
        "partition_signature": partition_signature(state["partition"]["sets"]),
    }]

    class Router:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return {"actions": [{
                "action": "need_more_evidence", "target_ids": ["C1"],
                **decision_fields(),
                "tool_requests": [{
                    "tool_name": "clinical_characterization", "target_ids": ["C1"],
                }],
            }]}

    router = Router()
    router_node(state, {"tool_registry": fake_registry(), "router_model": router})
    router_node(state, {"tool_registry": fake_registry(), "router_model": router})
    assert router.calls == 2
    assert state["control"]["status"] == "review_unavailable"


def test_router_requires_complete_nonoverlapping_coverage():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    plan = RouterPlan.model_validate({
        "actions": [{"action": "accept", "target_ids": ["C1"], **decision_fields(True)}],
    })
    with pytest.raises(ValueError, match="cover every current set"):
        validate_router_plan(plan, state, fake_registry())


def test_need_evidence_is_highest_priority_and_other_actions_do_not_execute():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))

    class Router:
        def invoke(self, payload):
            return {"actions": [
                {
                    "action": "need_more_evidence",
                    "target_ids": ["C1"],
                    **decision_fields(),
                    "tool_requests": [{
                        "tool_name": "clinical_characterization",
                        "target_ids": ["C1"],
                    }],
                },
                {"action": "accept", "target_ids": ["C2"], **decision_fields(True)},
            ]}

    router_node(state, {
        "tool_registry": fake_registry(),
        "router_model": Router(),
    })
    assert state["control"]["round"] == 1
    assert state["control"]["next"] == "prepare_round"
    assert state["partition"]["sets"][0] == {
        "set_id": "C1", "member_ids": ["P1"], "revision_lineage": []
    }
    assert state["control"]["extra_tool_requests"] == [{
        "tool_name": "clinical_characterization",
        "target_ids": ["C1"],
    }]


def test_default_tools_recompute_and_extra_tool_runs_in_next_round():
    class Verifier:
        def __init__(self):
            self.acquisitions = []

        def invoke(self, payload):
            if payload["mode"] == "acquire":
                self.acquisitions.append(payload["tool_requests"])
                return {
                    "tool_calls": [
                        {"name": item["tool_name"], "id": item["tool_name"]}
                        for item in payload["tool_requests"]
                    ]
                }
            return reports_for_requests(payload)

    class Router:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                return {"actions": [
                    {
                        "action": "need_more_evidence",
                        "target_ids": ["C1"],
                        **decision_fields(),
                        "tool_requests": [{
                            "tool_name": "clinical_characterization",
                            "target_ids": ["C1"],
                        }],
                    },
                    {"action": "drop", "target_ids": ["C2"], **decision_fields()},
                ]}
            return {"actions": [
                {"action": "drop", "target_ids": ["C1"], **decision_fields()},
                {"action": "drop", "target_ids": ["C2"], **decision_fields()},
            ]}

    verifier = Verifier()
    router = Router()
    runtime = {
        "data_root": "/tmp",
        "artifact_root": "/tmp",
        "config_dir": "configs",
        "patient_states_by_id": {},
        "tool_registry": fake_registry(),
        "verifier_model": verifier,
        "router_model": router,
        "reviser_model": object(),
    }
    result = build_review_graph().invoke(
        make_state(("C1", ["P1"]), ("C2", ["P2"])),
        context=runtime,
    )
    assert result["control"]["status"] == "complete"
    assert result["control"]["round"] == 2
    assert [item["tool_name"] for item in verifier.acquisitions[0]] == [
        "pathway_enrichment", "mutation_enrichment", "cnv_characterization",
        "multimodal_consistency_check", "confound_test", "known_label_echo_test",
    ]
    assert verifier.acquisitions[1][-1] == {
        "tool_name": "clinical_characterization",
        "target_ids": ["C1"],
    }
    assert len(result["history"]) == 2


def test_successful_extra_tool_in_current_round_cannot_be_requested_again():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["round_evidence"] = [{
        "tool_name": "clinical_characterization",
        "status": "success",
        "target_ids": ["C1", "C2"],
        "partition_signature": partition_signature(state["partition"]["sets"]),
    }]
    request = {
        "tool_name": "clinical_characterization",
        "target_ids": ["C1"],
    }
    assert completed_tool_keys(state)
    with pytest.raises(ValueError, match="already succeeded"):
        validate_tool_request(
            ToolRequest.model_validate(request), state,
            {"tool_registry": fake_registry()}, {"C1"},
        )


def test_scientific_unavailable_extra_tool_cannot_be_requested_again():
    state = make_state(("C1", ["P1"]))
    state["round_evidence"] = [{
        "tool_name": "clinical_characterization",
        "status": "scientific_unavailable",
        "target_ids": ["C1"],
        "partition_signature": partition_signature(state["partition"]["sets"]),
    }]
    with pytest.raises(ValueError, match="already succeeded"):
        validate_tool_request(
            ToolRequest(tool_name="clinical_characterization", target_ids=["C1"]),
            state,
            {"tool_registry": fake_registry()},
            {"C1"},
        )


def test_reports_reject_tool_used_for_another_set():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["control"]["pending_tools"] = [
        {"tool_name": "pathway_enrichment", "target_ids": ["C1", "C2"]},
        {"tool_name": "clinical_characterization", "target_ids": ["C1"]},
    ]
    state["round_evidence"] = [
        {"tool_name": "pathway_enrichment", "dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1", "C2"], "metric_refs": []},
        {"tool_name": "clinical_characterization", "dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1"], "metric_refs": []},
    ]
    batch = EvidenceReportBatch.model_validate({"reports": [
        {"dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1"], "observations": [], "tool_refs": ["clinical_characterization", "pathway_enrichment"], "metric_refs": []},
        {"dimension": "biological_support", "scope": "set_identity", "target_ids": ["C2"], "observations": [], "tool_refs": ["clinical_characterization", "pathway_enrichment"], "metric_refs": []},
    ]})
    with pytest.raises(ValueError, match="target-aware|target|tool_refs"):
        validate_reports(batch, state, fake_registry())


def test_reports_must_account_for_all_tools_used_for_each_set():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["control"]["pending_tools"] = [
        {"tool_name": "pathway_enrichment", "target_ids": ["C1", "C2"]},
        {"tool_name": "mutation_enrichment", "target_ids": ["C1", "C2"]},
    ]
    state["round_evidence"] = [
        {"tool_name": "pathway_enrichment", "dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1", "C2"], "metric_refs": []},
        {"tool_name": "mutation_enrichment", "dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1", "C2"], "metric_refs": []},
    ]
    batch = EvidenceReportBatch.model_validate({"reports": [
        {"dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1"], "observations": [], "tool_refs": ["pathway_enrichment"], "metric_refs": []},
        {"dimension": "biological_support", "scope": "set_identity", "target_ids": ["C2"], "observations": [], "tool_refs": ["pathway_enrichment", "mutation_enrichment"], "metric_refs": []},
    ]})
    with pytest.raises(ValueError, match="account|tool_refs"):
        validate_reports(batch, state, fake_registry())


def make_audit_state_with_clinical_extra():
    state = make_state(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    tools = fake_registry()
    state["control"]["pending_tools"] = [
        {"tool_name": name, "target_ids": [] if metadata["scope"] == "partition" else ["C1", "C2"]}
        for name, metadata in tools.items()
        if metadata["default_every_round"]
    ] + [{"tool_name": "clinical_characterization", "target_ids": ["C1"]}]
    state["control"]["next"] = "verifier_audit"
    signature = partition_signature(state["partition"]["sets"])
    state["round_evidence"] = [
        {
            "tool_name": request["tool_name"],
            "dimension": tools[request["tool_name"]]["dimension"],
            "scope": tools[request["tool_name"]]["scope"],
            "target_ids": request["target_ids"],
            "status": "success",
            "metrics": {},
            "metric_refs": [],
            "partition_signature": signature,
        }
        for request in state["control"]["pending_tools"]
    ]
    return state, tools


def test_expected_tool_refs_preserve_clinical_targeting():
    state, tools = make_audit_state_with_clinical_extra()
    expected = expected_tool_refs_for_round(state, {"tool_registry": tools})
    assert expected[("biological_support", "set_identity", "C1")] == [
        "clinical_characterization", "cnv_characterization",
        "mutation_enrichment", "pathway_enrichment",
    ]
    assert expected[("biological_support", "set_identity", "C2")] == [
        "cnv_characterization", "mutation_enrichment", "pathway_enrichment",
    ]


@pytest.mark.parametrize("returned_refs", [[], ["wrong_tool"]])
def test_verifier_overwrites_llm_tool_refs_before_strict_validation(returned_refs):
    state, tools = make_audit_state_with_clinical_extra()

    class Verifier:
        def invoke(self, payload):
            reports = reports_for_requests(payload)["reports"]
            for report in reports:
                report["tool_refs"] = returned_refs
            return {"reports": reports}

    state = verifier_node(state, {"tool_registry": tools, "verifier_model": Verifier()})
    assert state["control"]["next"] == "router"
    biology = {
        tuple(report["target_ids"]): report["tool_refs"]
        for report in state["reports"]
        if report["dimension"] == "biological_support"
    }
    assert biology[("C1",)] == [
        "clinical_characterization", "cnv_characterization",
        "mutation_enrichment", "pathway_enrichment",
    ]
    assert biology[("C2",)] == [
        "cnv_characterization", "mutation_enrichment", "pathway_enrichment",
    ]


def test_validate_reports_remains_strict_after_python_attachment():
    state, tools = make_audit_state_with_clinical_extra()
    reports = reports_for_requests({
        "required_reports": required_reports_for_round(state, tools),
    })["reports"]
    reports[0]["tool_refs"] = ["wrong_tool"]
    with pytest.raises(ValueError, match="tool_refs mismatch|unrequested tool"):
        validate_reports(EvidenceReportBatch.model_validate({"reports": reports}), state, tools)


def test_validate_reports_failure_does_not_repeat_audit():
    state = make_state(("C1", ["P1", "P2"]))
    calls = {"audit": 0}

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                return {"tool_calls": [
                    {"name": item["tool_name"], "id": item["tool_name"]}
                    for item in payload["tool_requests"]
                ]}
            calls["audit"] += 1
            reports = reports_for_requests(payload)["reports"]
            return {"reports": reports[:-1]}

    result = build_review_graph().invoke(state, context={
        "data_root": "/tmp", "artifact_root": "/tmp", "config_dir": "configs",
        "patient_states_by_id": {}, "tool_registry": fake_registry(),
        "verifier_model": Verifier(), "router_model": object(), "reviser_model": object(),
    })
    assert calls["audit"] == 1
    assert result["control"]["status"] == "review_unavailable"


def test_split_with_singleton_child_is_rejected_before_new_partition(monkeypatch, tmp_path):
    import tools.cross_modal_structure as cross_modal_structure

    monkeypatch.setattr(
        cross_modal_structure,
        "execute_split_membership",
        lambda *args, **kwargs: [["P1"], ["P2", "P3"]],
    )
    state = make_state(("C1", ["P1", "P2", "P3"]))
    state["router_plan"] = {
        "actions": [{"action": "split", "target_ids": ["C1"], **decision_fields(), "tool_requests": [], "reason": ""}]
    }
    state["round_evidence"] = [{
        "tool_name": "multimodal_consistency_check",
        "metrics": {"structural_characterization": {"internal_structure_by_set": {"C1": {}}}},
        "metric_refs": [],
    }]

    class Reviser:
        def invoke(self, payload):
            return {"split_plans": [{
                "target_id": "C1", "n_children": 2,
                "structural_basis": ["fused"],
                "execution_strategy": "fused_similarity_spectral",
                "metric_refs": [],
            }], "merge_plans": []}

    result = reviser_node(state, {
        "data_root": str(tmp_path),
        "tool_registry": fake_registry(),
        "reviser_model": Reviser(),
    })
    assert result["partition"]["sets"][0]["set_id"] == "C1"
    assert result["control"]["next"] == "reviser"
    assert "non-estimable" in result["control"]["error"]


def test_reviser_retry_uses_error_feedback_and_accepts_corrected_plan(monkeypatch, tmp_path):
    import tools.cross_modal_structure as cross_modal_structure

    def split_membership(output_root, member_ids, n_children, strategy, basis):
        if n_children == 3:
            return [["P1"], ["P2"], ["P3", "P4"]]
        return [["P1", "P2"], ["P3", "P4"]]

    monkeypatch.setattr(cross_modal_structure, "execute_split_membership", split_membership)
    state = make_state(("C1", ["P1", "P2", "P3", "P4"]))
    state["router_plan"] = {
        "actions": [{"action": "split", "target_ids": ["C1"], **decision_fields(), "tool_requests": [], "reason": ""}]
    }
    state["history"] = [{"revision_plan": None, "revision_result": None}]
    state["control"]["history_index"] = 0
    state["round_evidence"] = [{
        "tool_name": "multimodal_consistency_check",
        "metrics": {"structural_characterization": {"internal_structure_by_set": {"C1": {}}}},
        "metric_refs": [],
    }]

    class Reviser:
        def __init__(self):
            self.payloads = []

        def invoke(self, payload):
            self.payloads.append(payload)
            n_children = 3 if len(self.payloads) == 1 else 2
            return {"split_plans": [{
                "target_id": "C1", "n_children": n_children,
                "structural_basis": ["fused"],
                "execution_strategy": "fused_similarity_spectral",
                "metric_refs": [],
            }], "merge_plans": []}

    reviser = Reviser()
    runtime = {
        "data_root": str(tmp_path),
        "tool_registry": fake_registry(),
        "reviser_model": reviser,
    }
    first = reviser_node(state, runtime)
    assert first["control"]["next"] == "reviser"
    assert first["control"]["revision_validation_error"]
    second = reviser_node(first, runtime)
    assert len(reviser.payloads) == 2
    assert reviser.payloads[1]["previous_revision_plan"]["split_plans"][0]["n_children"] == 3
    assert "binary" in reviser.payloads[1]["revision_validation_error"]
    assert {item["set_id"] for item in second["partition"]["sets"]} == {"C1_S1", "C1_S2"}


def test_clinical_extra_tool_outputs_only_requested_targets(tmp_path):
    result = clinical_characterization(
        {"set_id": "C1", "member_ids": ["P1"]},
        {
            "P1": {"clinical": {"stage": "I", "grade": "2"}},
            "P2": {"clinical": {"stage": "III", "grade": "4"}},
        },
        str(tmp_path),
        all_cluster_states=[
            {"set_id": "C1", "member_ids": ["P1"]},
            {"set_id": "C2", "member_ids": ["P2"]},
        ],
        target_ids=["C1"],
    )
    assert set(result["results"]["decision_metrics"]["clinical_by_set"]) == {"C1"}


def test_same_set_level_extra_tool_requests_are_merged_before_verifier_call():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["control"]["max_rounds"] = 1

    class Router:
        def invoke(self, payload):
            return {"actions": [
                {
                    "action": "need_more_evidence",
                    "target_ids": ["C1"],
                    **decision_fields(),
                    "tool_requests": [{
                        "tool_name": "clinical_characterization",
                        "target_ids": ["C1"],
                    }],
                },
                {
                    "action": "need_more_evidence",
                    "target_ids": ["C2"],
                    **decision_fields(),
                    "tool_requests": [{
                        "tool_name": "clinical_characterization",
                        "target_ids": ["C2"],
                    }],
                },
            ]}

    router_node(state, {"tool_registry": fake_registry(), "router_model": Router()})
    prepare_round_node(state, {"tool_registry": fake_registry()})
    clinical = [
        request for request in state["control"]["pending_tools"]
        if request["tool_name"] == "clinical_characterization"
    ]
    assert clinical == [{
        "tool_name": "clinical_characterization",
        "target_ids": ["C1", "C2"],
    }]


def test_verifier_failure_retries_the_same_stage():
    class Verifier:
        def __init__(self):
            self.acquire_calls = 0

        def invoke(self, payload):
            if payload["mode"] == "acquire":
                self.acquire_calls += 1
                if self.acquire_calls == 1:
                    raise RuntimeError("temporary verifier failure")
                return {"tool_calls": [
                    {"name": item["tool_name"], "id": item["tool_name"]}
                    for item in payload["tool_requests"]
                ]}
            return reports_for_requests(payload)

    class Router:
        def invoke(self, payload):
            return {"actions": [{"action": "drop", "target_ids": ["C1"], **decision_fields()}]}

    verifier = Verifier()
    runtime = {
        "data_root": "/tmp", "artifact_root": "/tmp", "config_dir": "configs",
        "patient_states_by_id": {}, "tool_registry": fake_registry(),
        "verifier_model": verifier, "router_model": Router(), "reviser_model": object(),
    }
    result = build_review_graph().invoke(make_state(("C1", ["P1"])), context=runtime)
    assert verifier.acquire_calls == 2
    assert result["control"]["status"] == "complete"


def test_length_finish_reason_error_ends_review_without_retry():
    class LengthFinishReasonError(RuntimeError):
        pass

    class Verifier:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            raise LengthFinishReasonError("completion limit reached")

    verifier = Verifier()
    runtime = {
        "data_root": "/tmp", "artifact_root": "/tmp", "config_dir": "configs",
        "patient_states_by_id": {}, "tool_registry": fake_registry(),
        "verifier_model": verifier, "router_model": object(), "reviser_model": object(),
    }
    result = build_review_graph().invoke(make_state(("C1", ["P1"])), context=runtime)
    assert verifier.calls == 1
    assert result["control"]["status"] == "review_unavailable"
    assert result["control"]["next"] == "end"


def test_graph_recognizes_direct_sdk_length_error():
    assert is_length_finish_error(LLMOutputLengthError("too long"))


def test_round_ten_accept_drop_decision_completes_without_round_eleven():
    state = make_state(("C1", ["P1"]))
    state["control"]["round"] = 9

    class Router:
        def invoke(self, payload):
            return {"actions": [{"action": "drop", "target_ids": ["C1"], **decision_fields()}]}

    router_node(state, {
        "tool_registry": fake_registry(),
        "router_model": Router(),
    })
    assert state["control"]["round"] == 10
    assert state["control"]["status"] == "complete"
    assert state["partition"]["sets"][0]["set_id"] == "C1"


def test_round_ten_need_evidence_runs_extra_tool_then_stops_without_round_eleven():
    class Verifier:
        def __init__(self):
            self.acquisitions = []

        def invoke(self, payload):
            if payload["mode"] == "acquire":
                self.acquisitions.append([item["tool_name"] for item in payload["tool_requests"]])
                return {"tool_calls": [
                    {"name": item["tool_name"], "id": item["tool_name"]}
                    for item in payload["tool_requests"]
                ]}
            return reports_for_requests(payload)

    class Router:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return {"actions": [
                {
                    "action": "need_more_evidence",
                    "target_ids": ["C1"],
                    **decision_fields(),
                    "tool_requests": [{
                        "tool_name": "clinical_characterization",
                        "target_ids": ["C1"],
                    }],
                },
                {"action": "drop", "target_ids": ["C2"], **decision_fields()},
            ]}

    verifier, router = Verifier(), Router()
    runtime = {
        "data_root": "/tmp", "artifact_root": "/tmp", "config_dir": "configs",
        "patient_states_by_id": {}, "tool_registry": fake_registry(),
        "verifier_model": verifier, "router_model": router, "reviser_model": object(),
    }
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["control"]["max_rounds"] = 1
    result = build_review_graph().invoke(state, context=runtime)
    assert result["control"]["round"] == 1
    assert result["control"]["status"] == "review_incomplete_due_to_round_budget"
    assert router.calls == 1
    assert len(verifier.acquisitions) == 2
    assert verifier.acquisitions[1] == ["clinical_characterization"]


def test_structural_diagnostics_reports_one_binary_probe_and_modalities():
    matrix = np.full((6, 6), 0.05)
    for start in (0, 2, 4):
        matrix[start:start + 2, start:start + 2] = 0.95
    np.fill_diagonal(matrix, 1.0)
    result = compute_structural_characterization(
        matrix,
        {name: matrix for name in ("ct", "wsi", "rna", "genomic")},
        [f"P{i}" for i in range(6)],
        {"C1": [f"P{i}" for i in range(6)]},
        resampling_iterations=5,
    )
    internal = result["internal_structure_by_set"]["C1"]
    assert internal["fused_binary_probe"]["child_sizes"]
    assert set(internal["probe_support_by_modality"]) == {"ct", "wsi", "rna", "genomic"}
    assert "k_diagnostics" not in internal


def test_split_execution_uses_actual_fused_binary_probe(tmp_path):
    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    ids = [f"P{i}" for i in range(6)]
    (candidate / "affinity_patient_order.json").write_text(json.dumps(ids), encoding="utf-8")
    np.save(candidate / "fused_similarity.npy", np.eye(6))
    rna = np.full((6, 6), 0.05)
    wsi = np.full((6, 6), 0.05)
    for start in (0, 2, 4):
        rna[start:start + 2, start:start + 2] = 0.95
        wsi[start:start + 2, start:start + 2] = 0.95
    np.save(candidate / "rna_affinity.npy", rna)
    np.save(candidate / "wsi_affinity.npy", wsi)
    groups = execute_split_membership(
        str(tmp_path), ids, 2, "fused_similarity_spectral", ["fused"]
    )
    assert len(groups) == 2
    assert sorted(sum(groups, [])) == ids
    assert all(len(group) >= 2 for group in groups)


def test_structural_actions_are_not_python_gated_and_merge_is_pairwise():
    state = make_state(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    validate_router_plan(
        RouterPlan(actions=[
            {"action": "split", "target_ids": ["C1"], **decision_fields(), "tool_requests": [], "reason": ""},
            {"action": "accept", "target_ids": ["C2"], **decision_fields(True), "tool_requests": [], "reason": ""},
        ]),
        state,
        fake_registry(),
    )
    with pytest.raises(ValueError, match="exactly two"):
        RouterAction(action="merge", target_ids=["C1", "C2", "C3"], **decision_fields(), reason="")
    with pytest.raises(ValueError):
        MergePlan(target_ids=["C1", "C2", "C3"])


def test_reviser_is_called_once_for_whole_partition_and_parent_is_removed(tmp_path):
    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    ids = [f"P{i}" for i in range(6)]
    (candidate / "affinity_patient_order.json").write_text(json.dumps(ids), encoding="utf-8")
    matrix = np.full((6, 6), 0.05)
    for start in (0, 2, 4):
        matrix[start:start + 2, start:start + 2] = 0.95
    np.fill_diagonal(matrix, 1.0)
    np.save(candidate / "fused_similarity.npy", matrix)
    state = make_state(("C1", ids), ("C2", ["P6", "P7"]))
    state["router_plan"] = {
        "actions": [
            {"action": "split", "target_ids": ["C1"], **decision_fields(), "tool_requests": [], "reason": ""},
            {"action": "drop", "target_ids": ["C2"], **decision_fields(), "tool_requests": [], "reason": ""},
        ]
    }
    state["round_evidence"] = [{
        "tool_name": "multimodal_consistency_check",
        "metrics": {"structural_characterization": {"internal_structure_by_set": {"C1": {}}}},
        "metric_refs": [],
    }]

    class Reviser:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return {
                "split_plans": [{
                    "target_id": "C1",
                    "n_children": 2,
                    "structural_basis": ["fused"],
                    "execution_strategy": "fused_similarity_spectral",
                    "metric_refs": [],
                }],
                "merge_plans": [],
            }

    reviser = Reviser()
    result = reviser_node(state, {
        "data_root": str(tmp_path),
        "tool_registry": fake_registry(),
        "reviser_model": reviser,
    })
    assert reviser.calls == 1
    assert {item["set_id"] for item in result["partition"]["sets"]} == {
        "C1_S1", "C1_S2", "C2"
    }
    assert result["revision_result"]["superseded_sets"][0]["set_id"] == "C1"
    assert result["history"] == []


def test_accept_only_final_report_and_drop_ledger(tmp_path):
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["router_plan"] = {
        "actions": [
            {"action": "accept", "target_ids": ["C1"], **decision_fields(True), "tool_requests": [], "reason": "accepted"},
            {"action": "drop", "target_ids": ["C2"], **decision_fields(), "tool_requests": [], "reason": "excluded"},
        ]
    }
    state["reports"] = [{
        "dimension": "biological_support",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "observations": [],
        "statistical_interpretation": "ok",
        "medical_interpretation": "ok",
        "limitations": [],
        "tool_refs": ["pathway_enrichment"],
        "metric_refs": ["tool_results.pathway_enrichment.metrics.x"],
    }]
    state["control"]["status"] = "complete"
    summary = save_review_outputs(state, str(tmp_path), direct=True)
    assert len(summary["accepted_subtype_reports"]) == 1
    assert len(summary["dropped_set_registry"]) == 1
    assert summary["accepted_subtype_reports"][0]["sections"]


def test_usage_tracker_only_records_usage():
    tracker = LLMUsageTracker()
    for _ in range(12):
        tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}})
    assert tracker.snapshot() == {
        "api_calls": 12,
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_runtime_tool_failure_stops_before_router_and_is_not_scientific_drop():
    registry = fake_registry()

    def broken_tool(*args, **kwargs):
        return {
            "tool_name": "pathway_enrichment",
            "status": "failure",
            "results": {"decision_metrics": {}, "missing_reason": ""},
            "errors": ["I/O failure"],
        }

    registry["pathway_enrichment"]["function"] = broken_tool

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                return {"tool_calls": [
                    {"name": item["tool_name"], "id": item["tool_name"]}
                    for item in payload["tool_requests"]
                ]}
            raise AssertionError("runtime failure must not reach audit")

    class Router:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return {"actions": [{"action": "drop", "target_ids": ["C1"], **decision_fields()}]}

    router = Router()
    runtime = {
        "data_root": "/tmp", "artifact_root": "/tmp", "config_dir": "configs",
        "patient_states_by_id": {}, "tool_registry": registry,
        "verifier_model": Verifier(), "router_model": router, "reviser_model": object(),
    }
    state = make_state(("C1", ["P1"]))
    state["control"]["max_failures"] = 1
    result = build_review_graph().invoke(state, context=runtime)
    assert result["control"]["status"] == "review_unavailable"
    assert router.calls == 0


def test_leaf_metric_refs_are_stable_and_specific():
    result = compact_tool_result({
        "status": "success",
        "results": {"decision_metrics": {"C1": {"q_value": 0.01, "effect": {"delta": 2}}}},
    }, "pathway_enrichment")
    assert result["metric_refs"] == [
        "tool_results.pathway_enrichment.metrics.C1.effect.delta",
        "tool_results.pathway_enrichment.metrics.C1.q_value",
    ]


def test_tool_status_separates_scientific_unavailability_from_runtime_failure():
    assert compact_tool_result({
        "status": "failure",
        "results": {"missing_reason": "no RNA table"},
        "errors": [],
    }, "pathway_enrichment")["status"] == "scientific_unavailable"
    assert compact_tool_result({
        "status": "failure",
        "results": {"missing_reason": ""},
        "errors": ["I/O failure"],
    }, "pathway_enrichment")["status"] == "runtime_failure"


def test_verifier_can_cite_one_leaf_metric_ref():
    state = make_state(("C1", ["P1"]))
    compact = compact_tool_result({
        "status": "success",
        "results": {"decision_metrics": {"C1": {"q_value": 0.01}}},
    }, "pathway_enrichment")
    state["round_evidence"] = [{
        **compact,
        "dimension": "biological_support",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "partition_signature": "p",
    }]
    state["control"]["pending_tools"] = [{
        "tool_name": "pathway_enrichment",
        "target_ids": ["C1"],
    }]
    validate_reports(EvidenceReportBatch.model_validate({"reports": [{
        "dimension": "biological_support",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "observations": [{
            "metric": "q_value",
            "finding": "significant",
            "metric_refs": ["tool_results.pathway_enrichment.metrics.C1.q_value"],
        }],
        "tool_refs": ["pathway_enrichment"],
        "metric_refs": [],
    }]}), state, fake_registry())


def test_graph_has_prepare_round_python_node_and_no_init_agent_node():
    graph = build_review_graph()
    assert "prepare_round" in graph.get_graph().nodes
    assert "init_agent" not in graph.get_graph().nodes


def test_router_payload_uses_reports_not_raw_structural_metrics():
    state = make_state(("C1", ["P1"]))
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [{"action": "drop", "target_ids": ["C1"], **decision_fields()}]}

    router_node(state, {"tool_registry": fake_registry(), "router_model": Router()})
    assert "raw_structural_metrics" not in captured


def test_router_payload_contains_compact_structural_index_without_action_flags():
    state = make_state(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    state["round_evidence"] = [{
        "tool_name": "multimodal_consistency_check",
        "full_metrics": {"structural_characterization": {
            "internal_structure_by_set": {
                "C1": {"member_n": 2, "fused_binary_probe": {
                    "child_sizes": [1, 1], "median_silhouette": 0.01,
                    "normalized_cut": 0.5, "resampling": {
                        "median_resample_ari": 0.8,
                        "consensus_separation": 0.7,
                        "pac": 0.2,
                        "degenerate_resample_fraction": 0.0,
                    }}, "probe_support_by_modality": {
                        name: {"median_silhouette": 0.1}
                        for name in ("ct", "wsi", "rna", "genomic")
                    }},
                "C2": {"member_n": 2},
            },
            "boundary_by_pair": {"C1+C2": {"fused": {
                "pair_median_silhouette": 0.02,
                "left_median_margin": 0.1,
                "right_median_margin": 0.2,
                "left_boundary_separation": 0.3,
                "right_boundary_separation": 0.4,
            }, "modalities": {}}},
        }},
    }]
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [
                {"action": "drop", "target_ids": ["C1"], **decision_fields()},
                {"action": "drop", "target_ids": ["C2"], **decision_fields()},
            ]}

    router_node(state, {"tool_registry": fake_registry(), "router_model": Router()})
    index = captured["structural_index"]
    assert index["per_set"]["C1"]["binary_probe"]["median_ari"] == 0.8
    assert index["boundary_by_pair"][0]["pair"] == ["C1", "C2"]
    assert "split_candidate" not in json.dumps(index)
    assert "merge_candidate" not in json.dumps(index)
