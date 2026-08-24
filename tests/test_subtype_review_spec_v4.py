from __future__ import annotations

import json

import numpy as np
import pytest

from agents.subtype_review.graph import (
    build_review_graph,
    initial_review_state,
    partition_signature,
    prepare_round_node,
    reviser_node,
    router_node,
    save_review_outputs,
    successful_tool_keys,
    validate_reports,
    validate_tool_request,
    validate_router_plan,
)
from agents.subtype_review.llm import LLMUsageTracker
from agents.subtype_review.schemas import EvidenceReport, EvidenceReportBatch, RouterPlan, ToolRequest
from agents.subtype_review.tools import TOOL_REGISTRY, clinical_characterization, compact_tool_result
from tools.structural_adequacy import execute_split_membership, structure_diagnostics


def make_state(*groups):
    return initial_review_state([
        {"cluster_id": name, "member_ids": members}
        for name, members in groups
    ])


def fake_registry():
    registry = {}
    for name, metadata in TOOL_REGISTRY.items():
        registry[name] = {**metadata}

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
    by_dimension = {}
    for request in requests:
        by_dimension.setdefault(request["tool_name"], request)
    grouped = {}
    for request in requests:
        grouped.setdefault(request["tool_name"], request)
    reports = []
    dimensions = {}
    for tool_name, request in grouped.items():
        metadata = TOOL_REGISTRY[tool_name]
        dimensions.setdefault(metadata["dimension"], []).append(tool_name)
    set_ids = sorted({
        target
        for request in requests
        for target in request.get("target_ids", [])
    })
    for dimension, tool_names in dimensions.items():
        request = next(item for item in requests if TOOL_REGISTRY[item["tool_name"]]["dimension"] == dimension)
        if request.get("target_ids"):
            reports.extend({
                "dimension": dimension,
                "scope": "set_identity",
                "target_ids": [set_id],
                "observations": [],
                "statistical_interpretation": "computed",
                "medical_interpretation": "computed",
                "limitations": [],
                "tool_refs": sorted(tool_names),
                "metric_refs": [],
            } for set_id in set_ids)
        else:
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


def test_router_requires_complete_nonoverlapping_coverage():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    plan = RouterPlan.model_validate({
        "actions": [{"action": "accept", "target_ids": ["C1"]}],
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
                    "tool_requests": [{
                        "tool_name": "clinical_characterization",
                        "target_ids": ["C1"],
                    }],
                },
                {"action": "accept", "target_ids": ["C2"]},
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
            return reports_for_requests(payload["tool_requests"])

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
                        "tool_requests": [{
                            "tool_name": "clinical_characterization",
                            "target_ids": ["C1"],
                        }],
                    },
                    {"action": "drop", "target_ids": ["C2"]},
                ]}
            return {"actions": [
                {"action": "drop", "target_ids": ["C1"]},
                {"action": "drop", "target_ids": ["C2"]},
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
    assert successful_tool_keys(state)
    with pytest.raises(ValueError, match="already succeeded"):
        validate_tool_request(ToolRequest.model_validate(request), state, fake_registry(), {"C1"})


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
                    "tool_requests": [{
                        "tool_name": "clinical_characterization",
                        "target_ids": ["C1"],
                    }],
                },
                {
                    "action": "need_more_evidence",
                    "target_ids": ["C2"],
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
            return reports_for_requests(payload["tool_requests"])

    class Router:
        def invoke(self, payload):
            return {"actions": [{"action": "drop", "target_ids": ["C1"]}]}

    verifier = Verifier()
    runtime = {
        "data_root": "/tmp", "artifact_root": "/tmp", "config_dir": "configs",
        "patient_states_by_id": {}, "tool_registry": fake_registry(),
        "verifier_model": verifier, "router_model": Router(), "reviser_model": object(),
    }
    result = build_review_graph().invoke(make_state(("C1", ["P1"])), context=runtime)
    assert verifier.acquire_calls == 2
    assert result["control"]["status"] == "complete"


def test_round_ten_accept_drop_decision_completes_without_round_eleven():
    state = make_state(("C1", ["P1"]))
    state["control"]["round"] = 9

    class Router:
        def invoke(self, payload):
            return {"actions": [{"action": "drop", "target_ids": ["C1"]}]}

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
            return reports_for_requests(payload["tool_requests"])

    class Router:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return {"actions": [
                {
                    "action": "need_more_evidence",
                    "target_ids": ["C1"],
                    "tool_requests": [{
                        "tool_name": "clinical_characterization",
                        "target_ids": ["C1"],
                    }],
                },
                {"action": "drop", "target_ids": ["C2"]},
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


def test_structural_diagnostics_reports_all_candidate_k_and_modalities():
    matrix = np.full((6, 6), 0.05)
    for start in (0, 2, 4):
        matrix[start:start + 2, start:start + 2] = 0.95
    np.fill_diagonal(matrix, 1.0)
    result = structure_diagnostics(
        {name: matrix for name in ("ct", "wsi", "rna", "genomic")},
        [f"P{i}" for i in range(6)],
        {"C1": [f"P{i}" for i in range(6)]},
    )
    rows = result["internal_structure_by_set"]["C1"]["k_diagnostics"]
    assert [row["k"] for row in rows] == [2, 3, 4, 5]
    assert all(set(row["modalities"]) == {"fused", "ct", "wsi", "rna", "genomic"} for row in rows)
    assert all("subsampling_stability" in row["modalities"]["ct"] for row in rows)
    assert not any(key.startswith("recommended") for key in result["internal_structure_by_set"]["C1"])


def test_declared_multimodal_basis_changes_split_execution(tmp_path):
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
    assert execute_split_membership(
        str(tmp_path), ids, 3, "multimodal_consensus", ["rna", "wsi"]
    ) == [["P0", "P1"], ["P2", "P3"], ["P4", "P5"]]


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
            {"action": "split", "target_ids": ["C1"], "tool_requests": [], "reason": ""},
            {"action": "drop", "target_ids": ["C2"], "tool_requests": [], "reason": ""},
        ]
    }
    state["round_evidence"] = [{
        "tool_name": "multimodal_consistency_check",
        "metrics": {"internal_structure_by_set": {"C1": {"positive_internal_heterogeneity": True}}},
        "metric_refs": [],
    }]

    class Reviser:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            return {
                "split_plans": [{
                    "target_id": "C1",
                    "n_children": 3,
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
        "C1_S1", "C1_S2", "C1_S3", "C2"
    }
    assert result["revision_result"]["superseded_sets"][0]["set_id"] == "C1"
    assert result["history"] == []


def test_accept_only_final_report_and_drop_ledger(tmp_path):
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["router_plan"] = {
        "actions": [
            {"action": "accept", "target_ids": ["C1"], "tool_requests": [], "reason": "accepted"},
            {"action": "drop", "target_ids": ["C2"], "tool_requests": [], "reason": "excluded"},
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
            return {"actions": [{"action": "drop", "target_ids": ["C1"]}]}

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
            return {"actions": [{"action": "drop", "target_ids": ["C1"]}]}

    router_node(state, {"tool_registry": fake_registry(), "router_model": Router()})
    assert "raw_structural_metrics" not in captured
