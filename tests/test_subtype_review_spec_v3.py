from __future__ import annotations

import json

import numpy as np
import pytest

from agents.subtype_review.graph import (
    apply_split,
    build_review_graph,
    initial_review_state,
    reset_after_structural_change,
    reviser_node,
    router_node,
    save_review_outputs,
    subject_signature,
    validate_router_output,
)
from agents.subtype_review.llm import LLMUsageTracker
from agents.subtype_review.schemas import EvidenceReport, RouterOutput
from tools.structural_adequacy import execute_split_membership, structure_diagnostics


def make_state(*groups):
    return initial_review_state([
        {"cluster_id": name, "member_ids": members}
        for name, members in groups
    ])


def test_evidence_report_is_detailed_and_has_no_status_label():
    report = EvidenceReport.model_validate({
        "dimension": "biological_support",
        "scope": "set_identity",
        "analysis": "round_validation",
        "target_ids": ["C1"],
        "observations": [{"metric": "q", "finding": "FDR-controlled signal"}],
        "statistical_interpretation": "The signal is reproducible.",
        "medical_interpretation": "The pattern is biologically coherent.",
    })
    assert not hasattr(report, "status")


def test_router_output_must_cover_every_current_set():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    output = RouterOutput.model_validate({
        "actions": [{"action": "accept", "target_ids": ["C1"]}],
    })
    with pytest.raises(ValueError, match="every current set"):
        validate_router_output(output, state)


def test_need_evidence_has_priority_and_tentative_actions_are_not_applied():
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))

    class Router:
        def invoke(self, payload):
            return {
                "actions": [
                    {
                        "action": "need_more_evidence",
                        "target_ids": ["C1"],
                        "requests": [{
                            "dimension": "biological_support",
                            "scope": "set_identity",
                            "analysis": "mutation_followup",
                            "target_ids": ["C1"],
                        }],
                    },
                    {"action": "accept", "target_ids": ["C2"]},
                ]
            }

    router_node(state, {}, Router())
    assert state["control"]["round"] == 1
    assert state["control"]["next"] == "supplement_acquire"
    assert all(item["status"] == "active" for item in state["sets"])
    assert state["control"]["pending_requests"][0]["analysis"] == "mutation_followup"


def test_successful_supplement_request_cannot_be_repeated():
    state = make_state(("C1", ["P1", "P2"]))
    signature = subject_signature(
        "biological_support", "set_identity", state["sets"], ["C1"], "followup"
    )
    state["evidence_history"].append({
        "raw": [{
            "dimension": "biological_support",
            "scope": "set_identity",
            "analysis": "followup",
            "target_ids": ["C1"],
            "subject_signature": signature,
            "status": "success",
        }],
        "reports": [],
    })
    output = RouterOutput.model_validate({
        "actions": [{
            "action": "need_more_evidence",
            "target_ids": ["C1"],
            "requests": [{
                "dimension": "biological_support",
                "scope": "set_identity",
                "analysis": "followup",
                "target_ids": ["C1"],
            }],
        }],
    })
    with pytest.raises(ValueError, match="already succeeded"):
        validate_router_output(output, state)


def test_router_round_counts_only_successful_router_decision():
    state = make_state(("C1", ["P1"]))
    state["reports"] = [{"dimension": "cross_modal_consistency", "scope": "set_identity", "target_ids": ["C1"]}]
    state["round_evidence"] = [{
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "analysis": "round_validation",
        "target_ids": ["C1"],
        "subject_signature": subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"]),
        "results": [{"tool_name": "multimodal_consistency_check", "metrics": {
            "identity_evidence_level_by_set": {"C1": "concordant"},
            "internal_structure_by_set": {"C1": {"positive_internal_heterogeneity": False}},
            "positive_weak_boundary_pairs": {},
        }}],
        "status": "success",
    }]

    class Router:
        def invoke(self, payload):
            return {"actions": [{"action": "accept", "target_ids": ["C1"]}]}

    router_node(state, {}, Router())
    assert state["control"]["round"] == 1
    assert state["sets"][0]["status"] == "accept"


def test_round_ten_router_decision_executes_but_no_round_eleven():
    state = make_state(("C1", ["P1"]))
    state["control"]["round"] = 9
    state["reports"] = [{"dimension": "cross_modal_consistency", "scope": "set_identity", "target_ids": ["C1"]}]
    state["round_evidence"] = [{
        "dimension": "cross_modal_consistency", "scope": "set_identity", "analysis": "round_validation",
        "target_ids": ["C1"], "subject_signature": subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"]),
        "results": [{"tool_name": "multimodal_consistency_check", "metrics": {
            "identity_evidence_level_by_set": {"C1": "concordant"},
            "internal_structure_by_set": {"C1": {"positive_internal_heterogeneity": False}},
            "positive_weak_boundary_pairs": {},
        }}], "status": "success",
    }]

    class Router:
        def invoke(self, payload):
            return {"actions": [{"action": "accept", "target_ids": ["C1"]}]}

    router_node(state, {}, Router())
    assert state["control"]["round"] == 10
    assert state["control"]["status"] == "complete"


def test_each_router_round_gets_fresh_full_partition_validation(monkeypatch):
    import agents.subtype_review.graph as graph_module

    def fake_validation(*args, **kwargs):
        dimension = kwargs["dimension"]
        metrics = {}
        if dimension == "cross_modal_consistency":
            metrics = {
                "identity_evidence_level_by_set": {"C1": "concordant", "C2": "concordant"},
                "internal_structure_by_set": {
                    "C1": {"positive_internal_heterogeneity": False},
                    "C2": {"positive_internal_heterogeneity": False},
                },
                "positive_weak_boundary_pairs": {},
            }
        return {
            "tool_name": dimension,
            "status": "success",
            "results": [{"tool_name": dimension, "status": "success", "metrics": metrics}],
        }

    monkeypatch.setattr(
        graph_module,
        "VALIDATION_FUNCTIONS",
        {name: (lambda *args, _name=name, **kwargs: fake_validation(*args, dimension=_name, **kwargs))
         for name in graph_module.EVIDENCE_DIMENSIONS},
    )

    class Verifier:
        def __init__(self):
            self.acquisitions = []

        def invoke(self, payload):
            if payload["mode"] == "acquire":
                self.acquisitions.append(payload["requests"])
                return {"tool_calls": [{"name": row["dimension"], "id": row["dimension"]} for row in payload["requests"]]}
            return {"reports": [{
                "dimension": row["dimension"],
                "scope": row["scope"],
                "analysis": row["analysis"],
                "target_ids": row["target_ids"],
                "observations": [],
                "statistical_interpretation": "computed",
                "medical_interpretation": "computed",
                "limitations": [],
                "metric_refs": [],
            } for row in payload["requests"]]}

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
                        "requests": [{
                            "dimension": "biological_support",
                            "scope": "set_identity",
                            "analysis": "mutation_followup",
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
    graph = build_review_graph(
        verifier_model=verifier,
        router_model=router,
        reviser_model=object(),
        runtime={"data_root": "/tmp", "output_root": "/tmp", "config_dir": "configs", "patient_states_by_id": {}},
    )
    result = graph.invoke(initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1"]},
        {"cluster_id": "C2", "member_ids": ["P2"]},
    ]))
    assert result["control"]["status"] == "complete"
    assert result["control"]["round"] == 2
    assert [len(requests) for requests in verifier.acquisitions] == [4, 1, 4]


def test_structural_diagnostics_has_all_k_and_all_modality_metrics():
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
    assert not any(key.startswith("recommended") for key in result["internal_structure_by_set"]["C1"])


def test_multimodal_split_execution_uses_declared_basis(tmp_path):
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


def test_reviser_returns_one_plan_and_python_executes_membership(tmp_path):
    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    ids = [f"P{i}" for i in range(6)]
    (candidate / "affinity_patient_order.json").write_text(json.dumps(ids), encoding="utf-8")
    matrix = np.full((6, 6), 0.05)
    for start in (0, 2, 4):
        matrix[start:start + 2, start:start + 2] = 0.95
    np.fill_diagonal(matrix, 1.0)
    np.save(candidate / "fused_similarity.npy", matrix)
    state = make_state(("C1", ids))
    state["round_evidence"] = [{
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "analysis": "round_validation",
        "target_ids": ["C1"],
        "results": [{"tool_name": "multimodal_consistency_check", "metrics": {
            "internal_structure_by_set": {"C1": {"positive_internal_heterogeneity": True}},
        }}],
        "status": "success",
    }]
    state["revision"] = {
        "pending_actions": [{"action": "split", "target_ids": ["C1"]}],
        "plans": [],
        "partition_signature": "before",
    }
    state["control"]["next"] = "revise"

    class Reviser:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            return {
                "action": "split",
                "target_ids": ["C1"],
                "n_children": 3,
                "structural_basis": ["fused"],
                "execution_strategy": "fused_similarity_spectral",
                "metric_refs": [],
            }

    reviser = Reviser()
    result = reviser_node(
        state,
        {"data_root": str(tmp_path), "output_root": str(tmp_path)},
        reviser,
    )
    assert reviser.calls == 1
    assert result["revision"] is None
    assert len([item for item in result["sets"] if item["status"] == "active"]) == 3


def test_split_parent_is_history_and_children_can_be_split_again():
    state = make_state(("C1", ["P1", "P2", "P3", "P4"]))
    apply_split(state, "C1", [["P1", "P2"], ["P3", "P4"]], {"action": "split", "target_ids": ["C1"]})
    child = next(item for item in state["sets"] if item["set_id"] == "C1_S1")
    apply_split(state, "C1_S1", [["P1"], ["P2"]], {"action": "split", "target_ids": ["C1_S1"]})
    assert next(item for item in state["sets"] if item["set_id"] == "C1")["status"] == "superseded_by_split"
    assert child["status"] == "superseded_by_split"
    assert {item["set_id"] for item in state["sets"] if item["status"] == "active"} == {"C1_S1_S1", "C1_S1_S2", "C1_S2"}


def test_terminal_drop_and_accept_only_report(tmp_path):
    state = make_state(("C1", ["P1"]), ("C2", ["P2"]))
    state["reports"] = [{"dimension": "confounder_exclusion", "scope": "set_identity", "target_ids": ["C1", "C2"]}]
    class Router:
        def invoke(self, payload):
            return {"actions": [
                {"action": "drop", "target_ids": ["C1"]},
                {"action": "drop", "target_ids": ["C2"]},
            ]}

    router_node(state, {}, Router())
    assert state["control"]["status"] == "complete"
    assert {item["status"] for item in state["sets"]} == {"drop"}
    summary = save_review_outputs(state, str(tmp_path), direct=True)
    assert summary["accepted_subtype_reports"] == []
    assert len(summary["dropped_set_registry"]) == 2


def test_accept_only_report_contains_lineage_dimensions_and_metric_refs(tmp_path):
    state = make_state(("C1", ["P1"]))
    state["sets"][0]["status"] = "accept"
    state["sets"][0]["revision_lineage"] = [{"action": "split", "target_ids": ["C0"]}]
    state["reports"] = [{
        "dimension": "biological_support",
        "scope": "set_identity",
        "analysis": "round_validation",
        "target_ids": ["C1"],
        "metric_refs": ["tool_results.rna.metrics.q_value"],
        "observations": [{"metric": "q", "finding": "signal", "metric_refs": ["tool_results.rna.metrics.q_value"]}],
    }]
    summary = save_review_outputs(state, str(tmp_path), direct=True)
    report = summary["accepted_subtype_reports"][0]
    assert report["membership"] == ["P1"]
    assert report["revision_lineage"]
    assert set(report["evidence_by_dimension"]) == {
        "biological_support", "cross_modal_consistency", "confounder_exclusion", "known_label_echo",
    }
    assert report["metric_refs"] == ["tool_results.rna.metrics.q_value"]


def test_usage_tracker_only_counts_calls_and_tokens():
    tracker = LLMUsageTracker()
    for _ in range(12):
        tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}})
    assert tracker.snapshot() == {"api_calls": 12, "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
