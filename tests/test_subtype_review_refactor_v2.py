from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents.subtype_review.graph import (
    apply_merge,
    apply_split,
    build_review_graph,
    initial_review_state,
    reset_after_structural_change,
    subject_signature,
    validate_router_output,
)
from agents.subtype_review.schemas import (
    EvidenceReport,
    EvidenceReportBatch,
    ReviserOutput,
    RouterOutput,
)


def state_with_sets(*groups):
    return initial_review_state([
        {"cluster_id": name, "member_ids": members}
        for name, members in groups
    ])


def test_evidence_report_contains_interpretation_not_status_label():
    report = EvidenceReport.model_validate({
        "dimension": "biological_support",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "observations": [{"metric": "q", "finding": "FDR-controlled signal", "metric_refs": ["r1"]}],
        "statistical_interpretation": "The signal is reproducible.",
        "medical_interpretation": "The pattern is biologically coherent.",
        "limitations": ["CNV coverage is incomplete."],
        "metric_refs": ["r1"],
    })
    assert report.target_ids == ["C1"]
    assert not hasattr(report, "status")


def test_router_requests_only_the_missing_dimension():
    state = state_with_sets(("C1", ["P1", "P2"]))
    output = RouterOutput.model_validate({
        "actions": [{
            "action": "need_more_evidence",
            "target_ids": ["C1"],
            "requests": [{
                "dimension": "confounder_exclusion",
                "scope": "set_identity",
                "target_ids": ["C1"],
            }],
            "reason": "Acquisition confounding has not been evaluated.",
        }],
    })
    validate_router_output(output, state)
    assert output.actions[0].requests[0].dimension == "confounder_exclusion"


def test_repeated_evidence_request_is_rejected():
    state = state_with_sets(("C1", ["P1", "P2"]))
    from agents.subtype_review.graph import subject_signature

    signature = subject_signature("confounder_exclusion", "set_identity", state["sets"], ["C1"])
    state["evidence"]["raw"].append({
        "dimension": "confounder_exclusion",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "subject_signature": signature,
        "results": [],
    })
    output = RouterOutput.model_validate({
        "actions": [{
            "action": "need_more_evidence",
            "target_ids": ["C1"],
            "requests": [{
                "dimension": "confounder_exclusion",
                "scope": "set_identity",
                "target_ids": ["C1"],
            }],
        }],
    })
    with pytest.raises(ValueError, match="already exists"):
        validate_router_output(output, state)


def test_one_set_cannot_have_two_actions_but_independent_actions_are_valid():
    state = state_with_sets(
        ("C1", ["P1", "P2"]),
        ("C2", ["P3", "P4"]),
    )
    state["evidence"]["raw"].append({
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "target_ids": [],
        "subject_signature": "",
        "results": [{
            "tool_name": "multimodal_consistency_check",
            "status": "success",
            "metrics": {
                "internal_structure_by_set": {
                    "C1": {"positive_internal_heterogeneity": True},
                },
                "positive_weak_boundary_pairs": {"C1+C2": ["ct", "rna"]},
            },
        }],
    })
    valid = RouterOutput.model_validate({
        "actions": [
            {"action": "split", "target_ids": ["C1"]},
            {"action": "merge", "target_ids": ["C2", "C1"]},
        ]
    })
    with pytest.raises(ValueError, match="only one action"):
        validate_router_output(valid, state)


def test_accept_drop_split_merge_can_be_valid_for_disjoint_sets():
    state = state_with_sets(
        ("C1", ["P1", "P2"]),
        ("C2", ["P3", "P4"]),
        ("C3", ["P5", "P6"]),
    )
    for name in ("C1", "C2", "C3"):
        for dimension in ("biological_support", "cross_modal_consistency", "confounder_exclusion", "known_label_echo"):
            state["reports"].append({
                "dimension": dimension,
                "scope": "set_identity" if dimension != "known_label_echo" else "partition",
                "target_ids": [name] if dimension != "known_label_echo" else [],
            })
    state["evidence"]["raw"].append({
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "subject_signature": subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"]),
        "results": [{
            "tool_name": "multimodal_consistency_check",
            "status": "success",
            "metrics": {
                "internal_structure_by_set": {"C2": {"positive_internal_heterogeneity": True}},
                "positive_weak_boundary_pairs": {"C1+C3": ["ct", "rna"]},
                "identity_supporting_modalities_by_set": {"C1": ["ct", "wsi"]},
            },
        }],
    })
    output = RouterOutput.model_validate({
        "actions": [
            {"action": "accept", "target_ids": ["C1"]},
            {"action": "split", "target_ids": ["C2"]},
            {"action": "merge", "target_ids": ["C3", "C1"]},
        ]
    })
    with pytest.raises(ValueError, match="only one action"):
        validate_router_output(output, state)


def test_router_can_select_accept_split_and_merge_in_one_round_when_disjoint():
    state = state_with_sets(
        ("C1", ["P1", "P2"]),
        ("C2", ["P3", "P4", "P5"]),
        ("C3", ["P6", "P7"]),
        ("C4", ["P8", "P9"]),
    )
    for dimension in ("biological_support", "cross_modal_consistency", "confounder_exclusion"):
        state["reports"].append({"dimension": dimension, "scope": "set_identity", "target_ids": ["C1"]})
    state["reports"].append({"dimension": "known_label_echo", "scope": "partition", "target_ids": []})
    state["evidence"]["raw"].append({
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "subject_signature": subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"]),
        "results": [{
            "tool_name": "multimodal_consistency_check",
            "status": "success",
            "metrics": {
                "internal_structure_by_set": {"C2": {"positive_internal_heterogeneity": True}},
                "positive_weak_boundary_pairs": {"C3+C4": ["ct", "wsi"]},
            },
        }],
    })
    output = RouterOutput.model_validate({
        "actions": [
            {"action": "accept", "target_ids": ["C1"]},
            {"action": "split", "target_ids": ["C2"]},
            {"action": "merge", "target_ids": ["C3", "C4"]},
        ]
    })
    validate_router_output(output, state)


def test_structural_actions_require_positive_signals():
    state = state_with_sets(("C1", ["P1", "P2", "P3"]), ("C2", ["P4", "P5", "P6"]))
    output = RouterOutput.model_validate({"actions": [{"action": "split", "target_ids": ["C1"]}]})
    with pytest.raises(ValueError, match="positive internal"):
        validate_router_output(output, state)


def test_split_creates_superseded_parent_and_exact_children():
    state = state_with_sets(("C1", ["P1", "P2", "P3", "P4"]))
    apply_split(state, "C1", [["P1", "P2"], ["P3", "P4"]])
    assert next(item for item in state["sets"] if item["set_id"] == "C1")["status"] == "superseded_by_split"
    assert [item["set_id"] for item in state["sets"] if item["status"] == "active"] == ["C1_S1", "C1_S2"]


def test_merge_creates_superseded_parents_and_union():
    state = state_with_sets(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))
    apply_merge(state, ["C2", "C1"])
    merged = next(item for item in state["sets"] if item["status"] == "active")
    assert merged["member_ids"] == ["P1", "P2", "P3", "P4"]
    assert {item["status"] for item in state["sets"] if item["set_id"] in {"C1", "C2"}} == {"superseded_by_merge"}


def test_structural_reset_invalidates_current_evidence_and_history():
    state = state_with_sets(("C1", ["P1", "P2"]))
    state["evidence"]["raw"].append({"dimension": "x"})
    state["reports"].append({"dimension": "x"})
    reset_after_structural_change(state, "new-partition")
    assert state["evidence"]["raw"] == []
    assert len(state["evidence"]["history"]) == 1
    assert state["reports"] == []
    assert all(item["status"] == "active" for item in state["sets"])


def test_reviser_schema_has_one_plan_and_no_membership_field():
    plan = ReviserOutput.model_validate({
        "action": "split",
        "target_ids": ["C1"],
        "n_children": 3,
        "structural_basis": ["rna", "wsi"],
        "execution_strategy": "multimodal_consensus",
    })
    assert plan.n_children == 3
    assert not hasattr(plan, "groups")


def test_max_round_and_failure_settings_are_workflow_only():
    state = state_with_sets(("C1", ["P1", "P2"]))
    assert set(state["control"]) == {
        "round", "failures", "status", "next", "error", "max_rounds", "max_failures",
        "visited_partitions", "trace",
    }


def test_graph_runs_request_acquisition_report_and_accept(monkeypatch):
    import agents.subtype_review.graph as graph_module

    def fake_validation(*args, **kwargs):
        dimension = kwargs["dimension"]
        metrics = {}
        if dimension == "cross_modal_consistency":
            metrics = {
                "identity_supporting_modalities_by_set": {"C1": ["ct", "wsi"]},
                "internal_structure_by_set": {"C1": {"positive_internal_heterogeneity": False}},
                "positive_weak_boundary_pairs": {},
            }
        return {
            "tool_name": dimension,
            "status": "success",
            "results": [{"tool_name": dimension, "status": "success", "metrics": metrics, "metric_refs": []}],
        }

    monkeypatch.setattr(
        graph_module,
        "VALIDATION_FUNCTIONS",
        {name: (lambda *args, _name=name, **kwargs: fake_validation(*args, dimension=_name, **kwargs)) for name in graph_module.EVIDENCE_DIMENSIONS},
    )

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                return {"tool_calls": [{"name": row["dimension"], "id": row["dimension"]} for row in payload["requests"]]}
            return {"reports": [{
                "dimension": row["dimension"], "scope": row["scope"], "target_ids": row["target_ids"],
                "observations": [], "statistical_interpretation": "ok", "medical_interpretation": "ok",
                "limitations": [], "metric_refs": [],
            } for row in payload["requests"]]}

    class Router:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                dimensions = list(graph_module.EVIDENCE_DIMENSIONS)
                return {"actions": [{
                    "action": "need_more_evidence",
                    "target_ids": ["C1"],
                    "requests": [{
                        "dimension": dimension,
                        "scope": "partition" if dimension == "known_label_echo" else "set_identity",
                        "target_ids": [] if dimension == "known_label_echo" else ["C1"],
                    } for dimension in dimensions],
                }]}
            return {"actions": [{"action": "accept", "target_ids": ["C1"], "reason": "closed"}]}

    graph = build_review_graph(
        verifier_model=Verifier(),
        router_model=Router(),
        reviser_model=object(),
        runtime={"data_root": "/tmp", "output_root": "/tmp", "config_dir": "configs", "patient_states_by_id": {}},
    )
    result = graph.invoke(initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}]))
    assert result["control"]["status"] == "complete"
    assert result["sets"][0]["status"] == "accept"


def test_reviser_applies_one_deterministic_k3_plan(tmp_path):
    from agents.subtype_review.graph import reviser_node

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    members = [f"P{i}" for i in range(6)]
    (candidate / "affinity_patient_order.json").write_text(json.dumps(members), encoding="utf-8")
    matrix = np.full((6, 6), 0.05, dtype=float)
    for start in (0, 2, 4):
        matrix[start:start + 2, start:start + 2] = 0.95
    np.fill_diagonal(matrix, 1.0)
    np.save(candidate / "fused_similarity.npy", matrix)
    state = initial_review_state([{"cluster_id": "C1", "member_ids": members}])
    state["action"] = {"action": "split", "target_ids": ["C1"]}
    state["revision"] = {
        "pending_actions": [state["action"]],
        "plans": [],
        "partition_signature": "before",
    }
    state["evidence"]["raw"].append({
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "results": [{
            "tool_name": "multimodal_consistency_check",
            "metrics": {"internal_structure_by_set": {"C1": {"recommended_k": 3}}},
        }],
    })

    class Reviser:
        def invoke(self, payload):
            return {
                "action": "split",
                "target_ids": ["C1"],
                "n_children": 3,
                "structural_basis": ["fused"],
                "execution_strategy": "fused_similarity_spectral",
                "metric_refs": [],
            }

    result = reviser_node(
        state,
        {"data_root": str(tmp_path), "output_root": str(tmp_path)},
        Reviser(),
    )
    assert result["revision"] is None
    assert result["sets"][0]["status"] == "superseded_by_split"
    assert len([item for item in result["sets"] if item["status"] == "active"]) == 3
