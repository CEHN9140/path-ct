from __future__ import annotations

import csv
import numpy as np
import pytest

from agents.subtype_review.graph import (
    apply_split,
    complete_audit,
    current_partition_evidence,
    partition_signature,
    reactivate_provisional_sets,
    required_evidence_requests,
    reviser_node,
    router_node,
    save_review_outputs,
    supported_structure_proposals,
    subject_signature,
    validate_router_action,
    validate_verifier_audit,
    verifier_node,
)
from agents.subtype_review.schemas import EVIDENCE_DIMENSIONS, RouterAction, VerifierOutput
from agents.subtype_review.graph import initial_review_state
from tools.subtype_review_common import bias_corrected_cramers_v, cliffs_delta, scoped_candidate_sets
from tools.tool_confound_test import CATEGORICAL_FIELDS, NUMERIC_FIELDS, global_categorical
from tools.tool_mutation_enrichment import enrichment_rows
from tools.tool_cnv_characterization import tool_cnv_characterization
from tools.tool_multimodal_consistency_check import compute_cross_modal_consistency


class StaticModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.payloads = []

    def invoke(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        return self.response


def evidence_row(state, dimension, scope, targets, proposal, tool_name, metrics, ref):
    return {
        "capability": dimension,
        "scope": scope,
        "proposal_id": proposal.get("proposal_id") if proposal else None,
        "subject_signature": subject_signature(
            dimension, scope, state["sets"], targets, proposal
        ),
        "results": [{
            "tool_name": tool_name,
            "status": "success",
            "metrics": metrics,
            "metric_refs": [ref],
        }],
    }


def add_identity_controls(state, target="C1", confound_status="supporting"):
    targets = [item["set_id"] for item in state["sets"] if item["status"] != "retired"]
    confound_ref = "tool_results.tool_confound_test.metrics.strong_technical_conflict"
    known_ref = "tool_results.tool_known_label_echo_test.metrics.near_identity"
    state["evidence"]["results"].extend([
        evidence_row(
            state,
            "confounder_exclusion",
            "set_identity",
            targets,
            {},
            "tool_confound_test",
            {"strong_technical_conflict": confound_status == "conflicting"},
            confound_ref,
        ),
        evidence_row(
            state,
            "known_label_echo",
            "partition",
            [],
            {},
            "tool_known_label_echo_test",
            {"near_identity": False},
            known_ref,
        ),
    ])
    state["audit"]["findings"].extend([
        {
            "target_ids": [target],
            "dimension": "confounder_exclusion",
            "scope": "set_identity",
            "status": confound_status,
            "metric_refs": [confound_ref],
        },
        {
            "target_ids": [],
            "dimension": "known_label_echo",
            "scope": "partition",
            "status": "supporting",
            "metric_refs": [known_ref],
        },
    ])


def add_proposal_checks(state, proposal, scope, targets, include_biology=False):
    checks = [
        (
            "confounder_exclusion",
            "tool_confound_test",
            {"strong_technical_conflict": False},
            "tool_results.tool_confound_test.metrics.strong_technical_conflict",
            "supporting",
        ),
        (
            "known_label_echo",
            "tool_known_label_echo_test",
            {"near_identity": False},
            "tool_results.tool_known_label_echo_test.metrics.near_identity",
            "supporting",
        ),
    ]
    if include_biology:
        checks.append((
            "biological_support",
            "tool_pathway_enrichment",
            {"signal": 0},
            "tool_results.tool_pathway_enrichment.metrics.signal",
            "inconclusive",
        ))
    for dimension, tool_name, metrics, ref, status in checks:
        state["evidence"]["results"].append(evidence_row(
            state, dimension, scope, targets, proposal, tool_name, metrics, ref
        ))
        state["audit"]["findings"].append({
            "target_ids": targets,
            "proposal_id": proposal["proposal_id"],
            "dimension": dimension,
            "scope": scope,
            "status": status,
            "metric_refs": [ref],
        })


def test_v2_has_four_dimensions_and_scopes():
    assert EVIDENCE_DIMENSIONS == (
        "biological_support",
        "cross_modal_consistency",
        "confounder_exclusion",
        "known_label_echo",
    )
    action = RouterAction(
        action="need_more_evidence",
        target_ids=["C1"],
        dimension="biological_support",
        scope="split_proposal",
        proposal_id="split:C1:k2:p1",
    )
    assert action.scope == "split_proposal"


def test_split_scope_is_child_vs_child_not_child_vs_rest():
    sets = [{"set_id": "C1", "member_ids": ["A", "B", "C", "D"]}]
    proposal = {
        "plan_id": "split:C1:k2:p1",
        "groups": [["A", "B"], ["C", "D"]],
    }
    groups = scoped_candidate_sets("split_proposal", sets[0], sets, proposal)
    assert sorted(groups.values(), key=lambda x: sorted(x)) == [{"A", "B"}, {"C", "D"}]


def test_inconclusive_biology_does_not_block_accept_when_identity_is_supported():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    sig = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"])
    ref = "tool_results.tool_multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"] = {"results": [{
        "capability": "cross_modal_consistency",
        "scope": "set_identity",
        "subject_signature": sig,
        "proposal_id": None,
        "results": [{
            "tool_name": "tool_multimodal_consistency_check",
            "metrics": {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}},
            "metric_refs": [ref],
        }],
    }]}
    state["audit"] = {"findings": [{
        "target_ids": ["C1"],
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "subject_signature": sig,
        "status": "supporting",
        "summary": "two original modalities support identity",
        "metric_refs": [ref],
    }], "gaps": []}
    add_identity_controls(state)
    validate_router_action(
        RouterAction(action="accept", target_ids=["C1"], metric_refs=[ref]), state
    )


def test_drop_requires_positive_invalidating_evidence():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    with pytest.raises(ValueError, match="unavailable metrics"):
        validate_router_action(
            RouterAction(action="drop", target_ids=["C1"], metric_refs=["x"]), state
        )


def test_set_identity_evidence_invalidates_after_partition_change():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C3", "member_ids": ["P3", "P4"]},
    ])
    biology_sig = subject_signature("biological_support", "set_identity", state["sets"], ["C3"])
    cross_modal_sig = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C3"])
    state["evidence"] = {"results": [{
        "capability": "biological_support",
        "scope": "set_identity",
        "subject_signature": biology_sig,
        "proposal_id": None,
        "results": [{"tool_name": "tool_pathway_enrichment", "metrics": {"x": 1}}],
    }, {
        "capability": "cross_modal_consistency",
        "scope": "set_identity",
        "subject_signature": cross_modal_sig,
        "proposal_id": None,
        "results": [{"tool_name": "tool_multimodal_consistency_check", "metrics": {"x": 1}}],
    }]}
    apply_split(state, {"source_set_id": "C1", "groups": [["P1"], ["P2"]]})
    assert not current_partition_evidence(state)["results"]


def test_verifier_derives_subject_signature_in_python():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    model = StaticModel({
        "findings": [],
        "gaps": [{
            "target_ids": ["C1"],
            "dimension": "biological_support",
            "scope": "set_identity",
            "reason": "missing biology",
        }],
    })
    verifier_node(state, {}, model)
    gap = state["audit"]["gaps"][0]
    assert gap["subject_signature"] == subject_signature(
        "biological_support", "set_identity", state["sets"], ["C1"]
    )


def test_empty_initial_audit_defers_to_python_mandatory_requests():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]}
    ])

    verifier_node(state, {}, StaticModel({"findings": [], "gaps": []}))

    assert state["control"]["error"] is None
    assert state["control"]["next"] == "router"
    assert required_evidence_requests(state)


def test_dimension_aware_set_signature_dependencies():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C3", "member_ids": ["P3", "P4"]},
    ])
    biology_before = subject_signature("biological_support", "set_identity", state["sets"], ["C3"])
    cross_before = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C3"])
    apply_split(state, {"source_set_id": "C1", "groups": [["P1"], ["P2"]]})
    active = [item for item in state["sets"] if item["status"] != "retired"]
    assert biology_before != subject_signature("biological_support", "set_identity", active, ["C3"])
    assert cross_before != subject_signature("cross_modal_consistency", "set_identity", active, ["C3"])


def test_split_proposal_evidence_survives_unrelated_split():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C3", "member_ids": ["P3", "P4", "P5", "P6"]},
    ])
    proposal = {
        "proposal_id": "p3",
        "plan_id": "p3",
        "source_set_id": "C3",
        "groups": [["P3", "P4"], ["P5", "P6"]],
    }
    state["structure_proposals"] = {
        "split_proposals": [proposal],
        "merge_proposals": [],
    }
    state["evidence"] = {"results": [evidence_row(
        state,
        "biological_support",
        "split_proposal",
        ["C3"],
        proposal,
        "tool_pathway_enrichment",
        {"signal": 1},
        "tool_results.tool_pathway_enrichment.metrics.signal",
    )]}

    apply_split(state, {"source_set_id": "C1", "groups": [["P1"], ["P2"]]})

    assert len(current_partition_evidence(state)["results"]) == 1


def test_python_generates_tiered_mandatory_evidence_requests():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]},
        {"cluster_id": "C2", "member_ids": ["Q1", "Q2"]},
    ])
    proposal = {
        "proposal_id": "p1",
        "plan_id": "p1",
        "source_set_id": "C1",
        "eligible_for_review": True,
        "groups": [
            [f"P{i}" for i in range(10)],
            [f"P{i}" for i in range(10, 20)],
        ],
    }
    state["structure_proposals"] = {
        "split_proposals": [proposal],
        "merge_proposals": [],
    }
    initial = {
        (row["dimension"], row["scope"], row.get("proposal_id"))
        for row in required_evidence_requests(state)
    }
    assert initial == {
        ("cross_modal_consistency", "set_identity", None),
        ("confounder_exclusion", "set_identity", None),
        ("known_label_echo", "partition", None),
        ("cross_modal_consistency", "split_proposal", "p1"),
    }

    ref = "tool_results.tool_multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"] = {"results": [evidence_row(
        state,
        "cross_modal_consistency",
        "split_proposal",
        ["C1"],
        proposal,
        "tool_multimodal_consistency_check",
        {"split_supporting_modalities": ["ct", "rna"]},
        ref,
    )]}
    follow_up = {
        (row["dimension"], row["scope"], row.get("proposal_id"))
        for row in required_evidence_requests(state)
    }
    assert ("confounder_exclusion", "split_proposal", "p1") in follow_up
    assert ("known_label_echo", "split_proposal", "p1") in follow_up
    assert ("biological_support", "split_proposal", "p1") not in follow_up


def test_verifier_cannot_borrow_metric_ref_from_another_proposal():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2", "P3", "P4"]}
    ])
    p1 = {
        "proposal_id": "p1",
        "plan_id": "p1",
        "source_set_id": "C1",
        "groups": [["P1", "P2"], ["P3", "P4"]],
    }
    p2 = {
        "proposal_id": "p2",
        "plan_id": "p2",
        "source_set_id": "C1",
        "groups": [["P1", "P3"], ["P2", "P4"]],
    }
    state["structure_proposals"] = {
        "split_proposals": [p1, p2],
        "merge_proposals": [],
    }
    ref = "tool_results.tool_pathway_enrichment.metrics.signal"
    state["evidence"] = {"results": [evidence_row(
        state,
        "biological_support",
        "split_proposal",
        ["C1"],
        p1,
        "tool_pathway_enrichment",
        {"signal": 1},
        ref,
    )]}
    audit = VerifierOutput.model_validate({"findings": [{
        "target_ids": ["C1"],
        "dimension": "biological_support",
        "scope": "split_proposal",
        "proposal_id": "p2",
        "subject_signature": subject_signature(
            "biological_support", "split_proposal", state["sets"], ["C1"], p2
        ),
        "status": "supporting",
        "metric_refs": [ref],
    }], "gaps": []})

    with pytest.raises(ValueError, match="exact evidence"):
        validate_verifier_audit(audit, state)


def test_legal_drop_completes_review():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    state["sets"][0]["status"] = "provisionally_accepted"
    state["sets"][1]["status"] = "provisionally_dropped"
    state["audit"] = {"findings": [{
        "target_ids": ["C2"],
        "dimension": "confounder_exclusion",
        "scope": "set_identity",
        "status": "conflicting",
    }], "gaps": []}
    assert complete_audit(state)


def test_drop_evidence_does_not_reactivate_a_provisionally_dropped_set():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    state["sets"][0]["status"] = "provisionally_dropped"
    audit = VerifierOutput.model_validate({"findings": [{
        "target_ids": ["C1"], "dimension": "confounder_exclusion",
        "scope": "set_identity", "status": "conflicting",
    }], "gaps": []})
    reactivate_provisional_sets(state, audit)
    assert state["sets"][0]["status"] == "provisionally_dropped"


def test_partition_known_label_conflict_reopens_an_accepted_set():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]}
    ])
    state["sets"][0]["status"] = "provisionally_accepted"
    audit = VerifierOutput.model_validate({"findings": [{
        "target_ids": [],
        "dimension": "known_label_echo",
        "scope": "partition",
        "status": "conflicting",
    }], "gaps": []})
    state["audit"] = audit.model_dump()

    assert not complete_audit(state)
    reactivate_provisional_sets(state, audit)
    assert state["sets"][0]["status"] == "active"


def test_router_contract_failure_is_runtime_failure_not_scientific_unresolved():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    signature = subject_signature("biological_support", "set_identity", state["sets"], ["C1"])
    state["audit"] = {"findings": [], "gaps": [{
        "target_ids": ["C1"], "dimension": "biological_support", "scope": "set_identity",
        "subject_signature": signature, "proposal_id": None,
    }]}
    model = StaticModel({"action": "accept", "target_ids": ["BAD"], "metric_refs": []})
    router_node(state, {}, model)
    assert state["control"]["status"] == "review_failed_runtime"


def test_budget_exhaustion_uses_public_unresolved_status(tmp_path):
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]}
    ])
    state["sets"][0]["status"] = "provisionally_accepted"
    state["control"]["status"] = "unresolved_due_to_budget"

    summary = save_review_outputs(state, str(tmp_path))

    assert summary["status"] == "review_complete_with_unresolved_sets"


def test_evidence_exhaustion_is_scientifically_unresolved_without_calling_router():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    state["evidence"] = {"results": []}
    state["audit"] = {"findings": [], "gaps": []}
    cross_ref = "tool_results.tool_multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state,
        "cross_modal_consistency",
        "set_identity",
        ["C1"],
        {},
        "tool_multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {"C1": []}},
        cross_ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"],
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "status": "inconclusive",
        "metric_refs": [cross_ref],
    })
    add_identity_controls(state)
    model = StaticModel({"action": "accept", "target_ids": ["C1"], "metric_refs": []})
    router_node(state, {}, model)
    assert state["control"]["status"] == "unresolved"
    assert model.calls == 0


def test_supported_split_proposal_is_bound_to_exact_membership_and_two_modalities():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]}])
    p1 = {"proposal_id": "p1", "plan_id": "p1", "source_set_id": "C1", "eligible_for_review": True, "groups": [[f"P{i}" for i in range(10)], [f"P{i}" for i in range(10, 20)]]}
    p2 = {"proposal_id": "p2", "plan_id": "p2", "source_set_id": "C1", "eligible_for_review": True, "groups": [[f"P{i}" for i in range(0, 20, 2)], [f"P{i}" for i in range(1, 20, 2)]]}
    state["structure_proposals"] = {"split_proposals": [p1, p2], "merge_proposals": []}
    ref = "tool_results.tool_multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"] = {"results": [
        {"capability": "cross_modal_consistency", "scope": "split_proposal", "proposal_id": "p1", "subject_signature": subject_signature("cross_modal_consistency", "split_proposal", state["sets"], ["C1"], p1), "results": [{"tool_name": "tool_multimodal_consistency_check", "metrics": {"split_supporting_modalities": ["ct", "rna"]}, "metric_refs": [ref]}]},
        {"capability": "cross_modal_consistency", "scope": "split_proposal", "proposal_id": "p2", "subject_signature": subject_signature("cross_modal_consistency", "split_proposal", state["sets"], ["C1"], p2), "results": [{"tool_name": "tool_multimodal_consistency_check", "metrics": {"split_supporting_modalities": ["ct"]}, "metric_refs": [ref]}]},
    ]}
    state["audit"] = {"findings": [
        {"target_ids": ["C1"], "proposal_id": "p1", "dimension": "cross_modal_consistency", "scope": "split_proposal", "status": "supporting", "metric_refs": [ref]},
        {"target_ids": ["C1"], "proposal_id": "p2", "dimension": "cross_modal_consistency", "scope": "split_proposal", "status": "supporting", "metric_refs": [ref]},
    ], "gaps": []}
    add_proposal_checks(state, p1, "split_proposal", ["C1"])
    assert [row["proposal_id"] for row in supported_structure_proposals(state, "split", "C1")] == ["p1"]
    state["audit"]["findings"].append({
        "target_ids": ["C1"],
        "dimension": "confounder_exclusion",
        "scope": "set_identity",
        "status": "conflicting",
        "metric_refs": [ref],
    })
    with pytest.raises(ValueError, match="structural correction"):
        validate_router_action(
            RouterAction(action="drop", target_ids=["C1"], metric_refs=[ref]),
            state,
        )
    state["control"]["policy"]["split_min_supporting_modalities"] = 3
    assert not supported_structure_proposals(state, "split", "C1")


def test_reviser_only_receives_supported_proposals():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]}
    ])
    p1 = {
        "proposal_id": "p1",
        "plan_id": "p1",
        "source_set_id": "C1",
        "eligible_for_review": True,
        "groups": [
            [f"P{i}" for i in range(10)],
            [f"P{i}" for i in range(10, 20)],
        ],
    }
    p2 = {
        "proposal_id": "p2",
        "plan_id": "p2",
        "source_set_id": "C1",
        "eligible_for_review": True,
        "groups": [
            [f"P{i}" for i in range(0, 20, 2)],
            [f"P{i}" for i in range(1, 20, 2)],
        ],
    }
    state["structure_proposals"] = {
        "split_proposals": [p1, p2],
        "merge_proposals": [],
    }
    metric_ref = (
        "tool_results.tool_multimodal_consistency_check.metrics."
        "split_supporting_modalities"
    )
    state["evidence"] = {
        "results": [
            {
                "capability": "cross_modal_consistency",
                "scope": "split_proposal",
                "proposal_id": proposal["proposal_id"],
                "subject_signature": subject_signature(
                    "cross_modal_consistency",
                    "split_proposal",
                    state["sets"],
                    ["C1"],
                    proposal,
                ),
                "results": [
                    {
                        "tool_name": "tool_multimodal_consistency_check",
                        "metrics": {"split_supporting_modalities": modalities},
                        "metric_refs": [metric_ref],
                    }
                ],
            }
            for proposal, modalities in ((p1, ["ct", "rna"]), (p2, ["ct"]))
        ]
    }
    state["audit"] = {
        "findings": [
            {
                "target_ids": ["C1"],
                "proposal_id": proposal["proposal_id"],
                "dimension": "cross_modal_consistency",
                "scope": "split_proposal",
                "status": "supporting",
                "metric_refs": [metric_ref],
            }
            for proposal in (p1, p2)
        ],
        "gaps": [],
    }
    add_proposal_checks(state, p1, "split_proposal", ["C1"])
    state["action"] = {
        "action": "split",
        "target_ids": ["C1"],
        "metric_refs": [metric_ref],
    }
    structural_ref = "structure_proposals.split_proposals.0.plan_id"
    model = StaticModel(
        {"plan_id": "p1", "reason": "supported", "metric_refs": [structural_ref]}
    )

    reviser_node(state, {}, model)

    assert [row["proposal_id"] for row in model.payloads[0]["candidates"]] == ["p1"]
    assert {row["set_id"] for row in state["sets"] if row["status"] == "active"} == {
        "C1_S1",
        "C1_S2",
    }


def test_accept_uses_only_target_set_identity_evidence_and_hard_vetoes_confounder():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    signature = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"])
    ref = "tool_results.tool_multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"] = {"results": [{
        "capability": "cross_modal_consistency", "scope": "set_identity", "proposal_id": None,
        "subject_signature": signature, "results": [{"tool_name": "tool_multimodal_consistency_check", "metrics": {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}}, "metric_refs": [ref]}],
    }]}
    state["audit"] = {"findings": [
        {"target_ids": ["C1"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [ref]},
    ], "gaps": []}
    add_identity_controls(state, confound_status="conflicting")
    with pytest.raises(ValueError, match="vetoed"):
        validate_router_action(RouterAction(action="accept", target_ids=["C1"], metric_refs=[ref]), state)


def test_accept_cannot_borrow_identity_metrics_from_split_scope():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    identity_signature = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"])
    ref = "tool_results.tool_multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"] = {"results": [
        {"capability": "cross_modal_consistency", "scope": "set_identity", "proposal_id": None, "subject_signature": identity_signature, "results": [{"tool_name": "tool_multimodal_consistency_check", "metrics": {"identity_supporting_modalities_by_set": {"C1": ["ct"]}}, "metric_refs": [ref]}]},
        {"capability": "cross_modal_consistency", "scope": "split_proposal", "proposal_id": "p1", "subject_signature": "stale", "results": [{"tool_name": "tool_multimodal_consistency_check", "metrics": {"identity_supporting_modalities_by_set": {"C1": ["ct", "wsi", "rna"]}}, "metric_refs": [ref]}]},
    ]}
    state["audit"] = {"findings": [{"target_ids": ["C1"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [ref]}], "gaps": []}
    add_identity_controls(state)
    with pytest.raises(ValueError, match="at least two"):
        validate_router_action(RouterAction(action="accept", target_ids=["C1"], metric_refs=[ref]), state)


def test_merge_requires_two_weak_boundary_modalities_and_no_biology_veto():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    proposal = {"proposal_id": "m1", "plan_id": "m1", "set_ids": ["C1", "C2"], "memberships": [["P1", "P2"], ["P3", "P4"]], "eligible_for_review": True}
    state["structure_proposals"] = {"split_proposals": [], "merge_proposals": [proposal]}
    signature = subject_signature("cross_modal_consistency", "merge_proposal", state["sets"], ["C1", "C2"], proposal)
    ref = "tool_results.tool_multimodal_consistency_check.metrics.merge_supporting_modalities"
    state["audit"] = {"findings": [{"target_ids": ["C1", "C2"], "proposal_id": "m1", "dimension": "cross_modal_consistency", "scope": "merge_proposal", "status": "supporting", "metric_refs": [ref]}], "gaps": []}
    state["evidence"] = {"results": [{"capability": "cross_modal_consistency", "scope": "merge_proposal", "proposal_id": "m1", "subject_signature": signature, "results": [{"tool_name": "tool_multimodal_consistency_check", "metrics": {"merge_supporting_modalities": ["ct"], "merge_strong_boundary_modalities": []}, "metric_refs": [ref]}]}]}
    assert not supported_structure_proposals(state, "merge", "C1")
    state["evidence"]["results"][0]["results"][0]["metrics"]["merge_supporting_modalities"] = ["ct", "wsi"]
    add_proposal_checks(
        state, proposal, "merge_proposal", ["C1", "C2"], include_biology=True
    )
    assert supported_structure_proposals(state, "merge", "C1")
    state["audit"]["findings"].append({"target_ids": ["C1", "C2"], "proposal_id": "m1", "dimension": "biological_support", "scope": "merge_proposal", "status": "conflicting", "metric_refs": [ref]})
    assert not supported_structure_proposals(state, "merge", "C1")


def test_continuous_wxs_feature_does_not_reference_binary_ci():
    rows = enrichment_rows(["validation::estimated_TMB"], {
        "P1": {"validation::estimated_TMB": 1.0}, "P2": {"validation::estimated_TMB": 2.0},
        "P3": {"validation::estimated_TMB": 8.0}, "P4": {"validation::estimated_TMB": 9.0},
    }, {"C1": {"P1", "P2"}, "C2": {"P3", "P4"}})
    assert all(row["odds_ratio_ci95"] is None for row in rows)


def test_ct_spacing_is_numeric_and_sparse_categorical_has_permutation_p():
    assert "ct_pixel_spacing_row" in NUMERIC_FIELDS
    assert "ct_z_spacing" in NUMERIC_FIELDS
    assert "ct_pixel_spacing_row" not in CATEGORICAL_FIELDS
    metrics = global_categorical(
        "scanner",
        {"C1": ["P1", "P2"], "C2": ["P3", "P4"], "C3": ["P5"]},
        {"P1": {"scanner": "A"}, "P2": {"scanner": "A"}, "P3": {"scanner": "B"}, "P4": {"scanner": "C"}},
        permutations=99,
    )
    assert metrics["chi_square_p_value"] is not None
    assert metrics["test_method"] == "permutation_chi_square"


def test_cnv_reports_continuous_and_gain_loss_events_with_ranked_summary(tmp_path):
    path = tmp_path / "cnv" / "case_features.csv"
    path.parent.mkdir()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "chr3p", "fga"])
        writer.writerows([
            ["P1", -0.8, 0.1], ["P2", -0.7, 0.2],
            ["P3", 0.1, 0.8], ["P4", 0.2, 0.9],
        ])
    result = tool_cnv_characterization(
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {},
        str(tmp_path),
        all_cluster_states=[
            {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
            {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
        ],
    )
    rows = result["results"]["metrics"]["cnv_characterization"]
    assert {row["feature"] for row in rows} >= {"chr3p", "chr3p::loss", "chr3p::gain"}
    summary = result["results"]["decision_metrics"]["per_comparison_cnv"]
    assert summary["C1_vs_rest"]["top_by_q"]
    assert summary["C1_vs_rest"]["top_by_effect"]


def test_cnv_continuous_effect_ranking_uses_cliffs_delta(tmp_path):
    path = tmp_path / "cnv" / "case_features.csv"
    path.parent.mkdir()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "fga", "gain_burden"])
        writer.writerows([
            ["P1", 0, 2],
            ["P2", 1000, 3],
            ["P3", 1, 0],
            ["P4", 2, 1],
        ])
    result = tool_cnv_characterization(
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {},
        str(tmp_path),
        all_cluster_states=[
            {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
            {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
        ],
    )

    top = result["results"]["decision_metrics"]["per_comparison_cnv"]
    assert top["C1_vs_rest"]["top_by_effect"][0]["feature"] == "gain_burden"


def test_cross_modal_identity_uses_median_silhouette(monkeypatch):
    def separation(_similarity, _labels):
        row = {"normalized_affinity_separation": 1.0}
        per_set = {
            "C1": {
                "normalized_affinity_separation": 1.0,
                "median_affinity_margin": 1.0,
                "fraction_affinity_margin_positive": 0.75,
            },
            "C2": {
                "normalized_affinity_separation": 1.0,
                "median_affinity_margin": 1.0,
                "fraction_affinity_margin_positive": 0.75,
            },
        }
        return row, per_set

    monkeypatch.setattr(
        "tools.tool_multimodal_consistency_check.partition_separation", separation
    )
    monkeypatch.setattr(
        "sklearn.metrics.silhouette_samples",
        lambda _distance, _labels, metric: np.asarray([-0.1, -0.1, 0.5, 0.2, 0.2, 0.2]),
    )
    affinities = {name: np.eye(6) for name in ("ct", "wsi", "rna", "genomic")}
    metrics = compute_cross_modal_consistency(
        affinities,
        [f"P{i}" for i in range(6)],
        {"C1": ["P0", "P1", "P2"], "C2": ["P3", "P4", "P5"]},
        permanova_permutations=9,
    )

    assert not metrics["decision_metrics"]["identity_supporting_modalities_by_set"]["C1"]


def test_effect_helpers_are_material_not_p_value_only():
    table = np.asarray([[8, 2], [1, 9]])
    assert bias_corrected_cramers_v(table) > 0
    assert cliffs_delta([1, 2, 3], [8, 9, 10]) < 0
