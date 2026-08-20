from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pytest

import agents.subtype_review.tools as review_tools
from agents.subtype_review.graph import (
    abandon_revision,
    apply_split,
    build_review_graph,
    complete_audit,
    current_partition_evidence,
    evidence_request_key,
    execute_tool_calls,
    initial_revision_intents,
    partition_signature,
    reactivate_provisional_sets,
    required_evidence_requests,
    reset_after_structural_change,
    reviser_node,
    router_node,
    save_review_outputs,
    supported_revision_candidates,
    subject_signature,
    validate_router_action,
    validate_verifier_audit,
    verifier_node,
)
from agents.subtype_review.schemas import (
    EVIDENCE_DIMENSIONS,
    ReviserOutput,
    RouterAction,
    RouterLLMOutput,
    VerifierOutput,
)
from agents.subtype_review.graph import initial_review_state
from tools.subtype_review_common import bias_corrected_cramers_v, cliffs_delta, scoped_candidate_sets
from tools.confound_test import CATEGORICAL_FIELDS, NUMERIC_FIELDS, global_categorical
from tools.mutation_enrichment import enrichment_rows
from tools.cnv_characterization import cnv_characterization
from tools.multimodal_consistency_check import compute_cross_modal_consistency
from tools.structural_adequacy import generate_revision_candidates


class StaticModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.payloads = []

    def invoke(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        return self.response


class SequentialModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.payloads = []

    def invoke(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        return next(self.responses)


def test_router_transport_schema_defers_action_contract_to_router_node():
    malformed = RouterLLMOutput(action="accept", target_ids=["C1", "C2"])
    assert malformed.target_ids == ["C1", "C2"]
    with pytest.raises(ValueError, match="scientific actions require one target_id"):
        RouterAction.model_validate(malformed.model_dump())


def test_router_retries_transport_valid_but_contract_invalid_action():
    state = initial_review_state([{
        "cluster_id": "C1",
        "member_ids": [f"P{i}" for i in range(20)],
    }])
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state,
        "cross_modal_consistency",
        "set_identity",
        ["C1"],
        {},
        "multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}},
        cross_ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"],
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "status": "supporting",
        "metric_refs": [cross_ref],
    })
    add_identity_controls(state)
    model = SequentialModel([
        {"action": "accept", "target_ids": ["C1", "C2"]},
        {"action": "split", "target_ids": ["C1"]},
    ])

    router_node(state, {}, model)

    assert state["control"]["status"] == "reviewing"
    assert state["action"]["action"] == "split"
    assert model.payloads[1]["rejected_action"]["target_ids"] == ["C1", "C2"]
    assert "validation_error" in model.payloads[1]


def test_router_structural_actions_do_not_select_proposals():
    split = RouterAction(action="split", target_ids=["C1"])
    merge = RouterAction(action="merge", target_ids=["C2", "C1"])

    assert split.proposal_id is None
    assert merge.target_ids == ["C1", "C2"]
    with pytest.raises(ValueError):
        RouterAction(action="split", target_ids=["C1"], proposal_id="p1")
    with pytest.raises(ValueError):
        RouterAction(action="merge", target_ids=["C1"])


def test_reviser_output_contains_only_plan_and_reason():
    parsed = ReviserOutput.model_validate({"plan_id": "p1", "reason": "best"})
    assert parsed.model_dump() == {"plan_id": "p1", "reason": "best"}
    with pytest.raises(ValueError):
        ReviserOutput.model_validate({"plan_id": "p1", "reason": "best", "metric_refs": ["x"]})
    with pytest.raises(ValueError):
        RouterAction.model_validate({"action": "accept", "target_ids": ["C1"], "metric_refs": ["x"]})


def test_revision_candidates_are_filtered_to_exact_intent(monkeypatch):
    import tools.structural_adequacy as structural_adequacy

    monkeypatch.setattr(structural_adequacy, "generate_structure_proposal_metrics", lambda *args, **kwargs: {
        "status": "success",
        "results": {"metrics": {
            "split_candidates": [
                {"plan_id": "s1", "source_set_id": "C1", "child_count": 2, "groups": [["P1"], ["P2"]], "selection_adjusted_null": {"q_value": 0.01, "separation_gain_over_null": 0.2}},
                {"plan_id": "s2", "source_set_id": "C2", "child_count": 2, "groups": [["P3"], ["P4"]], "selection_adjusted_null": {"q_value": 0.01, "separation_gain_over_null": 0.2}},
            ],
            "merge_candidates": [
                {"plan_id": "m12", "set_ids": ["C2", "C1"]},
                {"plan_id": "m23", "set_ids": ["C2", "C3"]},
            ],
        }},
    })
    sets = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
        {"set_id": "C3", "member_ids": ["P5", "P6"]},
    ]

    split = generate_revision_candidates("split", ["C1"], {}, {}, "output", all_cluster_states=sets)
    merge = generate_revision_candidates("merge", ["C2", "C1"], {}, {}, "output", all_cluster_states=sets)

    assert [row["proposal_id"] for row in split] == ["s1"]
    assert split[0]["parent_members"] == ["P1", "P2"]
    assert split[0]["partition_signature"]
    assert [row["proposal_id"] for row in merge] == ["m12"]
    assert merge[0]["set_ids"] == ["C1", "C2"]
    assert merge[0]["memberships"] == [["P1", "P2"], ["P3", "P4"]]


def test_initial_revision_intents_review_acceptable_sets_before_accept():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]},
        {"cluster_id": "C2", "member_ids": [f"Q{i}" for i in range(20)]},
    ])
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state,
        "cross_modal_consistency",
        "set_identity",
        ["C1", "C2"],
        {},
        "multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {"C1": ["ct", "wsi"], "C2": ["ct", "rna"]}},
        cross_ref,
    ))
    state["audit"]["findings"].extend([
        {"target_ids": ["C1"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [cross_ref]},
        {"target_ids": ["C2"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [cross_ref]},
    ])
    add_identity_controls(state, "C1")
    state["audit"]["findings"].insert(-1, {
        "target_ids": ["C2"], "dimension": "confounder_exclusion", "scope": "set_identity", "status": "supporting", "metric_refs": ["tool_results.confound_test.metrics.strong_technical_conflict"],
    })

    intents = initial_revision_intents(state)

    assert {tuple([row["action"], *row["target_ids"]]) for row in intents} == {
        ("split", "C1"),
        ("split", "C2"),
        ("merge", "C1", "C2"),
    }
    with pytest.raises(ValueError, match="structural review"):
        validate_router_action(RouterAction(action="accept", target_ids=["C1"]), state)


def test_biology_status_does_not_gate_initial_structural_review():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]}])
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "set_identity", ["C1"], {},
        "multimodal_consistency_check", {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}}, cross_ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"], "dimension": "cross_modal_consistency",
        "scope": "set_identity", "status": "supporting", "metric_refs": [cross_ref],
    })
    add_identity_controls(state)
    biology = {
        "target_ids": ["C1"], "dimension": "biological_support",
        "scope": "set_identity", "status": "inconclusive", "metric_refs": ["biology"],
    }
    state["audit"]["findings"].append(biology)

    for status in ("inconclusive", "conflicting", "mixed", "unavailable"):
        biology["status"] = status
        assert initial_revision_intents(state) == [{"action": "split", "target_ids": ["C1"]}]


def test_structural_intent_blocks_are_exact():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]},
        {"cluster_id": "C2", "member_ids": [f"Q{i}" for i in range(20)]},
        {"cluster_id": "C3", "member_ids": [f"R{i}" for i in range(20)]},
    ])
    targets = ["C1", "C2", "C3"]
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "set_identity", targets, {},
        "multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {target: ["ct", "rna"] for target in targets}},
        cross_ref,
    ))
    state["audit"]["findings"].extend([
        {"target_ids": [target], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [cross_ref]}
        for target in targets
    ])
    add_identity_controls(state)
    state["audit"]["findings"].extend([
        {"target_ids": [target], "dimension": "confounder_exclusion", "scope": "set_identity", "status": "supporting", "metric_refs": ["confound"]}
        for target in ("C2", "C3")
    ])
    state["control"]["blocked_actions"] = ["split:C1", "merge:C1+C2"]
    intents = initial_revision_intents(state)

    assert {tuple(row["target_ids"]) for row in intents if row["action"] == "split"} == {
        ("C2",), ("C3",),
    }
    assert {tuple([row["action"], *row["target_ids"]]) for row in intents if row["action"] == "merge"} == {
        ("merge", "C1", "C3"),
        ("merge", "C2", "C3"),
    }


def test_blocked_intents_persist_until_partition_changes(monkeypatch):
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]},
        {"cluster_id": "C2", "member_ids": [f"Q{i}" for i in range(20)]},
    ])
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "set_identity", ["C1", "C2"], {},
        "multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"], "C2": ["ct", "rna"]}},
        cross_ref,
    ))
    state["audit"]["findings"].extend([
        {"target_ids": [target], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [cross_ref]}
        for target in ("C1", "C2")
    ])
    add_identity_controls(state)
    state["audit"]["findings"].append({
        "target_ids": ["C2"], "dimension": "confounder_exclusion",
        "scope": "set_identity", "status": "supporting",
        "metric_refs": ["tool_results.confound_test.metrics.strong_technical_conflict"],
    })
    abandon_revision(state, "split:C1")

    proposal = {
        "proposal_id": "p2", "plan_id": "p2", "source_set_id": "C2",
        "eligible_for_review": True,
        "groups": [[f"Q{i}" for i in range(10)], [f"Q{i}" for i in range(10, 20)]],
    }
    state["revision"] = {
        "action": "split", "target_ids": ["C2"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    state["action"] = RouterAction(
        action="need_more_evidence", target_ids=["C2"],
        dimension="cross_modal_consistency", scope="split_proposal", proposal_id="p2",
    ).model_dump()
    monkeypatch.setitem(review_tools.VALIDATION_FUNCTIONS, "cross_modal_consistency", lambda *args, **kwargs: {
        "dimension": "cross_modal_consistency",
        "status": "success",
        "results": [{
            "tool_name": "multimodal_consistency_check", "status": "success",
            "metrics": {"split_supporting_modalities": ["ct"]},
            "metric_refs": ["tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"],
        }],
    })

    execute_tool_calls(
        state,
        {"tool_calls": [{"name": "cross_modal_consistency", "id": "call-1"}]},
        {"patient_states_by_id": {}, "output_root": "output", "config_dir": "configs"},
    )

    assert "split:C1" in state["control"]["blocked_actions"]
    abandon_revision(state, "split:C2")
    assert not any(
        row["action"] == "split" and row["target_ids"] == ["C1"]
        for row in initial_revision_intents(state)
    )
    reset_after_structural_change(state, partition_signature(state["sets"]))
    assert state["control"]["blocked_actions"] == []


def test_first_structural_action_creates_revision_without_calling_model(monkeypatch):
    import tools.structural_adequacy as structural_adequacy

    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    state["action"] = RouterAction(action="split", target_ids=["C1"]).model_dump()
    candidate = {
        "proposal_id": "p1",
        "plan_id": "p1",
        "source_set_id": "C1",
        "parent_members": ["P1", "P2"],
        "groups": [["P1"], ["P2"]],
        "eligible_for_review": True,
        "partition_signature": partition_signature(state["sets"]),
    }
    monkeypatch.setattr(structural_adequacy, "generate_revision_candidates", lambda *args, **kwargs: [candidate])
    model = StaticModel({"plan_id": "p1", "reason": "unused"})

    reviser_node(state, {"patient_states_by_id": {}, "output_root": "output"}, model)

    assert model.calls == 0
    assert state["revision"]["status"] == "evidence_collection"
    assert state["revision"]["candidates"] == [candidate]
    assert state["action"] is None
    assert state["control"]["next"] == "router"
    assert state["sets"][0]["status"] == "active"


def test_review_graph_contains_only_three_agent_nodes():
    graph = build_review_graph(
        verifier_model=StaticModel({}),
        router_model=StaticModel({}),
        reviser_model=StaticModel({}),
    )

    nodes = set(graph.get_graph().nodes)
    assert "proposals" not in nodes
    assert {"verifier", "router", "reviser"}.issubset(nodes)


def test_validation_tools_match_the_four_scientific_dimensions():
    assert set(review_tools.VALIDATION_FUNCTIONS) == set(EVIDENCE_DIMENSIONS)
    assert {tool.name for tool in review_tools.build_validation_tools()} == set(EVIDENCE_DIMENSIONS)
    assert not hasattr(review_tools, "execute_capability")


def test_tools_directory_has_no_tool_prefixed_python_modules():
    assert not list(Path("tools").glob("tool_*.py"))


def test_biological_support_keeps_rna_wxs_and_cnv_provenance(monkeypatch):
    calls = []

    def fake_tool(name):
        def run(*args, **kwargs):
            calls.append((name, kwargs["scope"], kwargs["target_ids"], kwargs["proposal"]))
            return {
                "tool_name": name,
                "status": "success",
                "results": {"decision_metrics": {"signal": name}},
            }

        return run

    for name in (
        "mutation_enrichment",
        "pathway_enrichment",
        "cnv_characterization",
    ):
        monkeypatch.setattr(review_tools, name, fake_tool(name))

    result = review_tools.biological_support(
        {"cluster_id": "p1", "member_ids": ["A", "B"]},
        {"A": {}, "B": {}},
        "output",
        "configs",
        [],
        scope="split_proposal",
        target_ids=["C1"],
        proposal={"proposal_id": "s1"},
    )

    assert result["dimension"] == "biological_support"
    assert result["status"] == "success"
    assert [item["tool_name"] for item in result["results"]] == [
        "pathway_enrichment",
        "mutation_enrichment",
        "cnv_characterization",
    ]
    assert {name for name, *_ in calls} == {
        "pathway_enrichment",
        "mutation_enrichment",
        "cnv_characterization",
    }
    assert all(scope == "split_proposal" for _, scope, _, _ in calls)


def test_evidence_cache_key_uses_dimension():
    assert evidence_request_key({
        "dimension": "biological_support",
        "scope": "set_identity",
        "subject_signature": "abc",
        "proposal_id": None,
    }) == "biological_support|set_identity|abc|"


def evidence_row(state, dimension, scope, targets, proposal, tool_name, metrics, ref):
    return {
        "dimension": dimension,
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
    confound_ref = "tool_results.confound_test.metrics.strong_technical_conflict"
    known_ref = "tool_results.known_label_echo_test.metrics.near_identity"
    state["evidence"]["results"].extend([
        evidence_row(
            state,
            "confounder_exclusion",
            "set_identity",
            targets,
            {},
            "confound_test",
            {"strong_technical_conflict": confound_status == "conflicting"},
            confound_ref,
        ),
        evidence_row(
            state,
            "known_label_echo",
            "partition",
            [],
            {},
            "known_label_echo_test",
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
            "confound_test",
            {"strong_technical_conflict": False},
            "tool_results.confound_test.metrics.strong_technical_conflict",
            "supporting",
        ),
        (
            "known_label_echo",
            "known_label_echo_test",
            {"near_identity": False},
            "tool_results.known_label_echo_test.metrics.near_identity",
            "supporting",
        ),
    ]
    if include_biology:
        checks.append((
            "biological_support",
            "pathway_enrichment",
            {"signal": 0},
            "tool_results.pathway_enrichment.metrics.signal",
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
    ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"] = {"results": [{
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "subject_signature": sig,
        "proposal_id": None,
        "results": [{
            "tool_name": "multimodal_consistency_check",
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
        RouterAction(action="accept", target_ids=["C1"]), state
    )


def test_conflicting_set_identity_biology_blocks_accept_but_not_drop():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    biology_ref = "tool_results.pathway_enrichment.metrics.signal"
    state["evidence"]["results"].extend([
        evidence_row(
            state, "cross_modal_consistency", "set_identity", ["C1"], {},
            "multimodal_consistency_check",
            {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}},
            cross_ref,
        ),
        evidence_row(
            state, "biological_support", "set_identity", ["C1"], {},
            "pathway_enrichment", {"signal": "opposes_identity"}, biology_ref,
        ),
    ])
    state["audit"] = {"findings": [
        {
            "target_ids": ["C1"], "dimension": "cross_modal_consistency",
            "scope": "set_identity", "status": "supporting", "metric_refs": [cross_ref],
        },
        {
            "target_ids": ["C1"], "dimension": "biological_support",
            "scope": "set_identity", "status": "conflicting", "metric_refs": [biology_ref],
        },
    ], "gaps": []}
    add_identity_controls(state)

    with pytest.raises(ValueError, match="Accept is vetoed"):
        validate_router_action(
            RouterAction(action="accept", target_ids=["C1"]), state
        )
    with pytest.raises(ValueError, match="positive confounder"):
        validate_router_action(
            RouterAction(action="drop", target_ids=["C1"]), state
        )


def test_drop_requires_positive_invalidating_evidence():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "set_identity", ["C1"], {},
        "multimodal_consistency_check", {"identity_supporting_modalities_by_set": {"C1": []}}, ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"], "dimension": "cross_modal_consistency",
        "scope": "set_identity", "status": "inconclusive", "metric_refs": [ref],
    })
    add_identity_controls(state)
    with pytest.raises(ValueError, match="positive confounder"):
        validate_router_action(RouterAction(action="drop", target_ids=["C1"]), state)


def test_set_identity_evidence_invalidates_after_partition_change():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C3", "member_ids": ["P3", "P4"]},
    ])
    biology_sig = subject_signature("biological_support", "set_identity", state["sets"], ["C3"])
    cross_modal_sig = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C3"])
    state["evidence"] = {"results": [{
        "dimension": "biological_support",
        "scope": "set_identity",
        "subject_signature": biology_sig,
        "proposal_id": None,
        "results": [{"tool_name": "pathway_enrichment", "metrics": {"x": 1}}],
    }, {
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "subject_signature": cross_modal_sig,
        "proposal_id": None,
        "results": [{"tool_name": "multimodal_consistency_check", "metrics": {"x": 1}}],
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


def test_revision_evidence_is_invalidated_by_partition_change():
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
    state["revision"] = {
        "action": "split", "target_ids": ["C3"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    state["evidence"] = {"results": [evidence_row(
        state,
        "biological_support",
        "split_proposal",
        ["C3"],
        proposal,
        "pathway_enrichment",
        {"signal": 1},
        "tool_results.pathway_enrichment.metrics.signal",
    )]}

    apply_split(state, {"source_set_id": "C1", "groups": [["P1"], ["P2"]]})
    state["revision"] = None

    assert not current_partition_evidence(state)["results"]


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
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
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

    ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"] = {"results": [evidence_row(
        state,
        "cross_modal_consistency",
        "split_proposal",
        ["C1"],
        proposal,
        "multimodal_consistency_check",
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
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [p1, p2],
    }
    ref = "tool_results.pathway_enrichment.metrics.signal"
    state["evidence"] = {"results": [evidence_row(
        state,
        "biological_support",
        "split_proposal",
        ["C1"],
        p1,
        "pathway_enrichment",
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


def test_verifier_allows_an_existing_metric_subpath_under_its_declared_root():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]}
    ])
    root = (
        "tool_results.multimodal_consistency_check.metrics."
        "identity_supporting_modalities_by_set"
    )
    state["evidence"]["results"].append(evidence_row(
        state,
        "cross_modal_consistency",
        "set_identity",
        ["C1"],
        {},
        "multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}},
        root,
    ))
    audit = VerifierOutput.model_validate({"findings": [{
        "target_ids": ["C1"],
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "subject_signature": subject_signature(
            "cross_modal_consistency", "set_identity", state["sets"], ["C1"]
        ),
        "status": "supporting",
        "metric_refs": [f"{root}.C1"],
    }], "gaps": []})

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
        "scope": "set_identity", "status": "conflicting", "metric_refs": ["x"],
    }], "gaps": []})
    reactivate_provisional_sets(state, audit)
    assert state["sets"][0]["status"] == "provisionally_dropped"


def test_split_proposal_confound_blocks_split_without_invalidating_parent():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]}])
    proposal = {
        "proposal_id": "p1", "plan_id": "p1", "source_set_id": "C1",
        "eligible_for_review": True,
        "groups": [[f"P{i}" for i in range(10)], [f"P{i}" for i in range(10, 20)]],
    }
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    identity_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    split_ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"]["results"].extend([
        evidence_row(state, "cross_modal_consistency", "set_identity", ["C1"], {}, "multimodal_consistency_check", {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}}, identity_ref),
        evidence_row(state, "cross_modal_consistency", "split_proposal", ["C1"], proposal, "multimodal_consistency_check", {"split_supporting_modalities": ["ct", "rna"]}, split_ref),
    ])
    state["audit"] = {"findings": [
        {"target_ids": ["C1"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [identity_ref]},
        {"target_ids": ["C1"], "proposal_id": "p1", "dimension": "cross_modal_consistency", "scope": "split_proposal", "status": "supporting", "metric_refs": [split_ref]},
    ], "gaps": []}
    add_identity_controls(state)
    add_proposal_checks(state, proposal, "split_proposal", ["C1"])
    state["audit"]["findings"][-2]["status"] = "conflicting"
    state["sets"][0]["status"] = "provisionally_accepted"
    audit = VerifierOutput.model_validate(state["audit"])

    assert not supported_revision_candidates(state)
    reactivate_provisional_sets(state, audit)
    assert state["sets"][0]["status"] == "provisionally_accepted"
    state["sets"][0]["status"] = "active"
    state["revision"] = None
    state["control"]["blocked_actions"].append("split:C1")
    validate_router_action(RouterAction(action="accept", target_ids=["C1"]), state)
    with pytest.raises(ValueError, match="positive confounder"):
        validate_router_action(RouterAction(action="drop", target_ids=["C1"]), state)


def test_merge_known_label_conflict_does_not_invalidate_parent_sets():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    proposal = {
        "proposal_id": "m1", "plan_id": "m1", "set_ids": ["C1", "C2"],
        "memberships": [["P1", "P2"], ["P3", "P4"]], "eligible_for_review": True,
    }
    state["revision"] = {
        "action": "merge", "target_ids": ["C1", "C2"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    merge_ref = "tool_results.multimodal_consistency_check.metrics.merge_supporting_modalities"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "merge_proposal", ["C1", "C2"], proposal,
        "multimodal_consistency_check",
        {"merge_supporting_modalities": ["ct", "wsi"], "merge_strong_boundary_modalities": []},
        merge_ref,
    ))
    state["audit"] = {"findings": [{
        "target_ids": ["C1", "C2"], "proposal_id": "m1",
        "dimension": "cross_modal_consistency", "scope": "merge_proposal",
        "status": "supporting", "metric_refs": [merge_ref],
    }], "gaps": []}
    add_proposal_checks(state, proposal, "merge_proposal", ["C1", "C2"], include_biology=True)
    state["audit"]["findings"][-2]["status"] = "conflicting"
    state["sets"][0]["status"] = "provisionally_dropped"

    reactivate_provisional_sets(state, VerifierOutput.model_validate(state["audit"]))
    assert state["sets"][0]["status"] == "active"


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
        "metric_refs": ["x"],
    }], "gaps": []})
    state["audit"] = audit.model_dump()

    assert not complete_audit(state)
    reactivate_provisional_sets(state, audit)
    assert state["sets"][0]["status"] == "active"


def test_successful_mandatory_evidence_without_finding_is_contract_failure():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    confound_ref = "tool_results.confound_test.metrics.strong_technical_conflict"
    known_ref = "tool_results.known_label_echo_test.metrics.near_identity"
    state["evidence"] = {"results": [
        evidence_row(state, "cross_modal_consistency", "set_identity", ["C1"], {}, "multimodal_consistency_check", {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}}, cross_ref),
        evidence_row(state, "confounder_exclusion", "set_identity", ["C1"], {}, "confound_test", {"strong_technical_conflict": False}, confound_ref),
        evidence_row(state, "known_label_echo", "partition", [], {}, "known_label_echo_test", {"near_identity": False}, known_ref),
    ]}
    audit = VerifierOutput.model_validate({"findings": [
        {"target_ids": ["C1"], "dimension": "confounder_exclusion", "scope": "set_identity", "subject_signature": subject_signature("confounder_exclusion", "set_identity", state["sets"], ["C1"]), "status": "supporting", "metric_refs": [confound_ref]},
        {"target_ids": [], "dimension": "known_label_echo", "scope": "partition", "subject_signature": subject_signature("known_label_echo", "partition", state["sets"], []), "status": "supporting", "metric_refs": [known_ref]},
    ], "gaps": []})

    with pytest.raises(ValueError, match="mandatory evidence has no corresponding Finding"):
        validate_verifier_audit(audit, state)


def test_default_review_budget_allows_mandatory_evidence_rounds():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    assert state["control"]["max_rounds"] == 60


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


def test_router_receives_legal_actions_and_traces_each_contract_rejection():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state,
        "cross_modal_consistency",
        "set_identity",
        ["C1"],
        {},
        "multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}},
        cross_ref,
    ))
    state["audit"] = {"findings": [{
        "target_ids": ["C1"],
        "dimension": "cross_modal_consistency",
        "scope": "set_identity",
        "status": "supporting",
        "metric_refs": [cross_ref],
    }], "gaps": []}
    add_identity_controls(state)
    rejected = {
        "action": "split",
        "target_ids": ["C1"],
        "proposal_id": "unsupported",
        "metric_refs": [],
    }
    model = StaticModel(rejected)

    router_node(state, {}, model)

    candidates = model.payloads[0]["legal_action_candidates"]
    assert [(row["action"], row["target_ids"]) for row in candidates] == [
        ("accept", ["C1"])
    ]
    rejections = [
        row for row in state["control"]["trace"]
        if row.get("event") == "contract_rejection"
    ]
    assert len(rejections) == 3
    assert all(row["rejected_action"]["proposal_id"] == "unsupported" for row in rejections)
    assert state["control"]["status"] == "review_failed_runtime"


def test_supported_structural_action_preempts_unrelated_evidence_requests():
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
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "ready_for_revision",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"]["results"].append(evidence_row(
        state,
        "cross_modal_consistency",
        "split_proposal",
        ["C1"],
        proposal,
        "multimodal_consistency_check",
        {"split_supporting_modalities": ["ct", "rna"]},
        ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"],
        "proposal_id": "p1",
        "dimension": "cross_modal_consistency",
        "scope": "split_proposal",
        "status": "supporting",
        "metric_refs": [ref],
    })
    add_proposal_checks(state, proposal, "split_proposal", ["C1"])
    identity_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "set_identity", ["C1", "C2"], {},
        "multimodal_consistency_check",
        {"identity_supporting_modalities_by_set": {"C1": ["ct"], "C2": ["ct", "rna"]}},
        identity_ref,
    ))
    add_identity_controls(state)
    model = StaticModel({
        "action": "split",
        "target_ids": ["C1"],
    })

    router_node(state, {}, model)

    payload = model.payloads[0]
    assert payload["requestable_evidence"] == []
    assert [(row["action"], row["target_ids"]) for row in payload["legal_action_candidates"]] == [
        ("split", ["C1"])
    ]


def test_revision_waits_for_all_candidate_evidence_before_becoming_ready():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]}
    ])
    candidates = [
        {
            "proposal_id": proposal_id,
            "plan_id": proposal_id,
            "source_set_id": "C1",
            "eligible_for_review": True,
            "groups": groups,
        }
        for proposal_id, groups in (
            ("p1", [[f"P{i}" for i in range(10)], [f"P{i}" for i in range(10, 20)]]),
            ("p2", [[f"P{i}" for i in range(0, 20, 2)], [f"P{i}" for i in range(1, 20, 2)]]),
        )
    ]
    state["revision"] = {
        "action": "split",
        "target_ids": ["C1"],
        "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]),
        "candidates": candidates,
    }
    identity_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    split_ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"]["results"].extend([
        evidence_row(
            state, "cross_modal_consistency", "set_identity", ["C1"], {},
            "multimodal_consistency_check",
            {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}},
            identity_ref,
        ),
        evidence_row(
            state, "cross_modal_consistency", "split_proposal", ["C1"], candidates[0],
            "multimodal_consistency_check", {"split_supporting_modalities": ["ct", "rna"]},
            split_ref,
        ),
    ])
    state["audit"]["findings"].extend([
        {
            "target_ids": ["C1"], "dimension": "cross_modal_consistency",
            "scope": "set_identity", "status": "supporting", "metric_refs": [identity_ref],
        },
        {
            "target_ids": ["C1"], "proposal_id": "p1",
            "dimension": "cross_modal_consistency", "scope": "split_proposal",
            "status": "supporting", "metric_refs": [split_ref],
        },
    ])
    add_identity_controls(state)
    add_proposal_checks(state, candidates[0], "split_proposal", ["C1"])
    model = StaticModel({
        "action": "need_more_evidence",
        "target_ids": ["C1"],
        "dimension": "cross_modal_consistency",
        "scope": "split_proposal",
        "proposal_id": "p2",
    })

    router_node(state, {}, model)

    assert state["revision"]["status"] == "evidence_collection"
    assert state["action"]["action"] == "need_more_evidence"
    assert model.payloads[0]["legal_action_candidates"] == []
    assert any(
        row["proposal_id"] == "p2" and row["dimension"] == "cross_modal_consistency"
        for row in model.payloads[0]["requestable_evidence"]
    )

    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "split_proposal", ["C1"], candidates[1],
        "multimodal_consistency_check", {"split_supporting_modalities": ["ct"]}, split_ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"], "proposal_id": "p2",
        "dimension": "cross_modal_consistency", "scope": "split_proposal",
        "status": "inconclusive", "metric_refs": [split_ref],
    })
    ready_model = StaticModel({"action": "split", "target_ids": ["C1"]})

    router_node(state, {}, ready_model)

    assert state["revision"]["status"] == "ready_for_revision"
    assert ready_model.payloads[0]["requestable_evidence"] == []
    assert ready_model.payloads[0]["legal_action_candidates"][0]["action"] == "split"


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
    cross_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"]["results"].append(evidence_row(
        state,
        "cross_modal_consistency",
        "set_identity",
        ["C1"],
        {},
        "multimodal_consistency_check",
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
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [p1, p2],
    }
    ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"] = {"results": [
        {"dimension": "cross_modal_consistency", "scope": "split_proposal", "proposal_id": "p1", "subject_signature": subject_signature("cross_modal_consistency", "split_proposal", state["sets"], ["C1"], p1), "results": [{"tool_name": "multimodal_consistency_check", "metrics": {"split_supporting_modalities": ["ct", "rna"]}, "metric_refs": [ref]}]},
        {"dimension": "cross_modal_consistency", "scope": "split_proposal", "proposal_id": "p2", "subject_signature": subject_signature("cross_modal_consistency", "split_proposal", state["sets"], ["C1"], p2), "results": [{"tool_name": "multimodal_consistency_check", "metrics": {"split_supporting_modalities": ["ct"]}, "metric_refs": [ref]}]},
    ]}
    state["audit"] = {"findings": [
        {"target_ids": ["C1"], "proposal_id": "p1", "dimension": "cross_modal_consistency", "scope": "split_proposal", "status": "supporting", "metric_refs": [ref]},
        {"target_ids": ["C1"], "proposal_id": "p2", "dimension": "cross_modal_consistency", "scope": "split_proposal", "status": "supporting", "metric_refs": [ref]},
    ], "gaps": []}
    add_proposal_checks(state, p1, "split_proposal", ["C1"])
    assert [row["proposal_id"] for row in supported_revision_candidates(state)] == ["p1"]
    state["audit"]["findings"].append({
        "target_ids": ["C1"], "dimension": "confounder_exclusion",
        "scope": "set_identity", "status": "conflicting", "metric_refs": [ref],
    })
    assert not supported_revision_candidates(state)
    state["audit"]["findings"].pop()
    state["control"]["policy"]["split_min_supporting_modalities"] = 3
    assert not supported_revision_candidates(state)


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
    for proposal in (p1, p2):
        proposal["partition_signature"] = partition_signature(state["sets"])
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "ready_for_revision",
        "partition_signature": partition_signature(state["sets"]), "candidates": [p1, p2],
    }
    metric_ref = (
        "tool_results.multimodal_consistency_check.metrics."
        "split_supporting_modalities"
    )
    state["evidence"] = {
        "results": [
            {
                "dimension": "cross_modal_consistency",
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
                        "tool_name": "multimodal_consistency_check",
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
    }
    model = StaticModel({"plan_id": "p1", "reason": "supported"})

    reviser_node(state, {}, model)

    assert [row["proposal_id"] for row in model.payloads[0]["candidates"]] == ["p1"]
    assert {row["set_id"] for row in state["sets"] if row["status"] == "active"} == {
        "C1_S1",
        "C1_S2",
    }
    trace = state["control"]["trace"][-1]
    assert trace["structural_metric_refs"]
    assert trace["audit_metric_refs"]


def test_c0003_supported_12_plus_18_split_is_applied_exactly():
    members = [f"P{i:02d}" for i in range(30)]
    state = initial_review_state([{"cluster_id": "C0003", "member_ids": members}])
    proposal = {
        "proposal_id": "c0003_split",
        "plan_id": "c0003_split",
        "source_set_id": "C0003",
        "eligible_for_review": True,
        "groups": [members[:12], members[12:]],
        "partition_signature": partition_signature(state["sets"]),
    }
    state["revision"] = {
        "action": "split", "target_ids": ["C0003"], "status": "ready_for_revision",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "split_proposal", ["C0003"], proposal,
        "multimodal_consistency_check", {"split_supporting_modalities": ["ct", "wsi", "rna"]}, ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C0003"], "proposal_id": "c0003_split",
        "dimension": "cross_modal_consistency", "scope": "split_proposal",
        "status": "supporting", "metric_refs": [ref],
    })
    add_proposal_checks(state, proposal, "split_proposal", ["C0003"])
    state["action"] = RouterAction(action="split", target_ids=["C0003"]).model_dump()

    reviser_node(state, {}, StaticModel({"plan_id": "c0003_split", "reason": "supported"}))

    active = {item["set_id"]: item["member_ids"] for item in state["sets"] if item["status"] == "active"}
    assert active == {"C0003_S1": members[:12], "C0003_S2": members[12:]}


def test_imaging_only_split_waits_for_biology_support():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]}])
    proposal = {
        "proposal_id": "p1", "plan_id": "p1", "source_set_id": "C1",
        "eligible_for_review": True, "groups": [[f"P{i}" for i in range(10)], [f"P{i}" for i in range(10, 20)]],
    }
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"]["results"].append(evidence_row(
        state, "cross_modal_consistency", "split_proposal", ["C1"], proposal,
        "multimodal_consistency_check", {"split_supporting_modalities": ["ct", "wsi"]}, ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"], "proposal_id": "p1", "dimension": "cross_modal_consistency",
        "scope": "split_proposal", "status": "supporting", "metric_refs": [ref],
    })
    add_proposal_checks(state, proposal, "split_proposal", ["C1"])

    assert any(
        row["dimension"] == "biological_support" and row["proposal_id"] == "p1"
        for row in required_evidence_requests(state)
    )
    assert not supported_revision_candidates(state)
    biology_ref = "tool_results.pathway_enrichment.metrics.signal"
    state["evidence"]["results"].append(evidence_row(
        state, "biological_support", "split_proposal", ["C1"], proposal,
        "pathway_enrichment", {"signal": 1}, biology_ref,
    ))
    state["audit"]["findings"].append({
        "target_ids": ["C1"], "proposal_id": "p1", "dimension": "biological_support",
        "scope": "split_proposal", "status": "supporting", "metric_refs": [biology_ref],
    })
    assert supported_revision_candidates(state)


def test_unsupported_revision_blocks_exact_intent_and_clears_proposal_audit():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": [f"P{i}" for i in range(20)]}])
    proposal = {
        "proposal_id": "p1", "plan_id": "p1", "source_set_id": "C1",
        "eligible_for_review": True, "groups": [[f"P{i}" for i in range(10)], [f"P{i}" for i in range(10, 20)]],
    }
    state["revision"] = {
        "action": "split", "target_ids": ["C1"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    base_ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    split_ref = "tool_results.multimodal_consistency_check.metrics.split_supporting_modalities"
    state["evidence"]["results"].extend([
        evidence_row(state, "cross_modal_consistency", "set_identity", ["C1"], {}, "multimodal_consistency_check", {"identity_supporting_modalities_by_set": {"C1": ["ct"]}}, base_ref),
        evidence_row(state, "cross_modal_consistency", "split_proposal", ["C1"], proposal, "multimodal_consistency_check", {"split_supporting_modalities": ["ct"]}, split_ref),
    ])
    state["audit"]["findings"].extend([
        {"target_ids": ["C1"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "inconclusive", "metric_refs": [base_ref]},
        {"target_ids": ["C1"], "proposal_id": "p1", "dimension": "cross_modal_consistency", "scope": "split_proposal", "status": "inconclusive", "metric_refs": [split_ref]},
    ])
    add_identity_controls(state)
    model = StaticModel({"action": "split", "target_ids": ["C1"]})

    router_node(state, {}, model)

    assert state["revision"] is None
    assert "split:C1" in state["control"]["blocked_actions"]
    assert all(row["scope"] not in {"split_proposal", "merge_proposal"} for row in state["audit"]["findings"])
    assert model.calls == 0


def test_accept_uses_only_target_set_identity_evidence_and_hard_vetoes_confounder():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    signature = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"])
    ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"] = {"results": [{
        "dimension": "cross_modal_consistency", "scope": "set_identity", "proposal_id": None,
        "subject_signature": signature, "results": [{"tool_name": "multimodal_consistency_check", "metrics": {"identity_supporting_modalities_by_set": {"C1": ["ct", "rna"]}}, "metric_refs": [ref]}],
    }]}
    state["audit"] = {"findings": [
        {"target_ids": ["C1"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [ref]},
    ], "gaps": []}
    add_identity_controls(state, confound_status="conflicting")
    with pytest.raises(ValueError, match="vetoed"):
        validate_router_action(RouterAction(action="accept", target_ids=["C1"]), state)


def test_accept_cannot_borrow_identity_metrics_from_split_scope():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])
    identity_signature = subject_signature("cross_modal_consistency", "set_identity", state["sets"], ["C1"])
    ref = "tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set"
    state["evidence"] = {"results": [
        {"dimension": "cross_modal_consistency", "scope": "set_identity", "proposal_id": None, "subject_signature": identity_signature, "results": [{"tool_name": "multimodal_consistency_check", "metrics": {"identity_supporting_modalities_by_set": {"C1": ["ct"]}}, "metric_refs": [ref]}]},
        {"dimension": "cross_modal_consistency", "scope": "split_proposal", "proposal_id": "p1", "subject_signature": "stale", "results": [{"tool_name": "multimodal_consistency_check", "metrics": {"identity_supporting_modalities_by_set": {"C1": ["ct", "wsi", "rna"]}}, "metric_refs": [ref]}]},
    ]}
    state["audit"] = {"findings": [{"target_ids": ["C1"], "dimension": "cross_modal_consistency", "scope": "set_identity", "status": "supporting", "metric_refs": [ref]}], "gaps": []}
    add_identity_controls(state)
    with pytest.raises(ValueError, match="at least two"):
        validate_router_action(RouterAction(action="accept", target_ids=["C1"]), state)


def test_merge_requires_two_weak_boundary_modalities_and_no_biology_veto():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    proposal = {"proposal_id": "m1", "plan_id": "m1", "set_ids": ["C1", "C2"], "memberships": [["P1", "P2"], ["P3", "P4"]], "eligible_for_review": True}
    state["revision"] = {
        "action": "merge", "target_ids": ["C1", "C2"], "status": "evidence_collection",
        "partition_signature": partition_signature(state["sets"]), "candidates": [proposal],
    }
    signature = subject_signature("cross_modal_consistency", "merge_proposal", state["sets"], ["C1", "C2"], proposal)
    ref = "tool_results.multimodal_consistency_check.metrics.merge_supporting_modalities"
    state["audit"] = {"findings": [{"target_ids": ["C1", "C2"], "proposal_id": "m1", "dimension": "cross_modal_consistency", "scope": "merge_proposal", "status": "supporting", "metric_refs": [ref]}], "gaps": []}
    state["evidence"] = {"results": [{"dimension": "cross_modal_consistency", "scope": "merge_proposal", "proposal_id": "m1", "subject_signature": signature, "results": [{"tool_name": "multimodal_consistency_check", "metrics": {"merge_supporting_modalities": ["ct"], "merge_strong_boundary_modalities": []}, "metric_refs": [ref]}]}]}
    assert not supported_revision_candidates(state)
    state["evidence"]["results"][0]["results"][0]["metrics"]["merge_supporting_modalities"] = ["ct", "wsi"]
    add_proposal_checks(
        state, proposal, "merge_proposal", ["C1", "C2"], include_biology=True
    )
    assert supported_revision_candidates(state)
    state["audit"]["findings"].append({"target_ids": ["C1", "C2"], "proposal_id": "m1", "dimension": "biological_support", "scope": "merge_proposal", "status": "conflicting", "metric_refs": [ref]})
    assert not supported_revision_candidates(state)


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
    result = cnv_characterization(
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
    result = cnv_characterization(
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
        "tools.multimodal_consistency_check.partition_separation", separation
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
