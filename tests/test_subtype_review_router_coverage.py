import json
import pytest

from agents.subtype_review.graph import (
    eligible_tools_for_request,
    initial_review_state,
    partition_signature,
    reason_contradicts_action,
    router_node,
    terminal_accountability_refs,
    validate_router_plan,
)
from agents.subtype_review.schemas import EvidenceRequest, RouterAction, RouterPlan
from agents.subtype_review.tools import TOOL_REGISTRY
from agents.subtype_review.llm import summarize_reports


def test_accept_reason_contradiction_is_detected():
    assert reason_contradicts_action(
        "accept",
        "Terminal acceptance is not warranted on the current evidence.",
    )


def test_accept_reason_qualification_is_not_overblocked():
    assert not reason_contradicts_action(
        "accept",
        "Acceptance is not warranted by biology alone, but the supported "
        "boundary establishes independent representation.",
    )


def test_drop_reason_contradiction_is_detected():
    assert reason_contradicts_action(
        "drop",
        "The candidate should be retained.",
    )


def test_positive_accept_reason_is_not_flagged():
    assert not reason_contradicts_action(
        "accept",
        "The candidate remains a defensible independent discovery-stage unit.",
    )


def terminal_validation_state(reports):
    state = initial_review_state([
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    state["reports"] = [
        {
            "report_ref": "ER:partition",
            "dimension": "cross_modal_consistency",
            "aspect": "structural_diagnostics",
            "scope": "partition",
            "target_ids": [],
        },
        *reports,
    ]
    return state


def terminal_action(action, target="C1", refs=None):
    return RouterAction(
        action=action, target_ids=[target], n_children=None,
        evidence_report_refs=refs or ["ER:partition"], reason="reason",
    )


def test_question_focus_binds_cross_modal_tool_family():
    for focus, scope, target_ids, tool_name in (
        ("membership_representation", "set", ["C1"], "representation_concordance"),
        ("internal_subdivision", "set", ["C1"], "structural_diagnostics"),
        ("boundary_representation", "pair", ["C1", "C2"], "representation_concordance"),
        ("boundary_structure", "pair", ["C1", "C2"], "structural_diagnostics"),
    ):
        request = EvidenceRequest(
            dimension="cross_modal_consistency", scope=scope,
            target_ids=target_ids, focus=focus, question="Assess the declared focus.",
        )
        assert eligible_tools_for_request(
            request, TOOL_REGISTRY, set(), set(), partition_screen_done=True,
        ) == [tool_name]


def test_accept_requires_exact_set_membership_report():
    state = terminal_validation_state([{
        "report_ref": "ER:biology", "dimension": "biological_support",
        "aspect": "rna_pathway_enrichment", "scope": "set", "target_ids": ["C1"],
    }])
    plan = RouterPlan(actions=[terminal_action("accept", refs=["ER:biology"]), terminal_action("drop", "C2")])
    with pytest.raises(ValueError, match="membership_representation"):
        validate_router_plan(plan, state, set())


def test_pair_boundary_cannot_substitute_for_accept_membership():
    state = terminal_validation_state([{
        "report_ref": "ER:pair", "dimension": "cross_modal_consistency",
        "aspect": "affinity_geometry_concordance", "scope": "pair", "target_ids": ["C1", "C2"],
    }])
    plan = RouterPlan(actions=[terminal_action("accept", refs=["ER:pair"]), terminal_action("drop", "C2")])
    with pytest.raises(ValueError, match="membership_representation"):
        validate_router_plan(plan, state, set())


def test_accept_with_membership_report_passes_role_contract():
    state = terminal_validation_state([{
        "report_ref": "ER:membership", "dimension": "cross_modal_consistency",
        "aspect": "affinity_geometry_concordance", "scope": "set", "target_ids": ["C1"],
        "request_foci": ["membership_representation"],
    }, {
        "report_ref": "ER:biology", "dimension": "biological_support",
        "aspect": "rna_pathway_enrichment", "scope": "set", "target_ids": ["C1"],
    }])
    plan = RouterPlan(actions=[terminal_action("accept", refs=["ER:membership", "ER:biology"]), terminal_action("drop", "C2")])
    validate_router_plan(plan, state, set())


def test_drop_does_not_require_fixed_membership_report():
    state = terminal_validation_state([{
        "report_ref": "ER:biology", "dimension": "biological_support",
        "aspect": "rna_pathway_enrichment", "scope": "set", "target_ids": ["C1"],
    }])
    plan = RouterPlan(actions=[terminal_action("drop", refs=["ER:biology"]), terminal_action("drop", "C2")])
    validate_router_plan(plan, state, set())


def test_terminal_drop_must_cite_all_target_specific_reports():
    state = terminal_validation_state([{
        "report_ref": "ER:biology", "dimension": "biological_support",
        "aspect": "rna_pathway_enrichment", "scope": "set", "target_ids": ["C1"],
    }, {
        "report_ref": "ER:pair", "dimension": "cross_modal_consistency",
        "aspect": "affinity_geometry_concordance", "scope": "pair", "target_ids": ["C1", "C2"],
    }])
    plan = RouterPlan(actions=[terminal_action("drop", refs=["ER:biology"]), terminal_action("drop", "C2", refs=["ER:partition"])])
    with pytest.raises(ValueError, match="all target-specific"):
        validate_router_plan(plan, state, set(), terminal_accountability_refs_by_target=terminal_accountability_refs(state["partition"]["sets"], state["reports"]))


def test_partition_report_is_not_duplicated_into_terminal_accountability():
    state = terminal_validation_state([])
    refs = terminal_accountability_refs(state["partition"]["sets"], state["reports"])
    assert refs == {"C1": set(), "C2": set()}


def test_pair_report_is_accountable_to_both_involved_targets():
    state = terminal_validation_state([{
        "report_ref": "ER:pair", "dimension": "cross_modal_consistency",
        "aspect": "affinity_geometry_concordance", "scope": "pair",
        "target_ids": ["C1", "C2"],
    }])
    refs = terminal_accountability_refs(state["partition"]["sets"], state["reports"])
    assert refs["C1"] == {"ER:pair"}
    assert refs["C2"] == {"ER:pair"}


def test_structural_revision_does_not_require_terminal_accountability_refs():
    state = terminal_validation_state([{
        "report_ref": "ER:boundary", "dimension": "cross_modal_consistency",
        "scope": "pair", "target_ids": ["C1", "C2"],
        "request_foci": ["boundary_representation"],
    }, {
        "report_ref": "ER:struct", "dimension": "cross_modal_consistency",
        "aspect": "structural_diagnostics", "scope": "pair", "target_ids": ["C1", "C2"],
        "request_foci": ["boundary_structure"],
    }])
    state["partition"]["sets"].append({"set_id": "C3", "member_ids": ["P5", "P6"], "revision_lineage": []})
    state["tool_evidence"] = [{
        "tool_name": "structural_diagnostics", "scope": "pair", "target_ids": ["C1", "C2"],
        "partition_signature": partition_signature(state["partition"]["sets"]),
        "metrics": {},
    }]
    action = RouterAction(action="merge", target_ids=["C1", "C2"], evidence_report_refs=["ER:boundary", "ER:struct"], reason="reason")
    plan = RouterPlan(actions=[action])
    validate_router_plan(plan, state, set(), terminal_accountability_refs_by_target={"C1": {"ER:other"}, "C2": {"ER:other"}})


def test_accept_with_membership_but_without_biology_fails():
    state = terminal_validation_state([{
        "report_ref": "ER:membership", "dimension": "cross_modal_consistency",
        "aspect": "affinity_geometry_concordance", "scope": "set", "target_ids": ["C1"],
        "request_foci": ["membership_representation"],
    }])
    plan = RouterPlan(actions=[terminal_action("accept", refs=["ER:membership"]), terminal_action("drop", "C2")])
    with pytest.raises(ValueError, match="biological_support"):
        validate_router_plan(plan, state, set())


def test_accept_with_biology_but_without_membership_fails():
    state = terminal_validation_state([{
        "report_ref": "ER:biology", "dimension": "biological_support",
        "aspect": "rna_pathway_enrichment", "scope": "set", "target_ids": ["C1"],
    }])
    plan = RouterPlan(actions=[terminal_action("accept", refs=["ER:biology"]), terminal_action("drop", "C2")])
    with pytest.raises(ValueError, match="membership_representation"):
        validate_router_plan(plan, state, set())


def test_report_focus_provenance_survives_router_summary():
    summary = summarize_reports([{
        "report_ref": "ER:membership",
        "dimension": "cross_modal_consistency",
        "aspect": "affinity_geometry_concordance",
        "scope": "set",
        "target_ids": ["C1"],
        "request_foci": ["membership_representation"],
    }])
    assert summary[0]["request_foci"] == ["membership_representation"]


def test_structural_action_legality_is_explicit_and_pair_review_remains_available(tmp_path):
    for set_ids, allowed_actions in (
        (["C1", "C2"], ["split"]),
        (["C1", "C2", "C3", "C4"], ["split", "merge"]),
    ):
        state = initial_review_state([
            {"set_id": set_id, "member_ids": [f"{set_id}-P1", f"{set_id}-P2"]}
            for set_id in set_ids
        ])
        signature = partition_signature(state["partition"]["sets"])
        pair = ["C1", "C2"]
        state["tool_evidence"].append({
            "tool_name": "structural_diagnostics", "scope": "partition",
            "target_ids": [], "partition_signature": signature,
            "metrics": {"partition": {
                "nearest_pair_targets": [pair],
                "nearest_pair_affinities": [{"target_ids": pair, "mean_between_affinity": 0.1}],
            }},
        })
        state["reports"].append({
            "report_ref": "ER:partition", "dimension": "cross_modal_consistency",
            "aspect": "structural_diagnostics", "scope": "partition", "target_ids": [],
        })
        captured = {}

        class Router:
            def invoke(self, payload):
                captured.update(payload)
                return {"actions": [], "evidence_requests": [{
                    "dimension": "cross_modal_consistency", "scope": "pair",
                    "target_ids": pair, "focus": "boundary_representation",
                    "question": "Assess whether the current boundary is defensible.",
                }]}

        runtime = {
            "router_model": Router(), "tool_registry": TOOL_REGISTRY,
            "patient_states_by_id": {}, "data_root": "/tmp",
            "artifact_root": "/tmp", "config_dir": "configs",
            "runtime_trace_path": str(tmp_path / f"trace-{len(set_ids)}.jsonl"),
        }
        router_node(state, runtime)
        assert captured["workflow_constraints"]["allowed_structural_actions"] == allowed_actions
        assert any(
            item["dimension"] == "cross_modal_consistency"
            and item["scope"] == "pair" and item["target_ids"] == pair
            for item in captured["available_evidence_requests"]
        )
        trace_rows = [
            json.loads(line)
            for line in (tmp_path / f"trace-{len(set_ids)}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        context = next(row for row in trace_rows if row["event"] == "router_context")
        assert context["workflow_constraints"] == captured["workflow_constraints"]


def test_router_receives_remaining_question_foci_without_tool_or_aspect_names():
    state = initial_review_state([
        {"set_id": set_id, "member_ids": [f"{set_id}-P1", f"{set_id}-P2"]}
        for set_id in ("C1", "C2")
    ])
    signature = partition_signature(state["partition"]["sets"])
    state["tool_evidence"] = [
        {"tool_name": "structural_diagnostics", "scope": "partition", "target_ids": [], "partition_signature": signature,
         "metrics": {"partition": {
             "nearest_pair_targets": [["C1", "C2"]],
             "nearest_pair_affinities": [{"target_ids": ["C1", "C2"], "mean_between_affinity": 0.1}],
         }}},
        {"tool_name": "representation_concordance", "scope": "set", "target_ids": ["C1"], "partition_signature": signature},
    ]
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set",
                "target_ids": ["C1"], "focus": "transcriptomic_phenotype",
                "question": "Assess the candidate phenotype.",
            }]}

    router_node(state, {
        "router_model": Router(), "tool_registry": TOOL_REGISTRY,
        "patient_states_by_id": {}, "data_root": "/tmp",
        "artifact_root": "/tmp", "config_dir": "configs",
    })
    options = captured["available_evidence_requests"]
    c1 = next(item for item in options if item["dimension"] == "cross_modal_consistency"
              and item["scope"] == "set" and item["target_ids"] == ["C1"])
    c2 = next(item for item in options if item["dimension"] == "cross_modal_consistency"
              and item["scope"] == "set" and item["target_ids"] == ["C2"])
    assert c1["available_question_foci"] == ["internal_subdivision"]
    assert c2["available_question_foci"] == ["internal_subdivision", "membership_representation"]
    pair = next(item for item in options if item["dimension"] == "cross_modal_consistency"
                and item["scope"] == "pair" and item["target_ids"] == ["C1", "C2"])
    assert pair["available_question_foci"] == ["boundary_representation"]
    assert "available_aspects" not in str(options)
    assert "representation_concordance" not in str(options)
    assert "structural_diagnostics" not in str(options)


def pair_options_after_completed_tools(completed_pair_tools):
    state = initial_review_state([
        {"set_id": set_id, "member_ids": [f"{set_id}-P1", f"{set_id}-P2"]}
        for set_id in ("C1", "C2")
    ])
    signature = partition_signature(state["partition"]["sets"])
    state["tool_evidence"] = [{
        "tool_name": "structural_diagnostics", "scope": "partition", "target_ids": [],
        "partition_signature": signature,
        "metrics": {"partition": {
            "nearest_pair_targets": [["C1", "C2"]],
            "nearest_pair_affinities": [{"target_ids": ["C1", "C2"], "mean_between_affinity": 0.1}],
        }},
    }]
    state["tool_evidence"].extend({
        "tool_name": tool_name, "scope": "pair", "target_ids": ["C1", "C2"],
        "partition_signature": signature,
    } for tool_name in completed_pair_tools)
    if "representation_concordance" in completed_pair_tools:
        state["reports"].append({
            "report_ref": "ER:pair-boundary", "dimension": "cross_modal_consistency",
            "aspect": "affinity_geometry_concordance", "scope": "pair",
            "target_ids": ["C1", "C2"], "request_foci": ["boundary_representation"],
        })
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
                "focus": "transcriptomic_phenotype", "question": "Clarify the candidate phenotype.",
            }]}

    router_node(state, {
        "router_model": Router(), "tool_registry": TOOL_REGISTRY,
        "patient_states_by_id": {}, "data_root": "/tmp",
        "artifact_root": "/tmp", "config_dir": "configs",
    })
    return captured["available_evidence_requests"]


def test_pair_boundary_representation_leaves_union_structure_available():
    options = pair_options_after_completed_tools(["representation_concordance"])
    pair = next(item for item in options if item["dimension"] == "cross_modal_consistency"
                and item["scope"] == "pair" and item["target_ids"] == ["C1", "C2"])
    assert pair["available_question_foci"] == ["boundary_structure"]
    assert "representation_concordance" not in str(options)
    assert "structural_diagnostics" not in str(options)
    assert "affinity_geometry_concordance" not in str(options)


def test_pair_boundary_structure_is_gated_until_boundary_representation():
    options = pair_options_after_completed_tools([])
    pair = next(item for item in options if item["dimension"] == "cross_modal_consistency"
                and item["scope"] == "pair" and item["target_ids"] == ["C1", "C2"])
    assert pair["available_question_foci"] == ["boundary_representation"]


def test_completed_pair_boundary_and_union_structure_remove_pair_capability():
    options = pair_options_after_completed_tools([
        "representation_concordance", "structural_diagnostics",
    ])
    assert not any(item["dimension"] == "cross_modal_consistency"
                   and item["scope"] == "pair" and item["target_ids"] == ["C1", "C2"]
                   for item in options)
