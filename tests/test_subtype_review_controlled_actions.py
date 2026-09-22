import json
import pytest
import numpy as np

from agents.subtype_review.graph import initial_review_state, partition_signature, validate_router_plan, reviser_node, router_node
from agents.subtype_review.tools import TOOL_REGISTRY
from agents.subtype_review.schemas import RouterAction, RouterPlan


def report(ref, dimension, scope, target_ids, *, focus=None):
    row = {
        "report_ref": ref,
        "dimension": dimension,
        "aspect": "structural_diagnostics" if dimension == "cross_modal_consistency" else "rna_pathway_enrichment",
        "scope": scope,
        "target_ids": list(target_ids),
    }
    if focus:
        row["request_foci"] = [focus]
    return row


def state_with_reports(sets, reports, tool_evidence=()):
    state = initial_review_state(sets)
    state["reports"] = list(reports)
    state["tool_evidence"] = list(tool_evidence)
    return state


def test_controlled_action_fixtures_cover_accept_drop_split_merge():
    base_sets = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
    ]
    partition = report("ER:partition", "cross_modal_consistency", "partition", [])

    accept_state = state_with_reports(base_sets, [
        partition,
        report("ER:C1-membership", "cross_modal_consistency", "set", ["C1"], focus="membership_representation"),
        report("ER:C1-biology", "biological_support", "set", ["C1"]),
        report("ER:C2-biology", "biological_support", "set", ["C2"]),
    ])
    accept = RouterAction(action="accept", target_ids=["C1"], evidence_report_refs=[
        "ER:C1-membership", "ER:C1-biology",
    ], reason="identity and membership are supported")
    drop = RouterAction(action="drop", target_ids=["C2"], evidence_report_refs=[
        "ER:C2-biology",
    ], reason="independent retention is unsupported")
    validate_router_plan(RouterPlan(actions=[accept, drop]), accept_state, set())

    split_sets = [{"set_id": "C1", "member_ids": ["P1", "P2", "P3", "P4"]},
                  {"set_id": "C2", "member_ids": ["P5", "P6"]}]
    split_state = state_with_reports(
        split_sets,
        [partition, report("ER:C1-structure", "cross_modal_consistency", "set", ["C1"])],
        [{
            "tool_name": "structural_diagnostics", "scope": "set", "target_ids": ["C1"],
            "partition_signature": partition_signature(split_sets),
            "metrics": {"set": {"C1": {"solutions": {"2": {"child_sizes": [2, 2]}}}}},
        }],
    )
    split = RouterAction(action="split", target_ids=["C1"], n_children=2,
                         evidence_report_refs=["ER:C1-structure"], reason="feasible subdivision")
    validate_router_plan(RouterPlan(actions=[split]), split_state, set())

    merge_sets = split_sets + [{"set_id": "C3", "member_ids": ["P7", "P8"]}]
    merge_state = state_with_reports(
        merge_sets,
        [
            partition,
            report("ER:C1-C2-boundary", "cross_modal_consistency", "pair", ["C1", "C2"], focus="boundary_representation"),
            report("ER:C1-C2-structure", "cross_modal_consistency", "pair", ["C1", "C2"], focus="boundary_structure"),
        ],
    )
    merge = RouterAction(action="merge", target_ids=["C1", "C2"],
                         evidence_report_refs=["ER:C1-C2-boundary", "ER:C1-C2-structure"], reason="revise boundary")
    validate_router_plan(RouterPlan(actions=[merge]), merge_state, set())


def test_merge_requires_both_pair_review_stages():
    sets = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
        {"set_id": "C3", "member_ids": ["P5", "P6"]},
    ]
    state = state_with_reports(sets, [
        report("ER:partition", "cross_modal_consistency", "partition", []),
        report("ER:structure", "cross_modal_consistency", "pair", ["C1", "C2"], focus="boundary_structure"),
    ])
    action = RouterAction(
        action="merge", target_ids=["C1", "C2"],
        evidence_report_refs=["ER:structure"], reason="revise boundary",
    )
    with pytest.raises(ValueError, match="boundary_representation"):
        validate_router_plan(RouterPlan(actions=[action]), state, set())


def test_controlled_structural_fixtures_revise_and_reenter_review(tmp_path):
    patient_ids = [f"P{i}" for i in range(8)]
    candidate = tmp_path / "candidate_subtype"
    (candidate / "consensus_cluster").mkdir(parents=True)
    (candidate / "affinity_patient_order.json").write_text(json.dumps(patient_ids))
    matrix = np.full((8, 8), 0.05)
    matrix[:4, :4] = 0.9
    matrix[4:, 4:] = 0.9
    np.fill_diagonal(matrix, 0.0)
    np.save(candidate / "consensus_cluster" / "consensus_matrix_Ktest.npy", matrix)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        "cross_modal:\n  structural:\n    max_children: 3\n    min_child_size: 2\n",
        encoding="utf-8",
    )
    generator = {"geometry": {
        "type": "resampled_consensus_coassignment",
        "matrix_relative_path": "consensus_cluster/consensus_matrix_Ktest.npy",
        "patient_order_relative_path": "affinity_patient_order.json",
    }}

    def apply_revision(sets, action, plan):
        state = initial_review_state([{**item, "generator": generator} for item in sets])
        state["router_plan"] = RouterPlan(actions=[action]).model_dump()
        state["history"] = [{"round": 1}]
        state["control"]["round"] = 1

        class Reviser:
            config = {}
            def invoke(self, payload):
                return plan

        return reviser_node(state, {
            "reviser_model": Reviser(), "data_root": str(tmp_path),
            "config_dir": str(config_dir),
        })

    split_action = RouterAction(action="split", target_ids=["C1"], n_children=2,
                                evidence_report_refs=["ER:C1-structure"], reason="subdivision")
    split_result = apply_revision(
        [{"set_id": "C1", "member_ids": patient_ids[:4]}, {"set_id": "C2", "member_ids": patient_ids[4:]}],
        split_action,
        {"split_plans": [{"target_id": "C1", "n_children": 2,
                          "structural_basis": ["candidate_consensus"],
                          "execution_strategy": "candidate_consensus_spectral"}], "merge_plans": []},
    )
    assert {item["set_id"] for item in split_result["partition"]["sets"]} == {"C1_S1", "C1_S2", "C2"}
    assert split_result["control"]["next"] == "router"
    assert all(item["generator"] == generator for item in split_result["partition"]["sets"])

    merge_action = RouterAction(action="merge", target_ids=["C1", "C2"],
                                evidence_report_refs=["ER:boundary", "ER:structure"], reason="weak boundary")
    merge_result = apply_revision(
        [{"set_id": "C1", "member_ids": patient_ids[:2]},
         {"set_id": "C2", "member_ids": patient_ids[2:4]},
         {"set_id": "C3", "member_ids": patient_ids[4:]}],
        merge_action,
        {"split_plans": [], "merge_plans": [{"target_ids": ["C1", "C2"]}]},
    )
    assert {item["set_id"] for item in merge_result["partition"]["sets"]} == {"C1_M_C2", "C3"}
    assert merge_result["control"]["next"] == "router"
    assert all(item["generator"] == generator for item in merge_result["partition"]["sets"])


def test_controlled_fixture_evidence_is_visible_in_router_input(tmp_path):
    sets = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
        {"set_id": "C3", "member_ids": ["P5", "P6"]},
    ]
    state = initial_review_state(sets)
    state["reports"] = [
        report("ER:partition", "cross_modal_consistency", "partition", []),
        report("ER:A-membership", "cross_modal_consistency", "set", ["C1"], focus="membership_representation"),
        report("ER:A-biology", "biological_support", "set", ["C1"]),
        report("ER:B-membership", "cross_modal_consistency", "set", ["C2"], focus="membership_representation"),
        report("ER:C-structure", "cross_modal_consistency", "set", ["C3"], focus="internal_subdivision"),
        report("ER:D-boundary", "cross_modal_consistency", "pair", ["C1", "C2"], focus="boundary_representation"),
        report("ER:D-structure", "cross_modal_consistency", "pair", ["C1", "C2"], focus="boundary_structure"),
    ]
    state["tool_evidence"] = [{
        "tool_name": "structural_diagnostics", "scope": "partition", "target_ids": [],
        "partition_signature": partition_signature(sets),
        "metrics": {"partition": {"nearest_pair_targets": [["C1", "C2"]]}},
    }]
    captured = {}

    class Router:
        config = {"router_plan_validation_retries": 0}
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
                "focus": "transcriptomic_phenotype", "question": "Assess the phenotype.",
            }]}

    router_node(state, {
        "router_model": Router(), "tool_registry": TOOL_REGISTRY,
        "patient_states_by_id": {}, "data_root": str(tmp_path),
        "config_dir": "configs", "artifact_root": str(tmp_path),
    })
    refs = {row["report_ref"] for row in captured["evidence_reports"]}
    assert {"ER:A-membership", "ER:A-biology", "ER:B-membership", "ER:C-structure",
            "ER:D-boundary", "ER:D-structure"}.issubset(refs)
    assert {item["report_ref"] for item in captured["completed_evidence_requests"]} >= {
        "ER:A-membership", "ER:B-membership", "ER:C-structure", "ER:D-boundary", "ER:D-structure",
    }
