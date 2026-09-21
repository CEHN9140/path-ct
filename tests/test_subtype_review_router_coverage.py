import json

from agents.subtype_review.graph import initial_review_state, partition_signature, router_node
from agents.subtype_review.evidence_semantics import EVIDENCE_ROLE_CONTRACTS
from agents.subtype_review.schemas import EVIDENCE_DIMENSIONS
from agents.subtype_review.tools import TOOL_REGISTRY


def test_partition_reports_update_partition_coverage_without_nesting():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    signature = partition_signature(state["partition"]["sets"])
    state["tool_evidence"].append({
        "tool_name": "structural_diagnostics", "scope": "partition",
        "target_ids": [], "partition_signature": signature,
    })
    state["reports"].append({
        "dimension": "cross_modal_consistency", "scope": "partition",
        "target_ids": [], "report_ref": "ER:partition",
    })
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set",
                "target_ids": ["C1"], "question": "Clarify C1 phenotype.",
            }]}

    runtime = {
        "router_model": Router(), "tool_registry": TOOL_REGISTRY,
        "patient_states_by_id": {}, "data_root": "/tmp",
        "artifact_root": "/tmp", "config_dir": "configs",
    }
    router_node(state, runtime)

    coverage = captured["evidence_coverage"]["partition"]
    assert coverage["cross_modal_consistency"] == "assessed"
    assert set(coverage) == set(EVIDENCE_DIMENSIONS)


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
            "metrics": {"partition": {"nearest_pair_targets": [pair]}},
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
                    "target_ids": pair, "question": "Assess whether the current boundary is defensible.",
                }]}

        runtime = {
            "router_model": Router(), "tool_registry": TOOL_REGISTRY,
            "patient_states_by_id": {}, "data_root": "/tmp",
            "artifact_root": "/tmp", "config_dir": "configs",
            "runtime_trace_path": str(tmp_path / f"trace-{len(set_ids)}.jsonl"),
        }
        router_node(state, runtime)
        assert captured["workflow_constraints"]["allowed_structural_actions"] == allowed_actions
        assert captured["evidence_dimension_contracts"] == EVIDENCE_ROLE_CONTRACTS
        forbidden_actions = captured["workflow_constraints"]["forbidden_structural_actions"]
        assert forbidden_actions == ([] if "merge" in allowed_actions else [{
            "action": "merge",
            "reason": "This workflow does not permit a merge that leaves fewer than two current sets.",
        }])
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
        assert context["evidence_dimension_contracts"] == EVIDENCE_ROLE_CONTRACTS


def test_router_receives_remaining_question_foci_without_tool_or_aspect_names():
    state = initial_review_state([
        {"set_id": set_id, "member_ids": [f"{set_id}-P1", f"{set_id}-P2"]}
        for set_id in ("C1", "C2")
    ])
    signature = partition_signature(state["partition"]["sets"])
    state["tool_evidence"] = [
        {"tool_name": "structural_diagnostics", "scope": "partition", "target_ids": [], "partition_signature": signature,
         "metrics": {"partition": {"nearest_pair_targets": [["C1", "C2"]]}}},
        {"tool_name": "representation_concordance", "scope": "set", "target_ids": ["C1"], "partition_signature": signature},
    ]
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set",
                "target_ids": ["C1"], "question": "Assess the candidate phenotype.",
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
    assert pair["available_question_foci"] == ["boundary_representation", "boundary_structure"]
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
        "metrics": {"partition": {"nearest_pair_targets": [["C1", "C2"]]}},
    }]
    state["tool_evidence"].extend({
        "tool_name": tool_name, "scope": "pair", "target_ids": ["C1", "C2"],
        "partition_signature": signature,
    } for tool_name in completed_pair_tools)
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
                "question": "Clarify the candidate phenotype.",
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


def test_completed_pair_boundary_and_union_structure_remove_pair_capability():
    options = pair_options_after_completed_tools([
        "representation_concordance", "structural_diagnostics",
    ])
    assert not any(item["dimension"] == "cross_modal_consistency"
                   and item["scope"] == "pair" and item["target_ids"] == ["C1", "C2"]
                   for item in options)
