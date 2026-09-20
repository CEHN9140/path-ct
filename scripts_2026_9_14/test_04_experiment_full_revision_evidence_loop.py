import importlib.util
from pathlib import Path
from types import SimpleNamespace

from agents.subtype_review.graph import reviser_node, router_node
from agents.subtype_review.schemas import RevisionPlan

SCRIPT = Path(__file__).with_name("04_experiment_full_revision_evidence_loop.py")
SPEC = importlib.util.spec_from_file_location("full_revision_evidence_loop", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def capture_reviser_refs(state, plan):
    captured = {}
    state["router_plan"] = plan
    reviser_node(state, {"reviser_model": SimpleNamespace(
        invoke=lambda payload: captured.update(payload) or {},
    )})
    return captured["available_metric_refs"]


def test_revision_input_does_not_preload_post_revision_evidence(tmp_path):
    MODULE.write_synthetic_input(tmp_path)
    state = MODULE.build_initial_state("split")
    assert state["evidence_memory"] == {}


def test_synthetic_registry_preserves_real_tool_contracts():
    registry = MODULE.synthetic_registry()
    assert set(registry) >= {
        "pathway_enrichment",
        "mutation_enrichment",
        "cnv_characterization",
        "multimodal_consistency_check",
        "confound_test",
        "known_label_echo_test",
    }
    assert all(callable(metadata["function"]) for metadata in registry.values())


def test_full_loop_summary_requires_reacquisition():
    summary = MODULE.summarize_state(
        "split",
        {
            "control": {"status": "complete", "trace": [
                {"node": "reviser", "event": "revision_applied"},
                {"node": "router", "event": "decision", "plan": {
                    "evidence_requests": [{"dimension": "biological_support"}]
                }},
                {"node": "verifier", "event": "tool_selection"},
                {"node": "verifier", "event": "reports"},
                {"node": "router", "event": "decision", "plan": {"actions": []}},
            ]},
            "revision_result": {"new_partition_signature": "x"},
            "partition": {"sets": []},
            "reports": [{"dimension": "biological_support", "target_ids": []}],
            "evidence_memory": {
                MODULE.partition_signature([]): [
                    {"dimension": "biological_support", "target_ids": []}
                ]
            },
        },
        "initial-signature",
    )
    assert summary["verifier_reacquired_after_revision"] is True
    assert summary["full_loop_verified"] is True


def test_full_loop_summary_is_false_without_revision():
    summary = MODULE.summarize_state(
        "split",
        {
            "control": {"status": "complete", "trace": [
                {"node": "verifier", "event": "tool_selection"},
                {"node": "verifier", "event": "reports"},
            ]},
            "revision_result": {"new_partition_signature": "x"},
            "partition": {"sets": []},
        },
        "initial-signature",
    )
    assert summary["verifier_reacquired_after_revision"] is False
    assert summary["full_loop_verified"] is False


def test_controlled_structural_evidence_exposes_revision_metric_refs():
    for name in ("split", "merge"):
        state = MODULE.build_initial_state(name)
        row = next(
            item for item in state["round_evidence"]
            if item["tool_name"] == "multimodal_consistency_check"
        )
        assert row["metric_refs"]


def test_reviser_uses_only_metric_refs_for_structural_evidence():
    prompt = (Path(__file__).parents[1] / "agents/subtype_review/prompts/reviser.md").read_text()
    assert "schema field `metric_refs`" in prompt
    assert "structural_metric_refs" in prompt


def test_reviser_receives_only_refs_for_router_targets():
    state = MODULE.build_initial_state("split")
    plan = {
        "actions": [{
            "action": "split",
            "target_ids": ["C1"],
            "decision_state": {
                "identity": "supported",
                "structure": "incompatible",
                "alternative_explanation": "not_supported",
                "uncertainty": "no",
            },
        }]
    }
    refs = capture_reviser_refs(state, plan)
    assert refs
    assert all(
        ".multimodal_consistency_check.metrics.cross_modal_consistency.per_set.C1.internal_structure."
        in ref
        for ref in refs
    )


def test_reviser_merge_refs_are_limited_to_the_requested_pair():
    state = MODULE.build_initial_state("merge")
    plan = {
        "actions": [{
            "action": "merge",
            "target_ids": ["C1", "C2"],
            "decision_state": {
                "identity": "supported",
                "structure": "incompatible",
                "alternative_explanation": "not_supported",
                "uncertainty": "no",
            },
        }]
    }
    refs = capture_reviser_refs(state, plan)
    assert refs
    assert all(
        ".multimodal_consistency_check.metrics.cross_modal_consistency.per_set.C1.boundary_to_other_sets.C2."
        in ref
        or ".multimodal_consistency_check.metrics.cross_modal_consistency.per_set.C2.boundary_to_other_sets.C1."
        in ref
        for ref in refs
    )


def test_revision_refs_exclude_other_tool_metrics_for_the_same_target():
    state = MODULE.build_initial_state("split")
    state["round_evidence"].append({
        "tool_name": "pathway_enrichment",
        "metric_refs": ["tool_results.pathway_enrichment.metrics.sets.C1.effect"],
    })
    plan = {"actions": [{"action": "split", "target_ids": ["C1"],
                         "decision_state": MODULE.SANITY.assessed_state(structure="incompatible")}]}
    refs = capture_reviser_refs(state, plan)
    assert all("pathway_enrichment" not in ref for ref in refs)


def test_revision_retry_signature_includes_metric_refs(monkeypatch):
    first = RevisionPlan.model_validate({
        "split_plans": [{
            "target_id": "C1", "n_children": 2,
            "structural_basis": ["fused"],
            "execution_strategy": "fused_similarity_spectral",
            "metric_refs": ["ref.one"],
        }],
        "merge_plans": [],
    })
    second = first.model_copy(update={"split_plans": [first.split_plans[0].model_copy(update={"metric_refs": ["ref.two"]})]})
    state = MODULE.build_initial_state("split")
    router_plan = {"actions": [
        {"action": "split", "target_ids": ["C1"], "decision_state": MODULE.SANITY.assessed_state(structure="incompatible")},
        {"action": "accept", "target_ids": ["C2"], "decision_state": MODULE.SANITY.assessed_state()},
    ]}
    state.update(router_node(state, {"router_model": SimpleNamespace(invoke=lambda payload: router_plan)}))
    assert state["control"]["error"] is None
    for plan in (first, second):
        state.update(reviser_node(state, {"reviser_model": SimpleNamespace(invoke=lambda payload: plan.model_dump())}))
        assert "unavailable metrics" in state["control"]["error"]
    assert len(state["control"]["failed_revision_plan_signatures"]) == 2

    valid_refs = capture_reviser_refs(state, router_plan)
    fixed = first.model_copy(update={"split_plans": [first.split_plans[0].model_copy(update={"metric_refs": valid_refs})]})
    monkeypatch.setattr("tools.cross_modal_structure.execute_split_membership",
                        lambda root, members, *args: [members[:len(members)//2], members[len(members)//2:]])
    state.update(reviser_node(state, {"data_root": "/tmp", "reviser_model": SimpleNamespace(invoke=lambda payload: fixed.model_dump())}))
    assert state["control"]["error"] is None
    assert state["control"]["next"] == "router"
    assert state["revision_result"]["operations"][0]["action"] == "split"
