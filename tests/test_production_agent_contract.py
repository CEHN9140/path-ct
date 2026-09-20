import pytest

from agents.subtype_review.schemas import (
    EvidenceReport, EvidenceRequest, RouterAction, RouterDecisionState, RouterPlan,
)
from agents.subtype_review.graph import initial_review_state, validate_router_plan


def test_evidence_scopes_enforce_exact_target_cardinality():
    common = {
        "dimension": "cross_modal_consistency",
        "observations": [],
    }
    assert EvidenceReport(**common, scope="set", target_ids=["C1"]).scope == "set"
    assert EvidenceReport(**common, scope="pair", target_ids=["C1", "C2"]).scope == "pair"
    assert EvidenceReport(**common, scope="partition", target_ids=[]).scope == "partition"
    with pytest.raises(ValueError):
        EvidenceRequest(dimension="cross_modal_consistency", scope="pair", target_ids=["C1"], question="x")


def test_router_split_requires_suggested_child_count_and_other_actions_forbid_it():
    state = RouterDecisionState(
        identity="uncertain", structure="incompatible",
        alternative_explanation="unassessed", uncertainty="yes",
    )
    assert RouterAction(action="split", target_ids=["C1"], n_children=3, decision_state=state).n_children == 3
    with pytest.raises(ValueError):
        RouterAction(action="split", target_ids=["C1"], decision_state=state)
    with pytest.raises(ValueError):
        RouterAction(action="accept", target_ids=["C1"], n_children=2, decision_state=state)


def test_review_tool_registry_is_the_production_seven_tool_set():
    from agents.subtype_review.tools import TOOL_REGISTRY

    assert set(TOOL_REGISTRY) == {
        "representation_concordance",
        "structural_diagnostics",
        "rna_pathway_enrichment",
        "wxs_mutation_enrichment",
        "known_label_echo",
        "confounder_association",
        "confounder_representation_effect",
    }
    assert set(TOOL_REGISTRY["structural_diagnostics"]["scopes"]) == {"set", "pair", "partition"}


def test_confounder_numeric_factor_with_one_observed_group_is_not_estimable(monkeypatch, tmp_path):
    import agents.subtype_review.tools as review_tools

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text("confounder:\n  permutations: 9\n")
    monkeypatch.setattr(review_tools, "technical_values", lambda *_: {
        "P1": {"ct_slice_thickness": 2.5},
    })
    result = review_tools.confounder_association(
        {}, str(tmp_path), str(config_dir),
        [{"set_id": "C1", "member_ids": ["P1"]}, {"set_id": "C2", "member_ids": ["P2"]}],
        "set", ["C1"],
    )

    row = result["metrics"]["set"]["C1"][0]
    assert row["factor"] == "ct_slice_thickness"
    assert row["test"] == "not_estimable"
    assert row["p_value"] is None


def test_split_uses_verifier_suggested_child_count_and_is_the_only_revision_action():
    state = initial_review_state([
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    state["reports"] = [{
        "dimension": "cross_modal_consistency",
        "scope": "set",
        "target_ids": ["C1"],
        "internal_structure_assessment": "supports_subdivision",
        "suggested_k": 3,
    }]
    split = RouterAction(
        action="split", target_ids=["C1"], n_children=3,
        decision_state=RouterDecisionState(
            identity="uncertain", structure="incompatible",
            alternative_explanation="unassessed", uncertainty="yes",
        ),
    )
    validate_router_plan(RouterPlan(actions=[split]), state, set(), terminal_only=False)
    with pytest.raises(ValueError, match="suggested_k"):
        validate_router_plan(
            RouterPlan(actions=[split.model_copy(update={"n_children": 2})]),
            state, set(), terminal_only=False,
        )
    terminal = RouterAction(
        action="accept", target_ids=["C2"],
        decision_state=RouterDecisionState(
            identity="unassessed", structure="unassessed",
            alternative_explanation="unassessed", uncertainty="yes",
        ),
    )
    with pytest.raises(ValueError, match="exactly one split or merge"):
        validate_router_plan(RouterPlan(actions=[split, terminal]), state, set(), terminal_only=False)


def test_merge_requires_verifier_pair_boundary_assessment():
    state = initial_review_state([
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    state["reports"] = [{
        "dimension": "cross_modal_consistency",
        "scope": "pair",
        "target_ids": ["C1", "C2"],
        "pair_boundary_assessment": "insufficiently_separated",
    }]
    merge = RouterAction(
        action="merge", target_ids=["C1", "C2"],
        decision_state=RouterDecisionState(
            identity="uncertain", structure="incompatible",
            alternative_explanation="unassessed", uncertainty="yes",
        ),
    )
    validate_router_plan(RouterPlan(actions=[merge]), state, set(), terminal_only=False)


def test_three_node_graph_acquires_interprets_and_routes_without_external_llm(tmp_path):
    from agents.subtype_review.graph import build_review_graph

    class Router:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                return {"actions": [], "evidence_requests": [{
                    "dimension": "biological_support", "scope": "set",
                    "target_ids": ["C1"], "question": "Is there coherent biological support?",
                }]}
            return {"actions": [{
                "action": "accept", "target_ids": ["C1"], "n_children": None,
                "decision_state": {
                    "identity": "supported", "structure": "unassessed",
                    "alternative_explanation": "unassessed", "uncertainty": "yes",
                },
                "reason": "Biological evidence supports retaining this discovery candidate.",
            }], "evidence_requests": []}

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                return {"tool_calls": [{
                    "name": "test_biology", "id": "call1",
                    "args": {"scope": "set", "target_ids": ["C1"]},
                }]}
            return {"reports": [{
                "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
                "observations": [{"metric": "effect", "finding": "A coherent signal is present."}],
                "tool_refs": [], "metric_refs": [],
            }]}

    def test_biology(**kwargs):
        return {"status": "success", "metrics": {"set": {"C1": {"effect": 1.2}}}}

    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    context = {
        "patient_states_by_id": {"P1": {}, "P2": {}},
        "data_root": str(tmp_path), "config_dir": str(tmp_path),
        "tool_registry": {"test_biology": {
            "dimension": "biological_support", "scopes": ("set",),
            "description": "test tool", "function": test_biology,
        }},
        "router_model": Router(), "verifier_model": Verifier(),
        "reviser_model": object(),
    }
    result = build_review_graph().invoke(state, context=context)
    assert result["control"]["status"] == "complete"
    assert result["reports"][0]["tool_refs"] == ["test_biology"]
    assert result["reports"][0]["metric_refs"] == [
        "tool_results.test_biology.metrics.set.C1.effect"
    ]


def test_split_invalidates_partition_reports_and_reacquires_for_new_sets(tmp_path):
    import json

    import numpy as np

    from agents.subtype_review.graph import build_review_graph

    patient_ids = [f"P{index}" for index in range(8)]
    candidate_root = tmp_path / "candidate_subtype"
    candidate_root.mkdir()
    (candidate_root / "affinity_patient_order.json").write_text(json.dumps(patient_ids))
    fused = np.full((8, 8), 0.02)
    fused[:4, :4] = 0.95
    fused[4:, 4:] = 0.95
    np.fill_diagonal(fused, 1.0)
    np.save(candidate_root / "fused_similarity.npy", fused)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        "cross_modal:\n  structural:\n    max_children: 4\n    min_child_size: 4\n    nearest_merge_neighbors: 2\n"
    )

    class Router:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                return {"actions": [], "evidence_requests": [{
                    "dimension": "cross_modal_consistency", "scope": "set",
                    "target_ids": ["C1"], "question": "Assess C1 internal structure.",
                }]}
            if self.calls == 2:
                return {"actions": [{
                    "action": "split", "target_ids": ["C1"], "n_children": 2,
                    "decision_state": {
                        "identity": "uncertain", "structure": "incompatible",
                        "alternative_explanation": "unassessed", "uncertainty": "yes",
                    },
                }]}
            if self.calls == 3:
                return {"actions": [], "evidence_requests": [
                    {
                        "dimension": "cross_modal_consistency", "scope": "set",
                        "target_ids": [set_id], "question": f"Assess {set_id} after revision.",
                    }
                    for set_id in ("C1_S1", "C1_S2")
                ]}
            return {"actions": [
                {
                    "action": "accept", "target_ids": [set_id], "n_children": None,
                    "decision_state": {
                        "identity": "unassessed", "structure": "unassessed",
                        "alternative_explanation": "unassessed", "uncertainty": "yes",
                    },
                }
                for set_id in ("C1_S1", "C1_S2")
            ]}

    class Verifier:
        def __init__(self):
            self.acquired = []

        def invoke(self, payload):
            if payload["mode"] == "acquire":
                for request in payload["evidence_requests"]:
                    for target in request["target_ids"]:
                        self.acquired.append(target)
                return {"tool_calls": [
                    {
                        "name": "structural_diagnostics", "id": f"call-{target}",
                        "args": {"scope": "set", "target_ids": [target]},
                    }
                    for request in payload["evidence_requests"]
                    for target in request["target_ids"]
                ]}
            reports = []
            for required in payload["required_reports"]:
                target = required["target_ids"][0]
                initial = target == "C1"
                reports.append({
                    **required,
                    "observations": [{"metric": "local_structure", "finding": "Measured."}],
                    "internal_structure_assessment": "supports_subdivision" if initial else "supports_retention",
                    "pair_boundary_assessment": None,
                    "suggested_k": 2 if initial else None,
                })
            return {"reports": reports}

    class Reviser:
        def invoke(self, payload):
            refs = payload["available_metric_refs"]
            return {"split_plans": [{
                "target_id": "C1", "n_children": 2,
                "structural_basis": ["fused"],
                "execution_strategy": "fused_similarity_spectral",
                "metric_refs": [next(ref for ref in refs if ref.endswith("suggested_k"))],
            }]}

    def structural_diagnostics(**kwargs):
        target = kwargs["target_ids"][0]
        suggested_k = 2 if target == "C1" else None
        return {
            "status": "success",
            "metrics": {"set": {target: {"suggested_k": suggested_k, "eigengap": 0.4}}},
        }

    context = {
        "patient_states_by_id": {patient_id: {} for patient_id in patient_ids},
        "data_root": str(tmp_path),
        "config_dir": str(config_dir),
        "tool_registry": {"structural_diagnostics": {
            "dimension": "cross_modal_consistency", "scopes": ("set",),
            "description": "Local structure diagnostics.", "function": structural_diagnostics,
        }},
        "router_model": Router(), "verifier_model": Verifier(), "reviser_model": Reviser(),
    }
    result = build_review_graph().invoke(
        initial_review_state([{"set_id": "C1", "member_ids": patient_ids}]), context=context,
    )

    assert result["control"]["status"] == "complete"
    assert {row["set_id"] for row in result["partition"]["sets"]} == {"C1_S1", "C1_S2"}
    assert {row["target_ids"][0] for row in result["reports"]} == {"C1_S1", "C1_S2"}
    assert context["verifier_model"].acquired == ["C1", "C1_S1", "C1_S2"]
    assert result["revision_result"]["operation"] == "split"
    assert result["round_evidence"][0]["target_ids"] in (["C1_S1"], ["C1_S2"])
