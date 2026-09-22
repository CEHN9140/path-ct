import json
import pytest
import numpy as np
import types
import pandas as pd

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
    assert "available_evidence_requests" in captured


def test_single_eligible_tool_still_uses_verifier_selector():
    from agents.subtype_review.graph import verifier_node

    calls = {"select": 0, "audit": 0, "tool": 0}
    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "select":
                calls["select"] += 1
                assert payload["remaining_tools"] == ["only_tool"]
                assert payload["require_tool"] is True
                return {"selected_tool": "only_tool"}
            calls["audit"] += 1
            required = payload["required_reports"][0]
            required = {key: value for key, value in required.items() if key != "evidence_guidance"}
            return {"reports": [{**required, "observations": [],
                "dimension_interpretation": "Measured.", "cross_evidence_context": "None.",
                "limitations": [], "metric_refs": []}]}

    def tool(**kwargs):
        calls["tool"] += 1
        return {"status": "success", "metrics": {"set": {"C1": {"value": 1}}}}

    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
        "focus": "transcriptomic_phenotype", "question": "Assess phenotype.",
    }]
    registry = {"only_tool": {
        "aspect": "only_tool", "dimension": "biological_support", "scopes": ("set",),
        "question_foci": {"set": ("transcriptomic_phenotype",)}, "function": tool,
    }}
    verifier_node(state, {"verifier_model": Verifier(), "tool_registry": registry,
                          "patient_states_by_id": {}, "data_root": "/tmp", "config_dir": "configs"})
    assert calls == {"select": 1, "audit": 1, "tool": 1}


def test_mandatory_partition_screen_uses_verifier_tool_selection(tmp_path):
    from agents.subtype_review.graph import verifier_node

    calls = {"select": 0, "audit": 0, "tool": 0}

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "select":
                calls["select"] += 1
                assert payload["remaining_tools"] == ["structural_diagnostics"]
                assert payload["require_tool"] is True
                assert payload["evidence_request"]["focus"] == "partition_structural_screen"
                return {"selected_tool": "structural_diagnostics"}
            calls["audit"] += 1
            required = {
                key: value for key, value in payload["required_reports"][0].items()
                if key != "evidence_guidance"
            }
            return {"reports": [{
                **required,
                "observations": [],
                "dimension_interpretation": "Partition structure was assessed.",
                "cross_evidence_context": "No additional context.",
                "limitations": [],
                "tool_refs": [],
                "metric_refs": [],
            }]}

    def structural_diagnostics(**kwargs):
        calls["tool"] += 1
        return {"status": "success", "metrics": {"partition": {
            "internal_structure": {}, "nearest_pair_affinities": [],
        }}}

    state = initial_review_state([
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "cross_modal_consistency", "scope": "partition", "target_ids": [],
        "focus": "partition_structural_screen", "question": "Screen partition structure.",
    }]
    registry = {"structural_diagnostics": {
        "aspect": "structural_diagnostics", "dimension": "cross_modal_consistency",
        "scopes": ("partition",), "question_foci": {"partition": ("partition_structural_screen",)},
        "function": structural_diagnostics,
    }}
    trace_path = tmp_path / "trace.jsonl"
    verifier_node(state, {
        "verifier_model": Verifier(), "tool_registry": registry,
        "patient_states_by_id": {}, "data_root": "/tmp", "config_dir": "configs",
        "runtime_trace_path": str(trace_path),
    })
    assert calls == {"select": 1, "audit": 1, "tool": 1}
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    selection = next(row for row in trace if row["event"] == "verifier_tool_selection")
    assert selection["selected_tool"] == "structural_diagnostics"
    assert selection["selection_source"] == "llm_tool_call"


def test_nonfirst_request_with_one_remaining_tool_can_stop():
    from agents.subtype_review.graph import verifier_node

    calls = {"select": 0, "audit": 0, "tools": []}
    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "select":
                calls["select"] += 1
                if payload["remaining_tools"] == ["second"]:
                    return {"selected_tool": None, "stop_reason": "already sufficient"}
                return {"selected_tool": "first"}
            calls["audit"] += 1
            required = payload["required_reports"][0]
            required = {key: value for key, value in required.items() if key != "evidence_guidance"}
            return {"reports": [{**required, "observations": [],
                "dimension_interpretation": "Measured.", "cross_evidence_context": "None.",
                "limitations": [], "metric_refs": []}]}

    def tool(name):
        def run(**kwargs):
            calls["tools"].append(name)
            return {"status": "success", "metrics": {"set": {"C1": {"value": 1}}}}
        return run

    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
        "focus": "transcriptomic_phenotype", "question": "Assess phenotype.",
    }]
    registry = {
        name: {"aspect": name, "dimension": "biological_support", "scopes": ("set",),
               "question_foci": {"set": ("transcriptomic_phenotype",)}, "function": tool(name)}
        for name in ("first", "second")
    }
    verifier_node(state, {"verifier_model": Verifier(), "tool_registry": registry,
                          "patient_states_by_id": {}, "data_root": "/tmp", "config_dir": "configs"})
    assert calls["select"] == 2
    assert calls["tools"] == ["first"]


def test_rna_cache_reuses_same_membership_and_invalidates_on_membership_change(tmp_path, monkeypatch):
    import sys
    import agents.subtype_review.tools as review_tools

    counts_path = tmp_path / "counts.csv"
    pd.DataFrame({"case_id": ["P1", "P2", "P3", "P4"], "G1": [10, 11, 2, 3],
                  "G2": [8, 9, 1, 2]}).to_csv(counts_path, index=False)
    hallmark = tmp_path / "hallmark.gmt"
    hallmark.write_text("TEST\tdesc\tG1\tG2\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        f"rna:\n  hallmark_gene_sets_path: {hallmark}\n  min_pathway_overlap: 2\n"
        "  cache_version: 1\n  gsea_permutations: 3\n", encoding="utf-8",
    )
    calls = {"dds": 0, "stats": 0, "gsea": 0}
    class Dds:
        def __init__(self, **kwargs): calls["dds"] += 1
        def deseq2(self): pass
    class Stats:
        def __init__(self, *args, **kwargs):
            calls["stats"] += 1
            self.statistics = np.array([2.0, -1.0])
        def run_wald_test(self): pass
    def prerank(**kwargs):
        calls["gsea"] += 1
        return types.SimpleNamespace(res2d=pd.DataFrame([{
            "Term": "TEST", "NES": 1.2, "FDR q-val": 0.1, "Lead_genes": "G1",
        }]))
    monkeypatch.setitem(sys.modules, "pydeseq2", types.ModuleType("pydeseq2"))
    monkeypatch.setitem(sys.modules, "pydeseq2.dds", types.SimpleNamespace(DeseqDataSet=Dds))
    monkeypatch.setitem(sys.modules, "pydeseq2.ds", types.SimpleNamespace(DeseqStats=Stats))
    monkeypatch.setitem(sys.modules, "gseapy", types.SimpleNamespace(prerank=prerank))
    states = {case_id: {"omics_evidence": {"rna_raw_counts_path": str(counts_path)}}
              for case_id in ("P1", "P2", "P3", "P4")}
    sets = [{"set_id": "C1", "member_ids": ["P1", "P2"]},
            {"set_id": "C2", "member_ids": ["P3", "P4"]}]
    review_tools.rna_pathway_enrichment(states, str(tmp_path), str(config_dir), sets, "set", ["C1"])
    review_tools.rna_pathway_enrichment(states, str(tmp_path), str(config_dir), sets, "set", ["C1"])
    assert calls == {"dds": 1, "stats": 1, "gsea": 1}
    changed_sets = [{"set_id": "C1", "member_ids": ["P1", "P3"]},
                    {"set_id": "C2", "member_ids": ["P2", "P4"]}]
    review_tools.rna_pathway_enrichment(states, str(tmp_path), str(config_dir), changed_sets, "set", ["C1"])
    assert calls == {"dds": 2, "stats": 2, "gsea": 2}


def test_completed_request_retry_feedback_names_existing_report(tmp_path):
    sets = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
        {"set_id": "C3", "member_ids": ["P5", "P6"]},
    ]
    state = initial_review_state(sets)
    signature = partition_signature(sets)
    state["reports"] = [
        report("ER:partition", "cross_modal_consistency", "partition", []),
        report("ER:boundary", "cross_modal_consistency", "pair", ["C1", "C2"], focus="boundary_representation"),
    ]
    state["tool_evidence"] = [
        {"tool_name": "structural_diagnostics", "scope": "partition", "target_ids": [],
         "partition_signature": signature, "metrics": {"partition": {"nearest_pair_targets": [["C1", "C2"]]}}},
        {"tool_name": "representation_concordance", "scope": "pair", "target_ids": ["C1", "C2"],
         "partition_signature": signature, "metrics": {}},
    ]
    payloads = []
    class Router:
        config = {"router_plan_validation_retries": 1}
        def invoke(self, payload):
            payloads.append(payload)
            if len(payloads) == 1:
                return {"actions": [], "evidence_requests": [{
                    "dimension": "cross_modal_consistency", "scope": "pair", "target_ids": ["C1", "C2"],
                    "focus": "boundary_representation", "question": "Repeat the pair review.",
                }]}
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
                "focus": "transcriptomic_phenotype", "question": "Assess phenotype.",
            }]}

    router_node(state, {
        "router_model": Router(), "tool_registry": TOOL_REGISTRY,
        "patient_states_by_id": {}, "data_root": str(tmp_path),
        "config_dir": "configs", "artifact_root": str(tmp_path),
    })
    feedback = payloads[1]["validation_feedback"]
    assert "error" in feedback
    assert "available_evidence_requests" not in feedback


def test_structural_pair_candidates_are_visible_and_drop_requires_one_review(tmp_path):
    sets = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
        {"set_id": "C3", "member_ids": ["P5", "P6"]},
    ]
    state = initial_review_state(sets)
    signature = partition_signature(sets)
    state["reports"] = [report("ER:partition", "cross_modal_consistency", "partition", [])]
    state["tool_evidence"] = [{
        "tool_name": "structural_diagnostics", "scope": "partition", "target_ids": [],
        "partition_signature": signature,
        "metrics": {"partition": {"nearest_pair_targets": [["C1", "C2"], ["C1", "C3"], ["C2", "C3"]],
            "nearest_pair_affinities": [
                {"target_ids": ["C1", "C3"], "mean_between_affinity": 0.2},
                {"target_ids": ["C1", "C2"], "mean_between_affinity": 0.3},
                {"target_ids": ["C2", "C3"], "mean_between_affinity": 0.1},
            ]}},
    }]
    captured = {}
    class Router:
        config = {"router_plan_validation_retries": 0}
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
                "focus": "transcriptomic_phenotype", "question": "Assess phenotype.",
            }]}
    router_node(state, {"router_model": Router(), "tool_registry": TOOL_REGISTRY,
                        "patient_states_by_id": {}, "data_root": str(tmp_path),
                        "config_dir": "configs", "artifact_root": str(tmp_path)})
    assert captured["structural_pair_candidates"] == [
        {"target_ids": ["C1", "C2"], "mean_between_affinity": 0.3},
        {"target_ids": ["C1", "C3"], "mean_between_affinity": 0.2},
        {"target_ids": ["C2", "C3"], "mean_between_affinity": 0.1},
    ]

    drops = [RouterAction(action="drop", target_ids=[target], evidence_report_refs=["ER:partition"], reason="unsupported")
             for target in ("C1", "C2", "C3")]
    with pytest.raises(ValueError, match="Drop is premature"):
        validate_router_plan(RouterPlan(actions=drops), state, set())

    state["reports"].append(report("ER:C1-C2-boundary", "cross_modal_consistency", "pair", ["C1", "C2"], focus="boundary_representation"))
    with pytest.raises(ValueError, match="Drop is premature"):
        validate_router_plan(RouterPlan(actions=drops), state, set())

    state["reports"].append(report("ER:C1-C3-boundary", "cross_modal_consistency", "pair", ["C1", "C3"], focus="boundary_representation"))
    validate_router_plan(RouterPlan(actions=drops), state, set())


def test_accept_reason_cannot_explicitly_reject_acceptance():
    state = initial_review_state([
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
    ])
    state["reports"] = [
        report("ER:partition", "cross_modal_consistency", "partition", []),
        report("ER:membership", "cross_modal_consistency", "set", ["C1"], focus="membership_representation"),
        report("ER:biology", "biological_support", "set", ["C1"]),
    ]
    state["tool_evidence"] = [{"tool_name": "structural_diagnostics", "scope": "partition",
        "target_ids": [], "partition_signature": partition_signature(state["partition"]["sets"])}]
    action = RouterAction(action="accept", target_ids=["C1"],
                          evidence_report_refs=["ER:membership", "ER:biology"],
                          reason="The evidence is conflicting, so accept is not justified.")
    with pytest.raises(ValueError, match="contradicts its own rationale"):
        validate_router_plan(RouterPlan(actions=[action]), state, set())


def test_set_between_distance_matches_nearest_competing_label():
    from agents.subtype_review.tools import current_membership_alignment
    distance = np.array([
        [0.0, 0.1, 0.2, 0.2, 0.9],
        [0.1, 0.0, 0.2, 0.2, 0.8],
        [0.2, 0.2, 0.0, 0.2, 0.7],
        [0.2, 0.2, 0.2, 0.0, 0.9],
        [0.9, 0.8, 0.7, 0.9, 0.0],
    ])
    result = current_membership_alignment(distance, np.array(["C1", "C1", "C1", "C2", "C3"]), "C1")
    assert np.isclose(result["mean_between_distance"], np.mean([0.2, 0.2, 0.2]))
