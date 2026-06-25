from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.subtype_review_v3.graph import (
    global_review_init,
    global_router_node,
    global_verifier_node,
    run_global_review_graph,
    run_review_graph,
)
from agents.subtype_review_v3.llm import load_prompt_with_protocol
from agents.subtype_review_v3.runner import run_subtype_review_v3_from_candidate
from agents.subtype_review_v3.schemas import GlobalRouterDecision, GlobalVerifierReview, RouterDecision, VerifierReview
from agents.subtype_review_v3.tools import (
    compact_tool_result,
    execute_requested_tools,
    load_available_tool_functions,
)


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.payloads = []

    def invoke(self, payload):
        self.payloads.append(dict(payload))
        output = self.outputs.pop(0)
        return output(payload) if callable(output) else output


class FailingModel:
    def __init__(self, message: str):
        self.message = message

    def invoke(self, payload):
        raise ValueError(self.message)


def verifier_output(
    *,
    accept_ready=False,
    confidence_level="low",
    evidence_gaps=None,
    reason_codes=None,
    metric_refs=None,
):
    return {
        "accept_ready": accept_ready,
        "verification_vector": {"set_reliability": {"member_count": 2}},
        "evidence_gaps": list(evidence_gaps or []),
        "confidence_level": confidence_level,
        "reason_codes": list(reason_codes or ["needs_more_evidence"]),
        "metric_refs": list(metric_refs or ["cluster.member_count"]),
        "reasoning_summary": "fake verifier",
    }


def router_output(
    action,
    *,
    requested_tools=None,
    target_blocks=None,
    reason_codes=None,
    metric_refs=None,
):
    return {
        "action": action,
        "requested_tools": list(requested_tools or []),
        "target_blocks": list(target_blocks or []),
        "reason_codes": list(reason_codes or [f"{action}_reason"]),
        "metric_refs": list(metric_refs or ["cluster.member_count"]),
        "continue_review_reason": "need more evidence" if action == "continue_review" else "",
        "revision_signal": {"suggested_member_ids": ["SHOULD_NOT_BE_USED"]},
        "reasoning_summary": "fake router",
    }


def global_verifier_output(*, ready_for_revision=False, evidence_gaps=None, set_reviews=None):
    return {
        "ready_for_revision": ready_for_revision,
        "set_reviews": list(
            set_reviews
            or [
                {
                    "cluster_id": "C1",
                    "accept_ready": False,
                    "confidence_level": "low",
                    "verification_vector": {},
                    "evidence_gaps": list(evidence_gaps or ["biological_support"]),
                    "reason_codes": ["needs_more_evidence"],
                    "metric_refs": ["cluster.C1.member_count"],
                    "reasoning_summary": "fake set review",
                },
                {
                    "cluster_id": "C2",
                    "accept_ready": False,
                    "confidence_level": "low",
                    "verification_vector": {},
                    "evidence_gaps": list(evidence_gaps or ["biological_support"]),
                    "reason_codes": ["needs_more_evidence"],
                    "metric_refs": ["cluster.C2.member_count"],
                    "reasoning_summary": "fake set review",
                },
            ]
        ),
        "global_evidence_gaps": list(evidence_gaps or ["biological_support"]),
        "cross_set_findings": [],
        "reasoning_summary": "fake global verifier",
    }


def global_router_output(action, *, requested_tools=None, revision_plan=None):
    return {
        "action": action,
        "requested_tools": list(requested_tools or []),
        "target_sets": ["C1", "C2"],
        "target_blocks": ["biological_support"] if action == "continue_review" else [],
        "reason_codes": [f"{action}_reason"],
        "metric_refs": ["cluster.C1.member_count"],
        "continue_review_reason": "need more global evidence" if action == "continue_review" else "",
        "revision_plan": dict(revision_plan or {"accept": ["C1"], "drop": ["C2"], "merge": [], "split": []}),
        "reasoning_summary": "fake global router",
    }


def fake_tool(cluster_state, patient_states_by_id, output_root, config_dir="", all_cluster_states=None):
    return {
        "tool_name": "tool_fake_biology",
        "status": "success",
        "cluster_id": str(cluster_state.get("cluster_id", "")),
        "results": {
            "metrics": {"rna_pathway_enrichment": [{"pathway": "HALLMARK_TEST", "q_value": 0.01}]},
            "warnings": [],
            "missing_reason": "",
        },
        "artifacts": {"tables": {"rna_pathway_enrichment": "rna.csv"}},
        "errors": [],
    }


def fake_global_tool(cluster_state, patient_states_by_id, output_root, config_dir="", all_cluster_states=None):
    return {
        "tool_name": "tool_fake_biology",
        "status": "success",
        "cluster_id": str(cluster_state.get("cluster_id", "")),
        "results": {
            "metrics": {
                "set_support": {
                    str(item.get("cluster_id")): len(list(item.get("member_ids", []) or []))
                    for item in list(all_cluster_states or [])
                }
            },
            "warnings": [],
            "missing_reason": "",
        },
        "artifacts": {},
        "errors": [],
    }


def fake_mutation_tool(cluster_state, patient_states_by_id, output_root, config_dir="", all_cluster_states=None):
    rows = [
        {
            "candidate_set_id": "C1",
            "gene": f"GENE{i}",
            "delta_frequency": i / 100,
            "odds_ratio": 1 + i,
            "p_value": i / 1000,
            "q_value": i / 100,
        }
        for i in range(1, 31)
    ]
    rows.append(
        {
            "candidate_set_id": "C1",
            "gene": "VHL",
            "delta_frequency": 0.5,
            "odds_ratio": 8,
            "p_value": 0.0001,
            "q_value": 0.001,
        }
    )
    return {
        "tool_name": "tool_mutation_enrichment",
        "status": "success",
        "cluster_id": str(cluster_state.get("cluster_id", "")),
        "results": {
            "metrics": {
                "wxs_gene_enrichment": rows,
                "wxs_pathway_enrichment": [
                    {
                        "candidate_set_id": "C1",
                        "pathway": f"P{i}",
                        "delta_frequency": i / 100,
                        "odds_ratio": 2 + i,
                        "p_value": i / 1000,
                        "q_value": i / 100,
                    }
                    for i in range(1, 31)
                ],
            },
            "warnings": [],
            "missing_reason": "",
        },
        "artifacts": {},
        "errors": [],
    }


class SubtypeReviewV3Test(unittest.TestCase):
    def test_global_review_uses_all_sets_and_router_requested_tools(self) -> None:
        verifier = FakeModel(
            [
                global_verifier_output(ready_for_revision=False),
                global_verifier_output(
                    ready_for_revision=True,
                    set_reviews=[
                        {
                            "cluster_id": "C1",
                            "accept_ready": True,
                            "confidence_level": "high",
                            "verification_vector": {"biological_support": "present"},
                            "evidence_gaps": [],
                            "reason_codes": ["accepted"],
                            "metric_refs": ["tool_results.tool_fake_biology.metrics.set_support.C1"],
                            "reasoning_summary": "C1 accepted",
                        },
                        {
                            "cluster_id": "C2",
                            "accept_ready": False,
                            "confidence_level": "low",
                            "verification_vector": {"biological_support": "weak"},
                            "evidence_gaps": ["biological_support"],
                            "reason_codes": ["weak_biological_support"],
                            "metric_refs": ["tool_results.tool_fake_biology.metrics.set_support.C2"],
                            "reasoning_summary": "C2 dropped",
                        },
                    ],
                ),
            ]
        )
        router = FakeModel(
            [
                global_router_output("continue_review", requested_tools=["tool_fake_biology"]),
                global_router_output("revise", revision_plan={"accept": ["C1"], "drop": ["C2"], "merge": [], "split": []}),
            ]
        )

        state = run_global_review_graph(
            {
                "candidate_sets": [
                    {"cluster_id": "C1", "member_ids": ["A", "B"]},
                    {"cluster_id": "C2", "member_ids": ["C"]},
                ],
                "patient_states_by_id": {"A": {}, "B": {}, "C": {}},
                "tool_definitions": {"tool_fake_biology": {"evidence_blocks": ["biological_support"]}},
                "tool_functions": {"tool_fake_biology": fake_global_tool},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 3, "max_tool_calls": 1},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(len(verifier.payloads[0]["candidate_sets"]), 2)
        self.assertEqual(router.payloads[0]["set_reviews"][0]["cluster_id"], "C1")
        self.assertEqual(state["executed_tools"], ["tool_fake_biology"])
        self.assertEqual(state["final_sets"][0]["cluster_id"], "C1")
        self.assertEqual(state["final_sets"][0]["member_ids"], ["A", "B"])
        self.assertEqual(state["final_sets"][0]["status"], "accept")
        self.assertEqual(state["dropped_set_ids"], ["C2"])
        self.assertEqual(
            [item["node"] for item in state["round_trace"]],
            ["global_verifier", "global_router", "tool_executor", "global_verifier", "global_router", "global_reviser"],
        )

    def test_global_router_limits_tools_per_round(self) -> None:
        verifier = FakeModel([global_verifier_output(ready_for_revision=False)])
        router = FakeModel(
            [
                global_router_output(
                    "continue_review",
                    requested_tools=["tool_a", "tool_b", "tool_c"],
                )
            ]
        )

        state = run_global_review_graph(
            {
                "candidate_sets": [{"cluster_id": "C1", "member_ids": ["A"]}, {"cluster_id": "C2", "member_ids": ["B"]}],
                "patient_states_by_id": {"A": {}, "B": {}},
                "tool_definitions": {
                    "tool_a": {"evidence_blocks": ["set_reliability"]},
                    "tool_b": {"evidence_blocks": ["clinical_context"]},
                    "tool_c": {"evidence_blocks": ["biological_support"]},
                },
                "tool_functions": {"tool_a": fake_global_tool, "tool_b": fake_global_tool, "tool_c": fake_global_tool},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 1, "max_tool_calls": 3, "max_tools_per_round": 2},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(state["executed_tools"], ["tool_fake_biology", "tool_fake_biology"])
        self.assertEqual(state["requested_tools"], ["tool_a", "tool_b"])
        self.assertEqual(state["router_decision"]["rejected_tools"], ["tool_c"])
        self.assertEqual(router.payloads[0]["max_tools_this_round"], 2)

    def test_global_router_only_sees_unexecuted_tool_catalog(self) -> None:
        router = FakeModel([global_router_output("continue_review", requested_tools=["tool_b"])])

        runtime = {
            "candidate_sets": [{"cluster_id": "C1", "member_ids": ["A"]}],
            "patient_states_by_id": {"A": {}},
            "tool_definitions": {
                "tool_a": {"evidence_blocks": ["set_reliability"]},
                "tool_b": {"evidence_blocks": ["clinical_context"]},
            },
            "tool_functions": {"tool_a": fake_global_tool, "tool_b": fake_global_tool},
            "output_root": tempfile.mkdtemp(dir="/tmp"),
            "budget": {"max_tool_calls": 3, "max_tools_per_round": 2},
            "router_model": router,
        }
        state = global_review_init(runtime)
        state["executed_tools"] = ["tool_a"]
        state["verifier_review"] = global_verifier_output(
            ready_for_revision=False,
            set_reviews=[
                {
                    "cluster_id": "C1",
                    "accept_ready": False,
                    "confidence_level": "low",
                    "verification_vector": {},
                    "evidence_gaps": ["clinical_context"],
                    "reason_codes": ["needs_more_evidence"],
                    "metric_refs": ["cluster.C1.member_count"],
                    "reasoning_summary": "fake set review",
                }
            ],
        )
        state = global_router_node(state, runtime)

        self.assertEqual(list(router.payloads[0]["tool_catalog"].keys()), ["tool_b"])
        self.assertEqual(state["requested_tools"], ["tool_b"])

    def test_global_tool_executed_on_final_round_gets_followup_verifier(self) -> None:
        verifier = FakeModel(
            [
                global_verifier_output(ready_for_revision=False),
                global_verifier_output(
                    ready_for_revision=True,
                    set_reviews=[
                        {
                            "cluster_id": "C1",
                            "accept_ready": True,
                            "confidence_level": "high",
                            "verification_vector": {},
                            "evidence_gaps": [],
                            "reason_codes": ["accepted"],
                            "metric_refs": ["tool_results.tool_fake_biology.metrics.set_support.C1"],
                            "reasoning_summary": "accepted",
                        }
                    ],
                ),
            ]
        )
        router = FakeModel([global_router_output("continue_review", requested_tools=["tool_fake_biology"])])

        state = run_global_review_graph(
            {
                "candidate_sets": [{"cluster_id": "C1", "member_ids": ["A"]}],
                "patient_states_by_id": {"A": {}},
                "tool_definitions": {"tool_fake_biology": {"evidence_blocks": ["biological_support"]}},
                "tool_functions": {"tool_fake_biology": fake_global_tool},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 1, "max_tool_calls": 1},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(len(verifier.payloads), 2)
        self.assertEqual([item["node"] for item in state["round_trace"]], ["global_verifier", "global_router", "tool_executor", "global_verifier", "budget"])

    def test_verifier_accept_ready_high_confidence_reports_without_router(self) -> None:
        verifier = FakeModel([verifier_output(accept_ready=True, confidence_level="high", reason_codes=["accepted"])])
        router = FakeModel([])

        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A", "B"]},
                "patient_states_by_id": {"A": {}, "B": {}},
                "tool_definitions": {},
                "tool_functions": {},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 3, "max_tool_calls": 2},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(state["final_action"], "accept")
        self.assertEqual(len(router.payloads), 0)
        self.assertEqual([item["node"] for item in state["round_trace"]], ["verifier"])

    def test_non_accept_verifier_enters_router_and_drop(self) -> None:
        verifier = FakeModel([verifier_output(evidence_gaps=["biological_support"])])
        router = FakeModel([router_output("drop", reason_codes=["insufficient_biology"])])

        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A"]},
                "patient_states_by_id": {"A": {}},
                "tool_definitions": {},
                "tool_functions": {},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 3, "max_tool_calls": 2},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(state["final_action"], "drop")
        self.assertEqual(state["reason_codes"], ["insufficient_biology"])
        self.assertIn("router", [item["node"] for item in state["round_trace"]])

    def test_router_continue_review_executes_requested_tool_then_verifies_again(self) -> None:
        verifier = FakeModel(
            [
                verifier_output(evidence_gaps=["biological_support"]),
                verifier_output(accept_ready=True, confidence_level="high", reason_codes=["accepted_after_tool"]),
            ]
        )
        router = FakeModel(
            [
                router_output(
                    "continue_review",
                    requested_tools=["tool_fake_biology"],
                    target_blocks=["biological_support"],
                )
            ]
        )

        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A", "B"]},
                "patient_states_by_id": {"A": {}, "B": {}},
                "tool_definitions": {"tool_fake_biology": {"evidence_blocks": ["biological_support"]}},
                "tool_functions": {"tool_fake_biology": fake_tool},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 3, "max_tool_calls": 2},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(state["final_action"], "accept")
        self.assertEqual(state["executed_tools"], ["tool_fake_biology"])
        self.assertIn("rna_pathway_enrichment", state["tool_results"][0]["metrics"])
        self.assertEqual([item["node"] for item in state["round_trace"]], ["verifier", "router", "tool_executor", "verifier"])
        self.assertNotIn("evidence_pack", verifier.payloads[-1])

    def test_tool_executed_on_final_round_gets_followup_verifier(self) -> None:
        verifier = FakeModel(
            [
                verifier_output(evidence_gaps=["biological_support"]),
                verifier_output(accept_ready=True, confidence_level="high", reason_codes=["accepted_after_final_tool"]),
            ]
        )
        router = FakeModel(
            [
                router_output(
                    "continue_review",
                    requested_tools=["tool_fake_biology"],
                    target_blocks=["biological_support"],
                )
            ]
        )

        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A", "B"]},
                "patient_states_by_id": {"A": {}, "B": {}},
                "tool_definitions": {"tool_fake_biology": {"evidence_blocks": ["biological_support"]}},
                "tool_functions": {"tool_fake_biology": fake_tool},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 1, "max_tool_calls": 1},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(state["final_action"], "accept")
        self.assertEqual(len(verifier.payloads), 2)
        self.assertEqual(state["round_index"], 2)
        self.assertEqual([item["node"] for item in state["round_trace"]], ["verifier", "router", "tool_executor", "verifier"])

    def test_router_reason_codes_are_normalized_to_short_codes(self) -> None:
        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A"]},
                "patient_states_by_id": {"A": {}},
                "tool_definitions": {},
                "tool_functions": {},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 3, "max_tool_calls": 1},
                "verifier_model": FakeModel([verifier_output(evidence_gaps=["confounder_exclusion"])]),
                "router_model": FakeModel(
                    [
                        router_output(
                            "drop",
                            reason_codes=[
                                "confounder_exclusion: Strong confounding signal from CT manufacturer (Cramér's V = 0.54).",
                                "known_label_echo: Significant association with clinical stage.",
                            ],
                        )
                    ]
                ),
            }
        )

        self.assertEqual(state["reason_codes"], ["strong_ct_manufacturer_confounding", "stage_label_echo"])
        self.assertEqual(
            state["router_decision"]["reason_codes"],
            ["strong_ct_manufacturer_confounding", "stage_label_echo"],
        )

    def test_router_continue_review_rejects_disabled_duplicate_or_missing_tools(self) -> None:
        verifier = FakeModel([verifier_output(evidence_gaps=["biological_support"])])
        router = FakeModel(
            [
                router_output(
                    "continue_review",
                    requested_tools=["tool_fake_biology", "tool_fake_biology", "tool_disabled"],
                    target_blocks=["biological_support"],
                )
            ]
        )

        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A"]},
                "patient_states_by_id": {"A": {}},
                "tool_definitions": {"tool_fake_biology": {"evidence_blocks": ["biological_support"]}},
                "tool_functions": {"tool_fake_biology": fake_tool},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 1, "max_tool_calls": 2},
                "verifier_model": verifier,
                "router_model": router,
            }
        )

        self.assertEqual(state["executed_tools"], ["tool_fake_biology"])
        self.assertEqual(state["requested_tools"], ["tool_fake_biology"])
        self.assertIn("tool_disabled", state["router_decision"]["rejected_tools"])

    def test_budget_exhaustion_after_non_accept_drops_as_insufficient_evidence(self) -> None:
        verifier = FakeModel([verifier_output(evidence_gaps=["clinical_context"])])

        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A"]},
                "patient_states_by_id": {"A": {}},
                "tool_definitions": {},
                "tool_functions": {},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 1, "max_tool_calls": 0},
                "verifier_model": verifier,
                "router_model": FakeModel([]),
            }
        )

        self.assertEqual(state["status"], "drop")
        self.assertEqual(state["final_action"], "drop")
        self.assertEqual(state["reason_codes"], ["insufficient_evidence"])
        self.assertNotEqual(state["status"], "review_unavailable")

    def test_router_failure_is_review_unavailable_not_insufficient_evidence(self) -> None:
        state = run_review_graph(
            {
                "cluster": {"cluster_id": "C1", "member_ids": ["A"]},
                "patient_states_by_id": {"A": {}},
                "tool_definitions": {},
                "tool_functions": {},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 3, "max_tool_calls": 2},
                "verifier_model": FakeModel([verifier_output(evidence_gaps=["clinical_context"])]),
                "router_model": FailingModel("bad router payload"),
            }
        )

        self.assertEqual(state["status"], "review_unavailable")
        self.assertEqual(state["reason_codes"], ["router_failure"])
        self.assertEqual(state["system_error"]["node"], "router")

    def test_split_revision_uses_deterministic_plan_not_llm_member_list(self) -> None:
        state = run_review_graph(
            {
                "cluster": {
                    "cluster_id": "C1",
                    "member_ids": ["A", "B", "C"],
                    "split_plan": {"groups": [["A"], ["B", "C"]]},
                },
                "patient_states_by_id": {"A": {}, "B": {}, "C": {}},
                "tool_definitions": {},
                "tool_functions": {},
                "output_root": tempfile.mkdtemp(dir="/tmp"),
                "budget": {"max_rounds": 3, "max_tool_calls": 2},
                "verifier_model": FakeModel([verifier_output(evidence_gaps=["set_reliability"])]),
                "router_model": FakeModel([router_output("split", reason_codes=["internal_heterogeneity_supported"])]),
            }
        )

        self.assertEqual(state["final_action"], "split")
        self.assertEqual([child["member_ids"] for child in state["generated_sets"]], [["A"], ["B", "C"]])
        self.assertNotIn("SHOULD_NOT_BE_USED", json.dumps(state["generated_sets"]))

    def test_tool_registry_keeps_available_tools_when_some_imports_fail(self) -> None:
        tool_definitions = {
            "tool_fake_biology": {
                "module": "tests.test_subtype_review_v3",
                "function": "fake_tool",
                "evidence_blocks": ["biological_support"],
            },
            "tool_missing": {
                "module": "missing.module",
                "function": "missing_tool",
                "evidence_blocks": ["clinical_context"],
            },
        }

        functions, errors = load_available_tool_functions(tool_definitions)

        self.assertIn("tool_fake_biology", functions)
        self.assertNotIn("tool_missing", functions)
        self.assertIn("tool_missing", errors)
        self.assertIn("ModuleNotFoundError", errors["tool_missing"])

    def test_tool_executor_returns_compact_tool_results(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            results = execute_requested_tools(
                ["tool_fake_biology"],
                {"tool_fake_biology": fake_tool},
                {"cluster_id": "C1", "member_ids": ["A", "B"]},
                {"A": {}, "B": {}},
                temp_dir,
                "",
                [],
            )

        self.assertEqual(results[0]["tool_name"], "tool_fake_biology")
        self.assertIn("rna_pathway_enrichment", results[0]["metrics"])
        self.assertIn("tool_results.tool_fake_biology.metrics.rna_pathway_enrichment", results[0]["metric_refs"])

    def test_enrichment_tables_are_compacted_before_llm_review(self) -> None:
        result = compact_tool_result(fake_mutation_tool({"cluster_id": "C1"}, {}, "/tmp"))

        self.assertNotIn("wxs_gene_enrichment", result["metrics"])
        self.assertNotIn("wxs_pathway_enrichment", result["metrics"])
        self.assertEqual(result["metrics"]["gene_enrichment_summary"]["tested_count"], 31)
        self.assertEqual(result["metrics"]["gene_enrichment_summary"]["significant_count_q05"], 6)
        self.assertLessEqual(len(result["metrics"]["top_genes_by_q"]), 10)
        self.assertLessEqual(len(result["metrics"]["top_genes_by_effect"]), 10)
        self.assertEqual(result["metrics"]["known_driver_hits"][0]["gene"], "VHL")

    def test_global_enrichment_compaction_keeps_per_set_summaries(self) -> None:
        result = compact_tool_result(fake_mutation_tool({"cluster_id": "GLOBAL"}, {}, "/tmp"))

        self.assertIn("per_set_gene_enrichment", result["metrics"])
        self.assertIn("C1", result["metrics"]["per_set_gene_enrichment"])
        self.assertNotIn("wxs_gene_enrichment", result["metrics"])

    def test_prompts_include_verification_protocol_library(self) -> None:
        prompt_dir = Path("agents/subtype_review_v3/prompts")

        verifier_prompt = load_prompt_with_protocol(prompt_dir, "verifier.md")
        router_prompt = load_prompt_with_protocol(prompt_dir, "router.md")

        self.assertIn("Verification Protocol Library", verifier_prompt)
        self.assertIn("Acceptance Gate", verifier_prompt)
        self.assertIn("Router Rules", router_prompt)

    def test_runner_loads_checkpoint_and_candidate_clusters(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            output_root = root / "review"
            checkpoint_dir = source_root / "storage" / "pipeline_checkpoints"
            candidate_dir = source_root / "candidate_subtype"
            checkpoint_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            (checkpoint_dir / "evidence_ready.json").write_text(json.dumps({"patient_states": [{"case_id": "A"}]}), encoding="utf-8")
            (candidate_dir / "candidate_clusters.json").write_text(json.dumps([{"cluster_id": "C1", "member_ids": ["A"]}]), encoding="utf-8")

            summary = run_subtype_review_v3_from_candidate(
                str(source_root),
                str(output_root),
                str(root / "configs"),
                verifier_model=FakeModel(
                    [
                        global_verifier_output(
                            ready_for_revision=True,
                            set_reviews=[
                                {
                                    "cluster_id": "C1",
                                    "accept_ready": True,
                                    "confidence_level": "high",
                                    "verification_vector": {},
                                    "evidence_gaps": [],
                                    "reason_codes": ["accepted"],
                                    "metric_refs": ["cluster.C1.member_count"],
                                    "reasoning_summary": "accepted",
                                }
                            ],
                        )
                    ]
                ),
                router_model=FakeModel([global_router_output("revise", revision_plan={"accept": ["C1"], "drop": [], "merge": [], "split": []})]),
            )

            self.assertEqual(summary["stage"], "subtype_review_v3")
            self.assertEqual(summary["set_count"], 1)
            self.assertTrue((output_root / "subtype_review_v3" / "final_review_summary.json").exists())

    def test_runner_uses_global_revision_plan_for_all_candidate_sets(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            output_root = root / "output_kirc_v3"
            checkpoint_dir = source_root / "storage" / "pipeline_checkpoints"
            candidate_dir = source_root / "candidate_subtype"
            checkpoint_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            (checkpoint_dir / "evidence_ready.json").write_text(
                json.dumps({"patient_states": [{"case_id": "A"}, {"case_id": "B"}, {"case_id": "C"}]}),
                encoding="utf-8",
            )
            (candidate_dir / "candidate_clusters.json").write_text(
                json.dumps(
                    [
                        {"cluster_id": "C1", "member_ids": ["A", "B"]},
                        {"cluster_id": "C2", "member_ids": ["C"]},
                        {"cluster_id": "C3", "member_ids": ["A"]},
                    ]
                ),
                encoding="utf-8",
            )
            stale_path = output_root / "subtype_review_v3" / "stale.txt"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("old", encoding="utf-8")

            summary = run_subtype_review_v3_from_candidate(
                str(source_root),
                str(output_root),
                str(root / "configs"),
                verifier_model=FakeModel(
                    [
                        global_verifier_output(
                            ready_for_revision=True,
                            set_reviews=[
                                {
                                    "cluster_id": "C1",
                                    "accept_ready": True,
                                    "confidence_level": "high",
                                    "verification_vector": {},
                                    "evidence_gaps": [],
                                    "reason_codes": ["accepted"],
                                    "metric_refs": ["cluster.C1.member_count"],
                                    "reasoning_summary": "accepted",
                                },
                                {
                                    "cluster_id": "C2",
                                    "accept_ready": False,
                                    "confidence_level": "low",
                                    "verification_vector": {},
                                    "evidence_gaps": ["biological_support"],
                                    "reason_codes": ["weak_biological_support"],
                                    "metric_refs": ["cluster.C2.member_count"],
                                    "reasoning_summary": "dropped",
                                },
                                {
                                    "cluster_id": "C3",
                                    "accept_ready": True,
                                    "confidence_level": "high",
                                    "verification_vector": {},
                                    "evidence_gaps": [],
                                    "reason_codes": ["accepted"],
                                    "metric_refs": ["cluster.C3.member_count"],
                                    "reasoning_summary": "accepted",
                                },
                            ],
                        )
                    ]
                ),
                router_model=FakeModel(
                    [
                        global_router_output("revise", revision_plan={"accept": ["C1", "C3"], "drop": ["C2"], "merge": [], "split": []}),
                    ]
                ),
            )

            cluster_ids = [item["cluster_id"] for item in summary["sets"]]
            self.assertFalse(stale_path.exists())
            self.assertEqual(cluster_ids, ["C1", "C2", "C3"])
            self.assertEqual([item["final_action"] for item in summary["sets"]], ["accept", "drop", "accept"])


if __name__ == "__main__":
    unittest.main()
