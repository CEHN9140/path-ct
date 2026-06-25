from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.cluster_flow import (
    REVIEW_UNAVAILABLE_STATUS,
    agentic_router_decision,
    build_agentic_evidence_matrix,
    build_evidence_catalog,
    build_verifier_decision,
    build_pipeline_output,
    decision_consistency_check,
    generate_cluster_report,
    llm_evidence_audit,
    llm_action_decision,
    revision_engine,
    select_pac_stability_k,
    tools_for_evidence_blocks,
)
from agents.subtype_review.router_planner import router_planner_node
from agents.subtype_review.verifier import verifier_node
from utils.subtype_review_runtime import attach_global_figures_to_reports
from utils.llm_utils import call_llm_json


class RetryClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {"content": "not-json", "tool_calls": []}
        return {"content": '{"ok": true}', "tool_calls": []}


class TokenBudgetClient:
    def __init__(self) -> None:
        self.max_tokens = 2048


class SubtypeReviewAgenticContractTest(unittest.TestCase):
    def test_pac_stability_selects_k5_for_current_candidate_fixture(self) -> None:
        records = [
            {"n_clusters": 2, "pac": 0.425968, "delta_area": 0.471901, "cluster_sizes": [31, 59], "consensus_silhouette": 0.78},
            {"n_clusters": 3, "pac": 0.389763, "delta_area": 0.198921, "cluster_sizes": [29, 29, 32], "consensus_silhouette": 0.75},
            {"n_clusters": 4, "pac": 0.394507, "delta_area": 0.070563, "cluster_sizes": [11, 21, 25, 33], "consensus_silhouette": 0.72},
            {"n_clusters": 5, "pac": 0.349313, "delta_area": 0.051072, "cluster_sizes": [10, 13, 19, 24, 24], "consensus_silhouette": 0.74},
            {"n_clusters": 6, "pac": 0.317853, "delta_area": 0.033589, "cluster_sizes": [9, 10, 12, 13, 20, 26], "consensus_silhouette": 0.80},
            {"n_clusters": 7, "pac": 0.283396, "delta_area": 0.020664, "cluster_sizes": [6, 7, 7, 11, 16, 19, 24], "consensus_silhouette": 0.81},
            {"n_clusters": 8, "pac": 0.249688, "delta_area": 0.019620, "cluster_sizes": [1, 6, 7, 9, 10, 11, 21, 25], "consensus_silhouette": 0.83},
        ]

        selected = select_pac_stability_k(
            records,
            min_cluster_size=10,
            pac_tie_tolerance=0.02,
        )

        self.assertEqual(selected["n_clusters"], 5)
        self.assertEqual(selected["selection_metric"], "pac_stability")
        self.assertIn("minimum cluster size", selected["selected_k_reason"])

    def test_tools_for_evidence_blocks_uses_tool_mapping_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "allowed_tools:",
                        "  - tool_stability_check",
                        "  - tool_mutation_enrichment",
                    ]
                ),
                encoding="utf-8",
            )
            (config_dir / "subtype_review_tools.yaml").write_text(
                "\n".join(
                    [
                        "tools:",
                        "  tool_stability_check:",
                        "    module: tools.tool_stability_check",
                        "    function: tool_stability_check",
                        "    evidence_blocks: [set_reliability]",
                        "  tool_mutation_enrichment:",
                        "    module: tools.tool_mutation_enrichment",
                        "    function: tool_mutation_enrichment",
                        "    evidence_blocks: [biological_support]",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                tools_for_evidence_blocks(["biological_support"], str(config_dir)),
                [{"tool_name": "tool_mutation_enrichment"}],
            )

    def test_confounder_metrics_are_not_filtered_or_summarized_for_review(self) -> None:
        metrics = {
            "confounder_global_association": {
                "ct_manufacturer": {"field": "ct_manufacturer", "q_value": 0.01}
            },
            "confounder_set_association": {
                "C0001": {
                    "age_at_index": {
                        "field": "age_at_index",
                        "standardized_mean_difference": 1.2,
                    }
                }
            },
        }
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "validation_results": {
                "tool_confound_test": {"results": {"metrics": metrics}}
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})

        self.assertEqual(matrix["confounder_exclusion"]["metrics"], metrics)

    def test_invalid_llm_json_is_retried_once(self) -> None:
        client = RetryClient()

        payload = call_llm_json(
            [{"role": "user", "content": "Return JSON."}],
            llm_client=client,
        )

        self.assertEqual(client.calls, 2)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["_retry_used"])

    def test_evidence_matrix_has_six_strict_blocks(self) -> None:
        cluster = {"cluster_id": "C0001", "member_ids": ["A", "B"]}

        matrix = build_agentic_evidence_matrix(cluster, {})

        self.assertEqual(
            set(matrix),
            {
                "set_reliability",
                "biological_support",
                "multimodal_support",
                "clinical_context",
                "known_label_echo",
                "confounder_exclusion",
            },
        )
        for block in matrix.values():
            self.assertEqual(
                set(block),
                {"metrics", "llm_assessment", "figures", "source_paths"},
            )
            self.assertIn("assessment", block["llm_assessment"])
            self.assertIn("rationale", block["llm_assessment"])
            self.assertIn("metric_refs", block["llm_assessment"])
        self.assertEqual(
            set(matrix["set_reliability"]["metrics"]),
            {
                "set_reliability_global_consensus",
                "set_reliability_set_consensus",
            },
        )
        self.assertEqual(matrix["set_reliability"]["figures"], {})

    def test_router_forces_drop_after_third_insufficient_round(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "review_round": 3,
            "verifier_decision": {
                "final_action_ready": False,
                "blocks_to_update": ["biological_support"],
            },
            "budget_state": {},
        }

        decision = agentic_router_decision(cluster, {"budget": {"max_rounds": 3}})

        self.assertEqual(decision["route_action"], "send_to_revision")
        self.assertEqual(decision["send_to_revision_action"], "drop")
        self.assertEqual(decision["forced_drop_reason"], "insufficient_evidence")

    def test_continue_review_without_action_is_not_review_unavailable_at_max_round(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "review_round": 3,
            "verifier_decision": {
                "decision_state": "continue_review",
                "confidence_level": "moderate",
                "final_action_ready": False,
                "blocks_to_update": ["biological_support"],
            },
            "llm_audit": {
                "decision_state": "continue_review",
                "confidence_level": "moderate",
                "recommended_action": "",
                "tools_to_call": [],
                "blocks_to_update": ["biological_support"],
                "continue_review_reason": "Biological evidence needs stronger review.",
            },
            "budget_state": {},
        }

        decision = agentic_router_decision(cluster, {"budget": {"max_rounds": 3}})

        self.assertEqual(decision["send_to_revision_action"], "drop")
        self.assertEqual(decision["forced_drop_reason"], "insufficient_evidence")
        self.assertNotEqual(decision["send_to_revision_action"], REVIEW_UNAVAILABLE_STATUS)

    def test_router_replans_without_auto_requesting_full_evidence_when_planner_has_no_tools(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "review_round": 2,
            "verifier_decision": {
                "final_action_ready": False,
                "blocks_to_update": [],
            },
            "budget_state": {},
        }

        decision = agentic_router_decision(cluster, {"budget": {"max_rounds": 3}})

        self.assertEqual(decision["route_action"], "continue_review")
        self.assertEqual(decision.get("tools_to_call", []), [])
        self.assertEqual(decision["blocks_to_update"], [])

    def test_router_node_replans_insufficient_when_planner_has_no_tools_before_max_round(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review_budget.yaml").write_text(
                "budget:\n  max_rounds: 3\n  max_failures: 2\n",
                encoding="utf-8",
            )
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: agents/subtype_review/prompts",
                        "allowed_tools:",
                        "  - tool_stability_check",
                        "  - tool_mutation_enrichment",
                        "  - tool_pathway_enrichment",
                        "  - tool_multimodal_consistency_check",
                        "  - tool_survival_analysis",
                        "  - tool_known_label_echo_test",
                        "  - tool_confound_test",
                    ]
                ),
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "review_round": 2,
                "verifier_decision": {
                    "final_action_ready": False,
                    "blocks_to_update": [],
                },
                "llm_audit": {"recommended_action": "", "tools_to_call": []},
                "budget_state": {},
            }

            updated = router_planner_node(
                {
                    "inventory": {
                        "cluster_state": cluster,
                        "config_dir": str(config_dir),
                        "output_root": str(Path(temp_dir)),
                    }
                }
            )["inventory"]["cluster_state"]

            self.assertEqual(updated["next_action"], "continue_review")
            self.assertEqual(updated["final_action"], "")
            self.assertEqual(updated.get("drop_reason", ""), "")
            self.assertEqual(updated["recommended_tools"], [])

    def test_router_node_replans_insufficient_for_decider_blocks_without_tools_before_max_round(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review_budget.yaml").write_text(
                "budget:\n  max_rounds: 3\n  max_failures: 2\n  max_tool_calls: 8\n",
                encoding="utf-8",
            )
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: agents/subtype_review/prompts",
                        "allowed_tools:",
                        "  - tool_stability_check",
                        "  - tool_mutation_enrichment",
                    ]
                ),
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "review_round": 1,
                "verifier_decision": {
                    "final_action_ready": False,
                    "blocks_to_update": ["biological_support"],
                },
                "llm_audit": {
                    "decision_state": "continue_review",
                    "tools_to_call": [],
                    "blocks_to_update": ["biological_support"],
                    "continue_review_reason": "Biology could still change the action.",
                },
                "budget_state": {},
            }

            updated = router_planner_node(
                {
                    "inventory": {
                        "cluster_state": cluster,
                        "config_dir": str(config_dir),
                        "output_root": str(Path(temp_dir)),
                    }
                }
            )["inventory"]["cluster_state"]

            self.assertEqual(updated["next_action"], "continue_review")
            self.assertEqual(updated.get("drop_reason", ""), "")
            self.assertEqual(updated["recommended_tools"], [])
            self.assertEqual(
                updated["router_decision"]["blocks_to_update"],
                ["biological_support"],
            )
            self.assertNotIn("forced_drop_reason", updated["router_decision"])

    def test_router_node_calls_only_planner_requested_tools(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review_budget.yaml").write_text(
                "budget:\n  max_rounds: 3\n  max_failures: 2\n  max_tool_calls: 8\n",
                encoding="utf-8",
            )
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: agents/subtype_review/prompts",
                        "allowed_tools:",
                        "  - tool_stability_check",
                        "  - tool_mutation_enrichment",
                    ]
                ),
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "review_round": 1,
                "verifier_decision": {"final_action_ready": False},
                "llm_audit": {
                    "decision_state": "continue_review",
                    "tools_to_call": [{"tool_name": "tool_mutation_enrichment"}],
                    "tool_plan": [{"tool_name": "tool_mutation_enrichment"}],
                },
                "budget_state": {},
            }

            updated = router_planner_node(
                {
                    "inventory": {
                        "cluster_state": cluster,
                        "config_dir": str(config_dir),
                        "output_root": str(Path(temp_dir)),
                    }
                }
            )["inventory"]["cluster_state"]

            self.assertEqual(updated["next_action"], "call_tools")
            self.assertEqual(
                updated["recommended_tools"],
                [{"tool_name": "tool_mutation_enrichment"}],
            )

    def test_verifier_starts_with_planner_without_auto_running_tools(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "validation_results": {},
            }

            def planner_side_effect(cluster_state, output_root, config_dir):
                cluster_state["planner_decision"] = {
                    "need_more_evidence": True,
                    "tools_to_call": [{"tool_name": "tool_stability_check"}],
                    "tool_plan": [{"tool_name": "tool_stability_check"}],
                    "reasoning_summary": "Stability could change accept or split.",
                }
                cluster_state["llm_audit"] = dict(cluster_state["planner_decision"])
                cluster_state["tool_plan"] = [{"tool_name": "tool_stability_check"}]
                return cluster_state["planner_decision"]

            with patch(
                "agents.subtype_review.verifier.llm_evidence_audit",
                side_effect=planner_side_effect,
            ), patch(
                "agents.subtype_review.verifier.llm_action_decision"
            ) as decider:
                updated = verifier_node(
                    {
                        "inventory": {
                            "cluster_state": cluster,
                            "patient_states_by_id": {"A": {"case_id": "A"}},
                            "all_cluster_states": [cluster],
                            "output_root": str(Path(temp_dir)),
                            "config_dir": "/data/qijun/path-ct/configs",
                        }
                    }
                )["inventory"]["cluster_state"]

            self.assertEqual(updated["validation_results"], {})
            self.assertEqual(
                updated["tool_plan"], [{"tool_name": "tool_stability_check"}]
            )
            round_dir = Path(temp_dir) / "subtype_review" / "C0001" / "round_1"
            self.assertTrue((round_dir / "pre_tool_evidence_matrix.json").exists())
            self.assertFalse((round_dir / "evidence_matrix.json").exists())
            decider.assert_not_called()

    def test_decider_continue_review_is_replanned_before_drop(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "validation_results": {},
            }
            planner_calls = []

            def planner_side_effect(cluster_state, output_root, config_dir):
                planner_calls.append(dict(cluster_state))
                if len(planner_calls) == 1:
                    cluster_state["planner_decision"] = {
                        "need_more_evidence": False,
                        "tools_to_call": [],
                        "tool_plan": [],
                        "reasoning_summary": "No initial tool plan.",
                    }
                    cluster_state["llm_audit"] = dict(cluster_state["planner_decision"])
                    cluster_state["tool_plan"] = []
                    return cluster_state["planner_decision"]
                cluster_state["planner_decision"] = {
                    "need_more_evidence": True,
                    "tools_to_call": [{"tool_name": "tool_mutation_enrichment"}],
                    "tool_plan": [{"tool_name": "tool_mutation_enrichment"}],
                    "reasoning_summary": "Decider biology uncertainty could change accept/drop.",
                }
                cluster_state["llm_audit"] = dict(cluster_state["planner_decision"])
                cluster_state["tool_plan"] = [{"tool_name": "tool_mutation_enrichment"}]
                return cluster_state["planner_decision"]

            def decider_side_effect(cluster_state, output_root, config_dir):
                decision = {
                    "decision_state": "continue_review",
                    "recommended_action": "",
                    "confidence_level": "low",
                    "blocks_to_update": ["biological_support"],
                    "continue_review_reason": "Biology could change accept/drop.",
                    "metric_refs": ["evidence_matrix.biological_support.metrics"],
                    "reasoning_summary": "Need biology evidence before final action.",
                }
                cluster_state["decider_decision"] = decision
                cluster_state["llm_audit"] = decision
                return decision

            with patch(
                "agents.subtype_review.verifier.llm_evidence_audit",
                side_effect=planner_side_effect,
            ), patch(
                "agents.subtype_review.verifier.llm_action_decision",
                side_effect=decider_side_effect,
            ):
                updated = verifier_node(
                    {
                        "inventory": {
                            "cluster_state": cluster,
                            "patient_states_by_id": {"A": {"case_id": "A"}},
                            "all_cluster_states": [cluster],
                            "output_root": str(Path(temp_dir)),
                            "config_dir": "/data/qijun/path-ct/configs",
                        }
                    }
                )["inventory"]["cluster_state"]

            self.assertEqual(len(planner_calls), 2)
            self.assertEqual(
                planner_calls[1]["decider_requested_evidence"]["blocks_to_update"],
                ["biological_support"],
            )
            self.assertEqual(
                updated["tool_plan"], [{"tool_name": "tool_mutation_enrichment"}]
            )

    def test_decider_payload_uses_current_metrics_without_evidence_catalog(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: /data/qijun/path-ct/agents/subtype_review/prompts",
                        "allowed_tools: []",
                        "llm:",
                        "  action_decision_max_new_tokens: 1024",
                    ]
                ),
                encoding="utf-8",
            )
            matrix = build_agentic_evidence_matrix(
                {
                    "cluster_id": "C0001",
                    "member_ids": ["A"],
                    "validation_results": {
                        "tool_stability_check": {
                            "status": "success",
                            "results": {
                                "metrics": {
                                    "set_reliability_global_consensus": {
                                        "total_candidate_set_n": 1
                                    }
                                }
                            },
                        }
                    },
                },
                {},
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "evidence_matrix": matrix,
                "evidence_catalog": build_evidence_catalog("C0001", matrix),
                "validation_results": {
                    "tool_stability_check": {
                        "status": "success",
                        "results": {
                            "metrics": {
                                "set_reliability_global_consensus": {
                                    "total_candidate_set_n": 1
                                }
                            }
                        },
                    }
                },
            }
            captured = {}

            def capture_messages(messages, llm_client=None, tools=None):
                captured["content"] = messages[-1]["content"]
                return {
                    "decision_state": "final",
                    "recommended_action": "drop",
                    "confidence_level": "high",
                    "reason_codes": ["insufficient_evidence"],
                    "metric_refs": [
                        "evidence_matrix.set_reliability.metrics.set_reliability_global_consensus"
                    ],
                    "drop_reason": "insufficient_evidence",
                    "reasoning_summary": "Current metrics cannot support a subtype claim.",
                }

            with patch("utils.cluster_flow.load_llm_client", return_value=TokenBudgetClient()), patch(
                "utils.cluster_flow.call_llm_json", side_effect=capture_messages
            ):
                llm_action_decision(cluster, str(Path(temp_dir)), str(config_dir))

            self.assertIn('"evidence_matrix"', captured["content"])
            self.assertNotIn('"validation_results"', captured["content"])
            self.assertNotIn('"evidence_catalog"', captured["content"])
            self.assertNotIn('"structured_evidence"', captured["content"])
            self.assertNotIn('"raw_metric_snapshot"', captured["content"])

    def test_attaching_figures_preserves_complete_existing_report(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "metadata": {"cluster_id": "C0001"},
                        "verifier_decision": {"reason_codes": ["insufficient_evidence"]},
                        "round_summaries": [{"round_index": 1}],
                        "final_evidence_matrix": {},
                        "figures": {"set_reliability": {}},
                    }
                ),
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "report_draft": {"figures": {"set_reliability": {}}},
                "artifacts": {"report": str(report_path)},
            }

            attach_global_figures_to_reports(
                [cluster], {"consensus_matrix_heatmap": "/tmp/figure.png"}
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["metadata"]["cluster_id"], "C0001")
            self.assertEqual(report["round_summaries"], [{"round_index": 1}])
            self.assertEqual(
                report["figures"]["set_reliability"]["consensus_matrix_heatmap"],
                "/tmp/figure.png",
            )

    def test_tools_for_evidence_blocks_maps_full_matrix_to_review_tools(self) -> None:
        tool_names = {
            item["tool_name"] for item in tools_for_evidence_blocks([], "")
        }

        self.assertEqual(
            tool_names,
            {
                "tool_stability_check",
                "tool_mutation_enrichment",
                "tool_pathway_enrichment",
                "tool_multimodal_consistency_check",
                "tool_survival_analysis",
                "tool_known_label_echo_test",
                "tool_confound_test",
            },
        )

    def test_dict_metrics_populate_assessments_and_drop_reason_codes(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "known_label_echo",
                    "metric": "grade: G4 q_value: 0.025",
                    "evidence": "The cluster is enriched for grade G4.",
                    "limitations": "Known label echo should be considered.",
                },
                {
                    "dimension": "confounder_exclusion",
                    "metric": "manufacturer q_value: 0.0003",
                    "evidence": "Manufacturer is associated with the cluster.",
                    "limitations": "Potential technical confounding.",
                },
            ],
            "validation_results": {
                "tool_known_label_echo_test": {
                    "results": {
                        "metrics": {
                            "known_label_global_association": {
                                "grade": {
                                    "label": "grade",
                                    "available_n": 4,
                                    "missing_n": 0,
                                    "contingency_table": {"C0001": {"G4": 2}, "C0002": {"G2": 2}},
                                    "chi_square_p_value": 0.01,
                                    "low_expected_count": True,
                                    "cramers_v": 0.3,
                                }
                            },
                            "known_label_set_enrichment": {
                                "C0001": {
                                    "grade": {
                                        "candidate_set_id": "C0001",
                                        "label": "grade",
                                        "dominant_level": "G4",
                                        "overlap_count": 2,
                                        "set_available_n": 2,
                                        "level_total_n": 2,
                                        "set_fraction": 1.0,
                                        "level_recall": 1.0,
                                        "odds_ratio": "inf",
                                        "p_value": 0.01,
                                        "q_value": 0.02,
                                    }
                                }
                            },
                        }
                    }
                },
                "tool_confound_test": {
                    "results": {
                        "metrics": {
                            "confounder_global_association": {
                                "ct_manufacturer": {
                                    "field": "ct_manufacturer",
                                    "field_type": "categorical",
                                    "available_n": 4,
                                    "missing_n": 0,
                                    "contingency_table": {"C0001": {"GE": 2}, "C0002": {"SIEMENS": 2}},
                                    "chi_square_p_value": 0.001,
                                    "q_value": 0.003,
                                    "low_expected_count": True,
                                    "cramers_v": 1.0,
                                }
                            },
                            "confounder_set_association": {
                                "C0001": {
                                    "ct_manufacturer": {
                                        "GE": {
                                            "candidate_set_id": "C0001",
                                            "field": "ct_manufacturer",
                                            "field_type": "categorical",
                                            "level": "GE",
                                            "set_count": 2,
                                            "set_total": 2,
                                            "rest_count": 0,
                                            "rest_total": 2,
                                            "level_total_n": 2,
                                            "set_fraction": 1.0,
                                            "rest_fraction": 0.0,
                                            "delta_fraction": 1.0,
                                            "p_value": 0.001,
                                            "q_value": 0.003,
                                        }
                                    }
                                }
                            },
                        }
                    }
                },
            },
            "llm_audit": {
                "recommended_action": "drop",
                "confidence_level": "high",
                "reason_codes": ["known_label_echo", "confounder_driven"],
                "metric_refs": [
                    "evidence_matrix.known_label_echo.metrics.known_label_set_enrichment.C0001.grade.q_value",
                    "evidence_matrix.confounder_exclusion.metrics.confounder_set_association.C0001.ct_manufacturer.GE.q_value",
                ],
                "confidence_basis": {
                    name: {
                        "assessment": "limitation",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "Known label echo and manufacturer confounding dominate.",
            },
        }

        cluster["evidence_matrix"] = build_agentic_evidence_matrix(cluster, {})
        verifier = build_verifier_decision(cluster)

        self.assertEqual(
            set(cluster["evidence_matrix"]["known_label_echo"]["metrics"]),
            {"known_label_global_association", "known_label_set_enrichment"},
        )
        self.assertEqual(
            cluster["evidence_matrix"]["known_label_echo"]["metrics"]["known_label_global_association"]["grade"]["label"],
            "grade",
        )
        self.assertNotIn(
            "global_redundancy",
            cluster["evidence_matrix"]["known_label_echo"]["metrics"],
        )
        self.assertEqual(
            cluster["evidence_matrix"]["confounder_exclusion"]["metrics"]["confounder_set_association"]["C0001"]["ct_manufacturer"]["GE"]["field"],
            "ct_manufacturer",
        )
        self.assertEqual(
            cluster["evidence_matrix"]["known_label_echo"]["llm_assessment"]["assessment"],
            "reviewed",
        )
        self.assertIn("known_label_echo", verifier["reason_codes"])
        self.assertIn("confounder_driven", verifier["reason_codes"])

    def test_reason_codes_come_from_llm_not_keyword_inference(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "confounder_exclusion",
                    "metric": "manufacturer q_value: 1.0",
                    "evidence": "No significant association with confounders.",
                    "limitations": "Confounder checks were reviewed.",
                }
            ],
            "llm_audit": {
                "recommended_action": "drop",
                "confidence_level": "high",
                "reason_codes": ["biologically_unsupported"],
                "metric_refs": [
                    "evidence_matrix.confounder_exclusion.metrics.confounder_set_association"
                ],
                "confidence_basis": {
                    name: {
                        "assessment": "limitation",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "The word confounder appears, but the final reason is weak biology.",
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        verifier = build_verifier_decision({**cluster, "evidence_matrix": matrix})

        self.assertEqual(verifier["reason_codes"], ["biologically_unsupported"])
        self.assertNotIn("confounder_driven", verifier["reason_codes"])

    def test_negative_echo_and_weak_multimodal_are_not_drop_reasons(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "known_label_echo",
                    "metric": "stage/grade contingency tables show no dominant enrichment",
                    "evidence": "The cluster is not a known label echo.",
                    "limitations": "No known-label dominance was observed.",
                },
                {
                    "dimension": "multimodal_support",
                    "metric": "ARI values close to zero",
                    "evidence": "Multimodal alignment is weak.",
                    "limitations": "Weak support should be reported as a limitation.",
                },
            ],
            "validation_results": {
                "tool_known_label_echo_test": {
                    "results": {
                        "metrics": {
                            "known_label_global_association": {
                                "grade": {
                                    "label": "grade",
                                    "available_n": 2,
                                    "missing_n": 0,
                                    "contingency_table": {"C0001": {"G4": 1, "G2": 1}},
                                    "chi_square_p_value": 1.0,
                                    "low_expected_count": True,
                                    "cramers_v": 0.0,
                                }
                            },
                            "known_label_set_enrichment": {
                                "C0001": {
                                    "grade": 
                                    {
                                        "candidate_set_id": "C0001",
                                        "label": "grade",
                                        "dominant_level": "G4",
                                        "overlap_count": 1,
                                        "set_available_n": 2,
                                        "level_total_n": 1,
                                        "set_fraction": 0.5,
                                        "level_recall": 1.0,
                                        "odds_ratio": 1.0,
                                        "p_value": 0.4,
                                        "q_value": 0.8,
                                    }
                                }
                            },
                        }
                    }
                },
                "tool_multimodal_consistency_check": {
                    "results": {
                        "metrics": {
                            "modality_alignment": {
                                "ct": {"available_cases": 2, "silhouette": 0.01},
                                "rna": {"available_cases": 2, "silhouette": 0.02},
                            }
                        }
                    }
                },
            },
            "llm_audit": {
                "recommended_action": "drop",
                "confidence_level": "high",
                "reason_codes": ["biologically_unsupported"],
                "metric_refs": ["evidence_matrix.multimodal_support.metrics"],
                "confidence_basis": {
                    name: {
                        "assessment": "limitation",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "The cluster has weak multimodal support but is not a known label echo.",
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        reason_codes = build_verifier_decision(
            {**cluster, "evidence_matrix": matrix}
        )["reason_codes"]

        self.assertNotIn("known_label_echo", reason_codes)
        self.assertNotIn("major_multimodal_contradiction", reason_codes)
        self.assertEqual(
            matrix["known_label_echo"]["llm_assessment"]["assessment"],
            "reviewed",
        )

    def test_confounder_block_passes_complete_metric_tables_without_legacy_summary(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "validation_results": {
                "tool_confound_test": {
                    "results": {
                        "metrics": {
                            "confounder_global_association": {
                                "year_of_diagnosis": {
                                    "field": "year_of_diagnosis",
                                    "q_value": 0.01,
                                },
                            },
                            "confounder_set_association": {
                                "C0001": {
                                    "ct_manufacturer": {
                                        "GE": {
                                            "field": "ct_manufacturer",
                                            "level": "GE",
                                            "q_value": 0.001,
                                        }
                                    }
                                }
                            },
                        }
                    }
                }
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        metrics = matrix["confounder_exclusion"]["metrics"]

        self.assertEqual(
            set(metrics),
            {"confounder_global_association", "confounder_set_association"},
        )
        self.assertEqual(metrics["confounder_global_association"]["year_of_diagnosis"]["q_value"], 0.01)
        self.assertEqual(metrics["confounder_set_association"]["C0001"]["ct_manufacturer"]["GE"]["q_value"], 0.001)
        self.assertNotIn("confounder_summary", metrics)
        self.assertNotIn("technical_associations", metrics)

    def test_multimodal_block_passes_complete_crossmodal_alignment_tables(self) -> None:
        metrics = {
            "crossmodal_global_alignment": [
                {
                    "scope": "global",
                    "candidate_set_id": None,
                    "modality_a": "ct",
                    "modality_b": "wsi",
                    "distance_metric_a": "euclidean",
                    "distance_metric_b": "cosine",
                    "available_n": 4,
                    "pair_count": 6,
                    "low_pair_count": True,
                    "spearman_distance_correlation": 0.8,
                    "mantel_p_value": 0.05,
                    "cka_similarity_alignment": 0.7,
                    "similarity_scaling": "p95_distance_to_similarity",
                    "mantel_permutations": 999,
                    "mantel_alternative": "two-sided",
                    "missing_reason": None,
                }
            ],
            "crossmodal_set_alignment": {
                "C0001": [
                    {
                        "scope": "candidate_set",
                        "candidate_set_id": "C0001",
                        "modality_a": "ct",
                        "modality_b": "wsi",
                        "distance_metric_a": "euclidean",
                        "distance_metric_b": "cosine",
                        "available_n": 2,
                        "pair_count": 1,
                        "low_pair_count": True,
                        "spearman_distance_correlation": None,
                        "mantel_p_value": None,
                        "cka_similarity_alignment": None,
                        "similarity_scaling": "p95_distance_to_similarity",
                        "mantel_permutations": 999,
                        "mantel_alternative": "two-sided",
                        "missing_reason": "insufficient_common_cases",
                    }
                ]
            },
        }
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "validation_results": {
                "tool_multimodal_consistency_check": {"results": {"metrics": metrics}}
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        multimodal_metrics = matrix["multimodal_support"]["metrics"]

        self.assertEqual(multimodal_metrics, metrics)
        self.assertEqual(matrix["multimodal_support"]["figures"], {})
        self.assertNotIn("modality_coverage", multimodal_metrics)
        self.assertNotIn("modality_separation", multimodal_metrics)
        self.assertNotIn("global_single_modality_alignment", multimodal_metrics)

    def test_clinical_context_passes_complete_os_survival_metrics(self) -> None:
        metrics = {
            "survival_global_association": {
                "endpoint": "OS",
                "available_n": 4,
                "missing_n": 0,
                "event_n": 2,
                "censored_n": 2,
                "per_set_n": {"C0001": 2, "C0002": 2},
                "per_set_event_n": {"C0001": 1, "C0002": 1},
                "per_set_censored_n": {"C0001": 1, "C0002": 1},
                "per_set_median_os_days": {"C0001": 100.0, "C0002": None},
                "logrank_p_value": 0.04,
            },
            "survival_set_association": {
                "C0001": {
                    "candidate_set_id": "C0001",
                    "endpoint": "OS",
                    "available_n": 4,
                    "missing_n": 0,
                    "set_n": 2,
                    "rest_n": 2,
                    "set_event_n": 1,
                    "rest_event_n": 1,
                    "set_censored_n": 1,
                    "rest_censored_n": 1,
                    "set_median_os_days": 100.0,
                    "rest_median_os_days": None,
                    "hazard_ratio": 2.5,
                    "ci_95_lower": 0.2,
                    "ci_95_upper": 25.0,
                    "cox_p_value": 0.1,
                    "logrank_p_value": 0.08,
                    "direction": "worse_survival_in_set",
                }
            },
        }
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "validation_results": {
                "tool_survival_analysis": {"results": {"metrics": metrics}}
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        clinical_metrics = matrix["clinical_context"]["metrics"]

        self.assertEqual(clinical_metrics, metrics)
        self.assertEqual(matrix["clinical_context"]["figures"], {})
        self.assertNotIn("clinical_summary", clinical_metrics)
        self.assertNotIn("km_logrank", clinical_metrics)

    def test_weak_biology_text_is_not_marked_as_support(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "biological_support",
                    "metric": "VHL q_value: 0.32",
                    "evidence": "Mutation-enriched genes have high q-values and are not statistically significant after correction.",
                    "limitations": "No significant mutation pathways were identified.",
                }
            ],
            "llm_audit": {
                "recommended_action": "drop",
                "confidence_level": "high",
                "reason_codes": ["biologically_unsupported"],
                "metric_refs": ["evidence_matrix.biological_support.metrics"],
                "confidence_basis": {
                    name: {
                        "assessment": "limitation",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "The cluster lacks significant biological support.",
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        verifier = build_verifier_decision({**cluster, "evidence_matrix": matrix})

        self.assertEqual(
            matrix["biological_support"]["llm_assessment"]["assessment"],
            "reviewed",
        )
        self.assertIn("biologically_unsupported", verifier["reason_codes"])

    def test_rna_pathway_support_prevents_biologically_unsupported_reason(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "biological_support",
                    "metric": "RNA_pathways: HALLMARK_HYPOXIA q_value: 0.012",
                    "evidence": "The candidate has significant RNA pathway enrichment.",
                    "limitations": "No significant mutation-enriched genes were identified.",
                }
            ],
            "validation_results": {
                "tool_pathway_enrichment": {
                    "results": {
                        "metrics": {
                            "rna_pathway_enrichment": [
                                {
                                    "candidate_set_id": "C0001",
                                    "pathway": "HALLMARK_HYPOXIA",
                                    "p_value": 0.001,
                                    "q_value": 0.012,
                                    "standardized_mean_difference": 0.8,
                                }
                            ]
                        }
                    }
                }
            },
            "llm_audit": {
                "recommended_action": "drop",
                "confidence_level": "high",
                "reason_codes": ["other_with_explanation"],
                "metric_refs": ["evidence_matrix.biological_support.metrics"],
                "confidence_basis": {
                    name: {
                        "assessment": "limitation",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "No significant mutation-enriched genes were identified.",
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        verifier = build_verifier_decision({**cluster, "evidence_matrix": matrix})

        self.assertEqual(
            matrix["biological_support"]["llm_assessment"]["assessment"],
            "reviewed",
        )
        self.assertEqual(
            set(matrix["biological_support"]["metrics"]),
            {"wxs_gene_enrichment", "wxs_pathway_enrichment", "rna_pathway_enrichment"},
        )
        self.assertEqual(matrix["biological_support"]["figures"], {})
        self.assertNotIn("biologically_unsupported", verifier["reason_codes"])

    def test_large_biological_tables_are_compacted_before_llm_context(self) -> None:
        big_rows = [
            {
                "candidate_set_id": "C0001",
                "gene": f"G{i}",
                "q_value": 0.001 + i / 1000000,
                "delta_frequency": 0.5,
            }
            for i in range(9000)
        ]
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "validation_results": {
                "tool_mutation_enrichment": {
                    "tool_name": "tool_mutation_enrichment",
                    "status": "success",
                    "results": {
                        "summary": "WXS tables computed.",
                        "metrics": {
                            "wxs_gene_enrichment": big_rows,
                            "wxs_pathway_enrichment": big_rows[:60],
                        },
                    },
                    "artifacts": {"figures": {}},
                },
                "tool_pathway_enrichment": {
                    "tool_name": "tool_pathway_enrichment",
                    "status": "success",
                    "results": {
                        "summary": "RNA tables computed.",
                        "metrics": {"rna_pathway_enrichment": big_rows[:60]},
                    },
                    "artifacts": {"figures": {}},
                },
            },
            "verification_vector": {"raw": {}},
        }
        matrix = build_agentic_evidence_matrix(cluster, {})
        wxs_view = matrix["biological_support"]["metrics"]["wxs_gene_enrichment"]
        self.assertEqual(wxs_view["row_count"], 9000)
        self.assertEqual(len(wxs_view["preview_rows"]), 20)
        self.assertEqual(wxs_view["omitted_row_count"], 8980)

        captured = {}

        def fake_llm(messages, client, tools=None):
            captured["messages"] = messages
            return {
                "cluster_id": "C0001",
                "current_hypothesis": "",
                "hypothesis_type": "candidate_subtype",
                "structured_evidence": [],
                "strong_evidence": [],
                "supportive_evidence": [],
                "weak_evidence": [],
                "missing_evidence": [],
                "contradictory_evidence": [],
                "tools_to_call": [],
                "limitations_to_report": [],
                "reasoning_summary": "ok",
            }

        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            prompt_dir = Path.cwd() / "agents" / "subtype_review" / "prompts"
            (config_dir / "subtype_review.yaml").write_text(
                f"prompt_dir: {prompt_dir}\n",
                encoding="utf-8",
            )
            with patch("utils.cluster_flow.load_llm_client", return_value=object()), patch(
                "utils.cluster_flow.call_llm_json", side_effect=fake_llm
            ):
                llm_evidence_audit(cluster, str(Path(temp_dir)), str(config_dir))

        user_content = captured["messages"][1]["content"]
        input_json = json.loads(user_content.split("Input JSON:\n", 1)[1])
        compact_metrics = input_json["validation_results"]["tool_mutation_enrichment"]["metrics"]
        prompt_wxs = compact_metrics["wxs_gene_enrichment"]
        self.assertEqual(prompt_wxs["row_count"], 9000)
        self.assertEqual(len(prompt_wxs["preview_rows"]), 20)
        self.assertNotIn(big_rows[20]["gene"], user_content)

    def test_known_label_metrics_override_negative_echo_text(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "known_label_echo",
                    "metric": "grade G4 q_value: 0.025",
                    "evidence": "The cluster is not a known label echo.",
                    "limitations": "Not strong enough to suggest a known subtype echo.",
                }
            ],
            "validation_results": {
                "tool_known_label_echo_test": {
                    "results": {
                        "metrics": {
                            "known_label_set_enrichment": {
                                "C0001": {
                                    "grade": 
                                    {
                                        "candidate_set_id": "C0001",
                                        "label": "grade",
                                        "dominant_level": "G4",
                                        "overlap_count": 2,
                                        "set_available_n": 2,
                                        "level_total_n": 2,
                                        "set_fraction": 1.0,
                                        "level_recall": 1.0,
                                        "odds_ratio": "inf",
                                        "p_value": 0.008,
                                        "q_value": 0.025,
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "llm_audit": {
                "recommended_action": "drop",
                "confidence_level": "high",
                "reason_codes": ["known_label_echo"],
                "metric_refs": [
                    "evidence_matrix.known_label_echo.metrics.known_label_set_enrichment.C0001.grade.q_value"
                ],
                "confidence_basis": {
                    name: {
                        "assessment": "limitation",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "There is no strong known label echo.",
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        verifier = build_verifier_decision({**cluster, "evidence_matrix": matrix})

        self.assertEqual(
            matrix["known_label_echo"]["llm_assessment"]["assessment"],
            "reviewed",
        )
        self.assertIn("known_label_echo", verifier["reason_codes"])

    def test_known_label_negative_association_text_is_support(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "known_label_echo",
                    "metric": "stage/grade overlap tables show no enriched dominant label",
                    "evidence": "The candidate cluster is not strongly associated with known clinical labels.",
                    "limitations": "This is not a simple echo of known clinical subtypes.",
                }
            ],
            "llm_audit": {
                "recommended_action": "drop",
                "confidence_level": "high",
                "reason_codes": ["biologically_unsupported"],
                "metric_refs": ["evidence_matrix.known_label_echo.metrics"],
                "confidence_basis": {
                    name: {
                        "assessment": "limitation",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "The known label echo test indicates no strong known label echo.",
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        verifier = build_verifier_decision({**cluster, "evidence_matrix": matrix})

        self.assertEqual(
            matrix["known_label_echo"]["llm_assessment"]["assessment"],
            "reviewed",
        )
        self.assertNotIn("known_label_echo", verifier["reason_codes"])

    def test_placeholder_evidence_text_is_unavailable_not_support(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "multimodal_support",
                    "evidence": "Multimodal evidence is important to validate the cluster.",
                    "limitations": "",
                },
                {
                    "dimension": "known_label_echo",
                    "evidence": "No known label echo evidence available.",
                    "limitations": "",
                },
            ],
        }

        matrix = build_agentic_evidence_matrix(cluster, {})

        self.assertEqual(
            matrix["multimodal_support"]["llm_assessment"]["assessment"],
            "unavailable",
        )
        self.assertEqual(
            matrix["known_label_echo"]["llm_assessment"]["assessment"],
            "unavailable",
        )

    def test_llm_accept_is_not_overridden_by_concern_dimensions(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "structured_evidence": [
                {
                    "dimension": "set_reliability",
                    "evidence": "Stable cluster.",
                    "limitations": "",
                },
                {
                    "dimension": "biological_support",
                    "evidence": "Significant RNA pathway enrichment.",
                    "limitations": "",
                },
                {
                    "dimension": "known_label_echo",
                    "evidence": "The cluster is not a known label echo.",
                    "limitations": "",
                },
                {
                    "dimension": "confounder_exclusion",
                    "evidence": "Manufacturer is significantly associated with the cluster.",
                    "limitations": "Technical confounding concern.",
                },
            ],
            "validation_results": {
                "tool_pathway_enrichment": {
                    "results": {
                        "metrics": {
                            "rna_pathway_enrichment": [
                                {
                                    "candidate_set_id": "C0001",
                                    "pathway": "HALLMARK_HYPOXIA",
                                    "q_value": 0.01,
                                    "standardized_mean_difference": 0.8,
                                }
                            ]
                        }
                    }
                },
                "tool_confound_test": {
                    "results": {
                        "metrics": {
                            "confounder_set_association": {
                                "C0001": {
                                    "ct_manufacturer": {
                                        "GE": {
                                            "field": "ct_manufacturer",
                                            "level": "GE",
                                            "q_value": 0.001,
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            },
            "llm_audit": {
                "recommended_action": "accept",
                "confidence_level": "high",
                "reason_codes": ["reliable_biological_nonconfounded"],
                "metric_refs": ["evidence_matrix.biological_support.metrics.rna_pathway_enrichment.0.q_value"],
                "confidence_basis": {
                    name: {
                        "assessment": "support",
                        "supports_action": True,
                        "limitations": "Reviewed.",
                        "dominance_judgment": "not_dominant",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "The cluster is stable and biologically supported.",
            },
        }

        matrix = build_agentic_evidence_matrix(cluster, {})
        catalog = build_evidence_catalog("C0001", matrix)
        cluster["llm_audit"]["evidence_ids"] = [catalog[0]["evidence_id"]]
        verifier = build_verifier_decision(
            {**cluster, "evidence_matrix": matrix, "evidence_catalog": catalog}
        )

        self.assertTrue(verifier["final_action_ready"])
        self.assertEqual(verifier["final_action"], "accept")
        self.assertEqual(verifier["reason_codes"], ["reliable_biological_nonconfounded"])

    def test_moderate_confidence_accept_is_not_final_ready(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "evidence_matrix": build_agentic_evidence_matrix(
                {"cluster_id": "C0001", "member_ids": ["A", "B"]}, {}
            ),
            "llm_audit": {
                "recommended_action": "accept",
                "confidence_level": "moderate",
                "reason_codes": ["reliable_biological_nonconfounded"],
                "metric_refs": [
                    "evidence_matrix.set_reliability.metrics.set_reliability_set_consensus.C0001.set_n"
                ],
                "blocks_to_update": ["biological_support"],
                "continue_review_reason": "Biological support needs stronger review.",
                "confidence_basis": {
                    name: {
                        "assessment": "support",
                        "supports_action": True,
                        "limitations": "",
                        "dominance_judgment": "reviewed",
                        "metric_refs": [f"evidence_matrix.{name}.metrics"],
                    }
                    for name in [
                        "set_reliability",
                        "biological_support",
                        "multimodal_support",
                        "clinical_context",
                        "known_label_echo",
                        "confounder_exclusion",
                    ]
                },
                "reasoning_summary": "Promising but not high confidence.",
            },
        }

        verifier = build_verifier_decision(cluster)
        router = agentic_router_decision(
            {**cluster, "review_round": 1, "verifier_decision": verifier},
            {"budget": {"max_rounds": 3}},
        )

        self.assertFalse(verifier["final_action_ready"])
        self.assertEqual(verifier["confidence_level"], "moderate")
        self.assertEqual(router["route_action"], "continue_review")
        self.assertEqual(router["blocks_to_update"], ["biological_support"])
        self.assertEqual(router.get("tools_to_call", []), [])

    def test_router_does_not_rerun_tools_for_blocks_with_existing_metrics(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A", "B"]}, {}
        )
        verifier = {
            "final_action_ready": False,
            "confidence_level": "moderate",
            "blocks_to_update": ["biological_support", "multimodal_support"],
            "decision_consistency_check": {
                "passed": False,
                "issues": ["non_high_confidence:moderate"],
            },
        }

        router = agentic_router_decision(
            {
                "cluster_id": "C0001",
                "review_round": 2,
                "evidence_matrix": matrix,
                "verifier_decision": verifier,
                "llm_audit": {"recommended_action": "accept"},
            },
            {"budget": {"max_rounds": 3, "max_failures": 2}},
        )

        self.assertEqual(router["route_action"], "continue_review")
        self.assertEqual(
            router["blocks_to_update"],
            ["biological_support", "multimodal_support"],
        )
        self.assertEqual(router.get("tools_to_call", []), [])

    def test_continue_review_decision_state_allows_missing_recommended_action(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A", "B"]}, {}
        )
        catalog = build_evidence_catalog("C0001", matrix)
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "review_round": 1,
            "evidence_matrix": matrix,
            "evidence_catalog": catalog,
            "llm_audit": {
                "decision_state": "continue_review",
                "confidence_level": "moderate",
                "evidence_ids": [catalog[0]["evidence_id"]],
                "blocks_to_update": ["biological_support"],
                "continue_review_reason": "Biological support needs stronger review.",
                "reasoning_summary": "Evidence is not sufficient for a final action yet.",
            },
        }

        verifier = build_verifier_decision(cluster)
        router = agentic_router_decision(
            {**cluster, "verifier_decision": verifier},
            {"budget": {"max_rounds": 3, "max_failures": 2}},
        )

        self.assertFalse(verifier["final_action_ready"])
        self.assertEqual(verifier["decision_state"], "continue_review")
        self.assertEqual(verifier["decision_consistency_check"]["issues"], [])
        self.assertEqual(router["route_action"], "continue_review")
        self.assertEqual(router["send_to_revision_action"], None)
        self.assertEqual(router["blocks_to_update"], ["biological_support"])

    def test_list_confidence_basis_does_not_crash_verifier(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A"]}, {}
        )
        verifier = build_verifier_decision(
            {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "evidence_matrix": matrix,
                "llm_audit": {
                    "decision_state": "continue_review",
                    "confidence_level": "low",
                    "confidence_basis": [
                        "set_reliability: unavailable",
                        "biological_support: unavailable",
                    ],
                    "blocks_to_update": ["biological_support"],
                    "continue_review_reason": "Need more biology.",
                    "metric_refs": ["evidence_matrix.biological_support.metrics"],
                    "reasoning_summary": "Need more evidence.",
                },
            }
        )

        self.assertEqual(
            verifier["confidence_basis"]["summary"],
            ["set_reliability: unavailable", "biological_support: unavailable"],
        )
        self.assertEqual(verifier["decision_state"], "continue_review")

    def test_final_decision_state_requires_recommended_action(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A"]}, {}
        )
        catalog = build_evidence_catalog("C0001", matrix)
        issues = build_verifier_decision(
            {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "evidence_matrix": matrix,
                "evidence_catalog": catalog,
                "llm_audit": {
                    "decision_state": "final",
                    "confidence_level": "high",
                    "reason_codes": ["reliable_biological_nonconfounded"],
                    "evidence_ids": [catalog[0]["evidence_id"]],
                    "reasoning_summary": "Evidence supports a final action.",
                },
            }
        )["decision_consistency_check"]["issues"]

        self.assertIn("missing_recommended_action", issues)

    def test_third_continue_review_round_becomes_insufficient_drop(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A", "B"]}, {}
        )
        catalog = build_evidence_catalog("C0001", matrix)
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A", "B"],
            "review_round": 3,
            "evidence_matrix": matrix,
            "evidence_catalog": catalog,
            "llm_audit": {
                "decision_state": "continue_review",
                "confidence_level": "moderate",
                "evidence_ids": [catalog[0]["evidence_id"]],
                "blocks_to_update": ["biological_support"],
                "continue_review_reason": "Evidence remains insufficient.",
                "reasoning_summary": "No high-confidence final action is supported.",
            },
            "budget_state": {},
        }
        verifier = build_verifier_decision(cluster)

        router = agentic_router_decision(
            {**cluster, "verifier_decision": verifier},
            {"budget": {"max_rounds": 3, "max_failures": 2}},
        )

        self.assertEqual(router["route_action"], "send_to_revision")
        self.assertEqual(router["send_to_revision_action"], "drop")
        self.assertEqual(router["forced_drop_reason"], "insufficient_evidence")

    def test_high_confidence_accept_does_not_require_confidence_basis(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A"]}, {}
        )
        catalog = build_evidence_catalog("C0001", matrix)
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A"],
            "evidence_matrix": matrix,
            "evidence_catalog": catalog,
            "llm_audit": {
                "recommended_action": "accept",
                "confidence_level": "high",
                "reason_codes": ["reliable_biological_nonconfounded"],
                "evidence_ids": [catalog[0]["evidence_id"]],
                "reasoning_summary": "Evidence supports accepting this candidate subtype.",
            },
        }

        verifier = build_verifier_decision(cluster)

        self.assertTrue(verifier["final_action_ready"])
        self.assertEqual(verifier["final_action"], "accept")
        self.assertEqual(verifier["evidence_ids"], [catalog[0]["evidence_id"]])
        self.assertEqual(verifier["metric_refs"], catalog[0]["metric_refs"])
        self.assertEqual(verifier["decision_consistency_check"]["issues"], [])

    def test_evidence_catalog_uses_stable_ids_and_real_metric_refs(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A", "B"]}, {}
        )

        catalog = build_evidence_catalog("C0001", matrix)

        self.assertTrue(catalog)
        self.assertTrue(all(item["evidence_id"].startswith("C0001__") for item in catalog))
        self.assertTrue(all(item["metric_refs"] for item in catalog))
        self.assertIn(
            "evidence_matrix.set_reliability.metrics.set_reliability_set_consensus",
            [ref for item in catalog for ref in item["metric_refs"]],
        )

    def test_evidence_catalog_omits_duplicate_block_overview_items(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A", "B"]}, {}
        )

        catalog = build_evidence_catalog("C0001", matrix)

        self.assertFalse(
            any(item["evidence_id"].endswith("__metrics") for item in catalog)
        )
        self.assertIn(
            "C0001__set_reliability__set_reliability_global_consensus",
            [item["evidence_id"] for item in catalog],
        )
        self.assertIn(
            "C0001__clinical_context__survival_global_association",
            [item["evidence_id"] for item in catalog],
        )

    def test_invalid_evidence_id_blocks_final_action(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A"]}, {}
        )
        verifier = build_verifier_decision(
            {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "evidence_matrix": matrix,
                "evidence_catalog": build_evidence_catalog("C0001", matrix),
                "llm_audit": {
                    "recommended_action": "accept",
                    "confidence_level": "high",
                    "reason_codes": ["reliable_biological_nonconfounded"],
                    "evidence_ids": ["missing_evidence"],
                    "reasoning_summary": "Evidence supports accepting this candidate subtype.",
                },
            }
        )

        self.assertFalse(verifier["final_action_ready"])
        self.assertIn(
            "missing_evidence_id:missing_evidence",
            verifier["decision_consistency_check"]["issues"],
        )

    def test_action_decision_contract_repair_counts_failure_without_new_round(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: /data/qijun/path-ct/agents/subtype_review/prompts",
                        "allowed_tools: []",
                        "llm:",
                        "  model_name: test",
                        "  base_url: http://127.0.0.1:9/v1",
                        "  temperature: 0",
                        "  max_new_tokens: 64",
                    ]
                ),
                encoding="utf-8",
            )
            matrix = build_agentic_evidence_matrix(
                {"cluster_id": "C0001", "member_ids": ["A"]}, {}
            )
            catalog = build_evidence_catalog("C0001", matrix)
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "evidence_matrix": matrix,
                "evidence_catalog": catalog,
                "budget_state": {},
            }
            invalid = {
                "recommended_action": "accept",
                "confidence_level": "high",
                "reason_codes": ["confounder_driven"],
                "evidence_ids": ["missing_evidence"],
                "reasoning_summary": "Evidence supports accepting this candidate subtype.",
            }
            repaired = {
                "recommended_action": "accept",
                "confidence_level": "high",
                "reason_codes": ["reliable_biological_nonconfounded"],
                "evidence_ids": [catalog[0]["evidence_id"]],
                "reasoning_summary": "Evidence supports accepting this candidate subtype.",
            }

            with patch("utils.cluster_flow.load_llm_client", return_value=object()), patch(
                "utils.cluster_flow.call_llm_json", side_effect=[invalid, repaired]
            ) as call:
                llm_action_decision(cluster, str(Path(temp_dir)), str(config_dir))

            self.assertEqual(call.call_count, 2)
            self.assertEqual(cluster["review_round"], 1)
            self.assertEqual(cluster["budget_state"].get("agent_failures", 0), 0)
            self.assertEqual(
                cluster["llm_audit"]["reason_codes"],
                ["reliable_biological_nonconfounded"],
            )
            self.assertEqual(
                cluster["llm_audit"]["evidence_ids"],
                [catalog[0]["evidence_id"]],
            )

    def test_action_decision_uses_configured_token_budget(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: /data/qijun/path-ct/agents/subtype_review/prompts",
                        "allowed_tools: []",
                        "llm:",
                        "  model_name: test",
                        "  base_url: http://127.0.0.1:9/v1",
                        "  temperature: 0",
                        "  max_new_tokens: 2048",
                        "  action_decision_max_new_tokens: 1024",
                    ]
                ),
                encoding="utf-8",
            )
            matrix = build_agentic_evidence_matrix(
                {"cluster_id": "C0001", "member_ids": ["A"]}, {}
            )
            catalog = build_evidence_catalog("C0001", matrix)
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "evidence_matrix": matrix,
                "evidence_catalog": catalog,
                "budget_state": {},
            }
            decision = {
                "recommended_action": "accept",
                "confidence_level": "high",
                "reason_codes": ["reliable_biological_nonconfounded"],
                "evidence_ids": [catalog[0]["evidence_id"]],
                "reasoning_summary": "Evidence supports accepting this candidate subtype.",
            }
            client = TokenBudgetClient()
            seen_budgets = []

            def capture_budget(_messages, llm_client=None, tools=None):
                seen_budgets.append(llm_client.max_tokens)
                return decision

            with patch("utils.cluster_flow.load_llm_client", return_value=client), patch(
                "utils.cluster_flow.call_llm_json", side_effect=capture_budget
            ):
                llm_action_decision(cluster, str(Path(temp_dir)), str(config_dir))

            self.assertEqual(seen_budgets, [1024])
            self.assertEqual(client.max_tokens, 2048)

    def test_evidence_audit_uses_configured_token_budget(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: /data/qijun/path-ct/agents/subtype_review/prompts",
                        "allowed_tools: []",
                        "llm:",
                        "  model_name: test",
                        "  base_url: http://127.0.0.1:9/v1",
                        "  temperature: 0",
                        "  max_new_tokens: 2048",
                        "  evidence_audit_max_new_tokens: 1024",
                    ]
                ),
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "validation_results": {},
                "budget_state": {},
            }
            audit = {
                "cluster_id": "C0001",
                "current_hypothesis": "",
                "hypothesis_type": "candidate_subtype",
                "structured_evidence": [],
                "strong_evidence": [],
                "supportive_evidence": [],
                "weak_evidence": [],
                "missing_evidence": [],
                "contradictory_evidence": [],
                "tools_to_call": [],
                "limitations_to_report": [],
                "reasoning_summary": "ok",
            }
            client = TokenBudgetClient()
            seen_budgets = []

            def capture_budget(_messages, llm_client=None, tools=None):
                seen_budgets.append(llm_client.max_tokens)
                return audit

            with patch("utils.cluster_flow.load_llm_client", return_value=client), patch(
                "utils.cluster_flow.call_llm_json", side_effect=capture_budget
            ):
                llm_evidence_audit(cluster, str(Path(temp_dir)), str(config_dir))

            self.assertEqual(seen_budgets, [1024])
            self.assertEqual(client.max_tokens, 2048)

    def test_failed_action_decision_repair_counts_agent_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: /data/qijun/path-ct/agents/subtype_review/prompts",
                        "allowed_tools: []",
                        "llm:",
                        "  model_name: test",
                        "  base_url: http://127.0.0.1:9/v1",
                        "  temperature: 0",
                        "  max_new_tokens: 64",
                    ]
                ),
                encoding="utf-8",
            )
            matrix = build_agentic_evidence_matrix(
                {"cluster_id": "C0001", "member_ids": ["A"]}, {}
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "evidence_matrix": matrix,
                "evidence_catalog": build_evidence_catalog("C0001", matrix),
                "budget_state": {},
            }
            invalid = {
                "decision_state": "final",
                "confidence_level": "high",
                "reason_codes": ["confounder_driven"],
                "evidence_ids": ["missing_evidence"],
                "reasoning_summary": "Evidence supports a final action.",
            }

            with patch("utils.cluster_flow.load_llm_client", return_value=object()), patch(
                "utils.cluster_flow.call_llm_json", side_effect=[invalid, invalid]
            ):
                llm_action_decision(cluster, str(Path(temp_dir)), str(config_dir))

            self.assertEqual(cluster["budget_state"]["agent_failures"], 1)

    def test_reason_code_must_match_action(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A"]}, {}
        )
        verifier = build_verifier_decision(
            {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "evidence_matrix": matrix,
                "llm_audit": {
                    "recommended_action": "accept",
                    "confidence_level": "high",
                    "reason_codes": ["confounder_driven"],
                    "metric_refs": [
                        "evidence_matrix.set_reliability.metrics.set_reliability_set_consensus.C0001.set_n"
                    ],
                    "reasoning_summary": "Evidence supports accepting this candidate subtype.",
                },
            }
        )

        self.assertFalse(verifier["final_action_ready"])
        self.assertIn("reason_code_not_allowed_for_action:confounder_driven", verifier["decision_consistency_check"]["issues"])

    def test_action_rationale_contradiction_is_not_final_ready(self) -> None:
        cluster = {
            "cluster_id": "C0001",
            "member_ids": ["A"],
            "evidence_matrix": build_agentic_evidence_matrix(
                {"cluster_id": "C0001", "member_ids": ["A"]}, {}
            ),
            "llm_audit": {
                "recommended_action": "drop",
                "drop_reason": "known_label_echo",
                "confidence_level": "high",
                "reason_codes": ["known_label_echo"],
                "metric_refs": [
                    "evidence_matrix.set_reliability.metrics.set_reliability_set_consensus.C0001.set_n"
                ],
                "reasoning_summary": "Overall evidence supports accepting this candidate subtype.",
            },
        }

        verifier = build_verifier_decision(cluster)

        self.assertFalse(verifier["final_action_ready"])
        self.assertIn("action_rationale_conflict", verifier["decision_consistency_check"]["issues"])
        self.assertEqual(verifier["blocks_to_update"], [])

    def test_invalid_llm_action_becomes_review_unavailable_not_drop(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review_budget.yaml").write_text(
                "budget:\n  max_rounds: 3\n  max_failures: 2\n",
                encoding="utf-8",
            )
            (config_dir / "subtype_review.yaml").write_text(
                "allowed_tools: []\n",
                encoding="utf-8",
            )
            matrix = build_agentic_evidence_matrix(
                {"cluster_id": "C0001", "member_ids": ["A"]}, {}
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "review_round": 1,
                "status": "under_review",
                "evidence_matrix": matrix,
                "llm_audit": {
                    "_parse_error": True,
                    "recommended_action": "",
                    "review_unavailable_reason": "llm_parse_failure",
                    "reasoning_summary": "Action decision LLM parsing failed.",
                },
            }
            verifier = build_verifier_decision(cluster)
            router = agentic_router_decision(
                {**cluster, "verifier_decision": verifier},
                {"budget": {"max_rounds": 3, "max_failures": 2}},
            )
            reported = generate_cluster_report(
                {
                    **cluster,
                    "status": REVIEW_UNAVAILABLE_STATUS,
                    "review_unavailable_reason": router["forced_drop_reason"],
                    "router_decision": router,
                    "verifier_decision": verifier,
                    "budget_state": {"report_calls_used": 0},
                },
                str(Path(temp_dir)),
                str(config_dir),
            )

            self.assertEqual(router["send_to_revision_action"], REVIEW_UNAVAILABLE_STATUS)
            self.assertEqual(reported["report_draft"]["metadata"]["final_action"], "")
            self.assertEqual(
                reported["report_draft"]["metadata"]["review_status"],
                REVIEW_UNAVAILABLE_STATUS,
            )
            self.assertEqual(
                reported["report_draft"]["verifier_decision"]["reason_codes"],
                ["review_unavailable"],
            )
            self.assertEqual(
                set(reported["report_draft"]["figures"]),
                {
                    "set_reliability",
                    "biological_support",
                    "multimodal_support",
                    "clinical_context",
                    "known_label_echo",
                    "confounder_exclusion",
                },
            )
            self.assertEqual(reported["report_draft"]["figures"]["known_label_echo"], {})
            self.assertEqual(reported["report_draft"]["figures"]["confounder_exclusion"], {})
            output = build_pipeline_output(
                patient_states=[{"case_id": "A", "qc": "success"}],
                candidate_clusters=[{"cluster_id": "C0001", "member_ids": ["A"]}],
                cluster_states=[reported],
                patient_store_paths={},
                cluster_store_paths={},
                graph_paths={},
            )
            summary = output["final_review_summary"]
            self.assertEqual(summary["overview"]["action_counts"], {REVIEW_UNAVAILABLE_STATUS: 1})
            self.assertIn("visualization_policy", summary)
            self.assertIn(
                "known_label_echo",
                summary["visualization_policy"]["table_only_dimensions"],
            )
            self.assertIn(
                "confounder_figures",
                summary["visualization_policy"]["not_generated_by_default"],
            )
            self.assertNotIn("global_known_label_association_matrix", summary["global_figures"])
            self.assertNotIn("global_confounder_association_matrix", summary["global_figures"])
            self.assertNotIn("global_multimodal_support_dotplot", summary["global_figures"])
            self.assertEqual(summary["dropped_clusters"], [])
            self.assertEqual(
                summary["review_unavailable_clusters"][0]["cluster_id"],
                "C0001",
            )
            self.assertEqual(
                summary["all_cluster_reports"][0]["review_unavailable_reason"],
                router["forced_drop_reason"],
            )
            self.assertEqual(
                summary["review_unavailable_clusters"][0]["reason"],
                router["forced_drop_reason"],
            )

    def test_tool_budget_exhaustion_is_insufficient_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review_budget.yaml").write_text(
                "budget:\n  max_rounds: 3\n  max_tool_calls: 0\n  max_failures: 2\n  max_revisions: 3\n  max_split_depth: 2\n  max_merge_attempts: 2\n",
                encoding="utf-8",
            )
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "artifact_policy:",
                        "  save_debug_rounds: false",
                        "allowed_tools:",
                        "  - tool_stability_check",
                    ]
                ),
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "next_action": "call_tools",
                "tool_plan": [{"tool_name": "tool_stability_check"}],
                "budget_state": {"tool_calls_used": 0, "report_calls_used": 0},
            }

            updated = revision_engine(
                cluster,
                {"A": {"case_id": "A"}},
                str(Path(temp_dir)),
                str(config_dir),
            )

            self.assertEqual(updated["status"], "drop")
            self.assertEqual(updated["final_action"], "drop")
            self.assertEqual(updated["drop_reason"], "insufficient_evidence")
            self.assertEqual(
                updated["report_draft"]["verifier_decision"]["reason_codes"],
                ["insufficient_evidence"],
            )

    def test_decision_consistency_requires_existing_metric_refs(self) -> None:
        matrix = build_agentic_evidence_matrix(
            {"cluster_id": "C0001", "member_ids": ["A"]}, {}
        )

        check = decision_consistency_check(
            final_action="accept",
            reason_codes=["reliable_biological_nonconfounded"],
            rationale="Evidence supports accepting this candidate subtype.",
            metric_refs=["evidence_matrix.not_a_block.metrics"],
            evidence_matrix=matrix,
        )

        self.assertFalse(check["passed"])
        self.assertIn("missing_metric_ref:evidence_matrix.not_a_block.metrics", check["issues"])

    def test_patient_coverage_uses_qc_passed_denominator(self) -> None:
        output = build_pipeline_output(
            patient_states=[
                {"case_id": "A", "qc": "success"},
                {"case_id": "B", "qc": "success"},
                {"case_id": "C", "qc": "failed"},
            ],
            candidate_clusters=[{"cluster_id": "C0001", "member_ids": ["A", "B"]}],
            cluster_states=[
                {
                    "cluster_id": "C0001",
                    "member_ids": ["A", "B"],
                    "status": "accept",
                    "final_action": "accept",
                    "final_decision": "accept",
                }
            ],
            patient_store_paths={},
            cluster_store_paths={},
            graph_paths={},
        )

        coverage = output["final_review_summary"]["overview"]["patient_coverage"]

        self.assertEqual(coverage["input_patient_count"], 3)
        self.assertEqual(coverage["qc_passed_patient_count"], 2)
        self.assertEqual(coverage["covered_fraction"], 1.0)

    def test_split_children_restart_review_round_at_one(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review_budget.yaml").write_text("budget:\n  max_rounds: 3\n", encoding="utf-8")
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: agents/subtype_review/prompts",
                        "allowed_tools: []",
                    ]
                ),
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B", "C", "D"],
                "parent_cluster_ids": ["C0001"],
                "review_round": 2,
                "next_action": "split",
                "llm_audit": {
                    "split_plan": {
                        "new_clusters": [
                            {"cluster_id": "C0001_S1", "member_ids": ["A", "B"]},
                            {"cluster_id": "C0001_S2", "member_ids": ["C", "D"]},
                        ]
                    }
                },
                "budget_state": {},
            }

            updated = revision_engine(cluster, {}, str(Path(temp_dir)), str(config_dir), [])

            self.assertEqual(
                [child["review_round"] for child in updated["generated_clusters"]],
                [0, 0],
            )
            for child in updated["generated_clusters"]:
                self.assertEqual(child["validation_results"], {})

    def test_split_members_are_generated_when_llm_plan_has_no_member_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review_budget.yaml").write_text("budget:\n  max_rounds: 3\n", encoding="utf-8")
            (config_dir / "subtype_review.yaml").write_text(
                "prompt_dir: agents/subtype_review/prompts\nallowed_tools: []\n",
                encoding="utf-8",
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B", "C", "D"],
                "parent_cluster_ids": ["C0001"],
                "next_action": "split",
                "llm_audit": {"split_plan": {"reason": "internal heterogeneity"}},
                "budget_state": {},
                "consensus": {
                    "patient_ids": ["A", "B", "C", "D"],
                    "matrix": [
                        [1.0, 0.9, 0.1, 0.1],
                        [0.9, 1.0, 0.1, 0.1],
                        [0.1, 0.1, 1.0, 0.85],
                        [0.1, 0.1, 0.85, 1.0],
                    ],
                },
            }

            updated = revision_engine(cluster, {}, str(Path(temp_dir)), str(config_dir), [])
            children = updated["generated_clusters"]

            self.assertEqual(len(children), 2)
            self.assertEqual(sorted(len(child["member_ids"]) for child in children), [2, 2])

    def test_cluster_report_and_summary_include_agentic_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            (config_dir / "subtype_review.yaml").write_text(
                "\n".join(
                    [
                        "prompt_dir: agents/subtype_review/prompts",
                        "allowed_tools: []",
                    ]
                ),
                encoding="utf-8",
            )
            evidence_matrix = build_agentic_evidence_matrix(
                {"cluster_id": "C0001", "member_ids": ["A"]},
                {"A": {"case_id": "A"}},
            )
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "parent_cluster_ids": ["C0001"],
                "final_action": "accept",
                "final_decision": "accept",
                "status": "accept",
                "review_round": 1,
                "evidence_matrix": evidence_matrix,
                "verifier_decision": {
                    "final_action": "accept",
                    "reason_codes": ["reliable_biological_nonconfounded"],
                    "rationale": "Evidence supports accept.",
                    "metric_refs": [
                        "final_evidence_matrix.set_reliability.metrics.set_reliability_set_consensus.C0001.set_n"
                    ],
                    "dimension_assessments": {
                        key: block["llm_assessment"]["assessment"]
                        for key, block in evidence_matrix.items()
                    },
                },
                "budget_state": {"report_calls_used": 1},
            }

            reported = generate_cluster_report(cluster, str(Path(temp_dir)), str(config_dir))
            report = reported["report_draft"]

            self.assertEqual(
                set(report),
                {
                    "metadata",
                    "round_summaries",
                    "final_evidence_matrix",
                    "verifier_decision",
                    "revision_result",
                    "decision_basis",
                    "limitations",
                    "figures",
                },
            )
            output = build_pipeline_output(
                patient_states=[{"case_id": "A", "qc": "success"}],
                candidate_clusters=[{"cluster_id": "C0001", "member_ids": ["A"]}],
                cluster_states=[reported],
                patient_store_paths={},
                cluster_store_paths={},
                graph_paths={},
            )

            summary = output["final_review_summary"]
            self.assertIn("overview", summary)
            self.assertIn("accepted_candidate_subtypes", summary)
            self.assertIn("all_cluster_reports", summary)
            self.assertIn("review_convergence_summary", summary)
            self.assertEqual(summary["all_cluster_reports"][0]["reason_codes"], ["reliable_biological_nonconfounded"])

    def test_revised_cluster_summary_keeps_primary_and_all_reason_codes(self) -> None:
        output = build_pipeline_output(
            patient_states=[{"case_id": "A", "qc": "success"}],
            candidate_clusters=[{"cluster_id": "C0001", "member_ids": ["A"]}],
            cluster_states=[
                {
                    "cluster_id": "C0001",
                    "member_ids": ["A"],
                    "final_action": "drop",
                    "final_decision": "drop",
                    "status": "drop",
                    "drop_reason": "confounder_driven",
                    "report_draft": {
                        "metadata": {"cluster_id": "C0001", "final_action": "drop", "rounds_used": 2, "parent_cluster_ids": ["C0001"]},
                        "verifier_decision": {
                            "reason_codes": ["confounder_driven", "known_label_echo"],
                            "dimension_assessments": {},
                        },
                    },
                }
            ],
            patient_store_paths={},
            cluster_store_paths={},
            graph_paths={},
        )

        entry = output["final_review_summary"]["all_cluster_reports"][0]
        self.assertEqual(entry["primary_reason_code"], "confounder_driven")
        self.assertEqual(entry["reason_codes"], ["confounder_driven", "known_label_echo"])


if __name__ == "__main__":
    unittest.main()
