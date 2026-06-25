from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as pipeline_main
from utils.cluster_flow import choose_router_action, revision_engine


def write_budget(root: Path, *, allowed_tools: list[str] | None = None, **budget) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review_budget.yaml").write_text(
        "\n".join(
            [
                "budget:",
                f"  max_rounds: {budget.get('max_rounds', 3)}",
                f"  max_tool_calls: {budget.get('max_tool_calls', 8)}",
                f"  max_failures: {budget.get('max_failures', 2)}",
                f"  max_revisions: {budget.get('max_revisions', 3)}",
                f"  max_split_depth: {budget.get('max_split_depth', 2)}",
                f"  max_merge_attempts: {budget.get('max_merge_attempts', 2)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tool_lines = [f"  - {tool}" for tool in list(allowed_tools or [])]
    (config_dir / "subtype_review.yaml").write_text(
        "\n".join(
            [
                "llm:",
                "  model_path: local",
                "  model_name: local",
                "  base_url: http://127.0.0.1:9/v1",
                "  temperature: 0.1",
                "  max_new_tokens: 256",
                f"prompt_dir: {Path.cwd() / 'agents/subtype_review/prompts'}",
                "allowed_tools:",
                *tool_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_dir


class FakeSubtypeReviewGraph:
    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.invoked_cluster_ids: list[str] = []

    def invoke(self, payload, config=None):
        cluster = dict(payload["inventory"]["cluster"])
        cluster_id = str(cluster.get("cluster_id", ""))
        self.invoked_cluster_ids.append(cluster_id)
        response = dict(self.responses[cluster_id])
        inventory = dict(payload["inventory"])
        inventory["cluster_state"] = response
        return {"inventory": inventory}


class SubtypeReviewClosedLoopTest(unittest.TestCase):
    def test_illegal_llm_action_routes_to_drop(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = write_budget(Path(temp_dir))
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B"],
                "llm_audit": {"recommended_action": "manual_review"},
                "budget_state": {},
            }

            self.assertEqual(choose_router_action(cluster, str(config_dir)), "drop")
            self.assertEqual(cluster["final_action"], "drop")
            self.assertEqual(cluster["drop_reason"], "invalid_or_missing_action")

    def test_tool_plan_routes_to_internal_call_tools(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = write_budget(Path(temp_dir), allowed_tools=["tool_stability_check"])
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B"],
                "llm_audit": {
                    "recommended_action": "",
                    "tools_to_call": [{"tool_name": "tool_stability_check"}],
                },
                "budget_state": {},
            }

            self.assertEqual(choose_router_action(cluster, str(config_dir)), "call_tools")
            self.assertEqual(cluster["tool_plan"], [{"tool_name": "tool_stability_check"}])

    def test_call_tools_writes_completed_round_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = write_budget(Path(temp_dir), allowed_tools=["tool_stability_check"])

            def fake_tool(cluster_state, patient_states_by_id, output_root, config_dir="", all_cluster_states=None):
                return {
                    "tool_name": "tool_stability_check",
                    "status": "success",
                    "cluster_id": str(cluster_state.get("cluster_id", "")),
                    "results": {
                        "summary": "fake stability evidence",
                        "metrics": {
                            "set_reliability_global_consensus": {
                                "total_candidate_set_n": 2
                            },
                            "set_reliability_set_consensus": {
                                "C0001": {"set_n": 2, "within_consensus_mean": 0.9}
                            },
                        },
                        "evidence_hints": [{"dimension": "set_reliability"}],
                        "warnings": [],
                    },
                    "artifacts": {},
                    "errors": [],
                }

            fake_module = type("FakeModule", (), {"fake_tool": staticmethod(fake_tool)})
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B"],
                "review_round": 1,
                "next_action": "call_tools",
                "tool_plan": [{"tool_name": "tool_stability_check"}],
                "budget_state": {},
                "evidence_matrix": {},
            }

            with patch(
                "utils.subtype_review_runtime.subtype_review_tool_imports",
                return_value={"tool_stability_check": ("fake_module", "fake_tool")},
            ), patch(
                "utils.subtype_review_runtime.importlib.import_module",
                return_value=fake_module,
            ):
                updated = revision_engine(
                    cluster,
                    {"A": {"case_id": "A"}, "B": {"case_id": "B"}},
                    str(Path(temp_dir)),
                    str(config_dir),
                    [],
                )

            round_dir = Path(temp_dir) / "subtype_review" / "C0001" / "round_1"
            self.assertTrue((round_dir / "tool_results.json").exists())
            self.assertTrue((round_dir / "evidence_matrix.json").exists())
            matrix = json.loads((round_dir / "evidence_matrix.json").read_text())
            self.assertEqual(
                matrix["set_reliability"]["metrics"]["set_reliability_set_consensus"]["C0001"]["within_consensus_mean"],
                0.9,
            )
            self.assertIn("round_1_evidence_matrix", updated["artifacts"])

    def test_disallowed_tool_plan_drops_without_not_implemented_result(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = write_budget(Path(temp_dir), allowed_tools=["tool_stability_check"])
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B"],
                "llm_audit": {
                    "recommended_action": "",
                    "tools_to_call": [{"tool_name": "tool_not_allowed"}],
                },
                "budget_state": {},
            }

            self.assertEqual(choose_router_action(cluster, str(config_dir)), "drop")
            self.assertEqual(cluster["drop_reason"], "invalid_tool_plan")

    def test_split_revision_generates_all_child_clusters(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = write_budget(Path(temp_dir))
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B", "C", "D"],
                "parent_cluster_ids": ["C0001"],
                "next_action": "split",
                "llm_audit": {
                    "split_plan": {
                        "reason": "internal structure",
                        "new_clusters": [
                            {"cluster_id": "C0001_S1", "member_ids": ["A", "B"]},
                            {"cluster_id": "C0001_S2", "member_ids": ["C", "D"]},
                        ],
                    }
                },
                "validation_results": {"old": {"status": "success"}},
                "structured_evidence": [{"evidence_id": "old"}],
                "budget_state": {"review_rounds_used": 1},
            }

            updated = revision_engine(cluster, {}, str(Path(temp_dir)), str(config_dir), [])

            children = updated["generated_clusters"]
            self.assertEqual([child["cluster_id"] for child in children], ["C0001_S1", "C0001_S2"])
            self.assertEqual([child["member_ids"] for child in children], [["A", "B"], ["C", "D"]])
            self.assertEqual(children[0]["status"], "under_review")
            self.assertEqual(children[0]["validation_results"], {})
            self.assertEqual(children[0]["structured_evidence"], [])
            self.assertTrue(children[0]["previous_evidence"])
            self.assertEqual(children[0]["revision_history"][0]["action"], "split")

    def test_revision_budget_exhaustion_forces_drop(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = write_budget(Path(temp_dir), max_revisions=1)
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A", "B"],
                "next_action": "split",
                "llm_audit": {"split_plan": {"member_ids": ["A"]}},
                "budget_state": {"split_plans_used": 1, "merge_plans_used": 0},
            }

            updated = revision_engine(cluster, {}, str(Path(temp_dir)), str(config_dir), [])

            self.assertEqual(updated["final_action"], "drop")
            self.assertEqual(updated["status"], "drop")
            self.assertEqual(updated["drop_reason"], "budget_exhausted")
            self.assertTrue(Path(updated["artifacts"]["report"]).exists())

    def test_merge_revision_generates_merged_cluster_and_absorbed_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = write_budget(Path(temp_dir))
            cluster = {
                "cluster_id": "C0001",
                "member_ids": ["A"],
                "parent_cluster_ids": ["C0001"],
                "next_action": "merge",
                "llm_audit": {
                    "merge_plan": {
                        "target_cluster_ids": ["C0002"],
                        "reason": "similar evidence",
                    }
                },
                "budget_state": {},
            }
            all_clusters = [{"cluster_id": "C0002", "member_ids": ["B", "C"]}]

            updated = revision_engine(cluster, {}, str(Path(temp_dir)), str(config_dir), all_clusters)

            self.assertEqual(updated["absorbed_cluster_ids"], ["C0002"])
            merged = updated["generated_clusters"][0]
            self.assertEqual(merged["member_ids"], ["A", "B", "C"])
            self.assertEqual(merged["parent_cluster_ids"], ["C0001", "C0002"])
            self.assertEqual(merged["revision_history"][0]["action"], "merge")

    def test_subtype_review_agent_reviews_split_children(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            graph = FakeSubtypeReviewGraph(
                {
                    "C0001": {
                        "cluster_id": "C0001",
                        "member_ids": ["A", "B", "C", "D"],
                        "generated_clusters": [
                            {
                                "cluster_id": "C0001_S1",
                                "member_ids": ["A", "B"],
                                "parent_cluster_ids": ["C0001"],
                                "status": "under_review",
                            },
                            {
                                "cluster_id": "C0001_S2",
                                "member_ids": ["C", "D"],
                                "parent_cluster_ids": ["C0001"],
                                "status": "under_review",
                            },
                        ],
                        "status": "under_review",
                    },
                    "C0001_S1": {
                        "cluster_id": "C0001_S1",
                        "member_ids": ["A", "B"],
                        "parent_cluster_ids": ["C0001"],
                        "final_action": "accept",
                        "final_decision": "accept",
                        "status": "accept",
                    },
                    "C0001_S2": {
                        "cluster_id": "C0001_S2",
                        "member_ids": ["C", "D"],
                        "parent_cluster_ids": ["C0001"],
                        "final_action": "drop",
                        "final_decision": "drop",
                        "status": "drop",
                        "drop_reason": "weak_evidence",
                    },
                }
            )
            inventory = {
                "patient_states_by_id": {"A": {"case_id": "A"}, "B": {"case_id": "B"}, "C": {"case_id": "C"}, "D": {"case_id": "D"}},
                "candidate_clusters": [{"cluster_id": "C0001", "member_ids": ["A", "B", "C", "D"]}],
                "candidate_clusters_path": "candidate_clusters.json",
            }

            result = pipeline_main.subtype_review_agent(
                {"inventory": inventory},
                graph,
                str(root / "output"),
                str(root / "configs"),
            )

            cluster_states = result["inventory"]["cluster_states"]
            self.assertEqual(graph.invoked_cluster_ids, ["C0001", "C0001_S1", "C0001_S2"])
            self.assertEqual(
                [state["cluster_id"] for state in cluster_states],
                ["C0001", "C0001_S1", "C0001_S2"],
            )

    def test_subtype_review_agent_marks_merge_target_absorbed_and_skips_it(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            graph = FakeSubtypeReviewGraph(
                {
                    "C0001": {
                        "cluster_id": "C0001",
                        "member_ids": ["A"],
                        "generated_clusters": [
                            {
                                "cluster_id": "C0001_M1",
                                "member_ids": ["A", "B"],
                                "parent_cluster_ids": ["C0001", "C0002"],
                                "absorbed_cluster_ids": ["C0002"],
                                "status": "under_review",
                            }
                        ],
                        "absorbed_cluster_ids": ["C0002"],
                        "status": "under_review",
                    },
                    "C0001_M1": {
                        "cluster_id": "C0001_M1",
                        "member_ids": ["A", "B"],
                        "parent_cluster_ids": ["C0001", "C0002"],
                        "final_action": "accept",
                        "final_decision": "accept",
                        "status": "accept",
                    },
                }
            )
            inventory = {
                "patient_states_by_id": {"A": {"case_id": "A"}, "B": {"case_id": "B"}},
                "candidate_clusters": [
                    {"cluster_id": "C0001", "member_ids": ["A"]},
                    {"cluster_id": "C0002", "member_ids": ["B"]},
                ],
                "candidate_clusters_path": "candidate_clusters.json",
            }

            result = pipeline_main.subtype_review_agent(
                {"inventory": inventory},
                graph,
                str(root / "output"),
                str(root / "configs"),
            )

            cluster_states = result["inventory"]["cluster_states"]
            self.assertEqual(graph.invoked_cluster_ids, ["C0001", "C0001_M1"])
            self.assertEqual([state["cluster_id"] for state in cluster_states], ["C0001", "C0001_M1", "C0002"])
            absorbed = cluster_states[1]
            absorbed = cluster_states[2]
            self.assertEqual(absorbed["final_action"], "merge")
            self.assertEqual(absorbed["absorbed_into"], "C0001_M1")


if __name__ == "__main__":
    unittest.main()
