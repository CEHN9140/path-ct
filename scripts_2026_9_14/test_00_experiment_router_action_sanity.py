import importlib.util
from pathlib import Path

from agents.subtype_review.graph import validate_router_plan
from agents.subtype_review.llm import parse_router_plan


SCRIPT = Path(__file__).with_name("00_experiment_router_action_sanity.py")
SPEC = importlib.util.spec_from_file_location("router_action_sanity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_controlled_cases_have_expected_partition_shapes():
    assert MODULE.CASES.keys() == {"split", "merge", "drop"}
    assert [len(MODULE.build_case(name)["partition"]["sets"]) for name in MODULE.CASES] == [2, 2, 1]


def test_split_and_merge_plans_pass_existing_structural_guards():
    for name, raw_plan in {
        "split": {
            "actions": [
                {"action": "split", "target_ids": ["C1"], "decision_state": MODULE.assessed_state(structure="incompatible"), "reason": "stable internal subdivision"},
                {"action": "accept", "target_ids": ["C2"], "decision_state": MODULE.assessed_state(), "reason": "retained control set"},
            ],
            "evidence_requests": [],
        },
        "merge": {
            "actions": [{"action": "merge", "target_ids": ["C1", "C2"], "decision_state": MODULE.assessed_state(structure="incompatible"), "reason": "weak pairwise boundary"}],
            "evidence_requests": [],
        },
    }.items():
        plan = parse_router_plan(raw_plan)
        state = MODULE.build_case(name)
        validate_router_plan(plan, state, {"tool_registry": MODULE.TOOL_REGISTRY})


def test_drop_plan_is_a_valid_terminal_action():
    plan = parse_router_plan({
        "actions": [{"action": "drop", "target_ids": ["C1"], "decision_state": MODULE.assessed_state(identity="unsupported", alternative_explanation="concerning"), "reason": "no defensible identity"}],
        "evidence_requests": [],
    })
    validate_router_plan(plan, MODULE.build_case("drop"), {"tool_registry": MODULE.TOOL_REGISTRY})


def test_controlled_cases_keep_evidence_dimensions_separate():
    split = MODULE.build_case("split")["reports"]
    merge = MODULE.build_case("merge")["reports"]
    drop = MODULE.build_case("drop")["reports"]
    split_cross = next(row for row in split if row["dimension"] == "cross_modal_consistency")
    merge_biology = [row for row in merge if row["dimension"] == "biological_support"]
    drop_cross = next(row for row in drop if row["dimension"] == "cross_modal_consistency")
    drop_biology = next(row for row in drop if row["dimension"] == "biological_support")
    drop_confound = next(row for row in drop if row["dimension"] == "confounder_exclusion")

    assert "outer boundary remains acceptable" in split_cross["observations"][0]["finding"]
    assert all("coherent biological identity" in row["observations"][0]["finding"] for row in merge_biology)
    assert "lacks a defensible identity" not in drop_cross["observations"][0]["finding"]
    assert "substantial competing technical explanation" not in drop_cross["observations"][0]["finding"]
    assert "no coherent biological identity" in drop_biology["observations"][0]["finding"].lower()
    assert "substantial competing technical explanation" in drop_confound["observations"][0]["finding"]


def test_action_summary_preserves_merge_targets():
    plan = {"actions": [{"action": "merge", "target_ids": ["C2", "C1"]}]}
    assert MODULE.action_summary(plan) == {"C1+C2": "merge"}


def test_run_case_compares_a_single_router_plan_without_aggregation():
    class FakeRouter:
        def invoke(self, payload):
            assert payload["available_evidence_requests"] == []
            return {
                "actions": [
                    {"action": "split", "target_ids": ["C1"], "decision_state": MODULE.assessed_state(structure="incompatible"), "reason": "stable internal subdivision"},
                    {"action": "accept", "target_ids": ["C2"], "decision_state": MODULE.assessed_state(), "reason": "retained control set"},
                ],
                "evidence_requests": [],
            }

    result = MODULE.run_case("split", FakeRouter())
    assert result["status"] == "success"
    assert result["passed"] is True
    assert result["actual_actions"] == {"C1": "split", "C2": "accept"}
