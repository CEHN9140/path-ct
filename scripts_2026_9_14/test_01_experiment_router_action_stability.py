import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("01_experiment_router_action_stability.py")
SPEC = importlib.util.spec_from_file_location("router_action_stability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stability_uses_five_independent_replays():
    assert MODULE.DEFAULT_REPEATS == 5


def test_replay_summary_counts_exact_actions():
    results = [
        {"status": "success", "passed": True, "actual_actions": {"C1": "split"}},
        {"status": "success", "passed": True, "actual_actions": {"C1": "split"}},
        {"status": "router_error", "passed": False, "actual_actions": {}},
    ]
    summary = MODULE.summarize_replays("split", results)
    assert summary["repeats"] == 3
    assert summary["successful_replays"] == 2
    assert summary["correct_replays"] == 2
    assert summary["router_errors"] == 1
    assert summary["action_consistency"] is False
