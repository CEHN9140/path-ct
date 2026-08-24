from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.subtype_review.graph import initial_review_state
from agents.subtype_review.llm import (
    LLMUsageTracker,
    build_default_router,
)
from agents.subtype_review.tools import TOOL_REGISTRY
from scripts_2026_8_17.experiment_initial_k_review_sensitivity import (
    labels_to_candidate_sets,
    load_patient_states,
    build_parser,
    summarize_run,
)
import scripts_2026_8_17.experiment_initial_k_review_sensitivity as sensitivity


def make_labels(k: int, count: int = 102) -> dict[str, str]:
    return {
        f"P{i:03d}": str(i % k)
        for i in range(count)
    }


def test_labels_to_candidate_sets_validates_complete_partition():
    sets = labels_to_candidate_sets({"labels": make_labels(3)}, initial_k=3)

    assert [item["cluster_id"] for item in sets] == ["C0001", "C0002", "C0003"]
    assert [len(item["member_ids"]) for item in sets] == [34, 34, 34]
    members = [member for item in sets for member in item["member_ids"]]
    assert len(members) == 102
    assert len(set(members)) == 102


def test_labels_to_candidate_sets_rejects_wrong_group_count():
    with pytest.raises(ValueError, match="groups instead of K=3"):
        labels_to_candidate_sets({"labels": make_labels(2)}, initial_k=3)


def test_fresh_review_states_do_not_share_membership_or_control():
    k2 = initial_review_state(labels_to_candidate_sets({"labels": make_labels(2)}, 2))
    k3 = initial_review_state(labels_to_candidate_sets({"labels": make_labels(3)}, 3))

    k2["partition"]["sets"][0]["member_ids"].append("local")
    assert all("local" not in item["member_ids"] for item in k3["partition"]["sets"])
    assert k3["control"]["trace"] == []


def test_usage_tracker_counts_unbounded_requests():
    tracker = LLMUsageTracker()
    for _ in range(13):
        tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}})
    assert tracker.snapshot() == {
        "api_calls": 13,
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_usage_tracker_reads_langchain_token_metadata():
    tracker = LLMUsageTracker()
    tracker.before_request()
    tracker.record_response({
        "usage_metadata": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    })
    assert tracker.snapshot() == {
        "api_calls": 1,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


def test_unbounded_tracker_only_records_calls():
    tracker = LLMUsageTracker()
    for _ in range(101):
        tracker.before_request()
    assert tracker.snapshot()["api_calls"] == 101


def test_tool_registry_keeps_real_implementations():
    assert TOOL_REGISTRY["multimodal_consistency_check"]["function"]
    assert TOOL_REGISTRY["known_label_echo_test"]["scope"] == "partition"


def test_sensitivity_exposes_no_agent_protocol_overrides():
    args = build_parser().parse_args([])
    assert not hasattr(args, "max_rounds")


def test_router_default_does_not_force_self_review(tmp_path):
    config = {
        "llm": {
            "structured_output": "json_object",
            "model_name": "test",
            "base_url": "http://localhost",
            "api_key_env": "TEST_KEY",
        },
        "prompt_dir": "agents/subtype_review/prompts",
    }
    default = build_default_router(config, "/data/qijun/path-ct/configs")
    assert not hasattr(default, "draft_model")


def test_load_patient_states_reads_upstream_root_only(tmp_path):
    source = tmp_path / "output_kirc"
    path = source / "storage" / "patient_states" / "patient_states.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"case_id": "P001"}) + "\n", encoding="utf-8")

    states = load_patient_states(source)

    assert states == {"P001": {"case_id": "P001"}}


def test_sensitivity_run_uses_shared_review_without_agent_overrides(tmp_path, monkeypatch):
    initial_sets = [{"cluster_id": "C0001", "member_ids": ["P1", "P2"]}]
    captured = {}
    monkeypatch.setattr(sensitivity, "load_patient_states", lambda root: {"P1": {}, "P2": {}})
    monkeypatch.setattr(sensitivity, "load_affinity_patient_ids", lambda root: {"P1", "P2"})
    monkeypatch.setattr(
        sensitivity,
        "load_initial_partition",
        lambda root, k, expected_patient_ids=None: (root / "K3.json", initial_sets),
    )

    def fake_review(*args, **kwargs):
        captured.update({"args": args, "kwargs": kwargs})
        state = initial_review_state(initial_sets)
        state["control"].update({"status": "complete", "max_rounds": 10})
        return state

    monkeypatch.setattr(sensitivity, "run_subtype_review", fake_review)
    monkeypatch.setattr(sensitivity, "save_review_outputs", lambda *args, **kwargs: {})
    review_root = tmp_path / "results"

    sensitivity.run_one(3, tmp_path / "data", review_root, tmp_path / "configs")

    assert captured["args"] == (
        initial_sets,
        {"P1": {}, "P2": {}},
        str(tmp_path / "data"),
        str(tmp_path / "configs"),
    )
    assert captured["kwargs"] == {
        "artifact_root": str(review_root / "K3"),
    }
    metadata = json.loads((review_root / "K3" / "run_metadata.json").read_text())
    assert metadata["max_rounds"] == 10


def test_summarize_run_contains_required_sensitivity_fields():
    summary = summarize_run(
        initial_k=3,
        initial_sets=[{"member_ids": ["P1", "P2"]}],
        state={
            "partition": {"sets": [{"set_id": "C1", "member_ids": ["P1", "P2"]}]},
            "router_plan": None,
            "control": {"status": "review_incomplete_due_to_round_budget", "round": 4},
        },
        usage={"api_calls": 60, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    )
    assert summary["initial_k"] == 3
    assert summary["terminal_status"] == "review_incomplete_due_to_round_budget"
    assert summary["api_calls"] == 60
