from __future__ import annotations

import agents.subtype_review.runner as runner


def test_runner_only_reads_workflow_budget(monkeypatch):
    config = {"budget": {"max_rounds": 10, "max_failures": 3}}
    calls = []
    monkeypatch.setattr(runner, "load_yaml_file", lambda path: config)

    def builder(name):
        def build(config, config_dir, **kwargs):
            calls.append((name, kwargs["usage_tracker"]))
            return name
        return build

    monkeypatch.setattr(runner, "build_default_verifier", builder("verifier"))
    monkeypatch.setattr(runner, "build_default_reviser", builder("reviser"))
    monkeypatch.setattr(runner, "build_default_router", builder("router"))

    class Graph:
        def invoke(self, state, context):
            return state

    monkeypatch.setattr(runner, "build_review_graph", lambda: Graph())
    state = runner.run_subtype_review(
        [{"cluster_id": "C1", "member_ids": ["P1", "P2"]}],
        {"P1": {}, "P2": {}},
        "/data/output",
        "/data/configs",
    )
    assert state["control"]["max_rounds"] == 10
    assert state["control"]["max_failures"] == 3
    assert {name for name, _ in calls} == {"verifier", "router", "reviser"}
    assert len({id(tracker) for _, tracker in calls}) == 1
