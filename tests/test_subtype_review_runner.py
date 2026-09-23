from __future__ import annotations

import json
from pathlib import Path

import agents.subtype_review.runner as runner


def test_review_budget_matches_configured_round_cap():
    config = runner.load_yaml_file("configs/subtype_review.yaml")
    assert config["budget"]["max_rounds"] == 20
    assert all(config["llm"][key] == 3 for key in (
        "structured_output_retries", "tool_selection_retries",
        "verifier_audit_coverage_retries", "router_plan_validation_retries",
        "reviser_plan_validation_retries",
    ))


def test_runner_only_reads_workflow_budget(monkeypatch):
    config = {"budget": {"max_rounds": 10}, "llm": {}}
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
        "/data/output/run",
    )
    assert state["control"]["max_rounds"] == 10
    assert {name for name, _ in calls} == {"verifier", "router", "reviser"}
    assert len({id(tracker) for _, tracker in calls}) == 1


def test_main_uses_multi_k_grid_from_config_when_cli_grid_is_omitted(tmp_path, monkeypatch):
    import main

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        "multi_k:\n  initial_ks: [3, 4]\n  repeats: [2]\n", encoding="utf-8"
    )
    data_path = tmp_path / "cases.json"
    data_path.write_text("[]", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(main, "run_pipeline", lambda args, cases: captured.update(vars(args)) or {})
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--data-json-path", str(data_path), "--config-dir", str(config_dir)],
    )

    main.main()

    assert captured["initial_ks"] == [3, 4]
    assert captured["repeats"] == [2]

    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py", "--data-json-path", str(data_path), "--config-dir", str(config_dir),
            "--initial-k", "8", "--repeat", "3",
        ],
    )
    main.main()
    assert captured["initial_ks"] == [8]
    assert captured["repeats"] == [3]


def test_main_accepts_explicit_run_pairs_and_rejects_mixed_grid_flags(tmp_path, monkeypatch):
    import main

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        "multi_k:\n  initial_ks: [2, 3]\n  repeats: [1, 2]\n", encoding="utf-8"
    )
    data_path = tmp_path / "cases.json"
    data_path.write_text("[]", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(main, "run_pipeline", lambda args, cases: captured.update(vars(args)) or {})
    monkeypatch.setattr(
        "sys.argv", ["main.py", "--data-json-path", str(data_path),
        "--config-dir", str(config_dir), "--run", "3:2", "--run", "2:1"]
    )
    main.main()
    assert captured["run_pairs"] == [(3, 2), (2, 1)]

    monkeypatch.setattr(
        "sys.argv", ["main.py", "--data-json-path", str(data_path),
        "--config-dir", str(config_dir), "--run", "3:2", "--repeat", "2"]
    )
    try:
        main.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("mixed --run and --repeat flags must be rejected")
