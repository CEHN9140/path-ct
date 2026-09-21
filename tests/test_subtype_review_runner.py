from __future__ import annotations

import json
from pathlib import Path

import agents.subtype_review.runner as runner


def test_review_budget_allows_pair_union_review_and_reassessment():
    config = runner.load_yaml_file("configs/subtype_review.yaml")
    assert config["budget"]["max_rounds"] == 14


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


def test_active_tool_registry_hides_disabled_omics_tools():
    assert "pathway_enrichment" not in runner.active_tool_registry(("ct", "wsi", "wxs", "cnv"))
    assert "mutation_enrichment" not in runner.active_tool_registry(("ct", "wsi", "rna", "cnv"))
    assert "cnv_characterization" not in runner.active_tool_registry(("ct", "wsi", "rna", "wxs"))
    assert "multimodal_consistency_check" in runner.active_tool_registry(("ct", "wsi", "wxs", "cnv"))


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
