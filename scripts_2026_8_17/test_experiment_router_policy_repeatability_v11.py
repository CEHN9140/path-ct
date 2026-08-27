from __future__ import annotations

import json

from scripts_2026_8_17 import experiment_router_policy_repeatability_v11 as experiment
from scripts_2026_8_17 import experiment_multi_k_accepted_core_stability as multi_k


def test_multi_k_default_output_is_v11_isolated_from_v10():
    assert multi_k.DEFAULT_EXPERIMENT_ROOT.parts[-2:] == (
        "output_kirc_v11",
        "experiment_multi_k_accepted_core_stability",
    )


def test_v11_router_calibration_repeats_saved_evidence_without_v10_labels(tmp_path, monkeypatch):
    for repeat in (1, 2, 3):
        run_root = tmp_path / f"run{repeat}" / "K2"
        run_root.mkdir(parents=True)
        (run_root / "review_history.json").write_text(
            json.dumps([{"repeat": repeat, "partition": {"sets": [{"set_id": "C1"}]}}]),
            encoding="utf-8",
        )

    monkeypatch.setattr(experiment, "load_yaml_file", lambda path: {"llm": {"model_name": "test", "temperature": 0}})
    monkeypatch.setattr(experiment, "review_signature_manifest", lambda config, config_dir: {
        "verifier_prompt_sha256": "v",
        "router_prompt_sha256": "r",
        "reviser_prompt_sha256": "x",
        "review_signature": "sig",
    })
    monkeypatch.setattr(experiment, "build_default_router", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        experiment,
        "replay_entry",
        lambda entry, router: ({"evidence": "saved"}, {
            "actions": [{"action": "accept" if entry["repeat"] != 2 else "drop", "target_ids": ["C1"]}]
        }),
    )

    summary = experiment.calibrate(tmp_path, tmp_path / "configs", (2,), (1, 2, 3), tmp_path / "calibration")

    assert summary["run_count"] == 3
    assert summary["successful_run_count"] == 3
    assert summary["discordant_set_count"] == 1
    assert summary["partition_plan_exact_match_fraction"] == 1 / 3
    row = summary["set_rows"][0]
    assert row["repeat1_action"] == "accept"
    assert row["repeat2_action"] == "drop"
    assert row["repeat3_action"] == "accept"
    assert row["v10_action_used_as_ground_truth"] is False
    assert (tmp_path / "calibration" / "prompt_manifest.json").exists()
