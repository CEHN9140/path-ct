from __future__ import annotations

import json

from scripts_2026_8_31 import experiment_router_policy_repeatability_v11 as experiment
from scripts_2026_8_31 import experiment_multi_k_accepted_core_stability as multi_k


def test_multi_k_default_output_is_v11_isolated_from_v10():
    assert multi_k.DEFAULT_EXPERIMENT_ROOT.parts[-2:] == (
        "output_kirc_v12",
        "03_multi_k_accepted_core_stability_v11",
    )


def test_v11_router_calibration_reuses_one_canonical_payload_for_all_replays(tmp_path, monkeypatch):
    source_root = tmp_path / "run2" / "K2"
    source_root.mkdir(parents=True)
    (source_root / "review_history.json").write_text(
        json.dumps([{"partition": {"sets": [{"set_id": "C1", "member_ids": ["P1", "P2"]}]}}]),
        encoding="utf-8",
    )

    class Router:
        def __init__(self):
            self.payloads = []

        def invoke(self, payload):
            self.payloads.append(json.loads(json.dumps(payload, sort_keys=True)))
            return {"actions": [{"action": "accept", "target_ids": ["C1"], "evidence_requests": [], "reason": "consistent"}]}

    router = Router()
    monkeypatch.setattr(experiment, "load_yaml_file", lambda path: {"llm": {"model_name": "test", "temperature": 0}})
    monkeypatch.setattr(experiment, "review_signature_manifest", lambda config, config_dir: {
        "verifier_prompt_sha256": "v",
        "router_prompt_sha256": "r",
        "reviser_prompt_sha256": "x",
        "review_signature": "sig",
    })
    monkeypatch.setattr(experiment, "build_default_router", lambda *args, **kwargs: router)

    summary = experiment.calibrate(
        tmp_path,
        tmp_path / "configs",
        initial_ks=(2,),
        source_repeat=2,
        replay_count=3,
        output_root=tmp_path / "calibration",
    )

    assert len(router.payloads) == 3
    assert router.payloads[0] == router.payloads[1] == router.payloads[2]
    assert summary["successful_run_count"] == 3
    assert summary["discordant_set_count"] == 0
    assert summary["identical_payload_all_replays"] == {"K2": True}
    assert summary["partition_rows"][0]["invalid_for_repeatability"] is False
    assert summary["set_rows"][0]["invalid_for_repeatability"] is False
    assert summary["valid_discordant_set_count"] == 0
    assert summary["invalid_set_count"] == 0
    assert summary["source_repeat"] == 2
    assert summary["set_rows"][0]["replay1_action"] == "accept"
    assert summary["set_rows"][0]["replay3_action"] == "accept"
    assert summary["v10_actions_are_not_ground_truth"] is True
    assert (tmp_path / "calibration" / "router_repeatability.csv").exists()


def test_v11_calibration_marks_mutated_replay_payload_invalid(tmp_path, monkeypatch):
    source_root = tmp_path / "run2" / "K2"
    source_root.mkdir(parents=True)
    (source_root / "review_history.json").write_text(
        json.dumps([{"partition": {"sets": [{"set_id": "C1", "member_ids": ["P1", "P2"]}]}}]),
        encoding="utf-8",
    )

    class MutatingRouter:
        def invoke(self, payload):
            payload["mutated"] = True
            return {"actions": [{"action": "accept", "target_ids": ["C1"], "evidence_requests": [], "reason": "ok"}]}

    monkeypatch.setattr(experiment, "load_yaml_file", lambda path: {"llm": {}})
    monkeypatch.setattr(experiment, "review_signature_manifest", lambda config, config_dir: {})
    monkeypatch.setattr(experiment, "build_default_router", lambda *args, **kwargs: MutatingRouter())

    summary = experiment.calibrate(
        tmp_path,
        tmp_path / "configs",
        initial_ks=(2,),
        source_repeat=2,
        replay_count=2,
        output_root=tmp_path / "calibration",
    )

    assert summary["identical_payload_all_replays"] == {"K2": False}
    assert summary["invalid_set_count"] == 1
