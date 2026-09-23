from __future__ import annotations

from types import SimpleNamespace


def test_worker_count_does_not_change_radiomics_cache_config():
    from agents.evidence_builder import CT_RADIOMICS_RUNTIME_KEYS
    from utils.cache_utils import hash_payload, semantic_config

    config = {"num_workers": 1, "extractor": {"setting": {"binWidth": 25}}}
    more_workers = {**config, "num_workers": 4}
    changed_features = {"num_workers": 4, "extractor": {"setting": {"binWidth": 30}}}

    signature = lambda value: hash_payload(
        {"semantic_config": semantic_config(value, CT_RADIOMICS_RUNTIME_KEYS)}
    )

    assert signature(config) == signature(more_workers)
    assert signature(config) != signature(changed_features)


def test_main_uses_configured_process_workers_for_successful_cases(tmp_path, monkeypatch):
    import main

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "ct_radiomics.yaml").write_text("num_workers: 2\n", encoding="utf-8")
    submitted = []
    worker_counts = []

    class Completed:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class Executor:
        def __init__(self, *, max_workers, mp_context):
            worker_counts.append((max_workers, mp_context.get_start_method()))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, state, **kwargs):
            submitted.append(state["case_id"])
            return Completed(function(state, **kwargs))

    monkeypatch.setattr(main, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(main, "sync_case_membership", lambda *_args: None)
    monkeypatch.setattr(
        main,
        "inventory_case",
        lambda case: {"case_id": case["Case_ID"], "qc": case["qc"]},
    )
    monkeypatch.setattr(main, "ct_qc", lambda states, **_kwargs: states)
    monkeypatch.setattr(main, "wsi_qc", lambda states, **_kwargs: states)
    monkeypatch.setattr(main, "save_patient_states", lambda *_args: None)
    monkeypatch.setattr(
        main,
        "evidence_builder",
        lambda state, **_kwargs: {**state, "radiomics_done": True},
    )
    monkeypatch.setattr(main, "build_wsi_embeddings_cohort", lambda states, **_kwargs: states)
    monkeypatch.setattr(main, "build_evidence_states", lambda states, **_kwargs: states)
    monkeypatch.setattr(
        main,
        "candidate_proposer",
        lambda states, **_kwargs: {
            "candidate_partitions": {
                2: [{"set_id": "C1", "member_ids": ["A", "B"]}],
            },
            "candidate_signature": "candidate-signature",
            "patient_states_by_id": {state["case_id"]: state for state in states},
        },
    )
    monkeypatch.setattr(
        main,
        "run_review_grid",
        lambda *_args, **_kwargs: {
            "runs": [],
            "run_root": "runs",
            "input_signature": "input-signature",
            "requested_run_count": 1,
            "complete_run_count": 0,
            "failed_runs": [],
            "incomplete_runs": [],
        },
    )
    monkeypatch.setattr(main, "write_json", lambda *_args: None)

    result = main.run_pipeline(
        SimpleNamespace(
            output_root=str(tmp_path / "output"),
            config_dir=str(config_dir),
            initial_ks=(2,),
            repeats=(1,),
            force=False,
        ),
        [
            {"Case_ID": "A", "qc": "success"},
            {"Case_ID": "B", "qc": "failed"},
        ],
    )

    assert worker_counts == [(2, "spawn")]
    assert submitted == ["A"]
    assert result["agent_run_count"] == 0
