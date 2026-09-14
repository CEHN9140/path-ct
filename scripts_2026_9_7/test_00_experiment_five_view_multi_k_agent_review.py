import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("00_experiment_five_view_multi_k_agent_review.py")
SPEC = importlib.util.spec_from_file_location("five_view_multi_k_agent_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_main_inputs_requires_canonical_five_views(tmp_path):
    candidate_dir = tmp_path / "candidate_subtype"
    candidate_dir.mkdir()
    (tmp_path / "wxs").mkdir()
    patient_ids = ["P1", "P2"]
    paths = {}
    for name in MODULE.VIEWS:
        root = candidate_dir if name in {"ct", "wsi", "rna"} else tmp_path / "wxs"
        path = root / f"{name}.npy"
        np.save(path, np.eye(2))
        paths[name] = str(path)
    np.save(candidate_dir / "fused_similarity.npy", np.eye(2))
    (candidate_dir / "affinity_patient_order.json").write_text(
        json.dumps(patient_ids), encoding="utf-8"
    )
    (candidate_dir / "affinity_cache.json").write_text(
        json.dumps({"patient_ids": patient_ids, "paths": paths}), encoding="utf-8"
    )

    loaded_ids, matrices, fused = MODULE.load_main_inputs(tmp_path)
    assert loaded_ids == patient_ids
    assert set(matrices) == set(MODULE.VIEWS)
    assert fused.shape == (2, 2)


def test_load_main_inputs_ignores_stale_absolute_cache_paths(tmp_path):
    current = tmp_path / "output_kirc"
    candidate_dir = current / "candidate_subtype"
    wxs_dir = current / "wxs"
    stale = tmp_path / "old_output"
    candidate_dir.mkdir(parents=True)
    wxs_dir.mkdir()
    stale.mkdir()
    paths = {}
    for index, name in enumerate(MODULE.VIEWS, 1):
        filename = f"{name}_affinity.npy"
        local = candidate_dir / filename if name in {"ct", "wsi", "rna"} else wxs_dir / filename
        np.save(local, np.full((2, 2), index, dtype=float))
        stale_path = stale / filename
        np.save(stale_path, np.full((2, 2), -index, dtype=float))
        paths[name] = str(stale_path)
    np.save(candidate_dir / "fused_similarity.npy", np.eye(2))
    (candidate_dir / "affinity_patient_order.json").write_text(
        json.dumps(["P1", "P2"]), encoding="utf-8"
    )
    (candidate_dir / "affinity_cache.json").write_text(
        json.dumps({"patient_ids": ["P1", "P2"], "paths": paths}), encoding="utf-8"
    )

    _, matrices, _ = MODULE.load_main_inputs(current)

    assert all(np.all(matrices[name] == index) for index, name in enumerate(MODULE.VIEWS, 1))


def test_load_initial_partition_uses_main_saved_k_partition(tmp_path):
    candidate_dir = tmp_path / "candidate_subtype" / "consensus_cluster"
    candidate_dir.mkdir(parents=True)
    patient_ids = ["P1", "P2", "P3", "P4"]
    payload = {
        "n_clusters": 2,
        "labels": {"P1": 0, "P2": 0, "P3": 1, "P4": 1},
    }
    (candidate_dir / "consensus_hierarchical_K2.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    groups = MODULE.load_initial_partition(tmp_path, 2, patient_ids)
    assert [group["member_ids"] for group in groups] == [["P1", "P2"], ["P3", "P4"]]
    assert all(group["source_views"] == list(MODULE.VIEWS) for group in groups)


def test_cache_reuse_requires_complete_status_and_matching_identity():
    identity = {
        "review_signature": "review",
        "fused_similarity_sha256": "fused",
        "patient_order_sha256": "order",
        "scientific_input_sha256": "input",
        "initial_partition_sha256": "partition",
        "git_commit_sha": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    complete = {
        "status": "review_complete",
        "raw_control_status": "complete",
        "partition": {"sets": [{"set_id": "C1", "member_ids": ["P1", "P2"]}]},
    }
    metadata = {
        **identity,
        "status": "review_complete",
        "final_partition_signature": MODULE.partition_signature(
            complete["partition"]["sets"]
        ),
    }
    assert MODULE.cache_reusable(complete, metadata, identity)
    assert not MODULE.cache_reusable(
        {"status": "review_incomplete_due_to_round_budget", "raw_control_status": "review_incomplete_due_to_round_budget"},
        metadata,
        identity,
    )
    changed = {**metadata, "review_signature": "old"}
    assert not MODULE.cache_reusable(complete, changed, identity)
    changed = {**metadata, "final_partition_signature": "wrong"}
    assert not MODULE.cache_reusable(complete, changed, identity)


def test_source_identity_records_commit_cleanliness_and_source_hash():
    identity = MODULE.source_identity(MODULE.ROOT)
    assert len(identity["git_commit_sha"]) == 40
    assert isinstance(identity["git_worktree_clean"], bool)
    assert len(identity["source_tree_sha256"]) == 64


def test_scientific_input_identity_covers_canonical_inputs(tmp_path):
    candidate = tmp_path / "candidate_subtype"
    wxs = tmp_path / "wxs"
    states = tmp_path / "storage" / "patient_states"
    cnv = tmp_path / "cnv"
    ct_qc = tmp_path / "ct_qc" / "P1" / "dcm2nii"
    candidate.mkdir(parents=True)
    wxs.mkdir()
    states.mkdir(parents=True)
    cnv.mkdir()
    ct_qc.mkdir(parents=True)
    np.save(candidate / "ct_affinity.npy", np.eye(2))
    np.save(candidate / "wsi_affinity.npy", np.eye(2))
    np.save(candidate / "rna_affinity.npy", np.eye(2))
    np.save(candidate / "fused_similarity.npy", np.eye(2))
    np.save(wxs / "wxs_affinity.npy", np.eye(2))
    np.save(wxs / "cnv_affinity.npy", np.eye(2))
    (states / "patient_states.jsonl").write_text('{"case_id":"P1"}\n', encoding="utf-8")
    (cnv / "case_features.csv").write_text("case_id,x\nP1,1\n", encoding="utf-8")
    (ct_qc.parent / "selection_summary.json").write_text("{}", encoding="utf-8")
    (ct_qc / "CT1.json").write_text("{}", encoding="utf-8")

    first = MODULE.core_analysis.scientific_input_identity(tmp_path)
    assert len(first["scientific_input_sha256"]) == 64
    np.save(wxs / "wxs_affinity.npy", np.ones((2, 2)))
    second = MODULE.core_analysis.scientific_input_identity(tmp_path)
    assert first["scientific_input_sha256"] != second["scientific_input_sha256"]
    (cnv / "case_features.csv").write_text("case_id,x\nP1,2\n", encoding="utf-8")
    assert MODULE.core_analysis.scientific_input_identity(tmp_path)["scientific_input_sha256"] != second["scientific_input_sha256"]
    (ct_qc / "CT1.json").write_text('{"slice_thickness":2}', encoding="utf-8")
    third = MODULE.core_analysis.scientific_input_identity(tmp_path)
    assert third["scientific_input_sha256"] != second["scientific_input_sha256"]


def test_stable_core_analysis_uses_full_multi_k_universe(monkeypatch, tmp_path):
    captured = {}

    def analyze(root, patient_ids, initial_ks, repeats, min_core_size):
        captured.update({
            "root": root, "patient_ids": patient_ids, "initial_ks": initial_ks,
            "repeats": repeats, "min_core_size": min_core_size,
        })
        return {"analysis_status": "complete"}

    monkeypatch.setattr(MODULE.stability, "analyze", analyze)
    result = MODULE.analyze_stable_cores(tmp_path, ["P1", "P2"], 21)

    assert result["analysis_status"] == "complete"
    assert captured == {
        "root": tmp_path,
        "patient_ids": ["P1", "P2"],
        "initial_ks": MODULE.INITIAL_KS,
        "repeats": MODULE.REPEATS,
        "min_core_size": 5,
    }


def test_stable_core_analysis_waits_for_all_current_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MODULE.stability, "analyze",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = MODULE.analyze_stable_cores(tmp_path, ["P1", "P2"], 3)

    assert result == {
        "analysis_status": "pending",
        "valid_run_count": 3,
        "expected_run_count": 21,
    }


def test_stable_core_analysis_honors_supplied_universe(monkeypatch, tmp_path):
    captured = {}

    def analyze(root, patient_ids, initial_ks, repeats, min_core_size):
        captured.update({"initial_ks": initial_ks, "repeats": repeats})
        return {"analysis_status": "complete"}

    monkeypatch.setattr(MODULE.stability, "analyze", analyze)
    MODULE.analyze_stable_cores(tmp_path, ["P1"], 1, (2,), (1,))
    assert captured == {"initial_ks": (2,), "repeats": (1,)}


def test_leave_one_k_out_uses_each_reduced_k_universe(monkeypatch, tmp_path):
    for repeat in MODULE.REPEATS:
        for initial_k in MODULE.INITIAL_KS:
            path = tmp_path / f"run{repeat}" / f"K{initial_k}"
            path.mkdir(parents=True)
    calls = []

    def analyze(root, patient_ids, initial_ks, repeats, min_core_size):
        calls.append(tuple(initial_ks))
        return {"analysis_status": "complete", "valid_run_count": 18,
                "primary_core_count": 1, "primary_core_patient_count": 5,
                "primary_cores": []}

    monkeypatch.setattr(MODULE.stability, "analyze", analyze)
    MODULE.leave_one_k_out(tmp_path, ["P1"], MODULE.INITIAL_KS, MODULE.REPEATS, [])
    assert calls == [tuple(k for k in MODULE.INITIAL_KS if k != excluded) for excluded in MODULE.INITIAL_KS]


def test_matched_core_jaccard_uses_one_to_one_matching():
    reference = [{"member_ids": '["a", "b"]'}, {"member_ids": '["c", "d"]'}]
    candidate = [{"member_ids": '["a", "b"]'}, {"member_ids": '["c", "x"]'}]
    mean, minimum = MODULE.matched_core_jaccard(reference, candidate)
    assert mean == 2 / 3
    assert minimum == 1 / 3


def test_matched_core_jaccard_penalizes_unmatched_cores():
    reference = [{"member_ids": '["a"]'}, {"member_ids": '["b"]'}]
    candidate = [{"member_ids": '["a"]'}]
    mean, minimum = MODULE.matched_core_jaccard(reference, candidate)
    assert mean == 0.5
    assert minimum == 0.0
