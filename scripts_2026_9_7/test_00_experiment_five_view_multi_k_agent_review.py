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
    patient_ids = ["P1", "P2"]
    paths = {}
    for name in MODULE.VIEWS:
        path = candidate_dir / f"{name}.npy"
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
