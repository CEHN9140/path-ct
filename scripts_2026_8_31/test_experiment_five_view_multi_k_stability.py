import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("15_experiment_five_view_multi_k_stability.py")
SPEC = importlib.util.spec_from_file_location("five_view_multi_k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repeat_selection_does_not_append_to_default_repeats():
    assert MODULE.parse_repeats(None) == (1, 2, 3)
    assert MODULE.parse_repeats([1, 2]) == (1, 2)


def test_initial_k_selection_supports_pilot_values():
    assert MODULE.parse_initial_ks(None) == tuple(range(2, 9))
    assert MODULE.parse_initial_ks([8, 2, 4, 4]) == (2, 4, 8)


def test_loader_reads_canonical_five_view_artifacts(tmp_path):
    candidate_dir = tmp_path / "candidate_subtype"
    wxs_dir = tmp_path / "wxs"
    candidate_dir.mkdir()
    wxs_dir.mkdir()
    patient_ids = ["P1", "P2"]
    paths = {}
    for name in ("ct", "wsi", "rna"):
        path = candidate_dir / f"{name}_affinity.npy"
        np.save(path, np.eye(2))
        paths[name] = str(path)
    for name in ("wxs", "cnv"):
        path = wxs_dir / f"{name}_affinity.npy"
        np.save(path, np.eye(2))
        paths[name] = str(path)
    fused_path = candidate_dir / "fused_similarity.npy"
    np.save(fused_path, np.eye(2))
    (candidate_dir / "affinity_patient_order.json").write_text(
        json.dumps(patient_ids), encoding="utf-8"
    )
    (candidate_dir / "affinity_cache.json").write_text(
        json.dumps({"patient_ids": patient_ids, "paths": paths}), encoding="utf-8"
    )

    loaded_ids, matrices, fused, _ = MODULE.load_five_view_inputs(
        tmp_path, MODULE.ROOT / "configs"
    )
    assert loaded_ids == patient_ids
    assert set(matrices) == {"ct", "wsi", "rna", "wxs", "cnv"}
    assert fused.shape == (2, 2)


def test_loader_resolves_stale_recorded_paths_from_data_root(tmp_path):
    candidate_dir = tmp_path / "candidate_subtype"
    wxs_dir = tmp_path / "wxs"
    stale_dir = tmp_path / "old_output"
    candidate_dir.mkdir()
    wxs_dir.mkdir()
    stale_dir.mkdir()
    patient_ids = ["P1", "P2"]
    paths = {}
    for name in ("ct", "wsi", "rna"):
        path = candidate_dir / f"{name}_affinity.npy"
        np.save(path, np.eye(2))
        stale_path = stale_dir / path.name
        np.save(stale_path, 2 * np.eye(2))
        paths[name] = str(stale_path)
    for name in ("wxs", "cnv"):
        path = wxs_dir / f"{name}_affinity.npy"
        np.save(path, np.eye(2))
        stale_path = stale_dir / path.name
        np.save(stale_path, 2 * np.eye(2))
        paths[name] = str(stale_path)
    np.save(candidate_dir / "fused_similarity.npy", np.eye(2))
    (candidate_dir / "affinity_patient_order.json").write_text(
        json.dumps(patient_ids), encoding="utf-8"
    )
    (candidate_dir / "affinity_cache.json").write_text(
        json.dumps({"patient_ids": patient_ids, "paths": paths}), encoding="utf-8"
    )

    loaded_ids, matrices, fused, _ = MODULE.load_five_view_inputs(
        tmp_path, MODULE.ROOT / "configs"
    )
    assert loaded_ids == patient_ids
    assert set(matrices) == {"ct", "wsi", "rna", "wxs", "cnv"}
    assert all(np.array_equal(matrix, np.eye(2)) for matrix in matrices.values())
    assert fused.shape == (2, 2)
