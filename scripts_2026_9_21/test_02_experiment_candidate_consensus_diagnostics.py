import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("02_experiment_candidate_consensus_diagnostics.py")
SPEC = importlib.util.spec_from_file_location("candidate_consensus_diagnostics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_candidate_k_uses_off_diagonal_consensus_and_tracks_cluster_items():
    final = np.array([
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.1, 0.2],
        [0.2, 0.1, 1.0, 0.9],
        [0.1, 0.2, 0.9, 1.0],
    ])
    labels = np.array([0, 0, 1, 1])
    algorithms = {name: final.copy() for name in MODULE.ALGORITHMS}

    summary, clusters, items = MODULE.summarize_candidate_k(
        2, final, algorithms, labels, ["a", "b", "c", "d"], 0.1, 0.9
    )

    assert summary["pac"] == 1 / 3
    assert summary["mean_cluster_consensus"] == 0.9
    assert summary["min_cluster_consensus"] == 0.9
    assert np.isclose(summary["item_consensus_p10"], 0.9)
    assert summary["hierarchical_spectral_ari"] == 1.0
    assert summary["min_cluster_size"] == 2
    assert len(clusters) == 2
    assert [row["cluster_size"] for row in clusters] == [2, 2]
    assert len(items) == 4
    assert all(np.isclose(row["item_consensus"], 0.9) for row in items)


def test_singleton_cluster_item_consensus_is_missing_not_self_similarity():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.8], [0.0, 0.8, 1.0]])
    labels = np.array([0, 1, 1])
    algorithms = {name: matrix.copy() for name in MODULE.ALGORITHMS}

    summary, clusters, items = MODULE.summarize_candidate_k(
        2, matrix, algorithms, labels, ["a", "b", "c"], 0.1, 0.9
    )

    singleton = next(row for row in clusters if row["cluster_size"] == 1)
    item = next(row for row in items if row["case_id"] == "a")
    assert singleton["within_cluster_consensus"] is None
    assert item["item_consensus"] is None
    assert np.isclose(summary["item_consensus_p10"], 0.8)
