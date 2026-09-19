import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).with_name("01_four_view_feature_engineering.py")
SPEC = importlib.util.spec_from_file_location("four_view_feature_engineering", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ct_transform_applies_constant_correlation_and_zscore_without_residualization():
    trend = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    matrix = np.column_stack([
        trend,
        trend * 2.0,
        [0.0, 2.0, 1.0, 4.0, 3.0],
        np.ones(5),
    ])

    transformed, names, audit = MODULE.transform_ct_features(
        matrix,
        ["technical", "correlated", "independent", "constant"],
        correlation_threshold=0.95,
    )

    assert names == ["correlated", "independent"]
    assert audit["constant_removed"] == ["constant"]
    assert audit["correlation_pruned"] == ["technical"]
    assert audit["technical_residualization"] is False
    assert audit["radiomics_stability_filter"] is False
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(transformed.std(axis=0), 1.0)


def test_correlation_pruning_is_invariant_to_input_column_order():
    matrix = np.array([
        [0.0, 0.0, 2.0],
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 0.0],
        [3.0, 6.0, 3.0],
        [4.0, 8.0, 2.0],
    ])
    names = ["feature_a", "feature_b", "feature_c"]

    kept_first, _, _ = MODULE.prune_correlated_features(matrix, names, 0.95)
    order = [2, 1, 0]
    kept_permuted, _, _ = MODULE.prune_correlated_features(matrix[:, order], [names[i] for i in order], 0.95)

    assert set(kept_first) == set(kept_permuted) == {"feature_a", "feature_c"}


def test_patient_resampled_consensus_reuses_each_sample_across_k_and_algorithms(monkeypatch):
    rng = np.random.default_rng(12)
    points = rng.normal(size=(8, 3))
    distance = np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=2))
    affinity = np.exp(-distance)
    modalities = {name: affinity.copy() for name in ("ct", "wsi", "rna", "wxs")}
    original_fuse = MODULE.fuse_affinities
    fuse_calls = []

    def count_fuse(networks, config):
        fuse_calls.append(tuple(networks))
        return original_fuse(networks, config)

    monkeypatch.setattr(MODULE, "fuse_affinities", count_fuse)
    records = MODULE.build_patient_resampled_consensus(
        modalities,
        [f"P{i}" for i in range(8)],
        {"neighbor_count": 3, "iterations": 5, "alpha": 1.0},
        {
            "algorithms": {
                "hierarchical": {"linkage_options": ["average", "complete"]},
                "spectral": {"assign_labels_options": ["kmeans", "discretize", "cluster_qr"]},
                "kmedoids": {"init_options": ["k-medoids++", "random", "heuristic"]},
            }
        },
        candidate_ks=(2, 3),
        sample_fraction=0.875,
        n_resamples=30,
        random_seed=20260921,
    )

    assert len(fuse_calls) == 30
    assert [record["n_clusters"] for record in records] == [2, 3]
    for record in records:
        assert record["labels"].shape == (8,)
        assert sum(record["cluster_sizes"]) == 8
        assert np.all(record["pair_seen"][~np.eye(8, dtype=bool)] > 0)
        assert set(record["algorithm_consensus"]) == {"hierarchical", "spectral", "kmedoids"}
        mean_consensus = np.mean(list(record["algorithm_consensus"].values()), axis=0)
        assert np.allclose(record["consensus"], mean_consensus)
        assert np.allclose(record["consensus"], record["consensus"].T)
        assert np.allclose(np.diag(record["consensus"]), 1.0)


def test_patient_resampled_consensus_rejects_subsamples_smaller_than_max_k():
    with pytest.raises(ValueError, match="too small"):
        MODULE.build_patient_resampled_consensus(
            {"ct": np.eye(8)},
            [f"P{i}" for i in range(8)],
            {},
            {},
            candidate_ks=(2, 7),
            sample_fraction=0.5,
            n_resamples=5,
        )


def test_wxs_variant_uses_empty_mutation_distance_one_without_affinity_override():
    features = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=bool)
    snf_config = {"neighbor_count": 2, "mu": 0.5}

    distance, affinity = MODULE.build_wxs_view(features, snf_config)

    assert distance[0, 1] == 1.0
    assert np.allclose(np.diag(distance), 0.0)
    assert affinity[0, 1] > 0.0


def test_wxs_feature_selection_uses_prevalence_without_forced_genes():
    table = pd.DataFrame({
        "case_id": ["a", "b", "c", "d"],
        "mutation::common": [1, 1, 0, 0],
        "mutation::rare_driver": [1, 0, 0, 0],
    })

    selected_table, selected = MODULE.prevalence_only_wxs_features(table, ["a", "b", "c", "d"], 0.5)

    assert selected == ["mutation::common"]
    assert selected_table.columns.tolist() == ["case_id", "mutation::common"]
