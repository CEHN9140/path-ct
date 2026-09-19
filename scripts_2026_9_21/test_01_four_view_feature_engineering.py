import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("01_four_view_feature_engineering.py")
SPEC = importlib.util.spec_from_file_location("four_view_feature_engineering", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ct_transform_prunes_then_residualizes_and_standardizes():
    technical = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    matrix = np.column_stack([
        technical,
        technical * 2.0,
        [0.0, 2.0, 1.0, 4.0, 3.0],
        np.ones(5),
    ])

    transformed, names, audit = MODULE.transform_ct_features(
        matrix,
        ["technical", "correlated", "independent", "constant"],
        np.column_stack([np.ones(5), technical]),
        low_variance_threshold=1e-8,
        correlation_threshold=0.95,
    )

    assert names == ["technical", "independent"]
    assert audit["low_variance_removed"] == ["constant"]
    assert audit["correlation_pruned"] == ["correlated"]
    assert audit["post_residual_zero_variance_features"] == ["technical"]
    assert np.allclose(transformed[:, 0], 0.0)
    assert np.allclose(transformed[:, 1].mean(), 0.0, atol=1e-12)
    assert np.isclose(transformed[:, 1].std(), 1.0)


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
