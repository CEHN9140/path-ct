import numpy as np
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "genomic_fusion_sensitivity",
    Path(__file__).with_name("13_experiment_genomic_fusion_sensitivity.py"),
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_variants_keep_genomic_fusion_controls_separate():
    assert experiment.VARIANTS["M5_independent_wxs_cnv"] == ("ct", "wsi", "rna", "wxs", "cnv")
    assert experiment.VARIANTS["M_dupGenomic"] == ("ct", "wsi", "rna", "genomic", "genomic")


def test_coassignment_is_averaged_over_all_k():
    result = experiment.coassignment({2: np.array([0, 0, 1]), 3: np.array([0, 1, 1])})
    assert result[0, 1] == .5
    assert result[1, 2] == .5
    assert np.all(np.diag(result) == 1)


def test_bootstrap_boundary_excludes_self_affinity():
    matrix = np.array([
        [1., .8, .2, .2], [.8, 1., .2, .2],
        [.2, .2, 1., .8], [.2, .2, .8, 1.],
    ])
    effect, low, high = experiment.bootstrap_boundary(
        matrix, np.full((4, 4), .5), [0, 1], [2, 3],
        np.random.default_rng(20260614), 20,
    )
    assert effect > 0
    assert low <= high
