from pathlib import Path

import numpy as np

from scripts_2026_8_17 import experiment_macro_state_analysis as experiment


def test_macro_state_membership_preserves_five_core_membership():
    cores = {
        "CORE01": ["a", "b"],
        "CORE02": ["c"],
        "CORE03": ["d"],
        "CORE04": ["e"],
        "CORE05": ["f", "g"],
    }
    states = experiment.macro_state_members(cores)
    assert states == {
        "STATE_A": ["a", "b", "d"],
        "STATE_B": ["c", "f", "g"],
        "STATE_C": ["e"],
    }


def test_macro_fused_structure_plot_writes_png_only(tmp_path):
    ids = ["a", "b", "c", "d", "e", "f"]
    similarity = np.full((6, 6), 0.25)
    np.fill_diagonal(similarity, 1)
    states = {
        "STATE_A": ["a", "b"],
        "STATE_B": ["c", "d"],
        "STATE_C": ["e", "f"],
    }
    methods = experiment.plot_macro_fused_structure(
        tmp_path / "macro_fused_structure", similarity, ids, states, 42
    )
    assert methods == ["pcoa", "spectral", "diffusion"]
    assert (tmp_path / "macro_fused_structure.png").is_file()
    assert not list(tmp_path.glob("*.pdf"))


def test_model_comparison_contains_five_core_and_three_state_rows():
    rows = experiment.compare_fused_models(
        {"CORE01": ["a", "b"], "CORE02": ["c", "d"], "CORE03": ["e", "f"], "CORE04": ["g", "h"]},
        {"STATE_A": ["a", "b", "e", "f"], "STATE_B": ["c", "d"], "STATE_C": ["g", "h"]},
        np.full((8, 8), .25) + .75 * np.eye(8),
        ["a", "b", "c", "d", "e", "f", "g", "h"],
    )
    assert {row["model"] for row in rows} == {"five_core", "three_macro_state"}
