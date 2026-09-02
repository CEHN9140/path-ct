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


def test_stage_number_parses_grouped_stage():
    assert experiment.stage_number("III") == 3
    assert experiment.stage_number("stage IV") == 4
    assert experiment.stage_number("unknown") is None


def test_tss_code_parses_tcga_barcode():
    assert experiment.tss_code("TCGA-B0-4698") == "B0"


def test_evidence_table_reports_supported_pair_fraction():
    cores = {f"CORE0{i}": [str(i)] for i in range(1, 6)}
    states = {"STATE_A": ["1", "3"], "STATE_B": ["2", "5"], "STATE_C": ["4"]}
    pair = [{"core_a": "a", "core_b": "b", "q_value": .01}]
    fused = [{"median_group_silhouette": 0.1, "permanova_permanova_r2": .2}]
    rows = experiment.evidence_table(cores, states, pair, pair, pair, pair, pair, pair, pair, pair, fused, fused)
    assert rows[0]["three_macro_state_total_pairs"] == 3
    assert rows[0]["three_macro_state_supported_fraction"] == .33333333
