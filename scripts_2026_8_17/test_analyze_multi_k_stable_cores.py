import csv
import json
from pathlib import Path

import numpy as np

from scripts_2026_8_17 import analyze_multi_k_stable_cores as analysis


def test_mapping_and_composition_are_patient_set_based():
    cores = {"CORE01": {"a", "b", "c"}, "CORE02": {"d"}}
    main = [
        {"set_id": "C0001", "member_ids": ["a", "b", "x"], "decision": "accept"},
        {"set_id": "C0002", "member_ids": ["c", "d"], "decision": "drop"},
    ]
    rows, composition = analysis.main_mapping(cores, main)
    row = next(item for item in rows if item["core_id"] == "CORE01" and item["main_set_id"] == "C0001")
    assert row["intersection_n"] == 2
    assert row["core_fraction_in_main"] == 2 / 3
    assert row["main_fraction_captured"] == 2 / 3
    assert composition["CORE01"] == {"C0001": 2, "C0002": 1}


def test_core_vs_rest_and_pairwise_fdr_are_separate():
    rows = analysis.test_rows(
        {"CORE01": [1.0, 2.0], "CORE02": [8.0, 9.0]},
        [1.0, 2.0, 8.0, 9.0],
        feature="x",
    )
    assert len(rows) == 2
    assert all(row["q_value"] is not None for row in rows)
    table = {case_id: {"x": value} for case_id, value in {"a": 1.0, "b": 2.0, "c": 8.0, "d": 9.0}.items()}
    pair = analysis.pairwise_numeric_rows(
        table, {"CORE01": ["a", "b"], "CORE02": ["c", "d"]}, feature="x"
    )
    assert pair[0]["core_a"] == "CORE01"
    assert pair[0]["core_b"] == "CORE02"
    assert pair[0]["q_value"] is not None


def test_affinity_metrics_use_fixed_membership_and_core_distance():
    ids = ["a", "b", "c", "d"]
    similarity = np.array(
        [[1, .9, .1, .2], [.9, 1, .2, .1], [.1, .2, 1, .8], [.2, .1, .8, 1]],
        dtype=float,
    )
    result = analysis.affinity_characterization(
        similarity, ids, {"CORE01": ["a", "b"], "CORE02": ["c", "d"]}
    )
    assert result["per_core"]["CORE01"]["silhouette"] > 0
    assert result["distance_matrix"]["CORE01"]["CORE02"] > 0


def test_matplotlib_writer_creates_png_only(tmp_path):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    analysis.save_figure(figure, tmp_path / "figure")
    assert (tmp_path / "figure.png").exists()
    assert not (tmp_path / "figure.pdf").exists()


def test_pathway_selection_keeps_any_significant_core_and_fills_by_max_effect():
    rows = [
        {"pathway": "A", "core_id": "CORE01", "q_value": 0.01, "smd": 0.1},
        {"pathway": "A", "core_id": "CORE02", "q_value": 0.8, "smd": 0.2},
        {"pathway": "B", "core_id": "CORE01", "q_value": 0.8, "smd": 0.9},
        {"pathway": "B", "core_id": "CORE02", "q_value": 0.8, "smd": 0.8},
        {"pathway": "C", "core_id": "CORE01", "q_value": 0.8, "smd": 0.7},
    ]
    assert analysis.select_pathways(rows, 2) == ["A", "B"]


def test_cnv_pairwise_fdr_is_scoped_to_current_pair():
    table = {case: {"chr1": value} for case, value in {"a": -1, "b": -1, "c": 1, "d": 1, "e": 0, "f": 0}.items()}
    _, _, _, rows = analysis.cnv_analysis(table, ["chr1"], {"CORE01": ["a", "b"], "CORE02": ["c", "d"], "CORE03": ["e", "f"]}, list(table))
    assert {row["core_a"] + row["core_b"] for row in rows} == {"CORE01CORE02", "CORE01CORE03", "CORE02CORE03"}
    assert all("q_value" in row for row in rows)


def test_pairwise_cnv_count_uses_only_the_current_pair():
    rows = [{"core_a": "CORE01", "core_b": "CORE02"}]
    cnv = [
        {"core_a": "CORE01", "core_b": "CORE02", "q_value": 0.01},
        {"core_a": "CORE01", "core_b": "CORE03", "q_value": 0.01},
    ]
    result = analysis.update_pairwise_similarity(rows, [], [], cnv)
    assert result[0]["cnv_pairwise_fdr_feature_count"] == 1


def test_pairwise_distances_are_named_as_mean_between_core_distances():
    result = analysis.pairwise_similarity(
        {"CORE01": ["a"], "CORE02": ["b"]}, [], [], [],
        {modality: {"CORE01": {"CORE02": 0.5}} for modality in analysis.MODALITIES}, [], [],
    )[0]
    assert result["ct_mean_between_core_distance"] == 0.5
    assert "ct_centroid_distance" not in result


def test_cnv_heatmap_selection_is_unique_and_significance_first():
    rows = [
        {"feature": "chr1", "core_id": "CORE01", "q_value": 0.8, "cliffs_delta": 0.99},
        {"feature": "chr1", "core_id": "CORE02", "q_value": 0.01, "cliffs_delta": 0.1},
        {"feature": "chr2", "core_id": "CORE01", "q_value": 0.8, "cliffs_delta": 0.8},
    ]
    assert analysis.select_cnv_heatmap_features(rows, 1) == ["chr1"]


def test_cooccurrence_does_not_turn_unassignable_into_different_parent():
    rows, by_k = analysis.cooccurrence_from_runs(
        [{"run_id": "run1_K2", "initial_k": 2, "sets": {"C1": {"a"}}}],
        {"CORE01": ["a", "b"], "CORE02": ["c", "d"]},
    )
    assert rows[0]["valid_assignment"] == 0
    assert rows[0]["same_parent_set"] is None
    assert by_k[0]["conditional_same_parent_fraction"] is None
    assert by_k[0]["unconditional_same_parent_fraction"] == 0


def test_run_summary_has_non_core_and_manifest_hash(tmp_path):
    summary = analysis.build_summary(
        {"CORE01": ["a", "b"]},
        ["a", "b", "c"],
        {"CORE01": {"C0001": 2}},
        {"CORE01": []},
        {"CORE01": []},
    )
    assert summary["patient_count"] == 3
    assert summary["stable_core_patient_count"] == 2
    assert summary["non_core_patient_count"] == 1
