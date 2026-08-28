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


def test_survival_km_plot_contains_only_stable_cores(tmp_path):
    records = {
        "a": {"os_time": 100, "os_event": 0},
        "b": {"os_time": 200, "os_event": 1},
        "c": {"os_time": 120, "os_event": 0},
        "d": {"os_time": 240, "os_event": 1},
    }
    plotted = analysis.plot_survival_km(
        tmp_path / "clinical_overall_survival_km",
        records,
        {"CORE01": ["a", "b"], "CORE02": ["c", "d"]},
    )
    assert plotted == 2
    assert (tmp_path / "clinical_overall_survival_km.png").exists()


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


def test_radiomics_analysis_reports_core_rest_and_pairwise_effects():
    table = {case: {"original_shape_Sphericity": value, "original_glcm_Contrast": value * 2} for case, value in {"a": 1, "b": 2, "c": 8, "d": 9}.items()}
    rows, pair_rows = analysis.radiomics_analysis(
        table, list(table["a"]), {"CORE01": ["a", "b"], "CORE02": ["c", "d"]}, list(table)
    )
    assert len(rows) == 4 and len(pair_rows) == 2
    assert {row["feature_family"] for row in rows} == {"shape", "glcm"}
    assert all(row["effect_size"] == row["smd"] for row in rows)
    assert all("medical_imaging_domain" in row for row in pair_rows)


def test_clinical_analysis_reports_stage_grade_metastasis_and_survival():
    records = {
        "a": {"stage_group": "I", "grade": "G2", "m_stage": "M0", "os_time": 100, "os_event": 0},
        "b": {"stage_group": "I", "grade": "G2", "m_stage": "M0", "os_time": 200, "os_event": 0},
        "c": {"stage_group": "IV", "grade": "G4", "m_stage": "M1", "os_time": 50, "os_event": 1},
        "d": {"stage_group": "IV", "grade": "G4", "m_stage": "M1", "os_time": 80, "os_event": 1},
    }
    rows, pair_rows, survival = analysis.clinical_analysis(
        records, {"CORE01": ["a", "b"], "CORE02": ["c", "d"]}, list(records)
    )
    assert {row["clinical_variable"] for row in rows} == {"stage_group", "grade", "m_stage"}
    assert {row["clinical_variable"] for row in survival} == {"overall_survival"}
    assert {row["clinical_variable"] for row in pair_rows} == {"stage_group", "grade", "m_stage", "overall_survival"}
    assert all("q_value" in row for row in rows + pair_rows + survival)


def test_clinical_availability_excludes_indeterminate_n_and_m_levels():
    records = {
        case_id: {
            "age": 60,
            "gender": "male",
            "race": "white",
            "stage_group": "I",
            "t_stage": "T1",
            "n_stage": "N0" if case_id in {"a", "b", "e", "f"} else "NX",
            "m_stage": "M1" if case_id in {"a", "e"} else "M0" if case_id != "i" else "MX",
            "grade": "G2",
            "os_time": 100,
            "os_event": 0,
        }
        for case_id in "abcdefghij"
    }
    rows, summary = analysis.clinical_availability(
        records, {"CORE01": list("abcd"), "CORE02": list("efgh")}, list(records)
    )
    assert summary["n_stage"]["overall_fraction"] == 0.4
    assert summary["n_stage"]["stable_core_fraction"] == 0.5
    assert "n_stage" in summary["not_analyzed_variables"]
    assert summary["n_stage"]["reason"] == "not_analyzed_below_availability_threshold"
    assert summary["m_stage"]["overall_available_n"] == 9
    assert "m_stage" in summary["eligible_variables"]
    assert analysis.clinical_value(records["i"], "m_stage") is None
    assert {row["scope"] for row in rows if row["clinical_variable"] == "n_stage"} == {"ALL", "STABLE_CORES", "CORE01", "CORE02"}
    clinical_rows, _, _ = analysis.clinical_analysis(records, {"CORE01": list("abcd"), "CORE02": list("efgh")}, list(records), summary["eligible_variables"], summary["survival_eligible"])
    assert not any(row["clinical_variable"] == "n_stage" for row in clinical_rows)
    assert {row["level"] for row in clinical_rows if row["clinical_variable"] == "m_stage"} == {"M0", "M1"}


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
