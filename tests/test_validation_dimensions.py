from tools.confound import (
    assign_q_values,
    confound_decision_metrics,
    confounder_values,
)
from tools.known_label_echo import (
    cluster_labels_by_case,
    compare_label_structures,
    known_label_echo_test,
)


def test_confounder_values_parse_tss_and_ct_metadata(tmp_path):
    states = {
        "TCGA-B0-4698": {
            "inventory": {
                "Case_ID": "TCGA-B0-4698",
                "CT": [{
                    "Manufacturer": "GE MEDICAL SYSTEMS",
                    "ManufacturersModelName": "Discovery CT750 HD",
                    "ConvolutionKernel": "STANDARD",
                    "Study Date": "11-22-1985",
                    "Slice Thickness": "2.5",
                    "Pixel Spacing": "0.8\\0.9",
                    "Spacing Between Slices": "2.5",
                    "Number of Images": 100,
                }],
            }
        }
    }
    values = confounder_values(states, str(tmp_path))
    row = values["TCGA-B0-4698"]
    assert row["tissue_source_site"] == "B0"
    assert row["ct_manufacturer"] == "GE"
    assert row["ct_pixel_spacing"] == "0.85"
    assert row["ct_scanner_model"] == "Discovery CT750 HD"


def test_confound_threshold_only_warns_on_global_cramers_v():
    global_metrics = {
        "tissue_source_site": {
            "field_type": "categorical", "available_n": 10, "missing_n": 0,
            "cramers_v": 0.6, "q_value": 0.01,
        },
        "ct_slice_thickness": {
            "field_type": "numeric", "available_n": 10, "missing_n": 0,
            "epsilon_squared": 0.2, "q_value": 0.01,
        },
    }
    set_metrics = {
        "C1": {"tissue_source_site": {
            "B0": {"field_type": "categorical", "frequency_difference": 0.9, "q_value": 0.01}
        }}
    }
    result = confound_decision_metrics(global_metrics, set_metrics)
    flags = result["deterministic_flags"]
    assert flags["categorical_warning_association"] is True
    assert flags["numeric_significant_association"] is True
    assert flags["notable_technical_association"] is True
    assert "strong_technical_association" not in flags
    assert "ct_slice_thickness" in flags["numeric_association_fields"]
    assert "C1" in flags["sets_with_significant_technical_association"]
    assert "invalidated_set_ids" not in flags
    assert "categorical_strong_v" not in flags["thresholds"]


def test_confound_secondary_global_metrics_are_descriptive_only():
    global_metrics = {
        "ct_slice_thickness": {
            "field_type": "numeric", "kruskal_p_value": 0.01,
        },
        "ct_study_year": {
            "field_type": "numeric", "kruskal_p_value": 0.01,
        },
    }
    assign_q_values(global_metrics, {})
    assert global_metrics["ct_slice_thickness"]["q_value"] == 0.01
    assert global_metrics["ct_study_year"]["q_value"] is None


def test_known_label_uses_set_id_before_cluster_id():
    states = [
        {"set_id": "C1", "cluster_id": "old-1", "member_ids": ["P1"]},
        {"set_id": "C2", "cluster_id": "old-2", "member_ids": ["P2"]},
    ]
    assert cluster_labels_by_case(states, states[0]) == {"P1": "C1", "P2": "C2"}


def test_known_label_reports_ari_homogeneity_completeness_for_refinement_and_coarsening():
    clinical = {
        "P1": {"stage_group": "I"}, "P2": {"stage_group": "I"},
        "P3": {"stage_group": "II"}, "P4": {"stage_group": "II"},
        "P5": {"stage_group": "III"}, "P6": {"stage_group": "III"},
    }
    refined = compare_label_structures(
        "stage", "stage_group",
        {"P1": "C1", "P2": "C2", "P3": "C3", "P4": "C4", "P5": "C5", "P6": "C6"}, clinical
    )
    coarse = compare_label_structures(
        "stage", "stage_group",
        {"P1": "C1", "P2": "C1", "P3": "C1", "P4": "C1", "P5": "C2", "P6": "C2"}, clinical
    )
    for result in (refined, coarse):
        assert {"adjusted_rand_index", "homogeneity", "completeness"}.issubset(result)
        assert all(0 <= result[key] <= 1 for key in ("homogeneity", "completeness"))
    assert refined["homogeneity"] == 1.0
    assert coarse["completeness"] == 1.0


def test_known_label_keeps_stage_grade_missingness_and_limitations(tmp_path):
    states = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3"]},
    ]
    patients = {
        "P1": {"clinical": {"diagnoses": [{"ajcc_pathologic_stage": "Stage I", "tumor_grade": "G2"}]}},
        "P2": {"clinical": {"diagnoses": [{"ajcc_pathologic_stage": "", "tumor_grade": "G3"}]}},
        "P3": {"clinical": {"diagnoses": [{"ajcc_pathologic_stage": "Stage III", "tumor_grade": ""}]}},
    }
    result = known_label_echo_test(
        states[0], patients, str(tmp_path), all_cluster_states=states
    )
    decision = result["results"]["decision_metrics"]
    assert decision["assessable_labels"] == ["stage", "grade"]
    assert decision["limitations"]
    assert decision["by_label"]["stage"]["missing_n"] == 1
    assert decision["by_label"]["grade"]["missing_n"] == 1
    assert "near_identity" not in str(result)


def test_known_label_is_not_estimable_with_one_known_level():
    result = compare_label_structures(
        "stage", "stage_group",
        {"P1": "C1", "P2": "C2"},
        {"P1": {"stage_group": "I"}, "P2": {"stage_group": "I"}},
    )
    assert result["comparison_status"] == "not_estimable"
    assert result["not_estimable_reason"] == "single_label_level"
    assert result["adjusted_rand_index"] is None


def test_known_label_is_not_estimable_with_one_candidate_set():
    result = compare_label_structures(
        "stage", "stage_group",
        {"P1": "C1", "P2": "C1"},
        {"P1": {"stage_group": "I"}, "P2": {"stage_group": "III"}},
    )
    assert result["comparison_status"] == "not_estimable"
    assert result["not_estimable_reason"] == "single_candidate_set"
    assert result["homogeneity"] is None
