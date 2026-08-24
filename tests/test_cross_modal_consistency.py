import numpy as np

from tools.multimodal_consistency_check import (
    MODALITIES,
    compute_cross_modal_consistency,
)


def positive_affinity(size=4):
    matrix = np.full((size, size), 0.1)
    matrix[: size // 2, : size // 2] = 0.9
    matrix[size // 2 :, size // 2 :] = 0.9
    np.fill_diagonal(matrix, 1.0)
    return matrix


def negative_affinity(size=4):
    matrix = np.full((size, size), 0.9)
    matrix[: size // 2, : size // 2] = 0.1
    matrix[size // 2 :, size // 2 :] = 0.1
    np.fill_diagonal(matrix, 1.0)
    return matrix


def four_modalities(matrix, missing=()):
    return {
        modality: None if modality in missing else matrix.copy()
        for modality in MODALITIES
    }


def test_fixed_membership_silhouette_patient_support_and_decision_layers():
    result = compute_cross_modal_consistency(
        {
            "ct": positive_affinity(),
            "wsi": negative_affinity(),
            "rna": positive_affinity(),
            "genomic": positive_affinity(),
        },
        ["a", "b", "c", "d"],
        {"C1": ["a", "b"], "C2": ["c", "d"]},
        permanova_permutations=9,
        permdisp_permutations=9,
    )
    full = result["modality_partition_support"]
    decision = result["decision_metrics"]["cross_modal_consistency"]
    assert full["ct"]["per_set"]["C1"]["median_silhouette"] > 0
    assert full["wsi"]["per_set"]["C1"]["median_silhouette"] < 0
    profile = result["patient_membership_profile"]["a"]
    assert profile["support_count"] == 3
    assert set(profile["positive_modalities"]) == {"ct", "rna", "genomic"}
    assert profile["negative_modalities"] == ["wsi"]
    assert "supporting" not in str(decision)
    assert "stable" not in str(decision)
    assert set(decision["per_set"]["C1"]) == {
        "modality_support", "patient_membership_support"
    }
    assert "patient_membership_profile" not in decision


def test_patient_profile_and_top5_compression_are_complete():
    case_ids = [f"p{i}" for i in range(8)]
    result = compute_cross_modal_consistency(
        four_modalities(positive_affinity(8)),
        case_ids,
        {"C1": case_ids[:6], "C2": case_ids[6:]},
        permanova_permutations=5,
        permdisp_permutations=5,
        lowest_support_patients_to_report=5,
    )
    assert set(result["patient_membership_profile"]) == set(case_ids)
    reported = result["decision_metrics"]["cross_modal_consistency"]["per_set"]["C1"]
    assert len(reported["patient_membership_support"]["lowest_support_patients"]) == 5
    assert set(result["patient_membership_profile"]["p0"]["silhouette_by_modality"]) == set(MODALITIES)
    assert "mean_within_affinity" in result["modality_partition_support"]["ct"]["per_set"]["C1"]
    assert "mean_within_affinity" not in reported["modality_support"]["ct"]


def test_permanova_r2_and_permdisp_are_partition_diagnostics():
    result = compute_cross_modal_consistency(
        four_modalities(positive_affinity()),
        ["a", "b", "c", "d"],
        {"C1": ["a", "b"], "C2": ["c", "d"]},
        permanova_permutations=7,
        permdisp_permutations=7,
    )
    full = result["modality_partition_support"]["ct"]
    decision = result["decision_metrics"]["cross_modal_consistency"]["partition"]
    assert full["permanova"]["permutations"] == 7
    assert full["permdisp"]["permutations"] == 7
    assert full["permanova_r2"] is not None
    assert "permanova_q_value" in full["permanova"]
    assert "permdisp_q_value" in full["permdisp"]
    assert "ct" in decision["permanova_r2"]
    assert set(decision["permdisp"]["ct"]) == {"f", "p_value", "q_value"}


def test_affinity_normalization_is_finite_symmetric_bounded_and_audited():
    raw = positive_affinity()
    raw[0, 1] = raw[1, 0] = 1.5
    raw[0, 2] = raw[2, 0] = -0.5
    result = compute_cross_modal_consistency(
        four_modalities(raw),
        ["a", "b", "c", "d"],
        {"C1": ["a", "b"], "C2": ["c", "d"]},
        permanova_permutations=3,
        permdisp_permutations=3,
    )
    audit = result["affinity_audit"]["ct"]
    assert audit["gt1_clipping_count"] > 0
    normalized = result["modality_partition_support"]["ct"]
    assert 0 <= audit["normalized_min"] <= audit["normalized_max"] <= 1
    assert normalized["affinity_audit"]["normalized_min"] == audit["normalized_min"]


def test_scientific_non_estimability_and_partial_modality_availability():
    single = compute_cross_modal_consistency(
        four_modalities(positive_affinity(3)),
        ["a", "b", "c"],
        {"C1": ["a", "b", "c"]},
        permanova_permutations=3,
        permdisp_permutations=3,
    )
    assert single["modality_partition_support"]["ct"]["permanova"]["comparison_status"] == "not_estimable"
    singleton = compute_cross_modal_consistency(
        four_modalities(positive_affinity(3)),
        ["a", "b", "c"],
        {"C1": ["a"], "C2": ["b", "c"]},
        permanova_permutations=3,
        permdisp_permutations=3,
    )
    assert singleton["modality_partition_support"]["ct"]["permdisp"]["not_estimable_reason"] == "singleton_set"
    assert singleton["modality_partition_support"]["ct"]["per_set"]["C1"]["comparison_status"] == "not_estimable"
    partial = compute_cross_modal_consistency(
        four_modalities(positive_affinity(), missing=("genomic",)),
        ["a", "b", "c", "d"],
        {"C1": ["a", "b"], "C2": ["c", "d"]},
        permanova_permutations=3,
        permdisp_permutations=3,
    )
    assert partial["modality_partition_support"]["genomic"]["comparison_status"] == "scientific_unavailable"
    assert partial["patient_membership_profile"]["a"]["silhouette_by_modality"]["genomic"] is None
