import numpy as np

from tools.multimodal_consistency_check import compute_cross_modal_consistency
from tools.subtype_review_common import assign_groupwise_fdr, bh_fdr


def test_bh_fdr_accepts_empty_family():
    assert bh_fdr([]) == []


def test_groupwise_fdr_does_not_mix_candidate_sets():
    rows = [
        {"candidate_set_id": "A", "p_value": 0.01},
        {"candidate_set_id": "A", "p_value": 0.02},
        {"candidate_set_id": "B", "p_value": 0.9},
    ]
    assign_groupwise_fdr(rows, "candidate_set_id", "p_value", "q_value")
    assert rows[0]["q_value"] == 0.02
    assert rows[1]["q_value"] == 0.02
    assert rows[2]["q_value"] == 0.9


def test_cross_modal_effect_epsilon_filters_near_zero_support():
    memberships = {"C1": ["a", "b"], "C2": ["c", "d"]}
    strong = np.array([[1.0, 0.9, 0.1, 0.1], [0.9, 1.0, 0.1, 0.1], [0.1, 0.1, 1.0, 0.9], [0.1, 0.1, 0.9, 1.0]])
    near_zero = np.ones((4, 4))
    result = compute_cross_modal_consistency(
        {"ct": strong, "wsi": strong, "rna": strong, "genomic": near_zero},
        ["a", "b", "c", "d"], memberships, permanova_permutations=9,
    )

    metrics = result["decision_metrics"]
    assert "genomic" not in metrics["identity_supporting_modalities"]
    assert "genomic" not in metrics["merge_strong_boundary_modalities"]
    assert metrics["effect_epsilon"] == 0.001
