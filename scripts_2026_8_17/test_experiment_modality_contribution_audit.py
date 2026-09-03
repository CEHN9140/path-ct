import numpy as np

from scripts_2026_8_17 import experiment_modality_contribution_audit as experiment


def test_view_definitions_cover_all_nine_experiments():
    assert list(experiment.VIEWS) == [
        "ALL", "minus_RNA", "minus_WSI", "minus_CT", "minus_GENOMIC",
        "RNA_only", "WSI_only", "CT_only", "GENOMIC_only",
    ]


def test_core_recovery_uses_best_cluster_overlap():
    rows = experiment.core_recovery(
        {"CORE01": ["a", "b"]},
        {"k2": {"x": ["a", "b", "c"], "y": ["d"]}},
        "minus_RNA",
    )
    assert rows[0]["best_jaccard"] == .66666667
    assert rows[0]["best_recall"] == 1.0


def test_core_recovery_selects_jaccard_not_intersection_count():
    rows = experiment.core_recovery(
        {"CORE01": ["a", "b", "c", "d"]},
        {"k2": {"large": ["a", "b", "c", "d", "x", "y"], "tight": ["a", "b", "c"]}},
        "RNA_only",
    )
    assert rows[0]["best_jaccard"] == .75


def test_affinity_audit_returns_correlation_and_knn_overlap():
    matrix = np.eye(4)
    rows = experiment.affinity_audit(
        {"ALL": matrix, "RNA_only": matrix.copy()}, ["a", "b", "c", "d"], 2
    )
    row = next(item for item in rows if item["view"] == "RNA_only")
    assert row["spearman_vs_all"] == 1.0
    assert row["mean_knn_jaccard_vs_all"] == 1.0
