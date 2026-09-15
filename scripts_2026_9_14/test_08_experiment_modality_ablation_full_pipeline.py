import numpy as np
import importlib.util
from pathlib import Path

path = Path(__file__).with_name("08_experiment_modality_ablation_full_pipeline.py")
spec = importlib.util.spec_from_file_location("modality_ablation", path)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def test_leave_ct_out_uses_exactly_four_active_views():
    assert ablation.ACTIVE_MODALITIES == ("wsi", "rna", "wxs", "cnv")
    assert "ct" not in ablation.ACTIVE_MODALITIES


def test_fuse_active_views_does_not_depend_on_ct():
    views = {name: np.eye(4) * (index + 1) for index, name in enumerate(("ct", "wsi", "rna", "wxs", "cnv"))}
    config = {"neighbor_count": 2, "iterations": 1, "alpha": 1.0}
    first = ablation.fuse_active_views(views, config)
    views["ct"] = np.full((4, 4), 999.0)
    second = ablation.fuse_active_views(views, config)
    assert np.array_equal(first, second)


def test_variant_manifest_records_disabled_ct():
    manifest = ablation.variant_manifest(["p1", "p2"], "abc", "def")
    assert manifest["active_modalities"] == ["wsi", "rna", "wxs", "cnv"]
    assert manifest["disabled_modalities"] == ["ct"]
    assert manifest["artifact_schema"] == "minimal_active_modalities_v3"


def test_cross_modal_metrics_only_report_active_views():
    from tools.multimodal_consistency_check import compute_cross_modal_consistency

    case_ids = [f"p{i}" for i in range(6)]
    memberships = {"C1": case_ids[:3], "C2": case_ids[3:]}
    matrices = {
        name: np.eye(6) + 0.1
        for name in ("wsi", "rna", "wxs", "cnv")
    }
    result = compute_cross_modal_consistency(
        matrices,
        case_ids,
        memberships,
        permanova_permutations=5,
        permdisp_permutations=5,
        modalities=("wsi", "rna", "wxs", "cnv"),
    )
    assert set(result["modality_partition_support"]) == set(matrices)
    assert "ct" not in result["affinity_audit"]
    assert "ct" not in result["decision_metrics"]["cross_modal_consistency"]["partition"]["permanova_r2"]


def test_compare_cores_accepts_load_cores_mapping():
    canonical = {"CORE01": ["A", "B", "C"], "CORE02": ["D", "E"]}
    variant = {"CORE01": ["A", "B"], "CORE02": ["D", "E", "F"]}
    result = ablation.compare_cores(canonical, variant)
    assert [row["canonical_core"] for row in result] == ["CORE01", "CORE02"]
    assert [row["variant_core"] for row in result] == ["CORE01", "CORE02"]
    assert result[0]["intersection"] == 2
    assert result[0]["canonical_size"] == 3
    assert result[0]["variant_size"] == 2
    assert result[0]["jaccard"] == 2 / 3
    assert result[1]["jaccard"] == 2 / 3


def test_compare_cores_handles_empty_variant():
    result = ablation.compare_cores({"CORE01": ["A", "B"]}, {})
    assert len(result) == 1
    assert result[0]["canonical_core"] == "CORE01"
    assert result[0]["variant_core"] is None
    assert result[0]["jaccard"] == 0.0
    summary = ablation.summarize_core_comparison(
        {"CORE01": ["A", "B"]}, {}, result
    )
    assert summary["canonical_core_coverage"] == 0.0
    assert summary["variant_core_coverage"] is None
    assert summary["matched_core_mean_jaccard"] is None
    assert summary["penalized_core_mean_jaccard"] == 0.0
    assert summary["unmatched_canonical_cores"] == ["CORE01"]


def test_compare_cores_handles_empty_canonical():
    summary = ablation.summarize_core_comparison({}, {"CORE01": ["A"]}, [])
    assert summary["canonical_core_coverage"] is None
    assert summary["variant_core_coverage"] == 0.0
    assert summary["extra_variant_cores"] == ["CORE01"]


def test_compare_cores_handles_both_empty():
    summary = ablation.summarize_core_comparison({}, {}, [])
    assert summary["canonical_core_coverage"] is None
    assert summary["variant_core_coverage"] is None


def test_fragmentation_rows_report_full_variant_distribution():
    canonical = {"CORE01": ["A", "B", "C", "D"]}
    variant = {"CORE01": ["A", "B"], "CORE05": ["C"], "CORE06": ["X"]}
    comparison = ablation.compare_cores(canonical, variant)
    row = ablation.fragmentation_rows(canonical, variant, comparison)[0]
    assert row["matched_variant_core"] == "CORE01"
    assert row["retained_n"] == 2
    assert row["variant_core_count_with_members"] == 2
    assert row["any_variant_stable_n"] == 3
    assert row["any_variant_stable_fraction"] == 0.75
    assert row["no_variant_stable_n"] == 1
    assert row["no_variant_stable_fraction"] == 0.25
    assert row["variant_membership_distribution"] == '{"CORE01": 2, "CORE05": 1}'


def test_leave_ct_out_confound_evidence_keeps_only_tss(monkeypatch, tmp_path):
    import tools.confound as confound

    ids = ["TCGA-A-0001", "TCGA-B-0002", "TCGA-C-0003", "TCGA-D-0004"]
    monkeypatch.setattr(
        confound,
        "confounder_values",
        lambda *_: {case_id: {"tissue_source_site": "A"} for case_id in ids},
    )
    state = {"set_id": "C1", "member_ids": ids[:2], "active_modalities": ablation.ACTIVE_MODALITIES}
    other = {"set_id": "C2", "member_ids": ids[2:], "active_modalities": ablation.ACTIVE_MODALITIES}
    result = confound.confound_test(
        state,
        {case_id: {} for case_id in ids},
        str(tmp_path),
        all_cluster_states=[state, other],
    )
    metrics = result["results"]["metrics"]
    assert set(metrics["confounder_global_association"]) == {"tissue_source_site"}
    assert all(set(row) == {"tissue_source_site"} for row in metrics["confounder_set_association"].values())
