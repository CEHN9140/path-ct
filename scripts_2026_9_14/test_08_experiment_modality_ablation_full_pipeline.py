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
