import json

import numpy as np

from tools.cross_modal_structure import (
    compute_structural_characterization,
    normalized_cut,
)
from tools.multimodal_consistency_check import multimodal_consistency_check


def block_affinity(groups, same=0.9, different=0.1):
    groups = np.asarray(groups)
    matrix = np.where(groups[:, None] == groups[None, :], same, different).astype(float)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def test_structure_uses_actual_fused_binary_probe_and_fixed_modality_labels():
    case_ids = ["a", "b", "c", "d", "e", "f"]
    fused = block_affinity([0, 0, 0, 1, 1, 1])
    ct = block_affinity([0, 0, 0, 1, 1, 1], same=0.1, different=0.9)
    result = compute_structural_characterization(
        fused,
        {"ct": ct, "wsi": None, "rna": None, "genomic": None},
        case_ids,
        {"C1": case_ids},
        resampling_iterations=10,
    )
    probe = result["internal_structure_by_set"]["C1"]["fused_binary_probe"]
    assert probe["child_sizes"] == [3, 3]
    labels = probe["probe_labels_by_case"]
    assert len({labels[case_id] for case_id in case_ids[:3]}) == 1
    assert len({labels[case_id] for case_id in case_ids[3:]}) == 1
    assert labels["a"] != labels["d"]
    assert result["internal_structure_by_set"]["C1"]["probe_support_by_modality"]["ct"]["comparison_status"] == "estimable"
    assert result["internal_structure_by_set"]["C1"]["probe_support_by_modality"]["ct"]["median_silhouette"] < 0
    assert "positive_modalities" not in result["internal_structure_by_set"]["C1"]


def test_binary_probe_reports_ncut_and_resampling_continuous_metrics():
    case_ids = ["a", "b", "c", "d", "e", "f"]
    result = compute_structural_characterization(
        block_affinity([0, 0, 0, 1, 1, 1]),
        {modality: block_affinity([0, 0, 0, 1, 1, 1]) for modality in ("ct", "wsi", "rna", "genomic")},
        case_ids,
        {"C1": case_ids},
        resampling_fraction=0.8,
        resampling_iterations=10,
        pac_lower=0.1,
        pac_upper=0.9,
    )
    probe = result["internal_structure_by_set"]["C1"]["fused_binary_probe"]
    assert probe["normalized_cut"] < normalized_cut(
        block_affinity([0, 1, 0, 1, 0, 1]),
        np.array([0, 0, 0, 1, 1, 1]),
    )
    assert set(probe["resampling"]) >= {
        "iterations",
        "median_resample_ari",
        "consensus_separation",
        "pac",
        "degenerate_resample_fraction",
    }
    assert "k_diagnostics" not in result["internal_structure_by_set"]["C1"]


def test_pair_boundary_is_continuous_and_keeps_left_right_metrics():
    case_ids = ["a", "b", "c", "d"]
    result = compute_structural_characterization(
        block_affinity([0, 0, 1, 1]),
        {modality: block_affinity([0, 0, 1, 1]) for modality in ("ct", "wsi", "rna", "genomic")},
        case_ids,
        {"C1": ["a", "b"], "C2": ["c", "d"]},
        resampling_iterations=5,
    )
    pair = result["boundary_by_pair"]["C1+C2"]
    assert "weak_boundary" not in pair
    assert pair["fused"]["pair_median_silhouette"] > 0
    assert "left_median_margin" in pair["fused"]
    assert "right_median_margin" in pair["fused"]
    assert "left_boundary_separation" in pair["fused"]
    assert "right_boundary_separation" in pair["fused"]
    assert "left_within_affinity" in pair["fused"]
    assert "right_within_affinity" in pair["fused"]
    assert "between_affinity" in pair["fused"]


def test_missing_fused_and_singleton_are_structured_scientific_results():
    case_ids = ["a", "b", "c"]
    modalities = {modality: block_affinity([0, 1, 1]) for modality in ("ct", "wsi", "rna", "genomic")}
    missing = compute_structural_characterization(
        None, modalities, case_ids, {"C1": case_ids}, resampling_iterations=3
    )
    assert missing["internal_structure_by_set"]["C1"]["comparison_status"] == "scientific_unavailable"
    singleton = compute_structural_characterization(
        block_affinity([0, 1, 1]), modalities, case_ids, {"C1": ["a"], "C2": ["b", "c"]}, resampling_iterations=3
    )
    assert singleton["internal_structure_by_set"]["C1"]["comparison_status"] == "not_estimable"
    assert singleton["boundary_by_pair"]["C1+C2"]["fused"]["comparison_status"] == "partially_estimable"


def test_tool_keeps_probe_labels_in_full_metrics_only(tmp_path):
    ids = ["a", "b", "c", "d"]
    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text(json.dumps(ids), encoding="utf-8")
    matrix = block_affinity([0, 0, 1, 1])
    np.save(candidate / "fused_similarity.npy", matrix)
    for modality in ("ct", "wsi", "rna"):
        np.save(candidate / f"{modality}_affinity.npy", matrix)
    wxs = tmp_path / "wxs"
    wxs.mkdir()
    np.save(wxs / "genomic_affinity.npy", matrix)
    result = multimodal_consistency_check(
        {"cluster_id": "C1", "member_ids": ids},
        {},
        str(tmp_path),
        config_dir="configs",
        all_cluster_states=[{"set_id": "C1", "member_ids": ids}],
    )
    full = result["results"]["metrics"]["structural_characterization"]
    decision = result["results"]["decision_metrics"]["cross_modal_consistency"]
    assert "probe_labels_by_case" in full["internal_structure_by_set"]["C1"]["fused_binary_probe"]
    assert "probe_labels_by_case" not in decision["per_set"]["C1"]["internal_structure"]["fused_binary_probe"]
