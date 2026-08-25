import json

import numpy as np
import pytest

from tools.cross_modal_structure import (
    compute_structural_characterization,
    normalized_cut,
    pair_boundary_metrics,
    resampling_consensus,
    execute_split_membership,
)
from tools.multimodal_consistency_check import multimodal_consistency_check
from agents.subtype_review.tools import compact_tool_result


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


def test_resampling_clear_structure_is_more_stable_than_weak_structure():
    rng = np.random.default_rng(7)
    labels = np.asarray([0] * 6 + [1] * 6)
    same = labels[:, None] == labels[None, :]
    clear = np.where(same, 0.92, 0.08)
    np.fill_diagonal(clear, 1.0)
    noise = rng.normal(0, 0.22, clear.shape)
    weak = np.clip(np.where(same, 0.54, 0.46) + (noise + noise.T) / 2, 0, 1)
    np.fill_diagonal(weak, 1.0)
    clear_result = resampling_consensus(clear, labels, fraction=0.75, iterations=100, seed=3)
    weak_result = resampling_consensus(weak, labels, fraction=0.75, iterations=100, seed=3)
    assert clear_result["consensus_separation"] > weak_result["consensus_separation"]
    assert clear_result["pac"] < weak_result["pac"]


def test_structural_diagnosis_matches_split_execution(tmp_path):
    case_ids = ["a", "b", "c", "d", "e", "f"]
    matrix = block_affinity([0, 0, 0, 1, 1, 1])
    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text(json.dumps(case_ids), encoding="utf-8")
    np.save(candidate / "fused_similarity.npy", matrix)
    diagnostics = compute_structural_characterization(
        matrix, {}, case_ids, {"C1": case_ids}, resampling_iterations=10
    )
    labels = diagnostics["internal_structure_by_set"]["C1"]["fused_binary_probe"]["probe_labels_by_case"]
    diagnosed = {
        frozenset(case_id for case_id, label in labels.items() if label == value)
        for value in set(labels.values())
    }
    executed = {frozenset(group) for group in execute_split_membership(
        str(tmp_path), case_ids, 2, "fused_similarity_spectral", ["fused"]
    )}
    assert diagnosed == executed


def test_asymmetric_boundary_preserves_left_and_right_metrics():
    matrix = np.full((4, 4), 0.2)
    matrix[0, 1] = matrix[1, 0] = 0.95
    matrix[2, 3] = matrix[3, 2] = 0.55
    np.fill_diagonal(matrix, 1.0)
    result = pair_boundary_metrics(
        matrix, ["a", "b", "c", "d"], ["a", "b"], ["c", "d"]
    )
    assert result["left_within_affinity"] != result["right_within_affinity"]
    assert result["left_boundary_separation"] != result["right_boundary_separation"]


def test_structural_metrics_in_state_are_compact_but_artifact_metrics_are_not(tmp_path):
    raw = {
        "status": "success",
        "results": {"metrics": {
            "structural_characterization": {
                "internal_structure_by_set": {
                    "C1": {"probe_labels_by_case": {"a": 0}, "patient_silhouette": {"a": 1}}
                }
            }
        }},
        "artifacts": {"full_metrics": str(tmp_path / "metrics.json")},
    }
    compact = compact_tool_result(raw, "multimodal_consistency_check")
    structure = compact["full_metrics"]["structural_characterization"]
    assert "probe_labels_by_case" not in structure["internal_structure_by_set"]["C1"]
    assert "patient_silhouette" not in structure["internal_structure_by_set"]["C1"]


def test_resampling_config_is_validated_and_probe_can_be_partial():
    labels = np.asarray([0, 0, 1, 1, 1, 1])
    matrix = block_affinity(labels)
    with pytest.raises(ValueError):
        resampling_consensus(matrix, labels, iterations=0)
    with pytest.raises(ValueError):
        resampling_consensus(matrix, labels, fraction=1.0)
    with pytest.raises(ValueError):
        resampling_consensus(matrix, labels, pac_lower=0.9, pac_upper=0.1)
    result = compute_structural_characterization(
        block_affinity([0, 0, 1, 1]), {}, ["a", "b", "c", "d"], {"C1": ["a", "b", "c", "d"]}
    )
    probe = result["internal_structure_by_set"]["C1"]
    assert probe["comparison_status"] == "partially_estimable"
    assert probe["fused_binary_probe"]["resampling"]["comparison_status"] == "not_estimable"
