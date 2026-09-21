import json
import sys
import types

import numpy as np
import pandas as pd


def test_wxs_enrichment_tests_full_matrix_and_returns_all_drivers(tmp_path):
    from agents.subtype_review.tools import wxs_mutation_enrichment

    wxs_dir = tmp_path / "wxs"
    wxs_dir.mkdir()
    pd.DataFrame({
        "case_id": ["P1", "P2", "P3", "P4"],
        "VHL": [1, 1, 0, 0],
        "OTHER": [0, 1, 0, 1],
        "PBRM1": [0, 0, 0, 0],
    }).to_csv(wxs_dir / "wxs_interpretation_features.csv", index=False)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "wxs.yaml").write_text(
        "biological_support:\n  driver_genes: [VHL, PBRM1]\n  exploratory_report_top_n: 1\n"
    )

    result = wxs_mutation_enrichment(
        {}, str(tmp_path), str(config_dir),
        [
            {"set_id": "C1", "member_ids": ["P1", "P2"]},
            {"set_id": "C2", "member_ids": ["P3", "P4"]},
        ],
        "set", ["C1"],
    )
    metrics = result["metrics"]["set"]["C1"]

    assert metrics["tested_gene_n"] == 3
    assert {row["gene"] for row in metrics["driver_panel"]["results"]} == {"PBRM1", "VHL"}
    assert all(row["q_global"] is not None and row["q_driver"] is not None for row in metrics["driver_panel"]["results"])
    pbrm1 = next(row for row in metrics["driver_panel"]["results"] if row["gene"] == "PBRM1")
    assert pbrm1["set_mutated_n"] == 0
    assert pbrm1["zero_cell_correction_applied"] is True
    assert len(metrics["exploratory_top_genes"]) == 1
    complete_table = pd.read_csv(result["artifact_paths"]["C1"])
    assert set(complete_table["gene"]) == {"VHL", "OTHER", "PBRM1"}
    assert complete_table["q_global"].notna().all()
    assert result["artifact_paths"]["C1"].endswith(".csv")
    from agents.subtype_review.tools import compact_tool_result
    assert compact_tool_result(result, "wxs_mutation_enrichment")["artifact_paths"] == result["artifact_paths"]


def test_representation_concordance_uses_native_distances_and_candidate_labels(tmp_path):
    from agents.subtype_review.tools import representation_concordance

    patient_ids = [f"P{i}" for i in range(6)]
    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text(json.dumps(patient_ids))
    modalities = ("ct", "wsi", "rna", "wxs")
    for offset, modality in enumerate(modalities):
        matrix = np.full((6, 6), 0.2 + offset * 0.05)
        np.fill_diagonal(matrix, 0.0)
        np.save(candidate / f"{modality}_distance.npy", matrix)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        "cross_modal:\n  concordance:\n    permutations: 3\n"
        "    bootstrap_repeats: 4\n    random_seed: 9\n",
        encoding="utf-8",
    )

    result = representation_concordance(
        {}, str(tmp_path), [
            {"set_id": "C1", "member_ids": patient_ids[:3]},
            {"set_id": "C2", "member_ids": patient_ids[3:]},
        ], "partition", [], str(config_dir),
    )["metrics"]["partition"]["partition"]

    assert set(result["geometry_concordance"]) == {
        "ct__wsi", "ct__rna", "ct__wxs", "wsi__rna", "wsi__wxs", "rna__wxs",
    }
    assert all("bootstrap_ci95" in row and "permutation_p" in row for row in result["geometry_concordance"].values())
    assert set(result["current_membership_alignment"]) == set(modalities)
    assert result["patient_n"] == 6
    assert abs(result["current_membership_alignment"]["ct"]["silhouette"]) < 1e-12
    assert np.isclose(result["current_membership_alignment"]["ct"]["mean_within_distance"], 0.2)

    target_result = representation_concordance(
        {}, str(tmp_path), [
            {"set_id": "C1", "member_ids": patient_ids[:3]},
            {"set_id": "C2", "member_ids": patient_ids[3:]},
        ], "set", ["C1"], str(config_dir),
    )["metrics"]["set"]["C1"]
    assert target_result["comparison"] == "target_vs_rest"
    assert target_result["patient_n"] == 6


def test_rna_pathway_review_uses_deseq2_wald_rank_and_raw_counts(tmp_path, monkeypatch):
    from agents.subtype_review.tools import rna_pathway_enrichment

    counts_path = tmp_path / "raw_counts.csv"
    pd.DataFrame({
        "case_id": ["P1", "P2", "P3", "P4"],
        "G1": [10, 12, 2, 1],
        "G2": [4, 5, 3, 3],
    }).to_csv(counts_path, index=False)
    gmt = tmp_path / "hallmark.gmt"
    gmt.write_text("HALLMARK_TEST\tdesc\tG1\tG2\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        f"rna:\n  hallmark_gene_sets_path: {gmt}\n  min_pathway_overlap: 2\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeDeseqDataSet:
        def __init__(self, **kwargs):
            captured["dds"] = kwargs

        def deseq2(self):
            captured["deseq2_called"] = True

    class FakeDeseqStats:
        def __init__(self, dds, **kwargs):
            captured["stats"] = kwargs
            self.statistics = np.array([2.0, -1.0])

        def run_wald_test(self):
            captured["wald_called"] = True

    def fake_prerank(**kwargs):
        captured["ranking"] = kwargs["rnk"]
        return types.SimpleNamespace(res2d=pd.DataFrame([{
            "Term": "HALLMARK_TEST", "NES": 1.5, "FDR q-val": 0.02, "Lead_genes": "G1"
        }]))

    pydeseq2 = types.ModuleType("pydeseq2")
    dds_module = types.ModuleType("pydeseq2.dds")
    dds_module.DeseqDataSet = FakeDeseqDataSet
    ds_module = types.ModuleType("pydeseq2.ds")
    ds_module.DeseqStats = FakeDeseqStats
    monkeypatch.setitem(sys.modules, "pydeseq2", pydeseq2)
    monkeypatch.setitem(sys.modules, "pydeseq2.dds", dds_module)
    monkeypatch.setitem(sys.modules, "pydeseq2.ds", ds_module)
    monkeypatch.setitem(sys.modules, "gseapy", types.SimpleNamespace(prerank=fake_prerank))
    states = {
        case_id: {"omics_evidence": {"rna_raw_counts_path": str(counts_path)}}
        for case_id in ("P1", "P2", "P3", "P4")
    }

    result = rna_pathway_enrichment(
        states, str(tmp_path), str(config_dir),
        [{"set_id": "C1", "member_ids": ["P1", "P2"]}, {"set_id": "C2", "member_ids": ["P3", "P4"]}],
        "set", ["C1"],
    )
    metrics = result["metrics"]["set"]["C1"]

    assert captured["dds"]["counts"].loc["P1", "G1"] == 10
    assert captured["dds"]["metadata"].loc["P1", "condition"] == "target"
    assert captured["dds"]["metadata"].loc["P3", "condition"] == "rest"
    assert captured["stats"]["contrast"] == ["condition", "target", "rest"]
    assert captured["ranking"]["stat"].tolist() == [2.0, -1.0]
    assert metrics["ranking_method"] == "pydeseq2_wald_statistic"


def test_confounder_geometry_uses_native_distance_and_separate_test_families(tmp_path, monkeypatch):
    from agents.subtype_review import tools as review_tools

    patient_ids = ["P1", "P2", "P3", "P4"]
    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text(json.dumps(patient_ids))
    ct_distance = np.array([
        [0, 1, 1, 2**0.5], [1, 0, 2**0.5, 1],
        [1, 2**0.5, 0, 1], [2**0.5, 1, 1, 0],
    ])
    fused_distance = np.array([
        [0, 0.8, 0.6, 0.4], [0.8, 0, 0.5, 0.7],
        [0.6, 0.5, 0, 0.9], [0.4, 0.7, 0.9, 0],
    ])
    for modality in ("ct", "wsi", "rna", "wxs"):
        np.save(candidate / f"{modality}_distance.npy", ct_distance)
    np.save(candidate / "fused_distance.npy", fused_distance)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        "confounder:\n  permutations: 9\n  random_seed: 7\n", encoding="utf-8"
    )
    values = {
        "P1": {"tissue_source_site": "A", "ct_phase": "X", "ct_slice_thickness": 1.0},
        "P2": {"tissue_source_site": "A", "ct_phase": "Y", "ct_slice_thickness": 3.0},
        "P3": {"tissue_source_site": "B", "ct_phase": "X", "ct_slice_thickness": 2.0},
        "P4": {"tissue_source_site": "B", "ct_phase": "Y", "ct_slice_thickness": 5.0},
    }
    monkeypatch.setattr(review_tools, "technical_values", lambda *_: values)
    permanova_distances = []
    permanova_p = iter([0.01, 0.04])
    permdisp_p = iter([0.02, 0.08])

    class DistanceMatrix:
        def __init__(self, data, ids):
            self.data = np.asarray(data)
            self.ids = ids

    def fake_permanova(dm, **kwargs):
        permanova_distances.append(dm.data.copy())
        return {"test statistic": 2.0, "p-value": next(permanova_p)}

    def fake_permdisp(dm, **kwargs):
        return {"test statistic": 1.5, "p-value": next(permdisp_p)}

    skbio = types.ModuleType("skbio")
    skbio.DistanceMatrix = DistanceMatrix
    skbio_stats = types.ModuleType("skbio.stats")
    distance_stats = types.ModuleType("skbio.stats.distance")
    distance_stats.permanova = fake_permanova
    distance_stats.permdisp = fake_permdisp
    monkeypatch.setitem(sys.modules, "skbio", skbio)
    monkeypatch.setitem(sys.modules, "skbio.stats", skbio_stats)
    monkeypatch.setitem(sys.modules, "skbio.stats.distance", distance_stats)

    result = review_tools.confounder_representation_effect(
        {}, str(tmp_path), str(config_dir), [], "partition", [],
    )["metrics"]["partition"]

    assert np.array_equal(permanova_distances[0], fused_distance)
    assert np.array_equal(permanova_distances[1], ct_distance)
    assert result["tissue_source_site"]["permanova"]["r_squared"] == 0.5
    assert result["tissue_source_site"]["permanova"]["q_value"] == 0.02
    assert result["tissue_source_site"]["permdisp"]["q_value"] == 0.04
    assert result["ct_phase"]["permdisp"]["q_value"] == 0.08
    assert "distance_regression" in result["ct_slice_thickness"]
