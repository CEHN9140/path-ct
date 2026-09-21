import gzip
import json

import numpy as np
import pandas as pd


def test_ct_candidate_features_drop_constant_and_correlated_columns_then_zscore(
    tmp_path, monkeypatch
):
    import tools.ct_radiomics as ct

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "ct_radiomics.yaml").write_text("{}", encoding="utf-8")
    (config_dir / "candidate_proposer.yaml").write_text("snf: {}", encoding="utf-8")
    monkeypatch.setattr("utils.llm_utils.load_yaml_file", lambda _path: {})
    monkeypatch.setattr("utils.llm_utils.load_candidate_proposer_config", lambda _path: {
        "snf": {"ct_high_correlation_threshold": 0.95}
    })

    states = []
    vectors = {
        "a": {"constant": 1, "shape_a": 0, "shape_b": 0, "texture": 2},
        "b": {"constant": 1, "shape_a": 1, "shape_b": 1, "texture": 0},
        "c": {"constant": 1, "shape_a": 2, "shape_b": 2, "texture": 3},
        "d": {"constant": 1, "shape_a": 3, "shape_b": 3, "texture": 1},
        "e": {"constant": 1, "shape_a": 4, "shape_b": 4, "texture": 4},
    }
    for case_id, values in vectors.items():
        path = tmp_path / f"{case_id}.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        states.append({"case_id": case_id, "qc": "success", "ct_evidence": {"feature_path": str(path)}})

    result = ct.build_ct_discovery_feature_matrix(states, config_dir=str(config_dir))

    assert result["feature_names"] == ["shape_a", "texture"]
    assert np.allclose(result["matrix"].mean(axis=0), 0.0)
    assert np.allclose(result["matrix"].std(axis=0), 1.0)
    assert result["audit"]["technical_residualization"] is False
    assert result["audit"]["ccc_filter"] is False


def test_wxs_discovery_uses_prevalence_only_and_empty_pair_distance_one(
    tmp_path, monkeypatch
):
    import tools.wxs as wxs

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "candidate_proposer.yaml").write_text(
        "snf:\n  neighbor_count: 1\n  mu: 0.5\n", encoding="utf-8"
    )
    monkeypatch.setattr(wxs, "load_wxs_config", lambda _path: {
        "min_gene_prevalence": 0.5,
        "empty_mutation_distance": 1.0,
        "nonsynonymous_classes": ["Missense_Mutation"],
        "biological_support": {"driver_genes": ["VHL"]},
        "functional_classes": {"missense": ["Missense_Mutation"]},
    })

    paths = []
    for case_id, rows in {
        "A": [("GENE1", "Missense_Mutation")],
        "B": [("GENE1", "Missense_Mutation")],
    }.items():
        path = tmp_path / f"{case_id}.maf.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("Hugo_Symbol\tVariant_Classification\tVariant_Type\tChromosome\tStart_Position\n")
            for gene, classification in rows:
                handle.write(f"{gene}\t{classification}\tSNP\t3\t100\n")
        paths.append({"case_id": case_id, "file_path": str(path)})
    cache = {"manifest_path": str(tmp_path / "wxs_manifest.json"), "signature": "sig"}
    (tmp_path / "wxs_manifest.json").write_text(json.dumps({"input_cases": paths}), encoding="utf-8")

    artifacts = wxs.build_wxs_artifacts(
        cache,
        [{"Case_ID": "A"}, {"Case_ID": "B"}],
        output_root=str(tmp_path),
        config_dir=str(config_dir),
    )

    discovery = pd.read_csv(artifacts["wxs_discovery_feature_path"])
    interpretation = pd.read_csv(artifacts["wxs_interpretation_feature_path"])
    validation = pd.read_csv(artifacts["wxs_validation_feature_path"])
    audit = json.loads((tmp_path / "wxs" / "wxs_discovery_audit.json").read_text())
    distance = wxs.binary_mutation_distance(np.zeros((2, 1), dtype=bool), 1.0)
    assert list(discovery.columns) == ["case_id", "mutation::GENE1"]
    assert "mutation::VHL" not in discovery.columns
    assert list(interpretation.columns) == ["case_id", "GENE1", "VHL"]
    assert interpretation["VHL"].sum() == 0
    assert "estimated_TMB" not in " ".join(validation.columns)
    assert audit["empty_mutation_distance"] == 1.0
    assert "estimated_TMB" not in str(audit)
    assert distance[0, 1] == 1.0


def test_rna_quantification_sums_raw_counts_but_preserves_max_tpm(tmp_path):
    from tools.rna import read_rna_quantification

    path = tmp_path / "counts.tsv"
    path.write_text(
        "# metadata\n"
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\t"
        "tpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
        "# metadata\n# metadata\n# metadata\n# metadata\n"
        "ENSG1\tGENE1\tprotein_coding\t10\t0\t0\t2\t0\t0\n"
        "ENSG2\tGENE1\tprotein_coding\t20\t0\t0\t3\t0\t0\n"
        "ENSG3\tMIR1\tmiRNA\t5\t0\t0\t9\t0\t0\n",
        encoding="utf-8",
    )

    result = read_rna_quantification(str(path))

    assert result["tpm"].to_dict() == {"GENE1": 3.0}
    assert result["raw_counts"].to_dict() == {"GENE1": 30}


def test_rna_quantification_rejects_noninteger_raw_counts(tmp_path):
    from tools.rna import read_rna_quantification

    path = tmp_path / "counts.tsv"
    path.write_text(
        "# metadata\n"
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\t"
        "tpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
        "# metadata\n# metadata\n# metadata\n# metadata\n"
        "ENSG1\tGENE1\tprotein_coding\t2.5\t0\t0\t2\t0\t0\n",
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="integer-valued"):
        read_rna_quantification(str(path))


def test_rna_cohort_cache_writes_patient_by_gene_raw_counts(tmp_path):
    from tools.rna import build_rna_cohort_cache

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "rna.yaml").write_text(
        "top_gene_count: 1\nprotein_coding_only: true\nmin_tpm: 0.1\nmin_expressed_fraction: 0.5\n",
        encoding="utf-8",
    )
    cases = []
    for case_id, count, tpm in (("P1", 10, 2), ("P2", 20, 3)):
        path = tmp_path / f"{case_id}.tsv"
        path.write_text(
            "# metadata\n"
            "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\t"
            "tpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
            "# metadata\n# metadata\n# metadata\n# metadata\n"
            f"ENSG1\tGENE1\tprotein_coding\t{count}\t0\t0\t{tpm}\t0\t0\n"
            "ENSG2\tZERO\tprotein_coding\t0\t0\t0\t0\t0\t0\n",
            encoding="utf-8",
        )
        cases.append({"Case_ID": case_id, "RNA_Seq": [{"File Path": str(path)}]})

    result = build_rna_cohort_cache(cases, output_root=str(tmp_path / "out"), config_dir=str(config_dir))
    raw_counts = pd.read_csv(result["raw_counts_path"]).set_index("case_id")

    assert list(raw_counts.index) == ["P1", "P2"]
    assert list(raw_counts.columns) == ["GENE1"]
    assert raw_counts["GENE1"].tolist() == [10, 20]
    assert "raw_counts" in result["manifest"]["files"]


def test_modality_affinity_artifact_writer_saves_distances_and_affinities(tmp_path):
    from agents.evidence_builder import save_modality_affinity_artifacts

    matrices = {
        name: np.array([[0.0, 0.4], [0.4, 0.0]])
        for name in ("ct", "wsi", "rna", "wxs")
    }
    affinities = {name: 1.0 - matrix for name, matrix in matrices.items()}

    paths = save_modality_affinity_artifacts(
        output_root=str(tmp_path),
        patient_ids=["P1", "P2"],
        modality_affinities=affinities,
        modality_distances=matrices,
        audit={},
    )

    assert set(paths) == {"affinity_paths", "distance_paths"}
    assert all(np.array_equal(np.load(path), matrices[name]) for name, path in paths["distance_paths"].items())
