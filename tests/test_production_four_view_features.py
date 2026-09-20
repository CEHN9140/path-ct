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
        "capture_size_mb": 38.0,
        "functional_classes": {},
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
    audit = json.loads((tmp_path / "wxs" / "wxs_discovery_audit.json").read_text())
    distance = wxs.binary_mutation_distance(np.zeros((2, 1), dtype=bool), 1.0)
    assert list(discovery.columns) == ["case_id", "mutation::GENE1"]
    assert "mutation::VHL" not in discovery.columns
    assert audit["empty_mutation_distance"] == 1.0
    assert distance[0, 1] == 1.0
