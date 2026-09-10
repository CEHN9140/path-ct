from pathlib import Path


def test_build_evidence_states_ignores_feature_values_when_collecting_paths(
    tmp_path, monkeypatch
):
    import agents.evidence_builder as evidence_builder
    import tools.evidence_features as evidence_features
    import tools.rna as rna
    import tools.wxs as wxs

    for name in ("ct.json", "wsi.npy", "rna.csv", "cnv.csv", "wxs.npy", "order.json"):
        (tmp_path / name).write_text("[]", encoding="utf-8")
    (tmp_path / "candidate_proposer.yaml").write_text("snf: {}\n", encoding="utf-8")
    (tmp_path / "candidate_subtype").mkdir()

    def bundle(path, signature):
        return {
            "tool_result": {
                "artifacts": {"case_features_path": str(path)},
                "provenance": {"signature": signature},
            },
            "payload": {"feature_values": [1.0]},
        }

    monkeypatch.setattr(rna, "build_rna_cohort_cache", lambda *args, **kwargs: {})
    monkeypatch.setattr(rna, "rna_signature_extra", lambda *_args: {})
    monkeypatch.setattr(rna, "run_case_rna_features", lambda **kwargs: bundle(tmp_path / "rna.csv", "rna"))
    monkeypatch.setattr(wxs, "build_wxs_cohort_cache", lambda *args, **kwargs: {})
    monkeypatch.setattr(wxs, "build_cnv_cohort_cache", lambda *args, **kwargs: {"signature": "cnv"})
    monkeypatch.setattr(wxs, "run_case_cnv_features", lambda **kwargs: bundle(tmp_path / "cnv.csv", "cnv"))
    monkeypatch.setattr(
        wxs,
        "build_wxs_cnv_artifacts",
        lambda *args, **kwargs: {
            "wxs_affinity_path": str(tmp_path / "wxs.npy"),
            "cnv_affinity_path": str(tmp_path / "cnv.csv"),
            "wxs_patient_order_path": str(tmp_path / "order.json"),
        },
    )
    monkeypatch.setattr(evidence_builder, "build_cohort_signature", lambda *args, **kwargs: "rna")
    monkeypatch.setattr(evidence_builder, "collect_case_file_paths", lambda *args, **kwargs: [])
    monkeypatch.setattr(evidence_builder, "load_tool_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evidence_builder,
        "load_yaml_file",
        lambda path: {
            "ccc_threshold": 0.8,
            "ccc_comparison_bin_widths": [20],
            "confound_correction": {"enabled": False, "fields": []},
        }
        if Path(path).name == "ct_radiomics.yaml"
        else {},
    )
    monkeypatch.setattr(
        evidence_features,
        "build_modality_affinity_artifacts",
        lambda *args, **kwargs: {
            "modality_affinities": {name: None for name in ("ct", "wsi", "rna", "wxs", "cnv")},
            "audit": {},
        },
    )
    monkeypatch.setattr(
        evidence_builder,
        "save_modality_affinity_artifacts",
        lambda **kwargs: {
            "ct": str(tmp_path / "ct_affinity.npy"),
            "wsi": str(tmp_path / "wsi_affinity.npy"),
            "rna": str(tmp_path / "rna_affinity.npy"),
            "wxs": str(tmp_path / "wxs.npy"),
            "cnv": str(tmp_path / "cnv.csv"),
        },
    )
    monkeypatch.setattr(
        "tools.confound.confounder_values",
        lambda *args, **kwargs: {},
    )

    state = {
        "case_id": "A",
        "qc": "success",
        "inventory": {
            "Case_ID": "A",
            "CT": [{"File Path": str(tmp_path / "ct.json")}],
            "WSI": [{"File Path": str(tmp_path / "wsi.npy")}],
            "RNA_Seq": [{"File Path": str(tmp_path / "rna.csv")}],
            "WXS": [{"File Path": str(tmp_path / "wxs.npy")}],
            "CNV": [{"File Path": str(tmp_path / "cnv.csv")}],
        },
        "ct_evidence": {
            "features": {f"feature_{index}": index for index in range(500)},
            "feature_path": str(tmp_path / "ct.json"),
            "ccc_feature_paths": {"20": str(tmp_path / "ct.json")},
        },
        "wsi_evidence": {
            "features": [0.1, 0.2],
            "feature_path": str(tmp_path / "wsi.npy"),
        },
    }

    result = evidence_builder.build_evidence_states(
        [state], output_root=str(tmp_path), config_dir=str(tmp_path)
    )

    assert result[0]["qc"] == "success"
    assert set(result[0]["omics_evidence"]["modality_affinity_paths"]) == {
        "ct",
        "wsi",
        "rna",
        "wxs",
        "cnv",
    }
