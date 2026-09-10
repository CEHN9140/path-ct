import json

import numpy as np


def test_candidate_payload_uses_independent_five_views(tmp_path, monkeypatch):
    import agents.candidate_proposer as proposer

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "candidate_proposer.yaml").write_text(
        "snf:\n  neighbor_count: 1\n  iterations: 1\n  alpha: 1.0\n",
        encoding="utf-8",
    )
    paths = {}
    matrices = {}
    for index, name in enumerate(("ct", "wsi", "rna", "wxs", "cnv"), 1):
        path = tmp_path / f"{name}.npy"
        matrices[name] = np.full((3, 3), index, dtype=float)
        np.fill_diagonal(matrices[name], 1.0)
        np.save(path, matrices[name])
        paths[name] = str(path)
    order_path = tmp_path / "order.json"
    order_path.write_text(json.dumps(["A", "B", "C"]), encoding="utf-8")
    states = [{
        "case_id": case_id,
        "qc": "success",
        "omics_evidence": {
            "modality_affinity_paths": paths,
            "modality_affinity_patient_order_path": str(order_path),
        },
    } for case_id in ("A", "B", "C")]
    captured = {}
    def capture(networks, config):
        captured["networks"] = networks
        return np.eye(3)
    monkeypatch.setattr(
        proposer,
        "fuse_affinities",
        capture,
    )

    proposer.build_feature_store_payload(
        states, config_dir=str(config_dir), output_root=str(tmp_path)
    )

    assert len(captured["networks"]) == 5
    assert all(
        np.array_equal(matrix, matrices[name])
        for name, matrix in zip(("ct", "wsi", "rna", "wxs", "cnv"), captured["networks"])
    )


def test_candidate_affinity_manifest_keeps_wxs_and_cnv_independent(tmp_path):
    from agents.evidence_builder import save_modality_affinity_artifacts

    order_path = tmp_path / "order.json"
    order_path.write_text(json.dumps(["A", "B"]), encoding="utf-8")
    wxs_path, cnv_path = tmp_path / "wxs.npy", tmp_path / "cnv.npy"
    np.save(wxs_path, np.eye(2))
    np.save(cnv_path, np.eye(2))
    paths = save_modality_affinity_artifacts(
        output_root=str(tmp_path),
        patient_ids=["A", "B"],
        modality_affinities={name: np.eye(2) for name in ("ct", "wsi", "rna")},
        genomic_discovery={
            "wxs_affinity_path": str(wxs_path),
            "cnv_affinity_path": str(cnv_path),
            "wxs_patient_order_path": str(order_path),
        },
        audit={},
    )

    assert set(paths) == {"ct", "wsi", "rna", "wxs", "cnv"}
    assert "genomic" not in paths
