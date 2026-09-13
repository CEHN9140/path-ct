import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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
        matrices[name] = np.full((3, 3), index / 10, dtype=float)
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
        discovery_artifacts={
            "wxs_affinity_path": str(wxs_path),
            "cnv_affinity_path": str(cnv_path),
            "wxs_patient_order_path": str(order_path),
        },
        audit={},
    )

    assert set(paths) == {"ct", "wsi", "rna", "wxs", "cnv"}
    assert "genomic" not in paths


def test_candidate_payload_rejects_corrupt_affinity(tmp_path):
    import agents.candidate_proposer as proposer

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "candidate_proposer.yaml").write_text(
        "snf:\n  neighbor_count: 1\n  iterations: 1\n  alpha: 1.0\n",
        encoding="utf-8",
    )
    paths = {}
    for name in ("ct", "wsi", "rna", "wxs", "cnv"):
        matrix = np.eye(2)
        if name == "rna":
            matrix[0, 1] = np.nan
        path = tmp_path / f"{name}.npy"
        np.save(path, matrix)
        paths[name] = str(path)
    order_path = tmp_path / "order.json"
    order_path.write_text(json.dumps(["A", "B"]), encoding="utf-8")
    states = [
        {
            "case_id": case_id,
            "qc": "success",
            "omics_evidence": {
                "modality_affinity_paths": paths,
                "modality_affinity_patient_order_path": str(order_path),
            },
        }
        for case_id in ("A", "B")
    ]

    with pytest.raises(ValueError, match="rna affinity contains non-finite values"):
        proposer.build_feature_store_payload(
            states, config_dir=str(config_dir), output_root=str(tmp_path)
        )


def test_fusion_rejects_nonfinite_network():
    from tools.evidence_features import fuse_affinities

    with pytest.raises(ValueError, match="finite"):
        fuse_affinities([np.array([[1.0, np.nan], [np.nan, 1.0]])], {
            "neighbor_count": 1, "iterations": 1, "alpha": 1.0,
        })


def test_distance_to_affinity_preserves_valid_kernel_values_above_one():
    from tools.evidence_features import distance_to_affinity

    affinity = distance_to_affinity(
        np.array([[0.0, 1e-6], [1e-6, 0.0]]),
        {"neighbor_count": 1, "mu": 0.5},
    )
    assert float(affinity.max()) > 1.0


def test_k_selector_publishes_deterministic_diagnostics_not_free_text_claims(
    tmp_path, monkeypatch
):
    import agents.candidate_proposer as proposer

    prompt = tmp_path / "candidate_k_selector.md"
    prompt.write_text("Select K.", encoding="utf-8")
    monkeypatch.setattr(proposer, "load_candidate_proposer_config", lambda _: {
        "k_selector": {
            "prompt_path": str(prompt),
            "llm": {
                "provider": "deepseek", "model_name": "test", "temperature": 0,
                "max_new_tokens": 128, "json_retries": 0,
            },
        }
    })
    bad_reason = "K3 has weaker item consensus than K4."
    response = SimpleNamespace(
        id="response", model="test",
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                content=json.dumps({
                    "selected_k": 4,
                    "confidence": "medium",
                    "reasoning_summary": bad_reason,
                    "evidence_refs": ["K4.pac", "K4.item_consensus.p10"],
                }),
                reasoning_content="",
            ),
        )],
    )
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **_: response)
    ))
    records = []
    for k, pac, item_value in ((3, 0.7, 0.6), (4, 0.6, 0.5)):
        labels = np.repeat(np.arange(k), 2)
        consensus = np.full((2 * k, 2 * k), 0.1)
        for label in range(k):
            indexes = np.where(labels == label)[0]
            consensus[np.ix_(indexes, indexes)] = item_value
        np.fill_diagonal(consensus, 1.0)
        records.append({
            "n_clusters": k, "cluster_sizes": [2] * k,
            "consensus": consensus, "labels": labels,
            "relative_delta_area": 0.1, "pac": pac,
        })

    result = proposer.select_k_with_llm(
        records, output_root=str(tmp_path), config_dir=str(tmp_path),
        min_cluster_size=2, llm_client=client,
    )
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))

    assert result["selected_k"] == 4
    assert bad_reason not in result["reasoning_summary"]
    assert "K3[PAC=0.7" in result["reasoning_summary"]
    assert "item_p10=0.6" in result["reasoning_summary"]
    assert audit["llm_decision"]["reasoning_summary"] == bad_reason


def test_k_selector_rejects_unknown_evidence_reference(tmp_path, monkeypatch):
    import agents.candidate_proposer as proposer

    prompt = tmp_path / "candidate_k_selector.md"
    prompt.write_text("Select K.", encoding="utf-8")
    monkeypatch.setattr(proposer, "load_candidate_proposer_config", lambda _: {
        "k_selector": {
            "prompt_path": str(prompt),
            "llm": {
                "provider": "deepseek", "model_name": "test", "temperature": 0,
                "max_new_tokens": 128, "json_retries": 0,
            },
        }
    })
    response = SimpleNamespace(
        id="response", model="test",
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=json.dumps({
                "selected_k": 2, "confidence": "medium", "reasoning_summary": "x",
                "evidence_refs": ["K2.nonexistent"],
            }), reasoning_content=""),
        )],
    )
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **_: response)
    ))
    record = {
        "n_clusters": 2, "cluster_sizes": [2, 2],
        "consensus": np.eye(4), "labels": np.array([0, 0, 1, 1]),
        "relative_delta_area": 0.0, "pac": 0.5,
    }

    with pytest.raises(RuntimeError, match="unknown evidence_refs"):
        proposer.select_k_with_llm(
            [record], output_root=str(tmp_path), config_dir=str(tmp_path),
            min_cluster_size=2, llm_client=client,
        )
