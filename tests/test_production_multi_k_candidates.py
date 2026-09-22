import numpy as np
import json
import pytest
from pathlib import Path


def test_candidate_proposer_builds_patient_resampled_partitions_for_every_k():
    from agents.candidate_proposer import build_patient_resampled_candidates

    rng = np.random.default_rng(11)
    n = 20
    affinities = {}
    for name in ("ct", "wsi", "rna", "wxs"):
        values = rng.uniform(0.05, 0.95, size=(n, n))
        values = (values + values.T) / 2
        np.fill_diagonal(values, 1.0)
        affinities[name] = values
    config = {
        "candidate_ks": [2, 3],
        "patient_resampling": {"fraction": 0.8, "count": 30, "random_seed": 7},
        "clustering": {
            "final_linkage": "average",
            "algorithms": {
                "hierarchical": {"linkage_options": ["average", "complete"]},
                "spectral": {"assign_labels_options": ["kmeans", "discretize", "cluster_qr"]},
                "kmedoids": {"init_options": ["k-medoids++", "random", "heuristic"]},
            },
        },
        "snf": {"neighbor_count": 5, "iterations": 5, "mu": 0.5, "alpha": 1.0},
    }

    records = build_patient_resampled_candidates(
        affinities, [f"P{i:02d}" for i in range(n)], config
    )

    assert set(records) == {2, 3}
    for k, record in records.items():
        assert len(record["labels"]) == n
        assert len(set(record["labels"])) == k
        assert record["consensus"].shape == (n, n)
        assert set(record["algorithm_consensus"]) == {"hierarchical", "spectral", "kmedoids"}
        assert record["pair_seen"].shape == (n, n)
        assert np.allclose(record["consensus"], record["consensus"].T)


def test_candidate_proposer_reuses_cached_consensus_for_agent_subset_runs(tmp_path, monkeypatch):
    import yaml

    import agents.candidate_proposer as proposer

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "candidate_proposer.yaml").write_text(yaml.safe_dump({
        "candidate_ks": [2],
        "patient_resampling": {"fraction": 1.0, "count": 1, "random_seed": 7},
        "clustering": {
            "final_linkage": "average",
            "algorithms": {
                "hierarchical": {"linkage_options": ["average"]},
                "spectral": {"assign_labels_options": ["cluster_qr"]},
                "kmedoids": {"init_options": ["heuristic"]},
            },
        },
        "snf": {"neighbor_count": 2, "iterations": 2, "mu": 0.5, "alpha": 1.0},
    }), encoding="utf-8")
    patient_ids = [f"P{index}" for index in range(4)]
    candidate_dir = tmp_path / "candidate_subtype"
    candidate_dir.mkdir()
    order_path = candidate_dir / "affinity_patient_order.json"
    order_path.write_text(json.dumps(patient_ids), encoding="utf-8")
    for modality in proposer.CANDIDATE_VIEWS:
        matrix = np.eye(4) + (np.ones((4, 4)) - np.eye(4)) * 0.4
        path = tmp_path / f"{modality}_affinity.npy"
        np.save(path, matrix)
    states = [{
        "case_id": patient_id, "qc": "success",
        "omics_evidence": {
            "modality_affinity_patient_order_path": str(order_path),
            "modality_affinity_paths": {
                name: str(tmp_path / f"{name}_affinity.npy")
                for name in proposer.CANDIDATE_VIEWS
            },
        },
    } for patient_id in patient_ids]

    first = proposer.candidate_proposer(states, output_root=str(tmp_path), config_dir=str(config_dir))
    assert set(first["candidate_partitions"]) == {2}
    candidate_json = json.loads(
        (candidate_dir / "consensus_cluster" / "consensus_hierarchical_K2.json").read_text()
    )
    generator = candidate_json["candidate_sets"][0]["generator"]
    assert generator["geometry"]["type"] == "resampled_consensus_coassignment"
    assert generator["geometry"]["matrix_relative_path"] == "consensus_cluster/consensus_matrix_K2.npy"
    fused = np.load(candidate_dir / "fused_similarity.npy")
    fused_distance = np.load(candidate_dir / "fused_distance.npy")
    assert np.allclose(fused_distance, 1.0 - np.clip((fused + fused.T) / 2, 0, 1))
    assert np.allclose(np.diag(fused_distance), 0.0)
    monkeypatch.setattr(proposer, "build_patient_resampled_candidates", lambda *args: pytest.fail("cache was recomputed"))
    second = proposer.candidate_proposer(states, output_root=str(tmp_path), config_dir=str(config_dir))
    assert second["candidate_signature"] == first["candidate_signature"]
    assert second["candidate_partitions"] == first["candidate_partitions"]


def test_partial_review_grid_uses_k_then_repeat_directories(tmp_path, monkeypatch):
    import agents.subtype_review.runner as runner

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for filename in ("m1_m4.csv", "clearcode.csv", "hallmark.gmt"):
        (tmp_path / filename).write_text("reference_status,reference_subtype\n", encoding="utf-8")
    (config_dir / "subtype_review.yaml").write_text(
        f"""budget: {{max_rounds: 10}}
multi_k:
  initial_ks: [2]
  repeats: [1, 2]
known_label_echo:
  mrna_m1_m4_path: {tmp_path / 'm1_m4.csv'}
  clearcode34_path: {tmp_path / 'clearcode.csv'}
rna:
  hallmark_gene_sets_path: {tmp_path / 'hallmark.gmt'}
prompt_dir: {Path(__file__).resolve().parents[1] / 'agents/subtype_review/prompts'}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "run_subtype_review", lambda *args: {})

    def save_summary(_state, run_root, direct=True):
        summary = {"status": "review_complete", "raw_control_status": "complete"}
        run_path = Path(run_root)
        (run_path / "final_subtype_sets.json").write_text("[]", encoding="utf-8")
        (run_path / "final_review_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(runner, "save_review_outputs", save_summary)
    result = runner.run_review_grid(
        {2: [{"set_id": "K2_C0001", "member_ids": ["P1", "P2"]}]},
        {"P1": {}, "P2": {}},
        str(tmp_path / "output"),
        str(config_dir),
        (2,),
        (1,),
        "candidate-signature",
    )

    assert result["multi_k_ready"] is False
    assert (tmp_path / "output/subtype_review/runs/K2/repeat1").is_dir()
    assert not (tmp_path / "output/subtype_review/multi_k").exists()


def test_full_review_grid_requires_every_accepted_set_artifact(tmp_path, monkeypatch):
    import agents.subtype_review.runner as runner

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name in ("m1_m4.csv", "clearcode.csv", "hallmark.gmt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    prompt_dir = Path(__file__).resolve().parents[1] / "agents/subtype_review/prompts"
    config_text = (
        f"known_label_echo:\n  mrna_m1_m4_path: {tmp_path / 'm1_m4.csv'}\n"
        f"  clearcode34_path: {tmp_path / 'clearcode.csv'}\n"
        f"rna:\n  hallmark_gene_sets_path: {tmp_path / 'hallmark.gmt'}\n"
        "multi_k:\n  initial_ks: [2, 3]\n  repeats: [1, 2]\n"
        f"prompt_dir: {prompt_dir}\n"
    )
    (config_dir / "subtype_review.yaml").write_text(config_text, encoding="utf-8")
    reviewed = []
    monkeypatch.setattr(runner, "run_subtype_review", lambda *args: reviewed.append(args) or {})

    def save_summary(_state, run_root, direct=True):
        run_path = Path(run_root)
        (run_path / "final_subtype_sets.json").write_text("[]", encoding="utf-8")
        summary = {"status": "review_complete", "raw_control_status": "complete"}
        (run_path / "final_review_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(runner, "save_review_outputs", save_summary)
    partitions = {k: [{"set_id": f"K{k}_C1", "member_ids": ["P1"]}] for k in (2, 3)}
    output_root = str(tmp_path / "output")
    result = runner.run_review_grid(
        partitions, {"P1": {}}, output_root, str(config_dir),
        (2, 3), (1, 2), "candidate-signature",
    )
    assert result["multi_k_ready"] is True
    assert len(reviewed) == 4

    (Path(result["run_root"]) / "K3" / "repeat2" / "final_subtype_sets.json").unlink()
    partial = runner.run_review_grid(
        partitions, {"P1": {}}, output_root, str(config_dir),
        (2,), (1,), "candidate-signature",
    )
    assert partial["multi_k_ready"] is False
    assert len(reviewed) == 4


def test_changing_only_configured_grid_reuses_existing_agent_run(tmp_path, monkeypatch):
    import agents.subtype_review.runner as runner

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name in ("m1_m4.csv", "clearcode.csv", "hallmark.gmt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    prompt_dir = Path(__file__).resolve().parents[1] / "agents/subtype_review/prompts"
    config_path = config_dir / "subtype_review.yaml"

    def write_config(repeats):
        config_path.write_text(
            f"known_label_echo:\n  mrna_m1_m4_path: {tmp_path / 'm1_m4.csv'}\n"
            f"  clearcode34_path: {tmp_path / 'clearcode.csv'}\n"
            f"rna:\n  hallmark_gene_sets_path: {tmp_path / 'hallmark.gmt'}\n"
            f"multi_k:\n  initial_ks: [2]\n  repeats: {repeats}\n"
            f"prompt_dir: {prompt_dir}\n",
            encoding="utf-8",
        )

    write_config("[1]")
    reviewed = []
    monkeypatch.setattr(runner, "run_subtype_review", lambda *args: reviewed.append(args) or {})

    def save_summary(_state, run_root, direct=True):
        summary = {"status": "review_complete", "raw_control_status": "complete"}
        run_path = Path(run_root)
        (run_path / "final_subtype_sets.json").write_text("[]", encoding="utf-8")
        (run_path / "final_review_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(runner, "save_review_outputs", save_summary)
    args = (
        {2: [{"set_id": "K2_C1", "member_ids": ["P1"]}]},
        {"P1": {}}, str(tmp_path / "output"), str(config_dir), (2,), (1,), "candidate-signature",
    )
    runner.run_review_grid(*args)
    write_config("[1, 2]")
    result = runner.run_review_grid(*args)

    assert len(reviewed) == 1
    assert result["multi_k_ready"] is False
