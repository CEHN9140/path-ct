import numpy as np

from agents.candidate_proposer import consensus_records_from_similarity
from agents.candidate_proposer import build_k_selection_evidence
from utils.candidate_clustering_outputs import save_candidate_clustering_outputs


def test_five_view_consensus_uses_production_k_and_algorithms():
    matrix = np.eye(8, dtype=float)
    matrix[:4, :4] = 0.9
    matrix[4:, 4:] = 0.9
    np.fill_diagonal(matrix, 1.0)
    config = {
        "repeat_count": 2,
        "max_clusters": 3,
        "consensus_linkage": "average",
        "pac_lower": 0.1,
        "pac_upper": 0.9,
        "random_seed": 20260614,
        "algorithms": {
            "hierarchical": {"linkage_options": ["average"]},
            "spectral": {"assign_labels_options": ["kmeans"]},
            "kmedoids": {"init_options": ["heuristic"]},
        },
    }
    records, partitions = consensus_records_from_similarity(matrix, config)
    assert [record["n_clusters"] for record in records] == [2, 3]
    assert partitions
    assert all(record["valid"] for record in partitions)
    assert all(len(record["labels"]) == 8 for record in records)


def test_k_selection_evidence_declares_metric_directions():
    evidence = build_k_selection_evidence(
        [{
            "n_clusters": 2,
            "consensus": np.eye(4),
            "labels": (0, 0, 1, 1),
            "cluster_sizes": [2, 2],
            "pac": 0.5,
        }],
        min_cluster_size=1,
    )
    assert evidence["metric_directions"]["item_consensus.p10"] == "higher_is_better"
    assert evidence["metric_directions"]["pac"] == "lower_is_better"


def test_best_consensus_partition_preserves_pac_bounds(tmp_path):
    matrix = np.eye(8, dtype=float)
    matrix[:4, :4] = 0.9
    matrix[4:, 4:] = 0.9
    np.fill_diagonal(matrix, 1.0)
    config = {
        "repeat_count": 2,
        "max_clusters": 2,
        "consensus_linkage": "average",
        "pac_lower": 0.1,
        "pac_upper": 0.9,
        "random_seed": 20260614,
        "algorithms": {
            "hierarchical": {"linkage_options": ["average"]},
            "spectral": {"assign_labels_options": ["kmeans"]},
            "kmedoids": {"init_options": ["heuristic"]},
        },
    }
    records, partitions = consensus_records_from_similarity(matrix, config)
    save_candidate_clustering_outputs(
        str(tmp_path), [f"P{i}" for i in range(8)], partitions, records, records[0]
    )
    payload = (
        tmp_path / "candidate_subtype" / "consensus_cluster" / "best_consensus_partition.json"
    ).read_text(encoding="utf-8")
    assert '"pac_lower": 0.1' in payload
    assert '"pac_upper": 0.9' in payload
