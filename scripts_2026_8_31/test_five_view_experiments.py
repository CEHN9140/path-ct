import numpy as np

from agents.candidate_proposer import consensus_records_from_similarity


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
