import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).with_name("13_experiment_four_view_macro_state_robustness.py")
spec = importlib.util.spec_from_file_location("macro_robustness", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cluster_summary_reports_candidate_ari_for_four_states():
    labels = list(module.CORE_TO_STATE)
    matrix = np.eye(len(labels))
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i != j and module.CORE_TO_STATE[left] == module.CORE_TO_STATE[right]:
                matrix[i, j] = 0.9
    result = module.cluster_summary(matrix, labels, 4, "average")
    assert result["actual_cluster_count"] == 4
    assert result["candidate_state_ari"] == 1.0


def test_cluster_summary_marks_one_cluster_silhouette_unavailable():
    result = module.cluster_summary(np.ones((4, 4)), list(module.CORE_TO_STATE)[:4], 2, "complete")
    assert result["actual_cluster_count"] == 1
    assert result["silhouette"] is None
