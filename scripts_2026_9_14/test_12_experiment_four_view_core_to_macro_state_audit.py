import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).with_name("12_experiment_four_view_core_to_macro_state_audit.py")
spec = importlib.util.spec_from_file_location("macro_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_normalize_affinity_uses_off_diagonal_scale():
    matrix = np.array([[4.0, 2.0, 6.0], [2.0, 9.0, 4.0], [6.0, 4.0, 1.0]])
    result = module.normalize_affinity(matrix)
    assert np.allclose(np.diag(result), 1.0)
    assert result.min() == 0.0
    assert result.max() == 1.0


def test_aggregate_core_matrix_preserves_core_order():
    cores = {"CORE01": ["A"], "CORE02": ["B"]}
    result = module.aggregate_core_matrix(np.array([[1.0, 0.2], [0.2, 1.0]]), ["A", "B"], cores)
    assert result.shape == (2, 2)
    assert np.allclose(result, [[1.0, 0.2], [0.2, 1.0]])


def test_hierarchy_uses_selected_macro_k():
    labels = [f"CORE{i:02d}" for i in range(1, 7)]
    matrix = np.full((6, 6), 0.1)
    np.fill_diagonal(matrix, 1.0)
    for left, right in ((0, 1), (2, 3), (4, 5)):
        matrix[left, right] = matrix[right, left] = 0.9
    states = {label: f"STATE_{'ABC'[index // 2]}" for index, label in enumerate(labels)}
    rows = module.hierarchy_rows(matrix, labels, "joint", states, macro_k=3)
    assert len({row["hierarchical_cluster"] for row in rows if row["linkage"] == "average"}) == 3
