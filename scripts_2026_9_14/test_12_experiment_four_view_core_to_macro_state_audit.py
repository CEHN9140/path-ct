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
