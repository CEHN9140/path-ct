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


def test_candidate_mapping_has_four_states_and_ten_cores():
    assert len(module.CORE_TO_STATE) == 10
    assert set(module.CORE_TO_STATE.values()) == set(module.STATE_ORDER)
