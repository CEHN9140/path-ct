from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("05_experiment_core_non_core_audit.py")
SPEC = spec_from_file_location("core_non_core_audit", PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bh_sorts_by_p_value_not_input_index():
    q_values = MODULE.bh([0.9, 0.001, 0.2])
    assert q_values == pytest.approx([0.9, 0.003, 0.3])
