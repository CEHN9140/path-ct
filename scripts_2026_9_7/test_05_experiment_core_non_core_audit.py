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


def test_binary_categorical_comparison_uses_fisher_exact():
    records = {
        "a": {"event": "1"}, "b": {"event": "1"},
        "c": {"event": "0"}, "d": {"event": "0"},
    }
    rows = MODULE.categorical_rows(
        records, {"core": ["a", "b"], "non_core": ["c", "d"]}, ("event",)
    )
    assert {row["test"] for row in rows} == {"fisher_exact"}
