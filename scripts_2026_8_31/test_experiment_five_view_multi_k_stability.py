import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("15_experiment_five_view_multi_k_stability.py")
SPEC = importlib.util.spec_from_file_location("five_view_multi_k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repeat_selection_does_not_append_to_default_repeats():
    assert MODULE.parse_repeats(None) == (1, 2, 3)
    assert MODULE.parse_repeats([1, 2]) == (1, 2)

