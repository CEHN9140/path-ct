import importlib.util
from pathlib import Path

import pytest


module_path = Path(__file__).with_name("01_experiment_five_view_four_state_macro_characterization.py")
spec = importlib.util.spec_from_file_location("four_state_macro", module_path)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_four_state_mapping_preserves_the_six_core_membership():
    cores = {
        "CORE01": [f"a{i}" for i in range(17)],
        "CORE02": [f"b{i}" for i in range(14)],
        "CORE03": [f"c{i}" for i in range(14)],
        "CORE04": [f"d{i}" for i in range(10)],
        "CORE05": [f"e{i}" for i in range(9)],
        "CORE06": [f"f{i}" for i in range(5)],
    }
    states = experiment.build_macro_states(cores)
    assert {key: len(value) for key, value in states.items()} == {
        "STATE_A": 17,
        "STATE_B": 14,
        "STATE_C": 14,
        "STATE_D": 24,
    }
    assert states["STATE_D"] == cores["CORE04"] + cores["CORE05"] + cores["CORE06"]


def test_validate_stable_cores_requires_six_disjoint_expected_cores():
    cores = {
        f"CORE0{i}": [f"{i}_{j}" for j in range(size)]
        for i, size in enumerate((17, 14, 14, 10, 9, 5), 1)
    }
    all_ids = set(sum(cores.values(), []))
    assert experiment.validate_stable_cores(cores, all_ids) == cores
    overlapping = dict(cores)
    overlapping["CORE06"] = ["1_0", "new_1", "new_2", "new_3", "new_4"]
    with pytest.raises(ValueError, match="互不重叠"):
        experiment.validate_stable_cores(overlapping, all_ids)


def test_evidence_comparison_uses_four_state_pair_denominator():
    six_pair = [{"core_a": "a", "core_b": "b", "q_value": 0.01}]
    four_pair = [{"core_a": "a", "core_b": "b", "q_value": None}]
    rows = experiment.evidence_comparison_rows(
        {f"CORE0{i}": [str(i)] for i in range(1, 7)},
        {"STATE_A": ["1"], "STATE_B": ["2"], "STATE_C": ["3"], "STATE_D": ["4", "5", "6"]},
        six_pair, four_pair,
        six_pair, four_pair,
        six_pair, four_pair,
        six_pair, four_pair,
    )
    assert rows[0]["six_core_total_pairs"] == 15
    assert rows[0]["four_state_total_pairs"] == 6
    assert rows[0]["six_core_supported_pairs"] == 1
    assert rows[0]["four_state_supported_pairs"] == 0
