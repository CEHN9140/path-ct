import numpy as np

from tools.post_discovery_characterization import (
    bh_adjust,
    binary_permutation_omnibus,
    continuous_omnibus,
    continuous_posthoc,
    holm_adjust,
    stable_analysis_universe,
)


def test_bh_adjust_sorts_by_p_value_before_adjusting():
    assert np.allclose(bh_adjust([0.736, 0.01, 0.2]), [0.736, 0.03, 0.3])


def test_holm_adjust_sorts_by_p_value_before_adjusting():
    assert np.allclose(holm_adjust([0.736, 0.01, 0.2]), [0.736, 0.03, 0.4])


def test_stable_analysis_universe_rejects_overlap_and_excludes_non_group_cases():
    assert stable_analysis_universe({"A": ["p2", "p1"], "B": ["p3"]}) == ["p1", "p2", "p3"]
    try:
        stable_analysis_universe({"A": ["p1"], "B": ["p1"]})
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping groups must be rejected")


def test_holm_adjust_preserves_none_and_is_not_smaller_than_raw_p():
    adjusted = holm_adjust([0.01, 0.04, None, 0.8])
    assert adjusted[2] is None
    assert adjusted[0] >= 0.01
    assert adjusted[1] >= 0.04
    assert adjusted[3] == 0.8


def test_continuous_omnibus_gates_posthoc_to_global_q_significant_features():
    groups = {"A": [f"a{i}" for i in range(5)], "B": [f"b{i}" for i in range(5)], "C": [f"c{i}" for i in range(5)]}
    table = {
        **{f"a{i}": {"signal": float(i) / 10, "noise": 1.0} for i in range(5)},
        **{f"b{i}": {"signal": 5 + float(i) / 10, "noise": 1.0} for i in range(5)},
        **{f"c{i}": {"signal": 10 + float(i) / 10, "noise": 1.0} for i in range(5)},
    }
    omnibus = continuous_omnibus(table, ["signal", "noise"], groups)
    selected = [row["feature"] for row in omnibus if row["q_value"] is not None and row["q_value"] < 0.05]
    assert selected == ["signal"]
    posthoc = continuous_posthoc(table, selected, groups, bootstrap_iterations=20, seed=42)
    assert {row["feature"] for row in posthoc} == {"signal"}
    assert all(row["cliffs_delta_ci_low"] is not None for row in posthoc)


def test_continuous_omnibus_marks_a_feature_not_estimable_if_a_group_is_empty():
    groups = {"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]}
    table = {
        "a1": {"x": 1.0}, "a2": {"x": 2.0},
        "b1": {"x": 3.0}, "b2": {"x": 4.0},
        "c1": {}, "c2": {},
    }
    row = continuous_omnibus(table, ["x"], groups)[0]
    assert row["comparison_status"] == "not_estimable"
    assert row["group_count"] == 3
    assert row["p_value"] is None


def test_binary_permutation_excludes_missing_values_instead_of_calling_them_wild_type():
    groups = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
    table = {"a1": {"m": 1}, "a2": {}, "b1": {"m": 0}, "b2": {"m": None}}
    row = binary_permutation_omnibus(table, ["m"], groups, permutations=20, seed=42)[0]
    assert row["available_n"] == 2
    assert row["missing_n"] == 2
    assert row["mutated_n"] == 1
    assert row["total_n"] == 2
