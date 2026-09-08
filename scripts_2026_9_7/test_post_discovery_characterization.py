import numpy as np

from tools.post_discovery_characterization import (
    continuous_omnibus,
    continuous_posthoc,
    holm_adjust,
    stable_analysis_universe,
)


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
