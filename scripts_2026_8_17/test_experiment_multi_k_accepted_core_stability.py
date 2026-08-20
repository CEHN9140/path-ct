from __future__ import annotations

import numpy as np
import pytest

from scripts_2026_8_17 import experiment_multi_k_accepted_core_stability as experiment


def test_accepted_assignments_exclude_dropped_sets():
    sets = [
        {"set_id": "A", "status": "provisionally_accepted", "member_ids": ["p1", "p2"]},
        {"set_id": "D", "status": "provisionally_dropped", "member_ids": ["p3"]},
    ]
    assert experiment.accepted_assignments(sets) == {"p1": "A", "p2": "A"}


def test_compare_runs_uses_shared_accepted_patients():
    result = experiment.compare_runs(
        {"p1": "A", "p2": "A", "p3": "B"},
        {"p1": "X", "p2": "X", "p4": "Y"},
    )
    assert result["accepted_union_count"] == 4
    assert result["accepted_intersection_count"] == 2
    assert result["accepted_coverage_jaccard"] == 0.5
    assert result["adjusted_rand_index"] == 1.0
    assert result["adjusted_mutual_information"] == 1.0
    assert result["matched_mean_jaccard"] == pytest.approx(0.5)
    assert result["matched_mean_dice"] == pytest.approx(0.5)


def test_coassignment_uses_all_runs_as_denominator():
    patients = ["p1", "p2", "p3"]
    runs = [
        {"p1": "A", "p2": "A"},
        {"p1": "B", "p2": "B", "p3": "C"},
        {"p1": "A", "p3": "A"},
    ]
    matrix, acceptance = experiment.coassignment(runs, patients)
    assert matrix[0, 1] == pytest.approx(2 / 3)
    assert matrix[0, 2] == pytest.approx(1 / 3)
    assert matrix[1, 2] == 0
    assert acceptance.tolist() == pytest.approx([1.0, 2 / 3, 2 / 3])


def test_extract_cores_uses_complete_linkage():
    patients = ["a1", "a2", "a3", "b1", "b2"]
    matrix = np.array(
        [
            [1, 0.9, 0.8, 0.1, 0.1],
            [0.9, 1, 0.85, 0.1, 0.1],
            [0.8, 0.85, 1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 1, 0.9],
            [0.1, 0.1, 0.1, 0.9, 1],
        ]
    )
    cores = experiment.extract_cores(
        matrix, np.ones(5), patients, threshold=0.8, min_size=2
    )
    assert [row["member_ids"] for row in cores] == [
        ["a1", "a2", "a3"],
        ["b1", "b2"],
    ]


def test_core_recurrence_reports_run_and_k_coverage():
    runs = [
        {
            "initial_k": 2,
            "repeat": 1,
            "assignments": {"a1": "A", "a2": "A", "a3": "A"},
        },
        {
            "initial_k": 2,
            "repeat": 2,
            "assignments": {"a1": "A", "a2": "A", "a3": "A"},
        },
        {
            "initial_k": 3,
            "repeat": 1,
            "assignments": {"a1": "X", "a2": "X", "a3": "Y"},
        },
    ]
    result = experiment.core_recurrence(["a1", "a2", "a3"], runs)
    assert result["run_recurrence_count"] == 2
    assert result["run_recurrence_fraction"] == pytest.approx(2 / 3)
    assert result["k_coverage_any"] == 1
    assert result["k_coverage_majority"] == 1
