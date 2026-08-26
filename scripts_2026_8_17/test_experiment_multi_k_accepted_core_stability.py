from __future__ import annotations

import csv
import json
import numpy as np
import pytest

from scripts_2026_8_17 import experiment_multi_k_accepted_core_stability as experiment


def test_accepted_assignments_exclude_dropped_sets():
    sets = [{"set_id": "A", "member_ids": ["p1", "p2"]}]
    assert experiment.accepted_assignments(sets) == {"p1": "A", "p2": "A"}


def test_scientifically_terminal_requires_normal_completion():
    assert experiment.scientifically_terminal({"status": "review_complete", "raw_control_status": "complete"})
    assert not experiment.scientifically_terminal({"status": "review_incomplete_due_to_round_budget", "raw_control_status": "review_incomplete_due_to_round_budget"})


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
    assert result["matched_set_pairs"][0]["overlap"] == pytest.approx(1.0)


def test_set_family_catalog_reports_overlap_and_cross_run_family():
    runs = [
        {"run_id": "run1_K2", "initial_k": 2, "repeat": 1, "assignments": {"p1": "A", "p2": "A", "p3": "B"}},
        {"run_id": "run1_K3", "initial_k": 3, "repeat": 1, "assignments": {"p1": "X", "p2": "X", "p3": "Y"}},
    ]
    catalog = experiment.accepted_set_catalog(runs)
    similarities = experiment.accepted_set_similarity(catalog)
    families = experiment.accepted_set_families(catalog, similarities, jaccard_threshold=0.5, overlap_threshold=1.0)
    assert similarities[0]["jaccard"] == pytest.approx(2 / 2)
    assert similarities[0]["overlap"] == pytest.approx(1.0)
    assert families[0]["node_ids"] == ["run1_K2::A", "run1_K3::X"]
    assert families[0]["k_coverage"] == 2


def test_relation_component_with_same_run_conflict_is_not_cohesive_family():
    runs = [
        {"run_id": "run1_K2", "initial_k": 2, "repeat": 1, "assignments": {"p1": "A1", "p2": "A1", "p3": "A2", "p4": "A2"}},
        {"run_id": "run2_K3", "initial_k": 3, "repeat": 2, "assignments": {"p1": "B", "p2": "B", "p3": "B", "p4": "B"}},
    ]
    catalog = experiment.accepted_set_catalog(runs)
    families = experiment.accepted_set_families(
        catalog,
        experiment.accepted_set_similarity(catalog),
        jaccard_threshold=0.5,
        overlap_threshold=0.8,
    )
    assert len(families) == 1
    assert families[0]["is_recurrent_relation_component"] is True
    assert families[0]["is_cohesive_family"] is False
    assert families[0]["same_run_conflict_count"] == 1


def test_relation_type_is_scoped_to_the_two_runs_being_compared():
    runs = [
        {"run_id": "run1_K3", "initial_k": 3, "repeat": 1, "assignments": {"p1": "A"}},
        {"run_id": "run1_K4", "initial_k": 4, "repeat": 1, "assignments": {"p1": "B"}},
        {"run_id": "run1_K5", "initial_k": 5, "repeat": 1, "assignments": {"p1": "C"}},
    ]
    rows = experiment.accepted_set_similarity(experiment.accepted_set_catalog(runs))
    assert {row["relation_type"] for row in rows} == {"one_to_one"}


def test_relation_type_reports_many_to_one_within_a_run_pair():
    runs = [
        {
            "run_id": "run1_K3",
            "initial_k": 3,
            "repeat": 1,
            "assignments": {"p1": "A1", "p2": "A1", "p3": "A2", "p4": "A2"},
        },
        {
            "run_id": "run2_K4",
            "initial_k": 4,
            "repeat": 2,
            "assignments": {"p1": "B", "p2": "B", "p3": "B", "p4": "B"},
        },
    ]
    rows = experiment.accepted_set_similarity(experiment.accepted_set_catalog(runs))
    assert {row["relation_type"] for row in rows} == {"many_to_one"}


def test_cohesive_subfamily_is_recovered_inside_a_bridged_component():
    runs = [
        {"run_id": "run1_K2", "initial_k": 2, "repeat": 1, "assignments": {"p1": "A", "p2": "A"}},
        {"run_id": "run1_K3", "initial_k": 3, "repeat": 1, "assignments": {"p1": "B", "p2": "B"}},
        {"run_id": "run1_K4", "initial_k": 4, "repeat": 1, "assignments": {"p1": "C", "p2": "C", "p3": "C", "p4": "C"}},
        {"run_id": "run1_K5", "initial_k": 5, "repeat": 1, "assignments": {"p1": "D", "p2": "D", "p3": "D", "p4": "D", "p5": "D"}},
    ]
    catalog = experiment.accepted_set_catalog(runs)
    similarities = experiment.accepted_set_similarity(catalog)
    components = experiment.accepted_set_families(catalog, similarities, jaccard_threshold=0.5, overlap_threshold=0.8)
    families = experiment.cohesive_set_families(catalog, similarities, components, jaccard_threshold=0.5, overlap_threshold=0.8)
    assert families[0]["node_ids"] == ["run1_K2::A", "run1_K3::B", "run1_K4::C"]


def test_cohesive_family_reports_within_k_membership_fraction():
    runs = [
        {"run_id": "run1_K2", "initial_k": 2, "repeat": 1, "assignments": {"p1": "A", "p2": "A"}},
        {"run_id": "run2_K2", "initial_k": 2, "repeat": 2, "assignments": {"p1": "B", "p2": "B"}},
        {"run_id": "run1_K3", "initial_k": 3, "repeat": 1, "assignments": {"p1": "C"}},
    ]
    families = experiment.family_layer(
        runs, jaccard_threshold=0.5, overlap_threshold=0.8
    )["families"]
    assert len(families) == 1
    assert families[0]["patient_unconditional_membership_by_k"]["p2"] == {
        "2": 1.0, "3": 0.0
    }
    assert families[0]["patient_unconditional_membership_by_k"]["p1"] == {
        "2": 1.0, "3": 1.0
    }


def test_family_presence_and_unconditional_membership_include_missing_repeats():
    runs = [
        {"run_id": "run1_K3", "initial_k": 3, "repeat": 1, "assignments": {"p1": "A", "p2": "A"}},
        {"run_id": "run1_K4", "initial_k": 4, "repeat": 1, "assignments": {"p1": "B", "p2": "B"}},
        {"run_id": "run2_K4", "initial_k": 4, "repeat": 2, "assignments": {"p3": "C"}},
        {"run_id": "run3_K4", "initial_k": 4, "repeat": 3, "assignments": {"p4": "D"}},
    ]
    families = experiment.family_layer(
        runs, jaccard_threshold=0.5, overlap_threshold=0.8
    )["families"]
    family = families[0]
    assert family["family_presence_fraction_by_k"]["4"] == pytest.approx(1 / 3)
    assert family["patient_membership_given_family_present_by_k"]["p1"]["4"] == 1.0
    assert family["patient_unconditional_membership_by_k"]["p1"]["4"] == pytest.approx(1 / 3)


def test_primary_families_exclude_k_with_only_one_valid_repeat(tmp_path):
    for initial_k, repeat in ((4, 1), (4, 2), (5, 1)):
        run_root = tmp_path / f"run{repeat}" / f"K{initial_k}"
        run_root.mkdir(parents=True)
        (run_root / "run_metadata.json").write_text("{}")
        (run_root / "final_review_summary.json").write_text(json.dumps({
            "status": "review_complete",
            "raw_control_status": "complete",
            "accepted_subtype_sets": [{"set_id": "A", "member_ids": ["p1", "p2"]}],
        }))
    summary = experiment.analyze(tmp_path, ["p1", "p2"], [4, 5], [1, 2, 3], 2)
    assert summary["primary_family_k_values"] == [4]
    primary_catalog = json.loads((tmp_path / "accepted_set_catalog.json").read_text())
    exploratory_catalog = json.loads((tmp_path / "exploratory_accepted_set_catalog.json").read_text())
    assert {item["initial_k"] for item in primary_catalog["sets"]} == {4}
    assert {item["initial_k"] for item in exploratory_catalog["sets"]} == {4, 5}


def test_primary_core_recurrence_excludes_exploratory_k(tmp_path):
    assignments = {
        (2, 1): [{"set_id": "A", "member_ids": ["p1", "p2"]}],
        (2, 2): [{"set_id": "A", "member_ids": ["p1", "p2"]}],
        (3, 1): [{"set_id": "A", "member_ids": ["p1", "p2"]}],
        (3, 2): [{"set_id": "A", "member_ids": ["p1", "p2"]}],
        (4, 1): [
            {"set_id": "A", "member_ids": ["p1"]},
            {"set_id": "B", "member_ids": ["p2"]},
        ],
    }
    for (initial_k, repeat), accepted_sets in assignments.items():
        run_root = tmp_path / f"run{repeat}" / f"K{initial_k}"
        run_root.mkdir(parents=True)
        (run_root / "run_metadata.json").write_text("{}")
        (run_root / "final_review_summary.json").write_text(json.dumps({
            "status": "review_complete",
            "raw_control_status": "complete",
            "accepted_subtype_sets": accepted_sets,
        }))
    summary = experiment.analyze(tmp_path, ["p1", "p2"], [2, 3, 4], [1, 2, 3], 2)
    assert summary["primary_cores"][0]["same_set_run_fraction"] == 1.0
    assert summary["primary_cores"][0]["all_members_accepted_run_count"] == 4


def test_compare_runs_does_not_treat_two_empty_results_as_perfect_agreement():
    result = experiment.compare_runs({}, {})
    assert result["both_empty"] is True
    assert result["accepted_coverage_jaccard"] is None
    assert result["matched_mean_jaccard"] is None
    assert result["matched_mean_dice"] is None


def test_stability_matrices_separate_acceptance_and_conditional_membership():
    patients = ["p1", "p2", "p3"]
    runs = [
        {"p1": "A", "p2": "A"},
        {"p1": "B", "p2": "B", "p3": "C"},
        {"p1": "A", "p3": "A"},
    ]
    joint, coacceptance, conditional, acceptance = experiment.stability_matrices(runs, patients)
    assert joint[0, 1] == pytest.approx(2 / 3)
    assert joint[0, 2] == pytest.approx(1 / 3)
    assert coacceptance[0, 1] == pytest.approx(2 / 3)
    assert conditional[0, 1] == 1.0
    assert coacceptance[1, 2] == pytest.approx(1 / 3)
    assert conditional[1, 2] == 0
    assert acceptance.tolist() == pytest.approx([1.0, 2 / 3, 2 / 3])


def test_aggregate_k_levels_gives_each_available_k_equal_weight():
    runs = [
        {"run_id": "run1_K2", "initial_k": 2, "assignments": {"p1": "A"}},
        {"run_id": "run2_K2", "initial_k": 2, "assignments": {"p1": "A"}},
        {"run_id": "run1_K3", "initial_k": 3, "assignments": {}},
    ]
    levels, matrices = experiment.aggregate_k_levels(runs, ["p1"], [2, 3], 2)
    assert [level["status"] for level in levels] == ["complete", "low_confidence"]
    assert len(matrices["primary"]) == 1
    assert len(matrices["exploratory"]) == 2
    assert matrices["primary"][0][3][0] == 1.0


def test_cross_k_conditional_is_joint_over_coacceptance():
    matrices = [
        (np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.ones((2, 2)), np.ones(2)),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.1], [0.1, 1.0]]), np.zeros((2, 2)), np.ones(2)),
    ]
    joint, coacceptance, conditional, _ = experiment.combine_k_matrices(matrices)
    assert joint[0, 1] == pytest.approx(0.5)
    assert coacceptance[0, 1] == pytest.approx(0.55)
    assert conditional[0, 1] == pytest.approx(0.5 / 0.55)


def test_extract_cores_uses_joint_recurrence_and_complete_linkage():
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
    cores = experiment.extract_cores(matrix, np.ones(5), patients, threshold=0.8, min_size=2)
    assert [row["member_ids"] for row in cores] == [
        ["a1", "a2", "a3"],
        ["b1", "b2"],
    ]


def test_extract_cores_does_not_replace_joint_with_conditional_metrics():
    patients = ["a1", "a2"]
    joint = np.array([[1.0, 0.64], [0.64, 1.0]])
    acceptance = np.ones(2)
    assert experiment.extract_cores(joint, acceptance, patients, threshold=2 / 3, min_size=2) == []


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
    assert result["all_members_accepted_run_count"] == 3
    assert result["same_set_run_count"] == 2
    assert result["same_set_run_fraction"] == pytest.approx(2 / 3)
    assert result["conditional_same_set_fraction"] == pytest.approx(2 / 3)
    assert result["k_coverage_any"] == 1
    assert result["k_coverage_majority"] == 1


def test_analyze_excludes_incomplete_and_unavailable_runs(tmp_path):
    complete = tmp_path / "run1" / "K2"
    incomplete = tmp_path / "run1" / "K3"
    contract_failure = tmp_path / "run1" / "K4"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    contract_failure.mkdir(parents=True)
    (complete / "run_metadata.json").write_text("{}")
    (incomplete / "run_metadata.json").write_text("{}")
    (contract_failure / "run_metadata.json").write_text("{}")
    (complete / "final_review_summary.json").write_text(json.dumps({
        "status": "review_complete", "raw_control_status": "complete",
        "accepted_subtype_sets": [{"set_id": "A", "member_ids": ["p1", "p2"]}],
    }))
    (incomplete / "final_review_summary.json").write_text(json.dumps({
        "status": "review_incomplete_due_to_round_budget",
        "raw_control_status": "review_incomplete_due_to_round_budget",
        "accepted_subtype_sets": [],
    }))
    (contract_failure / "final_review_summary.json").write_text(json.dumps({
        "status": "review_unavailable",
        "raw_control_status": "review_unavailable",
        "partition_sets": [],
    }))

    summary = experiment.analyze(tmp_path, ["p1", "p2"], [2, 3, 4], [1], 2)

    assert summary["analysis_status"] == "partial"
    assert summary["valid_run_count"] == 1
    assert summary["invalid_run_count"] == 2
    with (tmp_path / "patient_acceptance_frequency.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["acceptance_frequency"]) for row in rows] == [1.0, 1.0]
