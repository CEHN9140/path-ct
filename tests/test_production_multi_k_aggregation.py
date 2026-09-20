import json

import numpy as np
import pandas as pd


def test_multi_k_aggregation_requires_all_21_complete_runs(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    config = {
        "multi_k": {
            "coassignment_threshold": 2 / 3,
            "acceptance_threshold": 2 / 3,
            "min_core_size": 4,
            "max_states": 6,
        }
    }
    runs = tmp_path / "subtype_review" / "runs"
    patient_ids = [f"P{i:02d}" for i in range(16)]
    (tmp_path / "candidate_subtype").mkdir()
    (tmp_path / "candidate_subtype" / "affinity_patient_order.json").write_text(
        json.dumps(patient_ids), encoding="utf-8"
    )
    for k in range(2, 9):
        for repeat in range(1, 4):
            run_root = runs / f"K{k}" / f"repeat{repeat}"
            run_root.mkdir(parents=True)
            (run_root / "run_metadata.json").write_text(json.dumps({
                "status": "complete",
                "input_signature": "same-input",
            }), encoding="utf-8")
            clusters = [list(range(start, start + 4)) for start in (0, 4, 8, 12)]
            if repeat == 1 and k <= 5:
                clusters = [list(range(0, 8)), list(range(8, 16))]
            if (k, repeat) in {
                (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1),
            }:
                clusters = [members for members in clusters if 15 not in members]
            accepted = [
                {"set_id": f"S{i}", "member_ids": [patient_ids[index] for index in members]}
                for i, members in enumerate(clusters)
            ]
            (run_root / "final_subtype_sets.json").write_text(json.dumps(accepted), encoding="utf-8")
            (run_root / "final_review_summary.json").write_text(json.dumps({
                "status": "review_complete",
                "raw_control_status": "complete",
            }), encoding="utf-8")

    result = run_multi_k_aggregation(str(tmp_path), config, "same-input")

    assert result["run_count"] == 21
    assert result["core_count"] == 4
    assert result["selected_state_k"] == 2
    assert len(result["state_membership"]) == 16
    assert (tmp_path / "subtype_review/multi_k/accepted_coassignment_matrix.csv").is_file()
    frequency = pd.read_csv(
        tmp_path / "subtype_review/multi_k/patient_acceptance_frequency.csv"
    ).set_index("patient_id")
    assert frequency.loc["P15", "acceptance_frequency"] == 14 / 21


def test_multi_k_aggregation_does_not_start_for_incomplete_grid(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    try:
        run_multi_k_aggregation(str(tmp_path), {}, "missing")
    except (FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError("incomplete multi-K grid must fail before aggregation")
    assert not (tmp_path / "subtype_review/multi_k").exists()
