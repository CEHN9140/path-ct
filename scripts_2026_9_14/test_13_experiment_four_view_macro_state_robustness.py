import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).with_name("13_experiment_four_view_macro_state_robustness.py")
spec = importlib.util.spec_from_file_location("macro_robustness", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cluster_summary_reports_actual_cluster_count_and_reference_ari():
    labels = [f"CORE{i:02d}" for i in range(1, 11)]
    matrix = np.eye(10)
    for i in range(10):
        for j in range(10):
            if i // 2 == j // 2:
                matrix[i, j] = 0.9
    result = module.cluster_summary(matrix, labels, 5, "average", [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    assert result["actual_cluster_count"] == 5
    assert result["reference_ari"] == 1.0


def test_cluster_summary_marks_one_cluster_silhouette_unavailable():
    result = module.cluster_summary(np.ones((4, 4)), ["A", "B", "C", "D"], 2, "complete")
    assert result["actual_cluster_count"] == 1
    assert result["silhouette"] is None


def test_coassignment_uses_accepted_subtype_sets_and_reports_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REPEATS", (1, 2))
    monkeypatch.setattr(module, "INITIAL_KS", (2,))
    for repeat, members in ((1, ["A", "B"]), (2, ["A"])):
        run_root = tmp_path / f"run{repeat}" / "K2"
        run_root.mkdir(parents=True)
        (run_root / "final_review_summary.json").write_text(json.dumps({"status": "review_complete"}))
        (run_root / "final_partition_sets.json").write_text("[]")
        (run_root / "final_subtype_sets.json").write_text(json.dumps([
            {"set_id": "C1", "member_ids": members},
        ]))
    joint, conditional, audit = module.coassignment_from_runs(
        tmp_path, ["A", "B", "C"], return_audit=True
    )
    assert joint[0, 1] == 0.5
    assert joint[0, 2] == 0.0
    assert conditional[0, 1] == 1.0
    assert conditional[2, 2] == 0.0
    assert [row["assigned_patient_n"] for row in audit] == [2, 1]


@pytest.mark.parametrize("status,members", [
    ("review_unavailable", ["A"]),
    ("review_complete", ["A", "UNKNOWN"]),
])
def test_coassignment_rejects_incomplete_or_out_of_cohort_run(tmp_path, monkeypatch, status, members):
    monkeypatch.setattr(module, "REPEATS", (1,))
    monkeypatch.setattr(module, "INITIAL_KS", (2,))
    run_root = tmp_path / "run1" / "K2"
    run_root.mkdir(parents=True)
    (run_root / "final_review_summary.json").write_text(json.dumps({"status": status}))
    (run_root / "final_subtype_sets.json").write_text(json.dumps([
        {"set_id": "C1", "member_ids": members},
    ]))
    with pytest.raises(ValueError):
        module.coassignment_from_runs(tmp_path, ["A", "B"])
