import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).with_name("13_experiment_four_view_macro_state_robustness.py")
spec = importlib.util.spec_from_file_location("macro_robustness", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cluster_summary_reports_candidate_ari_for_four_states():
    labels = list(module.CORE_TO_STATE)
    matrix = np.eye(len(labels))
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i != j and module.CORE_TO_STATE[left] == module.CORE_TO_STATE[right]:
                matrix[i, j] = 0.9
    result = module.cluster_summary(matrix, labels, 4, "average")
    assert result["actual_cluster_count"] == 4
    assert result["candidate_state_ari"] == 1.0


def test_cluster_summary_marks_one_cluster_silhouette_unavailable():
    result = module.cluster_summary(np.ones((4, 4)), list(module.CORE_TO_STATE)[:4], 2, "complete")
    assert result["actual_cluster_count"] == 1
    assert result["silhouette"] is None


def test_coassignment_uses_accepted_subtype_sets_only(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REPEATS", (1,))
    monkeypatch.setattr(module, "INITIAL_KS", (2,))
    run_root = tmp_path / "run1" / "K2"
    run_root.mkdir(parents=True)
    import json
    (run_root / "final_partition_sets.json").write_text(json.dumps([
        {"set_id": "C1", "member_ids": ["A", "B"]},
        {"set_id": "C2", "member_ids": ["C"]},
    ]))
    (run_root / "final_subtype_sets.json").write_text(json.dumps([
        {"set_id": "C1", "member_ids": ["A", "B"]},
    ]))
    result = module.coassignment_from_runs(tmp_path, ["A", "B", "C"])
    assert result[0, 1] == 1.0
    assert result[0, 2] == 0.0
    assert result[2, 2] == 1.0
