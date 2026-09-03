import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("18_experiment_conditional_multimodal_support.py")
SPEC = importlib.util.spec_from_file_location("conditional_support", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_target_groups_include_five_view_cores_and_robust_intersection():
    groups = MODULE.target_groups(
        {"CORE01": ["a", "b"], "CORE02": ["c"]},
        [{"core_4v": "CORE01", "core_5v": "CORE01", "intersection_n": "2", "shared_patient_ids": '["a", "b"]', "representation_robust_candidate": "1"}],
    )
    assert groups["5V_CORE01"] == ["a", "b"]
    assert groups["ROBUST_4V_CORE01_5V_CORE01"] == ["a", "b"]


def test_support_summary_counts_only_fdr_supported_features():
    rows = [
        {"target_id": "C1", "modality": "rna", "feature": "P1", "q_value": "0.01", "effect_size": "1.2"},
        {"target_id": "C1", "modality": "rna", "feature": "P2", "q_value": "0.2", "effect_size": "2.0"},
    ]
    summary = MODULE.support_summary(rows)
    assert summary[0]["significant_feature_n"] == 1
    assert summary[0]["top_feature"] == "P1"
