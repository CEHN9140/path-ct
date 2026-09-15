import importlib.util
from pathlib import Path

path = Path(__file__).with_name("09_experiment_full_leave_one_modality_out.py")
spec = importlib.util.spec_from_file_location("leave_one", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_variants_disable_exactly_one_view():
    assert set(module.VARIANTS) == set(module.ALL_MODALITIES)
    for disabled, active in module.VARIANTS.items():
        assert disabled not in active
        assert len(active) == 4
        assert set(active) | {disabled} == set(module.ALL_MODALITIES)


def test_canonical_row_is_complete_baseline():
    row = module.canonical_row({"CORE01": ["A"], "CORE02": ["B", "C"]})
    assert row["stable_core_count"] == 2
    assert row["stable_core_patient_count"] == 3
    assert row["canonical_matched_jaccard"] == 1.0
    assert row["no_variant_stable_core_fraction"] == 0.0
