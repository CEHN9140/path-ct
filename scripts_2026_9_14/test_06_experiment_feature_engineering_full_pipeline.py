import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("06_experiment_feature_engineering_full_pipeline.py")
SPEC = importlib.util.spec_from_file_location("full_pipeline_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_variant_contract_is_explicit():
    assert MODULE.VARIANTS == (
        "rna_top_3000",
        "wxs_prevalence_only",
        "wxs_zero_distance_05",
    )
