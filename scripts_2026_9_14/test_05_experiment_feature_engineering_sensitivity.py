from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


path = Path(__file__).with_name("05_experiment_feature_engineering_sensitivity.py")
spec = spec_from_file_location("feature_sensitivity", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_core_jaccard_uses_patient_membership():
    reference = {"A": ["TCGA-1", "TCGA-2"], "B": ["TCGA-3", "TCGA-4"]}
    candidate = {"x": ["TCGA-1", "TCGA-2"], "y": ["TCGA-3", "TCGA-9"]}
    mean, minimum = module.core_jaccard(reference, candidate)
    assert mean == pytest.approx(2 / 3)
    assert minimum == pytest.approx(1 / 3)
