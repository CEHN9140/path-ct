import importlib.util
from pathlib import Path

import pandas as pd

path = Path(__file__).with_name("10_experiment_modality_dependency_matrix.py")
spec = importlib.util.spec_from_file_location("dependency", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_build_dependency_table_uses_fragmentation_metrics(tmp_path):
    frame = pd.DataFrame([
        {
            "canonical_core": "CORE01",
            "retention": 0.5,
            "any_variant_stable_fraction": 0.75,
            "no_variant_stable_fraction": 0.25,
            "jaccard": 0.4,
        }
    ])
    path = tmp_path / "fragmentation.csv"
    frame.to_csv(path, index=False)
    result = module.build_dependency_table(
        {modality: path for modality in module.MODALITIES}, cores=("CORE01",)
    )
    row = result.iloc[0]
    assert row["core"] == "CORE01"
    assert row["modality"] == "ct"
    assert row["matched_retention"] == 0.5
    assert row["any_stable_retention"] == 0.75
    assert row["lost_fraction"] == 0.25
    assert row["jaccard"] == 0.4
    assert row["membership_dependency"] == 0.25
    assert row["identity_disruption"] == 0.5


def test_build_dependency_table_rejects_missing_core(tmp_path):
    path = tmp_path / "fragmentation.csv"
    pd.DataFrame([{"canonical_core": "CORE01", "retention": 1.0}]).to_csv(path, index=False)
    try:
        module.build_dependency_table(
            {modality: path for modality in module.MODALITIES},
            cores=("CORE01", "CORE02"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("missing core rows must fail")
