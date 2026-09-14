from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_rna_reconstruction_matches_canonical_affinity():
    root = Path(__file__).resolve().parents[1]
    candidate = root / "output_kirc/candidate_subtype"
    table_path = root / "output_kirc/rna/case_pathway_features.csv"
    if not table_path.is_file() or not (candidate / "rna_affinity.npy").is_file():
        pytest.skip("canonical RNA artifacts are unavailable")
    ids = json.loads((candidate / "affinity_patient_order.json").read_text())
    config = module.load_candidate_proposer_config(root / "configs")["snf"]
    table = pd.read_csv(table_path).set_index("case_id").loc[ids]
    reconstructed, count = module.rna_affinity_from_full_table(table, 2000, config)
    canonical = np.load(candidate / "rna_affinity.npy")
    assert count == 2000
    assert np.allclose(reconstructed, canonical, atol=1e-10)


def test_cnv_reconstruction_matches_canonical_affinity():
    root = Path(__file__).resolve().parents[1]
    candidate = root / "output_kirc/candidate_subtype"
    table_path = root / "output_kirc/cnv/case_features.csv"
    if not table_path.is_file() or not (root / "output_kirc/wxs/cnv_affinity.npy").is_file():
        pytest.skip("canonical CNV artifacts are unavailable")
    ids = json.loads((candidate / "affinity_patient_order.json").read_text())
    config = module.load_candidate_proposer_config(root / "configs")["snf"]
    table = pd.read_csv(table_path).set_index("case_id").loc[ids]
    reconstructed = module.affinity(module.robust_scale_cnv(table.to_numpy(float)), "euclidean", config)
    canonical = np.load(root / "output_kirc/wxs/cnv_affinity.npy")
    assert np.allclose(reconstructed, canonical, atol=1e-10)
