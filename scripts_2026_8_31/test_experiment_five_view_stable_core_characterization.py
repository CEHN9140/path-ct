import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("16_experiment_five_view_stable_core_characterization.py")
SPEC = importlib.util.spec_from_file_location("five_view_characterization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_characterization_defaults_use_five_view_v12_artifacts():
    defaults = MODULE.ROOT / "output_kirc_v12"
    assert defaults in (MODULE.ROOT / "output_kirc_v12/15_five_view_multi_k_stability").parents
    assert defaults in (MODULE.ROOT / "output_kirc_v12/16_five_view_stable_core_characterization").parents


def test_wsi_embedding_table_uses_case_level_embedding_features():
    states = {
        "A": {"wsi_evidence": {"features": [1.0, 2.0]}},
        "B": {"wsi_evidence": {"features": [3.0, 4.0]}},
    }
    features, table = MODULE.wsi_embedding_table(states, ["A", "B"])
    assert features == ["embedding_0000", "embedding_0001"]
    assert table["A"]["embedding_0000"] == 1.0


def test_validate_stable_cores_rejects_wrong_core_sizes(tmp_path):
    membership = tmp_path / "stable_core_membership.csv"
    membership.write_text("core_id,patient_id\nCORE01,A\n", encoding="utf-8")
    try:
        MODULE.validate_stable_cores(tmp_path, {"A"})
    except ValueError as exc:
        assert "6组固定规模" in str(exc)
    else:
        raise AssertionError("invalid stable-core membership was accepted")
