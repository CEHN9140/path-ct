import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts_2026_8_17/experiment_ct_phase_confound_audit.py"
SPEC = importlib.util.spec_from_file_location("ct_phase_confound_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contingency_association_reports_cluster_phase_dependence():
    rows = [
        {"cluster": "0", "phase_group": "NC"},
        {"cluster": "0", "phase_group": "NC"},
        {"cluster": "1", "phase_group": "CE"},
        {"cluster": "1", "phase_group": "CE"},
    ]

    result = MODULE.contingency_association(rows, "phase_group")

    assert result["n"] == 4
    assert result["levels"] == ["CE", "NC"]
    assert result["cramers_v"] == 1.0
    assert result["p_value"] < 0.05


def test_contingency_association_ignores_missing_factor_values():
    rows = [
        {"cluster": "0", "phase_group": "NC"},
        {"cluster": "0", "phase_group": ""},
        {"cluster": "1", "phase_group": "CE"},
    ]

    result = MODULE.contingency_association(rows, "phase_group")

    assert result["n"] == 2
    assert result["missing_count"] == 1
