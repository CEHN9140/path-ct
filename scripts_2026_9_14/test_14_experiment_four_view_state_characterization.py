import importlib.util
from pathlib import Path


path = Path(__file__).with_name("14_experiment_four_view_state_characterization.py")
spec = importlib.util.spec_from_file_location("state_characterization", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_discovery_space_affinity_tests_are_adjusted_and_not_called_validation():
    rows = [
        {"permanova_p_value": 0.01, "permdisp_permdisp_p_value": 0.02},
        {"permanova_p_value": 0.04, "permdisp_permdisp_p_value": 0.50},
    ]
    survival = {"analysis_role": "primary"}
    module.finalize_diagnostic_roles(rows, survival)
    assert [row["permanova_q_value"] for row in rows] == [0.02, 0.04]
    assert [row["permdisp_q_value"] for row in rows] == [0.04, 0.5]
    assert all(row["analysis_role"] == "discovery_space_diagnostic" for row in rows)
    assert survival["analysis_role"] == "secondary_exploratory"
