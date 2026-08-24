import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts_2026_8_17/experiment_ct_phase_aware_radqy_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("ct_phase_aware_radqy_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(case_id, phase, rank, radqy_pass):
    return {
        "case_id": case_id,
        "inferred_phase": phase,
        "eligible_candidate": True,
        "selection_rank": rank,
        "radqy_iqm_pass": radqy_pass,
        "series_uid": f"{case_id}-{phase}-{rank}",
    }


def test_soft_keeps_phase_preference_while_strict_filters_radqy_failures():
    rows = [
        row("case-1", "MAIN_CE_HIGH", "4", "False"),
        row("case-1", "DEL", "1", "True"),
    ]

    soft, strict = MODULE.select_sensitivity_arms(rows)

    assert soft["case-1"]["inferred_phase"] == "MAIN_CE_HIGH"
    assert strict["case-1"]["inferred_phase"] == "DEL"


def test_strict_arm_drops_case_with_only_radqy_failures():
    soft, strict = MODULE.select_sensitivity_arms(
        [row("case-2", "NEPH", "1", "False")]
    )

    assert "case-2" in soft
    assert "case-2" not in strict
