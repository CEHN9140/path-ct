import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts_2026_8_17/experiment_ct_phase_aware_reselection.py"
SPEC = importlib.util.spec_from_file_location("ct_phase_aware_reselection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(case_id, phase, rank):
    return {
        "case_id": case_id,
        "inferred_phase": phase,
        "eligible_candidate": True,
        "selection_rank": rank,
    }


def test_phase_priority_beats_radqy_rank_across_phase_groups():
    selected = MODULE.select_phase_aware(
        [
            row("case-1", "DEL", "1"),
            row("case-1", "MAIN_CE_HIGH", "3"),
            row("case-1", "MAIN_CE_HIGH", "2"),
        ]
    )

    assert selected["case-1"]["inferred_phase"] == "MAIN_CE_HIGH"
    assert selected["case-1"]["selection_rank"] == "2"


def test_phase_aware_reselection_falls_back_to_best_unknown_series():
    selected = MODULE.select_phase_aware(
        [row("case-2", "UNKNOWN", "4"), row("case-2", "UNKNOWN", "1")]
    )

    assert selected["case-2"]["selection_rank"] == "1"
