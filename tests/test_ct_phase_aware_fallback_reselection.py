import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts_2026_8_17/experiment_ct_phase_aware_fallback_reselection.py"
SPEC = importlib.util.spec_from_file_location("ct_phase_aware_fallback_reselection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(case_id, phase, rank, radqy_pass="True"):
    return {
        "case_id": case_id,
        "inferred_phase": phase,
        "eligible_candidate": True,
        "selection_rank": rank,
        "radqy_iqm_pass": radqy_pass,
        "series_uid": f"{case_id}-{phase}-{rank}",
    }


def test_reliable_enhanced_candidate_precedes_nc_or_del():
    selected = MODULE.select_enhanced_or_radqy_best(
        [
            row("case-1", "NC", "1"),
            row("case-1", "DEL", "2"),
            row("case-1", "MAIN_CE_HIGH", "3"),
        ]
    )

    assert selected["case-1"]["inferred_phase"] == "MAIN_CE_HIGH"


def test_without_reliable_enhancement_falls_back_to_radqy_best():
    selected = MODULE.select_enhanced_or_radqy_best(
        [row("case-2", "NC", "2"), row("case-2", "DEL", "1")]
    )

    assert selected["case-2"]["inferred_phase"] == "DEL"


def test_radqy_failures_are_not_used_in_the_final_candidate_pool():
    selected = MODULE.select_enhanced_or_radqy_best(
        [row("case-3", "NEPH", "1", "False")]
    )

    assert "case-3" not in selected
