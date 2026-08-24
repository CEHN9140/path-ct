import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts_2026_8_17/experiment_ct_no_radqy_selection.py"
SPEC = importlib.util.spec_from_file_location("ct_no_radqy_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(case_id, phase, thickness, spacing, z_gap, rank):
    return {
        "case_id": case_id,
        "inferred_phase": phase,
        "eligible_candidate": True,
        "slice_thickness_median": thickness,
        "pixel_spacing_row": spacing,
        "pixel_spacing_col": spacing,
        "z_gap_max": z_gap,
        "n_slices": "100",
        "selection_rank": rank,
        "radqy_iqm_pass": "False",
        "series_uid": f"{case_id}-{phase}-{rank}",
    }


def test_no_radqy_selection_prefers_phase_then_technical_quality():
    selected = MODULE.select_without_radqy(
        [
            row("case-1", "NC", "1.0", "0.7", "0", "1"),
            row("case-1", "MAIN_CE_HIGH", "5.0", "1.0", "2", "9"),
            row("case-1", "MAIN_CE_HIGH", "2.0", "0.8", "0", "8"),
        ]
    )

    assert selected["case-1"]["inferred_phase"] == "MAIN_CE_HIGH"
    assert selected["case-1"]["slice_thickness_median"] == "2.0"


def test_no_radqy_fallback_uses_technical_quality_not_radqy_rank():
    selected = MODULE.select_without_radqy(
        [
            row("case-2", "UNKNOWN", "5.0", "1.0", "2", "1"),
            row("case-2", "DEL", "2.0", "0.8", "0", "9"),
        ]
    )

    assert selected["case-2"]["inferred_phase"] == "DEL"
