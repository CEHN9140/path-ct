from scripts_2026_8_17.experiment_ct_final_pass_fail_v2 import (
    candidate_fail_reasons,
    candidate_pass,
    ensure_project_root_on_path,
    final_status,
    phase_priority,
    pretreatment_pass,
    select_best_series,
    technical_key,
)


def test_project_root_is_available_for_project_module_imports():
    import sys

    ensure_project_root_on_path()
    assert "/data/qijun/path-ct" in sys.path


def make_row(case_id, series_uid, phase, thickness=2, spacing=1, z_gap=1, z_spacing=2, n_slices=100):
    return {
        "case_id": case_id,
        "series_uid": series_uid,
        "inferred_phase": phase,
        "eligible_candidate": "true",
        "prefilter_pass": "true",
        "nifti_qc_pass": "true",
        "totalseg_pass": "true",
        "slice_thickness_median": str(thickness),
        "pixel_spacing_row": str(spacing),
        "pixel_spacing_col": str(spacing),
        "z_gap_max": str(z_gap),
        "z_spacing_median": str(z_spacing),
        "n_slices": str(n_slices),
    }


def test_phase_priority_prefers_enhanced_phase_then_conservative_fallbacks():
    phases = ["NEPH", "MAIN_CE_HIGH", "MAIN_CE_MEDIUM", "CE_UNSPECIFIED", "ART", "DEL"]
    assert [phase_priority(phase) for phase in phases] == list(range(len(phases)))
    assert phase_priority("NC") == phase_priority("UNKNOWN") == 6


def test_selection_prefers_phase_before_technical_quality():
    rows = [
        make_row("A", "nc", "NC", thickness=1),
        make_row("A", "neph", "NEPH", thickness=4),
        make_row("B", "worse", "UNKNOWN", thickness=1, spacing=0.7),
        make_row("B", "better", "UNKNOWN", thickness=2, spacing=0.7),
        make_row("C", "nc", "NC", thickness=3),
        make_row("C", "unknown", "UNKNOWN", thickness=1),
    ]
    selected = select_best_series(rows)
    assert selected["A"]["series_uid"] == "neph"
    assert selected["B"]["series_uid"] == "worse"
    assert selected["C"]["series_uid"] == "unknown"
    assert len(selected) == 3


def test_technical_key_uses_thickness_spacing_continuity_coverage_and_uid():
    thin = make_row("A", "thin", "NEPH", thickness=1, spacing=1, z_gap=1, z_spacing=2, n_slices=100)
    thick = make_row("A", "thick", "NEPH", thickness=2, spacing=0.5, z_gap=0.5, z_spacing=2, n_slices=200)
    assert technical_key(thin) < technical_key(thick)


def test_unknown_phase_is_allowed_and_post_treatment_marker_fails():
    assert pretreatment_pass({"study_description": "RIGHT NEPHRECTOMY PLANNING"}) is True
    assert pretreatment_pass({"study_description": "STATUS POST RIGHT NEPHRECTOMY"}) is False


def test_series_description_controls_anatomy_gate_before_study_fallback():
    pure_chest = make_row("A", "chest", "UNKNOWN")
    pure_chest.update({
        "study_description": "CT THORAX WCONTRAST",
        "series_description": "RECON 2 CHEST",
        "protocol_name": "CHEST",
    })
    combined = make_row("B", "cap", "UNKNOWN")
    combined.update({
        "study_description": "CTCAP RENAL WC",
        "series_description": "RECON 2 CHEST",
        "protocol_name": "C/A/P RENAL",
    })
    assert candidate_pass(pure_chest) is False
    assert "non_abdominal_chest_series" in candidate_fail_reasons(pure_chest)
    assert candidate_pass(combined) is False

    combined["series_description"] = "CHEST ABD PELVIS"
    assert candidate_pass(combined) is True

    fallback = make_row("C", "fallback", "UNKNOWN")
    fallback.update({
        "study_description": "CT THORAX WCONTRAST",
        "series_description": "",
        "protocol_name": "CHEST",
    })
    assert candidate_pass(fallback) is False


def test_final_status_has_only_pass_or_fail():
    checks = {
        "segmentation_success": True,
        "mask_exists": True,
        "series_uid_match": True,
        "geometry_match": True,
        "mask_nonempty": True,
    }
    assert final_status(checks) == "PASS"
    checks["mask_nonempty"] = False
    assert final_status(checks) == "FAIL"
