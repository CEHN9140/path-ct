import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts_2026_8_17/experiment_ct_phase_audit.py"
SPEC = importlib.util.spec_from_file_location("ct_phase_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_row(series_number, description, phase, study_uid="study-1", acquisition_time=""):
    return {
        "study_uid": study_uid,
        "series_number": series_number,
        "acquisition_number": "",
        "acquisition_time": acquisition_time,
        "series_description": description,
        "protocol_name": "",
        "phase": phase,
        "eligible_candidate": True,
    }


def test_infer_main_ce_between_nc_and_del_skips_art_and_recon():
    rows = [
        make_row("2", "PRE-CONTRAST", "NC"),
        make_row("3", "POST CONTRAST", "CE_UNSPECIFIED"),
        make_row("4", "ARTERIAL", "ART"),
        make_row("5", "RECON POST", "UNKNOWN"),
        make_row("6", "3 MIN DELAY", "DEL"),
    ]

    updated = MODULE.infer_main_ce(rows)

    assert [row["inferred_phase"] for row in updated] == [
        "NC",
        "MAIN_CE_HIGH",
        "ART",
        "UNKNOWN",
        "DEL",
    ]
    assert updated[1]["phase_source"] == "study_order"


def test_high_confidence_uses_acquisition_time_and_nearest_nc_del_boundaries():
    rows = [
        make_row("1", "PRE-CONTRAST", "NC", acquisition_time="10:00:00"),
        make_row("2", "PRE-CONTRAST", "NC", acquisition_time="12:00:00"),
        make_row("3", "POST CONTRAST", "CE_UNSPECIFIED", acquisition_time="13:00:00"),
        make_row("4", "3 MIN DELAY", "DEL", acquisition_time="14:00:00"),
        make_row("5", "DELAY RECON", "DEL", acquisition_time="15:00:00"),
    ]

    updated = MODULE.infer_main_ce(rows)

    assert [row["inferred_phase"] for row in updated] == [
        "NC",
        "NC",
        "MAIN_CE_HIGH",
        "DEL",
        "DEL",
    ]


def test_medium_confidence_uses_post_contrast_after_nc_without_del():
    rows = [
        make_row("1", "PRE-CONTRAST", "NC", acquisition_time="10:00:00"),
        make_row("2", "POST CONTRAST", "CE_UNSPECIFIED", acquisition_time="11:00:00"),
        make_row("3", "ABDOMEN", "UNKNOWN", acquisition_time="12:00:00"),
    ]

    updated = MODULE.infer_main_ce(rows)

    assert [row["inferred_phase"] for row in updated] == [
        "NC",
        "MAIN_CE_MEDIUM",
        "UNKNOWN",
    ]


def test_infer_main_ce_requires_nc_and_del_in_same_study():
    rows = [
        make_row("2", "POST CONTRAST", "CE_UNSPECIFIED", study_uid="study-1"),
        make_row("3", "3 MIN DELAY", "DEL", study_uid="study-1"),
        make_row("4", "POST CONTRAST", "CE_UNSPECIFIED", study_uid="study-2"),
    ]

    updated = MODULE.infer_main_ce(rows)

    assert [row["inferred_phase"] for row in updated] == [
        "CE_UNSPECIFIED",
        "DEL",
        "CE_UNSPECIFIED",
    ]


def test_classify_phase_supports_neph_variants_without_protocol_overreach():
    assert MODULE.classify_phase(
        {"series_description": "COR NEPH", "study_description": ""}, {}
    )["phase"] == "NEPH"
    assert MODULE.classify_phase(
        {"series_description": "NEPHROGRAPHIC", "study_description": ""}, {}
    )["phase"] == "NEPH"
    assert MODULE.classify_phase(
        {"series_description": "PARANCHYMAL", "study_description": ""}, {}
    )["phase"] == "NEPH"
    assert MODULE.classify_phase(
        {"series_description": "UROGRAM", "study_description": ""}, {}
    )["phase"] == "UNKNOWN"
    assert MODULE.classify_phase(
        {"series_description": "SMART PREP", "study_description": ""}, {}
    )["phase"] == "UNKNOWN"
    assert MODULE.classify_phase(
        {"series_description": "12 MIN", "study_description": ""}, {}
    )["phase"] == "DEL"
    assert MODULE.classify_phase(
        {"series_description": "3MIN DELAYS", "study_description": ""}, {}
    )["phase"] == "DEL"
    assert MODULE.classify_phase(
        {"series_description": "90 SEC", "study_description": ""}, {}
    )["phase"] == "UNKNOWN"


def test_infer_main_ce_does_not_group_empty_study_uids_across_cases():
    rows = [
        make_row("1", "PRE-CONTRAST", "NC", study_uid="", acquisition_time="10:00:00"),
        make_row("2", "POST CONTRAST", "CE_UNSPECIFIED", study_uid="", acquisition_time="11:00:00"),
        make_row("3", "3 MIN DELAY", "DEL", study_uid="", acquisition_time="12:00:00"),
    ]
    for row, case_id in zip(rows, ("case-1", "case-2", "case-3")):
        row["case_id"] = case_id

    updated = MODULE.infer_main_ce(rows)

    assert [row["inferred_phase"] for row in updated] == [
        "NC",
        "CE_UNSPECIFIED",
        "DEL",
    ]
