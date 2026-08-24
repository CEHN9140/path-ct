from tools.ct_qc import (
    candidate_fail_reasons,
    build_case_summary,
    ct_case_signature,
    ct_qc_config_signature,
    phase_priority,
    pretreatment_pass,
    select_best_series,
    technical_key,
)


def make_series(case_id, series_uid, phase, thickness=2, spacing=1, z_gap=1, z_spacing=2, n_slices=100):
    return {
        "case_id": case_id,
        "series_uid": series_uid,
        "inferred_phase": phase,
        "prefilter_pass": True,
        "nifti_qc_pass": True,
        "totalseg_pass": True,
        "slice_thickness_median": thickness,
        "pixel_spacing_row": spacing,
        "pixel_spacing_col": spacing,
        "z_gap_max": z_gap,
        "z_spacing_median": z_spacing,
        "n_slices": n_slices,
    }


def test_formal_qc_phase_priority_and_unique_selection():
    assert phase_priority("NC") == phase_priority("UNKNOWN") == 6
    rows = [
        make_series("A", "nc", "NC", thickness=1),
        make_series("A", "neph", "NEPH", thickness=4),
        make_series("B", "nc", "NC", thickness=3),
        make_series("B", "unknown", "UNKNOWN", thickness=1),
    ]
    selected = select_best_series(rows)
    assert selected["A"]["series_uid"] == "neph"
    assert selected["B"]["series_uid"] == "unknown"
    assert len(selected) == 2


def test_formal_qc_technical_key_order_is_phase_internal_only():
    thin = make_series("A", "thin", "NEPH", thickness=1, spacing=1)
    thick = make_series("A", "thick", "NEPH", thickness=2, spacing=0.5)
    assert technical_key(thin) < technical_key(thick)


def test_formal_qc_rejects_pure_series_chest_but_keeps_combined_scan():
    pure_chest = {
        "series_description": "RECON 2 CHEST",
        "study_description": "CTCAP RENAL WC",
        "protocol_name": "C/A/P RENAL",
    }
    combined = {
        "series_description": "CHEST ABD PELVIS",
        "study_description": "CTCAP RENAL WC",
        "protocol_name": "C/A/P RENAL",
    }
    assert "non_abdominal_chest_series" in candidate_fail_reasons(pure_chest)
    assert candidate_fail_reasons(combined) == []


def test_case_summary_selects_only_candidate_series(tmp_path):
    chest_path = tmp_path / "chest.nii.gz"
    renal_path = tmp_path / "renal.nii.gz"
    chest_path.write_bytes(b"chest")
    renal_path.write_bytes(b"renal")
    chest = make_series("A", "chest", "NEPH")
    chest.update(
        ct_path=str(chest_path),
        series_description="RECON 2 CHEST",
        study_description="CTCAP RENAL WC",
        protocol_name="C/A/P RENAL",
    )
    renal = make_series("A", "renal", "UNKNOWN")
    renal.update(ct_path=str(renal_path), series_description="ABDOMEN")
    summary = build_case_summary("A", [chest, renal], str(tmp_path / "report.csv"))
    assert summary["selected_series"]["series_uid"] == "renal"


def test_formal_qc_rejects_explicit_post_treatment():
    assert pretreatment_pass({"study_description": "STATUS POST RIGHT NEPHRECTOMY"}) is False
    assert pretreatment_pass({"series_description": "RENAL CT"}) is True


def test_ct_qc_rebuilds_series_rows_when_config_changes(tmp_path, monkeypatch):
    import json
    import tools.ct_qc as ct_qc

    config_dir = tmp_path / "configs"
    output_root = tmp_path / "output"
    config_dir.mkdir()
    (config_dir / "ct_qc.yaml").write_text("threshold: 2\n", encoding="utf-8")
    ct_qc_dir = output_root / "ct_qc"
    ct_qc_dir.mkdir(parents=True)
    (ct_qc_dir / "cohort_cache.json").write_text(
        json.dumps({"config_signature": "old-signature", "cohort_signature": "old-cohort"}),
        encoding="utf-8",
    )
    observed = {}

    def fake_prepare(*args, **kwargs):
        observed["case_ids"] = [case["Case_ID"] for case in args[0]]
        (ct_qc_dir / "A").mkdir()
        return {"A": {"series_records": [], "prefilter_report_path": str(tmp_path / "report.json")}}

    monkeypatch.setattr(ct_qc, "prepare_ct_cases", fake_prepare)
    ct_qc.run_ct_qc(
        [{"Case_ID": "A", "CT": []}],
        output_root=str(output_root),
        config_dir=str(config_dir),
    )
    assert observed["case_ids"] == ["A"]


def test_ct_qc_signature_ignores_runtime_settings():
    config = {
        "dicom_prefilter": {"max_slice_thickness": 5},
        "totalsegmentator": {"devices": ["cuda:0"], "workers_per_gpu": 4, "task": "total"},
    }
    runtime_changed = {
        "dicom_prefilter": {"max_slice_thickness": 5},
        "totalsegmentator": {"devices": ["cuda:1", "cuda:2"], "workers_per_gpu": 8, "task": "total"},
    }
    semantic_changed = {
        "dicom_prefilter": {"max_slice_thickness": 3},
        "totalsegmentator": {"devices": ["cuda:1"], "workers_per_gpu": 8, "task": "total"},
    }
    assert ct_qc_config_signature(config) == ct_qc_config_signature(runtime_changed)
    assert ct_qc_config_signature(config) != ct_qc_config_signature(semantic_changed)


def test_ct_qc_reuses_unchanged_cases_and_processes_only_new_case(tmp_path, monkeypatch):
    import json
    import tools.ct_qc as ct_qc

    config_dir = tmp_path / "configs"
    output_root = tmp_path / "output"
    config_dir.mkdir()
    (config_dir / "ct_qc.yaml").write_text("threshold: 2\n", encoding="utf-8")
    ct_qc_dir = output_root / "ct_qc"
    case_a = {"Case_ID": "A", "CT": [{"File Path": "/a", "Series UID": "SA"}]}
    case_b = {"Case_ID": "B", "CT": [{"File Path": "/b", "Series UID": "SB"}]}
    (ct_qc_dir / "A").mkdir(parents=True)
    (ct_qc_dir / "A" / "selection_summary.json").write_text(
        json.dumps({"case_qc_passes_threshold": False, "selected_series": {}}),
        encoding="utf-8",
    )
    (ct_qc_dir / "cohort_cache.json").write_text(
        json.dumps({
            "semantic_config_signature": ct_qc_config_signature({"threshold": 2}),
            "case_signatures": {"A": ct_case_signature(case_a)},
        }),
        encoding="utf-8",
    )
    observed = {}

    def fake_prepare(cases, *_args, **kwargs):
        observed["case_ids"] = [case["Case_ID"] for case in cases]
        (ct_qc_dir / "B").mkdir()
        return {"B": {"series_records": [], "prefilter_report_path": str(tmp_path / "report.json")}}

    monkeypatch.setattr(ct_qc, "prepare_ct_cases", fake_prepare)
    monkeypatch.setattr(
        ct_qc,
        "build_case_summary",
        lambda case_id, *_args: {"case_id": case_id, "case_qc_passes_threshold": False, "selected_series": {}},
    )
    result = ct_qc.run_ct_qc(
        [case_a, case_b], output_root=str(output_root), config_dir=str(config_dir)
    )
    assert observed == {"case_ids": ["B"]}
    assert set(result["selection_summaries"]) == {"A", "B"}


def test_ct_qc_reprocesses_changed_case_but_not_unchanged_case(tmp_path, monkeypatch):
    import json
    import tools.ct_qc as ct_qc

    config_dir = tmp_path / "configs"
    output_root = tmp_path / "output"
    config_dir.mkdir()
    (config_dir / "ct_qc.yaml").write_text("threshold: 2\n", encoding="utf-8")
    ct_qc_dir = output_root / "ct_qc"
    for case_id in ("A", "B"):
        case_dir = ct_qc_dir / case_id
        case_dir.mkdir(parents=True)
        (case_dir / "selection_summary.json").write_text(
            json.dumps({"case_qc_passes_threshold": False, "selected_series": {}}),
            encoding="utf-8",
        )
    old_a = {"Case_ID": "A", "CT": [{"File Path": "/a", "Series UID": "SA"}]}
    old_b = {"Case_ID": "B", "CT": [{"File Path": "/b-old", "Series UID": "SB"}]}
    new_b = {"Case_ID": "B", "CT": [{"File Path": "/b-new", "Series UID": "SB"}]}
    (ct_qc_dir / "cohort_cache.json").write_text(
        json.dumps({
            "semantic_config_signature": ct_qc_config_signature({"threshold": 2}),
            "case_signatures": {"A": ct_case_signature(old_a), "B": ct_case_signature(old_b)},
        }),
        encoding="utf-8",
    )
    observed = {}

    def fake_prepare(cases, *_args, **kwargs):
        observed["case_ids"] = [case["Case_ID"] for case in cases]
        (ct_qc_dir / "B").mkdir(exist_ok=True)
        return {"B": {"series_records": [], "prefilter_report_path": str(tmp_path / "report.json")}}

    monkeypatch.setattr(ct_qc, "prepare_ct_cases", fake_prepare)
    monkeypatch.setattr(
        ct_qc,
        "build_case_summary",
        lambda case_id, *_args: {"case_id": case_id, "case_qc_passes_threshold": False, "selected_series": {}},
    )
    result = ct_qc.run_ct_qc(
        [old_a, new_b], output_root=str(output_root), config_dir=str(config_dir)
    )
    assert observed == {"case_ids": ["B"]}
    assert set(result["selection_summaries"]) == {"A", "B"}


def test_confound_audit_reads_selected_phase(tmp_path):
    import json

    from tools.confound import CATEGORICAL_FIELDS, selected_ct_metadata

    case_id = "A"
    selection_dir = tmp_path / "ct_qc" / case_id
    selection_dir.mkdir(parents=True)
    (selection_dir / "selection_summary.json").write_text(
        json.dumps({"selected_series": {"inferred_phase": "NEPH"}}),
        encoding="utf-8",
    )
    state = {"inventory": {"CT": []}}
    assert "ct_phase" in CATEGORICAL_FIELDS
    assert selected_ct_metadata(case_id, state, str(tmp_path))["phase"] == "NEPH"
