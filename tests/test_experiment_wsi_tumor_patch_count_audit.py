import json

from scripts_2026_8_17.experiment_wsi_tumor_patch_count_audit import (
    build_audit_rows,
    summarize_patch_counts,
)


def test_build_audit_rows_joins_grandqc_and_tumor_counts():
    tumor = {
        "A": {"case_id": "A", "slide_path": "/a.svs", "patch_count": 300, "tumor_patch_count": 250},
        "B": {"case_id": "B", "slide_path": "/b.svs", "patch_count": 100, "tumor_patch_count": 49},
    }
    grandqc = {"A": True, "B": False}
    rows = build_audit_rows(tumor, grandqc)
    assert rows[0]["case_id"] == "A"
    assert rows[0]["grandqc_pass"] is True
    assert rows[1]["tumor_patch_count"] == 49


def test_summarize_patch_counts_reports_requested_thresholds():
    rows = [
        {"case_id": "A", "tumor_patch_count": 250},
        {"case_id": "B", "tumor_patch_count": 100},
        {"case_id": "C", "tumor_patch_count": 49},
    ]
    summary = summarize_patch_counts(rows)
    assert summary["n"] == 3
    assert summary["min"] == 49
    assert summary["thresholds"]["50"]["pass_count"] == 2
    assert summary["thresholds"]["100"]["pass_count"] == 2
    assert summary["thresholds"]["250"]["pass_count"] == 1
