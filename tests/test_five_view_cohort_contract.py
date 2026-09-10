from pathlib import Path

from agents.inventory import inventory_case


def complete_case(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for modality in ("CT", "WSI", "RNA_Seq", "WXS", "CNV"):
        path = tmp_path / modality
        path.touch()
        paths[modality] = [{"File Path": str(path)}]
    return {"Case_ID": "A", **paths}


def test_inventory_marks_missing_wxs_as_failed(tmp_path):
    case = complete_case(tmp_path)
    case["WXS"] = []

    state = inventory_case(case)

    assert state["qc"] == "fail"
    assert state["missing_view_reason"] == ["missing_wxs"]


def test_five_view_filter_excludes_missing_omics_before_downstream_builders(tmp_path):
    from agents.evidence_builder import filter_five_view_states

    complete = complete_case(tmp_path / "complete")
    missing = complete_case(tmp_path / "missing")
    missing["WXS"] = []
    states = [
        {
            "case_id": "A",
            "qc": "success",
            "inventory": complete,
            "ct_evidence": {"feature_path": complete["CT"][0]["File Path"]},
            "wsi_evidence": {"feature_path": complete["WSI"][0]["File Path"]},
        },
        {
            "case_id": "B",
            "qc": "success",
            "inventory": missing,
            "ct_evidence": {"feature_path": missing["CT"][0]["File Path"]},
            "wsi_evidence": {"feature_path": missing["WSI"][0]["File Path"]},
        },
    ]

    updated, eligible, audit = filter_five_view_states(states)

    assert [state["case_id"] for state in eligible] == ["A"]
    assert updated[1]["qc"] == "fail"
    assert updated[1]["missing_view_reason"] == ["missing_wxs"]
    assert audit["pre_qc_count"] == 2
    assert audit["ct_wsi_pass_count"] == 2
    assert audit["five_view_complete_count"] == 1
    assert audit["excluded_missing_rna"] == []
    assert audit["excluded_missing_wxs"] == ["B"]
