import importlib.util
from pathlib import Path


module_path = Path(__file__).with_name("07_experiment_technical_confounder_audit.py")
spec = importlib.util.spec_from_file_location("technical_confounder_audit", module_path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

TECHNICAL_VARIABLES = audit.TECHNICAL_VARIABLES
build_availability_rows = audit.build_availability_rows
build_cnv_proxy = audit.build_cnv_proxy
build_wxs_proxy = audit.build_wxs_proxy
build_wsi_proxy = audit.build_wsi_proxy
bh = audit.bh


def test_unavailable_metadata_is_explicitly_marked():
    rows = build_availability_rows()
    by_name = {row["variable"]: row for row in rows}

    assert by_name["rna_batch"].get("status") == "unavailable"
    assert by_name["wxs_purity"].get("status") == "unavailable"
    assert by_name["cnv_ploidy"].get("status") == "unavailable"
    assert all(row["status"] in {"available", "unavailable"} for row in rows)
    assert set(TECHNICAL_VARIABLES) == set(by_name)


def test_wsi_proxy_uses_only_generated_qc_summary():
    row = build_wsi_proxy(
        "C1",
        {"patch_count": 100, "tumor_patch_count": 25, "model_mpp": 0.504, "model_tile_size": 512},
    )
    assert row["wsi_patch_count"] == 100
    assert row["wsi_tumor_patch_fraction"] == 0.25
    assert "wsi_scanner_vendor" not in row


def test_wxs_proxy_does_not_call_all_zero_missing_qc():
    row = build_wxs_proxy("C1", {"mutation::A": 0.0, "mutation::B": 0.0})
    assert row["wxs_discovery_mutation_count"] == 0
    assert row["wxs_discovery_all_zero_proxy"] is True
    assert row["wxs_discovery_all_zero_proxy_interpretation"] == "representation_proxy_not_qc"


def test_cnv_proxy_reports_feature_completeness():
    row = build_cnv_proxy("C1", {"chr1p": 0.1, "gain_burden": 0.2, "loss_burden": 0.3})
    assert row["cnv_missing_feature_count"] == 0
    assert row["cnv_gain_burden"] == 0.2
    assert row["cnv_loss_burden"] == 0.3
    assert "cnv_purity" not in row


def test_bh_sorts_by_p_value():
    assert bh([0.01, 0.04, 0.03, 0.002]) == [0.02, 0.04, 0.04, 0.008]


def test_primary_groups_are_four_stable_cores_only():
    assert hasattr(audit, "primary_core_groups")
    primary_core_groups = audit.primary_core_groups
    groups = primary_core_groups(
        {"CORE01": ["a"], "CORE02": ["b"], "CORE03": ["c"], "CORE04": ["d"]}
    )
    assert list(groups) == ["CORE01", "CORE02", "CORE03", "CORE04"]
    assert groups["CORE01"] == ["a"]
    assert "non_core" not in groups
