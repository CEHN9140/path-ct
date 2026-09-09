import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("04_experiment_known_ccrcc_subtype_core_mapping.py")
SPEC = importlib.util.spec_from_file_location("core_mapping", SCRIPT)
MAPPING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAPPING)

ROOT = Path(__file__).resolve().parent.parent


def test_four_view_core_sizes_union_and_disjointness_are_frozen():
    cores = MAPPING.load_core_membership(
        ROOT / "output_kirc_v12/03_multi_k_accepted_core_stability_v11/stable_core_membership.csv",
        {"CORE01": 18, "CORE02": 14, "CORE03": 11, "CORE04": 11, "CORE05": 5},
    )
    assert {key: len(value) for key, value in cores.items()} == {
        "CORE01": 18, "CORE02": 14, "CORE03": 11, "CORE04": 11, "CORE05": 5
    }
    assert len(set().union(*cores.values())) == 59


def test_five_view_core_sizes_are_frozen():
    cores = MAPPING.load_core_membership(
        ROOT / "output_kirc_v12/15_five_view_multi_k_stability/stable_core_membership.csv",
        {"CORE01": 17, "CORE02": 14, "CORE03": 14, "CORE04": 10, "CORE05": 9, "CORE06": 5},
    )
    assert {key: len(value) for key, value in cores.items()} == {
        "CORE01": 17, "CORE02": 14, "CORE03": 14, "CORE04": 10, "CORE05": 9, "CORE06": 5
    }
    assert len(set().union(*cores.values())) == 69


def test_core_union_must_equal_analysis_universe():
    MAPPING.validate_core_union({"A": {"p1", "p2"}}, {"p1", "p2"})
    try:
        MAPPING.validate_core_union({"A": {"p1"}}, {"p1", "p2"})
    except ValueError as error:
        assert "extra_in_universe" in str(error)
    else:
        raise AssertionError("core/version mismatch must fail")


def test_real_core_unions_equal_the_current_macro_analysis_universes():
    four = MAPPING.load_core_membership(ROOT / "output_kirc_v12/03_multi_k_accepted_core_stability_v11/stable_core_membership.csv", {"CORE01": 18, "CORE02": 14, "CORE03": 11, "CORE04": 11, "CORE05": 5})
    five = MAPPING.load_core_membership(ROOT / "output_kirc_v12/15_five_view_multi_k_stability/stable_core_membership.csv", {"CORE01": 17, "CORE02": 14, "CORE03": 14, "CORE04": 10, "CORE05": 9, "CORE06": 5})
    MAPPING.validate_core_union(four, MAPPING.load_universe(ROOT / "output_kirc_v13/02_post_discovery_characterization/4view_3state/analysis_universe.csv"))
    MAPPING.validate_core_union(five, MAPPING.load_universe(ROOT / "output_kirc_v13/02_post_discovery_characterization/5view_4state/analysis_universe.csv"))


def test_missing_reference_labels_are_not_in_contingency():
    table = MAPPING.core_contingency(
        {"CORE01": ["p1", "p2"], "CORE02": ["p3"]},
        {"p1": "m1", "p2": None, "p3": "m2"},
        ["m1", "m2"],
    )
    assert table.tolist() == [[1, 0], [0, 1]]


def test_status_low_information_and_unresolved_require_at_least_five_labels():
    assert MAPPING.evidence_status(4, []) == "low_information"
    assert MAPPING.evidence_status(5, []) == "no_fdr_supported_enrichment"
    assert MAPPING.unresolved_by_both_references(4, "no_fdr_supported_enrichment", 8, "no_fdr_supported_enrichment") is False


def test_component_heterogeneity_only_returns_multi_core_macros():
    macros = {"STATE_A": ("CORE01", "CORE03"), "STATE_C": ("CORE04",)}
    rows = MAPPING.component_heterogeneity_rows(
        "4V", macros, {"CORE01": ["p1"], "CORE03": ["p2"], "CORE04": ["p3"]},
        {"p1": "m1", "p2": "m2", "p3": "m1"}, "mRNA", ["m1", "m2"], 20,
    )
    assert {row["macro_state"] for row in rows} == {"STATE_A"}


def test_weighted_component_purity_and_purity_drop_are_calculated():
    row = MAPPING.macro_mixing_row(
        "4V", "STATE_A", ["CORE01", "CORE03"],
        {"CORE01": {"label_n": 10, "purity": .9}, "CORE03": {"label_n": 10, "purity": .5}},
        {"m1": 14, "m2": 6}, {"m1", "m2"}, None, None,
    )
    assert np.isclose(row["weighted_component_purity"], .7)
    assert np.isclose(row["macro_purity"], .7)
    assert np.isclose(row["purity_drop"], 0.0)


def test_core_enrichment_preserves_total_and_label_counts(tmp_path):
    results = MAPPING.analyze_reference(
        "4V-5core",
        {"CORE01": ["p1", "p2", "p3"], "CORE02": ["p4", "p5"]},
        {"p1": {"reference_subtype": "m1", "reference_status": "matched"}, "p2": {"reference_subtype": "m1", "reference_status": "matched"}, "p3": {"reference_subtype": None, "reference_status": "missing"}, "p4": {"reference_subtype": "m2", "reference_status": "matched"}, "p5": {"reference_subtype": "m2", "reference_status": "matched"}},
        "mRNA",
        ["m1", "m2"],
        tmp_path,
        20,
    )
    row = next(item for item in results["enrichment"] if item["core_id"] == "CORE01" and item["reference_subtype"] == "m1")
    assert row["core_total_n"] == 3
    assert row["core_label_n"] == 2
    assert row["rest_label_n"] == 2


def test_run_partition_returns_macro_rows_for_outer_aggregation(tmp_path):
    _, component_rows, mixing_rows = MAPPING.run_partition(
        "4V-5core",
        {"CORE01": ["p1", "p2"], "CORE03": ["p3", "p4"]},
        {"STATE_A": ("CORE01", "CORE03")},
        tmp_path,
        {"p1": {"reference_subtype": "m1", "reference_status": "matched"}, "p2": {"reference_subtype": "m1", "reference_status": "matched"}, "p3": {"reference_subtype": "m2", "reference_status": "matched"}, "p4": {"reference_subtype": "m2", "reference_status": "matched"}},
        {},
        20,
    )
    assert component_rows
    assert mixing_rows


def test_load_reference_treats_nan_label_as_missing(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("case_id,reference_subtype,reference_status\nTCGA-A1-0001,,missing\n", encoding="utf-8")
    assert MAPPING.load_reference(path, {"m1"})["TCGA-A1-0001"]["reference_subtype"] is None


def test_core_clearcode_score_reuses_holm_not_bh():
    assert MAPPING.BASE.holm_adjust([0.01, 0.02, 0.5]) == [0.03, 0.04, 0.5]
