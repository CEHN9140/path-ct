import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("04_experiment_known_ccrcc_subtype_core_mapping.py")
SPEC = importlib.util.spec_from_file_location("core_mapping", SCRIPT)
MAPPING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAPPING)

ROOT = Path(__file__).resolve().parent.parent


def test_current_seven_core_sizes_union_and_disjointness_are_frozen():
    cores = MAPPING.load_core_membership(
        ROOT / "output_kirc_v13/00_five_view_multi_k_agent_review/stable_core_membership.csv",
        MAPPING.SEVEN_CORE_SIZES,
    )
    assert {key: len(value) for key, value in cores.items()} == {
        "CORE01": 17, "CORE02": 15, "CORE03": 14, "CORE04": 10, "CORE05": 9, "CORE06": 7, "CORE07": 5
    }
    assert len(set().union(*cores.values())) == 77


def test_core_union_must_equal_analysis_universe():
    MAPPING.validate_core_union({"A": {"p1", "p2"}}, {"p1", "p2"})
    try:
        MAPPING.validate_core_union({"A": {"p1"}}, {"p1", "p2"})
    except ValueError as error:
        assert "extra_in_universe" in str(error)
    else:
        raise AssertionError("core/version mismatch must fail")


def test_real_core_union_matches_current_characterization_universe():
    cores = MAPPING.load_core_membership(ROOT / "output_kirc_v13/00_five_view_multi_k_agent_review/stable_core_membership.csv", MAPPING.SEVEN_CORE_SIZES)
    MAPPING.validate_core_union(cores, set().union(*cores.values()))


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
    assert MAPPING.no_fdr_supported_enrichment_in_either_reference(4, "no_fdr_supported_enrichment", 8, "no_fdr_supported_enrichment") is False


def test_core_enrichment_preserves_total_and_label_counts(tmp_path):
    results = MAPPING.analyze_reference(
        "5V-7state",
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


def test_run_partition_returns_current_reference_results(tmp_path):
    results = MAPPING.run_partition(
        "5V-7state",
        {"CORE01": ["p1", "p2"], "CORE03": ["p3", "p4"]},
        tmp_path,
        {"p1": {"reference_subtype": "m1", "reference_status": "matched"}, "p2": {"reference_subtype": "m1", "reference_status": "matched"}, "p3": {"reference_subtype": "m2", "reference_status": "matched"}, "p4": {"reference_subtype": "m2", "reference_status": "matched"}},
        {},
        20,
    )
    assert set(results) == {"mRNA", "ClearCode34"}


def test_load_reference_treats_nan_label_as_missing(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("case_id,reference_subtype,reference_status\nTCGA-A1-0001,,missing\n", encoding="utf-8")
    assert MAPPING.load_reference(path, {"m1"})["TCGA-A1-0001"]["reference_subtype"] is None


def test_core_clearcode_score_reuses_holm_not_bh():
    assert MAPPING.BASE.holm_adjust([0.01, 0.02, 0.5]) == [0.03, 0.04, 0.5]
