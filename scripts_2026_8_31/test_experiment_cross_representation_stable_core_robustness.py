import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("17_experiment_cross_representation_stable_core_robustness.py")
SPEC = importlib.util.spec_from_file_location("cross_representation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_core_overlap_reports_bidirectional_coverage_and_enrichment():
    row = MODULE.overlap_row("A", ["p1", "p2", "p3"], "B", ["p2", "p3", "p4"], 5)
    assert row["intersection_n"] == 2
    assert row["four_to_five_coverage"] == 2 / 3
    assert row["five_to_four_coverage"] == 2 / 3
    assert row["jaccard"] == 0.5

