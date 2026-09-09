import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np


SCRIPT = Path(__file__).with_name("03_experiment_known_ccrcc_subtype_mapping.py")
SPEC = importlib.util.spec_from_file_location("known_ccrcc_mapping", SCRIPT)
MAPPING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAPPING)


def test_normalize_tcga_patient_id_accepts_patient_and_sample_barcodes():
    assert MAPPING.normalize_tcga_patient_id("tcga-b0-4698-01a") == "TCGA-B0-4698"
    assert MAPPING.normalize_tcga_patient_id("bad-id") is None


def test_prepare_mrna_labels_detects_consistent_and_conflicting_duplicates(tmp_path):
    source = tmp_path / "mrna.xlsx"
    pd.DataFrame(
        [
            ["TCGA-A1-0001", 1, 2],
            ["TCGA-A1-0001-01", 1, 2],
            ["TCGA-A1-0002", 2, 1],
            ["TCGA-A1-0002", 3, 1],
        ],
        columns=["Patient", "mRNA_cluster", "microRNA_cluster"],
    ).to_excel(source, sheet_name="mRNA_miRNA_cluster_assignments", index=False)
    frame, audit = MAPPING.prepare_mrna_labels(source)
    rows = frame.set_index("case_id")
    assert rows.loc["TCGA-A1-0001", "reference_subtype"] == "m1"
    assert rows.loc["TCGA-A1-0001", "reference_status"] == "duplicate_consistent"
    assert pd.isna(rows.loc["TCGA-A1-0002", "reference_subtype"])
    assert rows.loc["TCGA-A1-0002", "reference_status"] == "conflict"
    assert audit["duplicate_consistent"] == 1
    assert audit["duplicate_conflict"] == 1


def test_parse_clearcode_uses_table3_only_and_published_label():
    text = """
Supplemental Table 3
TCGA-A1-0001
9.8E-01
2.0E-02
ccA
TCGA-A1-0002
4.0E-01
6.0E-01
ccB
Supplemental Table 4
40
1.0E+00
0.0E+00
ccA
"""
    frame, audit = MAPPING.parse_clearcode_text(text)
    assert frame["case_id"].tolist() == ["TCGA-A1-0001", "TCGA-A1-0002"]
    assert frame["reference_subtype"].tolist() == ["ccA", "ccB"]
    assert bool(frame.loc[0, "classification_consistency"])
    assert audit["valid_tcga_rows"] == 2


def test_contingency_and_bias_corrected_cramers_v_are_deterministic():
    table = MAPPING.build_contingency(
        {"A": "A", "B": "A", "C": "B", "D": "B"},
        {"A": "m1", "B": "m1", "C": "m2", "D": "m2"},
        ["A", "B"],
        ["m1", "m2"],
    )
    assert table.tolist() == [[2, 0], [0, 2]]
    assert MAPPING.bias_corrected_cramers_v(table) == 1.0


def test_bias_corrected_cramers_v_uses_nonempty_contingency_dimensions():
    table = np.array([[0, 8, 3, 2], [0, 1, 3, 1]])
    assert np.isclose(MAPPING.bias_corrected_cramers_v(table), 0.1961161351)


def test_enrichment_bh_is_applied_across_one_partition_reference_family():
    rows = [
        {"state": "A", "reference_subtype": "m1", "fisher_p": 0.01},
        {"state": "A", "reference_subtype": "m2", "fisher_p": 0.02},
        {"state": "B", "reference_subtype": "m1", "fisher_p": 0.5},
    ]
    MAPPING.attach_bh(rows, "fisher_p", "bh_q")
    assert [row["bh_q"] for row in rows] == [0.03, 0.03, 0.5]


def test_holm_adjust_for_clearcode_posthoc():
    assert MAPPING.holm_adjust([0.01, 0.02, 0.5]) == [0.03, 0.04, 0.5]


def test_raw_cramers_v_for_3x4_table():
    table = MAPPING.build_contingency(
        {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C", "c2": "C"},
        {"a1": "m1", "a2": "m1", "b1": "m2", "b2": "m2", "c1": "m3", "c2": "m3"},
        ["A", "B", "C"],
        ["m1", "m2", "m3", "m4"],
    )
    chi = MAPPING.chi_square_stat(table)
    expected = (chi / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1))) ** 0.5
    assert np.isclose(MAPPING.cramers_v_raw(table), expected)
