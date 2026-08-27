import csv
import json
from pathlib import Path

import numpy as np

from scripts_2026_8_17 import analyze_multi_k_stable_cores as analysis


def test_mapping_and_composition_are_patient_set_based():
    cores = {"CORE01": {"a", "b", "c"}, "CORE02": {"d"}}
    main = [
        {"set_id": "C0001", "member_ids": ["a", "b", "x"], "decision": "accept"},
        {"set_id": "C0002", "member_ids": ["c", "d"], "decision": "drop"},
    ]
    rows, composition = analysis.main_mapping(cores, main)
    row = next(item for item in rows if item["core_id"] == "CORE01" and item["main_set_id"] == "C0001")
    assert row["intersection_n"] == 2
    assert row["core_fraction_in_main"] == 2 / 3
    assert row["main_fraction_captured"] == 2 / 3
    assert composition["CORE01"] == {"C0001": 2, "C0002": 1}


def test_core_vs_rest_and_pairwise_fdr_are_separate():
    rows = analysis.test_rows(
        {"CORE01": [1.0, 2.0], "CORE02": [8.0, 9.0]},
        [1.0, 2.0, 8.0, 9.0],
        feature="x",
    )
    assert len(rows) == 2
    assert all(row["q_value"] is not None for row in rows)
    table = {case_id: {"x": value} for case_id, value in {"a": 1.0, "b": 2.0, "c": 8.0, "d": 9.0}.items()}
    pair = analysis.pairwise_numeric_rows(
        table, {"CORE01": ["a", "b"], "CORE02": ["c", "d"]}, feature="x"
    )
    assert pair[0]["core_a"] == "CORE01"
    assert pair[0]["core_b"] == "CORE02"
    assert pair[0]["q_value"] is not None


def test_affinity_metrics_use_fixed_membership_and_core_distance():
    ids = ["a", "b", "c", "d"]
    similarity = np.array(
        [[1, .9, .1, .2], [.9, 1, .2, .1], [.1, .2, 1, .8], [.2, .1, .8, 1]],
        dtype=float,
    )
    result = analysis.affinity_characterization(
        similarity, ids, {"CORE01": ["a", "b"], "CORE02": ["c", "d"]}
    )
    assert result["per_core"]["CORE01"]["silhouette"] > 0
    assert result["distance_matrix"]["CORE01"]["CORE02"] > 0


def test_png_only_writer_never_creates_pdf(tmp_path):
    path = analysis.write_png(tmp_path / "figure.png", 20, 20, lambda canvas: canvas.rectangle(0, 0, 19, 19, (255, 0, 0)))
    assert path.suffix == ".png" and path.exists()
    assert not list(tmp_path.glob("*.pdf"))


def test_run_summary_has_non_core_and_manifest_hash(tmp_path):
    summary = analysis.build_summary(
        {"CORE01": ["a", "b"]},
        ["a", "b", "c"],
        {"CORE01": {"C0001": 2}},
        {"CORE01": []},
        {"CORE01": []},
    )
    assert summary["patient_count"] == 3
    assert summary["stable_core_patient_count"] == 2
    assert summary["non_core_patient_count"] == 1
