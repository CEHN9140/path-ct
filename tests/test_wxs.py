import gzip
import json

import numpy as np
import pandas as pd
import pytest

from tools.wxs import (
    binary_mutation_distance,
    collect_wxs_file_paths,
    load_complete_cnv_matrix,
    read_wxs_mutations,
)


def test_binary_mutation_distance_uses_zero_for_two_empty_vectors():
    distance = binary_mutation_distance(np.array([[0, 0, 0], [0, 0, 0]]), 0.0)
    assert distance[0, 1] == 0.0


def test_binary_mutation_distance_keeps_standard_jaccard_for_nonempty_pairs():
    distance = binary_mutation_distance(np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0]]), 0.0)
    assert distance[0, 1] == 1.0
    assert distance[0, 2] == 0.5


def test_wxs_manifest_must_cover_every_target_patient(tmp_path):
    with gzip.open(tmp_path / "A.maf.gz", "wt", encoding="utf-8") as handle:
        handle.write("")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"input_cases": [{"case_id": "A", "file_path": str(tmp_path / "A.maf.gz")}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing target patients: B"):
        read_wxs_mutations(manifest, ["A", "B"])


def test_cnv_matrix_must_cover_every_target_patient(tmp_path):
    path = tmp_path / "cnv.csv"
    pd.DataFrame([{"case_id": "A", "feature": 0.1}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing target patients: B"):
        load_complete_cnv_matrix(path, ["A", "B"])


def test_cnv_matrix_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "cnv.csv"
    pd.DataFrame([{"case_id": "A", "feature": np.nan}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        load_complete_cnv_matrix(path, ["A"])


def test_wxs_selects_highest_depth_aliquot(tmp_path):
    columns = ["Hugo_Symbol", "t_depth"]
    paths = []
    for name, depth in (("low.maf.gz", 20), ("high.maf.gz", 80)):
        path = tmp_path / name
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("\t".join(columns) + "\nGENE\t" + str(depth) + "\n")
        paths.append(path)

    selected = collect_wxs_file_paths(
        [{"Case_ID": "A", "WXS": [{"File Path": str(path)} for path in paths]}]
    )

    assert selected == [("A", str(paths[1]))]
