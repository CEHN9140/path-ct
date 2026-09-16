#!/usr/bin/env python3
"""Quantify agreement between frozen macro-states and their micro-cores."""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from four_view_state_common import (
    DEFAULT_INPUT,
    DEFAULT_MEMBERSHIP,
    ROOT,
    groups,
    load_membership,
    load_states,
    write_manifest,
)
from tools.multimodal_consistency_check import normalized_affinity_with_audit
from tools.post_discovery_characterization import bh_adjust
from tools.subtype_review_common import clinical_table

PERMUTATIONS = 9999


def pair_statistic(pair_values, state_by_core, target=None, weights=None):
    within, outside = [], []
    for pair, value in pair_values.items():
        left, right = pair
        same = state_by_core[left] == state_by_core[right]
        if target is not None and target not in (state_by_core[left], state_by_core[right]):
            continue
        (within if same else outside).append((value, weights[pair] if weights else 1.0))
    if not within or not outside:
        return None
    average = lambda values: np.average(
        [value for value, _ in values], weights=[weight for _, weight in values]
    )
    return float(average(within) - average(outside))


def permutation_p(pair_values, state_by_core, target, rng, weights=None):
    observed = pair_statistic(pair_values, state_by_core, target, weights)
    if observed is None:
        return None, None
    cores = list(state_by_core)
    state_labels = list(state_by_core.values())
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = dict(zip(cores, rng.permutation(state_labels)))
        value = pair_statistic(pair_values, shuffled, target, weights)
        if value is not None:
            null.append(value)
    p_value = (1 + sum(abs(value) >= abs(observed) for value in null)) / (len(null) + 1)
    return observed, p_value


def run(
    input_root=DEFAULT_INPUT,
    membership=DEFAULT_MEMBERSHIP,
    output_root=ROOT / "output_kirc_v14/16_four_view_macro_micro_consistency",
    force=False,
):
    input_root, membership, output_root = Path(input_root), Path(membership), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame = load_membership(membership)
    state_groups = groups(frame)
    states = load_states(input_root)
    core_ids = sorted(frame.core_id.unique())
    state_by_core = {
        core: frame.loc[frame.core_id.eq(core), "state_id"].iloc[0] for core in core_ids
    }
    order = json.loads((input_root / "candidate_subtype/affinity_patient_order.json").read_text())
    index = {case_id: i for i, case_id in enumerate(order)}
    pair_rows, statistics, weighted_statistics = [], [], []
    rng = np.random.default_rng(20260916)
    paths = {
        "ct": input_root / "candidate_subtype/ct_affinity.npy",
        "wsi": input_root / "candidate_subtype/wsi_affinity.npy",
        "rna": input_root / "candidate_subtype/rna_affinity.npy",
        "wxs": input_root / "wxs/wxs_affinity.npy",
    }

    for modality, path in paths.items():
        matrix, _ = normalized_affinity_with_audit(np.load(path))
        pair_values, pair_weights = {}, {}
        for left, right in combinations(core_ids, 2):
            left_ids = frame.loc[frame.core_id.eq(left), "case_id"].tolist()
            right_ids = frame.loc[frame.core_id.eq(right), "case_id"].tolist()
            value = float(matrix[np.ix_([index[x] for x in left_ids], [index[x] for x in right_ids])].mean())
            pair = (left, right)
            pair_values[pair] = value
            pair_weights[pair] = len(left_ids) * len(right_ids)
            left_state, right_state = state_by_core[left], state_by_core[right]
            pair_rows.append({
                "modality": modality,
                "state_id": left_state if left_state == right_state else "between_states",
                "state_a": left_state,
                "state_b": right_state,
                "core_a": left,
                "core_b": right,
                "n_a": len(left_ids),
                "n_b": len(right_ids),
                "mean_similarity": value,
            })
        for weighted, output in ((False, statistics), (True, weighted_statistics)):
            weights = pair_weights if weighted else None
            for state in sorted(state_groups) + ["ALL"]:
                observed, p_value = permutation_p(
                    pair_values, state_by_core, None if state == "ALL" else state, rng, weights
                )
                output.append({
                    "modality": modality,
                    "state_id": state,
                    "within_minus_between": observed,
                    "permutation_p_value": p_value,
                    "weighted": weighted,
                    "analysis_role": "discovery_space_diagnostic",
                })

    for output in (statistics, weighted_statistics):
        for rows in (
            [row for row in output if row["state_id"] == "ALL"],
            [row for row in output if row["state_id"] != "ALL"],
        ):
            for row, q_value in zip(rows, bh_adjust([row["permutation_p_value"] for row in rows])):
                row["q_value"] = q_value
    pd.DataFrame(pair_rows).to_csv(output_root / "micro_core_pair_similarity.csv", index=False)
    pd.DataFrame(statistics).drop(columns="weighted").to_csv(
        output_root / "macro_micro_consistency_statistics.csv", index=False
    )
    pd.DataFrame(weighted_statistics).drop(columns="weighted").rename(
        columns={"within_minus_between": "weighted_within_minus_between"}
    ).to_csv(output_root / "macro_micro_consistency_weighted.csv", index=False)

    composition = [
        {
            "state_id": state,
            "state_n": len(members),
            "micro_core_count": frame.loc[frame.state_id.eq(state), "core_id"].nunique(),
            "micro_core_sizes": json.dumps(
                frame.loc[frame.state_id.eq(state), "core_id"].value_counts().sort_index().to_dict()
            ),
        }
        for state, members in state_groups.items()
    ]
    pd.DataFrame(composition).to_csv(output_root / "state_micro_core_composition.csv", index=False)

    records = clinical_table({case_id: states[case_id] for case_id in order if case_id in states})
    clinical_rows = []
    for state in sorted(state_groups):
        for core in sorted(frame.loc[frame.state_id.eq(state), "core_id"].unique()):
            case_ids = frame.loc[frame.core_id.eq(core), "case_id"].tolist()
            ages = [records[case_id]["age"] for case_id in case_ids if records[case_id].get("age") is not None]
            clinical_rows.append({
                "state_id": state,
                "core_id": core,
                "core_n": len(case_ids),
                "age_median": float(np.median(ages)) if ages else None,
                "stage_counts": json.dumps(
                    pd.Series([records[case_id].get("stage_group", "") for case_id in case_ids]).value_counts().to_dict()
                ),
            })
    pd.DataFrame(clinical_rows).to_csv(output_root / "within_state_micro_core_clinical.csv", index=False)
    write_manifest(output_root / "manifest.json", {
        "experiment": "four_view_macro_micro_consistency",
        "active_modalities": ["ct", "wsi", "rna", "wxs"],
        "membership_file": str(membership.resolve()),
        "analysis_patient_count": len(frame),
        "source_cohort_n": len(order),
        "cnv_included": False,
        "analysis_scope": "in-sample diagnostic; states were derived from these modalities",
        "consistency_test": "core-label permutation preserving macro-state core counts",
        "weighted_sensitivity": True,
        "weighted_metric": "pairwise mean similarity weighted by product of micro-core sizes",
        "permutations": PERMUTATIONS,
    })
    return {"output_root": str(output_root), "state_sizes": {k: len(v) for k, v in state_groups.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/16_four_view_macro_micro_consistency")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
