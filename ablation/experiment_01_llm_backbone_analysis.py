#!/usr/bin/env python3
"""Quantify agreement between completed LLM-backbone ablation outputs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ("gpt56sol", "deepseek_v4_pro", "qwen38max")


def load_partition(model_root: Path) -> dict:
    candidate_root = model_root / "candidate_subtype"
    patient_ids = [str(value) for value in json.loads(
        (candidate_root / "affinity_patient_order.json").read_text(encoding="utf-8")
    )]
    subtypes = json.loads(
        (model_root / "subtype_review" / "multi_k" / "patient_recurrence_subtypes.json")
        .read_text(encoding="utf-8")
    )
    labels_by_id = {}
    subtype_sets = []
    patient_set = set(patient_ids)
    for subtype_index, subtype in enumerate(subtypes, 1):
        members = {str(value) for value in subtype["member_ids"]}
        if not members.issubset(patient_set):
            raise ValueError(f"{model_root}: subtype contains an unknown patient.")
        if labels_by_id.keys() & members:
            raise ValueError(f"{model_root}: subtype memberships overlap.")
        subtype_sets.append(members)
        labels_by_id.update({patient_id: subtype_index for patient_id in members})
    labels = np.asarray([labels_by_id.get(patient_id, 0) for patient_id in patient_ids])
    return {
        "patient_ids": patient_ids,
        "labels": labels,
        "subtype_sets": subtype_sets,
        "covered_count": len(labels_by_id),
        "aggregation_summary": json.loads(
            (model_root / "subtype_review" / "multi_k" / "aggregation_summary.json")
            .read_text(encoding="utf-8")
        ),
    }


def matched_jaccard(left_sets: list[set[str]], right_sets: list[set[str]]) -> dict:
    if not left_sets and not right_sets:
        return {"matched_mean_jaccard": 1.0, "left_to_right_mean_jaccard": 1.0, "right_to_left_mean_jaccard": 1.0}
    matrix = np.zeros((len(left_sets), len(right_sets)))
    for i, left in enumerate(left_sets):
        for j, right in enumerate(right_sets):
            matrix[i, j] = len(left & right) / len(left | right)
    left_best = matrix.max(axis=1) if len(left_sets) else np.array([])
    right_best = matrix.max(axis=0) if len(right_sets) else np.array([])
    size = max(len(left_sets), len(right_sets))
    padded = np.zeros((size, size))
    padded[:matrix.shape[0], :matrix.shape[1]] = matrix
    rows, columns = linear_sum_assignment(-padded)
    matched = padded[rows, columns]
    return {
        "matched_mean_jaccard": float(matched.mean()),
        "left_to_right_mean_jaccard": float(left_best.mean()) if len(left_best) else 0.0,
        "right_to_left_mean_jaccard": float(right_best.mean()) if len(right_best) else 0.0,
    }


def compare_partitions(left: dict, right: dict) -> dict:
    if left["patient_ids"] != right["patient_ids"]:
        raise ValueError("LLM outputs use different candidate patient orders.")
    labels_left = left["labels"]
    labels_right = right["labels"]
    common_covered = (labels_left != 0) & (labels_right != 0)
    upper = np.triu_indices(len(labels_left), k=1)
    same_left = (labels_left[:, None] == labels_left[None, :]) & (labels_left[:, None] != 0) & (labels_left[None, :] != 0)
    same_right = (labels_right[:, None] == labels_right[None, :]) & (labels_right[:, None] != 0) & (labels_right[None, :] != 0)
    assigned_by_either = (
        ((labels_left[:, None] != 0) | (labels_right[:, None] != 0))
        & ((labels_left[None, :] != 0) | (labels_right[None, :] != 0))
    )
    valid_pairs = assigned_by_either[upper]
    return {
        "model_a": None,
        "model_b": None,
        "subtype_count_a": len(left["subtype_sets"]),
        "subtype_count_b": len(right["subtype_sets"]),
        "stable_patient_coverage_a": left["covered_count"] / len(labels_left),
        "stable_patient_coverage_b": right["covered_count"] / len(labels_right),
        "common_covered_patient_count": int(common_covered.sum()),
        "ari_all_candidates": float(adjusted_rand_score(labels_left, labels_right)),
        "nmi_all_candidates": float(normalized_mutual_info_score(labels_left, labels_right)),
        "coassignment_agreement_all_candidates": float((same_left[upper][valid_pairs] == same_right[upper][valid_pairs]).mean()) if valid_pairs.any() else None,
        "ari_common_covered": float(adjusted_rand_score(labels_left[common_covered], labels_right[common_covered])) if common_covered.sum() > 1 else None,
        "nmi_common_covered": float(normalized_mutual_info_score(labels_left[common_covered], labels_right[common_covered])) if common_covered.sum() > 1 else None,
        **matched_jaccard(left["subtype_sets"], right["subtype_sets"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "ablation/results/01_llm_backbone")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ablation/results/01_llm_backbone_analysis")
    parser.add_argument("--model", dest="model_keys", choices=MODEL_KEYS, action="append")
    args = parser.parse_args()
    model_keys = tuple(args.model_keys or MODEL_KEYS)
    if len(model_keys) < 2:
        raise ValueError("At least two model outputs are required for comparison.")
    partitions = {key: load_partition(args.input_root / key) for key in model_keys}
    model_rows = []
    for key, partition in partitions.items():
        model_rows.append({
            "model": key,
            "candidate_patient_count": len(partition["patient_ids"]),
            "stable_patient_count": partition["covered_count"],
            "stable_patient_coverage": partition["covered_count"] / len(partition["patient_ids"]),
            "stable_subtype_count": len(partition["subtype_sets"]),
            "included_run_count": partition["aggregation_summary"]["included_run_count"],
        })
    pair_rows = []
    for left_key, right_key in itertools.combinations(model_keys, 2):
        row = compare_partitions(partitions[left_key], partitions[right_key])
        row["model_a"] = left_key
        row["model_b"] = right_key
        pair_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pairwise_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "experiment": "llm_backbone_cross_model_analysis",
        "input_root": str(args.input_root.resolve()),
        "models": list(model_keys),
        "metrics": [
            "patient_overlap_jaccard",
            "ARI",
            "NMI",
            "coassignment_agreement",
            "stable_patient_coverage",
            "stable_subtype_count",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"models": list(model_keys), "pair_count": len(pair_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
