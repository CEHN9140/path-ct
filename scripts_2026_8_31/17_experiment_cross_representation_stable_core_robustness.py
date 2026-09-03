#!/usr/bin/env python3
"""Offline 4-view versus 5-view stable-core robustness audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_membership(path):
    cores = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cores.setdefault(row["core_id"], set()).add(row["patient_id"])
    return {key: sorted(value) for key, value in sorted(cores.items())}


def read_matrix(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    ids = rows[0][1:]
    matrix = np.asarray([[float(value) if value.lower() != "nan" else np.nan for value in row[1:]] for row in rows[1:]], dtype=float)
    return ids, matrix


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def overlap_row(core_a, members_a, core_b, members_b, universe_size):
    left, right = set(members_a), set(members_b)
    intersection = len(left & right)
    union = len(left | right)
    from scipy.stats import fisher_exact, hypergeom
    _, fisher_p = fisher_exact([[intersection, len(left) - intersection], [len(right) - intersection, universe_size - union]])
    hypergeom_p = hypergeom.sf(intersection - 1, universe_size, len(left), len(right))
    return {
        "core_4v": core_a,
        "core_5v": core_b,
        "core_4v_n": len(left),
        "core_5v_n": len(right),
        "intersection_n": intersection,
        "union_n": union,
        "jaccard": intersection / union if union else 0.0,
        "dice": 2 * intersection / (len(left) + len(right)) if left or right else 0.0,
        "overlap_coefficient": intersection / min(len(left), len(right)) if left and right else 0.0,
        "four_to_five_coverage": intersection / len(left) if left else 0.0,
        "five_to_four_coverage": intersection / len(right) if right else 0.0,
        "fisher_p": float(fisher_p),
        "hypergeometric_p": float(hypergeom_p),
    }


def pair_values(ids, matrix):
    return {(a, b): float(matrix[i, j]) for i, a in enumerate(ids) for j, b in enumerate(ids) if i < j and np.isfinite(matrix[i, j])}


def pair_rows(ids, matrix, representation):
    return [{"representation": representation, "patient_a": a, "patient_b": b, "conditional_same_set": value} for (a, b), value in pair_values(ids, matrix).items()]


def cross_pair_rows(four_ids, four_matrix, five_ids, five_matrix):
    four, five = pair_values(four_ids, four_matrix), pair_values(five_ids, five_matrix)
    rows = []
    for pair in sorted(set(four) & set(five)):
        a, b = four[pair], five[pair]
        rows.append({"patient_a": pair[0], "patient_b": pair[1], "four_view_conditional": a, "five_view_conditional": b, "robust_min": min(a, b), "robust_geometric_mean": math.sqrt(max(a, 0) * max(b, 0)), "absolute_difference": abs(a - b)})
    return rows


def plot_outputs(output_root, overlaps, cross_pairs, core4, core5):
    from utils.visualization import configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray([[next(row["jaccard"] for row in overlaps if row["core_4v"] == left and row["core_5v"] == right) for right in core5] for left in core4])
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(core5)), core5); axis.set_yticks(range(len(core4)), core4)
    for i in range(len(core4)):
        for j in range(len(core5)):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < .55 else "black")
    axis.set_xlabel("5-view stable core"); axis.set_ylabel("4-view stable core"); axis.set_title("4-view versus 5-view core overlap (Jaccard)")
    figure.colorbar(image, ax=axis, label="Jaccard"); figure.tight_layout(); figure.savefig(figures / "core_overlap_heatmap.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6)); left_y, right_y = {}, {}
    y = 0
    for core in core4: left_y[core] = y; y += len(core4[core])
    y = 0
    for core in core5: right_y[core] = y; y += len(core5[core])
    for row in sorted(overlaps, key=lambda item: (item["core_4v"], item["core_5v"])):
        width = row["intersection_n"]
        if not width: continue
        y0 = left_y[row["core_4v"]]; y1 = right_y[row["core_5v"]]
        axis.fill_between([0, 1], [y0, y1], [y0 + width, y1 + width], alpha=.22)
    for x, locations, cores, color in ((0, left_y, core4, "#457b9d"), (1, right_y, core5, "#e76f51")):
        for core, start in locations.items():
            axis.bar(x, len(cores[core]), bottom=start, width=.08, color=color)
            axis.text(x + (.06 if x == 0 else -.06), start + len(cores[core]) / 2, core, ha="left" if x == 0 else "right", va="center")
    axis.set_xlim(-.25, 1.25); axis.set_xticks([0, 1], ["4-view", "5-view"]); axis.set_ylabel("Patient count"); axis.set_title("Patient overlap flow: 4-view to 5-view")
    figure.tight_layout(); figure.savefig(figures / "core_alluvial_4v_to_5v.png", dpi=180); plt.close(figure)

    patients = sorted({row["patient_a"] for row in cross_pairs} | {row["patient_b"] for row in cross_pairs})
    lookup = {(row["patient_a"], row["patient_b"]): row["robust_min"] for row in cross_pairs}
    robust = np.full((len(patients), len(patients)), np.nan)
    np.fill_diagonal(robust, 1)
    for i, a in enumerate(patients):
        for j, b in enumerate(patients):
            if i < j and (a, b) in lookup: robust[i, j] = robust[j, i] = lookup[a, b]
    figure, axis = plt.subplots(figsize=(9, 8)); image = axis.imshow(robust, vmin=0, vmax=1, cmap="viridis"); axis.set_title("Cross-representation patient-pair stability (min of 4V/5V)"); axis.set_xlabel("Patients"); axis.set_ylabel("Patients"); figure.colorbar(image, ax=axis, label="min conditional co-assignment"); figure.tight_layout(); figure.savefig(figures / "patient_pair_robustness_heatmap.png", dpi=180); plt.close(figure)


def run(four_view_root, five_view_root, output_root, jaccard_threshold=.5, robust_threshold=.75, force=False):
    if output_root.exists() and any(output_root.iterdir()) and not force: raise FileExistsError(f"Output exists; pass --force: {output_root}")
    if force and output_root.exists(): shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    core4 = read_membership(four_view_root / "stable_core_membership.csv"); core5 = read_membership(five_view_root / "stable_core_membership.csv")
    patient_order = json.loads((ROOT / "output_kirc/candidate_subtype/affinity_patient_order.json").read_text(encoding="utf-8"))
    overlaps = [overlap_row(a, core4[a], b, core5[b], len(patient_order)) for a in core4 for b in core5]
    write_csv(output_root / "core_overlap_4v_vs_5v.csv", overlaps)
    pair4_ids, pair4_matrix = read_matrix(four_view_root / "conditional_membership_coassignment_matrix.csv")
    pair5_ids, pair5_matrix = read_matrix(five_view_root / "conditional_membership_coassignment_matrix.csv")
    write_csv(output_root / "patient_pair_coassignment_4v.csv", pair_rows(pair4_ids, pair4_matrix, "4-view")); write_csv(output_root / "patient_pair_coassignment_5v.csv", pair_rows(pair5_ids, pair5_matrix, "5-view"))
    cross_pairs = cross_pair_rows(pair4_ids, pair4_matrix, pair5_ids, pair5_matrix); write_csv(output_root / "patient_pair_cross_representation_stability.csv", cross_pairs)
    candidates = []
    for row in overlaps:
        shared = set(core4[row["core_4v"]]) & set(core5[row["core_5v"]]); pair_values_for_core = [item["robust_min"] for item in cross_pairs if item["patient_a"] in shared and item["patient_b"] in shared]
        row = dict(row, shared_patient_ids=json.dumps(sorted(shared), ensure_ascii=False), shared_pair_count=len(pair_values_for_core), shared_pair_robust_mean=float(np.mean(pair_values_for_core)) if pair_values_for_core else None, representation_robust_candidate=int(row["jaccard"] >= jaccard_threshold and (np.mean(pair_values_for_core) if pair_values_for_core else 0) >= robust_threshold))
        candidates.append(row)
    write_csv(output_root / "representation_robust_core_candidates.csv", candidates)
    plot_outputs(output_root, overlaps, cross_pairs, core4, core5)
    summary = {"experiment": "cross_representation_stable_core_robustness", "four_view_core_count": len(core4), "five_view_core_count": len(core5), "four_view_patient_count": len(set().union(*map(set, core4.values()))), "five_view_patient_count": len(set().union(*map(set, core5.values()))), "overlap_pair_count": len(overlaps), "patient_pair_cross_representation_count": len(cross_pairs), "robust_candidate_count": sum(row["representation_robust_candidate"] for row in candidates), "thresholds": {"jaccard": jaccard_threshold, "cross_pair_robust_mean": robust_threshold}, "interpretation": "Descriptive cross-representation overlap and pair stability; no new subtype labels are assigned."}
    write_json(output_root / "cross_representation_robustness_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-view-root", type=Path, default=ROOT / "output_kirc_v12/03_multi_k_accepted_core_stability_v11")
    parser.add_argument("--five-view-root", type=Path, default=ROOT / "output_kirc_v12/15_five_view_multi_k_stability")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v12/17_cross_representation_stable_core_robustness")
    parser.add_argument("--jaccard-threshold", type=float, default=.5); parser.add_argument("--robust-threshold", type=float, default=.75); parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
