#!/usr/bin/env python3
"""Audit a proposed 10-stable-core to 4-macro-state hierarchy offline."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.io import write_json
from utils.visualization import configure_matplotlib

CORE_TO_STATE = {
    "CORE01": "STATE_A", "CORE07": "STATE_A", "CORE09": "STATE_A",
    "CORE02": "STATE_B", "CORE04": "STATE_B", "CORE10": "STATE_B",
    "CORE03": "STATE_C", "CORE08": "STATE_C",
    "CORE05": "STATE_D", "CORE06": "STATE_D",
}
STATE_ORDER = ("STATE_A", "STATE_B", "STATE_C", "STATE_D")


def load_cores(summary_path):
    summary = pd.read_csv(summary_path)
    cores = {}
    for row in summary.to_dict("records"):
        members = row["member_ids"]
        if isinstance(members, str):
            members = json.loads(members)
        cores[str(row["core_id"])] = sorted(map(str, members))
    if set(cores) != set(CORE_TO_STATE):
        raise ValueError(f"Expected cores {sorted(CORE_TO_STATE)}, got {sorted(cores)}")
    if len(set().union(*map(set, cores.values()))) != sum(map(len, cores.values())):
        raise ValueError("Stable cores overlap")
    return dict(sorted(cores.items()))


def load_square_matrix(path, patient_ids):
    frame = pd.read_csv(path).set_index("patient_id")
    frame = frame.reindex(index=patient_ids, columns=patient_ids)
    if frame.isna().any().any():
        raise ValueError(f"Matrix does not cover the patient order: {path}")
    matrix = frame.to_numpy(float)
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError(f"Matrix is not symmetric: {path}")
    return matrix


def aggregate_core_matrix(matrix, patient_ids, cores):
    positions = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    core_ids = list(cores)
    result = np.zeros((len(core_ids), len(core_ids)), float)
    for i, core_a in enumerate(core_ids):
        left = [positions[patient_id] for patient_id in cores[core_a]]
        for j, core_b in enumerate(core_ids):
            right = [positions[patient_id] for patient_id in cores[core_b]]
            values = [matrix[a, b] for a in left for b in right if i != j or a != b]
            result[i, j] = float(np.mean(values)) if values else 1.0
    return result


def normalize_affinity(matrix):
    matrix = np.asarray(matrix, float)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    lower, upper = float(off_diagonal.min()), float(off_diagonal.max())
    if upper <= lower:
        normalized = np.zeros_like(matrix)
    else:
        normalized = (matrix - lower) / (upper - lower)
    normalized = np.clip(normalized, 0.0, 1.0)
    np.fill_diagonal(normalized, 1.0)
    return normalized


def write_matrix(path, matrix, labels):
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("core_id").to_csv(path)


def hierarchy_rows(matrix, labels, matrix_name):
    distance = np.clip(1.0 - (matrix + matrix.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    rows = []
    candidate = [CORE_TO_STATE[label] for label in labels]
    for method in ("average", "complete"):
        clusters = fcluster(linkage(squareform(distance, checks=False), method=method), 4, criterion="maxclust")
        ari = adjusted_rand_score(candidate, clusters)
        rows.extend({
            "matrix": matrix_name,
            "linkage": method,
            "core_id": label,
            "hierarchical_cluster": int(cluster),
            "candidate_state": CORE_TO_STATE[label],
            "candidate_state_ari": float(ari),
        } for label, cluster in zip(labels, clusters))
    return rows


def state_structure(matrix, labels, matrix_name):
    values = {(a, b): matrix[i, j] for i, a in enumerate(labels) for j, b in enumerate(labels)}
    rows = []
    for state in STATE_ORDER:
        members = [core for core in labels if CORE_TO_STATE[core] == state]
        within = [values[tuple(pair)] for pair in combinations(members, 2)]
        outside = [values[(a, b)] for a in members for b in labels if CORE_TO_STATE[b] != state]
        rows.append({
            "matrix": matrix_name,
            "state": state,
            "core_ids": json.dumps(members),
            "within_core_mean": float(np.mean(within)) if within else None,
            "between_state_mean": float(np.mean(outside)) if outside else None,
            "within_minus_between": (
                float(np.mean(within) - np.mean(outside)) if within and outside else None
            ),
        })
    return rows


def write_heatmap(matrices, labels, output_path):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, (name, matrix) in zip(axes, matrices.items()):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_title(name.replace("_", " ").title())
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_yticks(range(len(labels)), labels)
        for index, state in enumerate(labels):
            if index and CORE_TO_STATE[state] != CORE_TO_STATE[labels[index - 1]]:
                axis.axhline(index - 0.5, color="white", linewidth=1.5)
                axis.axvline(index - 0.5, color="white", linewidth=1.5)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Four-view stable-core to macro-state audit")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run(input_root, output_root, force=False):
    input_root, output_root = Path(input_root), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    review_root = input_root / "agent_review"
    summary_path = review_root / "stable_core_summary.csv"
    order_path = input_root / "inputs" / "four_view_no_cnv" / "candidate_subtype" / "affinity_patient_order.json"
    if not order_path.is_file():
        raise FileNotFoundError(f"Missing 4-view patient order: {order_path}")
    cores = load_cores(summary_path)
    patient_ids = list(map(str, json.loads(order_path.read_text(encoding="utf-8"))))
    input_dir = order_path.parent
    wxs_dir = input_root / "inputs" / "four_view_no_cnv" / "wxs"
    paths = {
        "joint_coassignment": review_root / "joint_accepted_coassignment_matrix.csv",
        "fused": input_dir / "fused_similarity.csv",
    }
    fused_npy = input_dir / "fused_similarity.npy"
    if fused_npy.is_file():
        fused = np.load(fused_npy)
        paths.pop("fused")
    else:
        fused = None
    matrices = {"joint_coassignment": load_square_matrix(paths["joint_coassignment"], patient_ids)}
    if fused is None:
        raise FileNotFoundError(fused_npy)
    if fused.shape != (len(patient_ids), len(patient_ids)):
        raise ValueError("4-view fused matrix shape does not match patient order")
    matrices["fused"] = fused
    for name in ("ct", "wsi", "rna"):
        matrices[name] = np.load(input_dir / f"{name}_affinity.npy")
    matrices["wxs"] = np.load(wxs_dir / "wxs_affinity.npy")
    if any(matrix.shape != (len(patient_ids), len(patient_ids)) for matrix in matrices.values()):
        raise ValueError("A patient-level matrix has the wrong shape")
    matrices = {name: normalize_affinity(matrix) for name, matrix in matrices.items()}

    output_root.mkdir(parents=True, exist_ok=True)
    labels = list(cores)
    core_matrices = {}
    pair_rows = []
    for name, matrix in matrices.items():
        core_matrix = aggregate_core_matrix(matrix, patient_ids, cores)
        core_matrices[name] = core_matrix
        write_matrix(output_root / f"core_{name}_similarity.csv", core_matrix, labels)
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i < j:
                    pair_rows.append({"matrix": name, "core_a": a, "core_b": b, "mean_similarity": float(core_matrix[i, j])})
    pd.DataFrame(pair_rows).to_csv(output_root / "core_pair_similarity.csv", index=False)
    hierarchy = []
    for name in ("joint_coassignment", "fused"):
        hierarchy.extend(hierarchy_rows(core_matrices[name], labels, name))
    pd.DataFrame(hierarchy).to_csv(output_root / "macro_state_hierarchy.csv", index=False)
    structures = []
    for name, matrix in core_matrices.items():
        structures.extend(state_structure(matrix, labels, name))
    pd.DataFrame(structures).to_csv(output_root / "macro_state_structure.csv", index=False)
    pd.DataFrame([{"core_id": core, "macro_state": CORE_TO_STATE[core], "core_size": len(members)} for core, members in cores.items()]).to_csv(
        output_root / "core_to_macro_state.csv", index=False
    )
    write_heatmap({"joint_coassignment": core_matrices["joint_coassignment"], "fused": core_matrices["fused"]}, labels, output_root / "core_to_macro_state_heatmap.png")
    write_json(output_root / "manifest.json", {
        "experiment": "four_view_core_to_macro_state_audit",
        "active_modalities": ["ct", "wsi", "rna", "wxs"],
        "core_count": len(cores),
        "macro_state_count": len(STATE_ORDER),
        "core_to_macro_state": CORE_TO_STATE,
        "source_input_root": str(input_root),
        "affinity_scaling": "each patient-level matrix min-max scaled on off-diagonal values, diagonal reset to 1",
    })
    return {"output_root": str(output_root), "core_count": len(cores), "macro_state_count": len(STATE_ORDER)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "output_kirc_v14/11_four_view_no_cnv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/12_four_view_core_to_macro_state_audit")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.input_root, args.output_root, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
