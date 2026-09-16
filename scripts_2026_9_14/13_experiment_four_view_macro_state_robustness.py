#!/usr/bin/env python3
"""Test the number and robustness of four-view core-level macro-states offline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.io import write_json

LINKAGES = ("average", "complete")
MACRO_KS = tuple(range(2, 7))
INITIAL_KS = tuple(range(2, 9))
REPEATS = (1, 2, 3)


def load_cores(path):
    table = pd.read_csv(path)
    cores = {}
    for row in table.to_dict("records"):
        members = json.loads(row["member_ids"]) if isinstance(row["member_ids"], str) else row["member_ids"]
        cores[str(row["core_id"])] = sorted(map(str, members))
    if len(cores) != 10:
        raise ValueError(f"Expected 10 four-view cores, got {sorted(cores)}")
    return dict(sorted(cores.items()))


def load_core_matrix(path, labels):
    frame = pd.read_csv(path).set_index("core_id").reindex(index=labels, columns=labels)
    if frame.isna().any().any():
        raise ValueError(f"Core matrix does not cover all labels: {path}")
    return frame.to_numpy(float)


def cluster_labels(matrix, k, method):
    distance = np.clip(1.0 - (matrix + matrix.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return fcluster(linkage(squareform(distance, checks=False), method=method), k, criterion="maxclust")


def cluster_summary(matrix, labels, k, method, reference=None):
    assignments = cluster_labels(matrix, k, method)
    distance = np.clip(1.0 - (matrix + matrix.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    sizes = [int(sum(assignments == cluster)) for cluster in sorted(set(assignments))]
    cluster_count = len(sizes)
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if assignments[i] == assignments[j] else between).append(matrix[i, j])
    return {
        "macro_k": k,
        "linkage": method,
        "actual_cluster_count": cluster_count,
        "silhouette": (
            float(silhouette_score(distance, assignments, metric="precomputed"))
            if 1 < cluster_count < len(labels) else None
        ),
        "mean_within_similarity": float(np.mean(within)) if within else None,
        "mean_between_similarity": float(np.mean(between)) if between else None,
        "within_minus_between": float(np.mean(within) - np.mean(between)) if within and between else None,
        "state_sizes_in_core_nodes": json.dumps(sorted(sizes)),
        "assignments": dict(zip(labels, map(int, assignments))),
        "reference_ari": (
            float(adjusted_rand_score(reference, assignments)) if reference is not None else None
        ),
    }


def select_macro_state(rows, assignments, labels):
    candidates = [row for row in rows if row["linkage"] == "average" and row["silhouette"] is not None]
    valid = []
    for row in candidates:
        k = row["macro_k"]
        average = [assignments[(k, "average")][core] for core in labels]
        complete = [assignments[(k, "complete")][core] for core in labels]
        complete_row = next(item for item in rows if item["macro_k"] == k and item["linkage"] == "complete")
        average_sizes = pd.Series(average).value_counts()
        complete_sizes = pd.Series(complete).value_counts()
        if (
            row["actual_cluster_count"] == k
            and complete_row["actual_cluster_count"] == k
            and adjusted_rand_score(average, complete) == 1.0
            and average_sizes.min() > 1
            and complete_sizes.min() > 1
        ):
            valid.append(row)
    return min(valid, key=lambda row: (-row["silhouette"], row["macro_k"])) if valid else None


def coassignment_from_runs(review_root, patient_ids, excluded_k=None, return_audit=False):
    same_count = np.zeros((len(patient_ids), len(patient_ids)), float)
    conditional = np.zeros_like(same_count)
    conditional_denominator = np.zeros_like(same_count)
    positions = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    run_count = 0
    audit = []
    for repeat in REPEATS:
        for initial_k in INITIAL_KS:
            if initial_k == excluded_k:
                continue
            path = review_root / f"run{repeat}" / f"K{initial_k}" / "final_subtype_sets.json"
            sets = json.loads(path.read_text(encoding="utf-8"))
            run_count += 1
            assigned = set()
            for item in sets:
                members = [str(member) for member in item.get("member_ids", []) if str(member) in positions]
                assigned.update(members)
                indexes = [positions[member] for member in members]
                for left in indexes:
                    same_count[left, indexes] += 1.0
            assigned_indexes = [positions[member] for member in assigned]
            conditional_denominator[np.ix_(assigned_indexes, assigned_indexes)] += 1.0
            audit.append({
                "run_id": f"run{repeat}/K{initial_k}",
                "repeat": repeat,
                "initial_k": initial_k,
                "assigned_patient_n": len(assigned),
                "unassigned_patient_n": len(patient_ids) - len(assigned),
            })
    if not run_count:
        raise ValueError("No runs available for coassignment recomputation")
    matrix = same_count / run_count
    np.divide(same_count, conditional_denominator, out=conditional, where=conditional_denominator > 0)
    np.fill_diagonal(matrix, 1.0)
    conditional[np.diag_indices_from(conditional)] = np.where(
        np.diag(conditional_denominator) > 0, 1.0, 0.0
    )
    return (matrix, conditional, audit) if return_audit else matrix


def aggregate_core_matrix(matrix, patient_ids, cores):
    positions = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    labels = list(cores)
    result = np.zeros((len(labels), len(labels)), float)
    for i, left_core in enumerate(labels):
        left = [positions[patient_id] for patient_id in cores[left_core]]
        for j, right_core in enumerate(labels):
            right = [positions[patient_id] for patient_id in cores[right_core]]
            values = [matrix[a, b] for a in left for b in right if i != j or a != b]
            result[i, j] = float(np.mean(values)) if values else 1.0
    return result


def run(input_root, data_root, config_dir, output_root, force=False):
    input_root, data_root, output_root = Path(input_root), Path(data_root), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    review_root = input_root / "agent_review"
    cores = load_cores(review_root / "stable_core_summary.csv")
    labels = list(cores)
    patient_ids = list(map(str, json.loads((input_root / "inputs/four_view_no_cnv/candidate_subtype/affinity_patient_order.json").read_text())))
    full_joint, full_conditional, coverage_audit = coassignment_from_runs(
        review_root, patient_ids, return_audit=True
    )
    coverage_audit = [{"excluded_initial_k": None, **row} for row in coverage_audit]
    full_matrix = aggregate_core_matrix(full_joint, patient_ids, cores)
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(full_conditional, index=patient_ids, columns=patient_ids).to_csv(
        output_root / "conditional_accepted_coassignment_matrix.csv"
    )
    full_rows = []
    full_assignments = {}
    for k in MACRO_KS:
        for method in LINKAGES:
            result = cluster_summary(full_matrix, labels, k, method)
            full_assignments[(k, method)] = result["assignments"]
            full_rows.append({key: value for key, value in result.items() if key != "assignments"})
            pd.DataFrame([{"core_id": core, "cluster": cluster} for core, cluster in result["assignments"].items()]).to_csv(
                output_root / f"full_macro_k{k}_{method}_assignment.csv", index=False
            )
    pd.DataFrame(full_rows).to_csv(output_root / "macro_k_summary.csv", index=False)
    selected = select_macro_state(full_rows, full_assignments, labels)
    if selected is None:
        raise ValueError("No macro-state K satisfies linkage agreement and no-singleton constraints")
    write_json(output_root / "selected_macro_state_model.json", {
        "selection_rule": "maximum average-linkage core-level silhouette subject to average/complete agreement and no singleton macro-state; ties choose smaller K",
        "selected_macro_k": int(selected["macro_k"]),
        "selected_linkage": selected["linkage"],
        "selected_silhouette": selected["silhouette"],
        "assignment_file": f"full_macro_k{int(selected['macro_k'])}_{selected['linkage']}_assignment.csv",
    })

    conditional_matrix = aggregate_core_matrix(full_conditional, patient_ids, cores)
    conditional_rows = []
    for k in MACRO_KS:
        for method in LINKAGES:
            result = cluster_summary(conditional_matrix, labels, k, method)
            conditional_rows.append({
                **{key: value for key, value in result.items() if key != "assignments"},
                "ari_vs_joint_assignment": adjusted_rand_score(
                    [full_assignments[(k, method)][core] for core in labels],
                    list(result["assignments"].values()),
                ),
                "coassignment_type": "conditional_on_joint_assignment",
            })
    pd.DataFrame(conditional_rows).to_csv(
        output_root / "conditional_coassignment_macro_k_summary.csv", index=False
    )

    joint_loko_rows, conditional_loko_rows, selected_loko_rows = [], [], []
    for excluded_k in INITIAL_KS:
        joint, conditional, audit_rows = coassignment_from_runs(
            review_root, patient_ids, excluded_k, return_audit=True
        )
        for item in audit_rows:
            coverage_audit.append({"excluded_initial_k": excluded_k, **item})
        matrix = aggregate_core_matrix(joint, patient_ids, cores)
        conditional_core_matrix = aggregate_core_matrix(conditional, patient_ids, cores)
        joint_assignments, conditional_assignments = {}, {}
        for k in MACRO_KS:
            for method in LINKAGES:
                reference = [full_assignments[(k, method)][core] for core in labels]
                result = cluster_summary(matrix, labels, k, method, reference)
                joint_assignments[(k, method)] = result["assignments"]
                joint_loko_rows.append({
                    "excluded_initial_k": excluded_k,
                    **{key: value for key, value in result.items() if key not in {"assignments", "reference_ari"}},
                    "leave_one_k_ari": result["reference_ari"],
                })
                result = cluster_summary(conditional_core_matrix, labels, k, method, reference)
                conditional_assignments[(k, method)] = result["assignments"]
                conditional_loko_rows.append({
                    "excluded_initial_k": excluded_k,
                    **{key: value for key, value in result.items() if key not in {"assignments", "reference_ari"}},
                    "leave_one_k_ari": result["reference_ari"],
                })
        joint_selected = select_macro_state(
            [row for row in joint_loko_rows if row["excluded_initial_k"] == excluded_k],
            joint_assignments,
            labels,
        )
        conditional_selected = select_macro_state(
            [row for row in conditional_loko_rows if row["excluded_initial_k"] == excluded_k],
            conditional_assignments,
            labels,
        )
        for coassignment_type, selected_row in (("joint", joint_selected), ("conditional", conditional_selected)):
            selected_loko_rows.append({
                "excluded_initial_k": excluded_k,
                "coassignment_type": coassignment_type,
                "selected_macro_k": int(selected_row["macro_k"]) if selected_row else None,
                "silhouette": selected_row["silhouette"] if selected_row else None,
                "linkage_agreement": bool(selected_row) if selected_row else False,
                "min_core_nodes": min(json.loads(selected_row["state_sizes_in_core_nodes"])) if selected_row else None,
            })
    pd.DataFrame(joint_loko_rows).to_csv(output_root / "joint_macro_state_leave_one_resolution_out.csv", index=False)
    pd.DataFrame(conditional_loko_rows).to_csv(output_root / "conditional_macro_state_leave_one_resolution_out.csv", index=False)
    pd.DataFrame(selected_loko_rows).to_csv(output_root / "leave_one_resolution_selected_k.csv", index=False)
    pd.DataFrame(coverage_audit).to_csv(output_root / "coassignment_coverage_audit.csv", index=False)

    write_json(output_root / "manifest.json", {
        "experiment": "four_view_macro_state_robustness",
        "discovery_modalities": ["ct", "wsi", "rna", "wxs"],
        "heldout_characterization": [],
        "macro_ks": list(MACRO_KS),
        "linkages": list(LINKAGES),
        "coassignment_source": "final_subtype_sets.json only; fixed full-data micro-cores",
        "coassignment_denominator": "joint matrix divides by included runs; coverage audit reports accepted-patient union per run",
        "conditional_coassignment_available": True,
        "leave_one_resolution_selection": "same macro-K rule reapplied separately to joint and conditional coassignment matrices",
        "selected_macro_k": int(selected["macro_k"]),
        "selected_assignment_file": f"full_macro_k{int(selected['macro_k'])}_{selected['linkage']}_assignment.csv",
        "input_root": str(input_root),
    })
    return {"output_root": str(output_root), "macro_ks": list(MACRO_KS), "loko_rows": len(joint_loko_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "output_kirc_v14/11_four_view_no_cnv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/13_four_view_macro_state_robustness")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.input_root, args.data_root, args.config_dir, args.output_root, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
