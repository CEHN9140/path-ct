from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from utils.io import write_json


def run_multi_k_aggregation(
    output_root: str,
    config: Mapping[str, Any],
    input_signature: str,
) -> dict[str, Any]:
    root = Path(output_root)
    runs_root = root / "subtype_review" / "runs"
    patient_ids = [
        str(value)
        for value in json.loads(
            (root / "candidate_subtype" / "affinity_patient_order.json").read_text()
        )
    ]
    patient_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    if len(patient_index) != len(patient_ids):
        raise ValueError("Candidate patient order contains duplicate IDs")

    params = config["multi_k"]
    run_files = []
    for k in params["initial_ks"]:
        for repeat in params["repeats"]:
            run_root = runs_root / f"K{k}" / f"repeat{repeat}"
            metadata_path = run_root / "run_metadata.json"
            summary_path = run_root / "final_review_summary.json"
            sets_path = run_root / "final_subtype_sets.json"
            if not metadata_path.is_file() or not summary_path.is_file() or not sets_path.is_file():
                raise FileNotFoundError(f"Incomplete multi-K review run: {run_root}")
            metadata = json.loads(metadata_path.read_text())
            summary = json.loads(summary_path.read_text())
            if (
                metadata.get("input_signature") != input_signature
                or metadata.get("status") != "complete"
                or summary.get("raw_control_status") != "complete"
                or summary.get("status") != "review_complete"
            ):
                raise ValueError(f"Review run is incomplete or has mismatched inputs: {run_root}")
            run_files.append(sets_path)

    run_count = len(run_files)
    coassignment = np.zeros((len(patient_ids), len(patient_ids)), dtype=float)
    acceptance = np.zeros(len(patient_ids), dtype=float)
    for path in run_files:
        sets = json.loads(path.read_text())
        assigned = set()
        for item in sets:
            members = [str(member) for member in item["member_ids"]]
            if not set(members).issubset(patient_index):
                raise ValueError(f"Accepted set contains patients outside candidate cohort: {path}")
            if assigned.intersection(members):
                raise ValueError(f"Accepted sets overlap within run: {path}")
            assigned.update(members)
            indices = [patient_index[member] for member in members]
            acceptance[indices] += 1
            coassignment[np.ix_(indices, indices)] += 1
    acceptance /= run_count
    coassignment /= run_count
    np.fill_diagonal(coassignment, acceptance)

    accepted_indices = np.flatnonzero(acceptance >= float(params["acceptance_threshold"]))
    if not len(accepted_indices):
        raise ValueError("No patients meet the configured accepted-run frequency threshold")
    patient_similarity = coassignment[np.ix_(accepted_indices, accepted_indices)]
    patient_distance = 1.0 - patient_similarity
    np.fill_diagonal(patient_distance, 0.0)
    patient_linkage = linkage(squareform(patient_distance, checks=False), method="complete")
    labels = fcluster(
        patient_linkage,
        t=1.0 - float(params["coassignment_threshold"]) + 1e-12,
        criterion="distance",
    )
    groups = {}
    for index, label in zip(accepted_indices, labels):
        groups.setdefault(int(label), []).append(patient_ids[int(index)])
    cores = sorted(
        (sorted(members, key=patient_index.get) for members in groups.values()
         if len(members) >= int(params["min_core_size"])),
        key=lambda members: patient_index[members[0]],
    )
    if not cores:
        raise ValueError("No recurrent core meets the configured minimum core size")
    core_ids = [f"CORE{index:02d}" for index in range(1, len(cores) + 1)]
    core_by_patient = {
        patient_id: core_ids[core_index]
        for core_index, members in enumerate(cores)
        for patient_id in members
    }

    core_similarity = np.zeros((len(cores), len(cores)), dtype=float)
    for i, left in enumerate(cores):
        left_indices = [patient_index[patient_id] for patient_id in left]
        for j, right in enumerate(cores):
            right_indices = [patient_index[patient_id] for patient_id in right]
            values = coassignment[np.ix_(left_indices, right_indices)]
            if i == j:
                values = values[~np.eye(len(left), dtype=bool)]
            core_similarity[i, j] = float(np.mean(values)) if values.size else 1.0
    core_similarity = np.clip((core_similarity + core_similarity.T) / 2, 0, 1)
    np.fill_diagonal(core_similarity, 1.0)
    core_distance = 1.0 - core_similarity
    np.fill_diagonal(core_distance, 0.0)

    state_candidates = []
    maximum_k = min(int(params["max_states"]), len(cores) - 1)
    for k in range(2, maximum_k + 1):
        average = fcluster(linkage(squareform(core_distance, checks=False), method="average"), k, criterion="maxclust")
        complete = fcluster(linkage(squareform(core_distance, checks=False), method="complete"), k, criterion="maxclust")
        if len(set(average)) != k or len(set(complete)) != k:
            continue
        if min(np.bincount(average)[1:]) < 2 or adjusted_rand_score(average, complete) != 1.0:
            continue
        score = float(silhouette_score(core_distance, average, metric="precomputed"))
        state_candidates.append({"k": k, "silhouette": score, "labels": average})
    if not state_candidates:
        raise ValueError("No state K meets linkage-agreement and no-singleton constraints")
    selected = min(state_candidates, key=lambda row: (-row["silhouette"], row["k"]))
    cluster_order = sorted(
        set(int(label) for label in selected["labels"]),
        key=lambda label: min(index for index, value in enumerate(selected["labels"]) if int(value) == label),
    )
    state_by_label = {label: f"STATE_{chr(65 + index)}" for index, label in enumerate(cluster_order)}
    core_to_state = {
        core_ids[index]: state_by_label[int(selected["labels"][index])]
        for index in range(len(cores))
    }
    state_membership = [
        {"case_id": patient_id, "core_id": core_by_patient[patient_id], "state_id": core_to_state[core_by_patient[patient_id]]}
        for patient_id in patient_ids
        if patient_id in core_by_patient
    ]

    output_dir = root / "subtype_review" / "multi_k"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coassignment, index=patient_ids, columns=patient_ids).rename_axis("patient_id").to_csv(output_dir / "accepted_coassignment_matrix.csv")
    np.save(output_dir / "accepted_coassignment_matrix.npy", coassignment)
    pd.DataFrame({"patient_id": patient_ids, "acceptance_frequency": acceptance}).to_csv(output_dir / "patient_acceptance_frequency.csv", index=False)
    pd.DataFrame([
        {"core_id": core_id, "patient_count": len(members), "member_ids": json.dumps(members)}
        for core_id, members in zip(core_ids, cores)
    ]).to_csv(output_dir / "recurrent_core_summary.csv", index=False)
    pd.DataFrame([
        {"case_id": patient_id, "core_id": core_by_patient[patient_id], "acceptance_frequency": acceptance[patient_index[patient_id]]}
        for patient_id in patient_ids if patient_id in core_by_patient
    ]).to_csv(output_dir / "recurrent_core_membership.csv", index=False)
    pd.DataFrame(core_similarity, index=core_ids, columns=core_ids).rename_axis("core_id").to_csv(output_dir / "core_similarity.csv")
    pd.DataFrame([{"core_id": core_id, "state_id": core_to_state[core_id]} for core_id in core_ids]).to_csv(output_dir / "core_to_state.csv", index=False)
    pd.DataFrame(state_membership).to_csv(output_dir / "final_state_membership.csv", index=False)
    summary = {
        "status": "complete",
        "run_count": run_count,
        "input_signature": input_signature,
        "patient_count": len(patient_ids),
        "accepted_patient_count": int(np.count_nonzero(acceptance >= float(params["acceptance_threshold"]))),
        "core_count": len(cores),
        "state_count": int(selected["k"]),
        "selected_state_k": int(selected["k"]),
        "selected_state_silhouette": selected["silhouette"],
        "state_merge_rule": "maximum average-linkage silhouette with average/complete agreement and at least two cores per state",
        "parameters": dict(params),
        "state_membership": state_membership,
    }
    write_json(output_dir / "state_merge_summary.json", summary)
    return summary
