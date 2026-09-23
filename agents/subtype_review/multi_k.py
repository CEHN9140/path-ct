from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np

from utils.cache_utils import hash_payload
from utils.io import write_json
from utils.llm_utils import load_yaml_file


def collect_accept_observations(run_files: list[tuple[int, int, Path]], patient_ids: set[str]) -> list[dict[str, Any]]:
    observations = []
    for initial_k, repeat, path in run_files:
        assigned: set[str] = set()
        for item in json.loads(path.read_text(encoding="utf-8")):
            members = [str(member) for member in item["member_ids"]]
            member_set = set(members)
            if len(member_set) != len(members):
                raise ValueError(f"Accepted set contains duplicate patients: {path}")
            if not member_set.issubset(patient_ids):
                raise ValueError(f"Accepted set contains patients outside candidate cohort: {path}")
            if assigned.intersection(member_set):
                raise ValueError(f"Accepted sets overlap within run: {path}")
            assigned.update(member_set)
            observations.append({
                "observation_id": f"K{initial_k}_repeat{repeat}_{item['set_id']}",
                "initial_k": initial_k, "repeat": repeat, "set_id": str(item["set_id"]),
                "member_ids": member_set, "patient_count": len(member_set),
            })
    return observations


def inspect_agent_run(run_root: Path) -> tuple[bool, dict[str, Any]]:
    metadata_path = run_root / "run_metadata.json"
    summary_path = run_root / "final_review_summary.json"
    sets_path = run_root / "final_subtype_sets.json"
    if not metadata_path.is_file():
        return False, {"status": "excluded", "reason": "missing_run_metadata"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, {"status": "excluded", "reason": "invalid_run_metadata_json"}
    if not isinstance(metadata, dict):
        return False, {"status": "excluded", "reason": "invalid_run_metadata_json"}
    if metadata.get("status") != "complete":
        return False, {"status": "excluded", "reason": f"run_status_{metadata.get('status', 'unknown')}"}
    if not summary_path.is_file():
        return False, {"status": "excluded", "reason": "missing_final_review_summary"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, {"status": "excluded", "reason": "invalid_final_review_summary_json"}
    if not isinstance(summary, dict):
        return False, {"status": "excluded", "reason": "invalid_final_review_summary_json"}
    if summary.get("raw_control_status") != "complete" or summary.get("status") != "review_complete":
        return False, {"status": "excluded", "reason": "review_not_complete"}
    if not sets_path.is_file():
        return False, {"status": "excluded", "reason": "missing_final_subtype_sets"}
    try:
        sets = json.loads(sets_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, {"status": "excluded", "reason": "invalid_final_subtype_sets_json"}
    if not isinstance(sets, list):
        return False, {"status": "excluded", "reason": "invalid_final_subtype_sets"}
    return True, {
        "status": "included",
        "input_signature": metadata.get("input_signature"),
        "candidate_signature": metadata.get("candidate_signature"),
        "sets_path": str(sets_path),
    }


def collect_usable_runs(
    output_root: str,
    config: Mapping[str, Any],
) -> tuple[list[tuple[int, int, Path]], list[dict[str, Any]]]:
    runs_root = Path(output_root) / "subtype_review" / "runs"
    params = config["multi_k"]
    usable_runs = []
    audit = []
    for initial_k in sorted(set(map(int, params["initial_ks"]))):
        for repeat in sorted(set(map(int, params["repeats"]))):
            run_root = runs_root / f"K{initial_k}" / f"repeat{repeat}"
            usable, detail = inspect_agent_run(run_root)
            audit.append({"initial_k": initial_k, "repeat": repeat, "run_root": str(run_root), **detail})
            if usable:
                usable_runs.append((initial_k, repeat, run_root / "final_subtype_sets.json"))
    return usable_runs, audit


def enumerate_intersection_candidates(observations: list[dict[str, Any]], min_subtype_size: int) -> set[frozenset[str]]:
    unique_sets = {frozenset(row["member_ids"]) for row in observations}
    candidates = {members for members in unique_sets if len(members) >= min_subtype_size}
    frontier = list(candidates)
    while frontier:
        current = frontier.pop()
        for accepted in unique_sets:
            intersection = frozenset(current.intersection(accepted))
            if len(intersection) >= min_subtype_size and intersection not in candidates:
                candidates.add(intersection)
                frontier.append(intersection)
    return candidates


def build_recurrent_sets(observations: list[dict[str, Any]], min_subtype_size: int, usable_run_count: int | None = None) -> list[dict[str, Any]]:
    records = []
    total = len(observations)
    usable_run_count = usable_run_count or len({(row["initial_k"], row["repeat"]) for row in observations})
    for members in enumerate_intersection_candidates(observations, min_subtype_size):
        supporting = [row for row in observations if members.issubset(row["member_ids"])]
        closure = frozenset.intersection(*(frozenset(row["member_ids"]) for row in supporting)) if supporting else frozenset()
        if closure != members:
            continue
        occurrence_count = len(supporting)
        supporting_ks = sorted({int(row["initial_k"]) for row in supporting})
        supporting_runs = [f"K{row['initial_k']}_repeat{row['repeat']}" for row in supporting]
        records.append({
            "member_ids": sorted(members), "member_count": len(members),
            "occurrence_count": occurrence_count, "total_accept_set_count": total,
            "accept_set_frequency": occurrence_count / total if total else 0.0,
            "supporting_observation_ids": [row["observation_id"] for row in supporting],
            "supporting_ks": supporting_ks, "supporting_k_count": len(supporting_ks),
            "supporting_runs": supporting_runs, "supporting_run_count": len(supporting_runs),
            "usable_run_count": usable_run_count,
            "run_support_frequency": len(supporting_runs) / usable_run_count if usable_run_count else 0.0,
        })
    records.sort(key=lambda row: (-row["accept_set_frequency"], -row["member_count"], tuple(row["member_ids"])))
    for rank, row in enumerate(records, 1):
        row["recurrent_set_id"] = f"RS{rank:03d}"
    return records


def build_run_acceptance_maps(
    run_files: list[tuple[int, int, Path]], patient_ids: list[str],
) -> list[dict[str, Any]]:
    patient_set = set(patient_ids)
    records = []
    for initial_k, repeat, path in run_files:
        assignments = {patient_id: None for patient_id in patient_ids}
        accepted_sets = []
        for item in json.loads(path.read_text(encoding="utf-8")):
            set_id = str(item["set_id"])
            members = [str(member) for member in item["member_ids"]]
            if len(set(members)) != len(members) or not set(members).issubset(patient_set):
                raise ValueError(f"Invalid accepted set membership: {path}")
            if any(assignments[member] is not None for member in members):
                raise ValueError(f"Accepted sets overlap within run: {path}")
            for member in members:
                assignments[member] = set_id
            accepted_sets.append({"set_id": set_id, "member_ids": set(members)})
        records.append({
            "initial_k": initial_k, "repeat": repeat,
            "run_id": f"K{initial_k}_repeat{repeat}",
            "assignments": assignments, "accepted_sets": accepted_sets,
        })
    return records


def build_patient_consensus_matrices(
    run_records: list[dict[str, Any]], patient_ids: list[str],
) -> dict[str, Any]:
    n = len(patient_ids)
    index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    coacceptance = np.zeros((n, n), dtype=float)
    separation = np.zeros((n, n), dtype=float)
    acceptance_frequency = np.zeros(n, dtype=float)
    ks = sorted({record["initial_k"] for record in run_records})
    runs_by_k = {k: [record for record in run_records if record["initial_k"] == k] for k in ks}
    for k, records in runs_by_k.items():
        weight = 1.0 / (len(ks) * len(records))
        for record in records:
            accepted = [set(item["member_ids"]) for item in record["accepted_sets"]]
            for members in accepted:
                for patient_id in members:
                    acceptance_frequency[index[patient_id]] += weight
                for left, right in itertools.combinations(sorted(members), 2):
                    i, j = index[left], index[right]
                    coacceptance[i, j] += weight
                    coacceptance[j, i] += weight
            for left_set, right_set in itertools.combinations(accepted, 2):
                for left in left_set:
                    for right in right_set:
                        i, j = index[left], index[right]
                        separation[i, j] += weight
                        separation[j, i] += weight
    return {
        "coacceptance": coacceptance,
        "separation": separation,
        "acceptance_frequency": acceptance_frequency,
        "run_weights": {
            record["run_id"]: 1.0 / (len(ks) * len(runs_by_k[record["initial_k"]]))
            for record in run_records
        },
    }


def cluster_consensus(matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    if matrix.shape[0] <= n_clusters:
        raise ValueError("family count must be smaller than patient count")
    try:
        return SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", assign_labels="discretize",
            random_state=0, n_init=20,
        ).fit_predict(np.clip((matrix + matrix.T) / 2, 0, 1))
    except ValueError:
        degree = matrix.sum(axis=1)
        inv_sqrt = np.zeros_like(degree)
        positive = degree > 0
        inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
        laplacian = np.eye(matrix.shape[0]) - inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
        _, vectors = np.linalg.eigh(laplacian)
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=0, n_init=20).fit_predict(
            vectors[:, :n_clusters]
        )


def consensus_family_solutions(
    run_records: list[dict[str, Any]], patient_ids: list[str],
    min_subtype_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    full = build_patient_consensus_matrices(run_records, patient_ids)
    matrix = full["coacceptance"]
    eigenvalues = np.linalg.eigvalsh(
        np.eye(len(patient_ids))
        - np.diag(np.divide(1.0, np.sqrt(matrix.sum(axis=1)), out=np.zeros(len(patient_ids)), where=matrix.sum(axis=1) > 0))
        @ matrix
        @ np.diag(np.divide(1.0, np.sqrt(matrix.sum(axis=1)), out=np.zeros(len(patient_ids)), where=matrix.sum(axis=1) > 0))
    )
    solutions = []
    for family_count in range(2, min(8, len(patient_ids) - 1) + 1):
        labels = cluster_consensus(matrix, family_count)
        sizes = [int(np.sum(labels == label)) for label in range(family_count)]
        distance = np.clip(1.0 - matrix, 0, 1)
        np.fill_diagonal(distance, 0.0)
        silhouette = float(silhouette_score(distance, labels, metric="precomputed")) if len(set(labels)) > 1 else 0.0
        same = labels[:, None] == labels[None, :]
        upper = np.triu(np.ones_like(matrix, dtype=bool), 1)
        within_values = matrix[same & upper]
        between_values = matrix[(~same) & upper]
        loo_k = []
        for k in sorted({record["initial_k"] for record in run_records}):
            subset = [record for record in run_records if record["initial_k"] != k]
            if not subset:
                continue
            loo_labels = cluster_consensus(build_patient_consensus_matrices(subset, patient_ids)["coacceptance"], family_count)
            loo_k.append(float(adjusted_rand_score(labels, loo_labels)))
        loo_run = []
        for index in range(len(run_records)):
            subset = run_records[:index] + run_records[index + 1:]
            if not subset:
                continue
            loo_labels = cluster_consensus(build_patient_consensus_matrices(subset, patient_ids)["coacceptance"], family_count)
            loo_run.append(float(adjusted_rand_score(labels, loo_labels)))
        eigengap = float(eigenvalues[family_count] - eigenvalues[family_count - 1])
        solutions.append({
            "family_count": family_count, "labels": labels, "family_sizes": sizes,
            "min_family_size": min(sizes), "valid_min_size": min(sizes) >= min_subtype_size,
            "eigengap": eigengap, "silhouette": silhouette,
            "within_consensus": float(np.mean(within_values)) if within_values.size else 0.0,
            "between_leakage": float(np.mean(between_values)) if between_values.size else 0.0,
            "loo_k_ari": float(np.mean(loo_k)) if loo_k else 0.0,
            "loo_run_ari": float(np.mean(loo_run)) if loo_run else 0.0,
            "loo_k_values": loo_k, "loo_run_values": loo_run,
        })
    valid = [solution for solution in solutions if solution["valid_min_size"]]
    selected = max(valid, key=lambda solution: (solution["eigengap"], -solution["family_count"])) if valid else None
    return solutions, {"selected": selected, "matrices": full}


def write_family_outputs(
    output_dir: Path, run_records: list[dict[str, Any]], patient_ids: list[str],
    recurrent_sets: list[dict[str, Any]], min_subtype_size: int,
) -> dict[str, Any]:
    solutions, selected_data = consensus_family_solutions(run_records, patient_ids, min_subtype_size)
    selected = selected_data["selected"]
    if selected is None:
        pd.DataFrame(solutions).drop(columns=["labels"], errors="ignore").to_csv(
            output_dir / "family_count_diagnostics.csv", index=False
        )
        np.save(output_dir / "patient_coacceptance.npy", selected_data["matrices"]["coacceptance"])
        np.save(output_dir / "patient_separation.npy", selected_data["matrices"]["separation"])
        write_json(output_dir / "stable_subtype_sets.json", [])
        write_json(output_dir / "stable_subtype_summary.json", {
            "selected_family_count": None,
            "selection_status": "no_solution_meets_min_subtype_size",
            "usable_run_count": len(run_records),
            "patient_count": len(patient_ids),
        })
        pd.DataFrame(columns=["patient_id", "subtype"]).to_csv(
            output_dir / "stable_subtype_membership.csv", index=False
        )
        pd.DataFrame(columns=["patient_id", "family_id"]).to_csv(
            output_dir / "patient_assignment_confidence.csv", index=False
        )
        pd.DataFrame(columns=["recurrent_set_id", "family_id", "overlap_count", "purity", "coverage"]).to_csv(
            output_dir / "recurrent_set_family_map.csv", index=False
        )
        pd.DataFrame(columns=["family_id", "initial_k", "repeat", "support"]).to_csv(
            output_dir / "family_support_by_run.csv", index=False
        )
        pd.DataFrame(columns=["family_id", "initial_k", "mean", "std", "min", "max", "count"]).to_csv(
            output_dir / "family_support_by_k.csv", index=False
        )
        return {"selected_family_count": None}
    labels = selected["labels"]
    families = []
    for label in sorted(set(labels)):
        family_id = f"Subtype_{chr(65 + label)}"
        members = [patient_id for patient_id, patient_label in zip(patient_ids, labels) if patient_label == label]
        families.append({"family_id": family_id, "member_ids": members, "member_count": len(members)})
    family_by_patient = {patient_id: family["family_id"] for family in families for patient_id in family["member_ids"]}
    coacceptance = selected_data["matrices"]["coacceptance"]
    index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    confidence_rows = []
    for patient_id in patient_ids:
        i = index[patient_id]
        affinities = []
        for family in families:
            peers = [index[member] for member in family["member_ids"] if member != patient_id]
            affinities.append((family["family_id"], float(np.mean(coacceptance[i, peers])) if peers else 0.0))
        affinities.sort(key=lambda item: (-item[1], item[0]))
        confidence_rows.append({
            "patient_id": patient_id, "family_id": family_by_patient[patient_id],
            "family_affinity": affinities[0][1], "second_family": affinities[1][0],
            "second_affinity": affinities[1][1], "assignment_margin": affinities[0][1] - affinities[1][1],
            "acceptance_frequency": selected_data["matrices"]["acceptance_frequency"][i],
        })
    pd.DataFrame(confidence_rows).to_csv(output_dir / "patient_assignment_confidence.csv", index=False)
    pd.DataFrame([
        {"patient_id": row["patient_id"], "subtype": row["family_id"]}
        for row in confidence_rows
    ]).to_csv(output_dir / "stable_subtype_membership.csv", index=False)
    np.save(output_dir / "patient_coacceptance.npy", selected_data["matrices"]["coacceptance"])
    np.save(output_dir / "patient_separation.npy", selected_data["matrices"]["separation"])
    pd.DataFrame([{
        key: value for key, value in solution.items()
        if key not in {"labels", "loo_k_values", "loo_run_values"}
    } | {"loo_k_values": json.dumps(solution["loo_k_values"]), "loo_run_values": json.dumps(solution["loo_run_values"])} for solution in solutions]).to_csv(output_dir / "family_count_diagnostics.csv", index=False)
    map_rows = []
    for recurrent in recurrent_sets:
        scores = []
        members = set(recurrent["member_ids"])
        for family in families:
            overlap = len(members.intersection(family["member_ids"]))
            scores.append((overlap / len(members), overlap / family["member_count"], family["family_id"], overlap))
        purity, coverage, family_id, overlap = max(scores)
        map_rows.append({"recurrent_set_id": recurrent["recurrent_set_id"], "family_id": family_id, "overlap_count": overlap, "purity": purity, "coverage": coverage})
    pd.DataFrame(map_rows).to_csv(output_dir / "recurrent_set_family_map.csv", index=False)
    support_rows = []
    for family in families:
        family_members = set(family["member_ids"])
        for record in run_records:
            best = max((2 * len(family_members.intersection(item["member_ids"])) / (len(family_members) + len(item["member_ids"])) for item in record["accepted_sets"]), default=0.0)
            support_rows.append({"family_id": family["family_id"], "initial_k": record["initial_k"], "repeat": record["repeat"], "support": best})
    support_frame = pd.DataFrame(support_rows)
    support_frame.to_csv(output_dir / "family_support_by_run.csv", index=False)
    by_k = support_frame.groupby(["family_id", "initial_k"])["support"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    by_k.to_csv(output_dir / "family_support_by_k.csv", index=False)
    summary = {
        "selected_family_count": selected["family_count"], "families": families,
        "family_count_selection": {key: value for key, value in selected.items() if key not in {"labels", "loo_k_values", "loo_run_values"}},
        "usable_run_count": len(run_records), "patient_count": len(patient_ids),
    }
    write_json(output_dir / "stable_subtype_sets.json", families)
    write_json(output_dir / "stable_subtype_summary.json", summary)
    return summary


def write_recurrent_outputs(output_dir: Path, observations: list[dict[str, Any]], recurrent_sets: list[dict[str, Any]], *, aggregation_manifest: Mapping[str, Any], aggregation_signature: str, run_count: int, min_subtype_size: int, run_audit: list[dict[str, Any]]) -> dict[str, Any]:
    pd.DataFrame([
        {"observation_id": row["observation_id"], "initial_k": row["initial_k"], "repeat": row["repeat"], "set_id": row["set_id"], "patient_count": row["patient_count"], "member_ids": json.dumps(sorted(row["member_ids"]), ensure_ascii=False)}
        for row in observations
    ], columns=["observation_id", "initial_k", "repeat", "set_id", "patient_count", "member_ids"]).to_csv(output_dir / "accept_set_observations.csv", index=False)
    pd.DataFrame([
        {"rank": rank, "recurrent_set_id": row["recurrent_set_id"], "member_count": row["member_count"], "occurrence_count": row["occurrence_count"], "total_accept_set_count": row["total_accept_set_count"], "accept_set_frequency": row["accept_set_frequency"], "supporting_k_count": row["supporting_k_count"], "supporting_ks": json.dumps(row["supporting_ks"]), "supporting_run_count": row["supporting_run_count"], "usable_run_count": row["usable_run_count"], "run_support_frequency": row["run_support_frequency"], "member_ids": json.dumps(row["member_ids"], ensure_ascii=False)}
        for rank, row in enumerate(recurrent_sets, 1)
    ], columns=["rank", "recurrent_set_id", "member_count", "occurrence_count", "total_accept_set_count", "accept_set_frequency", "supporting_k_count", "supporting_ks", "supporting_run_count", "usable_run_count", "run_support_frequency", "member_ids"]).to_csv(output_dir / "recurrent_set_summary.csv", index=False)
    write_json(output_dir / "recurrent_sets.json", recurrent_sets)
    pd.DataFrame([
        {"recurrent_set_id": row["recurrent_set_id"], "patient_id": patient_id}
        for row in recurrent_sets for patient_id in row["member_ids"]
    ], columns=["recurrent_set_id", "patient_id"]).to_csv(output_dir / "recurrent_set_membership.csv", index=False)
    included = [row for row in run_audit if row["status"] == "included"]
    excluded = [row for row in run_audit if row["status"] == "excluded"]
    usable_runs_by_k = {
        str(initial_k): sum(
            row["status"] == "included" and row["initial_k"] == initial_k
            for row in run_audit
        )
        for initial_k in sorted({row["initial_k"] for row in run_audit})
    }
    summary = {"status": "complete", "analysis_type": "closed_recurrent_accept_set_statistics", "aggregation_signature": aggregation_signature, "run_count": run_count, "included_run_count": len(included), "excluded_run_count": len(excluded), "total_accept_set_count": len(observations), "min_subtype_size": min_subtype_size, "recurrent_set_count": len(recurrent_sets), "automatic_subtype_selection": False}
    write_json(output_dir / "aggregation_manifest.json", {
        **aggregation_manifest,
        "aggregation_signature": aggregation_signature,
        "configured_run_count": len(run_audit),
        "included_run_count": len(included),
        "excluded_run_count": len(excluded),
        "excluded_runs": excluded,
        "usable_runs_by_k": usable_runs_by_k,
        "run_audit": run_audit,
    })
    write_json(output_dir / "aggregation_summary.json", summary)
    return summary


def run_multi_k_aggregation(
    output_root: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_root)
    patient_order_path = root / "candidate_subtype" / "affinity_patient_order.json"
    patient_values = json.loads(patient_order_path.read_text(encoding="utf-8"))
    patient_ids = [str(value) for value in patient_values]
    patient_set = set(patient_ids)
    if len(patient_set) != len(patient_ids):
        raise ValueError("Candidate patient order contains duplicate IDs")
    params = config["multi_k"]
    run_files, run_audit = collect_usable_runs(output_root, config)
    if not run_files:
        raise ValueError("No usable completed Agent runs were found.")
    candidate_signatures = {
        row.get("candidate_signature")
        for row in run_audit
        if row.get("status") == "included" and row.get("candidate_signature")
    }
    if len(candidate_signatures) > 1:
        raise ValueError(
            "Included Agent runs reference multiple candidate signatures: "
            + ", ".join(sorted(map(str, candidate_signatures)))
        )
    configured_initial_ks = sorted(set(map(int, params["initial_ks"])))
    configured_repeats = sorted(set(map(int, params["repeats"])))
    included_runs = [
        {"initial_k": initial_k, "repeat": repeat}
        for initial_k, repeat, _ in run_files
    ]
    included_initial_ks = sorted({row["initial_k"] for row in included_runs})
    included_repeats = sorted({row["repeat"] for row in included_runs})
    aggregation_params = {
        **params,
        "initial_ks": configured_initial_ks,
        "repeats": configured_repeats,
    }
    run_manifest = []
    for initial_k, repeat, sets_path in run_files:
        metadata = json.loads((sets_path.parent / "run_metadata.json").read_text(encoding="utf-8"))
        run_manifest.append({
            "initial_k": initial_k,
            "repeat": repeat,
            "agent_input_signature": metadata.get("input_signature"),
            "candidate_signature": metadata.get("candidate_signature"),
            "final_subtype_sets_path": str(sets_path.resolve()),
            "final_subtype_sets_sha256": hashlib.sha256(sets_path.read_bytes()).hexdigest(),
        })
    input_signatures = sorted({
        str(row["agent_input_signature"])
        for row in run_manifest
        if row.get("agent_input_signature")
    })
    aggregation_manifest = {
        "aggregation_version": 2,
        "agent_input_signatures": input_signatures,
        "candidate_signature": next(iter(candidate_signatures), None),
        "configured_initial_ks": configured_initial_ks,
        "configured_repeats": configured_repeats,
        "included_initial_ks": included_initial_ks,
        "included_repeats": included_repeats,
        "included_runs": included_runs,
        "parameters": aggregation_params,
        "patient_order": {
            "path": str(patient_order_path.resolve()),
            "sha256": hashlib.sha256(patient_order_path.read_bytes()).hexdigest(),
        },
        "agent_runs": run_manifest,
        "aggregation_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    aggregation_signature = hash_payload(aggregation_manifest)
    output_dir = root / "subtype_review" / "multi_k"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = collect_accept_observations(run_files, patient_set)
    recurrent_sets = build_recurrent_sets(
        observations, int(params["min_subtype_size"]), usable_run_count=len(run_files)
    )
    summary = write_recurrent_outputs(
        output_dir, observations, recurrent_sets,
        aggregation_manifest=aggregation_manifest,
        aggregation_signature=aggregation_signature,
        run_count=len(run_files), min_subtype_size=int(params["min_subtype_size"]),
        run_audit=run_audit,
    )
    run_records = build_run_acceptance_maps(run_files, patient_ids)
    family_summary = write_family_outputs(
        output_dir, run_records, patient_ids, recurrent_sets,
        int(params["min_subtype_size"]),
    )
    summary.update({"stable_family_count": family_summary["selected_family_count"]})
    write_json(output_dir / "aggregation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute recurrent ACCEPT-set statistics from completed Agent runs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    config = load_yaml_file(Path(args.config_dir) / "subtype_review.yaml")
    print(json.dumps(
        run_multi_k_aggregation(args.output_root, config),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
