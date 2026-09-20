from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from tools.evidence_features import fuse_affinities
from utils.cache_utils import file_identity, hash_payload, semantic_config
from utils.candidate_clustering_outputs import canonical_partition, fit_kmedoids
from utils.io import write_json
from utils.patient_store import save_patient_states

CANDIDATE_VIEWS = ("ct", "wsi", "rna", "wxs")


def build_patient_resampled_candidates(
    modality_affinities: Mapping[str, np.ndarray],
    patient_ids: list[str],
    config: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    candidate_ks = tuple(int(k) for k in config["candidate_ks"])
    resampling = config["patient_resampling"]
    clustering = config["clustering"]
    n_patients = len(patient_ids)
    sample_size = round(n_patients * float(resampling["fraction"]))
    if set(modality_affinities) != set(CANDIDATE_VIEWS):
        raise ValueError("Candidate generation requires exactly CT, WSI, RNA, and WXS affinities")
    if not candidate_ks or sample_size < max(candidate_ks):
        raise ValueError("Patient subsample is smaller than the largest candidate K")
    for name, matrix in modality_affinities.items():
        if matrix.shape != (n_patients, n_patients):
            raise ValueError(f"{name} affinity shape does not match patient order")

    algorithms = ("hierarchical", "spectral", "kmedoids")
    pair_seen = np.zeros((n_patients, n_patients), dtype=np.int32)
    pair_same = {
        k: {algorithm: np.zeros_like(pair_seen) for algorithm in algorithms}
        for k in candidate_ks
    }
    rng = np.random.default_rng(int(resampling["random_seed"]))

    for repeat_index in range(int(resampling["count"])):
        sampled = np.sort(rng.choice(n_patients, size=sample_size, replace=False))
        sample_pairs = np.ix_(sampled, sampled)
        pair_seen[sample_pairs] += 1
        sample_affinities = {
            name: np.asarray(matrix)[sample_pairs]
            for name, matrix in modality_affinities.items()
        }
        fused = fuse_affinities(sample_affinities, config["snf"])
        distance = 1.0 - np.clip((fused + fused.T) / 2, 0, 1)
        np.fill_diagonal(distance, 0.0)
        np.fill_diagonal(fused, 1.0)

        for k in candidate_ks:
            for algorithm in algorithms:
                options = clustering["algorithms"][algorithm]
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                if algorithm == "hierarchical":
                    linkage = options["linkage_options"][repeat_index % len(options["linkage_options"])]
                    labels = AgglomerativeClustering(
                        n_clusters=k, metric="precomputed", linkage=linkage
                    ).fit_predict(distance)
                elif algorithm == "spectral":
                    assign_labels = options["assign_labels_options"][repeat_index % len(options["assign_labels_options"])]
                    labels = SpectralClustering(
                        n_clusters=k,
                        affinity="precomputed",
                        assign_labels=assign_labels,
                        random_state=seed,
                    ).fit_predict(fused)
                else:
                    init = options["init_options"][repeat_index % len(options["init_options"])]
                    labels = fit_kmedoids(
                        distance, n_clusters=k, init=init, seed=seed
                    )
                labels = np.asarray(canonical_partition(np.asarray(labels, dtype=int)))
                if len(np.unique(labels)) != k:
                    raise ValueError(f"{algorithm} returned the wrong number of clusters for K={k}")
                same = labels[:, None] == labels[None, :]
                pair_same[k][algorithm][sample_pairs] += same

    off_diagonal = ~np.eye(n_patients, dtype=bool)
    if np.any(pair_seen[off_diagonal] == 0):
        raise ValueError("Some patient pairs were never co-sampled; increase patient_resampling.count")

    records = {}
    for k in candidate_ks:
        algorithm_consensus = {
            algorithm: pair_same[k][algorithm] / pair_seen
            for algorithm in algorithms
        }
        consensus = np.mean(list(algorithm_consensus.values()), axis=0)
        consensus = np.clip((consensus + consensus.T) / 2, 0, 1)
        np.fill_diagonal(consensus, 1.0)
        distance = 1.0 - consensus
        np.fill_diagonal(distance, 0.0)
        labels = np.asarray(canonical_partition(AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage=clustering["final_linkage"]
        ).fit_predict(distance)), dtype=int)
        records[k] = {
            "labels": labels,
            "consensus": consensus,
            "algorithm_consensus": algorithm_consensus,
            "pair_seen": pair_seen.copy(),
        }
    return records


def candidate_proposer(
    patient_states: list[dict[str, Any]],
    *,
    output_root: str,
    config_dir: str,
) -> dict[str, Any]:
    from utils.llm_utils import load_candidate_proposer_config

    states = [dict(state) for state in patient_states if state.get("qc") == "success"]
    if not states:
        raise ValueError("No patients passed four-view QC for candidate generation")
    patient_ids = [str(state["case_id"]) for state in states]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Four-view candidate cohort contains duplicate patient IDs")
    evidence = states[0]["omics_evidence"]
    order = json.loads(Path(evidence["modality_affinity_patient_order_path"]).read_text())
    if order != patient_ids:
        raise ValueError("Four-view affinity patient order does not match eligible states")
    paths = evidence["modality_affinity_paths"]
    if set(paths) != set(CANDIDATE_VIEWS):
        raise ValueError("Production affinity cache must contain exactly CT, WSI, RNA, and WXS")
    affinities = {name: np.load(paths[name]) for name in CANDIDATE_VIEWS}
    config = load_candidate_proposer_config(config_dir)
    cache_signature = hash_payload({
        "patient_ids": patient_ids,
        "affinities": {name: file_identity(paths[name]) for name in CANDIDATE_VIEWS},
        "config": semantic_config(config),
        "implementation": {
            name: file_identity(str(Path(__file__).resolve().parent.parent / name))
            for name in (
                "agents/candidate_proposer.py",
                "tools/evidence_features.py",
                "utils/candidate_clustering_outputs.py",
            )
        },
    })
    root = Path(output_root) / "candidate_subtype"
    consensus_dir = root / "consensus_cluster"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = consensus_dir / "resampling_manifest.json"
    candidate_paths = {
        k: consensus_dir / f"consensus_hierarchical_K{k}.json"
        for k in config["candidate_ks"]
    }
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    matrix_paths = [
        consensus_dir / f"{algorithm}_consensus_matrix_K{k}.npy"
        for k in config["candidate_ks"]
        for algorithm in ("hierarchical", "spectral", "kmedoids")
    ]
    cache_valid = (
        manifest.get("cache_signature") == cache_signature
        and all(path.is_file() for path in candidate_paths.values())
        and all((consensus_dir / f"consensus_matrix_K{k}.npy").is_file() for k in config["candidate_ks"])
        and all(path.is_file() for path in matrix_paths)
        and (consensus_dir / "pair_seen_counts.npy").is_file()
        and (root / "fused_similarity.npy").is_file()
    )
    if cache_valid:
        records = {
            int(k): {
                "labels": np.asarray(list(json.loads(path.read_text())["labels"].values()), dtype=int),
                "consensus": np.load(consensus_dir / f"consensus_matrix_K{k}.npy"),
            }
            for k, path in candidate_paths.items()
        }
    else:
        full_fused = fuse_affinities(affinities, config["snf"])
        np.save(root / "fused_similarity.npy", full_fused)
        records = build_patient_resampled_candidates(affinities, patient_ids, config)
        np.save(consensus_dir / "pair_seen_counts.npy", records[int(config["candidate_ks"][0])]["pair_seen"])
        for k, record in records.items():
            np.save(consensus_dir / f"consensus_matrix_K{k}.npy", record["consensus"])
            for algorithm, matrix in record["algorithm_consensus"].items():
                np.save(consensus_dir / f"{algorithm}_consensus_matrix_K{k}.npy", matrix)
            labels = record["labels"]
            sets = []
            for label in sorted(np.unique(labels)):
                member_ids = [patient_ids[index] for index, value in enumerate(labels) if value == label]
                set_id = f"K{k}_C{len(sets) + 1:04d}"
                sets.append({
                    "set_id": set_id,
                    "member_ids": member_ids,
                    "generator": {
                        "initial_k": int(k),
                        "partition_id": f"consensus_hierarchical_K{k}",
                    },
                })
            write_json(candidate_paths[k], {
                "initial_k": int(k),
                "labels": {patient_ids[index]: int(label) for index, label in enumerate(labels)},
                "candidate_sets": sets,
            })
        manifest = {
            "cache_signature": cache_signature,
            "patient_ids": patient_ids,
            "candidate_ks": [int(k) for k in config["candidate_ks"]],
            "patient_resampling": config["patient_resampling"],
            "algorithms": ["hierarchical", "spectral", "kmedoids"],
            "algorithm_consensus_weighting": "equal",
            "final_partition_algorithm": "average_linkage",
        }
        write_json(manifest_path, manifest)

    candidate_partitions = {
        int(k): json.loads(candidate_paths[k].read_text())["candidate_sets"]
        for k in candidate_paths
    }
    cluster_ids_by_patient = {patient_id: [] for patient_id in patient_ids}
    for clusters in candidate_partitions.values():
        for cluster in clusters:
            for patient_id in cluster["member_ids"]:
                cluster_ids_by_patient[patient_id].append(cluster["set_id"])
    updated_states = []
    for patient_state in patient_states:
        state = dict(patient_state)
        if state.get("qc") == "success":
            state["candidate_cluster_ids"] = cluster_ids_by_patient[state["case_id"]]
        updated_states.append(state)
    return {
        "patient_states": updated_states,
        "patient_states_by_id": {
            str(state["case_id"]): state for state in updated_states if state.get("case_id")
        },
        "candidate_partitions": candidate_partitions,
        "candidate_partition_paths": {int(k): str(path) for k, path in candidate_paths.items()},
        "candidate_signature": cache_signature,
        "patient_store_paths": save_patient_states(output_root, updated_states),
        "resampling_manifest_path": str(manifest_path),
        "fused_similarity_path": str(root / "fused_similarity.npy"),
    }
