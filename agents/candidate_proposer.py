from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.cluster_flow import (
    canonical_partition,
    cluster_sizes_from_labels,
    consensus_matrix_from_partitions,
    consensus_silhouette_score,
    save_candidate_clustering_outputs,
    select_pac_stability_k,
)
from utils.cluster_store import save_candidate_clusters
from utils.llm_utils import load_yaml_file
from utils.patient_store import save_patient_states


def snf_fuse(
    feature_matrices: list[np.ndarray],
    *,
    k: int,
    iterations: int,
    mu: float,
    alpha: float,
) -> np.ndarray:
    modality_specs = [
        ("ct", "euclidean", feature_matrices[0] if len(feature_matrices) > 0 else None),
        ("wsi", "cosine", feature_matrices[1] if len(feature_matrices) > 1 else None),
        ("rna", "spearman", feature_matrices[2] if len(feature_matrices) > 2 else None),
        ("wxs", "jaccard", feature_matrices[3] if len(feature_matrices) > 3 else None),
    ]
    matrices = [
        matrix
        for _, _, matrix in modality_specs
        if matrix is not None
        if int(matrix.shape[0]) > 0 and int(matrix.shape[1]) > 0
    ]
    if not matrices:
        return np.zeros((0, 0), dtype=float)

    n_cases = int(matrices[0].shape[0])
    if n_cases == 1:
        return np.ones((1, 1), dtype=float)

    import snf
    from scipy import stats
    from scipy.spatial import distance as scipy_distance
    from snf.compute import affinity_matrix

    k_eff = min(max(int(k), 1), n_cases - 1)
    affinity_networks = []
    for modality, metric, matrix in modality_specs:
        if matrix is None or int(matrix.shape[0]) == 0 or int(matrix.shape[1]) == 0:
            continue
        values = np.asarray(matrix, dtype=float)
        if modality == "rna":
            values = np.apply_along_axis(stats.rankdata, 1, values)
            metric = "correlation"
        distance_matrix = scipy_distance.cdist(values, values, metric=metric)
        off_diagonal = ~np.eye(n_cases, dtype=bool)
        finite_distances = distance_matrix[np.isfinite(distance_matrix) & off_diagonal]
        invalid_distance = (
            float(finite_distances.max()) if finite_distances.size else 1.0
        )
        invalid_distance = invalid_distance if invalid_distance > 0 else 1.0
        distance_matrix = np.nan_to_num(
            distance_matrix,
            nan=invalid_distance,
            posinf=invalid_distance,
            neginf=invalid_distance,
        )
        distance_matrix = np.maximum((distance_matrix + distance_matrix.T) / 2.0, 0.0)
        np.fill_diagonal(distance_matrix, 0.0)
        affinity_networks.append(
            affinity_matrix(distance_matrix, K=k_eff, mu=float(mu))
        )
    fused_network = (
        affinity_networks[0]
        if len(affinity_networks) == 1
        else snf.snf(
            *affinity_networks,
            K=k_eff,
            t=max(int(iterations), 0),
            alpha=float(alpha),
        )
    )
    fused_network = np.nan_to_num(
        np.asarray(fused_network, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    fused_network = np.maximum((fused_network + fused_network.T) / 2.0, 0.0)
    off_diagonal = ~np.eye(n_cases, dtype=bool)
    max_similarity = (
        float(fused_network[off_diagonal].max()) if off_diagonal.any() else 0.0
    )
    if max_similarity > 1.0:
        fused_network[off_diagonal] = fused_network[off_diagonal] / max_similarity
    fused_network = np.clip(fused_network, 0.0, 1.0)
    np.fill_diagonal(fused_network, 1.0)
    return fused_network


def calculate_consensus_cdf(
    consensus_values: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.nan_to_num(np.asarray(consensus_values, dtype=float), nan=0.0)
    threshold_values = (
        np.asarray(thresholds, dtype=float)
        if thresholds is not None
        else np.linspace(0.0, 1.0, 101)
    )
    cdf_values = np.asarray(
        [np.mean(values <= threshold) for threshold in threshold_values],
        dtype=float,
    )
    cdf_area = float(
        np.sum((cdf_values[:-1] + cdf_values[1:]) * 0.5 * np.diff(threshold_values))
    )
    return threshold_values, cdf_values, cdf_area


def calculate_delta_area(
    cdf_area: float,
    *,
    previous_cdf_area: float,
) -> tuple[float, float]:
    delta_area = float(cdf_area) - float(previous_cdf_area)
    relative_delta_area = (
        delta_area / max(abs(float(previous_cdf_area)), 1e-8)
        if previous_cdf_area
        else delta_area
    )
    return float(delta_area), float(relative_delta_area)


def calculate_pac(consensus_values: np.ndarray, *, lower: float, upper: float) -> float:
    values = np.nan_to_num(np.asarray(consensus_values, dtype=float), nan=0.0)
    return float(np.mean((values > float(lower)) & (values < float(upper))))


def build_feature_store_payload(
    patient_states: list[Mapping[str, Any]],
    *,
    config_dir: str = "",
) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser() if config_dir else Path("configs")
    snf_config = (
        load_yaml_file(config_path / "snf.yaml")
    ) or {}
    ct_low_variance_threshold = float(snf_config["ct_low_variance_threshold"])
    ct_high_correlation_threshold = float(snf_config["ct_high_correlation_threshold"])
    eligible_states = [
        dict(item) for item in patient_states if item.get("qc") == "success"
    ]
    patient_ids = [str(item.get("case_id", "")) for item in eligible_states]
    ct_vectors: list[list[float]] = []
    wsi_vectors: list[list[float]] = []
    rna_vectors: list[list[float]] = []
    wxs_vectors: list[list[float]] = []
    ct_names_by_case: list[list[str]] = []

    for patient_state in eligible_states:
        case_id = str(patient_state.get("case_id", "") or "")

        ct_evidence = dict(patient_state.get("ct_evidence", {}) or {})
        ct_values = ct_evidence.get("features", [])
        if isinstance(ct_values, list) and ct_values:
            ct_vectors.append([float(item) for item in ct_values])
            ct_names_by_case.append(
                [f"ct_{index:04d}" for index in range(len(ct_values))]
            )
        else:
            feature_path = Path(str(ct_evidence.get("feature_path", "") or ""))
            try:
                payload = (
                    json.loads(feature_path.read_text(encoding="utf-8"))
                    if feature_path.exists()
                    else {}
                )
                feature_items = list(dict(payload).items())
                ct_vectors.append([float(value) for _, value in feature_items])
                ct_names_by_case.append([str(name) for name, _ in feature_items])
            except Exception:
                ct_vectors.append([])
                ct_names_by_case.append([])

        wsi_evidence = dict(patient_state.get("wsi_evidence", {}) or {})
        wsi_values = wsi_evidence.get("features", [])
        if isinstance(wsi_values, list) and wsi_values:
            wsi_vectors.append([float(item) for item in wsi_values])
        else:
            path = Path(str(wsi_evidence.get("feature_path", "") or ""))
            if path.suffix == ".pt" and path.with_suffix(".npy").exists():
                path = path.with_suffix(".npy")
            try:
                if path.suffix == ".npy" and path.exists():
                    wsi_vectors.append(
                        np.asarray(np.load(path), dtype=float).reshape(-1).tolist()
                    )
                elif path.suffix == ".pt" and path.exists():
                    import torch

                    value = torch.load(path, map_location="cpu")
                    value = (
                        value.detach().cpu().numpy()
                        if hasattr(value, "detach")
                        else value
                    )
                    wsi_vectors.append(
                        np.asarray(value, dtype=float).reshape(-1).tolist()
                    )
                else:
                    wsi_vectors.append([])
            except Exception:
                wsi_vectors.append([])

        omics = dict(patient_state.get("omics_evidence", {}) or {})
        for prefix, vectors in [("rna", rna_vectors), ("wxs", wxs_vectors)]:
            values = omics.get(f"{prefix}_features", [])
            if isinstance(values, list) and values:
                vectors.append([float(item) for item in values])
                continue
            path = Path(str(omics.get(f"{prefix}_feature_path", "") or ""))
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    feature_names = [
                        name
                        for name in list(reader.fieldnames or [])
                        if name != "case_id"
                    ]
                    row = next(
                        (
                            row
                            for row in reader
                            if str(row.get("case_id", "") or "") == case_id
                        ),
                        None,
                    )
                    vectors.append(
                        [float(row.get(name)) for name in feature_names] if row else []
                    )
            except Exception:
                vectors.append([])

    z_ct = (
        np.asarray(ct_vectors, dtype=float)
        if any(ct_vectors)
        else np.zeros((len(ct_vectors), 0), dtype=float)
    )
    ct_feature_names = next(
        (
            list(names)
            for names in ct_names_by_case
            if len(names) == z_ct.shape[1] and z_ct.shape[1] > 0
        ),
        [f"ct_{index:04d}" for index in range(z_ct.shape[1])],
    )
    if z_ct.shape[1] > 0:
        keep_mask = np.var(z_ct, axis=0) > ct_low_variance_threshold
        z_ct = z_ct[:, keep_mask]
        ct_feature_names = [
            name for name, keep in zip(ct_feature_names, keep_mask) if bool(keep)
        ]
    if z_ct.shape[1] > 1 and 0 < ct_high_correlation_threshold < 1:
        corr = np.nan_to_num(np.corrcoef(z_ct, rowvar=False), nan=0.0)
        keep_mask = np.ones(z_ct.shape[1], dtype=bool)
        for index in range(z_ct.shape[1]):
            prior_corr = np.abs(corr[index, :index][keep_mask[:index]])
            if np.any(prior_corr > ct_high_correlation_threshold):
                keep_mask[index] = False
        z_ct = z_ct[:, keep_mask]
        ct_feature_names = [
            name for name, keep in zip(ct_feature_names, keep_mask) if bool(keep)
        ]
    if z_ct.shape[1] > 0:
        means = np.mean(z_ct, axis=0, keepdims=True)
        stds = np.std(z_ct, axis=0, keepdims=True)
        z_ct = np.divide(z_ct - means, stds, out=np.zeros_like(z_ct), where=stds > 0)
    z_wsi = (
        np.asarray(wsi_vectors, dtype=float)
        if any(wsi_vectors)
        else np.zeros((len(wsi_vectors), 0), dtype=float)
    )
    if z_wsi.shape[1] > 0:
        norms = np.linalg.norm(z_wsi, axis=1, keepdims=True)
        z_wsi = np.divide(z_wsi, norms, out=np.zeros_like(z_wsi), where=norms > 0)
    z_rna = (
        np.asarray(rna_vectors, dtype=float)
        if any(rna_vectors)
        else np.zeros((len(rna_vectors), 0), dtype=float)
    )
    if z_rna.shape[1] > 0:
        means = np.mean(z_rna, axis=0, keepdims=True)
        stds = np.std(z_rna, axis=0, keepdims=True)
        z_rna = np.divide(z_rna - means, stds, out=np.zeros_like(z_rna), where=stds > 0)
    z_wxs = (
        np.asarray(wxs_vectors, dtype=float)
        if any(wxs_vectors)
        else np.zeros((len(wxs_vectors), 0), dtype=float)
    )
    snf_neighbor_count = int(snf_config["neighbor_count"])
    snf_iterations = int(snf_config["iterations"])
    snf_mu = float(snf_config["mu"])
    snf_alpha = float(snf_config["alpha"])
    z_snf = snf_fuse(
        [z_ct, z_wsi, z_rna, z_wxs],
        k=snf_neighbor_count,
        iterations=snf_iterations,
        mu=snf_mu,
        alpha=snf_alpha,
    )
    return {
        "eligible_patient_ids": patient_ids,
        "z_ct": z_ct,
        "z_wsi": z_wsi,
        "z_rna": z_rna,
        "z_wxs": z_wxs,
        "z_snf": z_snf,
        "snf_config": {
            "neighbor_count": snf_neighbor_count,
            "iterations": snf_iterations,
            "mu": snf_mu,
            "alpha": snf_alpha,
            "ct_low_variance_threshold": ct_low_variance_threshold,
            "ct_high_correlation_threshold": ct_high_correlation_threshold,
            "wxs_feature_mode": "top_frequency_gene_binary",
            "modality_metrics": {
                "ct": "euclidean",
                "wsi": "cosine",
                "rna": "spearman",
                "wxs": "jaccard",
            },
        },
        "ct_feature_names": ct_feature_names,
    }


def candidate_proposer(
    patient_states: list[dict[str, Any]],
    *,
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    feature_payload = build_feature_store_payload(patient_states, config_dir=config_dir)
    eligible_states = [
        dict(item) for item in patient_states if item.get("qc") == "success"
    ]
    patient_ids = [str(item.get("case_id", "")) for item in eligible_states]
    n_cases = len(patient_ids)
    candidate_clusters: list[dict[str, Any]] = []
    if not eligible_states:
        print(
            "[candidate_cluster_generator] No QC-passed cases; skip candidate clustering.",
            flush=True,
        )
    elif n_cases == 1:
        print(
            "[candidate_cluster_generator] Only one QC-passed case; create one cluster.",
            flush=True,
        )
        candidate_clusters = [
            {
                "cluster_id": "C1",
                "member_ids": patient_ids,
                "source_views": ["snf"],
                "status": "under_review",
                "generator": {
                    "algorithm": "consensus_hierarchical",
                    "n_clusters": 1,
                    "seed": None,
                    "partition_id": "consensus_hierarchical_K1",
                    "cluster_label": 0,
                },
                "consensus": {
                    "partition_count": 0,
                    "secondary_algorithm": "hierarchical",
                },
            }
        ]
    else:
        z_snf = np.nan_to_num(
            np.asarray(feature_payload.get("z_snf", []), dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if z_snf.shape != (n_cases, n_cases):
            print(
                "[candidate_cluster_generator] Missing SNF matrix; skip candidate clustering.",
                flush=True,
            )
        else:
            from sklearn.cluster import AgglomerativeClustering, SpectralClustering

            z_snf = (z_snf + z_snf.T) / 2.0
            z_snf = np.clip(z_snf, 0.0, 1.0)
            np.fill_diagonal(z_snf, 1.0)
            snf_distance = 1.0 - z_snf
            np.fill_diagonal(snf_distance, 0.0)
            config_path = (
                Path(config_dir).expanduser() if config_dir else Path("configs")
            )
            cluster_config = (
                load_yaml_file(config_path / "candidate_clustering.yaml")
            )
            repeat_count = int(cluster_config["repeat_count"])
            max_clusters = int(cluster_config["max_clusters"])
            if repeat_count < 1:
                raise ValueError("candidate_clustering.repeat_count must be >= 1")
            if max_clusters < 2:
                raise ValueError("candidate_clustering.max_clusters must be >= 2")
            max_cluster_value = min(max_clusters, n_cases)
            algorithms_config = dict(cluster_config["algorithms"])
            consensus_linkage = str(cluster_config["consensus_linkage"])
            pac_lower = float(cluster_config["pac_lower"])
            pac_upper = float(cluster_config["pac_upper"])
            selection_method = str(cluster_config["selection_method"])
            pac_gain_threshold = float(cluster_config["pac_gain_threshold"])
            min_cluster_size = int(cluster_config.get("min_cluster_size", 10))
            pac_tie_tolerance = float(cluster_config.get("pac_tie_tolerance", 0.02))
            random_seed = cluster_config.get("random_seed")
            configured_algorithms = [
                (algorithm, dict(algorithms_config[algorithm]))
                for algorithm in ("hierarchical", "spectral", "pam")
                if algorithm in algorithms_config
            ]
            rng = np.random.default_rng(random_seed)
            clustering_jobs = []
            for algorithm, algorithm_config in configured_algorithms:
                for n_clusters in range(2, max_cluster_value + 1):
                    for run_index in range(1, repeat_count + 1):
                        run_config = dict(algorithm_config)
                        if algorithm == "hierarchical":
                            linkage_options = list(run_config.pop("linkage_options"))
                            run_config["linkage"] = str(rng.choice(linkage_options))
                        elif algorithm == "spectral":
                            assign_labels_options = list(
                                run_config.pop("assign_labels_options")
                            )
                            run_config["assign_labels"] = str(
                                rng.choice(assign_labels_options)
                            )
                        elif algorithm == "pam":
                            init_options = list(run_config.pop("init_options"))
                            run_config["init"] = str(rng.choice(init_options))
                        clustering_jobs.append(
                            {
                                "algorithm": algorithm,
                                "run_index": run_index,
                                "n_clusters": n_clusters,
                                "seed": int(rng.integers(0, np.iinfo(np.int32).max)),
                                "algorithm_config": run_config,
                            }
                        )
            n_cluster_values = sorted(
                {int(job["n_clusters"]) for job in clustering_jobs}
            )
            print(
                "[candidate_cluster_generator] "
                f"Start multi-candidate clustering: cases={n_cases}, "
                f"repeat_count={repeat_count}, max_clusters={max_cluster_value}, "
                f"algorithms={[name for name, _ in configured_algorithms]}.",
                flush=True,
            )

            partition_records: list[dict[str, Any]] = []
            for job in clustering_jobs:
                algorithm = str(job["algorithm"])
                run_index = int(job["run_index"])
                n_clusters = int(job["n_clusters"])
                seed = int(job["seed"])
                algorithm_config = dict(job.get("algorithm_config", {}) or {})
                partition_id = f"{algorithm}_run{run_index:03d}_K{n_clusters}_S{seed}"
                base_record = {
                    "algorithm": algorithm,
                    "run_index": run_index,
                    "n_clusters": n_clusters,
                    "seed": seed,
                    "partition_id": partition_id,
                    "algorithm_config": algorithm_config,
                }
                print(
                    "[candidate_cluster_generator] "
                    f"Run {algorithm} clustering: run={run_index}, K={n_clusters}, seed={seed}.",
                    flush=True,
                )
                try:
                    if algorithm == "hierarchical":
                        linkage = str(algorithm_config["linkage"])
                        labels = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            metric="precomputed",
                            linkage=linkage,
                        ).fit_predict(snf_distance)
                    elif algorithm == "spectral":
                        labels = SpectralClustering(
                            n_clusters=n_clusters,
                            affinity="precomputed",
                            assign_labels=str(algorithm_config["assign_labels"]),
                            random_state=seed,
                        ).fit_predict(z_snf)
                    else:
                        from sklearn_extra.cluster import KMedoids

                        labels = KMedoids(
                            n_clusters=n_clusters,
                            metric="precomputed",
                            method=str(algorithm_config["method"]),
                            init=str(algorithm_config["init"]),
                            random_state=seed,
                        ).fit_predict(snf_distance)
                    labels = np.asarray(labels, dtype=int)
                    partition_key = (
                        canonical_partition(labels)
                        if labels.shape == (n_cases,)
                        else tuple()
                    )
                    actual_cluster_count = len(set(partition_key))
                    valid_partition = (
                        labels.shape == (n_cases,)
                        and actual_cluster_count == n_clusters
                    )
                    partition_records.append(
                        {
                            **base_record,
                            "labels": partition_key,
                            "valid": valid_partition,
                            "skip_reason": ""
                            if valid_partition
                            else "label_shape_mismatch"
                            if labels.shape != (n_cases,)
                            else "single_cluster_partition"
                            if actual_cluster_count < 2
                            else "cluster_count_mismatch",
                        }
                    )
                except Exception as exc:
                    partition_records.append(
                        {
                            **base_record,
                            "labels": tuple(),
                            "valid": False,
                            "skip_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

            valid_partition_records = [
                record
                for record in partition_records
                if bool(record.get("valid", True))
            ]
            valid_partitions_by_k: dict[int, list[tuple[int, ...]]] = {
                n_clusters: [] for n_clusters in n_cluster_values
            }
            for record in valid_partition_records:
                n_clusters = int(record.get("n_clusters", 0) or 0)
                valid_partitions_by_k.setdefault(n_clusters, []).append(
                    tuple(int(item) for item in tuple(record.get("labels", ())))
                )
            print(
                "[candidate_cluster_generator] "
                f"Kept {len(valid_partition_records)} valid candidate partitions "
                f"from {len(clustering_jobs)} attempted partitions.",
                flush=True,
            )

            consensus_records: list[dict[str, Any]] = []
            previous_cdf_area = 0.0
            for n_clusters in n_cluster_values:
                k_partitions = valid_partitions_by_k.get(n_clusters, [])
                if not k_partitions:
                    print(
                        "[consensus_clustering] "
                        f"Skip K={n_clusters}: no valid partitions.",
                        flush=True,
                    )
                    continue
                print(
                    "[consensus_clustering] "
                    f"Build consensus matrix for K={n_clusters} "
                    f"from {len(k_partitions)} partitions.",
                    flush=True,
                )
                consensus = consensus_matrix_from_partitions(k_partitions)
                consensus_values = np.nan_to_num(
                    consensus[np.triu_indices(n_cases, k=1)],
                    nan=0.0,
                )
                pac = calculate_pac(
                    consensus_values,
                    lower=pac_lower,
                    upper=pac_upper,
                )
                cdf_thresholds_array, cdf_array, cdf_area = calculate_consensus_cdf(
                    consensus_values
                )
                delta_area, relative_delta_area = calculate_delta_area(
                    cdf_area,
                    previous_cdf_area=previous_cdf_area,
                )
                previous_cdf_area = cdf_area
                consensus_distance = 1.0 - consensus
                np.fill_diagonal(consensus_distance, 0.0)
                labels = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage=consensus_linkage,
                ).fit_predict(consensus_distance)
                partition_key = canonical_partition(labels)
                cluster_sizes = cluster_sizes_from_labels(partition_key)
                silhouette = consensus_silhouette_score(consensus, partition_key)
                print(
                    "[consensus_clustering] "
                    f"K={n_clusters}, CDF area={cdf_area:.6f}, "
                    f"delta area={delta_area:.6f}, PAC={pac:.6f}, "
                    f"sizes={cluster_sizes}.",
                    flush=True,
                )
                consensus_records.append(
                    {
                        "n_clusters": int(n_clusters),
                        "partition_count": len(k_partitions),
                        "consensus": consensus,
                        "cdf_thresholds": cdf_thresholds_array.tolist(),
                        "cdf_values": cdf_array.astype(float).tolist(),
                        "cdf_area": float(cdf_area),
                        "delta_area": float(delta_area),
                        "relative_delta_area": float(relative_delta_area),
                        "pac": float(pac),
                        "cluster_sizes": cluster_sizes,
                        "consensus_silhouette": silhouette,
                        "labels": partition_key,
                    }
                )
            if consensus_records:
                if selection_method not in {
                    "pac_elbow",
                    "delta_area_elbow",
                    "pac_stability",
                }:
                    raise ValueError(
                        "candidate_clustering.selection_method must be pac_elbow, "
                        "delta_area_elbow, or pac_stability"
                    )
                if selection_method == "pac_stability":
                    best_record = select_pac_stability_k(
                        consensus_records,
                        min_cluster_size=min_cluster_size,
                        pac_tie_tolerance=pac_tie_tolerance,
                    )
                    best_selection_metric = "pac_stability"
                    best_pac_gain = 0.0
                    next_n_clusters = None
                    next_pac_gain = None
                elif selection_method == "pac_elbow":
                    best_record = consensus_records[0]
                    best_selection_metric = "pac_elbow_single_k"
                    best_pac_gain = 0.0
                    next_n_clusters = None
                    next_pac_gain = None
                    smallest_gain_record = None
                    for previous, current in zip(
                        consensus_records, consensus_records[1:]
                    ):
                        pac_gain = float(previous.get("pac", 0.0)) - float(
                            current.get("pac", 0.0)
                        )
                        current["pac_gain"] = float(pac_gain)
                        if smallest_gain_record is None or pac_gain < float(
                            smallest_gain_record["next_pac_gain"]
                        ):
                            smallest_gain_record = {
                                "record": previous,
                                "selection_metric": "pac_elbow_smallest_gain",
                                "pac_gain": float(previous.get("pac_gain", 0.0) or 0.0),
                                "next_n_clusters": int(
                                    current.get("n_clusters", 0) or 0
                                ),
                                "next_pac_gain": float(pac_gain),
                            }
                        if pac_gain < pac_gain_threshold:
                            best_record = previous
                            best_selection_metric = "pac_elbow"
                            best_pac_gain = float(previous.get("pac_gain", 0.0) or 0.0)
                            next_n_clusters = int(current.get("n_clusters", 0) or 0)
                            next_pac_gain = float(pac_gain)
                            break
                    else:
                        if smallest_gain_record is not None:
                            best_record = smallest_gain_record["record"]
                            best_selection_metric = str(
                                smallest_gain_record["selection_metric"]
                            )
                            best_pac_gain = float(smallest_gain_record["pac_gain"])
                            next_n_clusters = int(
                                smallest_gain_record["next_n_clusters"]
                            )
                            next_pac_gain = float(smallest_gain_record["next_pac_gain"])
                else:  # delta_area_elbow
                    best_record = consensus_records[0]
                    best_selection_metric = "delta_area_elbow_single_k"
                    best_delta_gain = 0.0
                    next_n_clusters = None
                    next_pac_gain = None
                    best_pac_gain = 0.0
                    for previous, current in zip(
                        consensus_records, consensus_records[1:]
                    ):
                        delta_gain = float(
                            previous.get("delta_area", 0.0) or 0.0
                        ) - float(current.get("delta_area", 0.0) or 0.0)
                        current["delta_area_gain"] = float(delta_gain)
                        if delta_gain > best_delta_gain:
                            best_delta_gain = float(delta_gain)
                            best_record = previous
                            best_selection_metric = "delta_area_elbow"
                            next_n_clusters = int(current.get("n_clusters", 0) or 0)
                            next_pac_gain = float(current.get("pac_gain", 0.0) or 0.0)
                            best_pac_gain = float(previous.get("pac_gain", 0.0) or 0.0)
                best_record = {
                    **best_record,
                    "selection_metric": best_selection_metric,
                    "pac_lower": pac_lower,
                    "pac_upper": pac_upper,
                    "pac_gain_threshold": pac_gain_threshold,
                    "pac_gain": float(best_pac_gain),
                    "next_n_clusters": next_n_clusters,
                    "next_pac_gain": next_pac_gain,
                }
                best_n_clusters = int(best_record.get("n_clusters", 0) or 0)
                best_delta_area = float(best_record.get("delta_area", 0.0) or 0.0)
                best_cdf_area = float(best_record.get("cdf_area", 0.0) or 0.0)
                best_pac = float(best_record.get("pac", 0.0))
                selection_metric = str(best_record["selection_metric"])
                selected_k_reason = str(best_record.get("selected_k_reason", "") or "")
                best_partition = tuple(
                    int(item) for item in tuple(best_record.get("labels", ()))
                )
                print(
                    "[consensus_clustering] "
                    f"Selected best K by {selection_metric}: K={best_n_clusters}, "
                    f"PAC={best_pac:.6f}, delta area={best_delta_area:.6f}, "
                    f"CDF area={best_cdf_area:.6f}.",
                    flush=True,
                )
                source_algorithms = sorted(
                    {str(record["algorithm"]) for record in valid_partition_records}
                )
                consensus_paths = save_candidate_clustering_outputs(
                    output_root,
                    patient_ids,
                    partition_records,
                    consensus_records,
                    best_record,
                    snf_matrix=z_snf,
                )
                print(
                    "[consensus_clustering] "
                    f"Consensus matrix saved to {consensus_paths.get('matrix_path', '')}; "
                    f"metadata saved to {consensus_paths.get('metadata_path', '')}.",
                    flush=True,
                )
                for label in sorted(set(best_partition)):
                    members = [
                        patient_ids[index]
                        for index, value in enumerate(best_partition)
                        if value == label
                    ]
                    if members:
                        candidate_clusters.append(
                            {
                                "cluster_id": f"C{len(candidate_clusters) + 1:04d}",
                                "member_ids": members,
                                "source_views": ["snf"],
                                "status": "under_review",
                                "generator": {
                                    "algorithm": "consensus_hierarchical",
                                    "n_clusters": int(best_n_clusters),
                                    "seed": None,
                                    "partition_id": f"consensus_hierarchical_K{best_n_clusters}",
                                    "cluster_label": int(label),
                                    "selection_metric": selection_metric,
                                },
                                "consensus": {
                                    "partition_count": int(
                                        best_record["partition_count"]
                                    ),
                                    "attempted_partition_count": len(clustering_jobs),
                                    "saved_partition_count": len(partition_records),
                                    "source_algorithms": source_algorithms,
                                    "secondary_algorithm": "hierarchical",
                                    "secondary_n_clusters": int(best_n_clusters),
                                    "best_n_clusters": int(best_n_clusters),
                                    "selection_metric": selection_metric,
                                    "pac": float(best_pac),
                                    "pac_lower": float(pac_lower),
                                    "pac_upper": float(pac_upper),
                                    "pac_gain_threshold": pac_gain_threshold,
                                    "pac_gain": best_record.get("pac_gain"),
                                    "next_n_clusters": best_record.get(
                                        "next_n_clusters"
                                    ),
                                    "next_pac_gain": best_record.get("next_pac_gain"),
                                    "cdf_area": float(best_cdf_area),
                                    "delta_area": float(best_delta_area),
                                    "cluster_sizes": list(
                                        best_record.get("cluster_sizes", []) or []
                                    ),
                                    "consensus_silhouette": best_record.get(
                                        "consensus_silhouette"
                                    ),
                                    "selected_k_reason": selected_k_reason,
                                    **consensus_paths,
                                },
                            }
                        )
                print(
                    "[consensus_clustering] "
                    f"Generated {len(candidate_clusters)} candidate clusters; "
                    f"sizes={[len(cluster['member_ids']) for cluster in candidate_clusters]}.",
                    flush=True,
                )

    cluster_ids_by_case = {
        str(item.get("case_id", "") or ""): [] for item in patient_states
    }
    for cluster in candidate_clusters:
        cluster_id = str(cluster.get("cluster_id", "") or "")
        for member_id in list(cluster.get("member_ids", []) or []):
            member_key = str(member_id)
            if member_key in cluster_ids_by_case and cluster_id:
                cluster_ids_by_case[member_key].append(cluster_id)
    patient_states = [
        {
            **dict(patient_state),
            "candidate_cluster_ids": cluster_ids_by_case.get(
                str(dict(patient_state).get("case_id", "") or ""), []
            ),
        }
        for patient_state in patient_states
    ]
    return {
        "patient_states": patient_states,
        "patient_states_by_id": {
            str(patient_state["case_id"]): dict(patient_state)
            for patient_state in patient_states
        },
        "candidate_clusters": candidate_clusters,
        "patient_store_paths": save_patient_states(output_root, patient_states),
        "candidate_clusters_path": save_candidate_clusters(
            output_root, candidate_clusters
        ),
    }


def cohort_candidate_proposer(
    state: Mapping[str, Any], *, output_root: str, config_dir: str
) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    inventory.update(
        candidate_proposer(
            [dict(item) for item in list(inventory.get("patient_states", []) or [])],
            output_root=output_root,
            config_dir=config_dir,
        )
    )
    return {"inventory": inventory}
