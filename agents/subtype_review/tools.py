from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def scoped_groups(scope: str, target_ids: list[str], sets: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    members = {
        str(item["set_id"]): sorted(set(map(str, item["member_ids"])))
        for item in sets
    }
    if scope == "partition":
        return {"partition": sorted({case_id for values in members.values() for case_id in values})}
    if not set(target_ids).issubset(members):
        raise ValueError(f"Evidence targets are not current: {target_ids}")
    if scope == "set":
        return {target: members[target] for target in target_ids}
    pair_id = "|".join(sorted(target_ids))
    return {pair_id: sorted(set(members[target_ids[0]]) | set(members[target_ids[1]]))}


def tool_result(
    name: str,
    metrics: Mapping[str, Any],
    status: str = "success",
    reason: str = "",
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        "tool_name": name,
        "status": status,
        "metrics": dict(metrics),
        "warnings": [],
        "missing_reason": reason,
        "errors": [],
    }
    if artifacts:
        result["artifact_paths"] = dict(artifacts)
    return result


def candidate_consensus_geometry(
    output_root: str,
    all_cluster_states: list[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray]:
    geometries = []
    for item in all_cluster_states:
        geometry = dict((item.get("generator", {}) or {}).get("geometry", {}) or {})
        if geometry:
            geometries.append(geometry)
    if not geometries:
        raise ValueError("Current partition lacks candidate-generation geometry provenance")
    canonical = {json.dumps(geometry, sort_keys=True) for geometry in geometries}
    if len(canonical) != 1:
        raise ValueError("Current partition mixes incompatible candidate-generation geometries")
    geometry = geometries[0]
    if geometry.get("type") != "resampled_consensus_coassignment":
        raise ValueError("Unsupported candidate geometry type")
    root = Path(output_root) / "candidate_subtype"
    patient_ids = [
        str(value) for value in json.loads(
            (root / geometry["patient_order_relative_path"]).read_text()
        )
    ]
    matrix = np.asarray(np.load(root / geometry["matrix_relative_path"]), dtype=float)
    if matrix.shape != (len(patient_ids), len(patient_ids)):
        raise ValueError("Candidate consensus geometry shape mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("Candidate consensus geometry contains non-finite values")
    matrix = np.clip((matrix + matrix.T) / 2, 0, 1)
    np.fill_diagonal(matrix, 0.0)
    return patient_ids, matrix


def distance_kernel(distance: np.ndarray) -> np.ndarray:
    n = len(distance)
    centering = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * centering @ np.square(distance) @ centering


def modality_distance_matrices(output_root: str) -> tuple[list[str], dict[str, np.ndarray]]:
    candidate_root = Path(output_root) / "candidate_subtype"
    patient_ids = [
        str(value)
        for value in json.loads((candidate_root / "affinity_patient_order.json").read_text())
    ]
    shape = (len(patient_ids), len(patient_ids))
    matrices = {}
    for name in ("ct", "wsi", "rna", "wxs"):
        matrix = np.asarray(np.load(candidate_root / f"{name}_distance.npy"), dtype=float)
        if (
            matrix.shape != shape
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, atol=1e-8)
            or np.min(matrix) < 0
            or not np.allclose(np.diag(matrix), 0.0, atol=1e-8)
        ):
            raise ValueError(
                f"{name} native distance does not match patient order or is invalid"
            )
        matrices[name] = matrix
    return patient_ids, matrices


def generalized_rv(left_distance: np.ndarray, right_distance: np.ndarray) -> float | None:
    left_kernel = distance_kernel(left_distance)
    right_kernel = distance_kernel(right_distance)
    denominator = np.linalg.norm(left_kernel) * np.linalg.norm(right_kernel)
    return float(np.sum(left_kernel * right_kernel) / denominator) if denominator else None


def current_membership_alignment(
    distance: np.ndarray,
    labels: np.ndarray,
    target_label: str | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import silhouette_samples

    labels = np.asarray(labels, dtype=str)
    upper = np.triu(np.ones(distance.shape, dtype=bool), k=1)
    same = labels[:, None] == labels[None, :]
    within = distance[upper & same]
    between = distance[upper & ~same]
    unique_n = len(np.unique(labels))
    samples = (
        silhouette_samples(distance, labels, metric="precomputed")
        if 1 < unique_n < len(labels) else None
    )
    mask = np.ones(len(labels), dtype=bool) if target_label is None else labels == target_label
    target_within = within
    target_between = between
    competing = None
    if target_label is not None:
        target_positions = np.flatnonzero(mask)
        other_labels = sorted(set(labels[~mask]))
        within_means = []
        nearest_between_means = []
        for i in target_positions:
            same_positions = np.flatnonzero((labels == labels[i]) & (np.arange(len(labels)) != i))
            if len(same_positions):
                within_means.append(float(distance[i, same_positions].mean()))
            if other_labels:
                nearest_between_means.append(min(
                    float(distance[i, labels == label].mean()) for label in other_labels
                ))
        target_within = np.asarray(within_means)
        target_between = np.asarray(nearest_between_means)
        if other_labels:
            mean_to_other = {
                label: float(distance[np.ix_(target_positions, labels == label)].mean())
                for label in other_labels
            }
            competing = min(mean_to_other, key=mean_to_other.get)
        else:
            mean_to_other = {}
    else:
        mean_to_other = {}
    selected_samples = samples[mask] if samples is not None else None
    mean_silhouette = float(selected_samples.mean()) if selected_samples is not None else None
    return {
        "silhouette": mean_silhouette,
        "mean_silhouette": mean_silhouette,
        "median_silhouette": float(np.median(selected_samples)) if selected_samples is not None else None,
        "negative_fraction": float(np.mean(selected_samples < 0)) if selected_samples is not None else None,
        "mean_within_distance": float(target_within.mean()) if len(target_within) else None,
        "mean_between_distance": float(target_between.mean()) if len(target_between) else None,
        "nearest_competing_set": competing,
        "mean_distance_to_each_other_set": mean_to_other,
    }


def representation_concordance(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
    config_dir: str = "",
) -> dict[str, Any]:
    from utils.llm_utils import load_yaml_file

    patient_ids, matrices = modality_distance_matrices(output_root)
    index = {patient_id: position for position, patient_id in enumerate(patient_ids)}
    memberships = {
        str(item["set_id"]): set(map(str, item["member_ids"]))
        for item in all_cluster_states
    }
    partition_members = set().union(*memberships.values())
    missing_members = sorted(partition_members - set(index))
    if missing_members:
        raise ValueError(f"Current memberships are absent from native distance matrices: {missing_members}")
    if scope == "partition":
        case_ids = [case_id for case_id in patient_ids if case_id in partition_members]
        positions = [index[case_id] for case_id in case_ids]
        settings = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["cross_modal"]["concordance"]
        permutations = int(settings["permutations"])
        bootstrap_repeats = int(settings["bootstrap_repeats"])
        rng = np.random.default_rng(int(settings["random_seed"]))
        concordance = {}
        for left, right in combinations(("ct", "wsi", "rna", "wxs"), 2):
            left_distance = matrices[left][np.ix_(positions, positions)]
            right_distance = matrices[right][np.ix_(positions, positions)]
            observed = generalized_rv(left_distance, right_distance)
            if observed is None:
                ci, valid_n, permutation_p = None, 0, None
            else:
                bootstrapped = []
                for _ in range(bootstrap_repeats):
                    sample = rng.integers(0, len(positions), size=len(positions))
                    value = generalized_rv(
                        left_distance[np.ix_(sample, sample)],
                        right_distance[np.ix_(sample, sample)],
                    )
                    if value is not None:
                        bootstrapped.append(value)
                valid_n = len(bootstrapped)
                ci = [float(value) for value in np.quantile(bootstrapped, [0.025, 0.975])] if valid_n >= 2 else None
                left_kernel, right_kernel = distance_kernel(left_distance), distance_kernel(right_distance)
                denominator = np.linalg.norm(left_kernel) * np.linalg.norm(right_kernel)
                exceed = 0
                for _ in range(permutations):
                    perm = rng.permutation(len(positions))
                    permuted = right_kernel[np.ix_(perm, perm)]
                    exceed += float(np.sum(left_kernel * permuted) / denominator) >= observed
                permutation_p = (exceed + 1) / (permutations + 1)
            concordance[f"{left}__{right}"] = {
                "grv": observed, "bootstrap_ci95": ci,
                "bootstrap_valid_n": valid_n, "permutation_p": permutation_p,
                "permutations": permutations,
            }
        return tool_result("representation_concordance", {"partition": {"partition": {
            "geometry_concordance_patient_n": len(case_ids),
            "comparison": "partition_patient_geometry",
            "geometry_concordance": concordance,
        }}})
    elif scope == "set":
        target = target_ids[0]
        case_ids = [case_id for case_id in patient_ids if case_id in partition_members]
        labels = np.asarray([
            next(set_id for set_id, members in memberships.items() if case_id in members)
            for case_id in case_ids
        ])
        comparison = "full_partition_labels"
        report_key = target
    else:
        left, right = target_ids
        selected = memberships[left] | memberships[right]
        case_ids = [case_id for case_id in patient_ids if case_id in selected]
        labels = np.asarray([left if case_id in memberships[left] else right for case_id in case_ids])
        comparison = "candidate_pair"
        report_key = "|".join(sorted(target_ids))
    positions = [index[case_id] for case_id in case_ids]
    alignment = {
        name: current_membership_alignment(
            matrices[name][np.ix_(positions, positions)],
            labels,
            target_ids[0] if scope == "set" else None,
        )
        for name in ("ct", "wsi", "rna", "wxs")
    }
    output = {
        "alignment_patient_n": len(case_ids),
        "comparison": comparison,
        "native_view_membership_alignment": alignment,
    }
    return tool_result("representation_concordance", {scope: {report_key: output}})


def structural_diagnostics(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from utils.llm_utils import load_yaml_file

    patient_ids, consensus = candidate_consensus_geometry(
        output_root, all_cluster_states
    )
    native_patient_ids, native_distances = modality_distance_matrices(output_root)
    if native_patient_ids != patient_ids:
        raise ValueError("Native distance matrices do not match fused affinity patient order")
    index = {patient_id: position for position, patient_id in enumerate(patient_ids)}
    settings = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["cross_modal"]["structural"]
    max_children, min_size = int(settings["max_children"]), int(settings["min_child_size"])

    def eigengap(matrix: np.ndarray, max_k: int) -> dict[str, Any]:
        degree = np.maximum(matrix.sum(axis=1), np.finfo(float).eps)
        laplacian = np.eye(len(matrix)) - matrix / np.sqrt(np.outer(degree, degree))
        values = np.linalg.eigvalsh((laplacian + laplacian.T) / 2)
        gaps = {k: float(values[k] - values[k - 1]) for k in range(1, max_k + 1) if k < len(values)}
        best = max(gaps, key=gaps.get) if gaps else None
        return {
            "candidate_k": int(best) if best is not None else None,
            "candidate_eigengap": gaps.get(best) if best is not None else None,
            "eigengaps": {str(k): value for k, value in gaps.items()},
        }

    def normalized_cut(matrix: np.ndarray, labels: np.ndarray) -> float | None:
        degree = matrix.sum(axis=1)
        total = 0.0
        for label in np.unique(labels):
            inside = labels == label
            volume = float(degree[inside].sum())
            cut = float(matrix[np.ix_(inside, ~inside)].sum())
            if volume <= np.finfo(float).eps:
                return None
            total += cut / volume
        return total

    def native_separation(case_ids: list[str], labels: np.ndarray) -> dict[str, Any]:
        result = {}
        for name, matrix in native_distances.items():
            local = matrix[np.ix_([index[item] for item in case_ids], [index[item] for item in case_ids])]
            result[name] = {
                "mean_silhouette": (
                    float(silhouette_score(local, labels, metric="precomputed"))
                    if 1 < len(np.unique(labels)) < len(labels) else None
                )
            }
        return result

    set_members = {
        str(item["set_id"]): list(map(str, item["member_ids"]))
        for item in all_cluster_states
    }
    if scope == "partition":
        internal = {}
        for set_id, members in set_members.items():
            local = consensus[np.ix_([index[item] for item in members], [index[item] for item in members])]
            upper = local[np.triu_indices(len(local), 1)]
            max_k = min(max_children, len(members) // min_size)
            gaps = eigengap(local, max_k)
            internal[set_id] = {
                "member_n": len(members),
                "mean_within_affinity": float(upper.mean()) if len(upper) else None,
                "screen_candidate_k": gaps["candidate_k"],
                "candidate_eigengap": gaps["candidate_eigengap"],
                "eigengaps": gaps["eigengaps"],
            }
        boundaries = []
        for left, right in combinations(sorted(set_members), 2):
            cross = consensus[np.ix_([index[item] for item in set_members[left]], [index[item] for item in set_members[right]])]
            boundaries.append({"target_ids": [left, right], "mean_between_affinity": float(cross.mean())})
        neighbors = int(settings["nearest_merge_neighbors"])
        selected = {
            tuple(row["target_ids"])
            for set_name in set_members
            for row in sorted(
                (row for row in boundaries if set_name in row["target_ids"]),
                key=lambda row: (-row["mean_between_affinity"], row["target_ids"]),
            )[:neighbors]
        }
        boundaries = [row for row in boundaries if tuple(row["target_ids"]) in selected]
        boundaries.sort(key=lambda row: (-row["mean_between_affinity"], row["target_ids"]))
        return tool_result("structural_diagnostics", {
            "partition": {
                "internal_structure": internal,
                "nearest_pair_targets": [row["target_ids"] for row in boundaries],
                "nearest_pair_affinities": boundaries,
            },
            "geometry_basis": "candidate_generation_consensus",
        })

    members = scoped_groups(scope, target_ids, all_cluster_states)
    output = {}
    for key, case_ids in members.items():
        local = consensus[np.ix_([index[item] for item in case_ids], [index[item] for item in case_ids])]
        max_k = min(max_children, len(case_ids) // min_size)
        spectrum = eigengap(local, max_k)
        if scope == "set":
            solutions = {}
            for k in range(2, max_k + 1):
                labels = SpectralClustering(
                    n_clusters=k, affinity="precomputed", assign_labels="cluster_qr", random_state=0
                ).fit_predict(local)
                sizes = np.bincount(labels, minlength=k)
                if int(sizes.min()) < min_size or len(np.unique(labels)) != k:
                    continue
                local_distance = 1.0 - local
                np.fill_diagonal(local_distance, 0.0)
                solutions[str(k)] = {
                    "child_sizes": sorted(map(int, sizes)),
                    "normalized_cut": normalized_cut(local, labels),
                    "candidate_consensus_separation": current_membership_alignment(
                        local_distance,
                        labels,
                    ),
                    "native_view_separation": native_separation(case_ids, labels),
                }
            output[key] = {
                "member_n": len(case_ids),
                "screen_candidate_k": spectrum["candidate_k"],
                "candidate_eigengap": spectrum["candidate_eigengap"],
                "eigengaps": spectrum["eigengaps"],
                "solutions": solutions,
            }
        else:
            left, right = (set_members[target] for target in target_ids)
            labels = np.asarray([0 if item in left else 1 for item in case_ids])
            normalized_labels = labels
            within = [
                local[a, b] for a, b in combinations(range(len(case_ids)), 2)
                if (case_ids[a] in left) == (case_ids[b] in left)
            ]
            between = [
                local[a, b] for a, b in combinations(range(len(case_ids)), 2)
                if (case_ids[a] in left) != (case_ids[b] in left)
            ]
            within_mean = float(np.mean(within)) if within else None
            between_mean = float(np.mean(between)) if between else None
            current_ncut = normalized_cut(local, normalized_labels)
            independent = SpectralClustering(
                n_clusters=2, affinity="precomputed", assign_labels="cluster_qr", random_state=0
            ).fit_predict(local)
            output[key] = {
                "member_n": len(case_ids),
                "mean_within_affinity": within_mean,
                "mean_between_affinity": between_mean,
                "current_boundary_normalized_cut": current_ncut,
                "left_volume": float(local[labels == 0].sum()),
                "right_volume": float(local[labels == 1].sum()),
                "independent_two_way_spectral_ari": float(adjusted_rand_score(labels, independent)),
                "union_eigengap": spectrum,
            }
    return tool_result("structural_diagnostics", {
        scope: output,
        "geometry_basis": "candidate_generation_consensus",
    })


def rna_pathway_enrichment(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from gseapy import prerank
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from utils.llm_utils import load_yaml_file

    all_members = {
        str(member) for item in all_cluster_states for member in item["member_ids"]
    }
    missing_states = sorted(all_members - set(patient_states_by_id))
    if missing_states:
        raise ValueError(f"RNA biological-support tool is missing patient states: {missing_states}")

    rna_paths = set()
    missing_paths = []
    for case_id in sorted(all_members):
        omics = dict(patient_states_by_id[case_id].get("omics_evidence", {}) or {})
        path = str(omics.get("rna_raw_counts_path", "") or "")
        if path:
            rna_paths.add(path)
        else:
            missing_paths.append(case_id)
    if missing_paths:
        raise ValueError(
            "Candidate-cohort patient states are missing rna_raw_counts_path: "
            f"{missing_paths}"
        )
    if len(rna_paths) != 1:
        raise ValueError(
            "Candidate-cohort patient states must reference exactly one raw RNA count "
            f"feature artifact, found: {sorted(rna_paths)}"
        )

    gene_settings = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["rna"]
    counts_path = Path(next(iter(rna_paths))).resolve()
    gene_sets_path = Path(gene_settings["hallmark_gene_sets_path"]).resolve()
    min_overlap = int(gene_settings["min_pathway_overlap"])
    gsea_permutations = int(gene_settings.get("gsea_permutations", 1000))
    seed = int(gene_settings.get("seed", 20260920))
    cache_root = Path(output_root) / "subtype_review" / "tool_cache" / "rna_pathway_enrichment"
    cache_root.mkdir(parents=True, exist_ok=True)
    members_by_set = {
        set_id: set(next(item["member_ids"] for item in all_cluster_states if item["set_id"] == set_id))
        for set_id in target_ids
    }
    case_ids = sorted(all_members)
    results = {}
    uncached = []
    for set_id, members in members_by_set.items():
        target_ids_in_matrix = [case_id for case_id in case_ids if case_id in members]
        rest_ids = [case_id for case_id in case_ids if case_id not in members]
        cache_payload = {
            "version": int(gene_settings.get("cache_version", 1)),
            "target_ids": target_ids_in_matrix,
            "rest_ids": rest_ids,
            "counts": [str(counts_path), counts_path.stat().st_size, counts_path.stat().st_mtime_ns],
            "gene_sets": [str(gene_sets_path), gene_sets_path.stat().st_size, gene_sets_path.stat().st_mtime_ns],
            "min_pathway_overlap": min_overlap,
            "gsea_permutations": gsea_permutations,
            "seed": seed,
        }
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        cache_path = cache_root / f"{cache_key}.json"
        if cache_path.is_file():
            results[set_id] = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            uncached.append((set_id, members, target_ids_in_matrix, rest_ids, cache_path))
    if not uncached:
        return tool_result("rna_pathway_enrichment", {"set": results})

    frame = pd.read_csv(counts_path).set_index("case_id")
    frame.index = frame.index.map(str)
    missing_counts = sorted(all_members - set(frame.index))
    if missing_counts:
        raise ValueError("Raw RNA count matrix does not cover the full candidate cohort: " f"{missing_counts}")
    counts_array = frame.to_numpy(dtype=float)
    if not np.isfinite(counts_array).all() or np.any(counts_array < 0) or not np.allclose(counts_array, np.rint(counts_array)):
        raise ValueError("Raw RNA count matrix must contain finite nonnegative integers")
    gene_sets = {}
    for line in Path(gene_sets_path).read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) > 2:
            gene_sets[fields[0]] = set(fields[2:]) & set(frame.columns)
    gene_sets = {name: genes for name, genes in gene_sets.items() if len(genes) >= min_overlap}
    for set_id, members, target_ids_in_matrix, rest_ids, cache_path in uncached:
        set_n, rest_n = len(target_ids_in_matrix), len(rest_ids)
        if set_n < 2 or rest_n < 2:
            results[set_id] = {
                "analysis_status": "not_estimable", "set_n": set_n, "rest_n": rest_n,
                "ranking_method": "pydeseq2_wald_statistic",
                "finite_ranked_gene_n": 0,
                "reason": "DESeq2 target-versus-rest contrast requires at least two samples in both groups.",
                "pathways": [],
            }
            cache_path.write_text(json.dumps(results[set_id], ensure_ascii=False), encoding="utf-8")
            continue
        contrast_ids = target_ids_in_matrix + rest_ids
        counts = frame.loc[contrast_ids].astype(np.int64)
        metadata = pd.DataFrame(
            {"condition": ["target"] * set_n + ["rest"] * rest_n}, index=contrast_ids
        )
        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design="~condition",
            refit_cooks=True,
            n_cpus=int(gene_settings.get("n_cpus", 1)),
            quiet=True,
        )
        dds.deseq2()
        differential_stats = DeseqStats(
            dds,
            contrast=["condition", "target", "rest"],
            cooks_filter=False,
            independent_filter=False,
            n_cpus=int(gene_settings.get("n_cpus", 1)),
            quiet=True,
        )
        differential_stats.run_wald_test()
        ranking = pd.DataFrame({
            "gene": frame.columns.astype(str),
            "stat": np.asarray(differential_stats.statistics, dtype=float).reshape(-1),
        }).replace([np.inf, -np.inf], np.nan).dropna()
        ranking = ranking.sort_values(["stat", "gene"], ascending=[False, True])
        finite_ranked_gene_n = len(ranking)
        if finite_ranked_gene_n < 2:
            results[set_id] = {
                "analysis_status": "not_estimable", "set_n": set_n, "rest_n": rest_n,
                "ranking_method": "pydeseq2_wald_statistic",
                "finite_ranked_gene_n": finite_ranked_gene_n,
                "reason": "Fewer than two finite gene statistics were available for preranked enrichment.",
                "pathways": [],
            }
            cache_path.write_text(json.dumps(results[set_id], ensure_ascii=False), encoding="utf-8")
            continue
        ranked_genes = set(ranking["gene"].astype(str))
        eligible_gene_sets = {
            name: sorted(genes & ranked_genes)
            for name, genes in gene_sets.items()
            if len(genes & ranked_genes) >= min_overlap
        }
        if not eligible_gene_sets:
            results[set_id] = {
                "analysis_status": "not_estimable", "set_n": set_n, "rest_n": rest_n,
                "ranking_method": "pydeseq2_wald_statistic",
                "finite_ranked_gene_n": finite_ranked_gene_n,
                "reason": "No configured pathway met the minimum overlap with finite ranked genes.",
                "pathways": [],
            }
            cache_path.write_text(json.dumps(results[set_id], ensure_ascii=False), encoding="utf-8")
            continue
        result = prerank(
            rnk=ranking[["gene", "stat"]],
            gene_sets=eligible_gene_sets,
            min_size=min_overlap,
            max_size=500,
            permutation_num=gsea_permutations,
            seed=seed,
            threads=int(gene_settings.get("gsea_threads", 1)),
            outdir=None,
            verbose=False,
        ).res2d
        rows = []
        for row in result.to_dict("records"):
            rows.append({
                "pathway": str(row["Term"]),
                "nes": float(row["NES"]),
                "fdr_q": float(row["FDR q-val"]),
                "direction": "up" if float(row["NES"]) > 0 else "down",
                "leading_edge_genes": str(row.get("Lead_genes", "")),
            })
        rows.sort(key=lambda item: (item["fdr_q"], -abs(item["nes"])))
        results[set_id] = {
            "analysis_status": "success", "set_n": set_n, "rest_n": rest_n,
            "ranking_method": "pydeseq2_wald_statistic",
            "finite_ranked_gene_n": finite_ranked_gene_n, "pathways": rows[:30],
        }
        cache_path.write_text(json.dumps(results[set_id], ensure_ascii=False), encoding="utf-8")
    return tool_result("rna_pathway_enrichment", {"set": results})


def odds_ratio_with_ci(table: np.ndarray) -> dict[str, Any]:
    a, b, c, d = map(float, np.asarray(table).reshape(-1))
    if a + c == 0:
        return {
            "odds_ratio": None,
            "odds_ratio_ci95": None,
            "zero_cell_correction_applied": False,
            "effect_status": "not_estimable_no_events",
        }
    corrected = bool(np.any(np.asarray(table) == 0))
    if corrected:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = a * d / (b * c)
    standard_error = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    interval = np.exp(np.log(odds_ratio) + np.array([-1.96, 1.96]) * standard_error)
    return {
        "odds_ratio": float(odds_ratio),
        "odds_ratio_ci95": [float(interval[0]), float(interval[1])],
        "zero_cell_correction_applied": corrected,
        "effect_status": "estimated",
    }


def wxs_mutation_enrichment(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests
    from utils.llm_utils import load_yaml_file

    frame = pd.read_csv(Path(output_root) / "wxs" / "wxs_interpretation_features.csv").set_index("case_id")
    frame.index = frame.index.map(str)
    genes = list(frame.columns.astype(str))
    config = load_yaml_file(Path(config_dir) / "wxs.yaml")["biological_support"]
    driver_genes = {str(gene).upper() for gene in config["driver_genes"]}
    report_top_n = int(config["exploratory_report_top_n"])
    missing_drivers = sorted(driver_genes - {gene.upper() for gene in genes})
    if missing_drivers:
        raise ValueError(f"WXS interpretation matrix is missing configured drivers: {missing_drivers}")
    universe = {str(member) for item in all_cluster_states for member in item["member_ids"]}
    missing = sorted(universe - set(frame.index))
    if missing:
        raise ValueError(f"WXS interpretation matrix is missing candidate patients: {missing}")

    rows_by_set = {}
    artifact_paths = {}
    artifact_dir = Path(output_root) / "wxs" / "review_enrichment"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for set_id in target_ids:
        members = set(next(item["member_ids"] for item in all_cluster_states if item["set_id"] == set_id))
        set_ids = sorted(members & universe)
        rest_ids = sorted(universe - members)
        if not set_ids or not rest_ids:
            rows_by_set[set_id] = {
                "analysis_status": "not_estimable", "set_n": len(set_ids), "rest_n": len(rest_ids),
                "reason": "Fisher enrichment requires at least one patient in both set and rest.",
                "tested_gene_n": 0,
                "driver_panel": {"configured_genes": sorted(driver_genes), "results": []},
                "exploratory_top_genes": [],
            }
            continue

        rows = []
        for gene in genes:
            set_positive = int(frame.loc[set_ids, gene].astype(bool).sum())
            rest_positive = int(frame.loc[rest_ids, gene].astype(bool).sum())
            table = np.asarray([
                [set_positive, len(set_ids) - set_positive],
                [rest_positive, len(rest_ids) - rest_positive],
            ])
            rows.append({
                "gene": gene,
                "driver_panel_member": gene.upper() in driver_genes,
                "set_mutated_n": set_positive,
                "set_n": len(set_ids),
                "rest_mutated_n": rest_positive,
                "rest_n": len(rest_ids),
                **odds_ratio_with_ci(table),
                "p_value": float(fisher_exact(table).pvalue),
                "q_global": None,
                "q_driver": None,
            })
        for row, q_value in zip(rows, multipletests([row["p_value"] for row in rows], method="fdr_bh")[1]):
            row["q_global"] = float(q_value)
        driver_rows = [row for row in rows if row["driver_panel_member"]]
        for row, q_value in zip(driver_rows, multipletests([row["p_value"] for row in driver_rows], method="fdr_bh")[1]):
            row["q_driver"] = float(q_value)
        rows.sort(key=lambda row: (row["q_global"], row["p_value"], row["gene"]))
        membership_hash = hashlib.sha256(
            json.dumps({"target": set_ids, "rest": rest_ids}, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        safe_set_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", set_id)
        artifact_path = artifact_dir / f"{safe_set_id}_{membership_hash}.csv"
        pd.DataFrame([
            {
                **{key: value for key, value in row.items() if key != "odds_ratio_ci95"},
                "odds_ratio_ci95_lower": row["odds_ratio_ci95"][0] if row["odds_ratio_ci95"] else None,
                "odds_ratio_ci95_upper": row["odds_ratio_ci95"][1] if row["odds_ratio_ci95"] else None,
            }
            for row in rows
        ]).to_csv(artifact_path, index=False)
        artifact_paths[set_id] = str(artifact_path)
        rows_by_set[set_id] = {
            "analysis_status": "success", "set_n": len(set_ids), "rest_n": len(rest_ids),
            "tested_gene_n": len(rows),
            "driver_panel": {"configured_genes": sorted(driver_genes), "results": driver_rows},
            "exploratory_top_genes": [row for row in rows if not row["driver_panel_member"]][:report_top_n],
        }
    return tool_result("wxs_mutation_enrichment", {"set": rows_by_set}, artifacts=artifact_paths)


def clinical_labels(patient_states_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    result = {}
    for case_id, state in patient_states_by_id.items():
        clinical = dict(state.get("inventory", {}).get("Clinical", {}) or {})
        diagnoses = [dict(row) for row in clinical.get("diagnoses", [])]
        primary = [row for row in diagnoses if str(row.get("diagnosis_is_primary_disease", "")).lower() == "true"]
        diagnosis = (primary or diagnoses or [{}])[0]
        stage_text = str(diagnosis.get("ajcc_pathologic_stage", "")).upper()
        stage = next((value for value in ("IV", "III", "II", "I") if f"STAGE {value}" in stage_text or stage_text == value), "")
        t_match = re.search(r"T([1-4])", str(diagnosis.get("ajcc_pathologic_t", "")).upper())
        m_text = str(diagnosis.get("ajcc_pathologic_m", "")).upper()
        m_stage = m_text if m_text in {"M0", "M1"} else ""
        grade_match = re.search(r"G([1-4])", str(diagnosis.get("tumor_grade", "")).upper())
        result[str(case_id)] = {
            "stage": stage,
            "grade": f"G{grade_match.group(1)}" if grade_match else "",
            "t_stage": f"T{t_match.group(1)}" if t_match else "",
            "m_stage": m_stage,
        }
    return result


def cramers_v(table: np.ndarray) -> float:
    from scipy.stats import chi2_contingency

    observed = np.asarray(table, dtype=float)
    n = observed.sum()
    rows, cols = observed.shape
    if n <= 1 or min(rows, cols) <= 1:
        return 0.0
    chi2 = float(chi2_contingency(observed, correction=False)[0])
    phi2 = chi2 / n
    correction = (cols - 1) * (rows - 1) / (n - 1)
    phi2 = max(0.0, phi2 - correction)
    rows_adj = rows - (rows - 1) ** 2 / (n - 1)
    cols_adj = cols - (cols - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / min(rows_adj - 1, cols_adj - 1))) if min(rows_adj, cols_adj) > 1 else 0.0


def categorical_test(groups: list[str], values: list[str], permutations: int, seed: int, *, force_fisher: bool = False) -> dict[str, Any]:
    from scipy.stats import chi2_contingency

    table = pd.crosstab(pd.Series(groups, name="group"), pd.Series(values, name="label"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": "not_estimable", "p_value": None, "cramers_v": 0.0, "contingency": table.to_dict()}
    observed = float(chi2_contingency(table.to_numpy(), correction=False)[0])
    expected = chi2_contingency(table.to_numpy(), correction=False)[3]
    if force_fisher:
        if table.shape != (2, 2):
            raise ValueError("Fisher exact test requires a 2x2 table")
        from scipy.stats import fisher_exact

        p_value = float(fisher_exact(table.to_numpy()).pvalue)
        test = "fisher_exact"
    elif np.any(expected < 5):
        rng = np.random.default_rng(seed)
        group_values = np.asarray(groups)
        factor_values = np.asarray(values)
        exceed = 0
        for _ in range(permutations):
            shuffled = rng.permutation(group_values)
            permuted = pd.crosstab(shuffled, factor_values).reindex(index=table.index, columns=table.columns, fill_value=0)
            exceed += float(chi2_contingency(permuted, correction=False)[0]) >= observed
        p_value = (exceed + 1) / (permutations + 1)
        test = "permutation_chi_square"
    else:
        p_value = float(chi2_contingency(table.to_numpy(), correction=False)[1])
        test = "pearson_chi_square"
    return {"test": test, "p_value": p_value, "cramers_v": cramers_v(table.to_numpy()), "contingency": table.to_dict()}


def known_label_echo(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    all_cluster_states: list[Mapping[str, Any]],
    config_dir: str,
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from utils.llm_utils import load_yaml_file

    state_by_patient = {
        str(patient_id): str(item["set_id"])
        for item in all_cluster_states
        for patient_id in item["member_ids"]
    }
    labels = clinical_labels(patient_states_by_id)
    config = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["known_label_echo"]
    clinical_rows = {}
    for name in ("stage", "grade", "t_stage", "m_stage"):
        pairs = [(case_id, state_by_patient[case_id], row[name]) for case_id, row in labels.items() if case_id in state_by_patient and row[name]]
        if name == "m_stage":
            exact = {}
            for state_id in sorted(set(state_by_patient.values())):
                target = [value for case_id, group, value in pairs if group == state_id]
                rest = [value for case_id, group, value in pairs if group != state_id]
                exact[state_id] = categorical_test(
                    [state_id] * len(target) + ["rest"] * len(rest),
                    target + rest,
                    int(config["permutations"]),
                    20260920,
                    force_fisher=True,
                )
            clinical_rows[name] = {
                "available_n": len(pairs),
                "missing_n": len(state_by_patient) - len(pairs),
                "m0_m1_fisher_by_set": exact,
                "contingency": categorical_test([row[1] for row in pairs], [row[2] for row in pairs], int(config["permutations"]), 20260920),
            }
        else:
            result = categorical_test(
                [row[1] for row in pairs], [row[2] for row in pairs],
                int(config["permutations"]), 20260920,
            )
            clinical_rows[name] = {"available_n": len(pairs), "missing_n": len(state_by_patient) - len(pairs), **result}
    taxonomy = {}
    for name, filename in (("tcga_m1_m4", config["mrna_m1_m4_path"]), ("clearcode34", config["clearcode34_path"])):
        frame = pd.read_csv(filename)
        frame = frame[(frame["reference_status"] == "matched") & frame["reference_subtype"].notna()]
        frame = frame[frame["case_id"].astype(str).isin(state_by_patient)].copy()
        candidate = [state_by_patient[str(case_id)] for case_id in frame["case_id"]]
        known = frame["reference_subtype"].astype(str).tolist()
        table = pd.crosstab(pd.Series(candidate, name="state"), pd.Series(known, name="known_label"))
        taxonomy[name] = {
            "reference_n": len(frame),
            "contingency": table.to_dict(),
            "ari": float(adjusted_rand_score(candidate, known)),
            "ami": float(adjusted_mutual_info_score(candidate, known)),
            "limitation": "Expression-derived known taxonomy overlaps the RNA discovery view and is correspondence analysis, not independent validation or an accept/drop gate.",
        }
    return tool_result("known_label_echo", {"partition": {"clinical": clinical_rows, "molecular_taxonomy": taxonomy}})


def technical_values(patient_states_by_id: Mapping[str, Mapping[str, Any]], output_root: str) -> dict[str, dict[str, Any]]:
    values = {}
    root = Path(output_root)
    for case_id, state in patient_states_by_id.items():
        inventory = dict(state.get("inventory", {}) or {})
        ct_records = [dict(row) for row in inventory.get("CT", [])]
        qc_root = root / "ct_qc" / case_id
        summary_path = qc_root / "selection_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        series = dict(summary.get("selected_series", {}) or {})
        selected = dict((summary.get("selected_files") or [{}])[0])
        sidecar_path = qc_root / "dcm2nii" / f"{selected.get('selected_ct_id', '')}.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        source = str(selected.get("selected_source_file", ""))
        record = next((row for row in ct_records if str(row.get("File Path", "")) == source), {})
        case_parts = str(case_id).split("-")
        date = str(record.get("Study Date", ""))
        year_match = re.search(r"(?:19|20)\d{2}", date)
        spacing = series.get("pixel_spacing_row") or record.get("Pixel Spacing Row")
        if not spacing:
            spacing_text = str(record.get("PixelSpacing", record.get("Pixel Spacing", ""))).replace("\\", "/")
            spacing = spacing_text.split("/")[0] if spacing_text else ""
        def number(value: Any) -> float | None:
            try:
                parsed = float(value)
                return parsed if np.isfinite(parsed) else None
            except (TypeError, ValueError):
                return None
        values[str(case_id)] = {
            "tissue_source_site": case_parts[1] if len(case_parts) > 2 and case_parts[0] == "TCGA" else "",
            "ct_phase": str(series.get("inferred_phase", "") or ""),
            "ct_manufacturer": str(sidecar.get("Manufacturer", record.get("Manufacturer", "")) or ""),
            "ct_scanner_model": str(sidecar.get("ManufacturersModelName", sidecar.get("ManufacturerModelName", "")) or ""),
            "ct_reconstruction_kernel": str(sidecar.get("ConvolutionKernel", "") or ""),
            "ct_slice_thickness": number(series.get("slice_thickness_median", record.get("Slice Thickness"))),
            "ct_pixel_spacing": number(spacing),
            "ct_z_spacing": number(series.get("z_spacing_median", record.get("Spacing Between Slices"))),
            "study_year": int(year_match.group()) if year_match else None,
        }
    return values


def confounder_association(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from scipy.stats import kruskal
    from statsmodels.stats.multitest import multipletests
    from utils.llm_utils import load_yaml_file

    scoped = scoped_groups(scope, target_ids, all_cluster_states)
    universe = {str(member) for item in all_cluster_states for member in item["member_ids"]}
    if scope == "set":
        analyses = {
            target: {target: members, "rest": sorted(universe - set(members))}
            for target, members in scoped.items()
        }
    elif scope == "partition":
        analyses = {"partition": {
            str(item["set_id"]): sorted(set(map(str, item["member_ids"])))
            for item in all_cluster_states
        }}
    else:
        analyses = {scope: scoped}
    factors = technical_values(patient_states_by_id, output_root)
    permutations = int(load_yaml_file(Path(config_dir) / "subtype_review.yaml")["confounder"]["permutations"])
    result_by_group = {}
    for group_name, groups in analyses.items():
        group_by_patient = {case_id: key for key, members in groups.items() for case_id in members}
        rows = []
        for factor in ("tissue_source_site", "ct_phase", "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel", "ct_slice_thickness", "ct_pixel_spacing", "ct_z_spacing", "study_year"):
            observations = [(case_id, group_by_patient[case_id], values.get(factor)) for case_id, values in factors.items() if case_id in group_by_patient and values.get(factor) not in (None, "")]
            if not observations:
                continue
            group_labels = [row[1] for row in observations]
            factor_values = [str(row[2]) if isinstance(row[2], str) else float(row[2]) for row in observations]
            if isinstance(factor_values[0], str):
                test = categorical_test(group_labels, factor_values, permutations, 20260920)
                rows.append({"factor": factor, "type": "categorical", "n": len(observations), **test})
            else:
                levels = sorted(set(group_labels))
                samples = [[float(row[2]) for row in observations if row[1] == level] for level in levels]
                if len(samples) < 2:
                    rows.append({"factor": factor, "type": "numeric", "n": len(observations), "test": "not_estimable", "p_value": None, "epsilon_squared": None})
                else:
                    statistic, p_value = kruskal(*samples)
                    n, k = len(observations), len(samples)
                    epsilon = max(0.0, (float(statistic) - k + 1) / (n - k)) if n > k else 0.0
                    rows.append({"factor": factor, "type": "numeric", "n": n, "test": "kruskal_wallis", "p_value": float(p_value), "epsilon_squared": float(epsilon)})
        tested_rows = [row for row in rows if row["p_value"] is not None]
        if tested_rows:
            q_values = multipletests([row["p_value"] for row in tested_rows], method="fdr_bh")[1]
            for row, q_value in zip(tested_rows, q_values):
                row["q_value"] = float(q_value)
        result_by_group[group_name] = rows
    return tool_result("confounder_association", {scope: result_by_group})


def confounder_representation_effect(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from statsmodels.stats.multitest import multipletests
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova, permdisp
    from utils.llm_utils import load_yaml_file

    patient_ids, matrices = modality_distance_matrices(output_root)
    consensus_patient_ids, consensus = candidate_consensus_geometry(
        output_root, all_cluster_states
    )
    if consensus_patient_ids != patient_ids:
        raise ValueError("Candidate consensus patient order does not match native distance matrices")
    consensus_distance = 1.0 - consensus
    np.fill_diagonal(consensus_distance, 0.0)
    index = {case_id: position for position, case_id in enumerate(patient_ids)}
    factors = technical_values(patient_states_by_id, output_root)
    settings = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["confounder"]
    permutations = int(settings["permutations"])
    seed = int(settings["random_seed"])
    categorical_factors = ("tissue_source_site", "ct_phase", "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel")
    continuous_factors = ("ct_slice_thickness", "ct_pixel_spacing", "ct_z_spacing", "study_year")
    results = {}
    permanova_rows = []
    permdisp_rows = []
    regression_rows = []
    for factor in categorical_factors:
        available = [case_id for case_id in patient_ids if factors.get(case_id, {}).get(factor) not in (None, "")]
        labels = [str(factors[case_id][factor]) for case_id in available]
        levels, counts = np.unique(labels, return_counts=True)
        modality = "candidate_consensus" if factor == "tissue_source_site" else "ct"
        source = consensus_distance if modality == "candidate_consensus" else matrices["ct"]
        positions = [index[case_id] for case_id in available]
        local_distance = source[np.ix_(positions, positions)]
        result = {
            "modality": modality,
            "variable_type": "categorical",
            "n": len(available),
            "levels": len(levels),
            "level_counts": dict(zip(map(str, levels), map(int, counts))),
            "permanova": {"test": "not_estimable", "pseudo_f": None, "r_squared": None, "permutation_p": None, "q_value": None},
            "permdisp": {"test": "not_estimable", "f_statistic": None, "permutation_p": None, "q_value": None},
        }
        if len(levels) < 2 or len(available) <= len(levels):
            results[factor] = result
            continue
        metadata = pd.DataFrame({"group": labels}, index=available)
        dm = DistanceMatrix(local_distance, ids=available)
        permanova_result = permanova(
            dm, grouping=metadata, column="group", permutations=permutations, seed=seed
        )
        pseudo_f = float(permanova_result["test statistic"])
        permanova_p = float(permanova_result["p-value"])
        ratio = pseudo_f * (len(levels) - 1) / (len(available) - len(levels))
        permanova_r_squared = ratio / (1 + ratio) if np.isfinite(pseudo_f) else None
        result["permanova"] = {
            "test": "permanova",
            "pseudo_f": pseudo_f,
            "r_squared": float(permanova_r_squared) if permanova_r_squared is not None else None,
            "permutation_p": permanova_p,
            "q_value": None,
        }
        permanova_rows.append((factor, permanova_p))
        if np.min(counts) < 2:
            result["permdisp"] = {
                "test": "not_estimable",
                "reason": "PERMDISP requires at least two observations in every group.",
                "f_statistic": None,
                "permutation_p": None,
                "q_value": None,
            }
            results[factor] = result
            continue
        try:
            permdisp_result = permdisp(
                dm, grouping=metadata, column="group", test="median",
                permutations=permutations, seed=seed
            )
            dispersion_f = float(permdisp_result["test statistic"])
            dispersion_p = float(permdisp_result["p-value"])
            if not np.isfinite(dispersion_f) or not np.isfinite(dispersion_p):
                raise ValueError("PERMDISP returned a non-finite statistic or p-value")
            result["permdisp"] = {
                "test": "median",
                "f_statistic": dispersion_f,
                "permutation_p": dispersion_p,
                "q_value": None,
            }
            permdisp_rows.append((factor, dispersion_p))
        except (ValueError, np.linalg.LinAlgError) as error:
            results[factor] = {
                **result,
                "permdisp": {
                    "test": "not_estimable", "reason": str(error),
                    "f_statistic": None, "permutation_p": None, "q_value": None,
                },
            }
            continue
        else:
            results[factor] = result
    for factor in continuous_factors:
        available = [
            case_id for case_id in patient_ids
            if factors.get(case_id, {}).get(factor) is not None
            and np.isfinite(factors[case_id][factor])
        ]
        values = np.asarray([float(factors[case_id][factor]) for case_id in available])
        if len(values) < 3 or len(np.unique(values)) < 2:
            results[factor] = {
                "modality": "ct", "variable_type": "continuous", "n": len(values),
                "distance_regression": {
                    "test": "not_estimable", "pseudo_f": None, "r_squared": None,
                    "permutation_p": None, "q_value": None,
                },
            }
            continue
        positions = [index[case_id] for case_id in available]
        kernel = distance_kernel(matrices["ct"][np.ix_(positions, positions)])
        centered = values - values.mean()
        total = float(np.trace(kernel))
        predictor_ss = float(centered @ centered)
        if len(values) <= 2 or len(np.unique(values)) < 2 or total <= 0 or predictor_ss <= 0:
            results[factor] = {
                "modality": "ct", "variable_type": "continuous", "test": "not_estimable",
                "n": len(values), "distance_regression": {
                    "pseudo_f": None, "r_squared": None, "permutation_p": None, "q_value": None,
                },
            }
            continue
        model_ss = float(centered @ kernel @ centered / predictor_ss)
        residual_ss = total - model_ss
        if residual_ss <= 0:
            results[factor] = {
                "modality": "ct", "variable_type": "continuous", "n": len(values),
                "distance_regression": {
                    "test": "not_estimable", "pseudo_f": None, "r_squared": None,
                    "permutation_p": None, "q_value": None,
                },
            }
            continue
        observed = model_ss / (residual_ss / (len(values) - 2))
        rng = np.random.default_rng(seed)
        exceed = 0
        for _ in range(permutations):
            shuffled = rng.permutation(centered)
            permuted_ss = float(shuffled @ kernel @ shuffled / (shuffled @ shuffled))
            permuted_residual_ss = total - permuted_ss
            permuted_f = np.inf if permuted_residual_ss <= 0 else permuted_ss / (permuted_residual_ss / (len(values) - 2))
            exceed += permuted_f >= observed
        p_value = (exceed + 1) / (permutations + 1)
        results[factor] = {
            "modality": "ct",
            "variable_type": "continuous",
            "n": len(values),
            "distance_regression": {
                "test": "permutation_distance_regression",
                "pseudo_f": float(observed),
                "r_squared": float(model_ss / total),
                "permutation_p": p_value,
                "q_value": None,
            },
        }
        regression_rows.append((factor, p_value))
    for family, field, rows in (
        ("permanova", "q_value", permanova_rows),
        ("permdisp", "q_value", permdisp_rows),
        ("distance_regression", "q_value", regression_rows),
    ):
        if rows:
            for (factor, _), q_value in zip(rows, multipletests([p_value for _, p_value in rows], method="fdr_bh")[1]):
                results[factor][family][field] = float(q_value)
    return tool_result("confounder_representation_effect", {"partition": results})


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "representation_concordance": {
        "aspect": "affinity_geometry_concordance",
        "dimension": "cross_modal_consistency", "scopes": ("set", "pair", "partition"),
        "description": "Assess current candidate membership or pair-boundary representation in CT, WSI, RNA, and WXS native patient-distance geometries; partition scope additionally evaluates cross-view patient-geometry concordance with GRV uncertainty.",
        "question_foci": {"set": ("membership_representation",), "pair": ("boundary_representation",), "partition": ("patient_geometry_concordance",)},
        "selection_guidance": (
            "Use for current membership or pair-boundary representation in native modality geometries. Interpret "
            "the four views jointly; they are not votes and no single modality is automatically authoritative. "
            "Set/pair scope does not assess internal subdivision. Partition-scope patient_geometry_concordance is "
            "the only scope that computes global cross-view GRV uncertainty."
        ),
        "function": representation_concordance,
    },
    "structural_diagnostics": {
        "aspect": "structural_diagnostics",
        "dimension": "cross_modal_consistency", "scopes": ("set", "pair", "partition"),
        "description": "Return partition triage measurements or requested set/pair structural measurements and feasible spectral solutions.",
        "question_foci": {"partition": ("partition_structural_screen",), "set": ("internal_subdivision",), "pair": ("boundary_structure",)},
        "selection_guidance": (
            "Use for structural evidence concerning internal subdivision or pair-boundary geometry relevant to split "
            "or merge assessment. For set scope, select this tool only when the EvidenceRequest explicitly concerns "
            "internal subdivision, split, or granularity. Do not use set-scope structural diagnostics to answer whether "
            "current membership is represented relative to rest or neighboring patients. Pair scope can provide "
            "complementary current-boundary and union-structure evidence. Partition scope remains the mandatory "
            "structural screen."
        ),
        "function": structural_diagnostics,
    },
    "rna_pathway_enrichment": {
        "aspect": "rna_pathway_enrichment",
        "dimension": "biological_support", "scopes": ("set",),
        "description": "Run Hallmark preranked GSEA using PyDESeq2 target-versus-rest Wald statistics from raw RNA counts.",
        "question_foci": {"set": ("transcriptomic_phenotype",)},
        "selection_guidance": "Use when the unresolved biological question concerns coherent transcriptomic pathway programs.",
        "function": rna_pathway_enrichment,
    },
    "wxs_mutation_enrichment": {
        "aspect": "wxs_mutation_enrichment",
        "dimension": "biological_support", "scopes": ("set",),
        "description": "Run all-gene Fisher mutation tests on the full nonsynonymous interpretation matrix; return all configured ccRCC drivers, top exploratory results, and a complete result artifact.",
        "question_foci": {"set": ("mutation_phenotype",)},
        "selection_guidance": "Use when the unresolved biological question concerns somatic mutation patterns or whether an interpreted phenotype has a distinct mutation context.",
        "function": wxs_mutation_enrichment,
    },
    "known_label_echo": {
        "aspect": "known_label_echo",
        "dimension": "known_label_echo", "scopes": ("partition",),
        "description": "Compare the full partition with stage, grade, major T/M stage, TCGA m1-m4, and ClearCode34 labels.",
        "question_foci": {"partition": ("taxonomy_correspondence",)},
        "selection_guidance": "Use when the question concerns correspondence with clinical stratification or established ccRCC taxonomies.",
        "function": known_label_echo,
    },
    "confounder_association": {
        "aspect": "confounder_association",
        "dimension": "confounder_exclusion", "scopes": ("set", "partition"),
        "description": "Test association of candidate membership with measured site and CT acquisition metadata.",
        "question_foci": {"set": ("candidate_specific_confounder_association",), "partition": ("partition_confounder_association",)},
        "selection_guidance": "Use when the question concerns whether candidate membership is associated with measured technical, site, or acquisition factors.",
        "function": confounder_association,
    },
    "confounder_representation_effect": {
        "aspect": "confounder_representation_effect",
        "dimension": "confounder_exclusion", "scopes": ("partition",),
        "description": "Test categorical technical factors with PERMANOVA and PERMDISP on fused or CT native distances, and continuous factors with CT distance-based regression; apply separate BH families.",
        "question_foci": {"partition": ("partition_factor_geometry_association",)},
        "selection_guidance": "Use when the unresolved question concerns whether technical factors are associated with patient representation geometry rather than membership alone.",
        "function": confounder_representation_effect,
    },
}


def compact_tool_result(raw: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    metrics = dict(raw.get("metrics", {}) or {})
    refs = []
    def collect(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                collect(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")
        else:
            refs.append(path)
    collect(metrics, f"tool_results.{tool_name}.metrics")
    return {
        "tool_name": tool_name,
        "status": str(raw["status"]),
        "metrics": metrics,
        "metric_refs": refs,
        **({"artifact_paths": dict(raw["artifact_paths"])} if raw.get("artifact_paths") else {}),
        "warnings": list(raw.get("warnings", [])),
        "missing_reason": str(raw.get("missing_reason", "")),
        "errors": list(raw.get("errors", [])),
    }


def build_selection_tools(registry: Mapping[str, Mapping[str, Any]] | None = None) -> list[Any]:
    from langchain_core.tools import tool

    available = registry or TOOL_REGISTRY
    result = []
    for name, metadata in available.items():
        description = metadata["description"]
        guidance = str(metadata.get("selection_guidance", "") or "")
        if guidance:
            description = f"{description} Selection guidance: {guidance}"

        def select_tool() -> str:
            return "Selected scientific evidence tool."

        result.append(tool(name, description=description)(select_tool))
    return result
