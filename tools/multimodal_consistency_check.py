from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.subtype_review_common import bh_fdr, scoped_candidate_sets, tool_result


MODALITIES = ("ct", "wsi", "rna", "wxs", "cnv")


def modality_affinity_path(output_root: str, modality: str) -> Path:
    if modality == "wxs":
        return Path(output_root) / "wxs" / "wxs_affinity.npy"
    if modality == "cnv":
        return Path(output_root) / "wxs" / "cnv_affinity.npy"
    return Path(output_root) / "candidate_subtype" / f"{modality}_affinity.npy"


def round_value(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 6) if np.isfinite(value) else None


def normalize_affinity(network: np.ndarray) -> np.ndarray:
    matrix = np.asarray(network, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Affinity matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("Affinity matrix contains non-finite values")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise ValueError("Affinity diagonal must be positive")
    matrix = matrix / np.sqrt(np.outer(diagonal, diagonal))
    matrix = (matrix + matrix.T) / 2.0
    matrix = np.clip(matrix, 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def normalized_affinity_with_audit(network: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(network, dtype=float)
    normalized = normalize_affinity(raw)
    scaled = raw / np.sqrt(np.outer(np.diag(raw), np.diag(raw)))
    symmetric = (scaled + scaled.T) / 2.0
    high_count = int(np.sum(symmetric > 1.0))
    low_count = int(np.sum(symmetric < 0.0))
    clip_fraction = (high_count + low_count) / symmetric.size
    return normalized, {
        "raw_min": round_value(raw.min()),
        "raw_max": round_value(raw.max()),
        "normalized_min": round_value(normalized.min()),
        "normalized_max": round_value(normalized.max()),
        "gt1_clipping_count": high_count,
        "lt0_clipping_count": low_count,
        "clipping_fraction": round_value(clip_fraction),
        "warning": (
            "Affinity values required substantial clipping."
            if clip_fraction > 0.05 else ""
        ),
    }


def separation_score(within: float | None, between: float | None) -> float | None:
    if within is None or between is None:
        return None
    denominator = abs(within) + abs(between)
    return round_value((within - between) / denominator if denominator else 0.0)


def partition_separation(
    similarity: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    upper = np.triu_indices(len(labels), 1)
    values = similarity[upper]
    same = labels[upper[0]] == labels[upper[1]]
    within_values = values[same]
    between_values = values[~same]
    global_row = {
        "mean_within_affinity": round_value(within_values.mean()) if len(within_values) else None,
        "mean_between_affinity": round_value(between_values.mean()) if len(between_values) else None,
        "normalized_affinity_separation": separation_score(
            float(within_values.mean()) if len(within_values) else None,
            float(between_values.mean()) if len(between_values) else None,
        ),
    }
    per_set = {}
    labels_unique = np.unique(labels)
    for set_id in labels_unique:
        inside = np.flatnonzero(labels == set_id)
        others = [other for other in labels_unique if other != set_id]
        own_values = similarity[np.ix_(inside, inside)]
        own_values = own_values[~np.eye(len(inside), dtype=bool)]
        between_by_set = {
            str(other): float(similarity[np.ix_(inside, np.flatnonzero(labels == other))].mean())
            for other in others
        }
        nearest = max(between_by_set, key=between_by_set.get) if between_by_set else None
        margins = []
        if len(inside) > 1 and others:
            for patient in inside:
                own = float(np.delete(similarity[patient, inside], np.where(inside == patient)[0][0]).mean())
                other = max(
                    float(similarity[patient, np.flatnonzero(labels == other_set)].mean())
                    for other_set in others
                )
                margins.append(own - other)
        per_set[str(set_id)] = {
            "member_n": int(len(inside)),
            "mean_within_affinity": round_value(own_values.mean()) if len(own_values) else None,
            "nearest_other_set": nearest,
            "mean_nearest_between_affinity": (
                round_value(between_by_set[nearest]) if nearest else None
            ),
            "normalized_affinity_separation": separation_score(
                float(own_values.mean()) if len(own_values) else None,
                between_by_set.get(nearest) if nearest else None,
            ),
            "mean_affinity_margin": round_value(np.mean(margins)) if margins else None,
            "median_affinity_margin": round_value(np.median(margins)) if margins else None,
            "fraction_affinity_margin_positive": (
                round_value(np.mean(np.asarray(margins) > 0)) if margins else None
            ),
        }
    return global_row, per_set


def fixed_membership_silhouettes(
    distance: np.ndarray, labels: np.ndarray
) -> dict[str, float | None]:
    values = {}
    for index, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        others = [other for other in np.unique(labels) if other != label]
        if len(same) < 2 or not others:
            values[str(index)] = None
            continue
        own = float(np.delete(distance[index, same], np.where(same == index)[0][0]).mean())
        nearest_other = min(
            float(distance[index, np.flatnonzero(labels == other)].mean())
            for other in others
        )
        denominator = max(own, nearest_other)
        values[str(index)] = round_value(
            (nearest_other - own) / denominator if denominator else 0.0
        )
    return values


def silhouette_summary(
    silhouette_by_index: Mapping[str, float | None],
    labels: np.ndarray,
    case_ids: list[str],
) -> tuple[float | None, dict[str, dict[str, Any]], dict[str, float | None]]:
    per_set = {}
    for set_id in np.unique(labels):
        indices = np.flatnonzero(labels == set_id)
        values = [silhouette_by_index[str(index)] for index in indices]
        values = [value for value in values if value is not None]
        per_set[str(set_id)] = {
            "median_silhouette": round_value(np.median(values)) if values else None,
            "mean_silhouette": round_value(np.mean(values)) if values else None,
            "fraction_silhouette_positive": (
                round_value(np.mean(np.asarray(values) > 0)) if values else None
            ),
            "silhouette_available_n": len(values),
            "silhouette_missing_n": len(indices) - len(values),
            "comparison_status": "estimable" if values else "not_estimable",
            "not_estimable_reason": "insufficient_set_size" if not values else "",
        }
    available = [value for value in silhouette_by_index.values() if value is not None]
    return (
        round_value(np.median(available)) if available else None,
        per_set,
        {case_ids[int(index)]: value for index, value in silhouette_by_index.items()},
    )


def gower_matrix(distance: np.ndarray) -> np.ndarray:
    center = np.eye(len(distance)) - np.ones_like(distance) / len(distance)
    return -0.5 * center @ (distance**2) @ center


def pd_factor(labels: np.ndarray) -> np.ndarray:
    levels = {value: index for index, value in enumerate(np.unique(labels))}
    return np.asarray([levels[value] for value in labels], dtype=int)


def permanova_statistic(gower: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    design = np.eye(len(np.unique(labels)))[pd_factor(labels)]
    hat = design @ np.linalg.pinv(design)
    rank = int(np.linalg.matrix_rank(design))
    total = float(np.trace(gower))
    if total <= 0:
        return None, None
    between = max(float(np.trace(hat @ gower)), 0.0)
    residual = max(total - between, 0.0)
    numerator_df = rank - 1
    denominator_df = len(labels) - rank
    pseudo_f = (
        (between / numerator_df) / (residual / denominator_df)
        if numerator_df > 0 and denominator_df > 0 and residual > 0
        else None
    )
    return between / total, pseudo_f


def permanova_metrics(
    similarity: np.ndarray,
    labels: np.ndarray,
    permutations: int = 999,
    seed: int = 0,
) -> dict[str, Any]:
    if len(np.unique(labels)) < 2:
        reason = "single_set_partition"
    else:
        reason = ""
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    gower = gower_matrix(distance)
    r2, pseudo_f = permanova_statistic(gower, labels)
    if reason or r2 is None:
        return {
            "comparison_status": "not_estimable",
            "not_estimable_reason": reason or "zero_total_dispersion_or_residual_df",
            "available_n": len(labels),
            "set_count": int(len(np.unique(labels))),
            "permanova_r2": None,
            "pseudo_f": None,
            "permanova_p_value": None,
            "permutations": int(permutations),
        }
    rng = np.random.default_rng(seed)
    null = [permanova_statistic(gower, rng.permutation(labels))[0] for _ in range(permutations)]
    null = [value for value in null if value is not None]
    p_value = (1 + sum(value >= r2 for value in null)) / (len(null) + 1)
    return {
        "comparison_status": "estimable",
        "not_estimable_reason": "",
        "available_n": len(labels),
        "set_count": int(len(np.unique(labels))),
        "permanova_r2": round_value(r2),
        "pseudo_f": round_value(pseudo_f),
        "permanova_p_value": round_value(p_value),
        "permutations": int(permutations),
    }


def permdisp_coordinates(distance: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    coordinates = gower_matrix(distance)
    eigenvalues, eigenvectors = np.linalg.eigh(coordinates)
    negative = eigenvalues[eigenvalues < -1e-10]
    total = float(np.abs(eigenvalues).sum())
    audit = {
        "negative_eigenvalue_count": int(len(negative)),
        "negative_eigenvalue_sum": round_value(float(np.abs(negative).sum())),
        "negative_eigenvalue_fraction": round_value(
            float(np.abs(negative).sum()) / total if total else 0.0
        ),
        "limitations": (
            [
                "Negative PCoA eigenvalues were discarded; PERMDISP is an approximate diagnostic."
            ]
            if len(negative)
            else []
        ),
    }
    keep = eigenvalues > 1e-10
    if not np.any(keep):
        return None, audit
    return eigenvectors[:, keep] * np.sqrt(eigenvalues[keep]), audit


def permdisp_statistic(coordinates: np.ndarray, labels: np.ndarray) -> float | None:
    unique = np.unique(labels)
    if len(unique) < 2 or min(np.sum(labels == label) for label in unique) < 2:
        return None
    group_distances = [
        np.linalg.norm(
            coordinates[labels == label]
            - coordinates[labels == label].mean(axis=0),
            axis=1,
        )
        for label in unique
    ]
    grand = float(np.concatenate(group_distances).mean())
    between = sum(len(values) * (float(values.mean()) - grand) ** 2 for values in group_distances)
    within = sum(float(((values - values.mean()) ** 2).sum()) for values in group_distances)
    df_between = len(unique) - 1
    df_within = len(labels) - len(unique)
    return (between / df_between) / (within / df_within) if df_within > 0 and within > 0 else 0.0


def permdisp_metrics(
    similarity: np.ndarray,
    labels: np.ndarray,
    permutations: int = 999,
    seed: int = 100,
) -> dict[str, Any]:
    if len(np.unique(labels)) < 2:
        reason = "single_set_partition"
    elif min(np.sum(labels == label) for label in np.unique(labels)) < 2:
        reason = "singleton_set"
    else:
        reason = ""
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    coordinates, audit = permdisp_coordinates(distance)
    statistic = permdisp_statistic(coordinates, labels) if not reason and coordinates is not None else None
    if reason or statistic is None:
        return {
            "comparison_status": "not_estimable",
            "not_estimable_reason": reason or "zero_coordinate_dispersion",
            "permdisp_f": None,
            "permdisp_p_value": None,
            "permutations": int(permutations),
            **audit,
        }
    rng = np.random.default_rng(seed)
    null = [permdisp_statistic(coordinates, rng.permutation(labels)) for _ in range(permutations)]
    null = [value for value in null if value is not None]
    p_value = (1 + sum(value >= statistic for value in null)) / (len(null) + 1)
    return {
        "comparison_status": "estimable",
        "not_estimable_reason": "",
        "permdisp_f": round_value(statistic),
        "permdisp_p_value": round_value(p_value),
        "permutations": int(permutations),
        **audit,
    }


def compute_cross_modal_consistency(
    affinities: Mapping[str, np.ndarray | None],
    case_ids: list[str],
    memberships: Mapping[str, list[str]],
    *,
    permanova_permutations: int = 999,
    permdisp_permutations: int = 999,
    lowest_support_patients_to_report: int = 5,
) -> dict[str, Any]:
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Affinity patient order contains duplicates")
    labels_by_case = {
        case_id: set_id for set_id, members in memberships.items() for case_id in members
    }
    if set(case_ids) != set(labels_by_case):
        raise ValueError("Affinity patients and candidate members differ")
    labels = np.asarray([labels_by_case[case_id] for case_id in case_ids])
    rows = {}
    patient_profile = {
        case_id: {
            "silhouette_by_modality": {},
            "data_available_modalities": [],
            "membership_estimable_modalities": [],
            "membership_estimable_modality_count": 0,
            "positive_modalities": [],
            "negative_modalities": [],
            "support_count": 0,
        }
        for case_id in case_ids
    }
    for modality in MODALITIES:
        if affinities.get(modality) is None:
            rows[modality] = {
                "comparison_status": "scientific_unavailable",
                "not_estimable_reason": "missing_affinity_matrix",
                "per_set": {set_id: {"median_silhouette": None} for set_id in memberships},
                "permanova_r2": None,
                "permanova_p_value": None,
                "permanova_q_value": None,
                "permdisp_f": None,
                "permdisp_p_value": None,
                "permdisp_q_value": None,
                "limitations": ["Affinity matrix is unavailable for this modality."],
            }
            for case_id in case_ids:
                patient_profile[case_id]["silhouette_by_modality"][modality] = None
            continue
        for case_id in case_ids:
            patient_profile[case_id]["data_available_modalities"].append(modality)
        similarity, audit = normalized_affinity_with_audit(affinities[modality])
        if similarity.shape != (len(case_ids), len(case_ids)):
            raise ValueError(f"{modality} affinity shape does not match patient order")
        global_row, per_set = partition_separation(similarity, labels)
        distance = 1.0 - similarity
        silhouette_by_index = fixed_membership_silhouettes(distance, labels)
        median_silhouette, silhouette_sets, by_case = silhouette_summary(
            silhouette_by_index, labels, case_ids
        )
        for set_id, values in silhouette_sets.items():
            per_set[set_id].update(values)
        permanova = permanova_metrics(
            similarity, labels, permanova_permutations, MODALITIES.index(modality)
        )
        permdisp = permdisp_metrics(
            similarity, labels, permdisp_permutations, 100 + MODALITIES.index(modality)
        )
        limitations = []
        if any(values["comparison_status"] != "estimable" for values in silhouette_sets.values()):
            limitations.append("Some set silhouettes are not estimable under the fixed membership.")
        if permanova["comparison_status"] != "estimable":
            limitations.append(permanova["not_estimable_reason"])
        if permdisp["comparison_status"] != "estimable":
            limitations.append(permdisp["not_estimable_reason"])
        limitations.extend(permdisp.get("limitations", []))
        rows[modality] = {
            **global_row,
            "comparison_status": (
                "estimable"
                if all(values["comparison_status"] == "estimable" for values in silhouette_sets.values())
                and permanova["comparison_status"] == "estimable"
                and permdisp["comparison_status"] == "estimable"
                else "partially_estimable"
            ),
            "median_silhouette": median_silhouette,
            "mean_silhouette": round_value(
                np.mean([value for value in by_case.values() if value is not None])
            ) if any(value is not None for value in by_case.values()) else None,
            "fraction_silhouette_positive": round_value(
                np.mean([value > 0 for value in by_case.values() if value is not None])
            ) if any(value is not None for value in by_case.values()) else None,
            "per_set": per_set,
            "patient_silhouette": by_case,
            "permanova": permanova,
            "permdisp": permdisp,
            "permanova_r2": permanova["permanova_r2"],
            "permanova_p_value": permanova["permanova_p_value"],
            "permanova_q_value": None,
            "permdisp_f": permdisp["permdisp_f"],
            "permdisp_p_value": permdisp["permdisp_p_value"],
            "permdisp_q_value": None,
            "negative_eigenvalue_count": permdisp["negative_eigenvalue_count"],
            "negative_eigenvalue_fraction": permdisp["negative_eigenvalue_fraction"],
            "affinity_audit": audit,
            "limitations": limitations,
        }
        for case_id, value in by_case.items():
            patient_profile[case_id]["silhouette_by_modality"][modality] = value
            if value is not None and value > 0:
                patient_profile[case_id]["positive_modalities"].append(modality)
                patient_profile[case_id]["support_count"] += 1
            elif value is not None and value < 0:
                patient_profile[case_id]["negative_modalities"].append(modality)

    for profile in patient_profile.values():
        available = [
            value for value in profile["silhouette_by_modality"].values()
            if value is not None
        ]
        profile["membership_estimable_modalities"] = [
            modality for modality, value in profile["silhouette_by_modality"].items()
            if value is not None
        ]
        profile["membership_estimable_modality_count"] = len(
            profile["membership_estimable_modalities"]
        )
        profile["support_fraction"] = (
            round_value(
                profile["support_count"]
                / profile["membership_estimable_modality_count"]
            )
            if profile["membership_estimable_modality_count"] else None
        )
        profile["mean_estimable_silhouette"] = (
            round_value(np.mean(available)) if available else None
        )

    for field, key in (("permanova_p_value", "permanova_q_value"), ("permdisp_p_value", "permdisp_q_value")):
        available = [rows[modality][field] for modality in MODALITIES if rows[modality][field] is not None]
        q_values = bh_fdr(available)
        position = 0
        for modality in MODALITIES:
            if rows[modality][field] is not None:
                rows[modality][key] = round_value(q_values[position])
                nested = "permanova" if field == "permanova_p_value" else "permdisp"
                rows[modality][nested][key] = rows[modality][key]
                position += 1

    decision_sets = {}
    for set_id, members in memberships.items():
        profiles = [patient_profile[case_id] for case_id in members]
        support = [profile["support_count"] for profile in profiles]
        estimable_profiles = [
            profile
            for profile in profiles
            if profile["membership_estimable_modality_count"]
        ]
        estimable_support = [profile["support_count"] for profile in estimable_profiles]
        estimable_members = [
            case_id
            for case_id in members
            if patient_profile[case_id]["membership_estimable_modality_count"]
        ]
        decision_sets[set_id] = {
            "modality_support": {
                modality: {"median_silhouette": rows[modality]["per_set"][set_id].get("median_silhouette")}
                for modality in MODALITIES
            },
            "patient_membership_support": {
                "support_count_distribution": {
                    str(count): estimable_support.count(count)
                    for count in range(len(MODALITIES) + 1)
                },
                "median_support_count": (
                    round_value(np.median(estimable_support))
                    if estimable_support else None
                ),
                "all_estimable_positive_fraction": (
                    round_value(np.mean([
                        profile["support_count"] == profile["membership_estimable_modality_count"]
                        for profile in estimable_profiles
                    ])) if estimable_profiles else None
                ),
                "no_positive_among_estimable_fraction": (
                    round_value(np.mean([
                        profile["support_count"] == 0
                        for profile in estimable_profiles
                    ])) if estimable_profiles else None
                ),
                "membership_unestimable_patient_n": len(profiles) - len(estimable_profiles),
                "lowest_support_patients": [
                    {
                        "case_id": case_id,
                        "support_count": patient_profile[case_id]["support_count"],
                        "mean_estimable_silhouette": patient_profile[case_id]["mean_estimable_silhouette"],
                        "positive_modalities": patient_profile[case_id]["positive_modalities"],
                        "negative_modalities": patient_profile[case_id]["negative_modalities"],
                    }
                    for case_id in sorted(
                        estimable_members,
                        key=lambda item: (
                            patient_profile[item]["support_fraction"] is None,
                            patient_profile[item]["support_fraction"]
                            if patient_profile[item]["support_fraction"] is not None
                            else float("inf"),
                            patient_profile[item]["mean_estimable_silhouette"]
                            if patient_profile[item]["mean_estimable_silhouette"] is not None
                            else float("inf"),
                            item,
                        ),
                    )[:lowest_support_patients_to_report]
                ],
            },
        }
    limitations = sorted({
        limitation
        for row in rows.values()
        for limitation in row.get("limitations", [])
        if limitation
    })
    decision = {
        "cross_modal_consistency": {
            "per_set": decision_sets,
            "partition": {
                "permanova_r2": {
                    modality: rows[modality]["permanova_r2"] for modality in MODALITIES
                },
                "permdisp": {
                    modality: {
                        "f": rows[modality]["permdisp_f"],
                        "p_value": rows[modality]["permdisp_p_value"],
                        "q_value": rows[modality]["permdisp_q_value"],
                    }
                    for modality in MODALITIES
                },
            },
            "limitations": limitations,
        }
    }
    return {
        "modality_partition_support": rows,
        "patient_membership_profile": patient_profile,
        "affinity_audit": {
            modality: rows[modality].get("affinity_audit") for modality in MODALITIES
        },
        "decision_metrics": decision,
        "analysis_scope": (
            "Fixed candidate memberships were evaluated independently in CT, WSI, "
            "RNA, WXS, and CNV affinity spaces; no independent modality was reclustered. "
            "Structural characterization uses the actual SNF fused network for a binary probe."
        ),
    }


def multimodal_consistency_check(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
    scope="set_identity",
    target_ids=None,
    artifact_root=None,
):
    artifact_root = artifact_root or output_root
    cluster_id = str(cluster_state.get("cluster_id", "GLOBAL"))
    memberships = {
        key: sorted(value)
        for key, value in scoped_candidate_sets(scope, cluster_state, all_cluster_states).items()
    }
    candidate_dir = Path(output_root) / "candidate_subtype"
    patient_order = json.loads(
        (candidate_dir / "affinity_patient_order.json").read_text(encoding="utf-8")
    )
    active = {case_id for members in memberships.values() for case_id in members}
    positions = [index for index, case_id in enumerate(patient_order) if case_id in active]
    case_ids = [patient_order[index] for index in positions]
    affinities = {}
    for modality in MODALITIES:
        path = modality_affinity_path(output_root, modality)
        affinities[modality] = np.load(path)[np.ix_(positions, positions)] if path.exists() else None
    fused_path = candidate_dir / "fused_similarity.npy"
    fused = np.load(fused_path)[np.ix_(positions, positions)] if fused_path.exists() else None
    parameters = {}
    if config_dir:
        from tools.subtype_review_common import tool_parameters
        parameters = tool_parameters(config_dir, "cross_modal")
    metrics = compute_cross_modal_consistency(
        affinities,
        case_ids,
        memberships,
        permanova_permutations=int(parameters.get("permanova_permutations", 999)),
        permdisp_permutations=int(parameters.get("permdisp_permutations", 999)),
        lowest_support_patients_to_report=int(
            parameters.get("lowest_support_patients_to_report", 5)
        ),
    )
    from tools.cross_modal_structure import compute_structural_characterization
    structure_parameters = dict(parameters.get("structure", {}) or {})
    structure = compute_structural_characterization(
        fused,
        affinities,
        case_ids,
        memberships,
        resampling_fraction=float(structure_parameters.get("resampling_fraction", 0.8)),
        resampling_iterations=int(structure_parameters.get("resampling_iterations", 200)),
        pac_lower=float(structure_parameters.get("pac_lower", 0.1)),
        pac_upper=float(structure_parameters.get("pac_upper", 0.9)),
        random_seed=int(structure_parameters.get("random_seed", 0)),
    )
    decision = metrics["decision_metrics"]["cross_modal_consistency"]
    for set_id in memberships:
        full_internal = structure["internal_structure_by_set"][set_id]
        full_probe = dict(full_internal.get("fused_binary_probe", {}) or {})
        probe = {
            key: full_probe[key]
            for key in ("child_sizes", "normalized_cut", "median_silhouette", "mean_silhouette", "fraction_silhouette_positive")
            if key in full_probe
        }
        if "resampling" in full_probe:
            probe["resampling"] = full_probe["resampling"]
        decision["per_set"][set_id]["internal_structure"] = {
            "comparison_status": full_internal.get("comparison_status"),
            "member_n": full_internal.get("member_n"),
            "fused_binary_probe": probe,
            "probe_support_by_modality": {
                modality: {
                    key: value
                    for key, value in metrics_row.items()
                    if key in {"comparison_status", "not_estimable_reason", "median_silhouette"}
                }
                for modality, metrics_row in full_internal.get("probe_support_by_modality", {}).items()
            },
            "limitations": full_internal.get("limitations", []),
        }
        decision["per_set"][set_id]["boundary_to_other_sets"] = {}
    for pair in structure["boundary_by_pair"].values():
        left, right = pair["targets"]
        compact_pair = {
            "targets": pair["targets"],
            "fused": {
                metric: value
                for metric, value in pair["fused"].items()
                if metric not in {"patient_silhouette", "patient_margins"}
            },
            "modalities": {
                modality: {
                    metric: value
                    for metric, value in row.items()
                    if metric not in {"patient_silhouette", "patient_margins"}
                }
                for modality, row in pair["modalities"].items()
            },
        }
        decision["per_set"][left]["boundary_to_other_sets"][right] = compact_pair
        decision["per_set"][right]["boundary_to_other_sets"][left] = compact_pair
    decision["limitations"].extend(structure.get("limitations", []))
    metrics["structural_characterization"] = structure
    return tool_result(
        tool_name="multimodal_consistency_check",
        status="success",
        cluster_id=cluster_id,
        output_root=artifact_root,
        summary="Fixed candidate memberships were evaluated in five independent modality affinity networks.",
        metrics=metrics,
        decision_metrics=metrics["decision_metrics"],
        support_level="informational",
        concern_level="none",
        figures={},
    )
