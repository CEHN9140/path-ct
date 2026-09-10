from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.io import ensure_dir, write_json

MOFS_COLORS = [
    "#119da4",
    "#ff6666",
    "#ffc857",
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#8ab17d",
    "#6d597a",
]


def mofs_color(index: int) -> str:
    return MOFS_COLORS[int(index) % len(MOFS_COLORS)]


def style_mofs_axis(ax, *, grid_axis: str = "both") -> None:
    ax.set_facecolor("#f3f6f6")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=8, colors="black")
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color="#cacfd2",
            linestyle="--",
            linewidth=0.6,
            alpha=0.8,
        )
        ax.set_axisbelow(True)


def canonical_partition(labels: np.ndarray) -> tuple[int, ...]:
    label_map: dict[int, int] = {}
    canonical_labels: list[int] = []
    for label in labels.tolist():
        label = int(label)
        if label not in label_map:
            label_map[label] = len(label_map)
        canonical_labels.append(label_map[label])
    return tuple(canonical_labels)


def fit_kmedoids(
    distance_matrix: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    init: str,
    max_iter: int = 300,
) -> np.ndarray:
    values = np.asarray(distance_matrix, dtype=float)
    n_cases = int(values.shape[0])
    rng = np.random.default_rng(seed)
    if init == "heuristic":
        medoids = np.argsort(values.sum(axis=1))[:n_clusters].astype(int)
    elif init == "random":
        medoids = rng.choice(n_cases, size=n_clusters, replace=False).astype(int)
    elif init == "k-medoids++":
        medoids = [int(rng.integers(0, n_cases))]
        while len(medoids) < n_clusters:
            nearest = np.min(values[:, medoids], axis=1)
            weights = nearest**2
            weights[medoids] = 0.0
            total = float(np.sum(weights))
            if total <= 0:
                remaining = [index for index in range(n_cases) if index not in medoids]
                medoids.append(int(rng.choice(remaining)))
            else:
                medoids.append(int(rng.choice(n_cases, p=weights / total)))
        medoids = np.asarray(medoids, dtype=int)
    else:
        raise ValueError(f"Unsupported k-medoids init: {init}")

    for _ in range(max_iter):
        labels = np.argmin(values[:, medoids], axis=1)
        labels[medoids] = np.arange(n_clusters)
        new_medoids = medoids.copy()
        for cluster_index in range(n_clusters):
            members = np.flatnonzero(labels == cluster_index)
            costs = values[np.ix_(members, members)].sum(axis=1)
            new_medoids[cluster_index] = int(members[int(np.argmin(costs))])
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    else:
        raise RuntimeError(f"K-medoids did not converge within {max_iter} iterations")

    labels = np.asarray(np.argmin(values[:, medoids], axis=1), dtype=int)
    labels[medoids] = np.arange(n_clusters)
    return labels


def consensus_matrix_from_partitions(partitions: list[tuple[int, ...]]) -> np.ndarray:
    if not partitions:
        return np.zeros((0, 0), dtype=float)
    n_cases = len(partitions[0])
    consensus = np.zeros((n_cases, n_cases), dtype=float)
    for labels in partitions:
        values = np.asarray(labels, dtype=int)
        consensus += values[:, None] == values[None, :]
    consensus = consensus / float(len(partitions))
    np.fill_diagonal(consensus, 1.0)
    return (consensus + consensus.T) / 2.0


def cluster_sizes_from_labels(labels: list[int] | tuple[int, ...] | np.ndarray) -> list[int]:
    values = np.asarray(labels, dtype=int).reshape(-1)
    return [int(np.sum(values == label)) for label in sorted(set(values.tolist()))]


def consensus_silhouette_score(matrix: np.ndarray, labels: list[int] | tuple[int, ...] | np.ndarray) -> float | None:
    values = np.asarray(matrix, dtype=float)
    label_values = np.asarray(labels, dtype=int).reshape(-1)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] != label_values.shape[0]:
        return None
    if len(set(label_values.tolist())) < 2 or len(set(label_values.tolist())) >= len(label_values):
        return None
    distance = 1.0 - np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(distance, label_values, metric="precomputed"))
    except Exception:
        return None


def consensus_scatter_coordinates(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    n_cases = int(values.shape[0]) if values.ndim == 2 else 0
    if n_cases == 0:
        return np.zeros((0, 2), dtype=float)
    if n_cases == 1:
        return np.zeros((1, 2), dtype=float)
    distance = np.maximum(1.0 - np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0), 0.0)
    np.fill_diagonal(distance, 0.0)
    from sklearn.manifold import MDS

    coordinates = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
    ).fit_transform(distance)
    return np.nan_to_num(
        np.asarray(coordinates, dtype=float)[:, :2], nan=0.0, posinf=0.0, neginf=0.0
    )


def save_candidate_clustering_outputs(
    output_root: str,
    patient_ids: list[str],
    partition_records: list[dict[str, Any]],
    consensus_records: list[dict[str, Any]],
    best_record: Mapping[str, Any],
    *,
    snf_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    candidate_dir = ensure_dir(Path(output_root) / "candidate_subtype")
    algorithm_dirs: dict[str, str] = {}
    partition_outputs = []
    for record in partition_records:
        partition_id = str(record.get("partition_id", "") or "partition")
        algorithm = str(record.get("algorithm", "") or "unknown")
        if algorithm not in algorithm_dirs:
            algorithm_dirs[algorithm] = ensure_dir(candidate_dir / algorithm)
        labels = tuple(int(item) for item in tuple(record.get("labels", ())))
        label_map = {
            patient_ids[index]: int(label)
            for index, label in enumerate(labels)
            if index < len(patient_ids)
        }
        partition_output = {
            "partition_id": partition_id,
            "algorithm": algorithm,
            "run_index": int(record.get("run_index", 0) or 0),
            "n_clusters": int(record.get("n_clusters", 0) or 0),
            "seed": record.get("seed"),
            "algorithm_config": dict(record.get("algorithm_config", {}) or {}),
            "valid": bool(record.get("valid", True)),
            "skip_reason": str(record.get("skip_reason", "") or ""),
            "labels": label_map,
        }
        partition_path = write_json(
            Path(algorithm_dirs[algorithm]) / f"{partition_id}.json",
            partition_output,
        )
        partition_outputs.append({**partition_output, "json_path": partition_path})
    partition_records_path = write_json(
        candidate_dir / "partition_records.json",
        partition_outputs,
    )

    consensus_dir = ensure_dir(candidate_dir / "consensus_cluster")
    consensus_matrix_dir = ensure_dir(consensus_dir / "matrices")
    consensus_visualization_dir = ensure_dir(consensus_dir / "visualizations")
    from utils.visualization import configure_matplotlib

    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Ellipse

    consensus_cmap = LinearSegmentedColormap.from_list(
        "mofs_consensus", ["#f3f6f6", "#053061"]
    )
    consensus_outputs = []
    cdf_series = []
    delta_x = []
    relative_delta_y = []
    pac_y = []
    for index, record in enumerate(consensus_records):
        n_clusters = int(record.get("n_clusters", 0) or 0)
        matrix = np.asarray(record.get("consensus"), dtype=float)
        labels = tuple(int(item) for item in tuple(record.get("labels", ())))
        label_map = {
            patient_ids[label_index]: int(label)
            for label_index, label in enumerate(labels)
            if label_index < len(patient_ids)
        }
        matrix_path = consensus_matrix_dir / f"consensus_matrix_K{n_clusters}.npy"
        np.save(matrix_path, matrix)
        heatmap_path = (
            consensus_visualization_dir / f"consensus_heatmap_K{n_clusters}.png"
        )
        plot_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        plot_patient_ids = list(patient_ids)
        plot_labels = list(labels)
        if len(plot_labels) == len(plot_patient_ids):
            order = np.asarray(
                sorted(
                    range(len(plot_patient_ids)),
                    key=lambda item: (plot_labels[item], item),
                ),
                dtype=int,
            )
            plot_matrix = plot_matrix[np.ix_(order, order)]
            plot_patient_ids = [plot_patient_ids[int(item)] for item in order]
            plot_labels = [plot_labels[int(item)] for item in order]
        fig, ax = plt.subplots(
            figsize=(
                max(5.0, min(12.0, len(plot_patient_ids) * 0.12)),
                max(4.5, min(12.0, len(plot_patient_ids) * 0.12)),
            ),
            dpi=160,
        )
        image = ax.imshow(
            plot_matrix, cmap=consensus_cmap, vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        ax.set_title(f"Consensus heatmap K={n_clusters}", fontweight="bold")
        if plot_labels:
            for boundary in np.where(np.diff(np.asarray(plot_labels, dtype=int)) != 0)[
                0
            ]:
                ax.axhline(float(boundary) + 0.5, color="black", linewidth=0.6)
                ax.axvline(float(boundary) + 0.5, color="black", linewidth=0.6)
        if len(plot_patient_ids) <= 40:
            ax.set_xticks(range(len(plot_patient_ids)))
            ax.set_xticklabels(plot_patient_ids, rotation=90, fontsize=5)
            ax.set_yticks(range(len(plot_patient_ids)))
            ax.set_yticklabels(plot_patient_ids, fontsize=5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if plot_labels:
            top_ax = ax.inset_axes([0, 1.01, 1, 0.04], transform=ax.transAxes)
            label_order = sorted(set(plot_labels))
            top_ax.imshow(
                np.asarray([[label_order.index(label) for label in plot_labels]], dtype=float),
                cmap=ListedColormap([mofs_color(i) for i in range(len(label_order))]),
                aspect="auto",
                interpolation="nearest",
            )
            top_ax.set_xticks([])
            top_ax.set_yticks([])
            for spine in top_ax.spines.values():
                spine.set_visible(False)
        style_mofs_axis(ax, grid_axis="")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(heatmap_path)
        plt.close(fig)
        partition_path = write_json(
            consensus_dir / f"consensus_hierarchical_K{n_clusters}.json",
            {
                "partition_id": f"consensus_hierarchical_K{n_clusters}",
                "algorithm": "consensus_hierarchical",
                "n_clusters": n_clusters,
                "cdf_area": float(record.get("cdf_area", 0.0) or 0.0),
                "delta_area": float(record.get("delta_area", 0.0) or 0.0),
                "relative_delta_area": float(
                    record.get("relative_delta_area", 0.0) or 0.0
                ),
                "pac": float(record.get("pac", 0.0) or 0.0),
                "cluster_sizes": list(record.get("cluster_sizes", []) or []),
                "consensus_silhouette": record.get("consensus_silhouette"),
                "sampling_summary": dict(record.get("sampling_summary", {}) or {}),
                "labels": label_map,
            },
        )
        thresholds = [
            float(item) for item in list(record.get("cdf_thresholds", []) or [])
        ]
        cdf_values = [float(item) for item in list(record.get("cdf_values", []) or [])]
        cdf_series.append(
            {
                "label": f"K={n_clusters}",
                "x": thresholds,
                "y": cdf_values,
                "color": mofs_color(index),
            }
        )
        delta_x.append(float(n_clusters))
        relative_delta_y.append(float(record.get("relative_delta_area", 0.0) or 0.0))
        pac_y.append(float(record.get("pac", 0.0) or 0.0))
        consensus_outputs.append(
            {
                "n_clusters": n_clusters,
                "partition_count": int(record.get("partition_count", 0) or 0),
                "cdf_area": float(record.get("cdf_area", 0.0) or 0.0),
                "delta_area": float(record.get("delta_area", 0.0) or 0.0),
                "relative_delta_area": float(
                    record.get("relative_delta_area", 0.0) or 0.0
                ),
                "pac": float(record.get("pac", 0.0) or 0.0),
                "cluster_sizes": list(record.get("cluster_sizes", []) or []),
                "consensus_silhouette": record.get("consensus_silhouette"),
                "sampling_summary": dict(record.get("sampling_summary", {}) or {}),
                "matrix_path": str(matrix_path),
                "heatmap_path": str(heatmap_path),
                "partition_path": partition_path,
            }
        )

    selection_metric = str(best_record.get("selection_metric", "min_pac"))
    candidate_k_diagnostics = list(best_record.get("candidate_k_diagnostics", []) or [])
    selected_k_reason = str(best_record.get("selected_k_reason", "") or "")
    candidate_k_diagnostics_path = write_json(
        consensus_dir / "candidate_k_diagnostics.json",
        {
            "selection_metric": selection_metric,
            "selected_k": best_record.get("n_clusters"),
            "selected_k_reason": selected_k_reason,
            "llm_confidence": best_record.get("llm_confidence", ""),
            "llm_evidence_refs": best_record.get("llm_evidence_refs", []),
            "k_selection_audit_path": best_record.get("k_selection_audit_path", ""),
            "diagnostics": candidate_k_diagnostics,
        },
    )
    pac_lower = float(best_record["pac_lower"])
    pac_upper = float(best_record["pac_upper"])
    pac_gain_threshold = best_record.get("pac_gain_threshold")
    pac_gain = best_record.get("pac_gain")
    next_n_clusters = best_record.get("next_n_clusters")
    next_pac_gain = best_record.get("next_pac_gain")
    cdf_plot_path = consensus_dir / "cdf_plot.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    for item in cdf_series:
        ax.plot(
            item["x"],
            item["y"],
            label=item["label"],
            color=item["color"],
            linewidth=1.8,
        )
    ax.set_title("Consensus CDF")
    ax.set_xlabel("Consensus index")
    ax.set_ylabel("CDF")
    ax.set_ylim(bottom=0.0)
    style_mofs_axis(ax)
    if cdf_series:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(cdf_plot_path)
    plt.close(fig)

    delta_area_plot_path = consensus_dir / "delta_area_plot.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.plot(delta_x, relative_delta_y, marker="o", color="#2563eb", linewidth=1.8)
    ax.set_title("Relative delta area")
    ax.set_xlabel("K")
    ax.set_ylabel("Relative delta area")
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(delta_x)
    style_mofs_axis(ax)
    fig.tight_layout()
    fig.savefig(delta_area_plot_path)
    plt.close(fig)

    pac_plot_path = consensus_dir / "pac_plot.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.plot(delta_x, pac_y, marker="o", color="#dc2626", linewidth=1.8)
    ax.set_title("PAC")
    ax.set_xlabel("K")
    ax.set_ylabel("PAC")
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(delta_x)
    style_mofs_axis(ax)
    fig.tight_layout()
    fig.savefig(pac_plot_path)
    plt.close(fig)
    if "consensus" not in best_record:
        metadata_path = write_json(
            consensus_dir / "consensus_metadata.json",
            {
                "patient_ids": patient_ids,
                "best_n_clusters": None,
                "selection_metric": selection_metric,
                "pac_lower": pac_lower,
                "pac_upper": pac_upper,
                "pac_gain_threshold": pac_gain_threshold,
                "pac_gain": pac_gain,
                "next_n_clusters": next_n_clusters,
                "next_pac_gain": next_pac_gain,
                "partition_count": len(partition_records),
                "valid_partition_count": sum(
                    1 for record in partition_records if bool(record.get("valid", True))
                ),
                "algorithm_result_dirs": {
                    algorithm: str(path) for algorithm, path in algorithm_dirs.items()
                },
                "consensus_results": consensus_outputs,
                "candidate_k_diagnostics_path": candidate_k_diagnostics_path,
                "candidate_k_diagnostics": candidate_k_diagnostics,
                "selected_k_reason": selected_k_reason,
                "source_partitions_path": partition_records_path,
                "cdf_plot_path": str(cdf_plot_path),
                "delta_area_plot_path": str(delta_area_plot_path),
                "pac_plot_path": str(pac_plot_path),
            },
        )
        return {
            "candidate_subtype_dir": str(candidate_dir),
            "consensus_cluster_dir": str(consensus_dir),
            "metadata_path": metadata_path,
            "partition_records_path": partition_records_path,
            "partition_result_dir": str(candidate_dir),
            "algorithm_result_dirs": {
                algorithm: str(path) for algorithm, path in algorithm_dirs.items()
            },
            "cdf_plot_path": str(cdf_plot_path),
            "delta_area_plot_path": str(delta_area_plot_path),
            "pac_plot_path": str(pac_plot_path),
            "candidate_k_diagnostics_path": candidate_k_diagnostics_path,
            "best_n_clusters": None,
        }

    best_n_clusters = int(best_record.get("n_clusters", 0) or 0)
    best_consensus = np.asarray(best_record.get("consensus"), dtype=float)
    best_partition = tuple(int(item) for item in tuple(best_record.get("labels", ())))
    best_matrix_path = consensus_dir / "best_consensus_matrix.npy"
    np.save(best_matrix_path, best_consensus)
    consistency_heatmap_path = consensus_dir / "consistency_heatmap.png"
    plot_matrix = np.nan_to_num(best_consensus, nan=0.0, posinf=0.0, neginf=0.0)
    plot_patient_ids = list(patient_ids)
    plot_labels = list(best_partition)
    if len(plot_labels) == len(plot_patient_ids):
        order = np.asarray(
            sorted(
                range(len(plot_patient_ids)), key=lambda item: (plot_labels[item], item)
            ),
            dtype=int,
        )
        plot_matrix = plot_matrix[np.ix_(order, order)]
        plot_patient_ids = [plot_patient_ids[int(item)] for item in order]
        plot_labels = [plot_labels[int(item)] for item in order]
    fig, ax = plt.subplots(
        figsize=(
            max(5.0, min(12.0, len(plot_patient_ids) * 0.12)),
            max(4.5, min(12.0, len(plot_patient_ids) * 0.12)),
        ),
        dpi=160,
    )
    image = ax.imshow(
        plot_matrix, cmap=consensus_cmap, vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    ax.set_title(f"Best consensus heatmap K={best_n_clusters}", fontweight="bold")
    if plot_labels:
        for boundary in np.where(np.diff(np.asarray(plot_labels, dtype=int)) != 0)[0]:
            ax.axhline(float(boundary) + 0.5, color="black", linewidth=0.6)
            ax.axvline(float(boundary) + 0.5, color="black", linewidth=0.6)
    if len(plot_patient_ids) <= 40:
        ax.set_xticks(range(len(plot_patient_ids)))
        ax.set_xticklabels(plot_patient_ids, rotation=90, fontsize=5)
        ax.set_yticks(range(len(plot_patient_ids)))
        ax.set_yticklabels(plot_patient_ids, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if plot_labels:
        top_ax = ax.inset_axes([0, 1.01, 1, 0.04], transform=ax.transAxes)
        label_order = sorted(set(plot_labels))
        top_ax.imshow(
            np.asarray([[label_order.index(label) for label in plot_labels]], dtype=float),
            cmap=ListedColormap([mofs_color(i) for i in range(len(label_order))]),
            aspect="auto",
            interpolation="nearest",
        )
        top_ax.set_xticks([])
        top_ax.set_yticks([])
        for spine in top_ax.spines.values():
            spine.set_visible(False)
    style_mofs_axis(ax, grid_axis="")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(consistency_heatmap_path)
    plt.close(fig)
    best_labels = {
        patient_ids[index]: int(label)
        for index, label in enumerate(best_partition)
        if index < len(patient_ids)
    }
    scatter_source = (
        np.asarray(snf_matrix, dtype=float)
        if snf_matrix is not None
        and np.asarray(snf_matrix).shape == best_consensus.shape
        else best_consensus
    )
    scatter_coordinates = consensus_scatter_coordinates(scatter_source)
    scatter_points = [
        {
            "case_id": patient_ids[index],
            "cluster_label": int(best_partition[index])
            if index < len(best_partition)
            else 0,
            "x": float(scatter_coordinates[index, 0]),
            "y": float(scatter_coordinates[index, 1]),
        }
        for index in range(min(len(patient_ids), int(scatter_coordinates.shape[0])))
    ]
    scatter_coordinates_path = write_json(
        consensus_dir / "consensus_scatter_coordinates.json",
        scatter_points,
    )
    scatter_path = consensus_dir / "consensus_scatter.png"
    scatter_labels = sorted({int(point["cluster_label"]) for point in scatter_points})
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=160)
    if scatter_points:
        xy = np.asarray(
            [[point["x"], point["y"]] for point in scatter_points], dtype=float
        )
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
        fallback_width = span * 0.12
        for label in scatter_labels:
            label_points = np.asarray(
                [
                    [point["x"], point["y"]]
                    for point in scatter_points
                    if int(point["cluster_label"]) == label
                ],
                dtype=float,
            )
            center = label_points.mean(axis=0)
            width = fallback_width
            height = fallback_width
            angle = 0.0
            if label_points.shape[0] >= 2:
                covariance = np.cov(label_points, rowvar=False)
                if np.asarray(covariance).shape == (2, 2) and np.all(
                    np.isfinite(covariance)
                ):
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    order = np.argsort(eigenvalues)[::-1]
                    eigenvalues = np.maximum(
                        eigenvalues[order], (fallback_width / 2.0) ** 2
                    )
                    eigenvectors = eigenvectors[:, order]
                    width, height = 2.0 * np.sqrt(5.991 * eigenvalues)
                    angle = float(
                        np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    )
            color = mofs_color(label)
            ax.add_patch(
                Ellipse(
                    xy=center,
                    width=float(width),
                    height=float(height),
                    angle=angle,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=1.4,
                )
            )
            ax.scatter(
                label_points[:, 0],
                label_points[:, 1],
                s=34,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                label=f"C{label + 1}",
                zorder=3,
            )
    ax.set_title(f"Consensus scatter K={best_n_clusters}")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    style_mofs_axis(ax)
    if scatter_labels:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(scatter_path)
    plt.close(fig)
    best_partition_path = write_json(
        consensus_dir / "best_consensus_partition.json",
        {
            "partition_id": f"consensus_hierarchical_K{best_n_clusters}",
            "algorithm": "consensus_hierarchical",
            "n_clusters": best_n_clusters,
            "selection_metric": selection_metric,
            "pac": float(best_record.get("pac", 0.0) or 0.0),
            "pac_lower": pac_lower,
            "pac_upper": pac_upper,
            "pac_gain_threshold": pac_gain_threshold,
            "pac_gain": pac_gain,
            "next_n_clusters": next_n_clusters,
            "next_pac_gain": next_pac_gain,
            "cluster_sizes": list(best_record.get("cluster_sizes", []) or []),
            "consensus_silhouette": best_record.get("consensus_silhouette"),
            "selected_k_reason": selected_k_reason,
            "llm_confidence": best_record.get("llm_confidence", ""),
            "llm_evidence_refs": best_record.get("llm_evidence_refs", []),
            "k_selection_audit_path": best_record.get("k_selection_audit_path", ""),
            "cdf_area": float(best_record.get("cdf_area", 0.0) or 0.0),
            "delta_area": float(best_record.get("delta_area", 0.0) or 0.0),
            "relative_delta_area": float(
                best_record.get("relative_delta_area", 0.0) or 0.0
            ),
            "labels": best_labels,
        },
    )
    metadata_path = write_json(
        consensus_dir / "consensus_metadata.json",
        {
            "patient_ids": patient_ids,
            "best_n_clusters": best_n_clusters,
            "selection_metric": selection_metric,
            "pac_lower": pac_lower,
            "pac_upper": pac_upper,
            "pac_gain_threshold": pac_gain_threshold,
            "pac_gain": pac_gain,
            "next_n_clusters": next_n_clusters,
            "next_pac_gain": next_pac_gain,
            "best_pac": float(best_record.get("pac", 0.0) or 0.0),
            "best_delta_area": float(best_record.get("delta_area", 0.0) or 0.0),
            "best_relative_delta_area": float(
                best_record.get("relative_delta_area", 0.0) or 0.0
            ),
            "best_cdf_area": float(best_record.get("cdf_area", 0.0) or 0.0),
            "best_cluster_sizes": list(best_record.get("cluster_sizes", []) or []),
            "best_consensus_silhouette": best_record.get("consensus_silhouette"),
            "selected_k_reason": selected_k_reason,
            "llm_confidence": best_record.get("llm_confidence", ""),
            "llm_evidence_refs": best_record.get("llm_evidence_refs", []),
            "k_selection_audit_path": best_record.get("k_selection_audit_path", ""),
            "candidate_k_diagnostics_path": candidate_k_diagnostics_path,
            "candidate_k_diagnostics": candidate_k_diagnostics,
            "shape": list(best_consensus.shape),
            "partition_count": len(partition_records),
            "valid_partition_count": sum(
                1 for record in partition_records if bool(record.get("valid", True))
            ),
            "algorithm_result_dirs": {
                algorithm: str(path) for algorithm, path in algorithm_dirs.items()
            },
            "consensus_results": consensus_outputs,
            "source_partitions_path": partition_records_path,
            "scatter_path": str(scatter_path),
            "scatter_coordinates_path": scatter_coordinates_path,
            "scatter_source": "snf_matrix"
            if scatter_source is not best_consensus
            else "consensus_matrix",
            "scatter_ellipse_count": len(scatter_labels),
            "scatter_ellipse_labels": scatter_labels,
            "cdf_plot_path": str(cdf_plot_path),
            "delta_area_plot_path": str(delta_area_plot_path),
            "pac_plot_path": str(pac_plot_path),
        },
    )
    return {
        "candidate_subtype_dir": str(candidate_dir),
        "consensus_cluster_dir": str(consensus_dir),
        "matrix_path": str(best_matrix_path),
        "metadata_path": metadata_path,
        "partition_records_path": partition_records_path,
        "partition_result_dir": str(candidate_dir),
        "algorithm_result_dirs": {
            algorithm: str(path) for algorithm, path in algorithm_dirs.items()
        },
        "consistency_heatmap_path": str(consistency_heatmap_path),
        "scatter_path": str(scatter_path),
        "scatter_coordinates_path": scatter_coordinates_path,
        "cdf_plot_path": str(cdf_plot_path),
        "delta_area_plot_path": str(delta_area_plot_path),
        "pac_plot_path": str(pac_plot_path),
        "candidate_k_diagnostics_path": candidate_k_diagnostics_path,
        "selected_k_reason": selected_k_reason,
        "candidate_k_diagnostics": candidate_k_diagnostics,
        "best_partition_path": best_partition_path,
        "best_n_clusters": best_n_clusters,
    }
