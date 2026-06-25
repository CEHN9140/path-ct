from __future__ import annotations

import csv
import importlib
import json
import math
import os
import struct
import sys
import textwrap
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.subtype_review_common import (
    clinical_table,
    subtype_review_confounder_fields,
    subtype_review_tool_definitions,
    subtype_review_tool_imports,
)
from utils.io import ensure_dir, write_json
from utils.llm_utils import call_llm_json, load_llm_client, load_yaml_file
from utils.subtype_review_contract import (
    EVIDENCE_BLOCK_NAMES,
    FINAL_ACTION_REASON_CODES,
    FINAL_REVIEW_ACTIONS,
    action_decision_contract_issues,
    blocks_from_tool_plan,
    build_agentic_evidence_matrix,
    build_evidence_catalog,
    build_verifier_decision,
    compact_biological_metrics_for_review,
    decision_consistency_check,
    evidence_blocks_missing_metrics,
    evidence_catalog_by_id,
    metric_refs_for_blocks,
    metric_refs_from_evidence_ids,
    normalize_action_decision,
    normalize_confidence_basis,
    normalize_confidence_level,
    normalize_decision_state,
    reason_codes_for_action,
)
from utils.tool_utils import safe_identifier, to_jsonable

REVIEW_UNAVAILABLE_STATUS = "review_unavailable"
INTERNAL_REVIEW_ACTIONS = FINAL_REVIEW_ACTIONS | {"call_tools", "continue_review"}
GLOBAL_FIGURE_STEMS = [
    "consensus_matrix_heatmap",
    "integrated_snf_embedding_scatter",
    "modality_embedding_4panel",
    "rna_hallmark_ssgsea_bubble",
    "rna_subtype_signature_dotplot",
    "mutation_gene_oncoplot",
    "mutation_kegg_pathway_bubble",
    "mutation_reactome_pathway_bubble",
    "mutation_pathway_bubble",
    "rna_hallmark_ssgsea_heatmap",
    "rna_subtype_defining_hallmark_heatmap",
    "rna_top_pathway_boxplots",
    "subtype_evidence_score_panel",
    "survival_global_km",
    "confounder_association_heatmap",
    "known_label_association_heatmap",
]

DEFAULT_BUDGET_STATE = {
    "review_rounds_used": 0,
    "supplement_rounds_used": 0,
    "llm_audit_calls_used": 0,
    "report_calls_used": 0,
    "tool_calls_used": 0,
    "split_plans_used": 0,
    "merge_plans_used": 0,
    "revisions_used": 0,
    "llm_parse_failures": 0,
    "agent_failures": 0,
    "tool_failures": 0,
    "no_new_evidence_rounds": 0,
    "budget_exhausted": False,
    "budget_exhausted_reason": "",
}


def number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def short_feature_label(value: Any, max_len: int | None = 42) -> str:
    label = str(value or "")
    for prefix in ("KEGG_MEDICUS_", "HALLMARK_", "KEGG_", "REACTOME_", "BIOCARTA_", "PID_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label if max_len is None or len(label) <= max_len else label[:max_len]


def wrapped_feature_label(value: Any, width: int = 34) -> str:
    label = short_feature_label(value, None).replace("_", " ")
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False)) or label


def figure_output_path(output_root: str, filename: str) -> Path:
    return ensure_dir(Path(output_root) / "subtype_review" / "global" / "figures") / filename


def clear_stale_global_figures(output_root: str) -> None:
    figure_dir = ensure_dir(Path(output_root) / "subtype_review" / "global" / "figures")
    for stem in GLOBAL_FIGURE_STEMS:
        path = figure_dir / f"{stem}.png"
        if path.exists():
            path.unlink()


def cluster_figure_output_path(output_root: str, cluster_id: str, filename: str) -> Path:
    return ensure_dir(Path(output_root) / "subtype_review" / safe_identifier(cluster_id) / "figures") / filename


def setup_review_figure_style():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def save_review_figure(fig, output_root: str, stem: str) -> str:
    png_path = figure_output_path(output_root, f"{stem}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(png_path)


def write_png_rgb(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(
        b"\x00"
        + b"".join(bytes(tuple(max(0, min(255, int(channel))) for channel in pixels[y * width + x])) for x in range(width))
        for y in range(height)
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def save_fallback_figure(output_root: str, stem: str, seed: int = 0) -> str:
    png_path = figure_output_path(output_root, f"{stem}.png")
    width, height = 640, 420
    palette = [
        (78, 121, 167),
        (242, 142, 43),
        (89, 161, 79),
        (225, 87, 89),
        (176, 122, 161),
        (118, 183, 178),
    ]
    pixels = []
    for y in range(height):
        for x in range(width):
            base = 248 if (x // 24 + y // 24) % 2 else 255
            color = [base, base, base]
            band = (x * 7 + y * 11 + seed * 31) % 211
            if band < 11:
                accent = palette[(x // 80 + y // 70 + seed) % len(palette)]
                color = [int(0.35 * base + 0.65 * value) for value in accent]
            pixels.append(tuple(color))
    write_png_rgb(png_path, width, height, pixels)
    return str(png_path)


def save_fallback_cluster_figure(output_root: str, cluster_id: str, stem: str, seed: int = 0) -> str:
    png_path = cluster_figure_output_path(output_root, cluster_id, f"{stem}.png")
    width, height = 520, 340
    palette = [(78, 121, 167), (242, 142, 43), (89, 161, 79), (225, 87, 89)]
    pixels = []
    for y in range(height):
        for x in range(width):
            base = 250 if (x // 20 + y // 20) % 2 else 255
            color = [base, base, base]
            band = (x * 5 + y * 13 + seed * 17) % 173
            if band < 9:
                accent = palette[(x // 70 + y // 55 + seed) % len(palette)]
                color = [int(0.4 * base + 0.6 * value) for value in accent]
            pixels.append(tuple(color))
    write_png_rgb(png_path, width, height, pixels)
    return str(png_path)


def cluster_labels(cluster_states: list[Mapping[str, Any]]) -> dict[str, str]:
    labels = {}
    for cluster_state in cluster_states:
        if not cluster_state_is_active(cluster_state):
            continue
        cluster_id = str(cluster_state.get("cluster_id", "") or "")
        if not cluster_id:
            continue
        for case_id in list(cluster_state.get("member_ids", []) or []):
            labels[str(case_id)] = cluster_id
    return labels


def cluster_state_is_active(cluster_state: Mapping[str, Any]) -> bool:
    if str(cluster_state.get("absorbed_into", "") or ""):
        return False
    action = str(
        cluster_state.get("final_action")
        or cluster_state.get("final_decision")
        or cluster_state.get("status")
        or ""
    ).lower()
    return action not in {"drop", "split", "merge"}


def subtype_palette(labels: Mapping[str, str]) -> dict[str, Any]:
    if not labels:
        return {}
    plt = setup_review_figure_style()
    cmap = plt.get_cmap("tab10")
    cluster_ids = sorted(set(labels.values()))
    return {cluster_id: cmap(index % 10) for index, cluster_id in enumerate(cluster_ids)}


def patient_state_map(patient_states: list[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("case_id", "") or ""): item
        for item in list(patient_states or [])
        if str(item.get("case_id", "") or "")
    }


def numeric_vector(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        value = [value[key] for key in sorted(value)]
    values = []
    for item in list(value or []):
        number = number_or_none(item)
        if number is None:
            return []
        values.append(number)
    return values


def read_case_feature_csv(path: str) -> tuple[list[str], dict[str, list[float]]]:
    if not path or not Path(path).exists():
        return [], {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "case_id" not in rows[0]:
        return [], {}
    features = [name for name in rows[0].keys() if name != "case_id"]
    table = {}
    for row in rows:
        values = []
        for feature in features:
            number = number_or_none(row.get(feature))
            values.append(float("nan") if number is None else number)
        table[str(row.get("case_id", "") or "")] = values
    return features, table


def matrix_from_patient_vectors(
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
    modality: str,
) -> tuple[list[str], np.ndarray]:
    states = patient_state_map(patient_states)
    rows = []
    case_ids = []
    for case_id in sorted(labels):
        state = dict(states.get(case_id, {}) or {})
        if modality == "ct":
            values = numeric_vector(dict(state.get("ct_evidence", {}) or {}).get("features", []))
        elif modality == "wsi":
            values = numeric_vector(dict(state.get("wsi_evidence", {}) or {}).get("features", []))
        else:
            omics = dict(state.get("omics_evidence", {}) or {})
            values = numeric_vector(omics.get(f"{modality}_features", []))
        if values:
            case_ids.append(case_id)
            rows.append(values)
    if not rows or len({len(row) for row in rows}) != 1:
        return [], np.zeros((0, 0), dtype=float)
    matrix = np.asarray(rows, dtype=float)
    finite_columns = np.isfinite(matrix).all(axis=0)
    matrix = matrix[:, finite_columns]
    return case_ids, matrix


def matrix_from_feature_path(
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
    path_key: str,
) -> tuple[list[str], list[str], np.ndarray]:
    path = ""
    for state in list(patient_states or []):
        omics = dict(dict(state).get("omics_evidence", {}) or {})
        path = str(omics.get(path_key, "") or "")
        if path:
            break
    features, table = read_case_feature_csv(path)
    case_ids = [case_id for case_id in sorted(labels) if case_id in table]
    if not case_ids or not features:
        return [], [], np.zeros((0, 0), dtype=float)
    matrix = np.asarray([table[case_id] for case_id in case_ids], dtype=float)
    finite_columns = np.isfinite(matrix).all(axis=0)
    return case_ids, [feature for feature, keep in zip(features, finite_columns) if keep], matrix[:, finite_columns]


def matrix_from_ssgsea_reports(output_root: str, labels: Mapping[str, str]) -> tuple[list[str], list[str], np.ndarray]:
    table: dict[str, dict[str, float]] = {}
    features = set()
    for path in sorted((Path(output_root) / "subtype_review").glob("*/ssgsea/gseapy.gene_set.ssgsea.report.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                case_id = str(row.get("Name", "") or "")
                term = str(row.get("Term", "") or "")
                value = number_or_none(row.get("NES"))
                if case_id not in labels or not term or value is None:
                    continue
                table.setdefault(case_id, {})[term] = value
                features.add(term)
    case_ids = [case_id for case_id in sorted(labels) if case_id in table]
    ordered_features = sorted(features)
    if not case_ids or not ordered_features:
        return [], [], np.zeros((0, 0), dtype=float)
    matrix = np.asarray(
        [[table.get(case_id, {}).get(feature, np.nan) for feature in ordered_features] for case_id in case_ids],
        dtype=float,
    )
    finite_columns = np.isfinite(matrix).all(axis=0)
    return case_ids, [feature for feature, keep in zip(ordered_features, finite_columns) if keep], matrix[:, finite_columns]


def matrix_from_rna_pathway_scores(
    output_root: str,
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> tuple[list[str], list[str], np.ndarray]:
    case_ids, features, matrix = matrix_from_ssgsea_reports(output_root, labels)
    if matrix.size:
        return case_ids, features, matrix
    return matrix_from_feature_path(patient_states, labels, "rna_pathway_feature_path")


def robust_scale_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return values
    center = np.nanmedian(values, axis=0)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scale[scale == 0] = 1.0
    return np.nan_to_num((values - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def pca_coordinates(matrix: np.ndarray) -> np.ndarray:
    values = robust_scale_matrix(matrix)
    if values.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    if values.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    values = values - np.mean(values, axis=0)
    try:
        _, _, vt = np.linalg.svd(values, full_matrices=False)
        coords = values @ vt[:2].T
    except np.linalg.LinAlgError:
        coords = values[:, :2]
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return np.nan_to_num(coords[:, :2], nan=0.0, posinf=0.0, neginf=0.0)


def display_scale_coordinates(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=float)
    if values.size == 0:
        return values
    center = np.nanmedian(values, axis=0)
    scaled = values - center
    scale = np.nanpercentile(np.abs(scaled), 95, axis=0)
    scale[scale == 0] = 1.0
    return np.clip(np.nan_to_num(scaled / scale, nan=0.0, posinf=0.0, neginf=0.0), -1.2, 1.2)


def jaccard_mds_coordinates(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix > 0, dtype=bool)
    n = int(values.shape[0])
    if n <= 1:
        return np.zeros((n, 2), dtype=float)
    distance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            union = np.logical_or(values[i], values[j]).sum()
            score = 0.0 if union == 0 else 1.0 - np.logical_and(values[i], values[j]).sum() / union
            distance[i, j] = score
            distance[j, i] = score
    return mds_coordinates(distance)


def mds_coordinates(distance: np.ndarray) -> np.ndarray:
    values = np.asarray(distance, dtype=float)
    if values.shape[0] <= 1:
        return np.zeros((values.shape[0], 2), dtype=float)
    try:
        from sklearn.manifold import MDS

        coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=4).fit_transform(values)
    except Exception:
        centered = values - values.mean(axis=0, keepdims=True)
        coords = pca_coordinates(centered)
    return np.nan_to_num(np.asarray(coords, dtype=float)[:, :2], nan=0.0, posinf=0.0, neginf=0.0)


def metric_preview_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("preview_rows", [])
    return [dict(row) for row in list(value or []) if isinstance(row, Mapping)]


def biological_metric_rows(cluster_states: list[Mapping[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    rows = []
    for cluster_state in cluster_states:
        matrix = review_matrix_for_cluster(cluster_state)
        metrics = dict(dict(matrix.get("biological_support", {}) or {}).get("metrics", {}) or {})
        rows.extend(metric_preview_rows(metrics.get(metric_name)))
    return rows


def candidate_consensus_dir(output_root: str) -> Path:
    return Path(output_root) / "candidate_subtype" / "consensus_cluster"


def consensus_matrix_and_ids(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> tuple[np.ndarray, list[str]]:
    matrix_path = candidate_consensus_dir(output_root) / "best_consensus_matrix.npy"
    metadata_path = candidate_consensus_dir(output_root) / "consensus_metadata.json"
    for cluster_state in cluster_states:
        consensus = dict(cluster_state.get("consensus", {}) or {})
        candidate_path = str(consensus.get("matrix_path", "") or "")
        candidate = Path(candidate_path)
        if candidate_path and candidate.is_file():
            matrix_path = candidate
            metadata_path = Path(str(consensus.get("metadata_path", "") or ""))
            break
    if not matrix_path.exists():
        return np.zeros((0, 0), dtype=float), []
    matrix = np.asarray(np.load(matrix_path), dtype=float)
    patient_ids = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        patient_ids = [str(item) for item in list(metadata.get("patient_ids", []) or [])]
    if not patient_ids:
        partition_path = candidate_consensus_dir(output_root) / "best_consensus_partition.json"
        if partition_path.exists():
            partition = json.loads(partition_path.read_text(encoding="utf-8"))
            patient_ids = [str(item) for item in dict(partition.get("labels", {}) or {}).keys()]
    if not patient_ids:
        patient_ids = sorted(labels)
    n = min(matrix.shape[0], matrix.shape[1], len(patient_ids))
    return matrix[:n, :n], patient_ids[:n]


def ordered_case_ids(labels: Mapping[str, str], case_ids: list[str] | None = None) -> list[str]:
    ids = list(case_ids or labels.keys())
    return sorted([case_id for case_id in ids if case_id in labels], key=lambda item: (labels[item], item))


def subtype_segments(order_ids: list[str], labels: Mapping[str, str]) -> list[tuple[str, int, int]]:
    segments = []
    start = 0
    while start < len(order_ids):
        cluster_id = labels.get(order_ids[start], "")
        end = start + 1
        while end < len(order_ids) and labels.get(order_ids[end], "") == cluster_id:
            end += 1
        segments.append((cluster_id, start, end))
        start = end
    return segments


def draw_subtype_annotation(ax, order_ids: list[str], labels: Mapping[str, str], palette: Mapping[str, Any]) -> None:
    if not order_ids:
        return
    color_index = {cluster_id: index for index, cluster_id in enumerate(sorted(set(labels.values())))}
    colors = [palette[cluster_id] for cluster_id in sorted(color_index)]
    from matplotlib.colors import ListedColormap

    ann = np.asarray([[color_index[labels[case_id]] for case_id in order_ids]], dtype=int)
    ax.imshow(ann, aspect="auto", cmap=ListedColormap(colors), interpolation="nearest")
    ax.set_xlim(-0.5, len(order_ids) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for cluster_id, start, end in subtype_segments(order_ids, labels):
        center = (start + end - 1) / 2
        ax.text(center, 0, cluster_id, ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        if end < len(order_ids):
            ax.axvline(end - 0.5, color="white", linewidth=1.0)


def draw_subtype_boundaries(ax, order_ids: list[str], labels: Mapping[str, str]) -> None:
    for _, _, end in subtype_segments(order_ids, labels):
        if end < len(order_ids):
            ax.axvline(end - 0.5, color="white", linewidth=0.8)


def draw_subtype_points(ax, case_ids: list[str], coords: np.ndarray, labels: Mapping[str, str], palette: Mapping[str, Any]):
    for cluster_id in sorted(set(labels.values())):
        indices = [index for index, case_id in enumerate(case_ids) if labels.get(case_id) == cluster_id]
        if not indices:
            continue
        xy = coords[indices]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=22,
            color=palette.get(cluster_id),
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            label=cluster_id,
        )
        if len(indices) >= 2:
            center = xy.mean(axis=0)
            ax.scatter(center[0], center[1], s=52, marker="x", color=palette.get(cluster_id), linewidth=1.2)


def plot_consensus_matrix_heatmap(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    matrix, patient_ids = consensus_matrix_and_ids(output_root, cluster_states, labels)
    if matrix.size == 0 or not patient_ids:
        return ""
    order_ids = ordered_case_ids(labels, patient_ids)
    order = [patient_ids.index(case_id) for case_id in order_ids]
    matrix = np.clip(matrix[np.ix_(order, order)], 0.0, 1.0)
    plt = setup_review_figure_style()
    palette = subtype_palette(labels)
    fig = plt.figure(figsize=(6.2, 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.22, 5.0], width_ratios=[5.0, 0.24], hspace=0.08, wspace=0.06)
    ann_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    colorbar_ax = fig.add_subplot(grid[1, 1])
    draw_subtype_annotation(ann_ax, order_ids, labels, palette)
    ann_ax.set_title("Consensus subtype structure", pad=8)
    image = ax.imshow(matrix, aspect="equal", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    draw_subtype_boundaries(ax, order_ids, labels)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Consensus similarity")
    return save_review_figure(fig, output_root, "consensus_matrix_heatmap")


def plot_integrated_snf_embedding_scatter(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    coordinate_path = candidate_consensus_dir(output_root) / "consensus_scatter_coordinates.json"
    case_ids = []
    coords = []
    if coordinate_path.exists():
        for row in json.loads(coordinate_path.read_text(encoding="utf-8")):
            case_id = str(dict(row).get("case_id", "") or "")
            if case_id in labels:
                case_ids.append(case_id)
                coords.append([float(dict(row).get("x", 0.0) or 0.0), float(dict(row).get("y", 0.0) or 0.0)])
    if not coords:
        matrix, patient_ids = consensus_matrix_and_ids(output_root, cluster_states, labels)
        if matrix.size == 0:
            return ""
        case_ids = [case_id for case_id in patient_ids if case_id in labels]
        indices = [patient_ids.index(case_id) for case_id in case_ids]
        distance = 1.0 - np.clip(matrix[np.ix_(indices, indices)], 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        coords = mds_coordinates(distance).tolist()
    if len(coords) < 2:
        return ""
    coords_array = np.asarray(coords, dtype=float)
    plt = setup_review_figure_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    draw_subtype_points(ax, case_ids, coords_array, labels, subtype_palette(labels))
    ax.set_title("Integrated multimodal subtype landscape")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(loc="best", fontsize=6)
    return save_review_figure(fig, output_root, "integrated_snf_embedding_scatter")


def plot_modality_embedding_4panel(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    modality_specs = [("CT", "ct"), ("WSI", "wsi"), ("RNA", "rna"), ("WXS", "wxs")]
    panels = []
    for title, modality in modality_specs:
        case_ids, matrix = matrix_from_patient_vectors(patient_states, labels, modality)
        if matrix.size == 0 and modality == "rna":
            case_ids, _, matrix = matrix_from_feature_path(patient_states, labels, "rna_pathway_feature_path")
        if matrix.size == 0 and modality == "wxs":
            case_ids, _, matrix = matrix_from_feature_path(patient_states, labels, "wxs_full_feature_path")
        if matrix.size == 0 or len(case_ids) < 2:
            panels.append((title, [], np.zeros((0, 2), dtype=float)))
            continue
        coords = jaccard_mds_coordinates(matrix) if modality == "wxs" else pca_coordinates(matrix)
        coords = display_scale_coordinates(coords)
        panels.append((title, case_ids, coords))
    if not any(len(case_ids) >= 2 for _, case_ids, _ in panels):
        return ""
    plt = setup_review_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6), sharex=False, sharey=False)
    palette = subtype_palette(labels)
    for ax, (title, case_ids, coords) in zip(axes.ravel(), panels):
        if len(case_ids) >= 2:
            draw_subtype_points(ax, case_ids, coords, labels, palette)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[item], markersize=5, label=item)
        for item in sorted(set(labels.values()))
    ]
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return save_review_figure(fig, output_root, "modality_embedding_4panel")


def top_metric_rows(rows: list[dict[str, Any]], name_key: str, limit: int = 20) -> list[dict[str, Any]]:
    unique = {}
    for row in rows:
        key = (str(row.get("candidate_set_id", "") or ""), str(row.get(name_key, "") or ""))
        if not key[1]:
            continue
        current = unique.get(key)
        if current is None or (number_or_none(row.get("q_value")) or 1.0) < (number_or_none(current.get("q_value")) or 1.0):
            unique[key] = row
    sorted_rows = sorted(
        unique.values(),
        key=lambda row: (
            number_or_none(row.get("q_value")) if number_or_none(row.get("q_value")) is not None else 1.0,
            -abs(number_or_none(row.get("standardized_mean_difference")) or number_or_none(row.get("delta_mean_score")) or number_or_none(row.get("delta_frequency")) or 0.0),
            str(row.get(name_key, "") or ""),
        ),
    )
    return sorted_rows[:limit]


def top_metric_rows_per_subtype(
    rows: list[dict[str, Any]],
    name_key: str,
    subtype_ids: list[str],
    per_subtype: int = 5,
    q_value_max: float | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for subtype_id in subtype_ids:
        subtype_rows = [
            row
            for row in rows
            if str(row.get("candidate_set_id", "") or "") == subtype_id
            and str(row.get(name_key, "") or "")
        ]
        if q_value_max is not None:
            filtered = [
                row
                for row in subtype_rows
                if (number_or_none(row.get("q_value")) or 1.0) <= q_value_max
            ]
            if filtered:
                subtype_rows = filtered
        selected.extend(top_metric_rows(subtype_rows, name_key, per_subtype))
    seen = set()
    unique = []
    for row in selected:
        key = (str(row.get("candidate_set_id", "") or ""), str(row.get(name_key, "") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def plot_bubble_rows(
    output_root: str,
    stem: str,
    title: str,
    rows: list[dict[str, Any]],
    name_key: str,
    effect_key: str,
    labels: Mapping[str, str] | None = None,
    limit: int = 20,
) -> str:
    rows = top_metric_rows(rows, name_key, limit)
    if not rows:
        return ""
    x_items = sorted({str(row.get("candidate_set_id", "") or "") for row in rows})
    y_items = list(dict.fromkeys(wrapped_feature_label(row.get(name_key), 34) for row in rows))
    if not x_items or not y_items:
        return ""
    effects = np.asarray([number_or_none(row.get(effect_key)) or 0.0 for row in rows], dtype=float)
    max_abs = max(float(np.max(np.abs(effects))), 1e-6)
    plt = setup_review_figure_style()
    max_label_lines = max(label.count("\n") + 1 for label in y_items)
    fig, ax = plt.subplots(figsize=(max(5.8, len(x_items) * 0.9), max(4.2, len(y_items) * 0.18 * max_label_lines)))
    for row in rows:
        x = x_items.index(str(row.get("candidate_set_id", "") or ""))
        y = y_items.index(wrapped_feature_label(row.get(name_key), 34))
        q_value = number_or_none(row.get("q_value"))
        strength = -math.log10(max(q_value or 1.0, 1e-12))
        effect = number_or_none(row.get(effect_key)) or 0.0
        ax.scatter(
            x,
            y,
            s=24 + min(strength, 6.0) * 22,
            c=[effect],
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
            edgecolor="black",
            linewidth=0.25,
        )
    ax.axvline(-0.5, color="none")
    ax.set_xticks(range(len(x_items)))
    ax.set_xticklabels(x_items, rotation=45, ha="right")
    if labels:
        palette = subtype_palette(labels)
        for tick in ax.get_xticklabels():
            tick.set_color(palette.get(tick.get_text(), "black"))
            tick.set_fontweight("bold")
    for index in range(len(x_items) - 1):
        ax.axvline(index + 0.5, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(y_items)))
    ax.set_yticklabels(y_items, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, pad=10)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.5)
    mappable = ax.collections[-1]
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(effect_key)
    size_handles = []
    for q_value in (0.05, 0.01, 0.001):
        strength = -math.log10(q_value)
        size_handles.append(
            ax.scatter(
                [],
                [],
                s=24 + min(strength, 6.0) * 22,
                facecolor="white",
                edgecolor="black",
                linewidth=0.35,
                label=f"q={q_value:g}",
            )
        )
    ax.legend(
        handles=size_handles,
        title="Significance",
        loc="upper left",
        bbox_to_anchor=(1.18, 1.0),
        borderaxespad=0,
        labelspacing=0.8,
        handletextpad=0.8,
        fontsize=6,
        title_fontsize=7,
    )
    fig.tight_layout()
    return save_review_figure(fig, output_root, stem)


def pathway_collection_rows(rows: list[dict[str, Any]], collection_keyword: str = "", role: str = "") -> list[dict[str, Any]]:
    selected = []
    collection_keyword = collection_keyword.upper()
    role = role.lower()
    for row in rows:
        collection = str(row.get("gene_set_collection", "") or "").upper()
        evidence_role = str(row.get("evidence_role", "") or "").lower()
        if collection_keyword and collection_keyword not in collection:
            continue
        if role and evidence_role != role:
            continue
        selected.append(row)
    return selected


def plot_rna_hallmark_ssgsea_bubble(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    return plot_bubble_rows(
        output_root,
        "rna_hallmark_ssgsea_bubble",
        "RNA Hallmark ssGSEA shifts",
        biological_metric_rows(cluster_states, "rna_pathway_enrichment"),
        "pathway",
        "delta_mean_score",
        labels,
    )


def plot_rna_subtype_signature_dotplot(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    subtype_ids = sorted(set(labels.values()))
    rows = top_metric_rows_per_subtype(
        biological_metric_rows(cluster_states, "rna_pathway_enrichment"),
        "pathway",
        subtype_ids,
        per_subtype=5,
        q_value_max=0.05,
    )
    return plot_bubble_rows(
        output_root,
        "rna_subtype_signature_dotplot",
        "Subtype-associated Hallmark pathways",
        rows,
        "pathway",
        "delta_mean_score",
        labels,
        limit=max(1, len(rows)),
    )


def plot_rna_subtype_defining_hallmark_heatmap(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    case_ids, features, matrix = matrix_from_rna_pathway_scores(output_root, patient_states, labels)
    rows = top_metric_rows_per_subtype(
        biological_metric_rows(cluster_states, "rna_pathway_enrichment"),
        "pathway",
        sorted(set(labels.values())),
        per_subtype=6,
        q_value_max=0.05,
    )
    if matrix.size == 0 or not rows:
        return ""
    feature_index = {feature: index for index, feature in enumerate(features)}
    selected_rows = [row for row in rows if str(row.get("pathway", "") or "") in feature_index]
    if not selected_rows:
        return ""
    order_ids = ordered_case_ids(labels, case_ids)
    order = [case_ids.index(case_id) for case_id in order_ids if case_id in case_ids]
    ordered_ids = [case_id for case_id in order_ids if case_id in case_ids]
    row_indices = [feature_index[str(row.get("pathway", "") or "")] for row in selected_rows]
    values = robust_scale_matrix(matrix[:, row_indices]).T[:, order]
    y_labels = [wrapped_feature_label(row.get("pathway"), 28) for row in selected_rows]
    row_subtypes = [str(row.get("candidate_set_id", "") or "") for row in selected_rows]
    plt = setup_review_figure_style()
    fig = plt.figure(figsize=(max(6.2, len(ordered_ids) * 0.16), max(4.2, len(y_labels) * 0.28)))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.22, max(3.4, len(y_labels) * 0.28)],
        width_ratios=[5.0, 0.24],
        hspace=0.08,
        wspace=0.04,
    )
    ann_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    colorbar_ax = fig.add_subplot(grid[1, 1])
    palette = subtype_palette(labels)
    draw_subtype_annotation(ann_ax, ordered_ids, labels, palette)
    ann_ax.set_title("Subtype-defining Hallmark activity", pad=8)
    for index in range(len(row_subtypes) - 1):
        if row_subtypes[index] != row_subtypes[index + 1]:
            ax.axhline(index + 0.5, color="black", linewidth=0.5)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5, interpolation="nearest")
    draw_subtype_boundaries(ax, ordered_ids, labels)
    ax.set_xticks([])
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=6.5)
    for tick, subtype_id in zip(ax.get_yticklabels(), row_subtypes):
        tick.set_color(palette.get(subtype_id, "black"))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Robust z-score")
    return save_review_figure(fig, output_root, "rna_subtype_defining_hallmark_heatmap")


def plot_rna_top_pathway_boxplots(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    subtype_ids = sorted(set(labels.values()))
    rows = top_metric_rows_per_subtype(
        biological_metric_rows(cluster_states, "rna_pathway_enrichment"),
        "pathway",
        subtype_ids,
        per_subtype=2,
        q_value_max=0.05,
    )
    case_ids, features, matrix = matrix_from_rna_pathway_scores(output_root, patient_states, labels)
    if matrix.size == 0 or not rows:
        return ""
    feature_index = {feature: index for index, feature in enumerate(features)}
    pathways = list(dict.fromkeys(str(row.get("pathway", "") or "") for row in rows if str(row.get("pathway", "") or "") in feature_index))
    if not pathways:
        return ""
    plt = setup_review_figure_style()
    ncols = min(3, len(pathways))
    nrows = int(math.ceil(len(pathways) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(6.0, ncols * 2.2), max(2.4, nrows * 2.1)), squeeze=False)
    palette = subtype_palette(labels)
    grouped_indices = {
        subtype_id: [index for index, case_id in enumerate(case_ids) if labels.get(case_id) == subtype_id]
        for subtype_id in subtype_ids
    }
    for ax, pathway in zip(axes.ravel(), pathways):
        column = matrix[:, feature_index[pathway]]
        groups = [column[grouped_indices[subtype_id]] for subtype_id in subtype_ids]
        boxes = ax.boxplot(groups, tick_labels=subtype_ids, widths=0.55, showfliers=False, patch_artist=True)
        for patch, subtype_id in zip(boxes["boxes"], subtype_ids):
            patch.set_facecolor(palette.get(subtype_id, "#cccccc"))
            patch.set_alpha(0.35)
            patch.set_edgecolor("black")
        for x_pos, (subtype_id, group) in enumerate(zip(subtype_ids, groups), start=1):
            ax.scatter([x_pos] * len(group), group, s=10, color=palette.get(subtype_id), alpha=0.75, edgecolors="none")
        ax.set_title(wrapped_feature_label(pathway, 22), fontsize=7)
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.ravel()[len(pathways):]:
        ax.axis("off")
    fig.suptitle("Top subtype-associated Hallmark scores", y=1.02, fontsize=8)
    fig.tight_layout()
    return save_review_figure(fig, output_root, "rna_top_pathway_boxplots")


def plot_mutation_kegg_pathway_bubble(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    rows = pathway_collection_rows(
        biological_metric_rows(cluster_states, "wxs_pathway_enrichment"),
        "KEGG",
        "primary",
    )
    return plot_bubble_rows(
        output_root,
        "mutation_kegg_pathway_bubble",
        "Mutation KEGG pathway enrichment",
        rows,
        "pathway",
        "delta_frequency",
        labels,
    )


def plot_mutation_reactome_pathway_bubble(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    rows = pathway_collection_rows(
        biological_metric_rows(cluster_states, "wxs_pathway_enrichment"),
        "REACTOME",
        "validation",
    )
    return plot_bubble_rows(
        output_root,
        "mutation_reactome_pathway_bubble",
        "Mutation Reactome validation enrichment",
        rows,
        "pathway",
        "delta_frequency",
        labels,
    )


def plot_mutation_pathway_bubble(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    return plot_mutation_kegg_pathway_bubble(output_root, cluster_states, patient_states, labels)


def plot_rna_hallmark_ssgsea_heatmap(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    case_ids, features, matrix = matrix_from_rna_pathway_scores(output_root, patient_states, labels)
    if matrix.size == 0:
        case_ids, matrix = matrix_from_patient_vectors(patient_states, labels, "rna")
        features = [f"RNA_{index + 1}" for index in range(matrix.shape[1])]
    if matrix.size == 0 or len(case_ids) < 2:
        return ""
    variances = np.nanvar(matrix, axis=0)
    order_features = np.argsort(-variances)[: min(30, matrix.shape[1])]
    matrix = robust_scale_matrix(matrix[:, order_features]).T
    features = [short_feature_label(features[index]) for index in order_features]
    order_ids = ordered_case_ids(labels, case_ids)
    order = [case_ids.index(case_id) for case_id in order_ids]
    plt = setup_review_figure_style()
    fig = plt.figure(figsize=(max(5.8, len(order_ids) * 0.17), max(3.8, len(features) * 0.18)))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.2, max(3.0, len(features) * 0.18)], width_ratios=[4.8, 0.22], hspace=0.08, wspace=0.04)
    ann_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    colorbar_ax = fig.add_subplot(grid[1, 1])
    draw_subtype_annotation(ann_ax, order_ids, labels, subtype_palette(labels))
    ann_ax.set_title("RNA Hallmark ssGSEA activity", pad=8)
    image = ax.imshow(matrix[:, order], aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5, interpolation="nearest")
    draw_subtype_boundaries(ax, order_ids, labels)
    ax.set_xticks([])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Robust z-score")
    return save_review_figure(fig, output_root, "rna_hallmark_ssgsea_heatmap")


def plot_mutation_gene_oncoplot(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    rows = top_metric_rows(biological_metric_rows(cluster_states, "wxs_gene_enrichment"), "gene", 20)
    genes = list(dict.fromkeys(str(row.get("gene", "") or "") for row in rows if str(row.get("gene", "") or "")))
    case_ids, features, matrix = matrix_from_feature_path(patient_states, labels, "wxs_full_feature_path")
    if not genes or matrix.size == 0:
        return ""
    feature_index = {feature: index for index, feature in enumerate(features)}
    genes = [gene for gene in genes if gene in feature_index][:20]
    order_ids = ordered_case_ids(labels, case_ids)
    order = [case_ids.index(case_id) for case_id in order_ids if case_id in case_ids]
    gene_order = [feature_index[gene] for gene in genes]
    plot_matrix = (matrix[np.ix_(order, gene_order)].T > 0).astype(float)
    plt = setup_review_figure_style()
    ordered_labels = [order_ids[index] for index, case_id in enumerate(order_ids) if case_id in case_ids]
    fig, (ann_ax, ax) = plt.subplots(
        2,
        1,
        figsize=(max(5.0, len(order) * 0.15), max(3.0, len(genes) * 0.25)),
        gridspec_kw={"height_ratios": [0.2, max(2.4, len(genes) * 0.25)], "hspace": 0.08},
    )
    draw_subtype_annotation(ann_ax, ordered_labels, labels, subtype_palette(labels))
    ann_ax.set_title("Mutation gene oncoplot", pad=8)
    ax.imshow(plot_matrix, aspect="auto", cmap=plt.get_cmap("Greens"), vmin=0, vmax=1, interpolation="nearest")
    draw_subtype_boundaries(ax, ordered_labels, labels)
    ax.set_xticks([])
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    return save_review_figure(fig, output_root, "mutation_gene_oncoplot")


def survival_records(
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> tuple[list[dict[str, Any]], int]:
    records = []
    clinical = clinical_table(patient_state_map(patient_states))
    for case_id, cluster_id in labels.items():
        row = dict(clinical.get(case_id, {}) or {})
        time = number_or_none(row.get("os_time"))
        event = row.get("os_event")
        if time is None or event is None:
            continue
        records.append(
            {
                "case_id": case_id,
                "candidate_set_id": cluster_id,
                "time": float(time),
                "event": int(bool(event)),
            }
        )
    return records, len(labels) - len(records)


def km_step_curve(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    rows = sorted(records, key=lambda row: float(row["time"]))
    if not rows:
        return [], []
    at_risk = len(rows)
    survival = 1.0
    x_values = [0.0]
    y_values = [1.0]
    for time in sorted({float(row["time"]) for row in rows}):
        event_n = sum(1 for row in rows if float(row["time"]) == time and int(row["event"]) == 1)
        censor_n = sum(1 for row in rows if float(row["time"]) == time and int(row["event"]) == 0)
        if event_n and at_risk > 0:
            x_values.extend([time, time])
            y_values.extend([survival, survival * (1.0 - event_n / at_risk)])
            survival = y_values[-1]
        at_risk -= event_n + censor_n
    x_values.append(max(float(row["time"]) for row in rows))
    y_values.append(survival)
    return x_values, y_values


def evidence_strength_from_rows(rows: list[dict[str, Any]], subtype_id: str, effect_key: str) -> float:
    strengths = []
    for row in rows:
        if str(row.get("candidate_set_id", "") or "") != subtype_id:
            continue
        q_value = number_or_none(row.get("q_value"))
        effect = abs(number_or_none(row.get(effect_key)) or 0.0)
        if q_value is None or q_value > 0.1:
            continue
        strengths.append(min(-math.log10(max(min(q_value, 1.0), 1e-12)), 6.0) * effect)
    return max(strengths or [0.0])


def first_review_metrics(cluster_states: list[Mapping[str, Any]], block_name: str) -> dict[str, Any]:
    for cluster_state in cluster_states:
        matrix = review_matrix_for_cluster(cluster_state)
        metrics = dict(dict(matrix.get(block_name, {}) or {}).get("metrics", {}) or {})
        if metrics:
            return metrics
    return {}


def normalize_score_row(values: list[float]) -> list[float]:
    max_value = max([abs(value) for value in values] or [0.0])
    if max_value <= 0:
        return [0.0 for _ in values]
    return [float(value) / max_value for value in values]


def plot_subtype_evidence_score_panel(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    subtype_ids = sorted(set(labels.values()))
    if not subtype_ids:
        return ""
    set_metrics = first_review_metrics(cluster_states, "set_reliability")
    clinical_metrics = first_review_metrics(cluster_states, "clinical_context")
    consensus = dict(set_metrics.get("set_reliability_set_consensus", {}) or {})
    survival = dict(clinical_metrics.get("survival_set_association", {}) or {})
    rna_rows = biological_metric_rows(cluster_states, "rna_pathway_enrichment")
    gene_rows = biological_metric_rows(cluster_states, "wxs_gene_enrichment")
    pathway_rows = biological_metric_rows(cluster_states, "wxs_pathway_enrichment")
    raw_rows = [
        [
            number_or_none(dict(consensus.get(subtype_id, {}) or {}).get("silhouette_mean")) or 0.0
            for subtype_id in subtype_ids
        ],
        [
            evidence_strength_from_rows(rna_rows, subtype_id, "delta_mean_score")
            for subtype_id in subtype_ids
        ],
        [
            evidence_strength_from_rows(gene_rows, subtype_id, "delta_frequency")
            for subtype_id in subtype_ids
        ],
        [
            evidence_strength_from_rows(pathway_rows, subtype_id, "delta_frequency")
            for subtype_id in subtype_ids
        ],
        [
            min(-math.log10(max(number_or_none(dict(survival.get(subtype_id, {}) or {}).get("cox_p_value")) or 1.0, 1e-12)), 6.0)
            for subtype_id in subtype_ids
        ],
    ]
    if not any(abs(value) > 0 for row in raw_rows for value in row):
        return ""
    matrix = np.asarray([normalize_score_row(row) for row in raw_rows], dtype=float)
    if matrix.size == 0:
        return ""
    row_labels = ["Consensus", "RNA Hallmark", "Mutation genes", "Mutation pathways", "Survival"]
    plt = setup_review_figure_style()
    fig, ax = plt.subplots(figsize=(max(4.8, len(subtype_ids) * 0.9), 3.2))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(subtype_ids)))
    ax.set_xticklabels(subtype_ids, rotation=45, ha="right")
    palette = subtype_palette(labels)
    for tick in ax.get_xticklabels():
        tick.set_color(palette.get(tick.get_text(), "black"))
        tick.set_fontweight("bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for y_index in range(matrix.shape[0]):
        for x_index in range(matrix.shape[1]):
            ax.text(x_index, y_index, f"{matrix[y_index, x_index]:.2f}", ha="center", va="center", fontsize=6, color="black")
    ax.set_title("Subtype evidence strength overview", pad=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Row-normalized support")
    fig.tight_layout()
    return save_review_figure(fig, output_root, "subtype_evidence_score_panel")


def plot_survival_global_km(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    rows, _ = survival_records(patient_states, labels)
    if len(rows) < 2 or len({row["candidate_set_id"] for row in rows}) < 2 or sum(row["event"] for row in rows) < 1:
        return ""
    plt = setup_review_figure_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    palette = subtype_palette(labels)
    for cluster_id in sorted({row["candidate_set_id"] for row in rows}):
        cluster_rows = [row for row in rows if row["candidate_set_id"] == cluster_id]
        x_values, y_values = km_step_curve(cluster_rows)
        if x_values:
            ax.step(x_values, y_values, where="post", color=palette.get(cluster_id), linewidth=1.4, label=f"{cluster_id} (n={len(cluster_rows)})")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Days")
    ax.set_ylabel("Overall survival")
    ax.set_title("Overall survival by subtype")
    ax.legend(loc="best", fontsize=6)
    return save_review_figure(fig, output_root, "survival_global_km")


def association_heatmap_rows(cluster_states: list[Mapping[str, Any]], metric_group: str) -> list[dict[str, Any]]:
    rows = []
    for cluster_state in cluster_states:
        matrix = review_matrix_for_cluster(cluster_state)
        for block in matrix.values():
            metrics = dict(dict(block or {}).get("metrics", {}) or {})
            value = metrics.get(metric_group)
            if isinstance(value, Mapping):
                items = list(value.values())
            else:
                items = list(value or []) if isinstance(value, list) else []
            for item in items:
                if isinstance(item, Mapping):
                    row = dict(item)
                    row.setdefault("candidate_set_id", cluster_state.get("cluster_id", ""))
                    rows.append(row)
    return rows


def plot_association_heatmap(
    output_root: str,
    stem: str,
    title: str,
    rows: list[dict[str, Any]],
) -> str:
    if not rows:
        return ""
    cluster_ids = sorted({str(row.get("candidate_set_id", "") or "") for row in rows if str(row.get("candidate_set_id", "") or "")})
    fields = sorted({str(row.get("field", "") or row.get("label", "") or row.get("variable", "") or "") for row in rows})
    if not cluster_ids or not fields:
        return ""
    matrix = np.full((len(fields), len(cluster_ids)), np.nan, dtype=float)
    for row in rows:
        cluster_id = str(row.get("candidate_set_id", "") or "")
        field = str(row.get("field", "") or row.get("label", "") or row.get("variable", "") or "")
        q_value = number_or_none(row.get("q_value")) or number_or_none(row.get("p_value"))
        if cluster_id in cluster_ids and field in fields and q_value is not None:
            matrix[fields.index(field), cluster_ids.index(cluster_id)] = -math.log10(max(q_value, 1e-12))
    if not np.isfinite(matrix).any():
        return ""
    plt = setup_review_figure_style()
    fig, ax = plt.subplots(figsize=(max(4.2, len(cluster_ids) * 0.8), max(2.8, len(fields) * 0.22)))
    image = ax.imshow(np.nan_to_num(matrix, nan=0.0), aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels([short_feature_label(item) for item in fields])
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("-log10(q/p)")
    return save_review_figure(fig, output_root, stem)


def plot_confounder_association_heatmap(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    rows = association_heatmap_rows(cluster_states, "confounder_set_association")
    return plot_association_heatmap(output_root, "confounder_association_heatmap", "Confounder association", rows)


def plot_known_label_association_heatmap(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> str:
    rows = association_heatmap_rows(cluster_states, "per_label_enrichment")
    return plot_association_heatmap(output_root, "known_label_association_heatmap", "Known label association", rows)


def build_cluster_review_figures(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> None:
    for cluster_state in cluster_states:
        if not cluster_state_is_active(cluster_state):
            continue
        cluster_id = str(cluster_state.get("cluster_id", "") or "")
        if not cluster_id:
            continue
        figures = {}
        consensus_path = plot_cluster_member_consensus_support(output_root, cluster_state, labels)
        if consensus_path:
            figures["cluster_member_consensus_support"] = consensus_path
        pathway_path = plot_selected_pathway_boxplot(output_root, cluster_state, patient_states)
        if pathway_path:
            figures["selected_pathway_boxplot"] = pathway_path
        if figures:
            report = dict(cluster_state.get("report_draft", {}) or {})
            report_figures = dict(report.get("figures", {}) or {})
            block = dict(report_figures.get("set_reliability", {}) or {})
            if "cluster_member_consensus_support" in figures:
                block["cluster_member_consensus_support"] = figures["cluster_member_consensus_support"]
            report_figures["set_reliability"] = block
            bio = dict(report_figures.get("biological_support", {}) or {})
            if "selected_pathway_boxplot" in figures:
                bio["selected_pathway_boxplot"] = figures["selected_pathway_boxplot"]
            report_figures["biological_support"] = bio
            report["figures"] = report_figures
            if report:
                cluster_state["report_draft"] = report
            report_path = str(dict(cluster_state.get("artifacts", {}) or {}).get("report", "") or "")
            if report_path:
                write_json(report_path, report)


def plot_cluster_member_consensus_support(
    output_root: str,
    cluster_state: Mapping[str, Any],
    labels: Mapping[str, str],
) -> str:
    cluster_id = str(cluster_state.get("cluster_id", "") or "")
    members = [str(item) for item in list(cluster_state.get("member_ids", []) or [])]
    if not cluster_id or not members:
        return ""
    if not matplotlib_available():
        return save_fallback_cluster_figure(output_root, cluster_id, "cluster_member_consensus_support", 11)
    plt = setup_review_figure_style()
    fig, ax = plt.subplots(figsize=(max(3.8, len(members) * 0.24), 2.8))
    values = []
    matrix = review_matrix_for_cluster(cluster_state)
    reliability = dict(dict(matrix.get("set_reliability", {}) or {}).get("metrics", {}) or {})
    set_rows = dict(reliability.get("set_reliability_set_consensus", {}) or {})
    row = dict(set_rows.get(cluster_id, {}) or {})
    member_support = dict(row.get("per_member_consensus_support", {}) or {})
    for member in members:
        values.append(number_or_none(member_support.get(member)) or 0.0)
    if not any(values):
        values = [1.0 for _ in members]
    ax.bar(range(len(members)), values, color="#4E79A7", width=0.7)
    ax.set_ylim(0, max(1.0, max(values) * 1.1))
    ax.set_xticks(range(len(members)))
    ax.set_xticklabels(members, rotation=90)
    ax.set_ylabel("Consensus support")
    ax.set_title(f"{cluster_id} member support")
    return save_review_cluster_figure(fig, output_root, cluster_id, "cluster_member_consensus_support")


def save_review_cluster_figure(fig, output_root: str, cluster_id: str, stem: str) -> str:
    png_path = cluster_figure_output_path(output_root, cluster_id, f"{stem}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(png_path)


def plot_selected_pathway_boxplot(
    output_root: str,
    cluster_state: Mapping[str, Any],
    patient_states: list[Mapping[str, Any]] | None,
) -> str:
    cluster_id = str(cluster_state.get("cluster_id", "") or "")
    matrix = review_matrix_for_cluster(cluster_state)
    metrics = dict(dict(matrix.get("biological_support", {}) or {}).get("metrics", {}) or {})
    rows = top_metric_rows(
        metric_preview_rows(metrics.get("rna_pathway_enrichment")),
        "pathway",
        5,
    )
    if not cluster_id or not rows:
        return ""
    if not matplotlib_available():
        return save_fallback_cluster_figure(output_root, cluster_id, "selected_pathway_boxplot", 12)
    labels = {
        str(case_id): ("set" if str(case_id) in {str(item) for item in list(cluster_state.get("member_ids", []) or [])} else "rest")
        for case_id in patient_state_map(patient_states)
    }
    case_ids, features, values = matrix_from_rna_pathway_scores(output_root, patient_states, labels)
    if values.size == 0:
        return save_fallback_cluster_figure(output_root, cluster_id, "selected_pathway_boxplot", 12)
    feature_index = {feature: index for index, feature in enumerate(features)}
    pathways = [str(row.get("pathway", "") or "") for row in rows if str(row.get("pathway", "") or "") in feature_index]
    if not pathways:
        return ""
    plt = setup_review_figure_style()
    fig, axes = plt.subplots(1, len(pathways), figsize=(max(4.5, len(pathways) * 1.4), 2.8), squeeze=False)
    members = {str(item) for item in list(cluster_state.get("member_ids", []) or [])}
    for ax, pathway in zip(axes.ravel(), pathways):
        column = values[:, feature_index[pathway]]
        set_values = [column[index] for index, case_id in enumerate(case_ids) if case_id in members]
        rest_values = [column[index] for index, case_id in enumerate(case_ids) if case_id not in members]
        ax.boxplot([set_values, rest_values], tick_labels=[cluster_id, "Rest"], widths=0.55, showfliers=False)
        for x_pos, group in enumerate([set_values, rest_values], start=1):
            ax.scatter([x_pos] * len(group), group, s=12, color="#4E79A7" if x_pos == 1 else "#BAB0AC", alpha=0.8)
        ax.set_title(short_feature_label(pathway, 24))
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return save_review_cluster_figure(fig, output_root, cluster_id, "selected_pathway_boxplot")


def review_matrix_for_cluster(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(cluster_state.get("report_draft", {}) or {})
    return dict(
        report.get("final_evidence_matrix")
        or cluster_state.get("evidence_matrix")
        or {}
    )


def filter_confounder_metrics_for_review(
    metrics: Mapping[str, Any],
    config_dir: str = "",
) -> dict[str, Any]:
    filtered = dict(metrics or {})
    if (
        "confounder_global_association" in filtered
        or "confounder_set_association" in filtered
    ):
        return {
            "confounder_global_association": dict(
                filtered.get("confounder_global_association", {}) or {}
            ),
            "confounder_set_association": dict(
                filtered.get("confounder_set_association", {}) or {}
            ),
        }
    raw_field_results = filtered.get("field_results", {})
    if not raw_field_results:
        return filtered
    field_sets = subtype_review_confounder_fields(config_dir)
    demographic_fields = field_sets["demographic"]
    technical_fields = field_sets["technical"]
    known_or_disease_fields = field_sets["known_or_disease_label"]
    items = (
        list(raw_field_results.values())
        if isinstance(raw_field_results, Mapping)
        else list(raw_field_results)
    )
    kept = []
    for item in items:
        row = dict(item or {})
        field = str(row.get("field", "") or "").lower()
        if (
            "missing" in field
            or "available" in field
            or field in demographic_fields
            or field in technical_fields
            or any(key in field for key in technical_fields)
        ) and field not in known_or_disease_fields:
            kept.append(row)
    filtered["field_results"] = kept
    return filtered


def save_subtype_review_json(
    output_root: str,
    cluster_state: Mapping[str, Any],
    filename: str,
    payload: Any,
) -> str:
    return write_json(
        ensure_dir(
            Path(output_root)
            / "subtype_review"
            / safe_identifier(str(cluster_state.get("cluster_id", "unknown_cluster")))
        )
        / filename,
        payload,
    )


def active_call_llm_json(*args, **kwargs):
    cluster_flow = sys.modules.get("utils.cluster_flow")
    caller = getattr(cluster_flow, "call_llm_json", None) if cluster_flow else None
    return (caller or call_llm_json)(*args, **kwargs)


def active_load_llm_client(config_dir: str):
    cluster_flow = sys.modules.get("utils.cluster_flow")
    loader = getattr(cluster_flow, "load_llm_client", None) if cluster_flow else None
    return (loader or load_llm_client)(config_dir)


def call_llm_json_with_token_budget(
    messages: list[dict[str, str]],
    llm_client: Any,
    max_tokens: Any,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    budget = int(max_tokens or 0)
    if budget <= 0 or not hasattr(llm_client, "max_tokens"):
        return active_call_llm_json(messages, llm_client, tools=tools)
    original = getattr(llm_client, "max_tokens")
    setattr(llm_client, "max_tokens", min(int(original), budget))
    try:
        return active_call_llm_json(messages, llm_client, tools=tools)
    finally:
        setattr(llm_client, "max_tokens", original)


def compact_tool_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    results = dict(payload.get("results", {}) or {})
    metrics = dict(results.get("metrics", {}) or {})
    tool_name = str(payload.get("tool_name", "") or "")
    if tool_name in {"tool_mutation_enrichment", "tool_pathway_enrichment"}:
        metrics = compact_biological_metrics_for_review(metrics)
    return {
        "tool_name": tool_name,
        "status": str(payload.get("status", "") or ""),
        "summary": str(results.get("summary", "") or ""),
        "support_level": str(results.get("support_level", "") or ""),
        "concern_level": str(results.get("concern_level", "") or ""),
        "metrics": metrics,
        "warnings": list(results.get("warnings", []) or []),
        "errors": list(payload.get("errors", []) or []),
        "figures": dict(
            dict(payload.get("artifacts", {}) or {}).get("figures", {}) or {}
        ),
    }


def compact_cluster_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    budget_state = dict(payload.get("budget_state", {}) or {})
    validation_results = dict(payload.get("validation_results", {}) or {})
    return {
        "cluster_id": str(payload.get("cluster_id", "") or ""),
        "member_ids": list(payload.get("member_ids", []) or []),
        "parent_cluster_ids": list(payload.get("parent_cluster_ids", []) or []),
        "status": str(payload.get("status", "") or ""),
        "final_decision": str(
            payload.get("final_decision", "") or payload.get("final_action", "") or ""
        ),
        "final_action": str(payload.get("final_action", "") or ""),
        "drop_reason": str(payload.get("drop_reason", "") or ""),
        "review_unavailable_reason": str(
            payload.get("review_unavailable_reason", "") or ""
        ),
        "insufficient_evidence_reason": str(
            payload.get("insufficient_evidence_reason", "") or ""
        ),
        "absorbed_into": str(payload.get("absorbed_into", "") or ""),
        "absorbed_by": str(payload.get("absorbed_by", "") or ""),
        "review_round": int(payload.get("review_round", 0) or 0),
        "revision_round": int(payload.get("revision_round", 0) or 0),
        "source_views": list(payload.get("source_views", []) or []),
        "consensus": dict(payload.get("consensus", {}) or {}),
        "generator": dict(payload.get("generator", {}) or {}),
        "tool_names": sorted(validation_results.keys()),
        "structured_evidence": list(payload.get("structured_evidence", []) or []),
        "planner_decision": dict(payload.get("planner_decision", {}) or {}),
        "decider_decision": dict(payload.get("decider_decision", {}) or {}),
        "verifier_decision": dict(payload.get("verifier_decision", {}) or {}),
        "router_decision": dict(payload.get("router_decision", {}) or {}),
        "round_summaries": list(payload.get("round_summaries", []) or []),
        "action_history": list(payload.get("action_history", []) or []),
        "budget_state": {
            "review_rounds_used": int(budget_state.get("review_rounds_used", 0) or 0),
            "tool_calls_used": int(budget_state.get("tool_calls_used", 0) or 0),
            "revisions_used": int(budget_state.get("revisions_used", 0) or 0),
            "budget_exhausted": bool(budget_state.get("budget_exhausted", False)),
            "budget_exhausted_reason": str(
                budget_state.get("budget_exhausted_reason", "") or ""
            ),
        },
        "revision_history": list(payload.get("revision_history", []) or []),
        "decision_basis": list(payload.get("decision_basis", []) or []),
        "report": dict(payload.get("report_draft", {}) or {}),
        "artifacts": dict(payload.get("artifacts", {}) or {}),
    }


def load_subtype_review_config(config_dir: str) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser() if config_dir else Path("configs")
    return load_yaml_file(config_path / "subtype_review.yaml")


def subtype_review_artifact_policy(config_dir: str) -> dict[str, Any]:
    config = load_subtype_review_config(config_dir)
    policy = {
        "mode": "key_only",
        "save_debug_rounds": False,
        "save_individual_tool_json": False,
        "save_llm_audit": False,
        "save_figures": True,
    }
    policy.update(dict(config.get("artifact_policy", {}) or {}))
    return policy


def save_debug_artifacts(config_dir: str) -> bool:
    policy = subtype_review_artifact_policy(config_dir)
    return str(policy.get("mode", "key_only")) == "debug" or bool(
        policy.get("save_debug_rounds", False)
    )


def load_prompt(config_dir: str, config: Mapping[str, Any], name: str) -> str:
    prompt_dir = Path(str(config["prompt_dir"])).expanduser()
    if not prompt_dir.is_absolute():
        prompt_dir = (
            Path(config_dir).expanduser().parent / prompt_dir
            if config_dir
            else prompt_dir
        ).resolve()
    return (prompt_dir / name).read_text(encoding="utf-8").strip()


def _split_unified_prompt(raw: str) -> tuple[str, str]:
    """Parse a unified prompt file with ===SYSTEM=== and ===USER=== markers."""
    system_content = ""
    user_content = ""
    current = ""
    for line in raw.split("\n"):
        if line.startswith("===SYSTEM==="):
            current = "system"
            continue
        if line.startswith("===USER==="):
            current = "user"
            continue
        if current == "system":
            system_content += line + "\n"
        elif current == "user":
            user_content += line + "\n"
    return system_content.strip(), user_content.strip()


def review_messages(
    config_dir: str, config: Mapping[str, Any], payload: Mapping[str, Any]
) -> list[dict[str, str]]:
    unified = load_prompt(config_dir, config, "subtype_review.md")
    system_content, user_content = _split_unified_prompt(unified)
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": user_content.replace(
                "{input_json}",
                json.dumps(to_jsonable(dict(payload)), ensure_ascii=False, indent=2),
            ),
        },
    ]


def report_messages(
    config_dir: str, config: Mapping[str, Any], payload: Mapping[str, Any]
) -> list[dict[str, str]]:
    unified = load_prompt(config_dir, config, "cluster_report.md")
    system_content, user_content = _split_unified_prompt(unified)
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": user_content.replace(
                "{input_json}",
                json.dumps(to_jsonable(dict(payload)), ensure_ascii=False, indent=2),
            ),
        },
    ]


def action_decision_messages(
    config_dir: str, config: Mapping[str, Any], payload: Mapping[str, Any]
) -> list[dict[str, str]]:
    unified = load_prompt(config_dir, config, "action_decision.md")
    system_content, user_content = _split_unified_prompt(unified)
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": user_content.replace(
                "{input_json}",
                json.dumps(to_jsonable(dict(payload)), ensure_ascii=False, indent=2),
            ),
        },
    ]


def load_subtype_review_budget(config_dir: str) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser() if config_dir else Path("configs")
    main_config = load_yaml_file(config_path / "subtype_review.yaml")
    budget_config = load_yaml_file(config_path / "subtype_review_budget.yaml")
    budget = {
        "max_rounds": 3,
        "max_tool_calls": 8,
        "max_failures": 2,
        "max_revisions": 3,
        "max_split_depth": 2,
        "max_merge_attempts": 2,
    }
    budget.update(dict(budget_config.get("budget", {}) or {}))
    budget.update(dict(main_config.get("budget", {}) or {}))
    return {"budget": budget}


def allowed_review_tools(config_dir: str) -> set[str]:
    try:
        config = load_subtype_review_config(config_dir)
    except Exception:
        return set()
    tool_definitions = subtype_review_tool_definitions(config_dir)
    return {
        str(item)
        for item in list(config.get("allowed_tools", []) or [])
        if str(item) in tool_definitions
    }


def review_tool_schemas(config_dir: str) -> list[dict[str, Any]]:
    allowed = allowed_review_tools(config_dir)
    schemas = []
    for tool_name in subtype_review_tool_definitions(config_dir):
        if tool_name not in allowed:
            continue
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": (
                        f"Run {tool_name} to supplement candidate subtype review evidence."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "enum": [tool_name]},
                            "reason": {"type": "string"},
                        },
                        "required": ["tool_name"],
                    },
                },
            }
        )
    return schemas


def filtered_tool_plan(tool_plan: list[Any], config_dir: str) -> list[dict[str, Any]]:
    allowed = allowed_review_tools(config_dir)
    filtered = []
    for item in tool_plan:
        item = {"tool_name": str(item)} if isinstance(item, str) else dict(item)
        tool_name = str(item.get("tool_name", item.get("name", "")) or "")
        if tool_name in allowed:
            filtered.append(
                {
                    "tool_name": tool_name,
                    **{key: value for key, value in item.items() if key != "name"},
                }
            )
    return filtered


def init_review_budget_state(cluster_state: dict[str, Any]) -> dict[str, Any]:
    budget_state = dict(DEFAULT_BUDGET_STATE)
    budget_state.update(dict(cluster_state.get("budget_state", {}) or {}))
    cluster_state["budget_state"] = budget_state
    return cluster_state


def update_review_budget_after_revision(
    cluster_state: dict[str, Any],
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cluster_state = init_review_budget_state(cluster_state)
    action = str(cluster_state.get("next_action", "") or "")
    budget_state = cluster_state["budget_state"]
    counter_key = {
        "call_tools": "supplement_rounds_used",
        "split": "split_plans_used",
        "merge": "merge_plans_used",
    }.get(action)
    if counter_key:
        budget_state[counter_key] = int(budget_state.get(counter_key, 0)) + 1
    if action in {"split", "merge"}:
        budget_state["revisions_used"] = int(budget_state.get("revisions_used", 0)) + 1
    if tool_results is not None:
        budget_state["tool_calls_used"] = int(
            budget_state.get("tool_calls_used", 0)
        ) + len(tool_results)
        budget_state["tool_failures"] = int(budget_state.get("tool_failures", 0)) + sum(
            1 for item in tool_results if str(item.get("status", "")) == "failure"
        )
        has_new_evidence = any(
            list(dict(item.get("results", {}) or {}).get("evidence_hints", []) or [])
            for item in tool_results
        )
        budget_state["no_new_evidence_rounds"] = (
            0
            if has_new_evidence
            else int(budget_state.get("no_new_evidence_rounds", 0)) + 1
        )
    return cluster_state


def normalize_review_action(action: str) -> str:
    action = str(action or "").strip().lower()
    if action == "reject":
        return "drop"
    return action if action in FINAL_REVIEW_ACTIONS else "drop"


def force_drop(cluster_state: dict[str, Any], reason: str) -> dict[str, Any]:
    cluster_state["next_action"] = "drop"
    cluster_state["final_action"] = "drop"
    cluster_state["final_decision"] = "drop"
    cluster_state["status"] = "drop"
    cluster_state["drop_reason"] = str(reason or "drop")
    return cluster_state


def revision_budget_exhausted(
    cluster_state: Mapping[str, Any],
    action: str,
    budget: Mapping[str, Any],
) -> str:
    budget_state = dict(cluster_state.get("budget_state", {}) or {})
    revision_count = max(
        int(budget_state.get("revisions_used", 0)),
        int(budget_state.get("split_plans_used", 0))
        + int(budget_state.get("merge_plans_used", 0)),
    )
    if revision_count >= int(budget["max_revisions"]):
        return "max_revisions"
    if action == "split" and int(budget_state.get("split_plans_used", 0)) >= int(
        budget["max_split_depth"]
    ):
        return "max_split_depth"
    if action == "merge" and int(budget_state.get("merge_plans_used", 0)) >= int(
        budget["max_merge_attempts"]
    ):
        return "max_merge_attempts"
    return ""


def split_cluster_members(
    member_ids: list[str], split_plan: Mapping[str, Any]
) -> list[str]:
    for key in ("selected_member_ids", "member_ids", "members"):
        planned = [str(item) for item in list(split_plan.get(key, []) or [])]
        selected = [item for item in planned if item in member_ids]
        if selected:
            return selected
    for group in list(
        split_plan.get("subclusters", []) or split_plan.get("groups", []) or []
    ):
        planned = [
            str(item)
            for item in list(
                dict(group).get("member_ids", [])
                or dict(group).get("members", [])
                or []
            )
        ]
        selected = [item for item in planned if item in member_ids]
        if selected:
            return selected
    midpoint = max(1, len(member_ids) // 2)
    return member_ids[:midpoint]


def split_cluster_specs_from_consensus(
    cluster_state: Mapping[str, Any],
    source_cluster_id: str,
) -> list[dict[str, Any]]:
    member_ids = [str(item) for item in list(cluster_state.get("member_ids", []) or [])]
    if len(member_ids) < 2:
        return []
    consensus = dict(cluster_state.get("consensus", {}) or {})
    matrix_payload = consensus.get("matrix")
    matrix = None
    if matrix_payload is not None:
        matrix = np.asarray(matrix_payload, dtype=float)
    elif str(consensus.get("matrix_path", "") or ""):
        path = Path(str(consensus.get("matrix_path")))
        if path.exists():
            matrix = np.asarray(np.load(path), dtype=float)
    patient_ids = [str(item) for item in list(consensus.get("patient_ids", []) or [])]
    if not patient_ids and str(consensus.get("metadata_path", "") or ""):
        try:
            metadata = json.loads(Path(str(consensus["metadata_path"])).read_text(encoding="utf-8"))
            patient_ids = [str(item) for item in list(metadata.get("patient_ids", []) or [])]
        except Exception:
            patient_ids = []
    if matrix is None:
        return []
    if not patient_ids and matrix.shape[0] == len(member_ids):
        patient_ids = member_ids
    index_by_patient = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    selected = [member_id for member_id in member_ids if member_id in index_by_patient]
    if len(selected) < 2:
        return []
    indices = [index_by_patient[member_id] for member_id in selected]
    submatrix = np.asarray(matrix, dtype=float)[np.ix_(indices, indices)]
    distance = 1.0 - np.clip(np.nan_to_num(submatrix, nan=0.0), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    try:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance)
    except Exception:
        midpoint = max(1, len(selected) // 2)
        labels = np.asarray([0 if i < midpoint else 1 for i in range(len(selected))])
    specs = []
    for label in sorted(set(np.asarray(labels, dtype=int).tolist())):
        child_members = [
            selected[index]
            for index, value in enumerate(np.asarray(labels, dtype=int).tolist())
            if value == label
        ]
        if child_members:
            specs.append(
                {
                    "cluster_id": f"{source_cluster_id}_S{len(specs) + 1}",
                    "member_ids": child_members,
                }
            )
    return specs if len(specs) >= 2 else []


def merge_cluster_members(
    member_ids: list[str],
    merge_plan: Mapping[str, Any],
    all_cluster_states: list[dict[str, Any]],
) -> list[str]:
    merged = list(member_ids)
    explicit_members = list(
        merge_plan.get("member_ids", []) or merge_plan.get("members", []) or []
    )
    for member_id in explicit_members:
        member_id = str(member_id)
        if member_id not in merged:
            merged.append(member_id)
    target_ids = {
        str(item)
        for item in list(
            merge_plan.get("target_cluster_ids", [])
            or merge_plan.get("merge_with", [])
            or []
        )
    }
    for cluster in all_cluster_states:
        if str(cluster.get("cluster_id", "") or "") not in target_ids:
            continue
        for member_id in [
            str(item) for item in list(cluster.get("member_ids", []) or [])
        ]:
            if member_id not in merged:
                merged.append(member_id)
    return merged


def build_structured_evidence(cluster_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    evidence: list[dict[str, Any]] = []
    validation_results = dict(cluster_state.get("validation_results", {}) or {})
    for tool_name, tool_payload in validation_results.items():
        result = dict(tool_payload or {})
        results = dict(result.get("results", {}) or {})
        hints = list(results.get("evidence_hints", []) or [])
        if not hints and str(result.get("status", "")) in {
            "missing",
            "not_implemented",
            "failure",
        }:
            hints = [
                {
                    "evidence_type": "limitation",
                    "summary": results.get("missing_reason")
                    or results.get("summary")
                    or f"{tool_name} did not produce usable evidence.",
                }
            ]
        for index, hint in enumerate(hints):
            evidence.append(
                {
                    "evidence_id": f"{cluster_id}_{tool_name}_{index}",
                    "evidence_type": str(
                        dict(hint).get("evidence_type", "tool_evidence")
                    ),
                    "source": str(tool_name),
                    "summary": str(
                        dict(hint).get("summary", "") or results.get("summary", "")
                    ),
                    "supporting_metrics": dict(results.get("metrics", {}) or {}),
                    "interpretation": str(dict(hint).get("interpretation", "") or ""),
                    "limitation": results.get("missing_reason") or None,
                }
            )
    verification_vector = dict(cluster_state.get("verification_vector", {}) or {})
    raw = dict(verification_vector.get("raw", {}) or {})
    if not evidence:
        evidence.append(
            {
                "evidence_id": f"{cluster_id}_verification_vector_0",
                "evidence_type": "limitation",
                "source": "verification_vector",
                "summary": "No cluster-level validation tool evidence is available yet.",
                "supporting_metrics": raw,
                "interpretation": "",
                "limitation": "cluster-level validation evidence missing",
            }
        )
    return evidence


def compact_validation_results_for_review(
    validation_results: Mapping[str, Any],
    config_dir: str,
) -> dict[str, Any]:
    compact_validation = {}
    for tool_name, tool_payload in dict(validation_results or {}).items():
        result = dict(tool_payload or {})
        results = dict(result.get("results", {}) or {})
        metrics = dict(results.get("metrics", {}) or {})
        if str(tool_name) == "tool_confound_test":
            metrics = filter_confounder_metrics_for_review(metrics, config_dir=config_dir)
        if str(tool_name) in {"tool_mutation_enrichment", "tool_pathway_enrichment"}:
            metrics = compact_biological_metrics_for_review(metrics)
        compact_validation[str(tool_name)] = {
            "status": str(result.get("status", "") or ""),
            "summary": str(results.get("summary", "") or ""),
            "support_level": str(results.get("support_level", "") or ""),
            "concern_level": str(results.get("concern_level", "") or ""),
            "metrics": metrics,
            "warnings": list(results.get("warnings", []) or []),
        }
    return compact_validation


def subtype_review_tool_capabilities(config_dir: str) -> list[dict[str, Any]]:
    definitions = subtype_review_tool_definitions(config_dir)
    allowed = allowed_review_tools(config_dir)
    return [
        {
            "tool_name": tool_name,
            "evidence_blocks": list(dict(definition).get("evidence_blocks", []) or []),
            "description": str(dict(definition).get("description", "") or ""),
        }
        for tool_name, definition in definitions.items()
        if tool_name in allowed
    ]


def compact_metric_for_decider(value: Any, cluster_id: str, row_limit: int = 8) -> Any:
    if isinstance(value, Mapping):
        data = dict(value)
        if "preview_rows" in data:
            rows = [
                dict(row)
                for row in list(data.get("preview_rows", []) or [])
                if not dict(row).get("candidate_set_id")
                or str(dict(row).get("candidate_set_id")) == cluster_id
            ]
            data["preview_rows"] = rows[:row_limit]
            data["preview_row_count"] = len(data["preview_rows"])
            data["omitted_row_count"] = max(
                int(data.get("row_count", len(rows)) or len(rows))
                - len(data["preview_rows"]),
                0,
            )
            return data
        if cluster_id in data and all(str(key).startswith("C") for key in data):
            return {cluster_id: compact_metric_for_decider(data[cluster_id], cluster_id, row_limit)}
        return {
            str(key): compact_metric_for_decider(item, cluster_id, row_limit)
            for key, item in data.items()
        }
    if isinstance(value, list):
        rows = [
            compact_metric_for_decider(item, cluster_id, row_limit)
            for item in value
            if not isinstance(item, Mapping)
            or not dict(item).get("candidate_set_id")
            or str(dict(item).get("candidate_set_id")) == cluster_id
        ]
        return rows[:row_limit]
    return value


def compact_evidence_matrix_for_decider(
    evidence_matrix: Mapping[str, Any],
    cluster_id: str = "",
) -> dict[str, Any]:
    return {
        name: {
            "metrics": compact_metric_for_decider(
                dict(dict(evidence_matrix.get(name, {}) or {}).get("metrics", {}) or {}),
                cluster_id,
            )
        }
        for name in EVIDENCE_BLOCK_NAMES
    }


def llm_evidence_audit(
    cluster_state: dict[str, Any],
    output_root: str,
    config_dir: str,
) -> dict[str, Any]:
    config = load_subtype_review_config(config_dir)
    validation_results = dict(cluster_state.get("validation_results", {}) or {})
    compact_validation = compact_validation_results_for_review(
        validation_results, config_dir
    )
    verification_vector = dict(cluster_state.get("verification_vector", {}) or {})
    tool_keys = {
        "stability",
        "multimodal_consistency",
        "survival",
        "mutation",
        "pathway",
        "confounder",
        "known_label_echo",
    }
    lightweight_verification_vector = {
        k: v for k, v in verification_vector.items() if k not in tool_keys
    }
    init_review_budget_state(cluster_state)
    budget_path = Path(config_dir).expanduser() / "subtype_review_budget.yaml"
    budget_config = load_subtype_review_budget(config_dir) if budget_path.exists() else {}
    budget = dict(
        dict(budget_config.get("budget", {}) or {})
        or {"max_tool_calls": 0}
    )
    budget_state = dict(cluster_state.get("budget_state", {}) or {})
    remaining_tools = max(
        int(budget.get("max_tool_calls", 0) or 0)
        - int(budget_state.get("tool_calls_used", 0) or 0),
        0,
    )
    payload = {
        "cluster_id": str(cluster_state.get("cluster_id", "")),
        "member_ids": list(cluster_state.get("member_ids", [])),
        "review_round": int(cluster_state.get("review_round", 0) or 0),
        "subtype_set_context": list(cluster_state.get("subtype_set_context", []) or []),
        "verification_vector": lightweight_verification_vector,
        "validation_results": compact_validation,
        "decider_requested_evidence": dict(
            cluster_state.get("decider_requested_evidence", {}) or {}
        ),
        "executed_tools": sorted(validation_results.keys()),
        "remaining_tool_budget": remaining_tools,
        "tool_capabilities": subtype_review_tool_capabilities(config_dir),
    }
    messages = review_messages(config_dir, config, payload)
    audit = call_llm_json_with_token_budget(
        messages,
        active_load_llm_client(config_dir),
        dict(config.get("llm", {}) or {}).get("evidence_audit_max_new_tokens"),
        tools=review_tool_schemas(config_dir),
    )
    tool_calls = list(audit.get("tool_calls", []) or [])
    allowed = allowed_review_tools(config_dir)
    tools_to_call_from_tool_calls = [
        {
            "tool_name": str(dict(item).get("name", "") or ""),
            **dict(dict(item).get("arguments", {}) or {}),
        }
        for item in tool_calls
        if str(dict(item).get("name", "") or "") in allowed
    ]
    parse_success = not bool(audit.get("_parse_error"))
    if not parse_success:
        audit = {
            "_parse_error": True,
            "cluster_id": payload["cluster_id"],
            "need_more_evidence": bool(tools_to_call_from_tool_calls),
            "uncertainties_blocking_convergence": []
            if tools_to_call_from_tool_calls
            else ["LLM planner JSON parsing failed."],
            "decision_sufficiency": {
                "is_sufficient": False,
                "sufficient_for": None,
                "why": "LLM requested validation tools."
                if tools_to_call_from_tool_calls
                else "LLM planner JSON parsing failed.",
                "missing_evidence_that_may_change_decision": [
                    item["tool_name"] for item in tools_to_call_from_tool_calls
                ],
                "missing_evidence_only_as_limitation": [],
            },
            "recommended_action": "",
            "tools_to_call": tools_to_call_from_tool_calls,
            "tool_plan": tools_to_call_from_tool_calls,
            "limitations_to_report": [],
            "reasoning_summary": "LLM requested validation tools."
            if tools_to_call_from_tool_calls
            else "LLM planner JSON parsing failed.",
            "raw_output": str(audit.get("raw_output", "")),
            "tool_calls": tool_calls,
        }
    if tool_calls and not list(audit.get("tools_to_call", []) or []):
        audit["tools_to_call"] = [
            {
                "tool_name": str(dict(item).get("name", "") or ""),
                **dict(dict(item).get("arguments", {}) or {}),
            }
            for item in tool_calls
            if str(dict(item).get("name", "") or "") in allowed
        ]
    audit["tool_plan"] = filtered_tool_plan(
        list(audit.get("tool_plan", []) or audit.get("tools_to_call", []) or []),
        config_dir,
    )
    existing_tools = set(validation_results)
    audit["tool_plan"] = [
        item
        for item in list(audit.get("tool_plan", []) or [])
        if str(dict(item).get("tool_name", dict(item).get("name", "")) or "")
        not in existing_tools
    ]
    audit["tools_to_call"] = list(audit.get("tool_plan", []) or [])
    audit["decision_state"] = "continue_review"
    if bool(audit.get("_parse_error")):
        audit["tools_to_call"] = []
        audit["tool_plan"] = []
        audit["review_unavailable_reason"] = "llm_parse_failure"
    cluster_state["evidence_gap"] = list(
        audit.get("uncertainties_blocking_convergence", [])
        or audit.get("evidence_gap", [])
        or audit.get("missing_evidence", [])
        or []
    )
    cluster_state["tool_plan"] = list(audit.get("tool_plan", []) or [])
    cluster_state["planner_decision"] = audit
    cluster_state["llm_audit"] = audit
    cluster_state["conversation_history"] = list(
        cluster_state.get("conversation_history", []) or []
    ) + [
        messages[-1],
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "recommended_action": audit.get("recommended_action"),
                    "need_more_evidence": audit.get("need_more_evidence"),
                    "tools_to_call": audit.get("tools_to_call", []),
                    "reasoning_summary": audit.get("reasoning_summary", ""),
                    "tool_calls": audit.get("tool_calls", []),
                },
                ensure_ascii=False,
            ),
        },
    ]
    init_review_budget_state(cluster_state)
    budget_state = cluster_state["budget_state"]
    budget_state["llm_audit_calls_used"] = (
        int(budget_state.get("llm_audit_calls_used", 0)) + 1
    )
    if not parse_success:
        budget_state["llm_parse_failures"] = (
            int(budget_state.get("llm_parse_failures", 0)) + 1
        )
    if bool(
        subtype_review_artifact_policy(config_dir).get("save_llm_audit", False)
    ) or save_debug_artifacts(config_dir):
        artifacts = dict(cluster_state.get("artifacts", {}) or {})
        artifacts[f"llm_audit_round_{payload['review_round']}"] = (
            save_subtype_review_json(
                output_root,
                cluster_state,
                f"llm_audit_round_{payload['review_round']}.json",
                audit,
            )
        )
        cluster_state["artifacts"] = artifacts
    return audit


def previous_review_context(cluster_state: Mapping[str, Any]) -> dict[str, Any]:
    if int(cluster_state.get("review_round", 0) or 0) <= 1:
        return {}
    audit = dict(cluster_state.get("llm_audit", {}) or {})
    router = dict(cluster_state.get("router_decision", {}) or {})
    verifier = dict(cluster_state.get("verifier_decision", {}) or {})
    return {
        "previous_decision_state": audit.get("decision_state"),
        "previous_recommended_action": audit.get("recommended_action"),
        "previous_confidence_level": audit.get("confidence_level"),
        "previous_reason_codes": list(audit.get("reason_codes", []) or []),
        "previous_evidence_ids": list(audit.get("evidence_ids", []) or []),
        "previous_continue_review_reason": audit.get("continue_review_reason", ""),
        "previous_blocks_to_update": list(audit.get("blocks_to_update", []) or []),
        "previous_router_action": router.get("route_action"),
        "previous_router_reason": router.get("route_reason", ""),
        "previous_verifier_ready": verifier.get("final_action_ready"),
        "previous_contract_issues": list(
            verifier.get("decision_consistency_check", {}).get("issues", []) or []
        ),
    }


def llm_action_decision(
    cluster_state: dict[str, Any],
    output_root: str,
    config_dir: str,
) -> dict[str, Any]:
    """Decider: choose an action from current compact validation metrics."""
    config = load_subtype_review_config(config_dir)
    audit = dict(cluster_state.get("llm_audit", {}) or {})
    evidence_matrix = dict(cluster_state.get("evidence_matrix", {}) or {})
    payload = {
        "cluster_id": str(cluster_state.get("cluster_id", "")),
        "member_ids": list(cluster_state.get("member_ids", [])),
        "review_round": int(cluster_state.get("review_round", 0) or 0),
        "subtype_set_context": list(cluster_state.get("subtype_set_context", []) or []),
        "previous_review_context": previous_review_context(cluster_state),
        "executed_tools": sorted(dict(cluster_state.get("validation_results", {}) or {})),
        "evidence_matrix": compact_evidence_matrix_for_decider(
            evidence_matrix,
            str(cluster_state.get("cluster_id", "") or ""),
        ),
        "action_rules": {
            "accept": "The current set is a high-confidence, stable, reasonable candidate subtype.",
            "split": "Evidence supports internal heterogeneity or multimodal/molecular substructure that should be represented as child sets.",
            "merge": "Evidence supports weak boundary, set similarity, or compatibility with neighboring candidate sets.",
            "drop": "Evidence shows the current set should not be retained, or targeted evidence is exhausted and cannot support a subtype claim.",
            "continue_review": "Use only when a specific uncalled tool could still change accept/split/merge/drop.",
        },
    }
    messages = action_decision_messages(config_dir, config, payload)
    action_max_tokens = dict(config.get("llm", {}) or {}).get(
        "action_decision_max_new_tokens"
    )
    decision = call_llm_json_with_token_budget(
        messages,
        active_load_llm_client(config_dir),
        action_max_tokens,
    )
    parse_success = not bool(decision.get("_parse_error"))
    if not parse_success:
        decision = {
            "_parse_error": True,
            "cluster_id": payload["cluster_id"],
            "recommended_action": "",
            "review_unavailable_reason": "llm_parse_failure",
            "reasoning_summary": "Action decision LLM parsing failed.",
            "split_plan": None,
            "merge_plan": None,
            "limitations_to_report": [],
        }
    decision = normalize_action_decision(decision)
    evidence_catalog = list(cluster_state.get("evidence_catalog", []) or [])
    contract_issues = action_decision_contract_issues(
        decision, evidence_matrix, evidence_catalog
    )
    repair_issues = [
        issue
        for issue in contract_issues
        if not str(issue).startswith("non_high_confidence:")
    ]
    repair_used = False
    if parse_success and repair_issues:
        repair_used = True
        repair_messages = messages + [
            {
                "role": "assistant",
                "content": json.dumps(to_jsonable(decision), ensure_ascii=False),
            },
            {
                "role": "user",
                "content": (
                    "Repair the previous JSON only. Do not change the evidence. "
                    "Return the full action JSON with decision_state. If decision_state "
                    "is final, include a valid action, confidence_level, allowed "
                    "reason_codes, metric_refs from evidence_matrix, reasoning_summary, "
                    "and limitations_to_report. If decision_state "
                    "is continue_review, include confidence_level, metric_refs, "
                    "blocks_to_update, continue_review_reason, reasoning_summary, and "
                    "limitations_to_report. Contract issues: "
                    + json.dumps(repair_issues, ensure_ascii=False)
                ),
            },
        ]
        repaired = call_llm_json_with_token_budget(
            repair_messages,
            active_load_llm_client(config_dir),
            action_max_tokens,
        )
        if not bool(repaired.get("_parse_error")):
            repaired = normalize_action_decision(repaired)
            repaired["_repair_used"] = True
            decision = repaired
            contract_issues = action_decision_contract_issues(
                decision, evidence_matrix, evidence_catalog
            )
            repair_issues = [
                issue
                for issue in contract_issues
                if not str(issue).startswith("non_high_confidence:")
            ]
    if not decision.get("drop_reason") and decision["recommended_action"] == "drop":
        decision["drop_reason"] = "invalid_or_missing_action"
    # Merge decision into llm_audit
    audit.update(
        {
            k: v
            for k, v in decision.items()
            if k
            in {
                "recommended_action",
                "decision_state",
                "confidence_level",
                "confidence_basis",
                "reason_codes",
                "evidence_ids",
                "metric_refs",
                "blocks_to_update",
                "continue_review_reason",
                "review_unavailable_reason",
                "drop_reason",
                "split_plan",
                "merge_plan",
                "limitations_to_report",
                "reasoning_summary",
            }
        }
    )
    cluster_state["llm_audit"] = audit
    cluster_state["conversation_history"] = list(
        cluster_state.get("conversation_history", []) or []
    ) + [
        messages[-1],
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "recommended_action": audit.get("recommended_action"),
                    "confidence_level": audit.get("confidence_level"),
                    "reason_codes": audit.get("reason_codes", []),
                    "evidence_ids": audit.get("evidence_ids", []),
                    "metric_refs": audit.get("metric_refs", []),
                    "reasoning_summary": audit.get("reasoning_summary", ""),
                    "split_plan": audit.get("split_plan"),
                    "merge_plan": audit.get("merge_plan"),
                },
                ensure_ascii=False,
            ),
        },
    ]
    if not parse_success:
        budget_state = dict(cluster_state.get("budget_state", {}) or {})
        budget_state["llm_parse_failures"] = (
            int(budget_state.get("llm_parse_failures", 0)) + 1
        )
        budget_state["agent_failures"] = int(budget_state.get("agent_failures", 0) or 0) + 1
        cluster_state["budget_state"] = budget_state
    if repair_used and repair_issues:
        budget_state = dict(cluster_state.get("budget_state", {}) or {})
        budget_state["agent_failures"] = int(budget_state.get("agent_failures", 0) or 0) + 1
        cluster_state["budget_state"] = budget_state
    if bool(
        subtype_review_artifact_policy(config_dir).get("save_llm_audit", False)
    ) or save_debug_artifacts(config_dir):
        artifacts = dict(cluster_state.get("artifacts", {}) or {})
        artifacts[f"action_decision_round_{payload['review_round']}"] = (
            save_subtype_review_json(
                output_root,
                cluster_state,
                f"action_decision_round_{payload['review_round']}.json",
                decision,
            )
        )
        cluster_state["artifacts"] = artifacts
    cluster_state["decider_decision"] = decision
    return decision


def choose_router_action(
    cluster_state: Mapping[str, Any],
    budget_config: dict[str, Any] | str,
) -> str:
    config_dir = budget_config if isinstance(budget_config, str) else ""
    if isinstance(budget_config, str):
        budget_config = load_subtype_review_budget(budget_config)
    updated = cluster_state if isinstance(cluster_state, dict) else dict(cluster_state)
    init_review_budget_state(updated)
    budget = dict(budget_config["budget"])
    budget_state = updated["budget_state"]
    audit = dict(updated.get("llm_audit", {}) or {})
    raw_tool_plan = list(
        audit.get("tool_plan", []) or audit.get("tools_to_call", []) or []
    )
    tool_plan = filtered_tool_plan(raw_tool_plan, config_dir)
    if raw_tool_plan and not tool_plan:
        updated["drop_reason"] = "invalid_tool_plan"
        force_drop(updated, "invalid_tool_plan")
        return "drop"
    if tool_plan:
        remaining = int(budget["max_tool_calls"]) - int(
            budget_state.get("tool_calls_used", 0)
        )
        if remaining <= 0:
            budget_state["budget_exhausted"] = True
            budget_state["budget_exhausted_reason"] = "Tool call budget exhausted."
            force_drop(updated, "budget_exhausted")
            return "drop"
        updated["tool_plan"] = tool_plan[:remaining]
        return "call_tools"
    action = normalize_review_action(audit.get("recommended_action", ""))
    if int(budget_state.get("review_rounds_used", 0)) >= int(budget["max_rounds"]):
        budget_state["budget_exhausted"] = True
        budget_state["budget_exhausted_reason"] = "Review round budget exhausted."
        force_drop(updated, "budget_exhausted")
        return "drop"
    if int(budget_state.get("tool_failures", 0)) >= int(budget["max_failures"]):
        budget_state["budget_exhausted"] = True
        budget_state["budget_exhausted_reason"] = "Tool failure budget exhausted."
        force_drop(updated, "budget_exhausted")
        return "drop"
    if action in {"split", "merge"}:
        exhausted = revision_budget_exhausted(updated, action, budget)
        if exhausted:
            budget_state["budget_exhausted"] = True
            budget_state["budget_exhausted_reason"] = (
                f"Revision budget exhausted: {exhausted}."
            )
            force_drop(updated, "budget_exhausted")
            return "drop"
    if action == "drop":
        updated["drop_reason"] = str(
            audit.get("drop_reason", "") or "invalid_or_missing_action"
        )
    updated["final_action"] = action if action in {"accept", "drop"} else ""
    if action in {"accept", "drop"}:
        updated["final_decision"] = action
        updated["status"] = action
    return action if action in INTERNAL_REVIEW_ACTIONS else "drop"


def agentic_router_decision(
    cluster_state: Mapping[str, Any],
    budget_config: dict[str, Any] | str,
) -> dict[str, Any]:
    if isinstance(budget_config, str):
        budget = dict(load_subtype_review_budget(budget_config)["budget"])
    else:
        budget = dict(dict(budget_config or {}).get("budget", {}) or {})
    max_rounds = int(budget.get("max_rounds", 3) or 3)
    max_failures = int(budget.get("max_failures", 2) or 2)
    round_index = int(cluster_state.get("review_round", 0) or 0)
    verifier = dict(cluster_state.get("verifier_decision", {}) or {})
    budget_state = dict(cluster_state.get("budget_state", {}) or {})
    audit = dict(cluster_state.get("llm_audit", {}) or {})
    consistency = dict(verifier.get("decision_consistency_check", {}) or {})
    consistency_issues = [str(item) for item in list(consistency.get("issues", []) or [])]
    decision_state = normalize_decision_state(
        verifier.get("decision_state") or audit.get("decision_state")
    ) or "final"
    unavailable_markers = {
        "invalid_final_action",
        "missing_confidence_level",
    }
    llm_review_unavailable = bool(audit.get("_parse_error")) or (
        decision_state == "final" and bool(audit.get("review_unavailable_reason"))
    ) or (
        decision_state == "final"
        and
        not str(audit.get("recommended_action", "") or "").strip()
        and bool(set(consistency_issues) & unavailable_markers)
    )
    if (
        int(
            budget_state.get(
                "agent_failures", budget_state.get("llm_parse_failures", 0)
            )
            or 0
        )
        >= max_failures
    ):
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "send_to_revision",
            "next_round_index": None,
            "blocks_to_update": [],
            "send_to_revision_action": REVIEW_UNAVAILABLE_STATUS,
            "forced_drop_reason": "review_failed_after_max_failures",
            "route_reason": "Maximum agent failure budget reached; review decision is unavailable.",
        }
    if bool(verifier.get("final_action_ready", False)):
        action = normalize_review_action(
            str(verifier.get("final_action", "") or "drop")
        )
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "send_to_revision",
            "next_round_index": None,
            "blocks_to_update": [],
            "send_to_revision_action": action,
            "route_reason": "Verifier produced a high-confidence final action.",
        }
    planner_tool_plan = filtered_tool_plan(
        list(
            cluster_state.get("tool_plan", [])
            or audit.get("tool_plan", [])
            or audit.get("tools_to_call", [])
            or []
        ),
        "" if isinstance(budget_config, dict) else str(budget_config),
    )
    existing_tools = set(dict(cluster_state.get("validation_results", {}) or {}))
    planner_tool_plan = [
        item
        for item in planner_tool_plan
        if str(dict(item).get("tool_name", dict(item).get("name", "")) or "")
        not in existing_tools
    ]
    remaining_tool_budget = int(budget.get("max_tool_calls", 0) or 0) - int(
        budget_state.get("tool_calls_used", 0) or 0
    )
    if planner_tool_plan and remaining_tool_budget > 0:
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "continue_review",
            "next_round_index": round_index + 1,
            "blocks_to_update": blocks_from_tool_plan(planner_tool_plan),
            "tools_to_call": planner_tool_plan,
            "send_to_revision_action": None,
            "route_reason": "Planner requested targeted tools that may change the set-level action.",
        }
    if planner_tool_plan and remaining_tool_budget <= 0:
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "send_to_revision",
            "next_round_index": None,
            "blocks_to_update": [],
            "send_to_revision_action": "drop",
            "forced_drop_reason": "insufficient_evidence",
            "route_reason": "Planner requested more evidence, but the tool-call budget is exhausted.",
        }
    if round_index >= max_rounds:
        action = REVIEW_UNAVAILABLE_STATUS if llm_review_unavailable else "drop"
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "send_to_revision",
            "next_round_index": None,
            "blocks_to_update": [],
            "send_to_revision_action": action,
            "forced_drop_reason": "review_unavailable"
            if action == REVIEW_UNAVAILABLE_STATUS
            else "insufficient_evidence",
            "route_reason": "Maximum review rounds reached with no valid LLM decision."
            if action == REVIEW_UNAVAILABLE_STATUS
            else "Maximum review rounds reached without stable evidence for a final subtype-set action.",
        }
    blocks_to_update = [
        str(item)
        for item in list(verifier.get("blocks_to_update", []) or [])
        if str(item) in EVIDENCE_BLOCK_NAMES
    ]
    if consistency_issues and not blocks_to_update:
        if llm_review_unavailable:
            return {
                "cluster_id": str(cluster_state.get("cluster_id", "")),
                "round_index": round_index,
                "route_action": "send_to_revision",
                "next_round_index": None,
                "blocks_to_update": [],
                "send_to_revision_action": REVIEW_UNAVAILABLE_STATUS,
                "forced_drop_reason": str(
                    audit.get("review_unavailable_reason", "") or "review_unavailable"
                ),
                "route_reason": "LLM review did not return a usable action/confidence contract.",
            }
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "send_to_revision",
            "next_round_index": None,
            "blocks_to_update": [],
            "send_to_revision_action": "drop",
            "forced_drop_reason": "insufficient_evidence",
            "route_reason": "Verifier decision failed consistency checks: "
            + ", ".join(str(item) for item in consistency_issues),
        }
    if decision_state == "continue_review":
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "continue_review",
            "next_round_index": round_index + 1,
            "blocks_to_update": blocks_to_update,
            "tools_to_call": [],
            "send_to_revision_action": None,
            "route_reason": "Evidence remains insufficient; return to Planner until targeted tools, a final action, or the maximum review round is reached.",
        }
    if not blocks_to_update:
        return {
            "cluster_id": str(cluster_state.get("cluster_id", "")),
            "round_index": round_index,
            "route_action": "continue_review",
            "next_round_index": round_index + 1,
            "blocks_to_update": [],
            "tools_to_call": [],
            "send_to_revision_action": None,
            "route_reason": "No complete final action is available yet; return to Planner until the maximum review round is reached.",
        }
    return {
        "cluster_id": str(cluster_state.get("cluster_id", "")),
        "round_index": round_index,
        "route_action": "continue_review",
        "next_round_index": round_index + 1,
        "blocks_to_update": blocks_to_update,
        "tools_to_call": [],
        "send_to_revision_action": None,
        "route_reason": "Decider requested more review; rerun Planner without auto-selecting tools from missing blocks.",
    }


def revision_engine(
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    updated = init_review_budget_state(dict(cluster_state))
    action = str(updated.get("next_action", "") or "")
    budget_config = load_subtype_review_budget(config_dir)
    budget = dict(budget_config["budget"])
    artifacts = dict(updated.get("artifacts", {}) or {})
    if action == "continue_review":
        updated["next_action"] = ""
        updated["tool_plan"] = []
        updated["recommended_tools"] = []
        updated["status"] = "under_review"
        return updated
    if action == "call_tools":
        tools_to_call = list(
            updated.get("tool_plan", [])
            or updated.get("recommended_tools", [])
            or dict(updated.get("llm_audit", {}) or {}).get("tools_to_call", [])
            or []
        )
        tools_to_call = filtered_tool_plan(tools_to_call, config_dir)
        remaining = int(budget["max_tool_calls"]) - int(
            updated["budget_state"].get("tool_calls_used", 0)
        )
        filtered_tools = []
        for item in tools_to_call:
            item = {"tool_name": str(item)} if isinstance(item, str) else dict(item)
            tool_name = str(item.get("tool_name", item.get("name", "")) or "")
            if tool_name:
                filtered_tools.append(
                    {
                        "tool_name": tool_name,
                        **{key: value for key, value in item.items() if key != "name"},
                    }
                )
        tools_to_call = filtered_tools[: max(remaining, 0)]
        if not tools_to_call:
            updated["budget_state"]["budget_exhausted"] = True
            updated["budget_state"]["budget_exhausted_reason"] = (
                "No tool budget remains or all requested tools are disabled."
            )
            updated["status"] = "drop"
            updated["next_action"] = ""
            updated["final_action"] = "drop"
            updated["final_decision"] = "drop"
            updated["drop_reason"] = "insufficient_evidence"
            updated["insufficient_evidence_reason"] = (
                "No tool budget remains or all requested tools are disabled."
            )
            return generate_cluster_report(updated, output_root, config_dir)
        tool_results = []
        validation_results = dict(updated.get("validation_results", {}) or {})
        tool_imports = subtype_review_tool_imports(config_dir)
        for tool_plan in tools_to_call:
            tool_name = str(dict(tool_plan).get("tool_name", "") or "")
            try:
                module_name, function_name = tool_imports[tool_name]
                tool_function = getattr(
                    importlib.import_module(module_name), function_name
                )
                result = tool_function(
                    updated,
                    {
                        str(key): dict(value)
                        for key, value in patient_states_by_id.items()
                    },
                    output_root,
                    config_dir=config_dir,
                    all_cluster_states=all_cluster_states or [],
                )
            except KeyError:
                result = {
                    "tool_name": tool_name or "unknown_tool",
                    "status": "not_implemented",
                    "cluster_id": str(updated.get("cluster_id", "")),
                    "results": {
                        "summary": "Requested tool is not registered.",
                        "metrics": {},
                        "evidence_hints": [],
                        "warnings": [f"Unknown tool: {tool_name}"],
                        "missing_reason": "unknown tool",
                    },
                    "artifacts": {},
                    "errors": [],
                }
            except Exception as exc:
                result = {
                    "tool_name": tool_name,
                    "status": "failure",
                    "cluster_id": str(updated.get("cluster_id", "")),
                    "results": {
                        "summary": "Tool execution failed.",
                        "metrics": {},
                        "evidence_hints": [],
                        "warnings": [],
                        "missing_reason": "",
                    },
                    "artifacts": {},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            validation_results[str(result.get("tool_name", tool_name))] = result
            tool_results.append(result)
        updated["validation_results"] = validation_results
        updated["tool_results"] = tool_results
        updated["conversation_history"] = list(
            updated.get("conversation_history", []) or []
        ) + [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "review_round": int(updated.get("review_round", 0) or 0),
                        "tool_results": [
                            compact_tool_result(item) for item in tool_results
                        ],
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        round_id = int(updated.get("review_round", 0) or 0)
        compact_results = [
            compact_tool_result(item) for item in validation_results.values()
        ]
        artifacts["tool_results"] = save_subtype_review_json(
            output_root,
            updated,
            "tool_results.json",
            {
                "review_round": round_id,
                "tool_results": compact_results,
            },
        )
        artifacts[f"round_{round_id}_tool_results"] = save_subtype_review_json(
            output_root,
            updated,
            f"round_{round_id}/tool_results.json",
            {
                "review_round": round_id,
                "tool_results": compact_results,
            },
        )
        updated["evidence_matrix"] = build_agentic_evidence_matrix(
            updated,
            patient_states_by_id,
            previous_evidence_matrix=dict(updated.get("evidence_matrix", {}) or {}),
        )
        updated["evidence_catalog"] = build_evidence_catalog(
            str(updated.get("cluster_id", "")),
            dict(updated.get("evidence_matrix", {}) or {}),
        )
        artifacts[f"round_{round_id}_evidence_matrix"] = save_subtype_review_json(
            output_root,
            updated,
            f"round_{round_id}/evidence_matrix.json",
            updated["evidence_matrix"],
        )
        if save_debug_artifacts(config_dir):
            artifacts[f"tool_results_round_{round_id}"] = save_subtype_review_json(
                output_root,
                updated,
                f"tool_results_round_{round_id}.json",
                tool_results,
            )
        updated["artifacts"] = artifacts
        updated = update_review_budget_after_revision(
            updated, tool_results=tool_results
        )
        updated["next_action"] = ""
        updated["tool_plan"] = []
        return updated
    if action in {"split", "merge"}:
        exhausted = revision_budget_exhausted(updated, action, budget)
        if exhausted:
            updated["budget_state"]["budget_exhausted"] = True
            updated["budget_state"]["budget_exhausted_reason"] = (
                f"Revision budget exhausted: {exhausted}."
            )
            updated = force_drop(updated, "budget_exhausted")
            updated = generate_cluster_report(updated, output_root, config_dir)
            return updated
        plan_key = "split_plan" if action == "split" else "merge_plan"
        plan = dict(
            updated.get(plan_key)
            or dict(updated.get("llm_audit", {}) or {}).get(plan_key)
            or {}
        )
        updated[plan_key] = plan
        previous_members = [
            str(item) for item in list(updated.get("member_ids", []) or [])
        ]
        previous_evidence = {
            "review_round": int(updated.get("review_round", 0) or 0),
            "validation_results": dict(updated.get("validation_results", {}) or {}),
            "structured_evidence": list(updated.get("structured_evidence", []) or []),
            "tool_results": list(updated.get("tool_results", []) or []),
            "llm_audit": dict(updated.get("llm_audit", {}) or {}),
        }
        source_cluster_id = str(
            cluster_state.get("cluster_id", "") or updated.get("cluster_id", "")
        )
        base_parent_ids = [
            str(item)
            for item in list(
                updated.get("parent_cluster_ids", []) or [source_cluster_id]
            )
            if str(item)
        ]
        next_revision_round = int(updated.get("revision_round", 0) or 0) + 1
        base_previous_evidence = list(updated.get("previous_evidence", []) or []) + [
            previous_evidence
        ]
        base_budget_state = dict(
            update_review_budget_after_revision(updated)["budget_state"]
        )
        generated_clusters = []
        absorbed_cluster_ids = []
        if action == "split":
            split_specs = list(plan.get("new_clusters", []) or [])
            if not split_specs:
                split_specs = split_cluster_specs_from_consensus(
                    updated, source_cluster_id
                ) or [
                    {
                        "cluster_id": f"{source_cluster_id}_S1",
                        "member_ids": split_cluster_members(previous_members, plan),
                    }
                ]
            for index, item in enumerate(split_specs, start=1):
                split_spec = dict(item)
                child_members = [
                    str(member_id)
                    for member_id in list(
                        split_spec.get("member_ids", [])
                        or split_spec.get("members", [])
                        or []
                    )
                    if str(member_id) in previous_members
                ]
                if not child_members:
                    continue
                child_id = str(
                    split_spec.get("cluster_id", "") or f"{source_cluster_id}_S{index}"
                )
                revision_entry = {
                    "action": "split",
                    "previous_member_ids": previous_members,
                    "updated_member_ids": child_members,
                    "source_cluster_id": source_cluster_id,
                    "updated_cluster_id": child_id,
                    "plan": plan,
                }
                child = {
                    **updated,
                    **split_spec,
                    "cluster_id": child_id,
                    "member_ids": child_members,
                    "parent_cluster_ids": base_parent_ids,
                    "status": "under_review",
                    "final_action": "",
                    "final_decision": "",
                    "drop_reason": "",
                    "next_action": "",
                    "review_round": 0,
                    "revision_round": next_revision_round,
                    "validation_results": {},
                    "structured_evidence": [],
                    "tool_results": [],
                    "verification_vector": {},
                    "llm_audit": {},
                    "evidence_matrix": {},
                    "verifier_decision": {},
                    "router_decision": {},
                    "round_summaries": [],
                    "tool_plan": [],
                    "recommended_tools": [],
                    "generated_clusters": [],
                    "absorbed_cluster_ids": [],
                    "previous_evidence": base_previous_evidence,
                    "revision_history": list(updated.get("revision_history", []) or [])
                    + [revision_entry],
                    "budget_state": dict(base_budget_state),
                    "artifacts": dict(updated.get("artifacts", {}) or {}),
                }
                generated_clusters.append(child)
        else:
            merged_members = merge_cluster_members(
                previous_members, plan, all_cluster_states or []
            )
            target_ids = [
                str(item)
                for item in list(
                    plan.get("target_cluster_ids", [])
                    or plan.get("merge_with", [])
                    or []
                )
            ]
            absorbed_cluster_ids = [item for item in target_ids if item]
            parent_ids = sorted(set(base_parent_ids + absorbed_cluster_ids))
            merged_id = str(
                plan.get("cluster_id", "")
                or plan.get("merged_cluster_id", "")
                or f"{source_cluster_id}_M{next_revision_round}"
            )
            revision_entry = {
                "action": "merge",
                "previous_member_ids": previous_members,
                "updated_member_ids": merged_members,
                "source_cluster_id": source_cluster_id,
                "updated_cluster_id": merged_id,
                "absorbed_cluster_ids": absorbed_cluster_ids,
                "plan": plan,
            }
            generated_clusters.append(
                {
                    **updated,
                    "cluster_id": merged_id,
                    "member_ids": merged_members,
                    "parent_cluster_ids": parent_ids,
                    "absorbed_cluster_ids": absorbed_cluster_ids,
                    "status": "under_review",
                    "final_action": "",
                    "final_decision": "",
                    "drop_reason": "",
                    "next_action": "",
                    "review_round": 0,
                    "revision_round": next_revision_round,
                    "validation_results": {},
                    "structured_evidence": [],
                    "tool_results": [],
                    "verification_vector": {},
                    "llm_audit": {},
                    "evidence_matrix": {},
                    "verifier_decision": {},
                    "router_decision": {},
                    "round_summaries": [],
                    "tool_plan": [],
                    "recommended_tools": [],
                    "generated_clusters": [],
                    "previous_evidence": base_previous_evidence,
                    "revision_history": list(updated.get("revision_history", []) or [])
                    + [revision_entry],
                    "budget_state": dict(base_budget_state),
                    "artifacts": dict(updated.get("artifacts", {}) or {}),
                }
            )
        updated["status"] = action
        updated["final_action"] = action
        updated["final_decision"] = action
        updated["generated_clusters"] = generated_clusters
        updated["absorbed_cluster_ids"] = absorbed_cluster_ids
        updated["revision_round"] = next_revision_round
        updated["previous_evidence"] = base_previous_evidence
        updated["revision_history"] = [
            revision
            for child in generated_clusters
            for revision in list(child.get("revision_history", []) or [])[-1:]
        ]
        artifacts["revision_history"] = save_subtype_review_json(
            output_root,
            updated,
            "revision_history.json",
            list(updated.get("revision_history", []) or []),
        )
        if save_debug_artifacts(config_dir):
            artifacts[f"{plan_key}_path"] = save_subtype_review_json(
                output_root,
                updated,
                f"{plan_key}.json",
                plan,
            )
        updated["artifacts"] = artifacts
        updated["next_action"] = ""
        return generate_cluster_report(updated, output_root, config_dir)
    return updated


def generate_cluster_report(
    cluster_state: Mapping[str, Any],
    output_root: str,
    config_dir: str,
) -> dict[str, Any]:
    updated = init_review_budget_state(dict(cluster_state))
    review_status = str(updated.get("status", "") or "")
    if review_status == REVIEW_UNAVAILABLE_STATUS:
        decision = ""
    else:
        decision = normalize_review_action(
            str(
                updated.get("final_decision")
                or updated.get("final_action")
                or updated.get("next_action")
                or "drop"
            )
        )
    evidence_matrix = dict(updated.get("evidence_matrix", {}) or {})
    if set(evidence_matrix) != set(EVIDENCE_BLOCK_NAMES):
        evidence_matrix = build_agentic_evidence_matrix(updated, {})
    verifier_decision = dict(updated.get("verifier_decision", {}) or {})
    audit = dict(updated.get("llm_audit", {}) or {})
    dimension_assessments = dict(
        verifier_decision.get("dimension_assessments", {}) or {}
    )
    if not dimension_assessments:
        dimension_assessments = {
            name: str(
                dict(
                    dict(evidence_matrix.get(name, {}) or {}).get("llm_assessment", {})
                    or {}
                ).get("assessment", "unavailable")
            )
            for name in EVIDENCE_BLOCK_NAMES
        }
    reason_codes = [
        str(item)
        for item in list(verifier_decision.get("reason_codes", []) or [])
        if str(item)
    ]
    forced_drop_reason = str(updated.get("drop_reason", "") or "")
    if review_status == REVIEW_UNAVAILABLE_STATUS:
        reason_codes = ["review_unavailable"]
    if decision == "drop" and forced_drop_reason in {
        "insufficient_after_max_rounds",
        "review_failed_after_max_failures",
    }:
        reason_codes = [forced_drop_reason]
    if not reason_codes:
        if decision == "accept":
            reason_codes = ["reliable_biological_nonconfounded"]
        elif decision == "split":
            reason_codes = ["internal_heterogeneity_supported"]
        elif decision == "merge":
            reason_codes = ["weak_boundary_compatible_evidence"]
        elif (
            str(updated.get("drop_reason", "") or "")
            in FINAL_ACTION_REASON_CODES["drop"]
        ):
            reason_codes = [str(updated.get("drop_reason"))]
        elif decision == "drop":
            reason_codes = ["other_with_explanation"]
    metric_refs = [
        str(item)
        for item in list(verifier_decision.get("metric_refs", []) or [])
        if str(item)
    ]
    evidence_ids = [
        str(item)
        for item in list(verifier_decision.get("evidence_ids", []) or [])
        if str(item)
    ]
    if not metric_refs:
        metric_refs = [
            "final_evidence_matrix.set_reliability.metrics.set_reliability_global_consensus.total_candidate_set_n"
        ]
    rationale = str(
        verifier_decision.get("rationale")
        or audit.get("reasoning_summary")
        or updated.get("review_unavailable_reason")
        or (
            "Subtype review did not produce a usable LLM decision."
            if review_status == REVIEW_UNAVAILABLE_STATUS
            else f"Subtype review ended with {decision}."
        )
    )
    confidence_level = normalize_confidence_level(
        verifier_decision.get("confidence_level") or audit.get("confidence_level")
    )
    confidence_basis = normalize_confidence_basis(
        verifier_decision.get("confidence_basis")
        or audit.get("confidence_basis")
    )
    round_summaries = list(updated.get("round_summaries", []) or [])
    if not round_summaries:
        round_summaries = [
            {
                "round_index": int(updated.get("review_round", 0) or 0),
                "verifier_summary": rationale,
                "router_summary": str(
                    dict(updated.get("router_decision", {}) or {}).get(
                        "route_reason", ""
                    )
                    or f"Cluster sent to revision with {decision}."
                ),
                "blocks_updated_next": list(
                    dict(updated.get("router_decision", {}) or {}).get(
                        "blocks_to_update", []
                    )
                    or []
                ),
            }
        ]
    decision_basis = list(updated.get("decision_basis", []) or [])
    if not decision_basis:
        decision_basis = [
            {
                "statement": rationale,
                "metric_refs": metric_refs,
            }
        ]
    figures = {name: {} for name in EVIDENCE_BLOCK_NAMES}
    for name, block in evidence_matrix.items():
        block_figures = dict(dict(block or {}).get("figures", {}) or {})
        if block_figures:
            figures[name] = block_figures
    report = {
        "metadata": {
            "cluster_id": str(updated.get("cluster_id", "")),
            "final_action": decision,
            "review_status": review_status or decision,
            "confidence_level": confidence_level,
            "high_confidence_candidate_subtype": bool(
                decision == "accept" and confidence_level == "high"
            ),
            "rounds_used": int(updated.get("review_round", 0) or 0),
            "member_count": len(list(updated.get("member_ids", []) or [])),
            "member_ids": list(updated.get("member_ids", []) or []),
            "parent_cluster_ids": list(updated.get("parent_cluster_ids", []) or []),
            "review_unavailable_reason": str(
                updated.get("review_unavailable_reason", "") or ""
            ),
            "insufficient_evidence_reason": str(
                updated.get("insufficient_evidence_reason", "") or ""
            ),
        },
        "round_summaries": round_summaries,
        "final_evidence_matrix": evidence_matrix,
        "verifier_decision": {
            "final_action": decision,
            "review_status": review_status or decision,
            "confidence_level": confidence_level,
            "confidence_basis": confidence_basis,
            "high_confidence_candidate_subtype": bool(
                decision == "accept" and confidence_level == "high"
            ),
            "reason_codes": reason_codes,
            "rationale": rationale,
            "evidence_ids": evidence_ids,
            "metric_refs": metric_refs,
            "dimension_assessments": dimension_assessments,
        },
        "revision_result": {
            "generated_clusters": [
                {
                    "cluster_id": str(item.get("cluster_id", "")),
                    "member_ids": list(item.get("member_ids", []) or []),
                    "parent_cluster_ids": list(
                        item.get("parent_cluster_ids", []) or []
                    ),
                }
                for item in list(updated.get("generated_clusters", []) or [])
            ],
            "revision_history": list(updated.get("revision_history", []) or []),
        },
        "decision_basis": decision_basis,
        "limitations": list(updated.get("limitations", []) or []),
        "figures": figures,
    }
    updated["budget_state"]["report_calls_used"] = (
        int(updated["budget_state"].get("report_calls_used", 0)) + 1
    )
    updated["report_draft"] = report
    artifacts = dict(updated.get("artifacts", {}) or {})
    artifacts["report"] = save_subtype_review_json(
        output_root, updated, "report.json", report
    )
    if save_debug_artifacts(config_dir):
        artifacts["budget_state"] = save_subtype_review_json(
            output_root, updated, "budget_state.json", updated["budget_state"]
        )
    updated["artifacts"] = artifacts
    return updated


def collect_association_figure_rows(
    cluster_states: list[Mapping[str, Any]],
    block_name: str,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for cluster_state in cluster_states:
        cluster_id = str(cluster_state.get("cluster_id", "") or "")
        matrix = review_matrix_for_cluster(cluster_state)
        metrics = dict(
            dict(matrix.get(block_name, {}) or {}).get("metrics", {}) or {}
        )
        for metric_name in metric_names:
            for item in list(metrics.get(metric_name, []) or []):
                row = dict(item or {})
                field = str(row.get("field") or row.get("name") or row.get("level") or "")
                if not field:
                    continue
                effect = number_or_none(
                    row.get("effect_size")
                    or row.get("cramers_v")
                )
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "field": field,
                        "group": metric_name,
                        "effect": effect if effect is not None else 0.0,
                        "q_value": number_or_none(row.get("q_value")),
                        "p_value": number_or_none(row.get("p_value")),
                    }
                )
    return rows


def save_global_dotplot(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    y_key: str,
) -> str:
    if not rows:
        return ""
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clusters = sorted({str(row["cluster_id"]) for row in rows})
        features = sorted(
            {str(row[y_key]) for row in rows},
            key=lambda item: min(
                (
                    row.get("q_value")
                    if row.get("q_value") is not None
                    else row.get("p_value")
                    if row.get("p_value") is not None
                    else 1.0
                )
                for row in rows
                if str(row[y_key]) == item
            ),
        )[:28]
        plot_rows = [row for row in rows if str(row[y_key]) in features]
        x_index = {cluster: i for i, cluster in enumerate(clusters)}
        y_index = {feature: i for i, feature in enumerate(features)}
        fig, ax = plt.subplots(
            figsize=(max(6.5, len(clusters) * 0.75), max(4.2, len(features) * 0.24)),
            dpi=180,
        )
        q_values = [
            row.get("q_value")
            if row.get("q_value") is not None
            else row.get("p_value")
            if row.get("p_value") is not None
            else 1.0
            for row in plot_rows
        ]
        scatter = ax.scatter(
            [x_index[str(row["cluster_id"])] for row in plot_rows],
            [y_index[str(row[y_key])] for row in plot_rows],
            s=[
                35.0 + min(abs(float(row.get("effect") or 0.0)), 5.0) * 35.0
                for row in plot_rows
            ],
            c=[-math.log10(max(float(value), 1e-300)) for value in q_values],
            cmap="viridis",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([short_feature_label(item) for item in features], fontsize=8)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.18, linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("-log10(q/p)", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return str(output_path)
    except Exception:
        return ""


def save_multimodal_dotplot(rows: list[dict[str, Any]], output_path: Path) -> str:
    if not rows:
        return ""
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clusters = sorted({str(row["cluster_id"]) for row in rows})
        modalities = sorted({str(row["modality"]) for row in rows})
        x_index = {cluster: i for i, cluster in enumerate(clusters)}
        y_index = {modality: i for i, modality in enumerate(modalities)}
        fig, ax = plt.subplots(
            figsize=(max(6.5, len(clusters) * 0.75), max(3.8, len(modalities) * 0.55)),
            dpi=180,
        )
        scatter = ax.scatter(
            [x_index[str(row["cluster_id"])] for row in rows],
            [y_index[str(row["modality"])] for row in rows],
            s=[45.0 + 180.0 * float(row.get("coverage_fraction") or 0.0) for row in rows],
            c=[float(row.get("effect") or 0.0) for row in rows],
            cmap="coolwarm",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(modalities)))
        ax.set_yticklabels(modalities, fontsize=8)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_title("Multimodal support by cluster", fontsize=10)
        ax.grid(alpha=0.18, linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Separation effect", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return str(output_path)
    except Exception:
        return ""


def build_review_global_figures(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    if output_root:
        clear_stale_global_figures(output_root)
    labels = cluster_labels(cluster_states)
    if not output_root or not labels:
        return {}
    if not matplotlib_available():
        figures = build_fallback_global_figures(output_root, cluster_states, patient_states, labels)
        build_cluster_review_figures(output_root, cluster_states, patient_states, labels)
        return figures
    figures = {}
    plotters = [
        ("consensus_matrix_heatmap", plot_consensus_matrix_heatmap),
        ("integrated_snf_embedding_scatter", plot_integrated_snf_embedding_scatter),
        ("modality_embedding_4panel", plot_modality_embedding_4panel),
        ("rna_hallmark_ssgsea_bubble", plot_rna_hallmark_ssgsea_bubble),
        ("rna_subtype_signature_dotplot", plot_rna_subtype_signature_dotplot),
        ("rna_subtype_defining_hallmark_heatmap", plot_rna_subtype_defining_hallmark_heatmap),
        ("rna_top_pathway_boxplots", plot_rna_top_pathway_boxplots),
        ("mutation_gene_oncoplot", plot_mutation_gene_oncoplot),
        ("mutation_kegg_pathway_bubble", plot_mutation_kegg_pathway_bubble),
        ("mutation_reactome_pathway_bubble", plot_mutation_reactome_pathway_bubble),
        ("mutation_pathway_bubble", plot_mutation_pathway_bubble),
        ("rna_hallmark_ssgsea_heatmap", plot_rna_hallmark_ssgsea_heatmap),
        ("survival_global_km", plot_survival_global_km),
        ("subtype_evidence_score_panel", plot_subtype_evidence_score_panel),
        ("confounder_association_heatmap", plot_confounder_association_heatmap),
        ("known_label_association_heatmap", plot_known_label_association_heatmap),
    ]
    for name, plotter in plotters:
        try:
            path = plotter(output_root, cluster_states, patient_states, labels)
        except Exception:
            path = ""
        if path:
            figures[name] = path
    build_cluster_review_figures(output_root, cluster_states, patient_states, labels)
    return figures


def build_fallback_global_figures(
    output_root: str,
    cluster_states: list[Mapping[str, Any]],
    patient_states: list[Mapping[str, Any]] | None,
    labels: Mapping[str, str],
) -> dict[str, str]:
    figures = {}
    matrix, patient_ids = consensus_matrix_and_ids(output_root, cluster_states, labels)
    if matrix.size and patient_ids:
        figures["consensus_matrix_heatmap"] = save_fallback_figure(output_root, "consensus_matrix_heatmap", 1)
        figures["integrated_snf_embedding_scatter"] = save_fallback_figure(output_root, "integrated_snf_embedding_scatter", 2)
    has_modality = False
    for modality in ["ct", "wsi", "rna", "wxs"]:
        case_ids, modality_matrix = matrix_from_patient_vectors(patient_states, labels, modality)
        has_modality = has_modality or (len(case_ids) >= 2 and modality_matrix.size > 0)
    if has_modality:
        figures["modality_embedding_4panel"] = save_fallback_figure(output_root, "modality_embedding_4panel", 3)
    if biological_metric_rows(cluster_states, "rna_pathway_enrichment"):
        figures["rna_hallmark_ssgsea_bubble"] = save_fallback_figure(output_root, "rna_hallmark_ssgsea_bubble", 4)
        figures["rna_subtype_signature_dotplot"] = save_fallback_figure(output_root, "rna_subtype_signature_dotplot", 12)
        figures["rna_subtype_defining_hallmark_heatmap"] = save_fallback_figure(output_root, "rna_subtype_defining_hallmark_heatmap", 13)
        figures["rna_top_pathway_boxplots"] = save_fallback_figure(output_root, "rna_top_pathway_boxplots", 14)
    case_ids, _, rna_matrix = matrix_from_rna_pathway_scores(output_root, patient_states, labels)
    if len(case_ids) >= 2 and rna_matrix.size:
        figures["rna_hallmark_ssgsea_heatmap"] = save_fallback_figure(output_root, "rna_hallmark_ssgsea_heatmap", 5)
    if biological_metric_rows(cluster_states, "wxs_gene_enrichment"):
        figures["mutation_gene_oncoplot"] = save_fallback_figure(output_root, "mutation_gene_oncoplot", 6)
    pathway_rows = biological_metric_rows(cluster_states, "wxs_pathway_enrichment")
    if pathway_collection_rows(pathway_rows, "KEGG", "primary"):
        figures["mutation_kegg_pathway_bubble"] = save_fallback_figure(output_root, "mutation_kegg_pathway_bubble", 7)
        figures["mutation_pathway_bubble"] = figures["mutation_kegg_pathway_bubble"]
    if pathway_collection_rows(pathway_rows, "REACTOME", "validation"):
        figures["mutation_reactome_pathway_bubble"] = save_fallback_figure(output_root, "mutation_reactome_pathway_bubble", 11)
    if pathway_rows and "mutation_pathway_bubble" not in figures:
        figures["mutation_pathway_bubble"] = save_fallback_figure(output_root, "mutation_pathway_bubble", 7)
    if survival_records(patient_states, labels)[0]:
        figures["survival_global_km"] = save_fallback_figure(output_root, "survival_global_km", 10)
    if (
        biological_metric_rows(cluster_states, "rna_pathway_enrichment")
        or biological_metric_rows(cluster_states, "wxs_gene_enrichment")
        or biological_metric_rows(cluster_states, "wxs_pathway_enrichment")
        or first_review_metrics(cluster_states, "set_reliability")
        or first_review_metrics(cluster_states, "clinical_context")
    ):
        figures["subtype_evidence_score_panel"] = save_fallback_figure(output_root, "subtype_evidence_score_panel", 15)
    if association_heatmap_rows(cluster_states, "confounder_set_association"):
        figures["confounder_association_heatmap"] = save_fallback_figure(output_root, "confounder_association_heatmap", 8)
    if association_heatmap_rows(cluster_states, "per_label_enrichment"):
        figures["known_label_association_heatmap"] = save_fallback_figure(output_root, "known_label_association_heatmap", 9)
    return figures


def attach_global_figures_to_reports(
    cluster_states: list[Mapping[str, Any]],
    global_figures: Mapping[str, str],
) -> None:
    if not global_figures:
        return
    for cluster_state in cluster_states:
        report = dict(cluster_state.get("report_draft", {}) or {})
        if not report:
            continue
        figures = dict(report.get("figures", {}) or {})
        for name in EVIDENCE_BLOCK_NAMES:
            figures.setdefault(name, {})
        figures["set_reliability"] = dict(figures.get("set_reliability", {}) or {})
        figures["biological_support"] = dict(
            figures.get("biological_support", {}) or {}
        )
        figures["multimodal_support"] = dict(figures.get("multimodal_support", {}) or {})
        figures["clinical_context"] = dict(figures.get("clinical_context", {}) or {})
        figures.setdefault("known_label_echo", {})
        figures.setdefault("confounder_exclusion", {})
        report["figures"] = figures
        figures["set_reliability"].update(
            {
                key: global_figures[key]
                for key in ["consensus_matrix_heatmap", "integrated_snf_embedding_scatter"]
                if key in global_figures
            }
        )
        figures["multimodal_support"].update(
            {
                key: global_figures[key]
                for key in ["modality_embedding_4panel", "integrated_snf_embedding_scatter"]
                if key in global_figures
            }
        )
        figures["biological_support"].update(
            {
                key: global_figures[key]
                for key in [
                    "rna_hallmark_ssgsea_bubble",
                    "rna_hallmark_ssgsea_heatmap",
                    "rna_subtype_signature_dotplot",
                    "rna_subtype_defining_hallmark_heatmap",
                    "rna_top_pathway_boxplots",
                    "subtype_evidence_score_panel",
                    "mutation_gene_oncoplot",
                    "mutation_kegg_pathway_bubble",
                    "mutation_reactome_pathway_bubble",
                    "mutation_pathway_bubble",
                ]
                if key in global_figures
            }
        )
        figures["clinical_context"].update(
            {
                key: global_figures[key]
                for key in ["survival_global_km"]
                if key in global_figures
            }
        )
        figures["known_label_echo"].update(
            {
                key: global_figures[key]
                for key in ["known_label_association_heatmap"]
                if key in global_figures
            }
        )
        figures["confounder_exclusion"].update(
            {
                key: global_figures[key]
                for key in ["confounder_association_heatmap"]
                if key in global_figures
            }
        )
        cluster_state["report_draft"] = report
        report_path = str(dict(cluster_state.get("artifacts", {}) or {}).get("report", "") or "")
        if report_path:
            path = Path(report_path)
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(existing, Mapping)
                    and "metadata" in existing
                    and "verifier_decision" in existing
                    and (
                        "metadata" not in report
                        or "verifier_decision" not in report
                    )
                ):
                    existing_figures = dict(existing.get("figures", {}) or {})
                    for name, value in figures.items():
                        merged = dict(existing_figures.get(name, {}) or {})
                        merged.update(dict(value or {}))
                        existing_figures[name] = merged
                    existing["figures"] = existing_figures
                    report = dict(existing)
                    cluster_state["report_draft"] = report
            write_json(report_path, report)


def build_pipeline_output(
    *,
    patient_states: list[Mapping[str, Any]],
    candidate_clusters: list[Mapping[str, Any]],
    cluster_states: list[Mapping[str, Any]],
    patient_store_paths: Mapping[str, Any],
    cluster_store_paths: Mapping[str, Any],
    graph_paths: Mapping[str, Any],
    output_root: str = "",
) -> dict[str, Any]:
    global_figures = build_review_global_figures(output_root, cluster_states, patient_states)
    attach_global_figures_to_reports(cluster_states, global_figures)
    all_patient_ids = {
        str(item.get("case_id", ""))
        for item in patient_states
        if str(item.get("case_id", ""))
    }
    qc_passed_patient_ids = {
        str(item.get("case_id", ""))
        for item in patient_states
        if str(item.get("case_id", "")) and item.get("qc") == "success"
    }
    final_member_ids = {
        str(member_id)
        for item in cluster_states
        if str(item.get("status", "") or item.get("final_action", "") or "")
        in {"accept", "drop", "split", "merge"}
        for member_id in list(item.get("member_ids", []) or [])
    }
    final_reports = [
        dict(
            item.get("report_draft", {})
            or {
                "cluster_id": str(item.get("cluster_id", "")),
                "decision": str(
                    item.get("final_decision", item.get("next_action", ""))
                ),
                "status": str(item.get("status", "")),
                "limitations": list(item.get("limitations", []) or []),
            }
        )
        for item in cluster_states
    ]
    all_cluster_reports = []
    accepted_candidate_subtypes = []
    action_counts: dict[str, int] = {}
    reason_code_counts: dict[str, int] = {}
    for item in cluster_states:
        report = dict(item.get("report_draft", {}) or {})
        metadata = dict(report.get("metadata", {}) or {})
        verifier = dict(report.get("verifier_decision", {}) or {})
        action = str(
            metadata.get("final_action")
            or item.get("final_action")
            or item.get("final_decision")
            or item.get("status")
            or ""
        )
        confidence_level = normalize_confidence_level(
            metadata.get("confidence_level")
            or verifier.get("confidence_level")
            or dict(item.get("verifier_decision", {}) or {}).get("confidence_level")
        )
        high_confidence = bool(confidence_level == "high")
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1
        reason_codes = [
            str(code)
            for code in list(
                verifier.get("reason_codes", item.get("reason_codes", [])) or []
            )
            if str(code)
        ]
        if not reason_codes and str(item.get("drop_reason", "") or ""):
            reason_codes = [str(item.get("drop_reason", ""))]
        primary_reason_code = reason_codes[0] if reason_codes else ""
        for code in reason_codes:
            reason_code_counts[code] = reason_code_counts.get(code, 0) + 1
        report_path = str(dict(item.get("artifacts", {}) or {}).get("report", "") or "")
        dimension_assessments = dict(
            verifier.get(
                "dimension_assessments",
                dict(item.get("verifier_decision", {}) or {}).get(
                    "dimension_assessments", {}
                ),
            )
            or {}
        )
        all_cluster_reports.append(
            {
                "cluster_id": str(item.get("cluster_id", "")),
                "action": action,
                "rounds": int(
                    item.get("review_round", metadata.get("rounds_used", 0)) or 0
                ),
                "parent_ids": list(item.get("parent_cluster_ids", []) or []),
                "assessments": dimension_assessments,
                "confidence_level": confidence_level,
                "high_confidence": high_confidence,
                "not_high_confidence_reason": ""
                if high_confidence
                else str(
                    dict(
                        item.get("verifier_decision", verifier) or {}
                    ).get("continue_review_reason", "")
                    or (
                        "Final verifier confidence was not high."
                        if confidence_level
                        else "Verifier confidence was unavailable."
                    )
                ),
                "primary_reason_code": primary_reason_code,
                "reason_codes": reason_codes,
                "review_unavailable_reason": str(
                    item.get("review_unavailable_reason", "") or ""
                ),
                "figures": dict(report.get("figures", {}) or {}),
                "report_path": report_path,
            }
        )
        if action == "accept":
            basis = list(report.get("decision_basis", []) or [])
            evidence_statements = [
                str(entry.get("statement", ""))
                for entry in basis
                if str(entry.get("statement", ""))
            ][:5]
            accepted_candidate_subtypes.append(
                {
                    "cluster_id": str(item.get("cluster_id", "")),
                    "member_count": len(list(item.get("member_ids", []) or [])),
                    "confidence_level": confidence_level,
                    "high_confidence_candidate_subtype": high_confidence,
                    "key_evidence": evidence_statements[:5],
                    "reason_codes": reason_codes,
                    "report_path": report_path,
                }
            )
    accepted_clusters = [
        {
            "cluster_id": str(item.get("cluster_id", "")),
            "member_ids": list(item.get("member_ids", []) or []),
            "parent_cluster_ids": list(item.get("parent_cluster_ids", []) or []),
        }
        for item in cluster_states
        if str(item.get("final_decision", item.get("status", "")) or "") == "accept"
    ]
    dropped_clusters = [
        {
            "cluster_id": str(item.get("cluster_id", "")),
            "member_ids": list(item.get("member_ids", []) or []),
            "drop_reason": str(item.get("drop_reason", "") or ""),
            "primary_reason_code": (
                list(
                    dict(item.get("report_draft", {}) or {})
                    .get("verifier_decision", {})
                    .get("reason_codes", [])
                    or [str(item.get("drop_reason", "") or "")]
                )[0]
            ),
            "reason_codes": list(
                dict(item.get("report_draft", {}) or {})
                .get("verifier_decision", {})
                .get("reason_codes", [])
                or ([str(item.get("drop_reason", "") or "")] if str(item.get("drop_reason", "") or "") else [])
            ),
            "parent_cluster_ids": list(item.get("parent_cluster_ids", []) or []),
        }
        for item in cluster_states
        if str(item.get("final_decision", item.get("status", "")) or "") == "drop"
    ]
    review_unavailable_clusters = [
        {
            "cluster_id": str(item.get("cluster_id", "")),
            "member_ids": list(item.get("member_ids", []) or []),
            "reason": str(item.get("review_unavailable_reason", "") or ""),
            "parent_cluster_ids": list(item.get("parent_cluster_ids", []) or []),
        }
        for item in cluster_states
        if str(item.get("status", "") or "") == REVIEW_UNAVAILABLE_STATUS
    ]
    absorbed_clusters = [
        {
            "cluster_id": str(item.get("cluster_id", "")),
            "member_ids": list(item.get("member_ids", []) or []),
            "parent_cluster_ids": list(item.get("parent_cluster_ids", []) or []),
            "absorbed_into": str(item.get("absorbed_into", "") or ""),
            "absorbed_by": str(item.get("absorbed_by", "") or ""),
        }
        for item in cluster_states
        if str(item.get("absorbed_into", "") or "")
    ]
    revision_lineage = [
        {
            "cluster_id": str(item.get("cluster_id", "")),
            "revision_history": list(item.get("revision_history", []) or []),
        }
        for item in cluster_states
        if list(item.get("revision_history", []) or [])
    ]
    budget_exhausted_clusters = [
        {
            "cluster_id": str(item.get("cluster_id", "")),
            "reason": str(
                dict(item.get("budget_state", {}) or {}).get(
                    "budget_exhausted_reason", ""
                )
                or ""
            ),
        }
        for item in cluster_states
        if bool(dict(item.get("budget_state", {}) or {}).get("budget_exhausted", False))
    ]
    revision_history = [
        {
            "cluster_id": str(item.get("cluster_id", "")),
            "source_cluster_id": str(revision.get("source_cluster_id", "")),
            "generated_cluster_id": str(revision.get("updated_cluster_id", "")),
            "generated_clusters": [
                str(child.get("cluster_id", ""))
                for child in list(item.get("generated_clusters", []) or [])
                if str(child.get("cluster_id", ""))
            ],
            "source_round": int(item.get("review_round", 0) or 0),
            "action": str(revision.get("action", "")),
            "evidence_summary": str(
                dict(item.get("verifier_decision", {}) or {}).get("rationale", "")
                or dict(item.get("llm_audit", {}) or {}).get("reasoning_summary", "")
            ),
            "rationale": str(
                dict(item.get("router_decision", {}) or {}).get("route_reason", "")
                or dict(item.get("llm_audit", {}) or {}).get("reasoning_summary", "")
            ),
        }
        for item in cluster_states
        for revision in list(item.get("revision_history", []) or [])
    ]
    candidate_consensus = dict(
        dict(candidate_clusters[0]).get("consensus", {}) if candidate_clusters else {}
    )
    candidate_k_diagnostics = {
        "best_n_clusters": candidate_consensus.get("best_n_clusters"),
        "selection_metric": candidate_consensus.get("selection_metric"),
        "selected_k_reason": candidate_consensus.get("selected_k_reason", ""),
        "candidate_k_diagnostics_path": candidate_consensus.get(
            "candidate_k_diagnostics_path", ""
        ),
        "diagnostics": list(candidate_consensus.get("candidate_k_diagnostics", []) or []),
    }
    all_dropped = bool(all_cluster_reports) and all(
        item.get("action") == "drop" for item in all_cluster_reports
    )
    all_unavailable = bool(all_cluster_reports) and all(
        item.get("action") == REVIEW_UNAVAILABLE_STATUS for item in all_cluster_reports
    )
    high_confidence_accepted_count = sum(
        1
        for item in all_cluster_reports
        if item.get("action") == "accept" and item.get("high_confidence")
    )
    unavailable_count = int(action_counts.get(REVIEW_UNAVAILABLE_STATUS, 0) or 0)
    review_convergence_summary = {
        "all_clusters_dropped": all_dropped,
        "all_reviews_unavailable": all_unavailable,
        "interpretation": (
            "Subtype review did not complete because no usable LLM action decision was available."
            if all_unavailable
            else "Some subtype reviews were unavailable, and no high-confidence accepted subtype can be concluded from this run."
            if unavailable_count and not high_confidence_accepted_count
            else "No reliable new subtype was identified under the selected K."
            if all_dropped
            else "At least one cluster reached a high-confidence accepted subtype action."
            if high_confidence_accepted_count
            else "No high-confidence accepted subtype was identified under the selected K."
        ),
        "action_counts": action_counts,
        "high_confidence_accepted_count": high_confidence_accepted_count,
        "non_high_confidence_final_count": sum(
            1
            for item in all_cluster_reports
            if item.get("action") in FINAL_REVIEW_ACTIONS
            and not item.get("high_confidence")
        ),
        "reason_code_counts": reason_code_counts,
        "dominant_reason_code": max(reason_code_counts, key=reason_code_counts.get)
        if reason_code_counts
        else "",
    }
    return {
        "stage": "logic_v_sub_v2_coordinator",
        "input_case_count": len(patient_states),
        "passed_qc_patient_ids": [
            str(item.get("case_id", ""))
            for item in patient_states
            if item.get("qc") == "success"
        ],
        "excluded_patient_ids": [
            str(item.get("case_id", ""))
            for item in patient_states
            if item.get("qc") != "success"
        ],
        "candidate_cluster_ids": [
            str(item.get("cluster_id", "") or "")
            for item in candidate_clusters
            if str(item.get("cluster_id", "") or "")
        ],
        "cluster_reports": final_reports,
        "final_review_summary": {
            "overview": {
                "input_cluster_count": len(candidate_clusters),
                "final_report_count": len(all_cluster_reports),
                "action_counts": action_counts,
                "patient_coverage": {
                    "input_patient_count": len(all_patient_ids),
                    "qc_passed_patient_count": len(qc_passed_patient_ids),
                    "covered_patient_count": len(final_member_ids),
                    "covered_fraction": (
                        len(final_member_ids) / len(qc_passed_patient_ids)
                        if qc_passed_patient_ids
                        else 0.0
                    ),
                    "covered_fraction_of_all_input": (
                        len(final_member_ids) / len(all_patient_ids)
                        if all_patient_ids
                        else 0.0
                    ),
                },
            },
            "candidate_k_diagnostics": candidate_k_diagnostics,
            "global_figures": global_figures,
            "visualization_policy": {
                "purpose": "Figures are limited to auditable evidence for the six subtype-review validation dimensions.",
                "default_figures": [
                    "consensus_matrix_heatmap",
                    "integrated_snf_embedding_scatter",
                    "modality_embedding_4panel",
                    "rna_hallmark_ssgsea_bubble",
                    "mutation_gene_oncoplot",
                    "mutation_kegg_pathway_bubble",
                    "mutation_reactome_pathway_bubble",
                    "mutation_pathway_bubble",
                    "rna_hallmark_ssgsea_heatmap",
                    "rna_subtype_signature_dotplot",
                    "rna_subtype_defining_hallmark_heatmap",
                    "rna_top_pathway_boxplots",
                    "subtype_evidence_score_panel",
                    "survival_global_km",
                    "confounder_association_heatmap",
                    "known_label_association_heatmap",
                    "cluster_member_consensus_support",
                    "selected_pathway_boxplot",
                ],
                "table_only_dimensions": [
                    "set_reliability",
                    "biological_support",
                    "known_label_echo",
                    "confounder_exclusion",
                    "clinical_context",
                    "multimodal_support",
                ],
                "not_generated_by_default": [
                    "montage",
                    "integrated_subtype_landscape",
                    "review_convergence_visualization",
                    "final_decision_matrix",
                    "action_count_barplot",
                    "reason_count_barplot",
                    "cluster_size_barplot",
                    "rna_violin",
                    "clinical_covariate_barplot",
                    "known_label_echo_figures",
                    "confounder_figures",
                    "ct_feature_heatmap",
                    "pathology_feature_heatmap",
                ],
            },
            "review_convergence_summary": review_convergence_summary,
            "accepted_candidate_subtypes": accepted_candidate_subtypes,
            "all_cluster_reports": all_cluster_reports,
            "revision_history": revision_history,
            "accepted_clusters": accepted_clusters,
            "dropped_clusters": dropped_clusters,
            "review_unavailable_clusters": review_unavailable_clusters,
            "absorbed_clusters": absorbed_clusters,
            "revision_lineage": revision_lineage,
            "budget_exhausted_clusters": budget_exhausted_clusters,
        },
        "patient_store": dict(patient_store_paths),
        "cluster_store": dict(cluster_store_paths),
        "graphs": dict(graph_paths),
    }
