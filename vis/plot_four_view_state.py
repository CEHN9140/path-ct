#!/usr/bin/env python3
"""Create publication-oriented figures and tables for the frozen four-view states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.sans-serif": ["Noto Sans CJK SC"],
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output_kirc_v14/11_four_view_no_cnv/inputs/four_view_no_cnv"
from tools.ct_radiomics import build_ct_discovery_feature_matrix  # noqa: E402
from tools import post_discovery_characterization as stats  # noqa: E402
from utils.llm_utils import load_yaml_file  # noqa: E402

INPUT = ROOT / "output_kirc_v14/11_four_view_no_cnv/inputs/four_view_no_cnv"
STATE = ROOT / "output_kirc_v14/12_four_view_core_to_macro_state_audit/final_macro_state_membership.csv"
CHAR = ROOT / "output_kirc_v14/14_four_view_state_characterization"
MAP = ROOT / "output_kirc_v14/15_four_view_state_known_ccrcc_mapping"
ROBUST = ROOT / "output_kirc_v14/13_four_view_macro_state_robustness"
COMPLETION = ROOT / "output_kirc_v14/18_four_view_state_result_completion"
AUDIT = ROOT / "output_kirc_v14/12_four_view_core_to_macro_state_audit"
COLORS = {"STATE_A": "#3366a8", "STATE_B": "#d95f02", "STATE_C": "#1b9e77", "STATE_D": "#7570b3"}
STATE_ORDER = ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]
RAW = ROOT / "output_kirc_raw"


def save(fig, out, name):
    for axis in fig.axes:
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def pcoa(similarity):
    sim = np.asarray(similarity, float)
    sim = (sim + sim.T) / 2
    distance = np.sqrt(np.clip(1 - sim, 0, None))
    n = len(distance)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance ** 2) @ centering
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    keep = np.maximum(values[:2], 0)
    return vectors[:, :2] * np.sqrt(keep)


def heatmap(frame, out, name, title, cmap="vlag", center=0, vmin=None, vmax=None, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(7.2, max(4.3, min(10, 0.28 * len(frame) + 2))))
    sns.heatmap(frame, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                annot=frame.size <= 80, fmt=fmt, xticklabels=False if len(frame.columns) > 40 else True,
                linewidths=.5, linecolor="white", cbar_kws={"label": "value"})
    ax.set_title(title, pad=12, weight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, out, name)


def consensus_heatmap(matrix, labels, out):
    order = np.argsort([{"STATE_A": 0, "STATE_B": 1, "STATE_C": 2, "STATE_D": 3}[x] for x in labels])
    matrix, labels = matrix[np.ix_(order, order)], np.asarray(labels)[order]
    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[.15, 1], height_ratios=[.15, 1], wspace=.02, hspace=.02)
    top, side, ax = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    colors = [COLORS[x] for x in labels]
    for strip, axis in [(colors, top), (colors, side)]:
        axis.imshow(np.array([[mpl.colors.to_rgba(c) for c in strip]] if axis is top else [[mpl.colors.to_rgba(c)] for c in strip]))
        axis.axis("off")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="none", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Patients ordered by macro-state", labelpad=8)
    ax.set_ylabel("Patients ordered by macro-state", labelpad=8)
    ax.set_title("K=4 consensus co-assignment", pad=10, weight="bold")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS[x], markersize=7, label=x) for x in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, frameon=False, fontsize=8)
    save(fig, out, "figure_k4_consensus_heatmap")


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_table(path):
    return pd.read_csv(path).set_index("case_id")


def representative_cases(similarity, case_ids, state_by_case, per_state=3):
    """Select modality-specific medoids, with deterministic within-state diversity."""
    similarity = np.asarray(similarity, float)
    rows = []
    for state in STATE_ORDER:
        ids = [case_id for case_id in case_ids if state_by_case[case_id] == state]
        indices = [case_ids.index(case_id) for case_id in ids]
        sub = similarity[np.ix_(indices, indices)]
        selected = []
        for _ in range(min(per_state, len(ids))):
            scores = sub.mean(axis=1) if not selected else sub.mean(axis=1) - sub[:, selected].max(axis=1)
            scores[selected] = -np.inf
            selected.append(int(np.argmax(scores)))
        rows.extend({"state_id": state, "case_id": ids[index], "state_n": len(ids), "representative_rank": rank + 1}
                    for rank, index in enumerate(selected))
    return pd.DataFrame(rows)


def resolve_path(path):
    if not str(path).strip():
        return Path("__missing_visualization_input__")
    path = Path(str(path))
    if path.exists():
        return path
    return Path(str(path).replace("/output_kirc/", "/output_kirc_raw/"))


def load_patient_state_records():
    path = RAW / "storage/patient_states/patient_states.jsonl"
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            records[str(record["case_id"])] = record
    return records


def plot_ct_representatives(representatives, out):
    import nibabel as nib

    def crop_to_mask(image, mask, margin=.2):
        ys, xs = np.where(mask)
        if not len(xs):
            return image, mask
        dx = max(8, int((xs.max() - xs.min() + 1) * margin))
        dy = max(8, int((ys.max() - ys.min() + 1) * margin))
        x0, x1 = max(0, xs.min() - dx), min(image.shape[1], xs.max() + dx + 1)
        y0, y1 = max(0, ys.min() - dy), min(image.shape[0], ys.max() + dy + 1)
        return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    rows = []
    fig, axes = plt.subplots(len(representatives), 3, figsize=(8.8, 2.45 * len(representatives)), squeeze=False)
    state_by_case = dict(zip(representatives.case_id, representatives.state_id))
    for row_index, item in representatives.iterrows():
        case_id, state = item.case_id, item.state_id
        radiomics_json = RAW / f"ct_radiomics/{case_id}.json"
        if not radiomics_json.exists():
            radiomics_json = ROOT / f"output_kirc/ct_radiomics/{case_id}.json"
        if not radiomics_json.exists():
            continue
        payload = json.loads(radiomics_json.read_text()).get("payload", {})
        ct_path = resolve_path(payload.get("ct_path", ""))
        mask_path = resolve_path(payload.get("mask_path", ""))
        if not ct_path.exists() or not mask_path.exists():
            continue
        image = np.asarray(nib.load(str(ct_path)).get_fdata(), dtype=float)
        mask = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
        if image.shape != mask.shape or not mask.any():
            continue
        center = np.argwhere(mask).mean(axis=0).round().astype(int)
        z = int(np.argmax(mask.sum(axis=(0, 1))))
        slices = [(image[:, :, z].T, mask[:, :, z].T, "Axial"),
                  (image[:, center[1], :].T, mask[:, center[1], :].T, "Coronal"),
                  (image[center[0], :, :].T, mask[center[0], :, :].T, "Sagittal")]
        for col, (slice_image, slice_mask, view) in enumerate(slices):
            slice_image, slice_mask = crop_to_mask(slice_image, slice_mask)
            axis = axes[row_index, col]
            axis.imshow(np.clip(slice_image, -150, 250), cmap="gray", vmin=-150, vmax=250)
            if slice_mask.any():
                axis.contour(slice_mask, levels=[.5], colors="#e66101", linewidths=1.1)
            axis.set_title(view, fontsize=9)
            axis.set_aspect("equal")
            axis.axis("off")
            if col == 0:
                axis.text(-.04, .5, f"{state}\n{case_id}", transform=axis.transAxes, ha="right", va="center", fontsize=8, color=COLORS[state], weight="bold")
        rows.append({"state_id": state, "case_id": case_id, "ct_path": str(ct_path), "mask_path": str(mask_path), "axial_slice": z})
    fig.suptitle("Representative CT cases by macro-state", y=.995, weight="bold")
    fig.text(.5, .01, "Window: [-150, 250] HU; orange contour: tumor mask", ha="center", fontsize=8)
    fig.subplots_adjust(top=.94, bottom=.05, wspace=.03, hspace=.18)
    save(fig, out, "figure_ct_representative_cases")
    return rows


def plot_wsi_representatives(representatives, out):
    import h5py
    import torch
    from PIL import Image

    slide_vectors, available = {}, {}
    for case_id in representatives.case_id:
        json_path = RAW / f"wsi_embeddings/{case_id}.json"
        if not json_path.exists():
            json_path = ROOT / f"output_kirc/wsi_embeddings/{case_id}.json"
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text()).get("tool_result", {})
        artifact = payload.get("artifacts", {})
        embedding_path = resolve_path(artifact.get("slide_embedding_npy_path", ""))
        tile_path = resolve_path(artifact.get("tile_embeddings_path", ""))
        coords_path = resolve_path(artifact.get("tumor_coordinates_h5_path", ""))
        patch_dir = resolve_path(artifact.get("patch_dir", ""))
        if embedding_path.exists() and tile_path.exists() and coords_path.exists() and patch_dir.exists():
            slide_vectors[case_id] = np.load(embedding_path).astype(float)
            available[case_id] = (tile_path, coords_path, patch_dir)
    if not slide_vectors:
        return []
    vector_frame = pd.DataFrame(slide_vectors).T
    selected = []
    for state in STATE_ORDER:
        candidates = representatives.loc[representatives.state_id == state, "case_id"].tolist()
        candidates = [case_id for case_id in candidates if case_id in slide_vectors]
        if not candidates:
            continue
        state_center = vector_frame.loc[candidates].mean(axis=0).to_numpy()
        selected.extend((state, case_id) for case_id in sorted(candidates, key=lambda x: float(np.linalg.norm(slide_vectors[x] - state_center))))

    sampled_tiles, patient_records = {}, {}
    for state, case_id in selected:
        tile_path, coords_path, patch_dir = available[case_id]
        tiles = torch.load(str(tile_path), map_location="cpu", weights_only=True).numpy().astype(float)
        with h5py.File(coords_path, "r") as handle:
            coordinates = np.asarray(handle["coordinates"], dtype=int)
        if len(tiles) != len(coordinates):
            raise ValueError(f"WSI tile/coordinate length mismatch for {case_id}")
        sample_index = np.linspace(0, len(tiles) - 1, min(200, len(tiles)), dtype=int)
        sampled_tiles[(state, case_id)] = tiles[sample_index]
        patient_records[(state, case_id)] = (tiles, coordinates, patch_dir)

    state_centers = {state: np.vstack([sampled_tiles[key] for key in sampled_tiles if key[0] == state]).mean(axis=0) for state in STATE_ORDER}
    norms = {state: np.linalg.norm(center) or 1.0 for state, center in state_centers.items()}
    selected_rows = []
    figures = [("prototype", "State-prototypical WSI tumor patches", "figure_wsi_state_prototypical_patches"),
               ("discriminative", "State-discriminative WSI tumor patches", "figure_wsi_state_discriminative_patches")]
    selected_by_state = {state: [case_id for state_id, case_id in selected if state_id == state] for state in STATE_ORDER}
    for selection_type, title, filename in figures:
        fig, axes = plt.subplots(4, 6, figsize=(12.0, 8.2), squeeze=False)
        for row_index, state in enumerate(STATE_ORDER):
          for patient_index, case_id in enumerate(selected_by_state[state]):
            tiles, coordinates, patch_dir = patient_records[(state, case_id)]
            tile_norm = np.linalg.norm(tiles, axis=1) * norms[state]
            own_score = tiles @ state_centers[state] / np.maximum(tile_norm, 1e-12)
            other_score = np.column_stack([tiles @ state_centers[other] / np.maximum(np.linalg.norm(tiles, axis=1) * norms[other], 1e-12) for other in STATE_ORDER if other != state]).max(axis=1)
            score = own_score if selection_type == "prototype" else own_score - other_score
            chosen = np.argsort(score)[::-1][:2]
            for col, patch_index in enumerate(chosen):
                panel_col = patient_index * 2 + col
                x, y = coordinates[patch_index]
                patch_path = patch_dir / f"{y}_{x}.png"
                if not patch_path.exists():
                    continue
                with Image.open(patch_path) as image:
                    axes[row_index, panel_col].imshow(image.convert("RGB"))
                axes[row_index, panel_col].set_title(f"{case_id}\n({x}, {y})", fontsize=7)
                axes[row_index, panel_col].axis("off")
                selected_rows.append({"selection_type": selection_type, "state_id": state, "case_id": case_id, "patch_index": int(patch_index), "x": int(x), "y": int(y), "patch_path": str(patch_path), "embedding_score": float(score[patch_index])})
          axes[row_index, 0].text(-.03, .5, state, transform=axes[row_index, 0].transAxes, ha="right", va="center", fontsize=10, color=COLORS[state], weight="bold")
        fig.suptitle(title, y=.995, weight="bold")
        fig.text(.5, .01, "Three WSI-affinity representative patients per state; two tumor patches per patient selected from cross-patient state prototypes", ha="center", fontsize=8)
        fig.subplots_adjust(top=.91, bottom=.07, left=.09, right=.99, wspace=.12, hspace=.36)
        save(fig, out, filename)
    return selected_rows


def plot_radio_pathological_exemplars(ct_rows, wsi_rows, out):
    from PIL import Image
    import nibabel as nib

    ct_by_state = {row["state_id"]: row for row in ct_rows if row.get("representative_rank", 1) == 1}
    prototype_rows = [row for row in wsi_rows if row["selection_type"] == "prototype"]
    wsi_by_state = {state: [row for row in prototype_rows if row["state_id"] == state][:2] for state in STATE_ORDER}
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 7.8), squeeze=False)
    for col, state in enumerate(STATE_ORDER):
        axes[0, col].set_title(state, color=COLORS[state], weight="bold")
        for row_index, patch in enumerate(wsi_by_state[state], 1):
            with Image.open(patch["patch_path"]) as image:
                axes[row_index - 1, col].imshow(image.convert("RGB"))
            axes[row_index - 1, col].set_title(f"WSI\n{patch['case_id']}", fontsize=7)
            axes[row_index - 1, col].axis("off")
        ct = ct_by_state.get(state)
        if ct:
            image = np.asarray(nib.load(ct["ct_path"]).get_fdata(), dtype=float)
            mask = np.asarray(nib.load(ct["mask_path"]).get_fdata()) > 0
            z = int(ct["axial_slice"])
            axes[2, col].imshow(np.clip(image[:, :, z].T, -150, 250), cmap="gray", vmin=-150, vmax=250)
            axes[2, col].contour(mask[:, :, z].T, levels=[.5], colors="#e66101", linewidths=1.0)
            axes[2, col].set_title(f"CT\n{ct['case_id']}", fontsize=7)
        axes[2, col].axis("off")
    axes[0, 0].set_ylabel("WSI\nprototype", rotation=0, labelpad=30, va="center", fontsize=8)
    axes[1, 0].set_ylabel("WSI\nprototype", rotation=0, labelpad=30, va="center", fontsize=8)
    axes[2, 0].set_ylabel("CT\naxial", rotation=0, labelpad=30, va="center", fontsize=8)
    fig.suptitle("Radio-pathological exemplars of four macro-states", y=.995, weight="bold")
    fig.text(.5, .01, "Patients selected using modality-specific affinity medoids; orange contour denotes the CT tumor mask", ha="center", fontsize=8)
    fig.subplots_adjust(top=.91, bottom=.07, left=.10, right=.99, wspace=.08, hspace=.28)
    save(fig, out, "figure_radio_pathological_exemplars")


def plot_completion_heatmap(path, out, name, title, label_prefix):
    frame = pd.read_csv(path).set_index("state_id")
    frame = frame.drop(columns=[x for x in frame if x.startswith("state_n")], errors="ignore")
    frame = frame.apply(pd.to_numeric, errors="coerce").reindex(STATE_ORDER)
    frame = frame.loc[:, frame.notna().any()]
    if frame.empty:
        return
    z = frame.sub(frame.mean(axis=0), axis=1).div(frame.std(axis=0).replace(0, np.nan), axis=1).fillna(0)
    z.index = [f"{label_prefix} | {str(x).replace('_', ' ').title()}" for x in z.index]
    fig, ax = plt.subplots(figsize=(7.8, max(4.0, .28 * len(z) + 1.8)))
    sns.heatmap(z, ax=ax, cmap="vlag", center=0, vmin=-2.2, vmax=2.2, annot=frame.round(2), fmt=".2f",
                linewidths=.5, linecolor="white", cbar_kws={"label": "Within-feature z-score"})
    ax.set_xlabel("Macro-state"); ax.set_ylabel(""); ax.set_title(title, weight="bold", pad=12)
    save(fig, out, name)


def plot_noncore_stability(out):
    frame = pd.read_csv(COMPLETION / "noncore_uncertainty_scores.csv")
    frame["group"] = np.where(frame.is_non_core, "Non-core", "State-assigned")
    metrics = [("max_coassignment", "Maximum co-assignment"), ("state_affinity_margin", "Nearest-state affinity margin"), ("state_affinity_entropy", "State-affinity entropy")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), squeeze=False)
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = [frame.loc[frame.group == group, metric].dropna().to_numpy() for group in ["State-assigned", "Non-core"]]
        axis.boxplot(values, tick_labels=["Assigned", "Non-core"], patch_artist=True,
                     boxprops={"facecolor": "#d9d9d9"}, medianprops={"color": "#222222"})
        for i, vals in enumerate(values, 1):
            rng = np.random.default_rng(20260918 + i)
            axis.scatter(rng.normal(i, .04, len(vals)), vals, s=10, alpha=.65, color=["#3366a8", "#999999"][i - 1])
        axis.set_title(title, fontsize=9); axis.grid(axis="y", color="#eeeeee")
    fig.suptitle("Post hoc stability profile of state-assigned and non-core patients", weight="bold", y=1.02)
    fig.text(.5, .01, "Non-core patients are not assigned to a macro-state; this figure describes recurrent-membership uncertainty only.", ha="center", fontsize=8)
    fig.subplots_adjust(top=.80, bottom=.22, wspace=.35)
    save(fig, out, "supplement_noncore_stability")


def run(output_dir=Path("vis/figs")):
    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    membership = pd.read_csv(STATE, dtype=str)
    state_by_case = membership.set_index("case_id")["state_id"].to_dict()
    with (INPUT / "storage/patient_states/patient_states.jsonl").open(encoding="utf-8") as handle:
        states = {
            str(state["case_id"]): state
            for line in handle if line.strip()
            for state in [json.loads(line)]
            if state.get("qc") == "success"
        }
    order = json.loads((INPUT / "candidate_subtype/affinity_patient_order.json").read_text())
    fused = np.load(INPUT / "candidate_subtype/fused_similarity.npy")
    state_rank = {state: i for i, state in enumerate(["STATE_A", "STATE_B", "STATE_C", "STATE_D"])}
    core_order = sorted(
        (case_id for case_id in order if case_id in state_by_case),
        key=lambda case_id: (state_rank[state_by_case[case_id]], case_id),
    )
    positions = [order.index(case_id) for case_id in core_order]
    coords = pcoa(fused[np.ix_(positions, positions)])
    labels = [state_by_case[case_id] for case_id in core_order]
    ct_similarity = np.load(INPUT / "candidate_subtype/ct_affinity.npy")
    wsi_similarity = np.load(INPUT / "candidate_subtype/wsi_affinity.npy")
    ct_representatives = representative_cases(ct_similarity[np.ix_(positions, positions)], core_order, state_by_case)
    wsi_representatives = representative_cases(wsi_similarity[np.ix_(positions, positions)], core_order, state_by_case)
    ct_rows = plot_ct_representatives(ct_representatives, out)
    wsi_rows = plot_wsi_representatives(wsi_representatives, out)
    pd.DataFrame(ct_rows + wsi_rows).to_csv(out / "table_representative_imaging_cases.csv", index=False)

    # Figure 1: fused geometry and macro-state discovery diagnostics.
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    for group in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]:
        idx = np.array(labels) == group
        ax.scatter(coords[idx, 0], coords[idx, 1], s=28, alpha=.85, label=group, c=COLORS[group], edgecolor="none")
    ax.set(title="Fused-space visualization of four-view macro-states", xlabel="PCoA 1", ylabel="PCoA 2")
    ax.legend(frameon=True, title="Patient label", ncol=2)
    save(fig, out, "figure_fused_pcoa")

    coassign_path = INPUT / "candidate_subtype/consensus_cluster/matrices/consensus_matrix_K4.npy"
    if not coassign_path.exists():
        coassign_path = ROOT / "output_kirc_v14/11_four_view_no_cnv/agent_review/joint_accepted_coassignment_matrix.csv"
    coassign = np.load(coassign_path) if coassign_path.suffix == ".npy" else pd.read_csv(coassign_path, index_col=0).to_numpy(float)
    coassign = coassign[np.ix_(positions, positions)]
    consensus_heatmap(coassign, labels, out)

    k = read_csv(ROBUST / "macro_k_summary.csv")
    if not k.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for linkage, sub in k.groupby("linkage"):
            ax.plot(sub["macro_k"], sub["silhouette"], marker="o", label=linkage)
        ax.axvline(4, ls="--", color="black", lw=1, label="selected K=4")
        ax.set(xlabel="Macro-state K", ylabel="Core-level silhouette", title="Macro-state resolution diagnostics")
        ax.legend()
        save(fig, out, "figure_macro_k_stability")

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, modality in zip(axes.flat, ["ct", "wsi", "rna", "wxs"]):
        path = INPUT / "candidate_subtype" / f"{modality}_affinity.npy" if modality != "wxs" else INPUT / "wxs/wxs_affinity.npy"
        matrix = np.load(path)
        xy = pcoa(matrix[np.ix_(positions, positions)])
        for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]:
            idx = np.array([state_by_case[case_id] == state for case_id in core_order])
            ax.scatter(xy[idx, 0], xy[idx, 1], s=18, color=COLORS[state], label=state)
        ax.set_title(modality.upper())
        ax.set_xlabel("PCoA 1"); ax.set_ylabel("PCoA 2")
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle("Per-view state overlays (86 state patients)", y=1.01, weight="bold")
    save(fig, out, "supplement_per_view_pcoa")

    # Figure 2: state biology and known taxonomy.
    rna_omnibus = read_csv(CHAR / "rna_hallmark_omnibus.csv")
    # Show a readable representative set in figures; full inference remains in
    # rna_hallmark_omnibus.csv and is not restricted to this plotting subset.
    selected = rna_omnibus.sort_values(["q_value", "epsilon_squared"], ascending=[True, False]).head(15).feature.tolist()
    if selected:
        rna_genes = load_table(INPUT / "rna/case_pathway_features.csv").reindex(core_order)
        gmt = load_yaml_file(ROOT / "configs/subtype_review.yaml")["rna"]["hallmark_gene_sets_path"]
        gene_sets = {}
        for line in Path(gmt).read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) > 2:
                gene_sets[fields[0]] = sorted(set(fields[2:]) & set(rna_genes.columns))
        gene_sets = {name: [g for g in genes if g in rna_genes.columns] for name, genes in gene_sets.items()}
        gene_sets = {name: genes for name, genes in gene_sets.items() if len(genes) >= 15}
        from gseapy import ssgsea
        enrichment = ssgsea(
            data=rna_genes.transpose(), gene_sets=gene_sets, outdir=None, no_plot=True,
            threads=1, min_size=15, verbose=False, seed=123,
        ).res2d
        scores = enrichment.pivot(index="Name", columns="Term", values="NES")[selected].apply(pd.to_numeric, errors="coerce")
        state_means = pd.DataFrame({state: scores[[x == state for x in labels]].mean() for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]})

        # Patient-level pathway heatmap, matching the common subtype-figure
        # convention: rows are pathways, columns are patients ordered by state.
        pathway_z = scores[selected].sub(scores[selected].mean()).div(scores[selected].std().replace(0, np.nan)).T
        pathway_z.index = [x.replace("HALLMARK_", "").replace("_", " ").title() for x in pathway_z.index]
        pathway_z = pathway_z.loc[:, core_order].clip(-2.5, 2.5)
        fig = plt.figure(figsize=(11.5, max(4.8, .34 * len(pathway_z) + 1.7)), constrained_layout=False)
        grid = fig.add_gridspec(2, 2, width_ratios=[10, .38], height_ratios=[.42, len(pathway_z)], wspace=.035, hspace=.06)
        annotation_ax = fig.add_subplot(grid[0, 0])
        pathway_ax = fig.add_subplot(grid[1, 0])
        cbar_ax = fig.add_subplot(grid[1, 1])
        annotation_ax.imshow(
            [[mpl.colors.to_rgba(COLORS[state_by_case[x]]) for x in core_order]],
            aspect="auto", extent=(0, len(core_order), 0, 1), interpolation="none",
        )
        annotation_ax.set_xlim(0, len(core_order))
        annotation_ax.set_xticks([]); annotation_ax.set_yticks([])
        start = 0
        for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]:
            count = sum(state_by_case[x] == state for x in core_order)
            annotation_ax.text(start + count / 2, .5, state.replace("STATE_", "Subtype "),
                               ha="center", va="center", color="white", weight="bold", fontsize=12)
            if start:
                annotation_ax.axvline(start, color="white", linewidth=2.2)
            start += count
        sns.heatmap(pathway_z, ax=pathway_ax, cmap="vlag", center=0, vmin=-2.5, vmax=2.5,
                    xticklabels=False, yticklabels=True, linewidths=0, cbar=True,
                    cbar_ax=cbar_ax)
        pathway_ax.set_xlabel("")
        pathway_ax.set_ylabel("")
        pathway_ax.tick_params(axis="y", labelsize=11, length=0)
        cbar_ax.set_ylabel("Relative pathway activity", fontsize=12, fontweight="bold")
        cbar_ax.tick_params(labelsize=10)
        boundaries = np.cumsum([sum(state_by_case[x] == state for x in core_order)
                                for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]])[:-1]
        for boundary in boundaries:
            pathway_ax.axvline(boundary, color="white", linewidth=1.8, zorder=5)
            annotation_ax.axvline(boundary, color="white", linewidth=2.2, zorder=5)
        fig.subplots_adjust(left=.16, right=.96, top=.96, bottom=.12)
        save(fig, out, "figure_rna_pathway_heatmap")

        plot_features = selected[:4]
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), squeeze=False)
        for ax, feature in zip(axes.flat, plot_features):
            values = [scores.loc[[x == state for x in labels], feature].dropna().to_numpy() for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]]
            ax.boxplot(values, tick_labels=["A", "B", "C", "D"], patch_artist=True,
                       boxprops={"facecolor": "#e6e6e6"}, medianprops={"color": "black"})
            for i, (state, vals) in enumerate(zip(["STATE_A", "STATE_B", "STATE_C", "STATE_D"], values), 1):
                rng = np.random.default_rng(20260920 + i)
                ax.scatter(rng.normal(i, .045, len(vals)), vals, s=9, alpha=.65, color=COLORS[state], zorder=3)
            ax.set_title(feature.replace("HALLMARK_", "").replace("_", " ").title(), fontsize=9)
            ax.set_ylabel("ssGSEA score")
        fig.suptitle("RNA pathway activity by macro-state", y=.995, weight="bold")
        fig.subplots_adjust(top=.88, wspace=.28, hspace=.38)
        save(fig, out, "figure_rna_pathway_boxplots")

    # Gene-level RNA characterization: full normalized expression, global BH-FDR.
    rna_expression = load_table(INPUT / "rna/case_pathway_features.csv").reindex(core_order).apply(pd.to_numeric, errors="coerce")
    gene_table = {case_id: rna_expression.loc[case_id].to_dict() for case_id in core_order}
    gene_rows = stats.continuous_omnibus(gene_table, list(rna_expression.columns),
                                         {state: [x for x in core_order if state_by_case[x] == state] for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]})
    gene_results = pd.DataFrame(gene_rows)
    gene_results.to_csv(out / "table_rna_gene_characterization.csv", index=False)
    gene_results = gene_results[gene_results.q_value < .05].sort_values(["q_value", "epsilon_squared"], ascending=[True, False]).head(30)
    if not gene_results.empty:
        genes = gene_results.feature.tolist()
        z = rna_expression[genes].sub(rna_expression[genes].mean()).div(rna_expression[genes].std().replace(0, np.nan)).T
        z = z.loc[genes, core_order].clip(-2.5, 2.5)

        # Standard patient-level expression heatmap: state blocks, row z-scores,
        # and an explicit state annotation bar above the samples.
        fig = plt.figure(figsize=(10.2, max(5.0, 0.22 * len(genes) + 2.4)), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[10, .35], height_ratios=[.22, 1], wspace=.04, hspace=.03)
        annotation_ax = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[1, 0])
        cbar_ax = fig.add_subplot(grid[1, 1])
        state_codes = np.array([state_rank[state_by_case[case_id]] for case_id in core_order])[None, :]
        annotation_ax.imshow(state_codes, aspect="auto", cmap=mpl.colors.ListedColormap([COLORS[s] for s in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]]), vmin=0, vmax=3)
        annotation_ax.set_xticks([]); annotation_ax.set_yticks([]); annotation_ax.set_ylabel("State", rotation=0, labelpad=28)
        sns.heatmap(z, ax=ax, cmap="vlag", center=0, vmin=-2.5, vmax=2.5, xticklabels=False,
                    yticklabels=True, linewidths=0, cbar=True, cbar_ax=cbar_ax,
                    cbar_kws={"label": "RNA expression (row z-score)"})
        ax.set_xlabel("Patients ordered by macro-state"); ax.set_ylabel("Gene")
        ax.set_title("RNA expression patterns across four macro-states", pad=10, weight="bold")
        boundaries = np.cumsum([sum(state_by_case[x] == state for x in core_order) for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]])[:-1]
        for boundary in boundaries:
            ax.axvline(boundary, color="white", linewidth=1.2)
            annotation_ax.axvline(boundary - .5, color="white", linewidth=1.2)
        centers = []
        start = 0
        for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]:
            count = sum(state_by_case[x] == state for x in core_order)
            centers.append(start + (count - 1) / 2)
            start += count
        for center, state in zip(centers, ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]):
            annotation_ax.text(center, -.8, state, ha="center", va="bottom", fontsize=8, color=COLORS[state], weight="bold")
        save(fig, out, "figure_rna_gene_heatmap")

        # Multi-state enrichment dot plot: activity is encoded by color and
        # global omnibus FDR by bubble size.
        pathway_means = pd.DataFrame({state: scores[[state_by_case[x] == state for x in core_order]].mean() for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]})
        pathway_means = pathway_means.sub(pathway_means.mean(axis=1), axis=0).div(pathway_means.std(axis=1).replace(0, np.nan), axis=0)
        pathway_means.index.name = "pathway"
        dot = pathway_means.reset_index().melt(id_vars="pathway", var_name="state", value_name="activity_z")
        q_map = rna_omnibus.set_index("feature")["q_value"]
        dot["neg_log10_q"] = dot.pathway.map(lambda x: -np.log10(max(float(q_map[x]), 1e-300)))
        dot["pathway_label"] = dot.pathway.str.replace("HALLMARK_", "", regex=False).str.replace("_", " ", regex=False).str.title()
        pathway_order = selected[::-1]
        fig, ax = plt.subplots(figsize=(10.4, max(5.2, .34 * len(pathway_order) + 1.8)))
        fig.subplots_adjust(left=.24, right=.78, top=.94, bottom=.12)
        for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]:
            sub = dot[dot.state == state].set_index("pathway").reindex(pathway_order)
            x = np.full(len(sub), ["STATE_A", "STATE_B", "STATE_C", "STATE_D"].index(state))
            ax.scatter(x, np.arange(len(sub)), s=90 + 130 * sub.neg_log10_q.to_numpy(),
                       c=sub.activity_z.to_numpy(), cmap="vlag", vmin=-2, vmax=2,
                       edgecolors=COLORS[state], linewidths=1.0, alpha=.95)
        ax.set_xticks(range(4), ["STATE_A", "STATE_B", "STATE_C", "STATE_D"])
        ax.set_yticks(range(len(pathway_order)), [x.replace("HALLMARK_", "").replace("_", " ").title() for x in pathway_order])
        ax.set(xlabel="Macro-state", ylabel="", title="RNA Hallmark activity by macro-state")
        ax.grid(axis="x", visible=False); ax.grid(axis="y", color="#eeeeee", linewidth=.6)
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-2, 2), cmap="vlag")
        colorbar = fig.colorbar(sm, ax=ax, pad=.02, aspect=28)
        colorbar.ax.set_ylabel("")
        colorbar.ax.set_title("Relative\npathway activity", fontsize=10, pad=8)
        size_handles = []
        for q_value in (.05, .01, .001):
            size = 90 + 130 * (-np.log10(q_value))
            handle = mpl.lines.Line2D([], [], marker="o", linestyle="None",
                                      markersize=np.sqrt(size),
                                      markerfacecolor="#777777", markeredgecolor="none",
                                      label=f"q={q_value:g}")
            size_handles.append(handle)
        ax.legend(handles=size_handles, title="Global FDR", frameon=False,
                  loc="upper left", bbox_to_anchor=(1.42, .55), fontsize=9,
                  labelspacing=1.5, handletextpad=.8, borderpad=.5)
        save(fig, out, "figure_rna_pathway_dotplot")

    wxs_matrix = load_table(INPUT / "wxs/wxs_discovery_features.csv").reindex(order)
    wxs_matrix.columns = wxs_matrix.columns.str.replace("mutation::", "", regex=False)
    wxs_features = wxs_matrix.mean().sort_values(ascending=False).index.tolist()
    wxs_features = [feature for feature in wxs_features if wxs_matrix[feature].sum() > 0]
    wxs_features = wxs_features[:9]
    wxs_matrix = wxs_matrix.reindex(columns=wxs_features)
    if not wxs_matrix.empty:
        wxs_matrix = wxs_matrix.loc[core_order]
        state_names = ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]
        state_colors = [COLORS[state_by_case[x]] for x in core_order]
        n_genes = len(wxs_features)
        fig = plt.figure(figsize=(13.5, max(4.8, .34 * n_genes + 2.0)), constrained_layout=False)
        grid = fig.add_gridspec(2, 2, width_ratios=[11, 4], height_ratios=[.9, n_genes], wspace=.03, hspace=.08)
        subtype_ax = fig.add_subplot(grid[0, 0])
        subtype_right = fig.add_subplot(grid[0, 1])
        ax = fig.add_subplot(grid[1, 0])
        freq_ax = fig.add_subplot(grid[1, 1])
        fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.08, wspace=.03, hspace=0.0)
        for axis in (subtype_ax, subtype_right, ax, freq_ax):
            for spine in axis.spines.values():
                spine.set_visible(False)

        subtype_ax.imshow([[mpl.colors.to_rgba(c) for c in state_colors]], aspect="auto",
                          extent=(0, len(state_colors), 0, 1))
        subtype_ax.set_xticks([]); subtype_ax.set_yticks([])
        oncoplot_fontsize = 14
        start = 0
        for state in state_names:
            count = sum(state_by_case[x] == state for x in core_order)
            subtype_ax.text(start + count / 2, .5, state.replace("STATE_", "Subtype "),
                            ha="center", va="center", color="white", weight="bold", fontsize=oncoplot_fontsize)
            if start:
                subtype_ax.axvline(start, color="white", lw=2.5)
            start += count
        subtype_ax.set_xlim(0, len(state_colors))
        subtype_right.axis("off")

        sns.heatmap(wxs_matrix.T, cmap=mpl.colors.ListedColormap(["#f1f1f1", "#1186bd"]),
                    vmin=0, vmax=1, cbar=False, linewidths=0, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_xticks([])
        ax.set_yticks(np.arange(n_genes) + .5)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        for boundary in np.cumsum([sum(state_by_case[x] == state for x in core_order) for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]])[:-1]:
            ax.axvline(boundary, color="white", lw=3.0, zorder=5)

        frequencies = pd.DataFrame({state: wxs_matrix.loc[[state_by_case[x] == state for x in core_order]].mean()
                                    for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]})
        y = np.arange(n_genes) + .5
        state_names = ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]
        for offset, state in enumerate(state_names):
            freq_ax.barh(y, frequencies.loc[wxs_features, state].to_numpy(), left=offset,
                         height=.72, color=COLORS[state], alpha=.95)
        freq_ax.set_xlim(0, len(state_names))
        freq_ax.set_xticks([])
        freq_ax.set_xlabel("")
        freq_ax.set_yticks(y)
        freq_ax.set_yticklabels(wxs_features, fontsize=oncoplot_fontsize, fontweight="bold")
        freq_ax.yaxis.tick_right()
        freq_ax.tick_params(axis="y", length=0, pad=5)
        freq_ax.set_ylim(n_genes, 0)
        freq_ax.set_title("Mutation frequency", fontsize=oncoplot_fontsize, fontweight="bold")
        freq_ax.set_facecolor("#f1f1f1")
        for offset in range(1, len(state_names)):
            freq_ax.axvline(offset, color="white", lw=1.5)
        freq_ax.grid(False)
        save(fig, out, "figure_wxs_oncoplot")
        frequencies = pd.DataFrame({state: wxs_matrix.loc[[state_by_case[x] == state for x in core_order]].mean() for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]})

    ct = read_csv(CHAR / "ct_radiomics_omnibus.csv")
    ct = ct[ct.q_value < .05].sort_values("q_value").head(4)
    ct_post = read_csv(CHAR / "ct_radiomics_posthoc.csv")
    if not ct.empty and not ct_post.empty:
        ct_payload = build_ct_discovery_feature_matrix([states[x] for x in sorted(states)], config_dir=str(ROOT / "configs"), output_root=str(INPUT))
        ct_frame = pd.DataFrame(ct_payload["matrix"], index=ct_payload["patient_ids"], columns=ct_payload["feature_names"])
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), squeeze=False)
        short_names = {"original_shape_SurfaceVolumeRatio": "Surface / volume ratio",
                       "original_glrlm_LowGrayLevelRunEmphasis": "Low gray-level run emphasis",
                       "log-sigma-2-0-mm-3D_glrlm_RunEntropy": "Run entropy (σ=2 mm)",
                       "log-sigma-3-0-mm-3D_glrlm_RunEntropy": "Run entropy (σ=3 mm)"}
        for ax, feature in zip(axes.flat, ct.feature):
            values = [ct_frame.loc[membership.loc[membership.state_id == state, "case_id"], feature].dropna().to_numpy() for state in ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]]
            ax.boxplot(values, tick_labels=["A", "B", "C", "D"], patch_artist=True,
                       boxprops={"facecolor": "#e6e6e6"}, medianprops={"color": "black"})
            for i, (state, vals) in enumerate(zip(["STATE_A", "STATE_B", "STATE_C", "STATE_D"], values), 1):
                rng = np.random.default_rng(20260916 + i)
                ax.scatter(rng.normal(i, .045, len(vals)), vals, s=10, alpha=.65, color=COLORS[state], zorder=3)
            ax.set(title=short_names.get(feature, feature), ylabel="Feature value")
        ct_state_means = pd.DataFrame({state: [ct_frame.loc[membership.loc[membership.state_id == state, "case_id"], feature].mean() for feature in ct.feature] for state in STATE_ORDER}, index=["CT | " + short_names.get(feature, feature) for feature in ct.feature])
        fig.suptitle("Top CT radiomics features", y=.995, weight="bold")
        fig.subplots_adjust(top=.88, wspace=.28, hspace=.40)
        save(fig, out, "figure_ct_radiomics")
        effect_rows = []
        for feature in ct.feature:
            rows = ct_post[ct_post.feature == feature]
            for _, row in rows.iterrows():
                effect_rows.append({"feature": feature, "comparison": f"{row.group_a} vs {row.group_b}", "effect": row.cliffs_delta})
        effect = pd.DataFrame(effect_rows)
        effect = effect[effect.feature.isin(ct.feature)].copy()
        effect["feature"] = effect.feature.map(short_names).fillna(effect.feature)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        sns.stripplot(data=effect, x="effect", y="feature", hue="comparison", dodge=False, size=6, palette="Set2", ax=ax)
        ax.axvline(0, ls="--", color="#777777", lw=.8)
        ax.set(xlabel="Cliff's delta", ylabel="", title="Pairwise CT effect sizes")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        save(fig, out, "figure_ct_effect_sizes")

    wsi_within = {}
    for state in STATE_ORDER:
        indices = [i for i, case_id in enumerate(core_order) if state_by_case[case_id] == state]
        sub = wsi_similarity[np.ix_(indices, indices)]
        wsi_within[state] = float(sub[~np.eye(len(sub), dtype=bool)].mean()) if len(sub) > 1 else np.nan
    wsi_block = pd.DataFrame([wsi_within], index=["WSI | within-state affinity"])

    cc = read_csv(MAP / "clearcode34_state_by_label.csv").set_index("state_id")
    if not cc.empty:
        heatmap(cc, out, "figure_clearcode34_mapping", "ClearCode34 composition by macro-state", cmap="YlGnBu", center=None, vmin=0, fmt=".0f")

    # Figure 3: clinical and modality dependency.
    clinical = {}
    for case_id, patient_state in states.items():
        inventory = dict(patient_state.get("inventory", {}) or {})
        record = dict(inventory.get("Clinical", {}) or {})
        diagnoses = [dict(row) for row in record.get("diagnoses", [])]
        primary = [row for row in diagnoses if str(row.get("diagnosis_is_primary_disease", "")).lower() == "true"]
        diagnosis = (primary or diagnoses or [{}])[0]
        demographic = dict(record.get("demographic", {}) or {})
        days_to_death = pd.to_numeric(demographic.get("days_to_death"), errors="coerce")
        followup = pd.to_numeric(diagnosis.get("days_to_last_follow_up"), errors="coerce")
        if pd.isna(followup):
            followup_days = [pd.to_numeric(row.get("days_to_follow_up"), errors="coerce") for row in record.get("follow_ups", [])]
            followup_days = [value for value in followup_days if pd.notna(value)]
            followup = max(followup_days) if followup_days else np.nan
        vital_status = str(demographic.get("vital_status", "") or "")
        event = int(vital_status.strip().lower() in {"dead", "deceased", "1", "true", "yes"})
        os_time = days_to_death if event else followup
        stage_text = str(diagnosis.get("ajcc_pathologic_stage", "") or "")
        stage_group = next((stage for stage in ("IV", "III", "II", "I") if f"Stage {stage}" in stage_text or stage_text == stage), stage_text)
        clinical[case_id] = {
            "os_time": os_time,
            "os_event": event if pd.notna(os_time) else np.nan,
            "age": pd.to_numeric(demographic.get("age_at_index"), errors="coerce"),
            "stage_group": stage_group,
            "m_stage": str(diagnosis.get("ajcc_pathologic_m", "") or ""),
        }
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    try:
        from lifelines import KaplanMeierFitter
        for state, ids in membership.groupby("state_id").case_id:
            rows = [(clinical[i].get("os_time"), clinical[i].get("os_event")) for i in ids if clinical[i].get("os_time") is not None]
            if rows:
                km = KaplanMeierFitter().fit([x[0] for x in rows], [x[1] for x in rows], label=f"{state} (n={len(rows)})")
                km.plot_survival_function(ax=ax, color=COLORS[state])
        ax.set(xlabel="Days", ylabel="Overall survival", title="Overall survival by macro-state")
        ax.legend()
        save(fig, out, "figure_overall_survival")
    except ImportError:
        plt.close(fig)

    cox = read_csv(CHAR / "stage_binary_adjusted_survival.csv")
    cox = cox[(cox.model == "state_plus_binary_stage") & (cox.status == "estimable")].copy()
    if not cox.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        cox["label"] = cox.covariate.map(lambda x: "Advanced stage (III-IV)" if x == "advanced_stage" else x.replace("state_STATE_", "STATE "))
        y = np.arange(len(cox))
        ax.errorbar(cox.hazard_ratio, y, xerr=[cox.hazard_ratio - cox.ci_low, cox.ci_high - cox.hazard_ratio], fmt="o", color="#333333")
        ax.axvline(1, ls="--", color="#888888")
        ax.set(yticks=y, yticklabels=cox.label, xscale="log", xlabel="Hazard ratio (log scale)", title="Stage-adjusted exploratory Cox model")
        save(fig, out, "figure_stage_adjusted_cox")

    plot_radio_pathological_exemplars(ct_rows, wsi_rows, out)
    if COMPLETION.exists():
        plot_completion_heatmap(COMPLETION / "wsi_state_phenotype_means.csv", out,
                                "figure_wsi_state_phenotype_heatmap",
                                "WSI tumor-patch phenotype composition by macro-state", "WSI")
        plot_noncore_stability(out)

    # Tables: compact artifacts for manuscript assembly.
    table1 = membership.groupby("state_id").size().rename("state_n").reset_index()
    table1["micro_core_n"] = membership.groupby("state_id").core_id.nunique().values
    for state in table1.state_id:
        ids = membership.loc[membership.state_id == state, "case_id"]
        records = [clinical[x] for x in ids if x in clinical]
        table1.loc[table1.state_id == state, "age_median"] = np.nanmedian([r.get("age", np.nan) for r in records])
        table1.loc[table1.state_id == state, "os_event_n"] = np.nansum([r.get("os_event", 0) for r in records])
        table1.loc[table1.state_id == state, "stage_III_IV_n"] = sum(r.get("stage_group") in {"III", "IV"} for r in records)
        table1.loc[table1.state_id == state, "m1_n"] = sum(r.get("m_stage") == "M1" for r in records)
    table1.to_csv(out / "table_state_composition.csv", index=False)
    for source, target in [(CHAR / "rna_hallmark_omnibus.csv", "table_rna_characterization.csv"),
                           (CHAR / "wxs_mutation_omnibus.csv", "table_wxs_characterization.csv"),
                           (CHAR / "ct_radiomics_omnibus.csv", "table_ct_characterization.csv"),
                           (CHAR / "affinity_permanova_permdisp.csv", "table_affinity_diagnostics.csv"),
                           (MAP / "mapping_coverage.csv", "table_known_subtype_mapping.csv")]:
        frame = read_csv(source)
        if not frame.empty: frame.to_csv(out / target, index=False)
    k.to_csv(out / "table_macro_k_stability.csv", index=False)
    read_csv(CHAR / "survival_global.csv").to_csv(out / "table_survival_global.csv", index=False)

    manifest = {"experiment": "four_view_state_visualization", "input_root": str(INPUT), "state_membership": str(STATE), "output_root": str(out),
                "state_patient_count": int(len(membership)), "state_sizes": membership.groupby("state_id").size().to_dict(),
                "active_modalities": ["ct", "wsi", "rna", "wxs"],
                "rna_gene_level_test": "Kruskal-Wallis across four states with BH-FDR over all genes",
                "analysis_note": "All visualizations use the 86 patients assigned to one of the four macro-states. State characterization plots are in-sample discovery-space diagnostics; WSI is shown as affinity/QC structure because no interpretable pathomics table is available."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("vis/figs"))
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
