#!/usr/bin/env python3
"""Summarize canonical-core dependence on each modality from existing ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.io import write_json
from utils.visualization import configure_matplotlib

CORES = ("CORE01", "CORE02", "CORE03", "CORE04")
MODALITIES = ("ct", "wsi", "rna", "wxs", "cnv")


def build_dependency_table(paths, cores=CORES):
    rows = []
    for modality in MODALITIES:
        path = Path(paths[modality])
        frame = pd.read_csv(path).set_index("canonical_core")
        missing = set(cores) - set(frame.index)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {path}")
        for core in cores:
            row = frame.loc[core]
            rows.append({
                "core": core,
                "modality": modality,
                "matched_retention": float(row["retention"]),
                "any_stable_retention": float(row["any_variant_stable_fraction"]),
                "lost_fraction": float(row["no_variant_stable_fraction"]),
                "jaccard": float(row["jaccard"]),
            })
    return pd.DataFrame(rows)


def default_paths(output_root):
    output_root = Path(output_root)
    return {
        "ct": output_root.parent / "08_modality_ablation" / "canonical_core_leave_ct_out_fragmentation.csv",
        **{
            modality: output_root / f"leave_{modality}_out" / f"canonical_core_leave_{modality}_out_fragmentation.csv"
            for modality in ("wsi", "rna", "wxs", "cnv")
        },
    }


def write_heatmap(table, output_path):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    metrics = [
        ("matched_retention", "Matched retention"),
        ("any_stable_retention", "Any stable-core retention"),
        ("lost_fraction", "Lost from stable cores"),
        ("jaccard", "Matched Jaccard"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        matrix = table.pivot(index="core", columns="modality", values=metric).loc[list(CORES), list(MODALITIES)]
        image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlBu")
        axis.set_title(title)
        axis.set_xticks(range(len(MODALITIES)), [name.upper() for name in MODALITIES])
        axis.set_yticks(range(len(CORES)), CORES)
        for row in range(len(CORES)):
            for column in range(len(MODALITIES)):
                axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Canonical-core modality dependency", fontsize=14)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run(output_root, force=False):
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = default_paths(output_root.parent / "09_full_leave_one_modality_out")
    table = build_dependency_table(paths)
    table.to_csv(output_root / "core_modality_dependency_long.csv", index=False)
    for metric in ("matched_retention", "any_stable_retention", "lost_fraction", "jaccard"):
        table.pivot(index="core", columns="modality", values=metric).loc[list(CORES), list(MODALITIES)].to_csv(
            output_root / f"{metric}_matrix.csv"
        )
    write_heatmap(table, output_root / "core_modality_dependency_heatmap.png")
    write_json(output_root / "manifest.json", {
        "cores": list(CORES),
        "modalities": list(MODALITIES),
        "source_fragmentation": {name: str(path) for name, path in paths.items()},
        "metrics": ["matched_retention", "any_stable_retention", "lost_fraction", "jaccard"],
    })
    return {"output_root": str(output_root), "rows": len(table)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/10_modality_dependency_matrix")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
