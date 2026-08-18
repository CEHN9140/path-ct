from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.wxs import wxs_distance_affinity


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_existing_final_snf(
    output_root: Path,
    snf_config: dict[str, Any],
) -> list[dict[str, Any]]:
    import snf
    from tools.wxs import combine_genomic_affinities

    wxs_root = output_root / "wxs"
    subtype_root = output_root / "candidate_subtype"
    feature_path = wxs_root / "wxs_discovery_features.csv"
    order_path = wxs_root / "wxs_discovery_patient_order.json"
    cnv_path = wxs_root / "cnv_affinity.npy"
    order_artifact = subtype_root / "affinity_patient_order.json"
    modality_paths = [subtype_root / f"{name}_affinity.npy" for name in ("ct", "wsi", "rna")]
    required = [feature_path, order_path, cnv_path, order_artifact, *modality_paths]
    if not all(path.is_file() for path in required):
        return []

    patients = json.loads(order_path.read_text(encoding="utf-8"))
    if patients != json.loads(order_artifact.read_text(encoding="utf-8")):
        return []
    table = pd.read_csv(feature_path).set_index("case_id").reindex(patients)
    binary = table.fillna(0.0).to_numpy(bool)
    inter = binary.astype(int) @ binary.astype(int).T
    union = binary.sum(1)[:, None] + binary.sum(1)[None, :] - inter
    cnv_affinity = np.asarray(np.load(cnv_path), dtype=float)
    modalities = [np.asarray(np.load(path), dtype=float) for path in modality_paths]
    baseline_fused = None
    rows = []
    for empty_distance in (0.5, 1.0):
        distance = np.divide(
            union - inter,
            union,
            out=np.full(union.shape, empty_distance, dtype=float),
            where=union > 0,
        )
        np.fill_diagonal(distance, 0.0)
        wxs_affinity = wxs_distance_affinity(distance, snf_config)
        genomic_affinity = combine_genomic_affinities(wxs_affinity, cnv_affinity, snf_config)
        networks = [*modalities, genomic_affinity]
        fused = snf.snf(
            *networks,
            K=min(max(int(snf_config["neighbor_count"]), 1), len(patients) - 1),
            t=int(snf_config["iterations"]),
            alpha=float(snf_config["alpha"]),
        )
        fused = np.maximum((np.asarray(fused) + np.asarray(fused).T) / 2.0, 0.0)
        np.fill_diagonal(fused, 1.0)
        if baseline_fused is None:
            baseline_fused = fused
            mean_delta, max_delta, correlation = 0.0, 0.0, 1.0
        else:
            upper = np.triu_indices(len(patients), 1)
            left, right = baseline_fused[upper], fused[upper]
            mean_delta = float(np.mean(np.abs(left - right)))
            max_delta = float(np.max(np.abs(left - right)))
            correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else None
        rows.append(
            {
                "empty_mutation_distance": empty_distance,
                "patient_count": len(patients),
                "full_snf_affinity_correlation_vs_0_5": correlation,
                "full_snf_mean_absolute_delta_vs_0_5": mean_delta,
                "full_snf_max_absolute_delta_vs_0_5": max_delta,
            }
        )
    return rows


def load_cases(inventory_path: Path) -> list[tuple[str, str]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = []
    for case in inventory:
        case_id = str(case.get("Case_ID", "") or "")
        records = list(case.get("WXS", []) or [])
        if case_id and records:
            rows.append((case_id, str(records[0].get("File Path", "") or "")))
    return sorted(rows)


def read_mutations(file_path: str, nonsynonymous: set[str]) -> pd.DataFrame:
    with gzip.open(file_path, "rb") as handle:
        table = pd.read_csv(handle, sep="\t", comment="#", low_memory=False)
    required = {"Hugo_Symbol", "Variant_Classification", "Chromosome", "Start_Position"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{file_path} lacks WXS columns: {sorted(missing)}")
    table = table.loc[table["Variant_Classification"].isin(nonsynonymous)].copy()
    table["Hugo_Symbol"] = table["Hugo_Symbol"].astype(str)
    return table.drop_duplicates(
        ["Hugo_Symbol", "Chromosome", "Start_Position"]
    )


def run(
    inventory_path: Path,
    config_path: Path,
    snf_config_path: Path,
    experiment_root: Path,
    existing_output_root: Path,
) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    snf_config = yaml.safe_load(snf_config_path.read_text(encoding="utf-8")) or {}
    cases = load_cases(inventory_path)
    nonsynonymous = set(config["nonsynonymous_classes"])
    mutations = {
        case_id: read_mutations(file_path, nonsynonymous)
        for case_id, file_path in cases
        if Path(file_path).is_file()
    }
    patients = sorted(mutations)
    all_genes = sorted(
        set(config.get("driver_genes", []))
        | {
            str(gene)
            for table in mutations.values()
            for gene, count in table["Hugo_Symbol"].value_counts().items()
            if count / max(len(patients), 1) >= float(config["min_gene_prevalence"])
        }
    )
    binary = np.zeros((len(patients), len(all_genes)), dtype=bool)
    for index, case_id in enumerate(patients):
        genes = set(mutations[case_id]["Hugo_Symbol"])
        binary[index] = [gene in genes for gene in all_genes]

    empty_mask = ~binary.any(axis=1)
    distance_rows = []
    affinities = {}
    affinity_config = {
        "neighbor_count": int(snf_config["neighbor_count"]),
        "mu": float(snf_config["mu"]),
        "iterations": int(snf_config["iterations"]),
        "alpha": float(snf_config["alpha"]),
    }
    for empty_distance in (0.0, 0.5, 1.0):
        inter = binary.astype(int) @ binary.astype(int).T
        union = binary.sum(1)[:, None] + binary.sum(1)[None, :] - inter
        distance = np.divide(
            union - inter,
            union,
            out=np.full(union.shape, empty_distance, dtype=float),
            where=union > 0,
        )
        np.fill_diagonal(distance, 0.0)
        affinities[str(empty_distance)] = wxs_distance_affinity(distance, affinity_config)
        distance_rows.append(
            {
                "empty_mutation_distance": empty_distance,
                "empty_profile_count": int(empty_mask.sum()),
                "empty_profile_fraction": float(empty_mask.mean()) if len(empty_mask) else 0.0,
                "mean_pairwise_distance": float(distance[np.triu_indices(len(patients), 1)].mean())
                if len(patients) > 1
                else 0.0,
            }
        )

    baseline = affinities["0.5"]
    sensitivity = []
    for value, matrix in affinities.items():
        off_diagonal = ~np.eye(len(patients), dtype=bool)
        sensitivity.append(
            {
                "empty_mutation_distance": float(value),
                "mean_absolute_affinity_delta_vs_0_5": float(
                    np.mean(np.abs(matrix[off_diagonal] - baseline[off_diagonal]))
                )
                if off_diagonal.any()
                else 0.0,
                "max_absolute_affinity_delta_vs_0_5": float(
                    np.max(np.abs(matrix[off_diagonal] - baseline[off_diagonal]))
                )
                if off_diagonal.any()
                else 0.0,
            }
        )

    profile_rows = [
        {
            "case_id": case_id,
            "n_selected_genes": int(binary[index].sum()),
            "empty_profile": bool(empty_mask[index]),
        }
        for index, case_id in enumerate(patients)
    ]
    summary = {
        "experiment": "wxs_empty_profile_audit_v2",
        "inputs": {
            "inventory": str(inventory_path),
            "config": str(config_path),
            "snf_config": str(snf_config_path),
            "existing_output_root": str(existing_output_root),
        },
        "patient_count": len(patients),
        "selected_gene_count": len(all_genes),
        "missing_input_file_count": len(cases) - len(mutations),
        "empty_profile_count": int(empty_mask.sum()),
        "empty_profile_fraction": float(empty_mask.mean()) if len(empty_mask) else 0.0,
        "distance_sensitivity": distance_rows,
        "affinity_sensitivity_vs_0_5": sensitivity,
        "final_snf_sensitivity": audit_existing_final_snf(existing_output_root, affinity_config),
        "interpretation_policy": {
            "primary_value": float(config["empty_mutation_distance"]),
            "no_main_pipeline_change": True,
            "empty_profile_is_reported_not_excluded": True,
            "final_snf_uses_existing_main_affinities": True,
        },
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_csv(experiment_root / "wxs_empty_profile_cases.csv", profile_rows)
    write_csv(experiment_root / "wxs_distance_sensitivity.csv", distance_rows)
    write_csv(experiment_root / "wxs_affinity_sensitivity.csv", sensitivity)
    write_csv(experiment_root / "wxs_final_snf_sensitivity.csv", summary["final_snf_sensitivity"])
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit empty WXS mutation profiles.")
    parser.add_argument("--inventory", type=Path, default=Path("data/tcga_kirc_data.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/wxs.yaml"))
    parser.add_argument("--snf-config", type=Path, default=Path("configs/snf.yaml"))
    parser.add_argument("--existing-output-root", type=Path, default=Path("output_kirc"))
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_wxs_empty_profile_audit"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.inventory, args.config, args.snf_config, args.experiment_root, args.existing_output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
