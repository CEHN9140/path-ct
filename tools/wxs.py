from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.cache_utils import file_identity, hash_payload, semantic_config
from utils.io import ensure_dir, write_json
from utils.omics_utils import (
    build_cohort_signature,
    collect_case_file_paths,
    load_manifest_if_valid,
)


def load_wxs_config(config_dir: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((Path(config_dir) / "wxs.yaml").read_text(encoding="utf-8")) or {}


def wxs_distance_affinity(distance: np.ndarray, snf_config: Mapping[str, Any]) -> np.ndarray:
    from tools.evidence_features import distance_to_affinity

    return distance_to_affinity(distance, snf_config)


def binary_mutation_distance(binary: np.ndarray, empty_mutation_distance: float) -> np.ndarray:
    values = np.asarray(binary, dtype=bool)
    intersection = values.astype(int) @ values.astype(int).T
    union = values.sum(axis=1)[:, None] + values.sum(axis=1)[None, :] - intersection
    distance = np.divide(
        union - intersection,
        union,
        out=np.full(union.shape, float(empty_mutation_distance), dtype=float),
        where=union > 0,
    )
    np.fill_diagonal(distance, 0.0)
    return distance


def collect_wxs_file_paths(cases: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    rows = []
    for case in cases:
        case_id = str(case.get("Case_ID", "") or "").strip()
        paths = [
            Path(str(record.get("File Path", "") or ""))
            for record in list(case.get("WXS") or [])
            if Path(str(record.get("File Path", "") or "")).is_file()
        ]
        if not case_id or not paths:
            continue
        if len(paths) > 1:
            quality = {}
            for path in paths:
                with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                    reader = csv.DictReader(
                        (line for line in handle if not line.startswith("#")), delimiter="\t"
                    )
                    depths = [
                        float(row["t_depth"])
                        for row in reader
                        if str(row.get("t_depth", "")).strip()
                    ]
                quality[path] = float(np.median(depths)) if depths else -1.0
            paths = [max(paths, key=lambda path: (quality[path], str(path)))]
        rows.append((case_id, str(paths[0])))
    return sorted(rows)


def read_wxs_mutations(manifest_path: Path, patient_ids: list[str]) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {str(row["case_id"]): Path(row["file_path"]) for row in manifest["input_cases"]}
    missing = [patient_id for patient_id in patient_ids if not paths.get(patient_id, Path()).is_file()]
    if missing:
        raise ValueError(f"WXS manifest is missing target patients: {', '.join(missing)}")
    rows = []
    for patient_id in patient_ids:
        with gzip.open(paths[patient_id], "rt", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
            rows.extend({
                "patient_id": patient_id,
                "gene": str(row.get("Hugo_Symbol", "")).strip(),
                "classification": str(row.get("Variant_Classification", "")).strip(),
                "variant_type": str(row.get("Variant_Type", "")).strip(),
                "chromosome": str(row.get("Chromosome", "")).strip(),
                "start": str(row.get("Start_Position", "")).strip(),
            } for row in reader)
    return pd.DataFrame(rows, columns=[
        "patient_id", "gene", "classification", "variant_type", "chromosome", "start"
    ])


def build_wxs_artifacts(
    wxs_cache: Mapping[str, Any],
    cohort_cases: list[dict[str, Any]],
    *,
    output_root: str,
    config_dir: str,
) -> dict[str, str]:
    from utils.llm_utils import load_candidate_proposer_config

    config = load_wxs_config(config_dir)
    snf_config = load_candidate_proposer_config(config_dir)["snf"]
    patient_ids = [str(case["Case_ID"]) for case in cohort_cases]
    output_dir = ensure_dir(Path(output_root) / "wxs")
    discovery_path = output_dir / "wxs_discovery_features.csv"
    interpretation_path = output_dir / "wxs_interpretation_features.csv"
    validation_path = output_dir / "wxs_validation_features.csv"
    distance_path = output_dir / "wxs_distance.npy"
    affinity_path = output_dir / "wxs_affinity.npy"
    order_path = output_dir / "wxs_discovery_patient_order.json"
    audit_path = output_dir / "wxs_discovery_audit.json"
    signature = hash_payload({
        "cache_version": 4,
        "wxs_input_signature": wxs_cache["signature"],
        "patient_ids": patient_ids,
        "config": semantic_config({
            **config,
            "biological_support": {
                key: value
                for key, value in config["biological_support"].items()
                if key != "exploratory_report_top_n"
            },
        }),
        "snf": semantic_config(snf_config),
        "code": file_identity(str(Path(__file__).resolve())),
    })
    if all(path.is_file() for path in (
        discovery_path, interpretation_path, validation_path, distance_path,
        affinity_path, order_path, audit_path,
    )):
        if json.loads(audit_path.read_text(encoding="utf-8")).get("artifact_signature") == signature:
            return {
                "wxs_discovery_feature_path": str(discovery_path),
                "wxs_interpretation_feature_path": str(interpretation_path),
                "wxs_validation_feature_path": str(validation_path),
                "wxs_distance_path": str(distance_path),
                "wxs_affinity_path": str(affinity_path),
                "wxs_patient_order_path": str(order_path),
                "wxs_discovery_audit_path": str(audit_path),
            }

    mutations = read_wxs_mutations(Path(wxs_cache["manifest_path"]), patient_ids)
    nonsynonymous = set(config["nonsynonymous_classes"])
    rows = mutations.drop_duplicates(["patient_id", "gene", "chromosome", "start"])
    rows = rows[rows["gene"].ne("")]
    altered = rows[rows["classification"].isin(nonsynonymous)]
    prevalence = altered.groupby("gene")["patient_id"].nunique() / len(patient_ids)
    genes = sorted(prevalence[prevalence >= float(config["min_gene_prevalence"])].index)
    driver_genes = sorted({str(gene).upper() for gene in config["biological_support"]["driver_genes"]})
    interpretation_genes = sorted(set(altered["gene"]) | set(driver_genes))
    discovery = pd.DataFrame(False, index=patient_ids, columns=genes)
    interpretation = pd.DataFrame(False, index=patient_ids, columns=interpretation_genes)
    for gene, gene_rows in altered[altered["gene"].isin(genes)].groupby("gene"):
        carriers = set(gene_rows["patient_id"])
        discovery[gene] = [patient_id in carriers for patient_id in patient_ids]
    for gene, gene_rows in altered[altered["gene"].isin(interpretation_genes)].groupby("gene"):
        carriers = set(gene_rows["patient_id"])
        interpretation[gene] = [patient_id in carriers for patient_id in patient_ids]

    functional = pd.DataFrame(index=patient_ids)
    for name, classes in config["functional_classes"].items():
        functional[name] = rows[rows["classification"].isin(classes)].groupby("patient_id").size().reindex(patient_ids, fill_value=0)
    discovery.index.name = interpretation.index.name = functional.index.name = "case_id"
    discovery.add_prefix("mutation::").reset_index().to_csv(discovery_path, index=False)
    interpretation.reset_index().to_csv(interpretation_path, index=False)
    functional.add_prefix("validation::").reset_index().to_csv(validation_path, index=False)

    binary = discovery.to_numpy(dtype=bool)
    distance = binary_mutation_distance(binary, float(config["empty_mutation_distance"]))
    np.save(distance_path, distance)
    np.save(affinity_path, wxs_distance_affinity(distance, snf_config))
    write_json(order_path, patient_ids)
    write_json(audit_path, {
        "artifact_signature": signature,
        "patient_count": len(patient_ids),
        "discovery_gene_count": len(genes),
        "discovery_genes": genes,
        "minimum_gene_prevalence": float(config["min_gene_prevalence"]),
        "discovery_driver_forced_inclusion": False,
        "interpretation_includes_configured_drivers": True,
        "interpretation_gene_count": len(interpretation_genes),
        "driver_genes": driver_genes,
        "validation_features": [f"validation::{name}" for name in functional.columns],
        "distance_metric": "jaccard_binary",
        "empty_mutation_distance": float(config["empty_mutation_distance"]),
        "zero_vector_patient_count": int(np.sum(binary.sum(axis=1) == 0)),
    })
    return {
        "wxs_discovery_feature_path": str(discovery_path),
        "wxs_interpretation_feature_path": str(interpretation_path),
        "wxs_validation_feature_path": str(validation_path),
        "wxs_distance_path": str(distance_path),
        "wxs_affinity_path": str(affinity_path),
        "wxs_patient_order_path": str(order_path),
        "wxs_discovery_audit_path": str(audit_path),
    }


def build_wxs_cohort_cache(
    cohort_cases: Sequence[Mapping[str, Any]], *, output_root: str
) -> dict[str, Any]:
    case_file_rows = collect_wxs_file_paths(cohort_cases)
    output_dir = ensure_dir(Path(output_root) / "wxs")
    manifest_path = output_dir / "manifest.json"
    signature = build_cohort_signature(
        case_file_rows,
        extra={
            "modality": "WXS",
            "feature_mode": "prevalence_filtered_nonsynonymous_binary",
            "file_selection": "single_file_or_highest_median_tumor_depth",
        },
    )
    manifest = load_manifest_if_valid(manifest_path, signature=signature, required_paths=[])
    if manifest is None:
        manifest = {
            "signature": signature,
            "modality": "WXS",
            "cohort_case_count": len(case_file_rows),
            "feature_mode": "prevalence_filtered_nonsynonymous_binary",
            "file_selection": "single_file_or_highest_median_tumor_depth",
            "input_cases": [
                {"case_id": case_id, "file_path": file_path}
                for case_id, file_path in case_file_rows
            ],
        }
        write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "signature": signature,
    }
