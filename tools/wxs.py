from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from snf.compute import affinity_matrix

from utils.io import ensure_dir, write_json
from utils.cache_utils import hash_payload, semantic_config
from utils.omics_utils import (
    build_cohort_signature,
    collect_case_file_paths,
    load_manifest_if_valid,
)
from utils.tool_utils import make_tool_result


def fuse_genomic_affinities(
    networks: list[np.ndarray], config: Mapping[str, Any]
) -> np.ndarray:
    import snf

    if not networks:
        raise ValueError("At least one genomic affinity network is required")
    matrices = [np.asarray(network, dtype=float) for network in networks]
    n_cases = int(matrices[0].shape[0])
    if any(matrix.shape != (n_cases, n_cases) for matrix in matrices):
        raise ValueError("Genomic affinity networks must have identical square shapes")
    if len(matrices) == 1 or n_cases <= 1:
        return matrices[0].copy()
    k = min(max(int(config.get("neighbor_count", 20)), 1), n_cases - 1)
    fused = snf.snf(
        *matrices,
        K=k,
        t=int(config.get("iterations", 20)),
        alpha=float(config.get("alpha", 1.0)),
    )
    fused = np.maximum((np.asarray(fused) + np.asarray(fused).T) / 2.0, 0.0)
    np.fill_diagonal(fused, 1.0)
    return fused


def combine_genomic_affinities(
    wxs_affinity: np.ndarray,
    cnv_affinity: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    return fuse_genomic_affinities([wxs_affinity, cnv_affinity], config)


CNV_CHROMOSOME_LENGTHS = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555,
    5: 181538259, 6: 170805979, 7: 159345973, 8: 145138636,
    9: 138394717, 10: 133797422, 11: 135086622, 12: 133275309,
    13: 114364328, 14: 107043718, 15: 101991189, 16: 90338345,
    17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468,
}
CNV_CENTROMERES_GRCH38 = {
    1: 124250000, 2: 94950000, 3: 92450000, 4: 50900000, 5: 50100000,
    6: 60550000, 7: 60100000, 8: 45200000, 9: 43850000, 10: 39800000,
    11: 53400000, 12: 35500000, 13: 17700000, 14: 17150000,
    15: 19000000, 16: 36850000, 17: 24000000, 18: 18450000,
    19: 26150000, 20: 28050000, 21: 11950000, 22: 15550000,
}
CNV_DRIVER_LOCI_GRCH38 = {
    "VHL": (3, 10141783, 10153667),
    "SETD2": (3, 47057897, 47205467),
    "PBRM1": (3, 52579368, 52708481),
    "BAP1": (3, 52401007, 52410134),
    "JAK2": (9, 4985245, 5129948),
    "CDKN2A": (9, 21967751, 21995321),
    "PTEN": (10, 87863113, 87971930),
}


def load_wxs_config(config_dir: str) -> dict[str, Any]:
    import yaml
    return yaml.safe_load((Path(config_dir) / "wxs.yaml").read_text(encoding="utf-8")) or {}


def wxs_distance_affinity(distance: np.ndarray, snf_config: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(distance, dtype=float)
    if len(values) < 2:
        return np.ones(values.shape)
    values = np.nan_to_num((values + values.T) / 2.0, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(values, 0.0)
    scale = float(values.max())
    if scale > 0:
        values /= scale
    affinity = np.asarray(
        affinity_matrix(
            values,
            K=min(max(int(snf_config.get("neighbor_count", 20)), 1), len(values) - 1),
            mu=float(snf_config.get("mu", 0.5)),
        ),
        dtype=float,
    )
    diagonal = np.sqrt(np.maximum(np.diag(affinity)[:, None] * np.diag(affinity)[None, :], 1e-12))
    affinity = np.clip((affinity / diagonal + (affinity / diagonal).T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def binary_mutation_distance(binary: np.ndarray, empty_mutation_distance: float) -> np.ndarray:
    values = np.asarray(binary, dtype=bool)
    inter = values.astype(int) @ values.astype(int).T
    union = values.sum(1)[:, None] + values.sum(1)[None, :] - inter
    distance = np.divide(
        union - inter,
        union,
        out=np.full_like(union, float(empty_mutation_distance), dtype=float),
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
    missing = [
        patient_id
        for patient_id in patient_ids
        if paths.get(patient_id) is None or not paths[patient_id].is_file()
    ]
    if missing:
        raise ValueError(f"WXS manifest is missing target patients: {', '.join(missing)}")
    rows = []
    for patient_id in patient_ids:
        path = paths[patient_id]
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
            for row in reader:
                rows.append({
                    "patient_id": str(patient_id), "gene": str(row.get("Hugo_Symbol", "")).strip(),
                    "classification": str(row.get("Variant_Classification", "")).strip(),
                    "variant_type": str(row.get("Variant_Type", "")).strip(),
                    "chromosome": str(row.get("Chromosome", "")).strip(),
                    "start": str(row.get("Start_Position", "")).strip(),
                })
    return pd.DataFrame(rows, columns=["patient_id", "gene", "classification", "variant_type", "chromosome", "start"])


def load_complete_cnv_matrix(path: Path, patient_ids: list[str]) -> np.ndarray:
    table = pd.read_csv(path)
    case_ids = table["case_id"].astype(str)
    available = set(case_ids)
    missing = [patient_id for patient_id in patient_ids if patient_id not in available]
    if missing:
        raise ValueError(f"CNV features are missing target patients: {', '.join(missing)}")
    matrix = table.assign(case_id=case_ids).set_index("case_id").loc[patient_ids].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("CNV features contain non-finite values")
    return matrix


def build_wxs_cnv_artifacts(
    wxs_cache: Mapping[str, Any],
    cnv_cache: Mapping[str, Any],
    cohort_cases: list[dict[str, Any]],
    *,
    output_root: str,
    config_dir: str,
) -> dict[str, str]:
    config = load_wxs_config(config_dir)
    from utils.llm_utils import load_candidate_proposer_config
    snf = load_candidate_proposer_config(config_dir).get("snf", {})
    patients = [str(case["Case_ID"]) for case in cohort_cases]
    out = ensure_dir(Path(output_root) / "wxs")
    discovery = out / "wxs_discovery_features.csv"
    validation = out / "wxs_validation_features.csv"
    wxs_path = out / "wxs_affinity.npy"
    cnv_path = out / "cnv_affinity.npy"
    (out / "genomic_affinity.npy").unlink(missing_ok=True)
    order_path, audit_path = out / "wxs_discovery_patient_order.json", out / "wxs_discovery_audit.json"
    signature = hash_payload(
        {
            "cache_version": 1,
            "upstream": {
                "wxs": wxs_cache["signature"],
                "cnv": cnv_cache["signature"],
            },
            "semantic_config": {
                "wxs": semantic_config(config),
                "snf": semantic_config(snf),
            },
            "patient_ids": patients,
        }
    )
    if audit_path.is_file() and wxs_path.is_file() and cnv_path.is_file() and order_path.is_file() and discovery.is_file() and validation.is_file():
        if json.loads(audit_path.read_text()).get("artifact_signature") == signature:
            return {"wxs_discovery_feature_path": str(discovery), "wxs_validation_feature_path": str(validation), "wxs_affinity_path": str(wxs_path), "cnv_affinity_path": str(cnv_path), "wxs_patient_order_path": str(order_path), "wxs_discovery_audit_path": str(audit_path)}
    mutations = read_wxs_mutations(Path(wxs_cache["manifest_path"]), patients)
    if mutations.empty:
        mutations = pd.DataFrame(columns=["patient_id", "gene", "classification", "variant_type", "chromosome", "start"])
    nonsyn = set(config["nonsynonymous_classes"])
    rows = mutations.drop_duplicates(["patient_id", "gene", "chromosome", "start"])
    nonsynonymous = rows[rows.classification.isin(nonsyn)]
    prevalence = nonsynonymous.groupby("gene")["patient_id"].nunique() / max(len(patients), 1)
    genes = sorted(set(prevalence[prevalence >= float(config["min_gene_prevalence"])].index) | set(config["driver_genes"]))
    discovery_features = pd.DataFrame(0.0, index=patients, columns=genes)
    for gene in genes:
        altered = set(rows.loc[(rows.gene == gene) & rows.classification.isin(nonsyn), "patient_id"])
        discovery_features[gene] = [float(p in altered) for p in patients]
    function = pd.DataFrame(index=patients)
    for name, classes in config["functional_classes"].items():
        function[name] = rows.loc[rows.classification.isin(classes)].groupby("patient_id").size().reindex(patients, fill_value=0)
    global_features = pd.DataFrame(index=patients)
    global_features["estimated_TMB"] = rows.loc[rows.classification.isin(nonsyn)].groupby("patient_id").size().reindex(patients, fill_value=0) / float(config["capture_size_mb"])
    discovery_features = discovery_features.add_prefix("mutation::")
    discovery_features.index.name = "case_id"
    discovery_features.reset_index().to_csv(discovery, index=False)
    validation_features = pd.concat([function.add_prefix("validation::"), global_features.rename(columns={"estimated_TMB": "validation::estimated_TMB"})], axis=1)
    validation_features.index.name = "case_id"
    validation_features.reset_index().to_csv(validation, index=False)
    binary = discovery_features.to_numpy(bool)
    distance = binary_mutation_distance(binary, config["empty_mutation_distance"])
    wxs_affinity = wxs_distance_affinity(distance, snf)
    cnv = load_complete_cnv_matrix(Path(cnv_cache["case_features_path"]), patients)
    median = np.median(cnv, axis=0, keepdims=True)
    iqr = np.quantile(cnv, 0.75, axis=0, keepdims=True) - np.quantile(cnv, 0.25, axis=0, keepdims=True)
    cnv = np.divide(cnv - median, iqr, out=np.zeros_like(cnv), where=iqr > 0)
    cnv_affinity = wxs_distance_affinity(cdist(cnv, cnv), snf)
    np.save(wxs_path, wxs_affinity); np.save(cnv_path, cnv_affinity)
    order_path.write_text(json.dumps(patients, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps({
        "artifact_signature": signature,
        "patient_count": len(patients),
        "discovery_gene_count": len(genes),
        "discovery_genes": genes,
        "min_gene_prevalence": config["min_gene_prevalence"],
        "validation_features": list(validation_features.columns),
        "independent_affinities": ["wxs", "cnv"],
        "wxs_feature_count": int(discovery_features.shape[1]),
        "cnv_feature_count": int(cnv.shape[1]),
        "cnv_zero_variance_count": int(np.sum(iqr.reshape(-1) == 0)),
        "distance_metric": "jaccard_binary",
        "empty_mutation_distance": float(config["empty_mutation_distance"]),
        "zero_vector_patient_count": int(np.sum(binary.sum(axis=1) == 0)),
        "zero_zero_pair_count": int(np.triu(np.outer(binary.sum(axis=1) == 0, binary.sum(axis=1) == 0), 1).sum()),
    }, indent=2), encoding="utf-8")
    return {"wxs_discovery_feature_path": str(discovery), "wxs_validation_feature_path": str(validation), "wxs_affinity_path": str(wxs_path), "cnv_affinity_path": str(cnv_path), "wxs_patient_order_path": str(order_path), "wxs_discovery_audit_path": str(audit_path)}


def read_cnv_case_features(file_path: str, threshold: float = 0.2) -> dict[str, float]:
    table = pd.read_csv(file_path, sep="\t")
    required = {"Chromosome", "Start", "End", "Segment_Mean"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{file_path} lacks CNV columns: {sorted(missing)}")
    weighted = {
        f"chr{chromosome}{arm}": [0.0, 0]
        for chromosome in CNV_CHROMOSOME_LENGTHS
        for arm in ("p", "q")
    }
    loci = {name: [0.0, 0] for name in CNV_DRIVER_LOCI_GRCH38}
    total = altered = gain = loss = 0
    for row in table.itertuples(index=False):
        try:
            chromosome = int(row.Chromosome)
            start = max(int(row.Start) - 1, 0)
            end = min(int(row.End), CNV_CHROMOSOME_LENGTHS.get(chromosome, 0))
            segment_mean = float(row.Segment_Mean)
        except (TypeError, ValueError):
            continue
        if chromosome not in CNV_CHROMOSOME_LENGTHS or end <= start:
            continue
        length = end - start
        centromere = CNV_CENTROMERES_GRCH38[chromosome]
        for arm, arm_start, arm_end in (("p", 0, centromere), ("q", centromere, CNV_CHROMOSOME_LENGTHS[chromosome])):
            overlap = max(0, min(end, arm_end) - max(start, arm_start))
            if overlap:
                weighted[f"chr{chromosome}{arm}"][0] += segment_mean * overlap
                weighted[f"chr{chromosome}{arm}"][1] += overlap
        for name, (locus_chromosome, locus_start, locus_end) in CNV_DRIVER_LOCI_GRCH38.items():
            if chromosome != locus_chromosome:
                continue
            overlap = max(0, min(end, locus_end) - max(start, locus_start))
            if overlap:
                loci[name][0] += segment_mean * overlap
                loci[name][1] += overlap
        total += length
        altered += length if abs(segment_mean) >= threshold else 0
        gain += length if segment_mean >= threshold else 0
        loss += length if segment_mean <= -threshold else 0
    values = {
        name: (
            total_value / covered
            if covered
            else 0.0
        )
        for name, (total_value, covered) in weighted.items()
    }
    values.update({
        f"locus::{name}": total_value / covered if covered else 0.0
        for name, (total_value, covered) in loci.items()
    })
    values.update({
        "fga": altered / total if total else 0.0,
        "gain_burden": gain / total if total else 0.0,
        "loss_burden": loss / total if total else 0.0,
    })
    return values


def build_cnv_cohort_cache(
    cohort_cases: Sequence[Mapping[str, Any]], *, output_root: str, config_dir: str = ""
) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(
        (Path(config_dir).expanduser() / "wxs.yaml").read_text(encoding="utf-8")
    )
    threshold = float(config.get("cnv_segment_threshold", 0.2))
    case_file_rows = collect_case_file_paths(cohort_cases, "CNV")
    output_dir = ensure_dir(Path(output_root) / "cnv")
    manifest_path = output_dir / "manifest.json"
    case_features_path = output_dir / "case_features.csv"
    feature_names = [
        f"chr{chromosome}{arm}"
        for chromosome in CNV_CHROMOSOME_LENGTHS
        for arm in ("p", "q")
    ]
    feature_names += [f"locus::{name}" for name in CNV_DRIVER_LOCI_GRCH38]
    feature_names += ["fga", "gain_burden", "loss_burden"]
    signature = build_cohort_signature(
        case_file_rows,
        extra={"modality": "CNV", "threshold": threshold, "feature_names": feature_names},
    )
    manifest = load_manifest_if_valid(
        manifest_path, signature=signature, required_paths=[case_features_path]
    )
    if manifest is None:
        rows = [
            {"case_id": case_id, **read_cnv_case_features(file_path, threshold)}
            for case_id, file_path in case_file_rows
        ]
        pd.DataFrame(rows, columns=["case_id", *feature_names]).to_csv(
            case_features_path, index=False
        )
        manifest = {
            "signature": signature,
            "modality": "CNV",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "segment_threshold": threshold,
            "input_cases": [
                {"case_id": case_id, "file_path": file_path}
                for case_id, file_path in case_file_rows
            ],
            "files": {"case_features": str(case_features_path)},
        }
        write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "case_features_path": str(case_features_path),
        "feature_names": feature_names,
        "signature": signature,
    }


def run_case_cnv_features(
    *, case_id: str, cohort_cases: Sequence[Mapping[str, Any]], output_root: str, config_dir: str = ""
) -> dict[str, Any]:
    cache = build_cnv_cohort_cache(
        cohort_cases, output_root=output_root, config_dir=config_dir
    )
    features = pd.read_csv(cache["case_features_path"])
    row = features.loc[features["case_id"].astype(str) == str(case_id)]
    values = row.iloc[0].drop(labels=["case_id"]).astype(float).tolist() if not row.empty else []
    status = "success" if values else "failure"
    payload = {
        "case_id": str(case_id),
        "source_path": dict(collect_case_file_paths(cohort_cases, "CNV")).get(str(case_id), ""),
        "feature_names": cache["feature_names"],
        "feature_values": values,
    }
    return {
        "tool_result": make_tool_result(
            output_root=output_root,
            tool_name="cnv",
            status=status,
            identifier=str(case_id),
            metrics={"feature_count": len(values)},
            artifacts={
                "case_features_path": cache["case_features_path"],
                "manifest_path": cache["manifest_path"],
            },
            provenance={
                "backend": "gdc_copy_number_segment",
                "signature": cache["signature"],
            },
            errors=[] if status == "success" else ["CNV features are unavailable for this case."],
            payload=payload,
        ),
        "payload": payload,
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
            "feature_mode": "recurrent_nonsynonymous_binary",
            "file_selection": "single_file_or_highest_median_tumor_depth",
        },
    )
    manifest = load_manifest_if_valid(
        manifest_path, signature=signature, required_paths=[]
    )
    if manifest is None:
        manifest = {
            "signature": signature,
            "modality": "WXS",
            "cohort_case_count": len(case_file_rows),
            "feature_mode": "recurrent_nonsynonymous_binary",
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
