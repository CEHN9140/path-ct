from __future__ import annotations

from typing import Any, Mapping, Sequence
from pathlib import Path

import json

import numpy as np


def distance_to_affinity(distance: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    from snf.compute import affinity_matrix

    values = np.asarray(distance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("distance must be a square matrix")
    if not np.isfinite(values).all():
        raise ValueError("distance contains non-finite values")
    values = (values + values.T) / 2.0
    if np.min(values) < 0:
        raise ValueError("distance must be nonnegative")
    np.fill_diagonal(values, 0.0)
    if len(values) < 2:
        return np.ones(values.shape, dtype=float)
    affinity = np.asarray(
        affinity_matrix(
            values,
            K=min(max(int(config["neighbor_count"]), 1), len(values) - 1),
            mu=float(config["mu"]),
        ),
        dtype=float,
    )
    affinity = (affinity + affinity.T) / 2.0
    if not np.isfinite(affinity).all() or np.min(affinity) < 0:
        raise ValueError("affinity_matrix returned invalid values")
    return affinity


def build_modality_affinity_artifacts(
    patient_states: Sequence[Mapping[str, Any]],
    *,
    output_root: str,
    config_dir: str = "",
    discovery_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the five independent candidate-generation affinity networks."""
    from tools.ct_radiomics import build_ct_affinity
    from tools.rna import build_rna_affinity
    from tools.wsi_affinity import build_wsi_affinity

    ct = build_ct_affinity(patient_states, config_dir=config_dir, output_root=output_root)
    wsi = build_wsi_affinity(patient_states, config_dir=config_dir)
    rna = build_rna_affinity(patient_states, config_dir=config_dir)
    artifacts = dict(discovery_artifacts or {})
    order_path = Path(artifacts["wxs_patient_order_path"])
    wxs = {
        "affinity": np.load(artifacts["wxs_affinity_path"]),
        "patient_ids": json.loads(order_path.read_text(encoding="utf-8")),
        "audit": {"source": "wxs_discovery_features", "metric": "jaccard"},
    }
    cnv = {
        "affinity": np.load(artifacts["cnv_affinity_path"]),
        "patient_ids": wxs["patient_ids"],
        "audit": {"source": "cnv_case_features", "metric": "euclidean"},
    }
    patient_ids = list(ct["patient_ids"])
    for name, result in (("wsi", wsi), ("rna", rna), ("wxs", wxs), ("cnv", cnv)):
        if list(result["patient_ids"]) != patient_ids:
            raise ValueError(f"{name} feature patient order does not match CT cohort")
    return {
        "patient_ids": patient_ids,
        "modality_affinities": {
            "ct": ct["affinity"],
            "wsi": wsi["affinity"],
            "rna": rna["affinity"],
            "wxs": wxs["affinity"],
            "cnv": cnv["affinity"],
        },
        "audit": {
            "source": "tool_owned_modality_features",
            "ct": ct["audit"],
            "wsi": wsi["audit"],
            "rna": rna["audit"],
            "wxs": wxs["audit"],
            "cnv": cnv["audit"],
        },
    }


def build_legacy_feature_payload(
    patient_states: Sequence[Mapping[str, Any]], *, config_dir: str = ""
) -> dict[str, Any]:
    """Compatibility path for unit fixtures without evidence-stage artifacts."""
    from scipy.spatial.distance import cdist
    from snf.compute import affinity_matrix
    from utils.llm_utils import load_candidate_proposer_config, load_yaml_file

    config_path = config_dir or "configs"
    snf_config = load_candidate_proposer_config(config_path)["snf"]
    ct_config = load_yaml_file(f"{config_path}/ct_radiomics.yaml")
    states = [dict(state) for state in patient_states if state.get("qc") == "success"]
    patient_ids = [str(state.get("case_id", "")) for state in states]
    ct_values, ct_names, ccc_values = [], [], {str(int(float(width))): [] for width in ct_config.get("ccc_comparison_bin_widths", [])}
    wsi_values, rna_values, wxs_values = [], [], []
    for state in states:
        ct = dict(state.get("ct_evidence", {}) or {})
        if isinstance(ct.get("features"), list) and ct["features"]:
            ct_values.append([float(value) for value in ct["features"]])
            ct_names.append([f"ct_{index:04d}" for index in range(len(ct["features"]))])
        else:
            feature_path = str(ct.get("feature_path", "") or "")
            if not feature_path:
                ct_values.append([])
                ct_names.append([])
            else:
                payload = json.loads(Path(feature_path).read_text(encoding="utf-8"))
                ct_values.append([float(value) for value in payload.values()])
                ct_names.append([str(name) for name in payload])
                for label in ccc_values:
                    comparison = json.loads(Path(str(ct["ccc_feature_paths"][label])).read_text(encoding="utf-8"))
                    ccc_values[label].append([float(comparison[name]) for name in ct_names[-1]])
        wsi = dict(state.get("wsi_evidence", {}) or {})
        if isinstance(wsi.get("features"), list) and wsi["features"]:
            wsi_values.append([float(value) for value in wsi["features"]])
        else:
            wsi_values.append([])
        omics = dict(state.get("omics_evidence", {}) or {})
        rna_values.append([float(value) for value in omics.get("rna_features", [])])
        wxs_values.append([float(value) for value in omics.get("wxs_features", [])])

    ct = np.asarray(ct_values, dtype=float)
    names = next((names for names in ct_names if len(names) == ct.shape[1]), [f"ct_{i:04d}" for i in range(ct.shape[1])])
    keep = np.var(ct, axis=0) > float(snf_config["ct_low_variance_threshold"])
    ct = ct[:, keep]
    names = [name for name, value in zip(names, keep) if value]
    ccc_filter = {"applied": bool(any(ccc_values.values())), "retained_feature_count": len(names), "rejected_features": []}
    if any(ccc_values.values()) and ct.shape[1]:
        ccc_keep = np.ones(ct.shape[1], dtype=bool)
        ccc_matrices = {label: np.asarray(values, dtype=float)[:, keep] for label, values in ccc_values.items()}
        for index in range(ct.shape[1]):
            for values in ccc_matrices.values():
                x, y = ct[:, index], values[:, index]
                denominator = np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2
                ccc_keep[index] &= bool(2 * np.mean((x - x.mean()) * (y - y.mean())) / denominator >= float(ct_config["ccc_threshold"])) if denominator else False
        ccc_filter["rejected_features"] = [name for name, value in zip(names, ccc_keep) if not value]
        ct, names = ct[:, ccc_keep], [name for name, value in zip(names, ccc_keep) if value]
        ccc_filter["retained_feature_count"] = len(names)
    if ct.shape[1] > 1:
        corr = np.nan_to_num(np.corrcoef(ct, rowvar=False), nan=0.0)
        keep = np.ones(ct.shape[1], dtype=bool)
        for index in range(ct.shape[1]):
            keep[index] = not np.any(np.abs(corr[index, :index][keep[:index]]) > float(snf_config["ct_high_correlation_threshold"]))
        ct, names = ct[:, keep], [name for name, value in zip(names, keep) if value]
    def standardize(values):
        values = np.asarray(values, dtype=float)
        mean, std = values.mean(axis=0, keepdims=True), values.std(axis=0, keepdims=True)
        return np.divide(values - mean, std, out=np.zeros_like(values), where=std > 0)
    ct, rna = standardize(ct), standardize(rna_values)
    wsi = np.asarray(wsi_values, dtype=float)
    if wsi.ndim == 2:
        norms = np.linalg.norm(wsi, axis=1, keepdims=True)
        wsi = np.divide(wsi, norms, out=np.zeros_like(wsi), where=norms > 0)
    cnv_values = []
    for state in states:
        cnv_values.append([float(value) for value in dict(state.get("omics_evidence", {}) or {}).get("cnv_features", [])])
    matrices = {
        "ct": (ct, "euclidean"), "wsi": (wsi, "cosine"), "rna": (rna, "correlation"),
        "wxs": (np.asarray(wxs_values, dtype=float), "jaccard"),
        "cnv": (np.asarray(cnv_values, dtype=float), "euclidean"),
    }
    affinities = {}
    for modality, (values, metric) in matrices.items():
        if values.ndim != 2 or values.shape[1] == 0:
            affinities[modality] = np.eye(len(states), dtype=float)
        else:
            distance = cdist(values, values, metric=metric)
            distance = np.nan_to_num(distance, nan=1.0, posinf=1.0, neginf=1.0)
            np.fill_diagonal(distance, 0.0)
            affinities[modality] = affinity_matrix(distance, K=min(max(int(snf_config["neighbor_count"]), 1), len(states) - 1), mu=float(snf_config["mu"]))
    return {
        "eligible_patient_ids": patient_ids,
        "z_ct": ct,
        "z_wsi": wsi,
        "z_rna": rna,
        "z_wxs": np.asarray(wxs_values, dtype=float),
        "z_snf": fuse_affinities(affinities, snf_config),
        "snf_config": {**snf_config, "wxs_representation": "frequency_filtered_gene_binary", "modality_metrics": {"ct": "euclidean", "wsi": "cosine", "rna": "correlation", "wxs": "jaccard", "cnv": "euclidean"}},
        "ct_feature_names": names,
        "ct_ccc_filter": ccc_filter,
        "modality_affinities": affinities,
        "feature_engineering_audit": {"source": "legacy_test_fixture"},
    }


def fuse_affinities(affinities: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> np.ndarray:
    import snf

    networks = list(affinities.values()) if isinstance(affinities, Mapping) else list(affinities)
    if not networks:
        raise ValueError("at least one affinity network is required")
    networks = [np.asarray(network, dtype=float) for network in networks]
    shape = networks[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("affinity networks must be square")
    for network in networks:
        if network.shape != shape or not np.isfinite(network).all():
            raise ValueError("affinity networks must have equal finite shapes")
        if np.min(network) < 0 or not np.allclose(network, network.T, atol=1e-8):
            raise ValueError("affinity networks must be nonnegative and symmetric")
    networks = [(network + network.T) / 2.0 for network in networks]
    if len(networks) == 1:
        return networks[0]
    if shape[0] < 2:
        return np.ones(shape, dtype=float)
    fused = snf.snf(
        *networks,
        K=min(max(int(config["neighbor_count"]), 1), shape[0] - 1),
        t=int(config["iterations"]),
        alpha=float(config["alpha"]),
    )
    fused = (np.asarray(fused) + np.asarray(fused).T) / 2.0
    if not np.isfinite(fused).all() or np.min(fused) < 0:
        raise ValueError("SNF returned invalid fused similarity values")
    off_diagonal = ~np.eye(shape[0], dtype=bool)
    maximum = float(fused[off_diagonal].max()) if off_diagonal.any() else 0.0
    if maximum > 1.0:
        fused[off_diagonal] /= maximum
    np.fill_diagonal(fused, 1.0)
    np.fill_diagonal(fused, 1.0)
    return fused
