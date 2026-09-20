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
    discovery_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    from tools.ct_radiomics import build_ct_affinity
    from tools.rna import build_rna_affinity
    from tools.wsi_affinity import build_wsi_affinity

    ct = build_ct_affinity(patient_states, config_dir=config_dir, output_root=output_root)
    wsi = build_wsi_affinity(patient_states, config_dir=config_dir)
    rna = build_rna_affinity(patient_states, config_dir=config_dir)
    wxs_patient_ids = json.loads(
        Path(discovery_artifacts["wxs_patient_order_path"]).read_text(encoding="utf-8")
    )
    wxs = {
        "affinity": np.load(discovery_artifacts["wxs_affinity_path"]),
        "patient_ids": wxs_patient_ids,
        "audit": {
            "source": "wxs_discovery_features",
            "distance": "jaccard",
        },
    }
    patient_ids = list(ct["patient_ids"])
    modalities = {"ct": ct, "wsi": wsi, "rna": rna, "wxs": wxs}
    for name, result in modalities.items():
        if list(result["patient_ids"]) != patient_ids:
            raise ValueError(f"{name} patient order does not match the candidate cohort")
    return {
        "patient_ids": patient_ids,
        "modality_affinities": {
            name: result["affinity"] for name, result in modalities.items()
        },
        "audit": {
            "modalities": {
                name: result["audit"] for name, result in modalities.items()
            }
        },
    }


def fuse_affinities(
    affinities: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> np.ndarray:
    import snf

    if not affinities:
        raise ValueError("affinity networks must be nonempty")
    networks = [np.asarray(value, dtype=float) for value in affinities.values()]
    shape = networks[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("affinity networks must be square matrices")
    if any(
        network.shape != shape
        or not np.isfinite(network).all()
        or np.min(network) < 0
        or not np.allclose(network, network.T, atol=1e-8)
        for network in networks
    ):
        raise ValueError("affinity networks must have equal, finite, symmetric, nonnegative values")
    networks = [(network + network.T) / 2 for network in networks]
    if len(networks) == 1:
        return networks[0]
    n = shape[0]
    fused = snf.snf(
        *networks,
        K=min(max(int(config["neighbor_count"]), 1), n - 1),
        t=int(config["iterations"]),
        alpha=float(config["alpha"]),
    )
    fused = (np.asarray(fused, dtype=float) + np.asarray(fused, dtype=float).T) / 2
    if not np.isfinite(fused).all() or np.min(fused) < 0:
        raise ValueError("SNF returned invalid fused similarities")
    np.fill_diagonal(fused, 1.0)
    return fused
