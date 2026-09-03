"""Shared inputs and exact candidate construction for 5-view experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from agents.candidate_proposer import (
    consensus_records_from_similarity,
    fuse_affinities,
)
from utils.io import write_json


def load_five_view_inputs(data_root: Path, stable_root: Path, output_root: Path, config_dir: Path):
    module_path = Path(__file__).with_name("13_experiment_genomic_fusion_sensitivity.py")
    spec = importlib.util.spec_from_file_location("genomic_fusion_sensitivity", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order, matrices, _, config, snf, _ = module.load_inputs(
        data_root, stable_root, config_dir
    )
    fused = fuse_affinities([matrices[name] for name in ("ct", "wsi", "rna", "wxs", "cnv")], snf)
    np.save(output_root / "fused_similarity_5view.npy", fused)
    np.save(output_root / "affinity_patient_order.npy", np.asarray(order, dtype=object))
    return order, matrices, fused, config


def build_consensus(fused: np.ndarray, config: dict, output_root: Path, patient_ids: list[str]):
    clustering_config = dict(config["clustering"])
    records, partitions = consensus_records_from_similarity(fused, clustering_config)
    payload = [{
        key: value.tolist() if isinstance(value, np.ndarray) else list(value) if isinstance(value, tuple) else value
        for key, value in record.items() if key not in {"consensus"}
    } for record in records]
    write_json(output_root / "candidate_k_diagnostics.json", {"records": payload})
    return records, partitions


def candidate_sets(labels, patient_ids, source_views):
    return [
        {
            "cluster_id": f"C{i + 1:04d}",
            "member_ids": [patient_ids[j] for j, value in enumerate(labels) if value == label],
            "source_views": list(source_views),
            "status": "under_review",
            "generator": {
                "algorithm": "consensus_hierarchical",
                "n_clusters": len(set(labels)),
                "partition_id": f"five_view_consensus_K{len(set(labels))}",
                "cluster_label": int(label),
            },
        }
        for i, label in enumerate(sorted(set(labels)))
    ]
