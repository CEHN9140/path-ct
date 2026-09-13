"""Shared inputs and exact candidate construction for 5-view experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.candidate_proposer import consensus_records_from_similarity
from utils.llm_utils import load_candidate_proposer_config
from utils.io import write_json


def load_five_view_inputs(data_root: Path, config_dir: Path):
    candidate_dir = data_root / "candidate_subtype"
    order = [str(item) for item in json.loads(
        (candidate_dir / "affinity_patient_order.json").read_text(encoding="utf-8")
    )]
    cache = json.loads((candidate_dir / "affinity_cache.json").read_text(encoding="utf-8"))
    if [str(item) for item in cache.get("patient_ids", [])] != order:
        raise ValueError("Canonical affinity cache patient order mismatch.")
    paths = dict(cache.get("paths", {}) or {})
    view_names = ("ct", "wsi", "rna", "wxs", "cnv")
    if any(name not in paths for name in view_names):
        raise ValueError("Canonical 5-view affinity cache is incomplete.")
    def resolve_path(name: str) -> Path:
        recorded = Path(paths[name])
        candidates = [candidate_dir / recorded.name] if name in {"ct", "wsi", "rna"} else [data_root / "wxs" / recorded.name]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Canonical {name} affinity is missing; checked: "
            + ", ".join(str(path) for path in candidates)
        )

    matrices = {
        name: np.asarray(np.load(resolve_path(name)), dtype=float)
        for name in view_names
    }
    fused = np.asarray(np.load(candidate_dir / "fused_similarity.npy"), dtype=float)
    expected_shape = (len(order), len(order))
    if fused.shape != expected_shape or any(matrix.shape != expected_shape for matrix in matrices.values()):
        raise ValueError("Canonical 5-view affinity shapes do not match patient order.")
    config = load_candidate_proposer_config(config_dir)
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
