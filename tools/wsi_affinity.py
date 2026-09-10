from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def build_wsi_affinity(
    patient_states: Sequence[Mapping[str, Any]], *, config_dir: str = ""
) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.distance import cdist
    from tools.evidence_features import distance_to_affinity
    from utils.llm_utils import load_candidate_proposer_config

    config = load_candidate_proposer_config(config_dir).get("snf", {})
    states = [dict(state) for state in patient_states if state.get("qc") == "success"]
    case_ids = [str(state.get("case_id", "")) for state in states]
    vectors = []
    for state in states:
        evidence = dict(state.get("wsi_evidence", {}) or {})
        path = Path(str(evidence.get("feature_path", "") or ""))
        if path.suffix == ".pt" and path.with_suffix(".npy").is_file():
            path = path.with_suffix(".npy")
        vectors.append(np.asarray(np.load(path), dtype=float).reshape(-1))
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    if len(case_ids) == 1:
        affinity = np.ones((1, 1), dtype=float)
    else:
        distance = cdist(matrix, matrix, metric="cosine")
        affinity = distance_to_affinity(distance, config)
    return {
        "affinity": np.asarray(affinity, dtype=float),
        "patient_ids": case_ids,
        "feature_count": int(matrix.shape[1]),
        "audit": {
            "feature_count": int(matrix.shape[1]),
            "normalization": "row_l2",
            "metric": "cosine",
        },
    }
