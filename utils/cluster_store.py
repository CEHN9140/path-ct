from __future__ import annotations

from pathlib import Path
from typing import Mapping

from utils.cluster_flow import compact_cluster_state
from utils.io import build_output_path, ensure_dir, write_json
from utils.tool_utils import safe_identifier


def save_candidate_clusters(
    output_root: str,
    candidate_clusters: list[Mapping[str, object]],
) -> dict[str, object]:
    candidate_dir = ensure_dir(Path(output_root) / "candidate_subtype")
    list_path = write_json(candidate_dir / "candidate_clusters.json", candidate_clusters)
    saved_paths: dict[str, object] = {
        "candidate_subtype_dir": str(candidate_dir),
        "candidate_clusters": list_path,
    }
    cluster_paths: dict[str, str] = {}
    for cluster in candidate_clusters:
        cluster_id = str(cluster.get("cluster_id", "unknown_cluster"))
        cluster_paths[cluster_id] = write_json(
            candidate_dir / f"{safe_identifier(cluster_id)}.json",
            cluster,
        )
    saved_paths["cluster_paths"] = cluster_paths
    return saved_paths


def save_cluster_states(output_root: str, cluster_states: list[Mapping[str, object]]) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    list_path = Path(output_root) / "storage" / "cluster_states" / "cluster_states.json"
    compact_states = [compact_cluster_state(cluster_state) for cluster_state in cluster_states]
    saved_paths["cluster_states"] = write_json(list_path, compact_states)
    for cluster_state in cluster_states:
        cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
        output_path = build_output_path(output_root, "cluster_states", cluster_id, ".json")
        saved_paths[cluster_id] = write_json(output_path, compact_cluster_state(cluster_state))
    return saved_paths
