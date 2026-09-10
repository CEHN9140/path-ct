#!/usr/bin/env python3
"""Run the current Subtype Review on saved five-view K partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import save_review_outputs
from agents.subtype_review.llm import review_signature_manifest
from agents.subtype_review.runner import run_subtype_review
from utils.io import write_json
from utils.llm_utils import load_yaml_file

VIEWS = ("ct", "wsi", "rna", "wxs", "cnv")
INITIAL_KS = tuple(range(2, 9))
REPEATS = (1, 2, 3)


def parse_values(values, default):
    return tuple(sorted(set(values or default)))


def file_sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def cache_reusable(summary, metadata, identity):
    return (
        summary.get("status") == "review_complete"
        and summary.get("raw_control_status") == "complete"
        and all(metadata.get(key) == value for key, value in identity.items())
    )


def load_main_inputs(data_root: Path):
    candidate_dir = data_root / "candidate_subtype"
    order = [str(item) for item in json.loads(
        (candidate_dir / "affinity_patient_order.json").read_text(encoding="utf-8")
    )]
    cache = json.loads(
        (candidate_dir / "affinity_cache.json").read_text(encoding="utf-8")
    )
    if [str(item) for item in cache.get("patient_ids", [])] != order:
        raise ValueError("Canonical affinity cache patient order mismatch.")
    paths = dict(cache.get("paths", {}) or {})
    if any(name not in paths for name in VIEWS):
        raise ValueError("Current main output does not contain all five view affinities.")
    matrices = {name: np.asarray(np.load(paths[name]), dtype=float) for name in VIEWS}
    fused = np.asarray(np.load(candidate_dir / "fused_similarity.npy"), dtype=float)
    expected_shape = (len(order), len(order))
    if fused.shape != expected_shape or any(
        matrix.shape != expected_shape for matrix in matrices.values()
    ):
        raise ValueError("Canonical five-view affinity shapes do not match patient order.")
    return order, matrices, fused


def load_initial_partition(data_root: Path, initial_k: int, patient_ids: list[str]):
    path = (
        data_root
        / "candidate_subtype"
        / "consensus_cluster"
        / f"consensus_hierarchical_K{initial_k}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = {str(case_id): int(label) for case_id, label in payload["labels"].items()}
    if set(labels) != set(patient_ids) or int(payload["n_clusters"]) != initial_k:
        raise ValueError(f"Saved K={initial_k} partition does not match patient order.")
    groups = []
    for index, label in enumerate(sorted(set(labels.values())), 1):
        members = [case_id for case_id in patient_ids if labels[case_id] == label]
        groups.append({
            "cluster_id": f"C{index:04d}",
            "member_ids": members,
            "source_views": list(VIEWS),
            "status": "under_review",
            "generator": {
                "algorithm": "consensus_hierarchical",
                "n_clusters": initial_k,
                "partition_id": f"consensus_hierarchical_K{initial_k}",
                "cluster_label": label,
            },
        })
    if len(groups) != initial_k or any(not group["member_ids"] for group in groups):
        raise ValueError(f"Saved K={initial_k} partition contains an invalid cluster.")
    return groups


def load_patient_states(data_root: Path):
    path = data_root / "storage" / "patient_states" / "patient_states.jsonl"
    states = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                states[str(state["case_id"])] = state
    return states


def run(
    data_root: Path,
    config_dir: Path,
    output_root: Path,
    initial_ks: tuple[int, ...],
    repeats: tuple[int, ...],
    force: bool = False,
):
    patient_ids, _, fused = load_main_inputs(data_root)
    patient_states = load_patient_states(data_root)
    if not set(patient_ids).issubset(patient_states):
        raise ValueError("Main patient-state records do not cover affinity patient order.")
    output_root.mkdir(parents=True, exist_ok=True)
    review_config = load_yaml_file(config_dir / "subtype_review.yaml")
    review_signature = review_signature_manifest(review_config, config_dir)
    candidate_dir = data_root / "candidate_subtype"
    input_identity = {
        "review_signature": review_signature["review_signature"],
        "fused_similarity_sha256": file_sha256(candidate_dir / "fused_similarity.npy"),
        "patient_order_sha256": file_sha256(candidate_dir / "affinity_patient_order.json"),
    }
    write_json(output_root / "experiment_manifest.json", {
        "experiment": "five_view_multi_k_agent_review",
        "input": str((data_root / "candidate_subtype").resolve()),
        "views": list(VIEWS),
        "patient_count": len(patient_ids),
        "fused_shape": list(fused.shape),
        "initial_k": list(initial_ks),
        "repeats": list(repeats),
    })

    rows = []
    for repeat in repeats:
        for initial_k in initial_ks:
            run_root = output_root / f"run{repeat}" / f"K{initial_k}"
            summary_path = run_root / "final_review_summary.json"
            metadata_path = run_root / "run_metadata.json"
            initial_sets = load_initial_partition(data_root, initial_k, patient_ids)
            initial_partition = {
                "initial_k": initial_k,
                "repeat": repeat,
                "views": list(VIEWS),
                "candidate_sets": initial_sets,
            }
            identity = {
                **input_identity,
                "initial_partition_sha256": json_sha256(initial_partition),
            }
            if summary_path.exists() and not force:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if cache_reusable(summary, metadata, identity):
                    rows.append({"repeat": repeat, "initial_k": initial_k, **summary})
                    continue
                raise RuntimeError(
                    f"Cannot reuse run{repeat}/K{initial_k}: status or cache identity mismatch; "
                    "use --force to rerun this run."
                )
            if run_root.exists() and any(run_root.iterdir()) and not force:
                raise RuntimeError(
                    f"Run directory is incomplete: {run_root}; use --force to rerun it."
                )
            if force and run_root.exists():
                shutil.rmtree(run_root)
            run_root.mkdir(parents=True, exist_ok=True)
            write_json(run_root / "initial_partition.json", initial_partition)
            state = run_subtype_review(
                initial_sets,
                patient_states,
                str(data_root),
                str(config_dir),
                artifact_root=str(run_root),
            )
            summary = save_review_outputs(state, str(run_root), direct=True)
            metadata = {
                "experiment": "five_view_multi_k_agent_review",
                "initial_k": initial_k,
                "repeat": repeat,
                "patient_count": len(patient_ids),
                "status": summary.get("status"),
                **identity,
            }
            write_json(run_root / "run_metadata.json", metadata)
            if not cache_reusable(summary, metadata, identity):
                raise RuntimeError(
                    f"Subtype Review failed for run{repeat}/K{initial_k}: "
                    f"{summary.get('status')}"
                )
            rows.append({"repeat": repeat, "initial_k": initial_k, **summary})
    write_json(output_root / "agent_discovery_summary.json", {
        "experiment": "five_view_multi_k_agent_review",
        "views": list(VIEWS),
        "runs": rows,
    })
    return {
        "output_root": str(output_root),
        "initial_k": list(initial_ks),
        "repeats": list(repeats),
        "run_count": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output_kirc_v13/00_five_view_multi_k_agent_review",
    )
    parser.add_argument("--initial-k", dest="initial_ks", type=int, choices=INITIAL_KS, action="append")
    parser.add_argument("--repeat", dest="repeats", type=int, choices=REPEATS, action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.initial_ks = parse_values(args.initial_ks, INITIAL_KS)
    args.repeats = parse_values(args.repeats, REPEATS)
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
