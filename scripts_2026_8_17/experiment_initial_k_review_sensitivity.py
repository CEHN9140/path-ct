#!/usr/bin/env python3
"""Run one independent subtype review from each saved initial K partition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import (  # noqa: E402
    current_sets,
    save_review_outputs,
)
from agents.subtype_review.runner import run_subtype_review  # noqa: E402
from utils.io import write_json  # noqa: E402

INITIAL_KS = tuple(range(2, 9))
EXPECTED_PATIENT_COUNT = 102
DEFAULT_DATA_ROOT = ROOT / "output_kirc"
DEFAULT_REVIEW_ROOT = ROOT / "output_kirc_v9" / "initial_k_review_sensitivity" / "run1"
DEFAULT_CONFIG_DIR = ROOT / "configs"


def labels_to_candidate_sets(
    payload: Mapping[str, Any],
    initial_k: int,
    expected_patient_count: int = EXPECTED_PATIENT_COUNT,
    expected_patient_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    labels = {
        str(patient_id): str(label)
        for patient_id, label in dict(payload.get("labels", {}) or {}).items()
    }
    if len(labels) != expected_patient_count:
        raise ValueError(
            f"Expected {expected_patient_count} patients for K={initial_k}, got {len(labels)}"
        )
    if expected_patient_ids is not None and set(labels) != set(expected_patient_ids):
        raise ValueError(
            "Consensus labels do not exactly match the patient-state cohort"
        )
    unique_labels = sorted(
        set(labels.values()), key=lambda value: int(value) if value.isdigit() else value
    )
    if len(unique_labels) != initial_k:
        raise ValueError(
            f"Consensus labels contain {len(unique_labels)} groups instead of K={initial_k}"
        )
    candidate_sets = []
    for index, label in enumerate(unique_labels, 1):
        members = sorted(
            patient_id for patient_id, value in labels.items() if value == label
        )
        candidate_sets.append(
            {
                "cluster_id": f"C{index:04d}",
                "member_ids": members,
                "source_views": ["snf"],
                "status": "under_review",
                "generator": {
                    "algorithm": "consensus_hierarchical",
                    "n_clusters": initial_k,
                    "seed": None,
                    "partition_id": f"consensus_hierarchical_K{initial_k}",
                    "cluster_label": int(label) if label.isdigit() else label,
                },
            }
        )
    members = [member for item in candidate_sets for member in item["member_ids"]]
    if len(members) != len(set(members)):
        raise ValueError("Consensus labels contain overlapping patients")
    return candidate_sets


def load_patient_states(data_root: Path) -> dict[str, dict[str, Any]]:
    path = data_root / "storage" / "patient_states" / "patient_states.jsonl"
    states = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        patient_id = str(state.get("case_id", state.get("Case_ID", "")) or "")
        if not patient_id:
            raise ValueError(f"Patient state has no case_id: {path}")
        if patient_id in states:
            raise ValueError(f"Duplicate patient state: {patient_id}")
        states[patient_id] = dict(state)
    return states


def load_affinity_patient_ids(data_root: Path) -> set[str]:
    path = data_root / "candidate_subtype" / "affinity_patient_order.json"
    return {str(item) for item in json.loads(path.read_text(encoding="utf-8"))}


def load_initial_partition(
    data_root: Path,
    initial_k: int,
    expected_patient_ids: set[str] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    path = (
        data_root
        / "candidate_subtype"
        / "consensus_cluster"
        / f"consensus_hierarchical_K{initial_k}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, labels_to_candidate_sets(
        payload, initial_k, expected_patient_ids=expected_patient_ids
    )


def summarize_run(
    initial_k: int,
    initial_sets: list[Mapping[str, Any]],
    state: Mapping[str, Any],
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    usage = dict(usage or {})
    final_sets = current_sets(state)
    control = dict(state.get("control", {}) or {})
    initial_sizes = sorted(
        len(item.get("member_ids", []) or []) for item in initial_sets
    )
    final_sizes = sorted(len(item.get("member_ids", []) or []) for item in final_sets)
    accepted = [item for item in final_sets if item.get("status") == "accept"]
    dropped = [item for item in final_sets if item.get("status") == "drop"]
    return {
        "initial_k": initial_k,
        "initial_cluster_sizes": initial_sizes,
        "terminal_status": str(control.get("status", "")),
        "rounds_used": int(control.get("round", 0) or 0),
        "api_calls": usage.get("api_calls"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "final_k": len(final_sets),
        "final_cluster_sizes": final_sizes,
        "accepted_patient_count": len(
            {member for item in accepted for member in item.get("member_ids", [])}
        ),
        "dropped_set_count": len(dropped),
    }


def run_one(
    initial_k: int,
    data_root: Path,
    review_root: Path,
    config_dir: Path,
) -> dict[str, Any]:
    patient_states_by_id = load_patient_states(data_root)
    source_path, initial_sets = load_initial_partition(
        data_root,
        initial_k,
        expected_patient_ids=load_affinity_patient_ids(data_root),
    )
    run_root = review_root / f"K{initial_k}"
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "initial_partition.json",
        {
            "initial_k": initial_k,
            "source": str(source_path),
            "patient_count": sum(len(item["member_ids"]) for item in initial_sets),
            "candidate_sets": initial_sets,
        },
    )
    state = run_subtype_review(
        initial_sets,
        patient_states_by_id,
        str(data_root),
        str(config_dir),
        artifact_root=str(run_root),
    )
    save_review_outputs(state, str(run_root), direct=True)
    usage = dict(state.get("control", {}).get("llm_usage", {}) or {})
    summary = summarize_run(initial_k, initial_sets, state, usage)
    metadata = {
        "experiment": "initial_k_review_sensitivity",
        "run": 1,
        "initial_k": initial_k,
        "initial_partition_source": str(source_path),
        "patient_count": sum(len(item["member_ids"]) for item in initial_sets),
        "max_rounds": state["control"]["max_rounds"],
        "llm_usage": usage,
        "terminal_status": summary["terminal_status"],
        "initial_cluster_sizes": summary["initial_cluster_sizes"],
        "final_k": summary["final_k"],
        "final_cluster_sizes": summary["final_cluster_sizes"],
    }
    write_json(run_root / "run_metadata.json", metadata)
    return summary


def write_run_summary(review_root: Path, rows: list[dict[str, Any]]) -> None:
    write_json(
        review_root / "run1_summary.json",
        {"experiment": "initial_k_review_sensitivity", "run": 1, "runs": rows},
    )
    fields = [
        "initial_k",
        "initial_cluster_sizes",
        "terminal_status",
        "rounds_used",
        "api_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "final_k",
        "final_cluster_sizes",
        "accepted_patient_count",
        "dropped_set_count",
    ]
    with (review_root / "run1_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False)
                    if isinstance(row[key], list)
                    else row[key]
                    for key in fields
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--initial-k", type=int, choices=INITIAL_KS, action="append")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows = [
        run_one(k, args.data_root, args.review_root, args.config_dir)
        for k in args.initial_k or INITIAL_KS
    ]
    write_run_summary(args.review_root, rows)
    print(
        json.dumps(
            {
                "run": 1,
                "initial_k": [row["initial_k"] for row in rows],
                "output_root": str(args.review_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
