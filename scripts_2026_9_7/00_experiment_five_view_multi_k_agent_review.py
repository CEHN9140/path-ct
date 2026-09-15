#!/usr/bin/env python3
"""Run the current Subtype Review on saved five-view K partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import partition_signature, save_review_outputs
from agents.subtype_review.llm import review_signature_manifest
from agents.subtype_review.runner import run_subtype_review
from scripts_2026_9_7 import analyze_multi_k_stable_cores as core_analysis
from scripts_2026_9_7 import experiment_multi_k_accepted_core_stability as stability
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


def source_identity(root: Path):
    paths = [
        root / "agents",
        root / "tools",
        root / "utils",
        root / "configs",
        Path(__file__).resolve(),
        Path(core_analysis.__file__).resolve(),
        Path(stability.__file__).resolve(),
        ROOT / "scripts_2026_9_7" / "five_view_experiment.py",
        ROOT / "scripts_2026_9_7" / "experiment_initial_k_review_sensitivity.py",
    ]
    files = []
    for path in paths:
        files.extend(
            item for item in (path.rglob("*") if path.is_dir() else [path])
            if item.is_file() and item.suffix in {".py", ".md", ".yaml", ".yml"}
        )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "agents", "tools", "utils", "configs",
            str(Path(__file__).resolve().relative_to(root)),
            str(Path(core_analysis.__file__).resolve().relative_to(root)),
            str(Path(stability.__file__).resolve().relative_to(root)),
            "scripts_2026_9_7/five_view_experiment.py",
            "scripts_2026_9_7/experiment_initial_k_review_sensitivity.py",
        ],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    return {
        "git_commit_sha": git_commit,
        "git_worktree_clean": not bool(status.strip()),
        "source_tree_sha256": digest.hexdigest(),
    }


def cache_reusable(summary, metadata, identity):
    return (
        summary.get("status") == "review_complete"
        and summary.get("raw_control_status") == "complete"
        and all(metadata.get(key) == value for key, value in identity.items())
        and metadata.get("final_partition_signature")
        == partition_signature(summary.get("partition", {}).get("sets", []))
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
    matrices = {
        name: np.asarray(np.load(
            (candidate_dir if name in {"ct", "wsi", "rna"} else data_root / "wxs")
            / Path(paths[name]).name
        ), dtype=float)
        for name in VIEWS
    }
    fused = np.asarray(np.load(candidate_dir / "fused_similarity.npy"), dtype=float)
    expected_shape = (len(order), len(order))
    if fused.shape != expected_shape or any(
        matrix.shape != expected_shape for matrix in matrices.values()
    ):
        raise ValueError("Canonical five-view affinity shapes do not match patient order.")
    return order, matrices, fused


def load_initial_partition(data_root: Path, initial_k: int, patient_ids: list[str], active_modalities=VIEWS):
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
            "source_views": list(active_modalities),
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


def analyze_stable_cores(
    output_root: Path, patient_ids: list[str], valid_run_count: int,
    initial_ks: tuple[int, ...] = INITIAL_KS,
    repeats: tuple[int, ...] = REPEATS,
):
    expected_run_count = len(initial_ks) * len(repeats)
    if valid_run_count != expected_run_count:
        return {
            "analysis_status": "pending",
            "valid_run_count": valid_run_count,
            "expected_run_count": expected_run_count,
        }
    return stability.analyze(
        output_root, patient_ids, initial_ks, repeats, min_core_size=5
    )


def matched_core_jaccard(reference, candidate):
    if not reference and not candidate:
        return None, None
    if not reference or not candidate:
        return 0.0, 0.0
    def core_member_set(core):
        members = core["member_ids"]
        if isinstance(members, str):
            members = json.loads(members)
        return set(map(str, members))

    reference_members = [core_member_set(core) for core in reference]
    candidate_members = [core_member_set(core) for core in candidate]
    scores = np.asarray([
        [len(left & right) / len(left | right)
         for right in candidate_members]
        for left in reference_members
    ])
    rows, columns = linear_sum_assignment(1.0 - scores)
    matched = np.pad(
        scores[rows, columns], (0, max(len(reference), len(candidate)) - len(rows))
    )
    return float(matched.mean()), float(matched.min())


def leave_one_k_out(
    output_root: Path, patient_ids: list[str], initial_ks: tuple[int, ...],
    repeats: tuple[int, ...], reference_cores,
):
    rows = []
    reference_count = len(reference_cores)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for excluded_k in initial_ks:
            kept_ks = tuple(k for k in initial_ks if k != excluded_k)
            reduced_root = root / f"exclude{excluded_k}"
            for repeat in repeats:
                for initial_k in kept_ks:
                    source = output_root / f"run{repeat}" / f"K{initial_k}"
                    target = reduced_root / f"run{repeat}" / f"K{initial_k}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(source, target_is_directory=True)
            summary = stability.analyze(reduced_root, patient_ids, kept_ks, repeats, min_core_size=5)
            mean_jaccard, min_jaccard = matched_core_jaccard(
                reference_cores, summary.get("primary_cores", [])
            )
            candidate_count = summary.get("primary_core_count") or 0
            matched_count = min(reference_count, candidate_count)
            matched_mean = (
                mean_jaccard * max(reference_count, candidate_count) / matched_count
                if mean_jaccard is not None and matched_count else None
            )
            rows.append({
                "excluded_k": excluded_k,
                "valid_run_count": summary.get("valid_run_count"),
                "primary_core_count": summary.get("primary_core_count"),
                "primary_core_patient_count": summary.get("primary_core_patient_count"),
                "mean_matched_jaccard": mean_jaccard,
                "min_matched_jaccard": min_jaccard,
                "matched_core_mean_jaccard": matched_mean,
                "extra_core_count": max(0, candidate_count - reference_count),
                "missing_core_count": max(0, reference_count - candidate_count),
                "primary_cores": summary.get("primary_cores", []),
            })
    write_json(output_root / "leave_one_k_out_summary.json", {"results": rows})
    core_analysis.write_csv(
        output_root / "leave_one_k_out_summary.csv",
        [{key: value for key, value in row.items() if key != "primary_cores"} for row in rows],
    )
    return rows


def run(
    data_root: Path,
    config_dir: Path,
    output_root: Path,
    initial_ks: tuple[int, ...],
    repeats: tuple[int, ...],
    force: bool = False,
    analysis_only: bool = False,
    active_modalities: tuple[str, ...] = VIEWS,
):
    patient_ids, _, fused = load_main_inputs(data_root)
    patient_states = load_patient_states(data_root)
    if not set(patient_ids).issubset(patient_states):
        raise ValueError("Main patient-state records do not cover affinity patient order.")
    output_root.mkdir(parents=True, exist_ok=True)
    if analysis_only:
        if tuple(initial_ks) != INITIAL_KS or tuple(repeats) != REPEATS:
            raise ValueError("analysis-only requires the complete K2-K8 and repeat1-3 universe")
        summary = stability.analyze(output_root, patient_ids, initial_ks, repeats, min_core_size=5)
        if summary.get("analysis_status") != "complete":
            raise RuntimeError("analysis-only requires 21 scientifically complete runs")
        leave_one_k_out(output_root, patient_ids, initial_ks, repeats, summary.get("primary_cores", []))
        return {"output_root": str(output_root), "analysis_status": "complete", "analysis_only": True}
    review_config = load_yaml_file(config_dir / "subtype_review.yaml")
    review_signature = review_signature_manifest(review_config, config_dir)
    candidate_dir = data_root / "candidate_subtype"
    input_identity = {
        "review_signature": review_signature["review_signature"],
        "fused_similarity_sha256": file_sha256(candidate_dir / "fused_similarity.npy"),
        "patient_order_sha256": file_sha256(candidate_dir / "affinity_patient_order.json"),
        **core_analysis.scientific_input_identity(data_root),
        **source_identity(ROOT),
    }
    write_json(output_root / "experiment_manifest.json", {
        "experiment": "five_view_multi_k_agent_review",
        "input": str((data_root / "candidate_subtype").resolve()),
        "views": list(active_modalities),
        "patient_count": len(patient_ids),
        "fused_shape": list(fused.shape),
        "initial_k": list(initial_ks),
        "repeats": list(repeats),
        **input_identity,
    })

    rows = []
    for repeat in repeats:
        for initial_k in initial_ks:
            run_root = output_root / f"run{repeat}" / f"K{initial_k}"
            summary_path = run_root / "final_review_summary.json"
            metadata_path = run_root / "run_metadata.json"
            initial_sets = load_initial_partition(data_root, initial_k, patient_ids, active_modalities)
            initial_partition = {
                "initial_k": initial_k,
                "repeat": repeat,
                "views": list(active_modalities),
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
                active_modalities=active_modalities,
            )
            summary = save_review_outputs(state, str(run_root), direct=True)
            metadata = {
                "experiment": "five_view_multi_k_agent_review",
                "initial_k": initial_k,
                "repeat": repeat,
                "patient_count": len(patient_ids),
                "status": summary.get("status"),
                "final_partition_signature": partition_signature(
                    summary.get("partition", {}).get("sets", [])
                ),
                **identity,
            }
            write_json(run_root / "run_metadata.json", metadata)
            if not cache_reusable(summary, metadata, identity):
                raise RuntimeError(
                    f"Subtype Review failed for run{repeat}/K{initial_k}: "
                    f"{summary.get('status')}"
                )
            rows.append({"repeat": repeat, "initial_k": initial_k, **summary})
    completed = {}
    for summary_path in output_root.glob("run*/K*/final_review_summary.json"):
        run_root = summary_path.parent
        metadata_path = run_root / "run_metadata.json"
        if not metadata_path.exists():
            continue
        repeat = int(run_root.parent.name.removeprefix("run"))
        initial_k = int(run_root.name.removeprefix("K"))
        initial_partition = {
            "initial_k": initial_k,
            "repeat": repeat,
            "views": list(active_modalities),
            "candidate_sets": load_initial_partition(data_root, initial_k, patient_ids, active_modalities),
        }
        identity = {
            **input_identity,
            "initial_partition_sha256": json_sha256(initial_partition),
        }
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cache_reusable(summary, metadata, identity):
            completed[(repeat, initial_k)] = {
                "repeat": repeat,
                "initial_k": initial_k,
                **summary,
            }
    rows = [completed[key] for key in sorted(completed)]
    write_json(output_root / "agent_discovery_summary.json", {
        "experiment": "five_view_multi_k_agent_review",
        "views": list(active_modalities),
        "runs": rows,
    })
    if tuple(initial_ks) == INITIAL_KS and tuple(repeats) == REPEATS:
        stable_core_analysis = analyze_stable_cores(
            output_root, patient_ids, len(rows), initial_ks, repeats
        )
        if stable_core_analysis["analysis_status"] == "complete":
            leave_one_k_out(
                output_root, patient_ids, initial_ks, repeats,
                stable_core_analysis.get("primary_cores", []),
            )
    else:
        stable_core_analysis = {
            "analysis_status": "pending",
            "valid_run_count": len(rows),
            "expected_run_count": len(INITIAL_KS) * len(REPEATS),
        }
    return {
        "output_root": str(output_root),
        "initial_k": list(initial_ks),
        "repeats": list(repeats),
        "run_count": len(rows),
        "stable_core_analysis_status": stable_core_analysis["analysis_status"],
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
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    args.initial_ks = parse_values(args.initial_ks, INITIAL_KS)
    args.repeats = parse_values(args.repeats, REPEATS)
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
