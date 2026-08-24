#!/usr/bin/env python3
"""Review saved K=2-8 partitions three times and find recurrent accepted cores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.subtype_review.graph import save_review_outputs  # noqa: E402
from agents.subtype_review.runner import run_subtype_review  # noqa: E402
from scripts_2026_8_17.experiment_initial_k_review_sensitivity import (  # noqa: E402
    labels_to_candidate_sets,
    load_affinity_patient_ids,
    load_patient_states,
)
from utils.io import write_json  # noqa: E402

INITIAL_KS = tuple(range(2, 9))
REPEATS = (1, 2, 3)
THRESHOLDS = (0.5, 2 / 3, 0.75, 0.8)
DEFAULT_DATA_ROOT = ROOT / "output_kirc"
DEFAULT_EXPERIMENT_ROOT = (
    ROOT / "output_kirc_v9" / "experiment_multi_k_accepted_core_stability"
)


def accepted_assignments(final_sets: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    assignments = {}
    for item in final_sets:
        set_id = str(item.get("set_id", item.get("cluster_id", "")) or "")
        if not set_id:
            raise ValueError("Accepted set has no set_id")
        for patient_id in item.get("member_ids", []) or []:
            patient_id = str(patient_id)
            if patient_id in assignments:
                raise ValueError(f"Patient occurs in multiple accepted sets: {patient_id}")
            assignments[patient_id] = set_id
    return dict(sorted(assignments.items()))


def scientifically_terminal(summary: Mapping[str, Any]) -> bool:
    raw_status = str(summary.get("raw_control_status", ""))
    review_status = str(summary.get("status", ""))
    return (
        raw_status == "complete"
        and review_status == "review_complete"
    )


def compare_runs(left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, Any]:
    left_ids, right_ids = set(left), set(right)
    shared = sorted(left_ids & right_ids)
    union = left_ids | right_ids
    result = {
        "left_accepted_count": len(left_ids),
        "right_accepted_count": len(right_ids),
        "accepted_intersection_count": len(shared),
        "accepted_union_count": len(union),
        "both_empty": not union,
        "accepted_coverage_jaccard": len(shared) / len(union) if union else None,
        "adjusted_rand_index": None,
        "adjusted_mutual_information": None,
    }
    if len(shared) >= 2:
        left_labels = [left[item] for item in shared]
        right_labels = [right[item] for item in shared]
        result["adjusted_rand_index"] = float(
            adjusted_rand_score(left_labels, right_labels)
        )
        result["adjusted_mutual_information"] = float(
            adjusted_mutual_info_score(left_labels, right_labels)
        )

    left_groups = [
        {patient for patient, label in left.items() if label == group}
        for group in sorted(set(left.values()))
    ]
    right_groups = [
        {patient for patient, label in right.items() if label == group}
        for group in sorted(set(right.values()))
    ]
    denominator = max(len(left_groups), len(right_groups))
    if not denominator:
        result.update({"matched_mean_jaccard": None, "matched_mean_dice": None})
        return result
    scores = np.zeros((len(left_groups), len(right_groups)), dtype=float)
    dice = np.zeros_like(scores)
    for i, left_group in enumerate(left_groups):
        for j, right_group in enumerate(right_groups):
            overlap = len(left_group & right_group)
            scores[i, j] = overlap / len(left_group | right_group)
            dice[i, j] = 2 * overlap / (len(left_group) + len(right_group))
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        result["matched_mean_jaccard"] = float(scores[rows, columns].sum() / denominator)
        result["matched_mean_dice"] = float(dice[rows, columns].sum() / denominator)
    else:
        result.update({"matched_mean_jaccard": 0.0, "matched_mean_dice": 0.0})
    return result


def stability_matrices(
    assignments_by_run: Sequence[Mapping[str, str]], patient_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not assignments_by_run:
        raise ValueError("No completed Review runs were provided")
    patient_ids = list(patient_ids)
    index = {patient_id: i for i, patient_id in enumerate(patient_ids)}
    same_set = np.zeros((len(patient_ids), len(patient_ids)), dtype=float)
    coaccepted = np.zeros_like(same_set)
    acceptance = np.zeros(len(patient_ids), dtype=float)
    for assignments in assignments_by_run:
        accepted = []
        groups: dict[str, list[int]] = {}
        for patient_id, label in assignments.items():
            if patient_id not in index:
                raise ValueError(f"Unknown patient in accepted set: {patient_id}")
            patient_index = index[patient_id]
            acceptance[patient_index] += 1
            accepted.append(patient_index)
            groups.setdefault(str(label), []).append(patient_index)
        coaccepted[np.ix_(accepted, accepted)] += 1
        for members in groups.values():
            same_set[np.ix_(members, members)] += 1
    denominator = len(assignments_by_run)
    conditional = np.divide(
        same_set,
        coaccepted,
        out=np.zeros_like(same_set),
        where=coaccepted > 0,
    )
    return same_set / denominator, coaccepted / denominator, conditional, acceptance / denominator


def extract_cores(
    conditional: np.ndarray,
    coacceptance: np.ndarray,
    acceptance: np.ndarray,
    patient_ids: Sequence[str],
    *,
    threshold: float,
    min_size: int,
) -> list[dict[str, Any]]:
    eligible = np.flatnonzero(acceptance >= threshold)
    if len(eligible) < min_size:
        return []
    if len(eligible) == 1:
        labels = np.ones(1, dtype=int)
    else:
        similarity = np.minimum(
            conditional[np.ix_(eligible, eligible)],
            coacceptance[np.ix_(eligible, eligible)],
        ).clip(0, 1)
        np.fill_diagonal(similarity, 1)
        tree = linkage(squareform(1 - similarity, checks=False), method="complete")
        labels = fcluster(tree, t=1 - threshold + 1e-12, criterion="distance")
    groups = []
    for label in sorted(set(labels)):
        members = sorted(str(patient_ids[index]) for index in eligible[labels == label])
        if len(members) >= min_size:
            groups.append({"member_ids": members})
    return sorted(groups, key=lambda item: (-len(item["member_ids"]), item["member_ids"]))


def core_recurrence(
    member_ids: Sequence[str], runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    recovered_by_k: Counter[int] = Counter()
    all_accepted = 0
    same_set = 0
    for run in runs:
        assignments = dict(run["assignments"])
        labels = [assignments.get(patient_id) for patient_id in member_ids]
        if labels and None not in labels:
            all_accepted += 1
        if labels and None not in labels and len(set(labels)) == 1:
            same_set += 1
            recovered_by_k[int(run["initial_k"])] += 1
    return {
        "all_members_accepted_run_count": all_accepted,
        "all_members_accepted_run_fraction": all_accepted / len(runs) if runs else 0.0,
        "same_set_run_count": same_set,
        "same_set_run_fraction": same_set / len(runs) if runs else 0.0,
        "conditional_same_set_fraction": same_set / all_accepted if all_accepted else 0.0,
        "k_coverage_any": sum(count >= 1 for count in recovered_by_k.values()),
        "k_coverage_majority": sum(count >= 2 for count in recovered_by_k.values()),
        "recovered_runs_by_k": dict(sorted(recovered_by_k.items())),
    }


def analyze(
    experiment_root: Path,
    patient_ids: Sequence[str],
    initial_ks: Sequence[int],
    repeats: Sequence[int],
    min_core_size: int,
) -> dict[str, Any]:
    runs = []
    execution_rows = []
    for repeat in repeats:
        for initial_k in initial_ks:
            run_root = experiment_root / f"run{repeat}" / f"K{initial_k}"
            summary_path = run_root / "final_review_summary.json"
            metadata_path = run_root / "run_metadata.json"
            if not summary_path.exists() or not metadata_path.exists():
                execution_rows.append({
                    "run_id": f"run{repeat}_K{initial_k}", "initial_k": initial_k,
                    "repeat": repeat, "valid_for_analysis": False,
                    "raw_control_status": "missing", "review_status": "missing",
                    "partition_assessment": "missing",
                    "rounds_used": None, "accepted_set_count": None,
                    "accepted_patient_count": None, "api_calls": None, "total_tokens": None,
                })
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            raw_status = str(summary.get("raw_control_status", ""))
            review_status = str(summary.get("status", ""))
            valid_for_analysis = scientifically_terminal(summary)
            assignments = accepted_assignments(summary.get("accepted_subtype_sets", []))
            usage = dict(summary.get("llm_usage", {}) or {})
            row = {
                "run_id": f"run{repeat}_K{initial_k}", "initial_k": initial_k,
                "repeat": repeat, "valid_for_analysis": valid_for_analysis,
                "raw_control_status": raw_status, "review_status": review_status,
                "partition_assessment": summary.get("partition_assessment", "unavailable"),
                "rounds_used": summary.get("rounds_used"),
                "accepted_set_count": len(set(assignments.values())),
                "accepted_patient_count": len(assignments),
                "api_calls": usage.get("api_calls"), "total_tokens": usage.get("total_tokens"),
            }
            execution_rows.append(row)
            if valid_for_analysis:
                runs.append({**row, "assignments": assignments})

    experiment_root.mkdir(parents=True, exist_ok=True)
    with (experiment_root / "run_execution_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(execution_rows[0]) if execution_rows else [])
        if execution_rows:
            writer.writeheader()
            writer.writerows(execution_rows)
    write_json(experiment_root / "run_execution_status.json", {"runs": execution_rows})
    if not runs:
        summary = {
            "experiment": "multi_k_accepted_core_stability",
            "analysis_status": "unavailable",
            "expected_run_count": len(initial_ks) * len(repeats),
            "valid_run_count": 0,
            "invalid_run_count": len(execution_rows),
            "interpretation": "No scientifically complete Review run is available.",
        }
        write_json(experiment_root / "summary.json", summary)
        return summary

    run_rows = [
        {key: value for key, value in run.items() if key != "assignments"}
        for run in runs
    ]
    pairwise_rows = []
    for left, right in combinations(runs, 2):
        pairwise_rows.append(
            {
                "left_run": left["run_id"],
                "right_run": right["run_id"],
                "same_initial_k": left["initial_k"] == right["initial_k"],
                **compare_runs(left["assignments"], right["assignments"]),
            }
        )

    joint, coacceptance, conditional, acceptance = stability_matrices(
        [run["assignments"] for run in runs], patient_ids
    )
    patient_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    threshold_rows, cores_by_threshold = [], {}
    for threshold in THRESHOLDS:
        cores = extract_cores(
            conditional,
            coacceptance,
            acceptance,
            patient_ids,
            threshold=threshold,
            min_size=min_core_size,
        )
        cores_by_threshold[threshold] = cores
        threshold_rows.append(
            {
                "threshold": threshold,
                "core_count": len(cores),
                "core_patient_count": sum(len(core["member_ids"]) for core in cores),
                "core_sizes": json.dumps([len(core["member_ids"]) for core in cores]),
            }
        )

    primary_threshold = 2 / 3
    primary_cores = cores_by_threshold[primary_threshold]
    core_rows, membership_rows = [], []
    for core_number, core in enumerate(primary_cores, 1):
        core_id = f"CORE{core_number:02d}"
        members = core["member_ids"]
        indices = [patient_index[member] for member in members]
        within_conditional = conditional[np.ix_(indices, indices)]
        within_coacceptance = coacceptance[np.ix_(indices, indices)]
        pair_indices = np.triu_indices(len(indices), k=1)
        within_values = within_conditional[pair_indices]
        coacceptance_values = within_coacceptance[pair_indices]
        outside_indices = [index for index in range(len(patient_ids)) if index not in indices]
        outside_values = conditional[np.ix_(indices, outside_indices)].ravel()
        recurrence = core_recurrence(members, runs)
        core_rows.append(
            {
                "core_id": core_id,
                "core_size": len(members),
                **{
                    **recurrence,
                    "recovered_runs_by_k": json.dumps(
                        recurrence["recovered_runs_by_k"], ensure_ascii=False
                    ),
                },
                "mean_within_conditional_coassignment": float(within_values.mean()),
                "min_within_conditional_coassignment": float(within_values.min()),
                "mean_within_coacceptance": float(coacceptance_values.mean()),
                "min_within_coacceptance": float(coacceptance_values.min()),
                "mean_outside_conditional_coassignment": float(outside_values.mean()) if outside_values.size else 0.0,
                "max_outside_conditional_coassignment": float(outside_values.max()) if outside_values.size else 0.0,
                "mean_acceptance_frequency": float(acceptance[indices].mean()),
                "min_acceptance_frequency": float(acceptance[indices].min()),
                "member_ids": json.dumps(members, ensure_ascii=False),
            }
        )
        membership_rows.extend(
            {
                "core_id": core_id,
                "patient_id": member,
                "acceptance_frequency": float(acceptance[patient_index[member]]),
            }
            for member in members
        )

    write_json(experiment_root / "run_summary.json", {"runs": run_rows})
    for name, rows in (
        ("run_summary.csv", run_rows),
        ("pairwise_partition_consistency.csv", pairwise_rows),
        (
            "patient_acceptance_frequency.csv",
            [
                {"patient_id": patient_id, "acceptance_frequency": float(acceptance[index])}
                for index, patient_id in enumerate(patient_ids)
            ],
        ),
        ("stable_core_membership.csv", membership_rows),
        ("stable_core_summary.csv", core_rows),
        ("threshold_sensitivity.csv", threshold_rows),
    ):
        fields = list(rows[0]) if rows else []
        with (experiment_root / name).open("w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    for name, matrix in (
        ("joint_accepted_coassignment_matrix.csv", joint),
        ("pairwise_coacceptance_matrix.csv", coacceptance),
        ("conditional_membership_coassignment_matrix.csv", conditional),
    ):
        with (experiment_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", *patient_ids])
            for patient_id, row in zip(patient_ids, matrix):
                writer.writerow([patient_id, *map(float, row)])

    import matplotlib.pyplot as plt

    plot_matrix = conditional.copy()
    np.fill_diagonal(plot_matrix, 1)
    if len(patient_ids) > 1:
        order = leaves_list(
            linkage(squareform(1 - plot_matrix, checks=False), method="complete")
        )
    else:
        order = np.arange(len(patient_ids))
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(conditional[np.ix_(order, order)], vmin=0, vmax=1, cmap="viridis")
    axis.set_title(f"Conditional accepted-set co-assignment across {len(runs)} valid Review runs")
    axis.set_xlabel("Patients (complete-linkage order)")
    axis.set_ylabel("Patients (complete-linkage order)")
    figure.colorbar(image, ax=axis, label="P(same set | both accepted)")
    figure.tight_layout()
    figure.savefig(experiment_root / "coassignment_heatmap.png", dpi=180)
    plt.close(figure)

    same_k = [row for row in pairwise_rows if row["same_initial_k"]]
    different_k = [row for row in pairwise_rows if not row["same_initial_k"]]
    same_k_ari = [
        row["adjusted_rand_index"]
        for row in same_k
        if row["adjusted_rand_index"] is not None
    ]
    different_k_ari = [
        row["adjusted_rand_index"]
        for row in different_k
        if row["adjusted_rand_index"] is not None
    ]
    same_k_informative = [row for row in same_k if not row["both_empty"]]
    different_k_informative = [row for row in different_k if not row["both_empty"]]
    summary = {
        "experiment": "multi_k_accepted_core_stability",
        "analysis_status": "complete" if len(runs) == len(initial_ks) * len(repeats) else "partial",
        "initial_k_values": list(initial_ks),
        "review_repeats": list(repeats),
        "expected_run_count": len(initial_ks) * len(repeats),
        "valid_run_count": len(runs),
        "invalid_run_count": len(execution_rows) - len(runs),
        "partition_assessment_counts": dict(sorted(Counter(
            str(run.get("partition_assessment", "unavailable")) for run in runs
        ).items())),
        "patient_count": len(patient_ids),
        "accepted_only": True,
        "scientific_denominator": "raw_control_status=complete and status=review_complete",
        "primary_core_definition": {
            "acceptance_frequency": primary_threshold,
            "minimum_pairwise_coacceptance": primary_threshold,
            "minimum_conditional_coassignment": primary_threshold,
            "minimum_core_size": min_core_size,
        },
        "primary_core_count": len(core_rows),
        "primary_core_patient_count": len(membership_rows),
        "primary_cores": core_rows,
        "threshold_sensitivity": threshold_rows,
        "within_k_repeatability": {
            "pair_count": len(same_k),
            "informative_pair_count": len(same_k_informative),
            "mean_ari": float(np.mean(same_k_ari)) if same_k_ari else None,
            "mean_coverage_jaccard": float(np.mean([row["accepted_coverage_jaccard"] for row in same_k_informative])) if same_k_informative else None,
            "mean_matched_jaccard": float(np.mean([row["matched_mean_jaccard"] for row in same_k_informative])) if same_k_informative else None,
            "mean_matched_dice": float(np.mean([row["matched_mean_dice"] for row in same_k_informative])) if same_k_informative else None,
        },
        "between_k_sensitivity": {
            "pair_count": len(different_k),
            "informative_pair_count": len(different_k_informative),
            "mean_ari": float(np.mean(different_k_ari)) if different_k_ari else None,
            "mean_coverage_jaccard": float(np.mean([row["accepted_coverage_jaccard"] for row in different_k_informative])) if different_k_informative else None,
            "mean_matched_jaccard": float(np.mean([row["matched_mean_jaccard"] for row in different_k_informative])) if different_k_informative else None,
            "mean_matched_dice": float(np.mean([row["matched_mean_dice"] for row in different_k_informative])) if different_k_informative else None,
        },
        "interpretation": (
            "Recurrent accepted cores indicate cross-K structural repeatability, "
            "not biological subtype validation."
        ),
    }
    write_json(experiment_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--initial-k", type=int, choices=INITIAL_KS, action="append")
    parser.add_argument("--repeat", type=int, choices=REPEATS, action="append")
    parser.add_argument("--min-core-size", type=int, default=5)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    initial_ks = sorted(set(args.initial_k or INITIAL_KS))
    repeats = sorted(set(args.repeat or REPEATS))
    patient_ids = sorted(load_affinity_patient_ids(args.data_root))
    patient_states = load_patient_states(args.data_root)
    review_files = [
        *sorted((ROOT / "agents" / "subtype_review").rglob("*.py")),
        *sorted((ROOT / "agents" / "subtype_review").rglob("*.md")),
        ROOT / "configs" / "subtype_review.yaml",
        *[
            ROOT / "tools" / name
            for name in (
                "cnv_characterization.py", "confound.py", "known_label_echo.py",
                "multimodal_consistency_check.py", "mutation_enrichment.py",
                "pathway_enrichment.py", "structural_adequacy.py",
            )
        ],
    ]
    review_signature = hashlib.sha256(
        b"".join(path.read_bytes() for path in review_files)
    ).hexdigest()

    if not args.analyze_only:
        for repeat in repeats:
            for initial_k in initial_ks:
                run_root = args.experiment_root / f"run{repeat}" / f"K{initial_k}"
                source_path = (
                    args.data_root
                    / "candidate_subtype"
                    / "consensus_cluster"
                    / f"consensus_hierarchical_K{initial_k}.json"
                )
                metadata_path = run_root / "run_metadata.json"
                summary_path = run_root / "final_review_summary.json"
                reusable = False
                if metadata_path.exists() and summary_path.exists() and not args.force:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    reusable = (
                        metadata.get("initial_k") == initial_k
                        and metadata.get("repeat") == repeat
                        and metadata.get("source") == str(source_path)
                        and metadata.get("review_signature") == review_signature
                        and scientifically_terminal(summary)
                    )
                if reusable:
                    print(f"[reuse] run{repeat}/K{initial_k}")
                    continue
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                initial_sets = labels_to_candidate_sets(
                    payload,
                    initial_k,
                    expected_patient_count=len(patient_ids),
                    expected_patient_ids=set(patient_ids),
                )
                run_root.mkdir(parents=True, exist_ok=True)
                write_json(
                    run_root / "initial_partition.json",
                    {
                        "initial_k": initial_k,
                        "repeat": repeat,
                        "source": str(source_path),
                        "review_signature": review_signature,
                        "patient_count": len(patient_ids),
                        "candidate_sets": initial_sets,
                    },
                )
                state = run_subtype_review(
                    initial_sets,
                    patient_states,
                    str(args.data_root),
                    str(args.config_dir),
                    artifact_root=str(run_root),
                )
                summary = save_review_outputs(state, str(run_root), direct=True)
                write_json(
                    metadata_path,
                    {
                        "experiment": "multi_k_accepted_core_stability",
                        "initial_k": initial_k,
                        "repeat": repeat,
                        "source": str(source_path),
                        "review_signature": review_signature,
                        "patient_count": len(patient_ids),
                        "status": summary["status"],
                        "rounds_used": summary["rounds_used"],
                        "llm_usage": summary["llm_usage"],
                    },
                )
                print(f"[complete] run{repeat}/K{initial_k}: {summary['status']}")

    summary = analyze(
        args.experiment_root,
        patient_ids,
        INITIAL_KS,
        REPEATS,
        args.min_core_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
