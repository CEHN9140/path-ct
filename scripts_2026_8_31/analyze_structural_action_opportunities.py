#!/usr/bin/env python3
"""Audit existing structural evidence and Router actions without calling any model."""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
INITIAL_KS = tuple(range(2, 9))
REPEATS = (1, 2, 3)
MODALITIES = ("ct", "wsi", "rna", "genomic")


def json_cell(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value


def action_by_set(plan: Mapping[str, Any]) -> dict[str, str]:
    actions = {}
    for action in plan.get("actions", []) or []:
        for target in action.get("target_ids", []) or []:
            actions[str(target)] = str(action.get("action", ""))
    return actions


def value(row: Mapping[str, Any], key: str) -> Any:
    return row.get(key) if isinstance(row, Mapping) else None


def boundary_value(row: Mapping[str, Any], key: str) -> Any:
    values = [row.get(key) for key in ("left_" + key, "right_" + key)]
    values = [item for item in values if isinstance(item, (int, float))]
    return min(values) if values else None


def pair_is_mentioned(reports: list[Mapping[str, Any]], left: str, right: str) -> bool:
    for report in reports:
        if report.get("dimension") != "cross_modal_consistency":
            continue
        targets = {str(target) for target in report.get("target_ids", []) or []}
        text = json.dumps(report, ensure_ascii=False)
        if (left in targets and right in text) or (right in targets and left in text):
            return True
    return False


def audit_history(
    history: list[Mapping[str, Any]], initial_k: int, repeat: int, review_status: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_rows, merge_rows = [], []
    for entry in history:
        sets = list(entry.get("partition", {}).get("sets", []) or [])
        actions = action_by_set(entry.get("router_plan", {}) or {})
        structural = {}
        for row in entry.get("round_evidence", []) or []:
            if row.get("tool_name") == "multimodal_consistency_check":
                structural = dict(row.get("full_metrics", {}).get("structural_characterization", {}) or {})
                break
        internal = structural.get("internal_structure_by_set", {}) or {}
        boundaries = structural.get("boundary_by_pair", {}) or {}
        reports = list(entry.get("evidence_reports", []) or [])
        signature = str(entry.get("partition_signature", ""))
        partition_info = {
            "initial_k": initial_k,
            "repeat": repeat,
            "round": entry.get("round"),
            "partition_id": hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12],
            "review_status": review_status,
        }
        for item in sets:
            set_id = str(item.get("set_id") or item.get("cluster_id") or "")
            metrics = dict(internal.get(set_id, {}) or {})
            fused = dict(metrics.get("fused_binary_probe", {}) or {})
            resampling = dict(fused.get("resampling", {}) or {})
            modalities = dict(metrics.get("probe_support_by_modality", {}) or {})
            split_rows.append({
                **partition_info,
                "set_id": set_id,
                "member_n": len(item.get("member_ids", []) or []),
                "router_action": actions.get(set_id, ""),
                "fused_probe_child_sizes": fused.get("child_sizes"),
                "fused_probe_silhouette": fused.get("median_silhouette"),
                "normalized_cut": fused.get("normalized_cut"),
                "resample_median_ari": resampling.get("median_resample_ari"),
                "consensus_separation": resampling.get("consensus_separation"),
                "pac": resampling.get("pac"),
                "degenerate_fraction": resampling.get("degenerate_resample_fraction"),
                **{
                    f"{name}_probe_silhouette": value(modalities.get(name, {}), "median_silhouette")
                    for name in MODALITIES
                },
            })
        for left, right in combinations(sorted(str(item.get("set_id") or item.get("cluster_id") or "") for item in sets), 2):
            pair_key = f"{left}+{right}"
            pair = dict(boundaries.get(pair_key, {}) or {})
            fused = dict(pair.get("fused", {}) or {})
            modality_rows = dict(pair.get("modalities", {}) or {})
            row = {
                **partition_info,
                "set_a": left,
                "set_b": right,
                "action_a": actions.get(left, ""),
                "action_b": actions.get(right, ""),
                "fused_pair_silhouette": fused.get("pair_median_silhouette"),
                "fused_boundary_separation": boundary_value(fused, "boundary_separation"),
                "fused_left_boundary_separation": fused.get("left_boundary_separation"),
                "fused_right_boundary_separation": fused.get("right_boundary_separation"),
                "left_margin": fused.get("left_median_margin"),
                "right_margin": fused.get("right_median_margin"),
                "pair_metrics_available_in_raw_artifact": pair_key in boundaries,
                "pair_mentioned_in_verifier_report": pair_is_mentioned(reports, left, right),
            }
            for name in MODALITIES:
                metrics = dict(modality_rows.get(name, {}) or {})
                row[f"{name}_pair_silhouette"] = metrics.get("pair_median_silhouette")
                row[f"{name}_boundary_separation"] = boundary_value(metrics, "boundary_separation")
                row[f"{name}_left_boundary_separation"] = metrics.get("left_boundary_separation")
                row[f"{name}_right_boundary_separation"] = metrics.get("right_boundary_separation")
            merge_rows.append(row)
    return split_rows, merge_rows


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        output = [{key: json_cell(value) for key, value in row.items()} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def audit_experiment(
    experiment_root: Path,
    initial_ks: list[int] | tuple[int, ...] = INITIAL_KS,
    repeats: list[int] | tuple[int, ...] = (1,),
    output_root: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root or experiment_root / "structural_action_opportunities"
    output_root.mkdir(parents=True, exist_ok=True)
    split_rows, merge_rows = [], []
    audited_runs = []
    for repeat in repeats:
        for initial_k in initial_ks:
            run_root = experiment_root / f"run{repeat}" / f"K{initial_k}"
            history_path = run_root / "review_history.json"
            if not history_path.exists():
                continue
            summary_path = run_root / "final_review_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            current_split, current_merge = audit_history(
                json.loads(history_path.read_text(encoding="utf-8")),
                int(initial_k), int(repeat), str(summary.get("status", "missing")),
            )
            split_rows.extend(current_split)
            merge_rows.extend(current_merge)
            audited_runs.append(f"run{repeat}/K{initial_k}")
    write_table(output_root / "split_opportunities.csv", split_rows)
    write_table(output_root / "merge_opportunities.csv", merge_rows)
    (output_root / "split_opportunities.json").write_text(json.dumps(split_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "merge_opportunities.json").write_text(json.dumps(merge_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    action_counts = {}
    for row in split_rows:
        bucket = action_counts.setdefault(str(row["initial_k"]), {action: 0 for action in ("accept", "drop", "split", "merge", "need_more_evidence")})
        bucket[row["router_action"]] = bucket.get(row["router_action"], 0) + 1
    summary = {
        "experiment": "structural_action_opportunities",
        "audited_runs": audited_runs,
        "audited_k_values": sorted({int(row["initial_k"]) for row in split_rows}),
        "audited_partition_count": len({(row["initial_k"], row["repeat"], row["round"]) for row in split_rows}),
        "audited_set_count": len(split_rows),
        "audited_pair_count": len(merge_rows),
        "observed_split_action_count": sum(row["router_action"] == "split" for row in split_rows),
        "observed_merge_action_count": sum(row["action_a"] == "merge" and row["action_b"] == "merge" for row in merge_rows),
        "action_counts_by_k": action_counts,
        "pair_metrics_by_k": {},
        "raw_pair_available_verifier_unmentioned_count": sum(
            row["pair_metrics_available_in_raw_artifact"] and not row["pair_mentioned_in_verifier_report"]
            for row in merge_rows
        ),
    }
    for initial_k in summary["audited_k_values"]:
        pairs = [row for row in merge_rows if row["initial_k"] == initial_k]
        fused_silhouettes = [
            row["fused_pair_silhouette"] for row in pairs
            if isinstance(row["fused_pair_silhouette"], (int, float))
        ]
        fused_boundaries = [
            row["fused_boundary_separation"] for row in pairs
            if isinstance(row["fused_boundary_separation"], (int, float))
        ]
        summary["pair_metrics_by_k"][str(initial_k)] = {
            "pair_count": len(pairs),
            "verifier_mentioned_pair_count": sum(
                row["pair_mentioned_in_verifier_report"] for row in pairs
            ),
            "verifier_unmentioned_pair_count": sum(
                not row["pair_mentioned_in_verifier_report"] for row in pairs
            ),
            "fused_pair_silhouette": {
                "min": min(fused_silhouettes) if fused_silhouettes else None,
                "median": median(fused_silhouettes) if fused_silhouettes else None,
                "max": max(fused_silhouettes) if fused_silhouettes else None,
            },
            "fused_boundary_separation": {
                "min": min(fused_boundaries) if fused_boundaries else None,
                "median": median(fused_boundaries) if fused_boundaries else None,
                "max": max(fused_boundaries) if fused_boundaries else None,
            },
        }
    (output_root / "structural_action_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "output_kirc_v12" / "02_multi_k_accepted_core_stability_v10")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--initial-k", type=int, action="append")
    parser.add_argument("--repeat", type=int, action="append")
    args = parser.parse_args()
    summary = audit_experiment(
        args.experiment_root,
        args.initial_k or list(INITIAL_KS),
        args.repeat or [1],
        args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
