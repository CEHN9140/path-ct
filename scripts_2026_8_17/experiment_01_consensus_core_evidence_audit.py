#!/usr/bin/env python3
"""Map consensus cores to historical five-dimension Agent audits.

This is a read-only diagnostic. It does not invoke an Agent, rerun tools, or
change the main pipeline. Evidence is transferred only when every patient in
the consensus target is contained in a historical source set.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DIMS = (
    "biological_support",
    "cross_modal_consistency",
    "confounder_exclusion",
    "known_label_echo",
    "structural_adequacy",
)
STATUSES = ("supporting", "conflicting", "mixed", "inconclusive", "unavailable")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(input_root: Path):
    runs = []
    for report_path in sorted(input_root.glob("arms/K*/repeat_*/subtype_review/final_review_summary.json")):
        k = int(report_path.parent.parent.parent.name[1:])
        repeat = int(report_path.parent.parent.name.split("_")[-1])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audit_path = report_path.parent / "evidence_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        runs.append({
            "label": f"K{k}/repeat_{repeat:02d}",
            "k": k,
            "repeat": repeat,
            "sets": report.get("partition_sets", []),
            "findings": audit.get("findings", []),
            "status": report.get("status"),
        })
    if not runs:
        raise FileNotFoundError(f"No Agent reports under {input_root / 'arms'}")
    return runs


def load_targets(consensus_root: Path, consensus_k: int, core_threshold: float):
    partition = json.loads(
        (consensus_root / f"consensus_partition_k{consensus_k}.json").read_text(encoding="utf-8")
    )
    targets = [
        {
            "target_id": item["cluster_id"],
            "target_type": "consensus_cluster",
            "member_ids": item["member_ids"],
        }
        for item in partition["sets"]
        if len(item["member_ids"]) > 1
    ]
    core_path = consensus_root / "stable_cores.csv"
    if core_path.exists():
        grouped = {}
        with core_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["consensus_k"]) == consensus_k and float(row["threshold"]) == core_threshold:
                    grouped.setdefault(row["cluster_id"], []).append(row["patient_id"])
        for cluster_id, members in sorted(grouped.items()):
            if len(members) > 1:
                targets.append({
                    "target_id": f"{cluster_id}_CORE_T{core_threshold:g}",
                    "target_type": "stable_core",
                    "parent_id": cluster_id,
                    "member_ids": sorted(members),
                })
    return targets


def best_source(target: dict, run: dict):
    target_members = set(target["member_ids"])
    candidates = []
    for source in run["sets"]:
        source_members = set(source["member_ids"])
        intersection = len(target_members & source_members)
        if not intersection:
            continue
        candidates.append({
            "source_set": source["set_id"],
            "source_status": source.get("status", "unknown"),
            "intersection": intersection,
            "target_size": len(target_members),
            "source_size": len(source_members),
            "target_coverage": intersection / len(target_members),
            "source_coverage": intersection / len(source_members),
            "jaccard": intersection / len(target_members | source_members),
        })
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["target_coverage"], row["jaccard"], row["source_coverage"]))


def findings_for(run: dict, source_set: str, dimension: str):
    return [
        finding
        for finding in run["findings"]
        if finding.get("dimension") == dimension and source_set in finding.get("target_ids", [])
    ]


def aggregate_dimension(rows: list[dict], dimension: str, run_count: int):
    status_counts = Counter(row["status"] for row in rows)
    if not rows:
        status = "unavailable"
    elif status_counts["conflicting"] and status_counts["supporting"]:
        status = "mixed"
    elif status_counts["conflicting"]:
        status = "conflicting"
    elif status_counts["supporting"] and status_counts["inconclusive"]:
        status = "mixed"
    elif status_counts["supporting"]:
        status = "supporting"
    else:
        status = "inconclusive"
    return {
        "dimension": dimension,
        "status": status,
        "mapped_runs": len(rows),
        "supporting_runs": status_counts["supporting"],
        "conflicting_runs": status_counts["conflicting"],
        "mixed_runs": status_counts["mixed"],
        "inconclusive_runs": status_counts["inconclusive"],
        "unavailable_runs": run_count - len(rows),
        "metric_refs": sorted({ref for row in rows for ref in row["metric_refs"]}),
        "summaries": [row["summary"] for row in rows if row["summary"]],
    }


def audit_target(target: dict, runs: list[dict]):
    mappings = []
    dimension_rows = {dimension: [] for dimension in DIMS}
    for run in runs:
        source = best_source(target, run)
        if source is None:
            mappings.append({"run": run["label"], "target_id": target["target_id"], "transferable": False})
            continue
        transferable = source["target_coverage"] == 1.0
        mappings.append({
            "run": run["label"],
            "target_id": target["target_id"],
            "source_set": source["source_set"],
            "source_status": source["source_status"],
            "target_coverage": round(source["target_coverage"], 6),
            "source_coverage": round(source["source_coverage"], 6),
            "jaccard": round(source["jaccard"], 6),
            "transferable": transferable,
        })
        if not transferable:
            continue
        for dimension in DIMS:
            findings = findings_for(run, source["source_set"], dimension)
            for finding in findings:
                dimension_rows[dimension].append({
                    "status": finding.get("status", "unavailable"),
                    "metric_refs": finding.get("metric_refs", []),
                    "summary": finding.get("summary", ""),
                })
    dimensions = [aggregate_dimension(dimension_rows[dimension], dimension, len(runs)) for dimension in DIMS]
    transferable_runs = sum(row.get("transferable", False) for row in mappings)
    return {
        "target_id": target["target_id"],
        "target_type": target["target_type"],
        "parent_id": target.get("parent_id"),
        "member_count": len(target["member_ids"]),
        "transferable_runs": transferable_runs,
        "target_member_ids": target["member_ids"],
        "dimensions": dimensions,
        "mappings": mappings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--consensus-root", type=Path,
        default=Path("output_kirc_v8/experiment_18_consensus_core_analysis"),
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("output_kirc_v8/experiment_16_current_agent_multi_initial_k_convergence_v2"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("output_kirc_v9/experiment_01_consensus_core_evidence_audit"),
    )
    parser.add_argument("--consensus-k", type=int, default=6)
    parser.add_argument("--core-threshold", type=float, default=0.9)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.input_root)
    targets = load_targets(args.consensus_root, args.consensus_k, args.core_threshold)
    audits = [audit_target(target, runs) for target in targets]

    mapping_rows = []
    dimension_rows = []
    for audit in audits:
        for mapping in audit["mappings"]:
            mapping_rows.append(mapping)
        for dimension in audit["dimensions"]:
            dimension_rows.append({
                "target_id": audit["target_id"],
                "target_type": audit["target_type"],
                "member_count": audit["member_count"],
                "dimension": dimension["dimension"],
                "status": dimension["status"],
                "mapped_runs": dimension["mapped_runs"],
                "supporting_runs": dimension["supporting_runs"],
                "conflicting_runs": dimension["conflicting_runs"],
                "mixed_runs": dimension["mixed_runs"],
                "inconclusive_runs": dimension["inconclusive_runs"],
                "unavailable_runs": dimension["unavailable_runs"],
                "metric_refs": " | ".join(dimension["metric_refs"]),
                "summaries": " || ".join(dimension["summaries"]),
            })
    fields = ["target_id", "target_type", "member_count", "dimension", "status", "mapped_runs", "supporting_runs", "conflicting_runs", "mixed_runs", "inconclusive_runs", "unavailable_runs", "metric_refs", "summaries"]
    write_csv(args.output_root / "consensus_core_evidence_summary.csv", dimension_rows, fields)
    mapping_fields = ["run", "target_id", "source_set", "source_status", "target_coverage", "source_coverage", "jaccard", "transferable"]
    write_csv(args.output_root / "consensus_core_source_mapping.csv", mapping_rows, mapping_fields)

    result = {
        "experiment": "consensus_core_evidence_audit_v1",
        "scope": "K-balanced consensus K=6 clusters and thresholded stable cores; historical evidence mapping only",
        "run_count": len(runs),
        "target_count": len(targets),
        "consensus_k": args.consensus_k,
        "core_threshold": args.core_threshold,
        "transfer_rule": "transfer evidence only when every target patient is contained in the historical source set",
        "targets": audits,
    }
    (args.output_root / "consensus_core_evidence_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "target_count": len(targets),
        "output": str(args.output_root),
        "summary": str(args.output_root / "consensus_core_evidence_summary.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
