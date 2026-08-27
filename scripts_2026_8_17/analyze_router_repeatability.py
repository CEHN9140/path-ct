#!/usr/bin/env python3
"""Audit Router repeatability from saved Multi-K Review runs only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INITIAL_KS = tuple(range(2, 9))
REPEATS = (1, 2, 3)
MODALITIES = ("ct", "wsi", "rna", "genomic")


def without_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: without_paths(item)
            for key, item in value.items()
            if key not in {"artifact_paths", "artifact_root", "output_root"}
        }
    if isinstance(value, list):
        return [without_paths(item) for item in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(without_paths(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_by_set(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(target): str(action.get("action", ""))
        for action in plan.get("actions", []) or []
        for target in action.get("target_ids", []) or []
    }


def tool_metrics(entry: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    return next(
        (
            dict(item.get("metrics", {}) or {})
            for item in entry.get("round_evidence", []) or []
            if item.get("tool_name") == tool_name
        ),
        {},
    )


def evidence_summary(entry: Mapping[str, Any], set_id: str) -> dict[str, Any]:
    pathway = tool_metrics(entry, "pathway_enrichment").get("per_set_rna_pathway_enrichment", {}).get(set_id, {})
    mutation = tool_metrics(entry, "mutation_enrichment").get("per_set_wxs_feature_enrichment", {}).get(set_id, {})
    cnv = tool_metrics(entry, "cnv_characterization").get("per_comparison_cnv", {}).get(f"{set_id}_vs_rest", {})
    cross = tool_metrics(entry, "multimodal_consistency_check").get("cross_modal_consistency", {})
    support = cross.get("per_set", {}).get(set_id, {}).get("modality_support", {})
    structural = cross.get("per_set", {}).get(set_id, {}).get("internal_structure", {})
    probe = structural.get("fused_binary_probe", {}) or {}
    resampling = probe.get("resampling", {}) or {}
    confound = tool_metrics(entry, "confound_test").get("sets", {}).get(set_id, [])
    return {
        "rna_significant_count": pathway.get("summary", {}).get("significant_q05"),
        "wxs_significant_count": mutation.get("summary", {}).get("significant_q05"),
        "cnv_significant_count": sum(
            part.get("summary", {}).get("significant_q05", 0) or 0
            for part in (cnv.get("continuous", {}), cnv.get("gain_loss", {}))
        ),
        **{
            f"{name}_silhouette": (support.get(name, {}) or {}).get("median_silhouette")
            for name in MODALITIES
        },
        "fused_internal_silhouette": probe.get("median_silhouette"),
        "resampling_ari": resampling.get("median_resample_ari"),
        "pac": resampling.get("pac"),
        "technical_confound_q05_count": sum(
            isinstance(item.get("q_value"), (int, float)) and item["q_value"] < 0.05
            for item in confound
        ),
    }


def load_run(run_root: Path) -> dict[str, Any]:
    metadata_path = run_root / "run_metadata.json"
    summary_path = run_root / "final_review_summary.json"
    history_path = run_root / "review_history.json"
    if not history_path.exists():
        return {"status": "missing"}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return {"status": "failed", "error": "empty review_history.json"}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    entry = history[-1]
    return {
        "status": "complete" if summary.get("status") == "review_complete" else "failed",
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {},
        "entry": entry,
        "actions": action_by_set(entry.get("router_plan", {}) or {}),
        "partition_set_ids": {
            str(item.get("set_id") or item.get("cluster_id") or "")
            for item in entry.get("partition", {}).get("sets", []) or []
        },
        "raw_evidence": stable_json(entry.get("round_evidence", [])),
    }


def audit_router_repeatability(
    experiment_root: Path,
    initial_ks: tuple[int, ...] = INITIAL_KS,
    repeats: tuple[int, ...] = REPEATS,
    output_root: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root or experiment_root / "router_repeatability"
    output_root.mkdir(parents=True, exist_ok=True)
    runs = {
        (initial_k, repeat): load_run(experiment_root / f"run{repeat}" / f"K{initial_k}")
        for initial_k in initial_ks
        for repeat in repeats
    }
    rows, discordant_rows = [], []
    for initial_k in initial_ks:
        set_ids = sorted({set_id for (k, _), run in runs.items() if k == initial_k for set_id in run.get("partition_set_ids", set())})
        for set_id in set_ids:
            records = [runs[(initial_k, repeat)] for repeat in repeats]
            actions = [record.get("actions", {}).get(set_id, "") for record in records]
            available_actions = [action for action in actions if action]
            metadata = [record.get("metadata", {}) for record in records]
            evidence = [record.get("raw_evidence") for record in records if record.get("raw_evidence")]
            row = {
                "initial_k": initial_k,
                "set_id": set_id,
                **{
                    f"repeat{repeat}_action": runs[(initial_k, repeat)].get("actions", {}).get(set_id, "")
                    for repeat in repeats
                },
                **{f"{action}_count": actions.count(action) for action in ("accept", "drop", "split", "merge")},
                "decision_agreement_fraction": (
                    max(Counter(available_actions).values()) / len(available_actions)
                    if available_actions else None
                ),
                "source_sha256_equal": (
                    len({item.get("source_sha256") for item in metadata}) == 1
                    and all(item.get("source_sha256") for item in metadata)
                ) if all(record.get("status") == "complete" for record in records) else None,
                "input_data_signature_equal": (
                    len({item.get("input_data_signature") for item in metadata if item.get("input_data_signature")}) == 1
                ) if all(record.get("status") == "complete" for record in records) else None,
                "review_signature_equal": (
                    len({item.get("review_signature") for item in metadata if item.get("review_signature")}) == 1
                ) if all(record.get("status") == "complete" for record in records) else None,
                "raw_evidence_equal": len(set(evidence)) == 1 if len(evidence) == len(repeats) else None,
                "decision_discordant": len(set(available_actions)) > 1 if len(available_actions) > 1 else None,
            }
            rows.append(row)
            if row["decision_discordant"]:
                evidence_row = {"initial_k": initial_k, "set_id": set_id, **row}
                for repeat in repeats:
                    metrics = evidence_summary(runs[(initial_k, repeat)].get("entry", {}), set_id)
                    evidence_row.update({f"repeat{repeat}_{key}": value for key, value in metrics.items()})
                discordant_rows.append(evidence_row)

    fields = list(rows[0]) if rows else []
    with (output_root / "router_repeatability.csv").open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    evidence_fields = list(discordant_rows[0]) if discordant_rows else []
    with (output_root / "router_discordant_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        if evidence_fields:
            writer = csv.DictWriter(handle, fieldnames=evidence_fields)
            writer.writeheader()
            writer.writerows(discordant_rows)
    status_counts = Counter(record["status"] for record in runs.values())
    summary = {
        "experiment": "router_repeatability",
        "frozen_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "planned_run_count": len(runs),
        "complete_run_count": status_counts["complete"],
        "missing_run_count": status_counts["missing"],
        "failed_run_count": status_counts["failed"],
        "set_row_count": len(rows),
        "discordant_set_count": len(discordant_rows),
        "raw_evidence_equal_count": sum(row["raw_evidence_equal"] is True for row in rows),
        "rows": rows,
    }
    (output_root / "router_repeatability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "output_kirc_v10" / "experiment_multi_k_accepted_core_stability")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_router_repeatability(args.experiment_root, output_root=args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
