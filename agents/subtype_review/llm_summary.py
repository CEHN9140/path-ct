from __future__ import annotations

from typing import Any, Mapping


def summarize_evidence(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for item in evidence.get("results", []) or []:
        summary.append({
            "dimension": str(item.get("dimension", "")),
            "scope": str(item.get("scope", "")),
            "target_ids": list(item.get("target_ids", []) or []),
            "proposal_id": item.get("proposal_id"),
            "status": str(item.get("status", "")),
            "tool_results": [
                {
                    "tool_name": str(child.get("tool_name", "")),
                    "status": str(child.get("status", "")),
                    "metric_refs": list(child.get("metric_refs", []) or []),
                    "missing_reason": child.get("missing_reason"),
                }
                for child in item.get("results", []) or []
            ],
        })
    return summary


def summarize_audit(audit: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [
            {
                "target_ids": list(row.get("target_ids", []) or []),
                "dimension": str(row.get("dimension", "")),
                "scope": str(row.get("scope", "")),
                "proposal_id": row.get("proposal_id"),
                "status": row.get("status"),
                "summary": str(row.get("summary", ""))[:240],
                "reason": str(row.get("reason", ""))[:240],
                "metric_refs": list(row.get("metric_refs", []) or []),
            }
            for row in audit.get(key, []) or []
        ]
        for key in ("findings", "gaps")
    }
