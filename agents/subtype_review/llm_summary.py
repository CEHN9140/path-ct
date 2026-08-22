from __future__ import annotations

from typing import Any, Mapping


def summarize_evidence(evidence: Mapping[str, Any] | list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = evidence if isinstance(evidence, list) else evidence.get("raw", evidence.get("results", []))
    return [
        {
            "dimension": str(item.get("dimension", "")),
            "scope": str(item.get("scope", "")),
            "target_ids": list(item.get("target_ids", []) or []),
            "subject_signature": str(item.get("subject_signature", "")),
            "status": str(item.get("status", "")),
            "tool_results": [
                {
                    "tool_name": str(child.get("tool_name", "")),
                    "status": str(child.get("status", "")),
                    "metric_refs": list(child.get("metric_refs", []) or []),
                    "warnings": list(child.get("warnings", []) or []),
                }
                for child in item.get("results", []) or []
            ],
        }
        for item in rows
    ]


def summarize_reports(reports: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dimension": str(row.get("dimension", "")),
            "scope": str(row.get("scope", "")),
            "target_ids": list(row.get("target_ids", []) or []),
            "observations": [
                {
                    "metric": str(item.get("metric", "")),
                    "finding": str(item.get("finding", ""))[:600],
                }
                for item in row.get("observations", []) or []
            ],
            "statistical_interpretation": str(row.get("statistical_interpretation", ""))[:1000],
            "medical_interpretation": str(row.get("medical_interpretation", ""))[:1000],
            "limitations": list(row.get("limitations", []) or []),
            "metric_refs": list(row.get("metric_refs", []) or []),
        }
        for row in reports
    ]
