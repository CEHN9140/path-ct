from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def summarize_evidence(evidence: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": row.get("tool_name", ""),
            "dimension": row.get("dimension", ""),
            "scope": row.get("scope", ""),
            "target_ids": list(row.get("target_ids", []) or []),
            "status": row.get("status", ""),
            "warnings": list(row.get("warnings", []) or []),
            "errors": list(row.get("errors", []) or []),
            "missing_reason": row.get("missing_reason", ""),
        }
        for row in evidence
    ]


def summarize_reports(reports: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dimension": row.get("dimension", ""),
            "scope": row.get("scope", ""),
            "target_ids": list(row.get("target_ids", []) or []),
            "observations": list(row.get("observations", []) or []),
            "statistical_interpretation": row.get("statistical_interpretation", ""),
            "medical_interpretation": row.get("medical_interpretation", ""),
            "limitations": list(row.get("limitations", []) or []),
            "tool_refs": list(row.get("tool_refs", []) or []),
        }
        for row in reports
    ]
