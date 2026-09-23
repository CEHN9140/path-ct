from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def inspect_review_run(
    run_root: str | Path,
    *,
    expected_input_signature: str | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    metadata_path = root / "run_metadata.json"
    summary_path = root / "final_review_summary.json"
    sets_path = root / "final_subtype_sets.json"
    result: dict[str, Any] = {
        "status": "missing",
        "reason": "missing_run_metadata",
        "input_signature": None,
        "candidate_signature": None,
        "sets_path": str(sets_path),
        "summary": None,
        "error_type": None,
        "error_message": None,
    }
    if not metadata_path.is_file():
        return result
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**result, "status": "invalid", "reason": "invalid_run_metadata_json"}
    if not isinstance(metadata, dict):
        return {**result, "status": "invalid", "reason": "invalid_run_metadata_json"}
    result.update({
        "input_signature": metadata.get("input_signature"),
        "candidate_signature": metadata.get("candidate_signature"),
        "error_type": metadata.get("error_type"),
        "error_message": metadata.get("error_message"),
    })
    if expected_input_signature is not None and result["input_signature"] != expected_input_signature:
        return {**result, "status": "stale", "reason": "input_signature_mismatch"}
    if metadata.get("status") == "failed":
        return {**result, "status": "failed", "reason": "run_failed"}
    if metadata.get("status") != "complete":
        return {**result, "status": "incomplete", "reason": f"run_status_{metadata.get('status', 'unknown')}"}
    if not summary_path.is_file():
        return {**result, "status": "incomplete", "reason": "missing_final_review_summary"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**result, "status": "invalid", "reason": "invalid_final_review_summary_json"}
    if not isinstance(summary, dict):
        return {**result, "status": "invalid", "reason": "invalid_final_review_summary_json"}
    result["summary"] = summary
    if summary.get("raw_control_status") != "complete" or summary.get("status") != "review_complete":
        return {**result, "status": "incomplete", "reason": "review_not_complete"}
    if not sets_path.is_file():
        return {**result, "status": "incomplete", "reason": "missing_final_subtype_sets"}
    try:
        sets = json.loads(sets_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**result, "status": "invalid", "reason": "invalid_final_subtype_sets_json"}
    if not isinstance(sets, list):
        return {**result, "status": "invalid", "reason": "invalid_final_subtype_sets"}
    return {**result, "status": "complete", "reason": "complete"}
