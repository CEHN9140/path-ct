from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.cache_utils import artifacts_valid, file_identity, hash_payload


def collect_case_file_paths(
    cases: Sequence[Mapping[str, Any]], modality_key: str
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for case in cases:
        case_id = str(case.get("Case_ID", "") or "").strip()
        if not case_id:
            continue
        records = list(case.get(modality_key, []) or [])
        file_path = ""
        for record in records:
            file_path = str(record.get("File Path", "") or "").strip()
            if file_path:
                break
        if file_path:
            rows.append((case_id, file_path))
    return sorted(rows)


def build_cohort_signature(
    case_file_rows: Sequence[tuple[str, str]],
    *,
    extra: Mapping[str, Any],
) -> str:
    payload = {
        "cache_version": 1,
        "case_file_rows": [
            {
                "case_id": case_id,
                "input": file_identity(file_path) if Path(file_path).exists() else {"path": file_path},
            }
            for case_id, file_path in case_file_rows
        ],
        "extra": dict(extra),
    }
    return hash_payload(payload)


def load_manifest_if_valid(
    manifest_path: Path,
    *,
    signature: str,
    required_paths: Sequence[Path],
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if str(manifest.get("signature", "")) != signature:
        return None
    if not artifacts_valid(required_paths):
        return None
    return manifest
