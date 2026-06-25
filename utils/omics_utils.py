from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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
        "case_file_rows": [[case_id, file_path] for case_id, file_path in case_file_rows],
        "extra": dict(extra),
    }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


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
    for required_path in required_paths:
        if not required_path.exists():
            return None
    return manifest
