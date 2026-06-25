from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.tool_utils import safe_identifier, to_jsonable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> str:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(file_path)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(dict(row)), ensure_ascii=False))
            handle.write("\n")
    return str(file_path)


def write_text(path: str | Path, content: str) -> str:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def build_output_path(output_root: str, namespace: str, identifier: str, suffix: str) -> Path:
    return ensure_dir(Path(output_root) / "storage" / namespace) / f"{safe_identifier(identifier)}{suffix}"
