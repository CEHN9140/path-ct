from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from utils.tool_utils import to_jsonable


def file_identity(path: str) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def semantic_config(value: Any, runtime_keys: Iterable[str] = ()) -> Any:
    runtime_keys = {str(key).lower() for key in runtime_keys}
    if isinstance(value, Mapping):
        return {
            str(key): semantic_config(item, runtime_keys)
            for key, item in value.items()
            if str(key).lower() not in runtime_keys
        }
    if isinstance(value, (list, tuple)):
        return [semantic_config(item, runtime_keys) for item in value]
    return to_jsonable(value)


def hash_payload(payload: Mapping[str, Any]) -> str:
    content = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def artifacts_valid(paths: Iterable[str | Path]) -> bool:
    return all(Path(path).expanduser().exists() for path in paths if str(path))
