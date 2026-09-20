from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.tool_utils import to_jsonable


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
