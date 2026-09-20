from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.tool_utils import to_jsonable


TRACE_VERSION = 1


def append_runtime_trace(
    path: str | Path | None,
    *,
    node: str,
    event: str,
    round_id: int | None = None,
    payload: Mapping[str, Any] | None = None,
) -> None:
    if not path:
        return

    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "trace_version": TRACE_VERSION,
        "node": str(node),
        "event": str(event),
    }
    if round_id is not None:
        row["round"] = int(round_id)
    if payload:
        row.update(to_jsonable(dict(payload)))

    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def partition_snapshot(
    sets: list[Mapping[str, Any]],
    *,
    include_members: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for item in sets:
        members = sorted(map(str, item["member_ids"]))
        row = {"set_id": str(item["set_id"]), "member_n": len(members)}
        if include_members:
            row["member_ids"] = members
        rows.append(row)
    return sorted(rows, key=lambda row: row["set_id"])
