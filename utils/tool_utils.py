from __future__ import annotations

import json
import logging
import math
import os
import re
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, list["JSONValue"], dict[str, "JSONValue"]]
ToolStatus = Literal["success", "failure"]


class ToolResult(TypedDict):
    tool_name: str
    status: ToolStatus
    metrics: dict[str, JSONValue]
    artifacts: dict[str, JSONValue]
    provenance: dict[str, JSONValue]
    errors: list[str]


@contextmanager
def quiet_tool_logs():
    previous_disable = logging.root.manager.disable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                logging.disable(logging.CRITICAL)
                try:
                    yield
                finally:
                    logging.disable(previous_disable)


def to_jsonable(value: Any) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return repr(value)


def safe_identifier(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return safe or "artifact"


def compact_metrics(metrics: dict[str, Any]) -> dict[str, JSONValue]:
    compacted: dict[str, JSONValue] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            compacted[str(key)] = value
            continue
        if isinstance(value, float) and math.isfinite(value):
            compacted[str(key)] = value
    return compacted


def compact_artifacts(artifacts: dict[str, Any]) -> dict[str, JSONValue]:
    compacted: dict[str, JSONValue] = {}
    for key, value in artifacts.items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and value.strip():
            compacted[str(key)] = value
    return compacted


def compact_provenance(provenance: dict[str, Any]) -> dict[str, JSONValue]:
    compacted: dict[str, JSONValue] = {}
    for key, value in provenance.items():
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and not value.strip():
                continue
            compacted[str(key)] = to_jsonable(value)
    return compacted


def compact_errors(errors: list[str] | None) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for item in errors or []:
        message = str(item).strip()
        if not message or message in seen:
            continue
        seen.add(message)
        compacted.append(message)
    return compacted


def save_snapshot(
    output_root: str, namespace: str, identifier: str, payload: Any
) -> str:
    namespace_dir = Path(output_root) / namespace
    namespace_dir.mkdir(parents=True, exist_ok=True)
    output_path = namespace_dir / f"{safe_identifier(identifier)}.json"
    output_path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def make_tool_result(
    *,
    output_root: str,
    tool_name: str,
    status: ToolStatus,
    identifier: str,
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    provenance: dict[str, Any],
    errors: list[str] | None = None,
    payload: Any | None = None,
) -> ToolResult:
    normalized_status: ToolStatus = (
        "success" if str(status).strip().lower() == "success" else "failure"
    )
    result: ToolResult = {
        "tool_name": tool_name,
        "status": normalized_status,
        "metrics": compact_metrics(metrics),
        "artifacts": compact_artifacts(artifacts),
        "provenance": compact_provenance(provenance),
        "errors": compact_errors(errors),
    }
    saved_output_path = str(
        Path(output_root) / tool_name / f"{safe_identifier(identifier)}.json"
    )
    result["artifacts"]["saved_output_path"] = saved_output_path
    snapshot_payload = {
        "tool_result": result,
        "payload": to_jsonable(payload if payload is not None else artifacts),
    }
    save_snapshot(output_root, tool_name, identifier, snapshot_payload)
    return result
