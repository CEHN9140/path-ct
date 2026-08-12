from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import warnings
from multiprocessing import get_context
from collections.abc import Mapping
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


def semantic_execution_config(value: Any, key: str = "") -> JSONValue:
    runtime_keys = {"devices", "workers_per_gpu"}
    if isinstance(value, Mapping):
        result = {
            str(item_key): semantic_execution_config(item, str(item_key))
            for item_key, item in value.items()
            if str(item_key).lower() not in runtime_keys
        }
        if "devices" in value and "device" not in result:
            devices = [str(item).strip().lower() for item in value["devices"]]
            result["device"] = (
                "cuda"
                if devices and all(device.startswith(("cuda:", "gpu:")) for device in devices)
                else semantic_execution_config(devices[0], "device")
            )
        return result
    if isinstance(value, (list, tuple)):
        item_key = "device" if key.lower() == "devices" else ""
        return [semantic_execution_config(item, item_key) for item in value]
    if key.lower() == "device":
        device = str(value).strip().lower()
        if (
            device.isdigit()
            or device == "gpu"
            or re.fullmatch(r"(?:cuda|gpu):\d+", device)
        ):
            return "cuda"
    return to_jsonable(value)


def split_requests(requests: list[Any], worker_count: int) -> list[list[Any]]:
    count = min(max(1, int(worker_count)), len(requests))
    return [requests[index::count] for index in range(count)] if requests else []


def split_device_requests(
    requests: list[Any], devices: list[str], *, workers_per_device: int
) -> tuple[list[dict[str, Any]], str | None]:
    devices = [str(device).strip().lower() for device in devices]
    if not devices:
        raise ValueError("At least one execution device is required")
    cuda_devices = all(
        device.startswith("cuda:") and device.split(":", 1)[1].isdigit()
        for device in devices
    )
    if not cuda_devices and devices != ["cpu"]:
        raise ValueError("Devices must be CUDA devices or a single CPU device")
    groups = split_requests(requests, len(devices) * int(workers_per_device))
    assignments = []
    for position, group in enumerate(groups):
        device_index = position % len(devices)
        assignments.append(
            {
                "requests": group,
                "device": f"cuda:{device_index}" if cuda_devices else "cpu",
                "physical_device": devices[device_index],
                "position": position,
            }
        )
    visible = (
        ",".join(device.split(":", 1)[1] for device in devices)
        if cuda_devices
        else None
    )
    return assignments, visible


def run_json_workers(
    command: list[str],
    payloads: list[Mapping[str, Any]],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> list[int]:
    processes = []
    for payload in payloads:
        process = subprocess.Popen(
            command,
            env=dict(env) if env is not None else None,
            cwd=cwd,
            stdin=subprocess.PIPE,
            text=True,
        )
        if process.stdin is None:
            raise RuntimeError("Worker stdin pipe was not created.")
        process.stdin.write(json.dumps(payload, ensure_ascii=False))
        process.stdin.close()
        processes.append(process)
    return [int(process.wait()) for process in processes]


def run_command(
    command: list[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=dict(env) if env is not None else None,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def run_function_workers(function, payloads: list[Mapping[str, Any]]) -> list[int]:
    processes = [
        get_context("spawn").Process(target=function, args=(payload,))
        for payload in payloads
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    return [int(process.exitcode or 0) for process in processes]


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
