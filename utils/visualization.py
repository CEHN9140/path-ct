from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from utils.tool_utils import safe_identifier


def configure_matplotlib(cache_dir: str | Path = "tools/.cache/matplotlib") -> None:
    import os

    path = Path(cache_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(path)


def rewrite_paths(payload: Any, source_root: Path, target_root: Path) -> Any:
    if isinstance(payload, Mapping):
        return {
            str(key): rewrite_paths(value, source_root, target_root)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [rewrite_paths(value, source_root, target_root) for value in payload]
    if isinstance(payload, str):
        return payload.replace(
            str(source_root / "subtype_review"),
            str(target_root / "subtype_review"),
        )
    return payload


def copy_figures(source_dir: Path, target_dir: Path) -> dict[str, str]:
    if not source_dir.exists():
        return {}
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source in sorted(source_dir.iterdir()):
        if source.is_file() and source.suffix.lower() in {".png", ".pdf", ".svg"}:
            target = target_dir / source.name
            shutil.copy2(source, target)
            copied[source.stem] = str(target)
    return copied


def build_review_figures(
    output_root: str | Path,
    candidate_sets: list[Mapping[str, Any]],
    source_output_root: str | Path = "",
) -> dict[str, Any]:
    target_root = Path(output_root)
    source_root = Path(source_output_root) if source_output_root else target_root
    source_review = source_root / "subtype_review"
    target_review = target_root / "subtype_review"
    global_figures = copy_figures(
        source_review / "global" / "figures",
        target_review / "global" / "figures",
    )
    set_figures = {}
    for candidate in candidate_sets:
        set_id = str(candidate.get("cluster_id", candidate.get("set_id", "")))
        if not set_id:
            continue
        copied = copy_figures(
            source_review / safe_identifier(set_id) / "figures",
            target_review / safe_identifier(set_id) / "figures",
        )
        report_path = source_review / safe_identifier(set_id) / "report.json"
        grouped = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            grouped = rewrite_paths(
                report.get("figures", {}), source_root, target_root
            )
        if copied or grouped:
            set_figures[set_id] = {
                "files": copied,
                "by_validation_dimension": grouped,
            }
    return {"global": global_figures, "sets": set_figures}
