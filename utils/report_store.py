from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.io import ensure_dir, write_json


def save_final_output(output_root: str, final_output: Mapping[str, object]) -> str:
    return write_json(Path(output_root) / "storage" / "reports" / "final_output.json", final_output)


def save_graph_pngs(output_root: str, graphs: Mapping[str, Any]) -> dict[str, str]:
    output_dir = ensure_dir(Path(output_root) / "graphs")
    saved_paths: dict[str, str] = {}
    for existing_path in output_dir.iterdir():
        if existing_path.is_file():
            existing_path.unlink()
    for name, app in graphs.items():
        if app is None:
            continue
        try:
            graph_obj = app.get_graph() if hasattr(app, "get_graph") else app
        except Exception:
            continue
        if graph_obj is None or not hasattr(graph_obj, "draw_mermaid_png"):
            continue
        try:
            png_bytes = graph_obj.draw_mermaid_png()
        except Exception:
            continue
        if not png_bytes:
            continue
        output_path = output_dir / f"{name}.png"
        output_path.write_bytes(png_bytes)
        saved_paths[name] = str(output_path)
    return saved_paths
