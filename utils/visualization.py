from __future__ import annotations

from pathlib import Path


def configure_matplotlib(cache_dir: str | Path = "tools/.cache/matplotlib") -> None:
    import os

    path = Path(cache_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(path)
