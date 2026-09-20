from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_yaml_file(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_candidate_proposer_config(config_dir: str | Path = "configs") -> dict[str, Any]:
    return load_yaml_file(Path(config_dir) / "candidate_proposer.yaml")
