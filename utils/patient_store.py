from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from utils.tool_utils import to_jsonable


def save_patient_states(
    output_root: str, patient_states: list[Mapping[str, object]]
) -> dict[str, str]:
    patient_states = [dict(item) for item in patient_states]
    jsonl_path = (
        Path(output_root) / "storage" / "patient_states" / "patient_states.jsonl"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for state in patient_states:
            handle.write(json.dumps(to_jsonable(state), ensure_ascii=False) + "\n")
    return {"patient_states_jsonl": str(jsonl_path)}
