from __future__ import annotations

from pathlib import Path
from typing import Mapping

from utils.io import write_jsonl


def save_patient_states(
    output_root: str, patient_states: list[Mapping[str, object]]
) -> dict[str, str]:
    patient_states = [dict(item) for item in patient_states]
    jsonl_path = (
        Path(output_root) / "storage" / "patient_states" / "patient_states.jsonl"
    )
    write_jsonl(jsonl_path, patient_states)
    return {"patient_states_jsonl": str(jsonl_path)}
