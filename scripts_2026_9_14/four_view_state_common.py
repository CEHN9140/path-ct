"""Shared, read-only inputs for the four-view state analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = ROOT / "output_kirc_v14/11_four_view_no_cnv/inputs/four_view_no_cnv"
DEFAULT_MEMBERSHIP = ROOT / "output_kirc_v14/12_four_view_core_to_macro_state_audit/final_macro_state_membership.csv"


def load_membership(path=DEFAULT_MEMBERSHIP):
    frame = pd.read_csv(path, dtype=str)
    required = {"case_id", "core_id", "state_id"}
    if set(frame.columns) != required or frame.case_id.duplicated().any():
        raise ValueError("final_macro_state_membership.csv must contain one unique row per case_id")
    if frame.empty:
        raise ValueError("Macro-state membership is empty")
    return frame.sort_values(["state_id", "core_id", "case_id"]).reset_index(drop=True)


def load_states(input_root=DEFAULT_INPUT):
    path = Path(input_root) / "storage/patient_states/patient_states.jsonl"
    states = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                if state.get("qc") == "success":
                    states[str(state["case_id"])] = state
    return states


def load_table(path):
    frame = pd.read_csv(path)
    if "case_id" not in frame:
        raise ValueError(f"Missing case_id column: {path}")
    return frame.set_index("case_id")


def groups(membership):
    return {name: sorted(values.case_id) for name, values in membership.groupby("state_id")}


def write_manifest(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
