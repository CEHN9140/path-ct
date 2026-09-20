from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.tool_utils import to_jsonable

REQUIRED_VIEWS = ("CT", "WSI", "RNA_Seq", "WXS")


def missing_view_reasons(case: Mapping[str, Any]) -> list[str]:
    missing = []
    for modality in REQUIRED_VIEWS:
        records = list(case.get(modality) or [])
        paths = [str(dict(record).get("File Path", "") or "").strip() for record in records]
        if not any(path and Path(path).exists() for path in paths):
            name = "rna" if modality == "RNA_Seq" else modality.lower()
            missing.append(f"missing_{name}")
    return missing


def build_patient_state(case: Mapping[str, Any], *, overall: str) -> dict[str, Any]:
    source = {
        key: value
        for key, value in dict(case or {}).items()
        if key in {*REQUIRED_VIEWS, "Clinical", "Case_ID", "case_id"}
    }
    case_id = str(source.get("Case_ID") or source.get("case_id") or "unknown_case")
    return {
        "case_id": case_id,
        "inventory": to_jsonable({**source, "Case_ID": case_id}),
        "qc": "success" if str(overall).lower() == "success" else "fail",
        "ct_evidence": {},
        "wsi_evidence": {},
        "text_evidence": {},
        "omics_evidence": {},
        "candidate_cluster_ids": [],
    }


def inventory_case(case_payload: Mapping[str, Any]) -> dict[str, Any]:
    case_payload = dict(case_payload)
    case_id = str(case_payload.get("Case_ID", "") or "unknown_case")
    counts = {name: len(list(case_payload.get(name) or [])) for name in REQUIRED_VIEWS}
    available = [f"{modality}={count}" for modality, count in counts.items() if count]
    missing = [modality for modality, count in counts.items() if not count]
    print(
        f"[inventory] {case_id}: available ({', '.join(available) if available else 'none'}); missing ({', '.join(missing) if missing else 'none'}).",
        flush=True,
    )
    missing_view_reason = missing_view_reasons(case_payload)
    state = build_patient_state(
        {**case_payload, "Case_ID": case_id},
        overall="fail" if missing_view_reason else "success",
    )
    state["missing_view_reason"] = missing_view_reason
    return state
