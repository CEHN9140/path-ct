from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.tool_utils import to_jsonable

REQUIRED_FIVE_VIEW = ("CT", "WSI", "RNA_Seq", "WXS", "CNV")


def missing_five_view_reasons(case: Mapping[str, Any]) -> list[str]:
    missing = []
    for modality in REQUIRED_FIVE_VIEW:
        records = list(case.get(modality) or [])
        paths = [str(dict(record).get("File Path", "") or "").strip() for record in records]
        if not any(path and Path(path).exists() for path in paths):
            name = "rna" if modality == "RNA_Seq" else modality.lower()
            missing.append(f"missing_{name}")
    return missing


def build_patient_state(case: Mapping[str, Any], *, overall: str) -> dict[str, Any]:
    source = dict(case or {})
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
    counts = {
        "CT": len(list(case_payload.get("CT") or [])),
        "WSI": len(list(case_payload.get("WSI") or [])),
        "RNA_Seq": len(list(case_payload.get("RNA_Seq") or [])),
        "WXS": len(list(case_payload.get("WXS") or [])),
        "CNV": len(list(case_payload.get("CNV") or [])),
        "Clinical": 1 if dict(case_payload.get("Clinical") or {}) else 0,
    }
    available = [f"{modality}={count}" for modality, count in counts.items() if count]
    missing = [modality for modality, count in counts.items() if not count]
    print(
        f"[inventory] {case_id}: available ({', '.join(available) if available else 'none'}); missing ({', '.join(missing) if missing else 'none'}).",
        flush=True,
    )
    missing_view_reason = missing_five_view_reasons(case_payload)
    state = build_patient_state(
        {**case_payload, "Case_ID": case_id},
        overall="fail" if missing_view_reason else "success",
    )
    if missing_view_reason:
        state["missing_view_reason"] = missing_view_reason
    return state
