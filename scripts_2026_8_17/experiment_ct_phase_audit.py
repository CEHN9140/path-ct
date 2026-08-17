from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PHASE_PATTERNS = {
    "NC": (
        r"\bPRE[- ]?CONTRAST\b",
        r"\bUNENHANCED\b",
        r"\bNON[- ]?CONTRAST\b",
        r"\bNO CONTRAST\b",
        r"\bWITHOUT CONTRAST\b",
        r"(?<!W)\bW/O CONTRAST\b",
        r"(?<!W)\bWO CONTRAST\b",
        r"\bPRE LIVER\b",
        r"\bKIDNEYS PRE\b",
    ),
    "ART": (r"\bARTERIAL(?: PHASE)?\b", r"\bCORTICOMEDULLARY\b"),
    "NEPH": (r"\bNEPHRO(?:GRAPHIC)?\b", r"\bPARENCHYMAL\b"),
    "DEL": (
        r"\bDELAY(?:ED)?\b",
        r"\bEXCRET(?:ION|ORY)?\b",
        r"\b3 MIN(?:UTE)?\b",
        r"\bDELAY BLADDER\b",
        r"\bKIDNEYS & DELAY\b",
        r"\bUROGRAM\b",
    ),
}
CE_PATTERNS = (
    r"\bPOST[- ]?CONTRAST\b",
    r"\bWITH CONTRAST\b",
    r"\bW CONTRAST\b",
    r"\bVENOUS\b",
    r"\bSMART PREP\b",
)


def read_inventory(path: Path) -> dict[str, dict[str, Any]]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(series.get("File Path", "")): {
            "case_id": str(case.get("Case_ID", "")),
            "series_uid": str(series.get("Series UID", "")),
            "study_uid": str(series.get("Study UID", "")),
            "series_description": str(series.get("Series Description", "")),
            "study_description": str(series.get("Study Description", "")),
            "manufacturer": str(series.get("Manufacturer", "")),
        }
        for case in inventory
        for series in case.get("CT", [])
        if str(series.get("File Path", ""))
    }


def read_sidecars(paths: str) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    errors = []
    for raw_path in str(paths or "").split(";"):
        path = Path(raw_path)
        if not raw_path or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {type(exc).__name__}")
            continue
        for key in (
            "SeriesDescription",
            "StudyDescription",
            "ProtocolName",
            "ContrastBolusAgent",
            "ContrastBolusStartTime",
            "ContrastBolusVolume",
            "AcquisitionTime",
            "SeriesTime",
            "ContentTime",
            "SeriesNumber",
            "AcquisitionNumber",
        ):
            if str(payload.get(key, "") or "").strip():
                values.setdefault(key, payload[key])
    return values, errors


def classify_phase(
    inventory: dict[str, Any], sidecar: dict[str, Any]
) -> dict[str, Any]:
    text_fields = {
        "series_description": inventory.get("series_description", ""),
        "study_description": inventory.get("study_description", ""),
        "SeriesDescription": sidecar.get("SeriesDescription", ""),
        "StudyDescription": sidecar.get("StudyDescription", ""),
        "ProtocolName": sidecar.get("ProtocolName", ""),
    }
    evidence = [
        f"{name}={value}"
        for name, value in text_fields.items()
        if str(value or "").strip()
    ]
    series_text = " ".join(
        str(text_fields[name] or "")
        for name in ("series_description", "SeriesDescription")
    ).upper()
    protocol_text = str(text_fields["ProtocolName"] or "").upper()

    def find_matches(text: str) -> dict[str, list[str]]:
        matches = {
            phase: [pattern for pattern in patterns if re.search(pattern, text)]
            for phase, patterns in PHASE_PATTERNS.items()
        }
        return {phase: patterns for phase, patterns in matches.items() if patterns}

    series_matches = find_matches(series_text)
    protocol_matches = find_matches(protocol_text)
    protocol_multiphase = bool(
        re.search(r"\bNC\b", protocol_text)
        and re.search(r"\b(?:WC|WITH|CONTRAST|DELAY|ARTERIAL|NEPHRO)\b", protocol_text)
    )
    if protocol_multiphase:
        protocol_matches = {}
    matched = series_matches or (
        protocol_matches if len(protocol_matches) == 1 else {}
    )
    contrast_agent = str(sidecar.get("ContrastBolusAgent", "") or "").strip()
    if contrast_agent:
        evidence.append(f"ContrastBolusAgent={contrast_agent}")

    if len(series_matches) == 1:
        phase = next(iter(matched))
        return {
            "phase": phase,
            "broad_phase": "NC" if phase == "NC" else "CE",
            "confidence": 0.95,
            "phase_status": "high_confidence_text",
            "matched_patterns": matched[phase],
            "evidence": evidence,
        }
    if len(series_matches) > 1 or len(protocol_matches) > 1 or protocol_multiphase:
        return {
            "phase": "UNKNOWN",
            "broad_phase": "UNKNOWN",
            "confidence": 0.0,
            "phase_status": "conflicting_explicit_labels",
            "matched_patterns": [
                pattern
                for patterns in {**series_matches, **protocol_matches}.values()
                for pattern in patterns
            ],
            "evidence": evidence,
        }
    if len(protocol_matches) == 1:
        phase = next(iter(protocol_matches))
        return {
            "phase": phase,
            "broad_phase": "NC" if phase == "NC" else "CE",
            "confidence": 0.9,
            "phase_status": "single_protocol_phase",
            "matched_patterns": protocol_matches[phase],
            "evidence": evidence,
        }
    searchable = f"{series_text} {protocol_text}"
    if not protocol_multiphase and (
        contrast_agent or any(re.search(pattern, searchable) for pattern in CE_PATTERNS)
    ):
        return {
            "phase": "CE_UNSPECIFIED",
            "broad_phase": "CE",
            "confidence": 0.75,
            "phase_status": "contrast_unspecified_phase",
            "matched_patterns": [
                pattern for pattern in CE_PATTERNS if re.search(pattern, searchable)
            ],
            "evidence": evidence,
        }
    return {
        "phase": "UNKNOWN",
        "broad_phase": "UNKNOWN",
        "confidence": 0.0,
        "phase_status": "insufficient_evidence",
        "matched_patterns": [],
        "evidence": evidence,
    }


def selected_series_ids(output_root: Path) -> set[str]:
    selected = set()
    for path in (output_root / "ct_qc").glob("*/selection_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        series_id = str(payload.get("selected_series", {}).get("ct_id", "") or "")
        if series_id:
            selected.add(series_id)
    return selected


def read_report_rows(output_root: Path, inventory: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected_ids = selected_series_ids(output_root)
    rows = []
    for report_path in sorted((output_root / "ct_qc").glob("*/dicom_prefilter_report.csv")):
        with report_path.open(encoding="utf-8", newline="") as handle:
            for report in csv.DictReader(handle):
                source_path = str(report.get("source_path", ""))
                meta = inventory.get(source_path, {})
                sidecar, sidecar_errors = read_sidecars(report.get("generated_json_files", ""))
                phase = classify_phase(meta, sidecar)
                generated = ";".join(
                    [report.get("generated_json_files", ""), report.get("converted_path", "")]
                )
                phase["case_id"] = str(report.get("case_id", "") or meta.get("case_id", ""))
                phase.update(
                    {
                        "source_path": source_path,
                        "series_uid": meta.get("series_uid", ""),
                        "study_uid": meta.get("study_uid", ""),
                        "series_description": meta.get("series_description", ""),
                        "study_description": meta.get("study_description", ""),
                        "protocol_name": sidecar.get("ProtocolName", ""),
                        "contrast_agent": sidecar.get("ContrastBolusAgent", ""),
                        "contrast_start_time": sidecar.get("ContrastBolusStartTime", ""),
                        "contrast_volume": sidecar.get("ContrastBolusVolume", ""),
                        "acquisition_time": sidecar.get("AcquisitionTime", ""),
                        "series_time": sidecar.get("SeriesTime", ""),
                        "content_time": sidecar.get("ContentTime", ""),
                        "series_number": sidecar.get("SeriesNumber", ""),
                        "acquisition_number": sidecar.get("AcquisitionNumber", ""),
                        "report_path": str(report_path),
                        "record_index": report.get("record_index", ""),
                        "prefilter_pass": report.get("prefilter_pass", ""),
                        "nifti_qc_pass": report.get("nifti_qc_pass", ""),
                        "totalseg_pass": report.get("totalseg_pass", ""),
                        "totalseg_execution_success": report.get("totalseg_execution_success", ""),
                        "selected_by_current_qc": any(
                            series_id and series_id in generated for series_id in selected_ids
                        ),
                        "sidecar_errors": sidecar_errors,
                    }
                )
                phase["eligible_candidate"] = all(
                    str(report.get(key, "")).lower() == "true"
                    for key in ("prefilter_pass", "nifti_qc_pass", "totalseg_pass")
                )
                rows.append(phase)
    return rows


def build_case_rows(series_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in series_rows:
        if row["eligible_candidate"]:
            grouped[row["case_id"]].append(row)
    all_case_ids = sorted({row["case_id"] for row in series_rows})
    output = []
    for case_id in all_case_ids:
        candidates = grouped.get(case_id, [])
        fine_phases = sorted({row["phase"] for row in candidates if row["phase"] in {"NC", "ART", "NEPH", "DEL"}})
        broad_phases = sorted({row["broad_phase"] for row in candidates if row["broad_phase"] != "UNKNOWN"})
        output.append(
            {
                "case_id": case_id,
                "candidate_series_count": len(candidates),
                "fine_phases": ";".join(fine_phases),
                "broad_phases": ";".join(broad_phases),
                "has_nc": "NC" in fine_phases,
                "has_art": "ART" in fine_phases,
                "has_neph": "NEPH" in fine_phases,
                "has_del": "DEL" in fine_phases,
                "has_ce": "CE" in broad_phases,
                "has_unknown_only": bool(candidates) and not fine_phases and not broad_phases,
                "retained_any_qc_candidate": bool(candidates),
                "exclusion_reason": "" if candidates else "no_existing_qc_candidate",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(inventory_path: Path, output_root: Path, experiment_root: Path) -> dict[str, Any]:
    inventory = read_inventory(inventory_path)
    series_rows = read_report_rows(output_root, inventory)
    case_rows = build_case_rows(series_rows)
    candidate_rows = [row for row in series_rows if row["eligible_candidate"]]
    sidecar_fields = (
        "contrast_agent",
        "contrast_start_time",
        "contrast_volume",
        "acquisition_time",
        "series_time",
        "content_time",
        "series_number",
        "acquisition_number",
    )
    summary = {
        "experiment": "ct_phase_audit_v1",
        "inputs": {
            "inventory_path": str(inventory_path),
            "qc_output_root": str(output_root),
            "rerun_models": False,
            "new_aorta_segmentation": False,
        },
        "case_count": len(case_rows),
        "series_count": len(series_rows),
        "eligible_candidate_series_count": len(candidate_rows),
        "cases_with_existing_qc_candidate": sum(row["retained_any_qc_candidate"] for row in case_rows),
        "cases_without_existing_qc_candidate": sum(not row["retained_any_qc_candidate"] for row in case_rows),
        "candidate_phase_series_counts": dict(Counter(row["phase"] for row in candidate_rows)),
        "candidate_broad_phase_series_counts": dict(Counter(row["broad_phase"] for row in candidate_rows)),
        "sidecar_field_presence": {
            field: sum(bool(str(row.get(field, "") or "").strip()) for row in candidate_rows)
            for field in sidecar_fields
        },
        "case_retention": {
            "any_qc_candidate": sum(row["retained_any_qc_candidate"] for row in case_rows),
            "NC": sum(row["has_nc"] for row in case_rows),
            "ART": sum(row["has_art"] for row in case_rows),
            "NEPH": sum(row["has_neph"] for row in case_rows),
            "DEL": sum(row["has_del"] for row in case_rows),
            "CE": sum(row["has_ce"] for row in case_rows),
            "fine_phase_any": sum(bool(row["fine_phases"]) for row in case_rows),
        },
        "phase_policy": {
            "description_is_supporting_evidence": True,
            "generic_protocol_names_are_not_phase_labels": True,
            "eligible_candidate_requires": ["prefilter_pass", "nifti_qc_pass", "totalseg_pass"],
            "high_confidence_threshold": 0.95,
        },
        "output_files": {
            "series": str(experiment_root / "series_phase_audit.csv"),
            "cases": str(experiment_root / "case_phase_retention.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    write_csv(experiment_root / "series_phase_audit.csv", series_rows)
    write_csv(experiment_root / "case_phase_retention.csv", case_rows)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit conservative CT phase coverage from existing QC artifacts.")
    parser.add_argument("--inventory", type=Path, default=Path("data/tcga_kirc_data.json"))
    parser.add_argument("--output-root", type=Path, default=Path("output_kirc"))
    parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v9/experiment_ct_phase_audit"))
    args = parser.parse_args()
    summary = run(args.inventory, args.output_root, args.experiment_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
