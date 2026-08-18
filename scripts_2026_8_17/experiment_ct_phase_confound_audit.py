from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import chi2_contingency


ASSOCIATION_FIELDS = (
    "phase_group",
    "manufacturer",
    "scanner_model",
    "reconstruction_kernel",
    "slice_thickness",
)


def coarse_phase_group(value: str) -> str:
    phase = str(value or "UNKNOWN").strip().upper()
    if phase in {"NC", "ART", "NEPH", "DEL", "UNKNOWN"}:
        return phase
    if phase in {"MAIN_CE_HIGH", "MAIN_CE_MEDIUM", "CE_UNSPECIFIED"}:
        return "OTHER_CE"
    return "UNKNOWN"


def cramers_v(table: list[list[int]], chi2: float) -> float:
    n = sum(sum(row) for row in table)
    rows = len(table)
    columns = len(table[0]) if table else 0
    denominator = n * max(min(rows - 1, columns - 1), 1)
    return (chi2 / denominator) ** 0.5 if denominator else 0.0


def contingency_association(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    usable = [
        row for row in rows
        if str(row.get("cluster", "") or "").strip()
        and str(row.get(field, "") or "").strip()
    ]
    missing_count = len(rows) - len(usable)
    clusters = sorted({str(row["cluster"]) for row in usable})
    levels = sorted({str(row[field]) for row in usable})
    table = [
        [sum(str(row["cluster"]) == cluster and str(row[field]) == level for row in usable) for level in levels]
        for cluster in clusters
    ]
    result = {
        "field": field,
        "n": len(usable),
        "missing_count": missing_count,
        "clusters": clusters,
        "levels": levels,
        "table": table,
        "chi2": None,
        "p_value": None,
        "degrees_of_freedom": None,
        "cramers_v": None,
    }
    if len(clusters) < 2 or len(levels) < 2:
        return result
    chi2, p_value, degrees_of_freedom, _ = chi2_contingency(table, correction=False)
    result.update(
        {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(degrees_of_freedom),
            "cramers_v": float(cramers_v(table, float(chi2))),
        }
    )
    return result


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_selected_metadata(
    phase_csv: Path, qc_root: Path, patient_states_path: Path
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with phase_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("selected_by_current_qc", "").lower() != "true":
                continue
            selected.setdefault(row["case_id"], row)

    formal_values = {}
    if patient_states_path.is_file():
        states = {}
        with patient_states_path.open(encoding="utf-8") as handle:
            for line in handle:
                state = json.loads(line)
                states[str(state.get("case_id", "") or "")] = state
        from tools.tool_confound_test import confounder_values

        formal_values = confounder_values(states, str(qc_root.parent))

    output = []
    for case_id, phase_row in sorted(selected.items()):
        selection_path = qc_root / case_id / "selection_summary.json"
        selection = read_json(selection_path) if selection_path.is_file() else {}
        selected_series = dict(selection.get("selected_series", {}) or {})
        ct_id = str(selected_series.get("ct_id", "") or "")
        sidecar = {}
        if ct_id:
            sidecar_path = qc_root / case_id / "dcm2nii" / f"{ct_id}.json"
            if sidecar_path.is_file():
                sidecar = read_json(sidecar_path)
        formal = dict(formal_values.get(case_id, {}) or {})
        phase = str(formal.get("ct_phase", "") or phase_row.get("inferred_phase", "") or "UNKNOWN").strip().upper()
        output.append(
            {
                "case_id": case_id,
                "selected_ct_id": ct_id,
                "explicit_phase": phase_row.get("explicit_phase", ""),
                "inferred_phase": phase,
                "phase_label": phase,
                "phase_group": formal.get("ct_phase_group") or coarse_phase_group(phase),
                "phase_confidence": phase_row.get("inferred_confidence", ""),
                "phase_source": phase_row.get("phase_source", ""),
                "manufacturer": formal.get("ct_manufacturer") or str(sidecar.get("Manufacturer", "") or "").strip(),
                "scanner_model": formal.get("ct_scanner_model") or str(sidecar.get("ManufacturerModelName", "") or "").strip(),
                "reconstruction_kernel": formal.get("ct_reconstruction_kernel") or str(sidecar.get("ConvolutionKernel", "") or "").strip(),
                "slice_thickness": formal.get("ct_slice_thickness") or selected_series.get("slice_thickness_median", ""),
                "acquisition_time": phase_row.get("acquisition_time", ""),
                "series_number": phase_row.get("series_number", ""),
                "series_uid": phase_row.get("series_uid", ""),
            }
        )
    return output


def read_cluster_labels(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        clusters = payload if isinstance(payload, list) else payload.get("clusters", [])
        return {
            str(case_id): str(cluster.get("cluster_id", ""))
            for cluster in clusters
            for case_id in list(cluster.get("member_ids", []) or [])
            if cluster.get("cluster_id") and case_id
        }
    labels = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id", "") or "").strip()
            cluster = str(row.get("cluster", row.get("cluster_label", "")) or "").strip()
            if case_id and cluster:
                labels[case_id] = cluster
    return labels


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_bh_q_values(rows: list[dict[str, Any]]) -> None:
    usable = [(index, float(row["p_value"])) for index, row in enumerate(rows) if row.get("p_value") is not None]
    usable.sort(key=lambda item: item[1])
    q_values = [0.0] * len(usable)
    running = 1.0
    for rank in range(len(usable), 0, -1):
        index, p_value = usable[rank - 1]
        running = min(running, p_value * len(usable) / rank)
        q_values[rank - 1] = running
    for (index, _), q_value in zip(usable, q_values):
        rows[index]["q_value"] = q_value


def run(
    phase_csv: Path,
    qc_root: Path,
    cluster_paths: list[Path],
    patient_states_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    metadata = read_selected_metadata(phase_csv, qc_root, patient_states_path)
    metadata_by_case = {row["case_id"]: row for row in metadata}
    association_rows = []
    contingency_rows = []
    partitions = []
    for cluster_path in cluster_paths:
        labels = read_cluster_labels(cluster_path)
        joined = [
            {**metadata_by_case[case_id], "cluster": cluster}
            for case_id, cluster in sorted(labels.items())
            if case_id in metadata_by_case
        ]
        partition_id = cluster_path.stem
        associations = []
        for field in ASSOCIATION_FIELDS:
            result = contingency_association(joined, field)
            result["partition"] = partition_id
            associations.append(result)
            association_rows.append(result)
            for cluster, counts in zip(result["clusters"], result["table"]):
                for level, count in zip(result["levels"], counts):
                    contingency_rows.append(
                        {
                            "partition": partition_id,
                            "field": field,
                            "cluster": cluster,
                            "level": level,
                            "count": count,
                        }
                    )
        partitions.append(
            {
                "partition": partition_id,
                "source": str(cluster_path),
                "label_count": len(labels),
                "joined_case_count": len(joined),
                "cluster_counts": dict(Counter(row["cluster"] for row in joined)),
                "associations": associations,
            }
        )
    add_bh_q_values(association_rows)
    for partition in partitions:
        q_by_field = {
            row["field"]: row.get("q_value")
            for row in association_rows
            if row["partition"] == partition["partition"]
        }
        for association in partition["associations"]:
            association["q_value"] = q_by_field.get(association["field"])

    summary = {
        "experiment": "ct_phase_confound_audit_v2",
        "inputs": {
            "phase_audit_csv": str(phase_csv),
            "qc_root": str(qc_root),
            "cluster_files": [str(path) for path in cluster_paths],
            "patient_states": str(patient_states_path),
            "rerun_radiomics": False,
            "new_aorta_segmentation": False,
        },
        "selected_case_count": len(metadata),
        "selected_phase_counts": dict(Counter(row["phase_group"] for row in metadata)),
        "partitions": partitions,
        "interpretation_policy": {
            "phase_association_is_diagnostic_not_exclusion": True,
            "missing_metadata_is_not_imputed": True,
            "no_enhancement_score_included": True,
            "association_effect_size": "Cramer's V",
            "phase_grouping": {
                "NC": "NC",
                "ART": "ART",
                "NEPH": "NEPH",
                "DEL": "DEL",
                "MAIN_CE_HIGH/MAIN_CE_MEDIUM/CE_UNSPECIFIED": "OTHER_CE",
                "unrecognized_or_missing": "UNKNOWN",
            },
        },
        "output_files": {
            "selected_cases": str(experiment_root / "selected_ct_confounders.csv"),
            "associations": str(experiment_root / "cluster_confounder_associations.csv"),
            "contingencies": str(experiment_root / "cluster_confounder_contingencies.csv"),
            "summary": str(experiment_root / "summary.json"),
        },
    }
    write_csv(experiment_root / "selected_ct_confounders.csv", metadata)
    write_csv(experiment_root / "cluster_confounder_associations.csv", association_rows)
    write_csv(experiment_root / "cluster_confounder_contingencies.csv", contingency_rows)
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CT phase and technical confounding against existing clusters.")
    parser.add_argument(
        "--phase-csv",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_audit/series_phase_audit.csv"),
    )
    parser.add_argument("--qc-root", type=Path, default=Path("output_kirc/ct_qc"))
    parser.add_argument(
        "--patient-states",
        type=Path,
        default=Path("output_kirc/storage/patient_states/patient_states.jsonl"),
    )
    parser.add_argument("--cluster-json", type=Path, default=None)
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        default=Path("output_kirc_v8/experiment_06_four_modal_consensus_stability"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("output_kirc_v9/experiment_ct_phase_confound_audit"),
    )
    args = parser.parse_args()
    cluster_paths = [args.cluster_json] if args.cluster_json else sorted(args.cluster_dir.glob("consensus_labels_k*.csv"))
    if not cluster_paths:
        raise FileNotFoundError(f"No consensus label CSV found in {args.cluster_dir}")
    summary = run(args.phase_csv, args.qc_root, cluster_paths, args.patient_states, args.experiment_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
