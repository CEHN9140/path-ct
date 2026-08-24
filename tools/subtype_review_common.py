from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.io import ensure_dir, write_json
from utils.llm_utils import load_yaml_file
from utils.tool_utils import safe_identifier

def subtype_review_dir(output_root: str, cluster_id: str) -> Path:
    return ensure_dir(
        Path(output_root) / "subtype_review" / safe_identifier(cluster_id)
    )


def subtype_review_config(config_dir: str) -> dict[str, Any]:
    return load_yaml_file(Path(config_dir).expanduser() / "subtype_review.yaml")


def tool_parameters(config_dir: str, tool_key: str) -> dict[str, Any]:
    return dict(subtype_review_config(config_dir)[tool_key])


def tool_result(
    *,
    tool_name: str,
    status: str,
    cluster_id: str,
    output_root: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    decision_metrics: dict[str, Any] | None = None,
    evidence_hints: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    missing_reason: str = "",
    artifact_key: str = "",
    errors: list[str] | None = None,
    support_level: str = "none",
    concern_level: str = "none",
    figures: dict[str, str] | None = None,
) -> dict[str, Any]:
    full_metrics = metrics or {}
    full_metrics_path = ""
    if full_metrics:
        full_metrics_path = str(
            subtype_review_dir(output_root, cluster_id)
            / f"{safe_identifier(tool_name)}_full_metrics.json"
        )
        write_json(full_metrics_path, full_metrics)
    result = {
        "tool_name": tool_name,
        "status": status,
        "cluster_id": cluster_id,
        "results": {
            "summary": summary,
            "metrics": full_metrics,
            "decision_metrics": (
                full_metrics if decision_metrics is None else decision_metrics
            ),
            "support_level": support_level,
            "concern_level": concern_level,
            "evidence_hints": evidence_hints or [],
            "warnings": warnings or [],
            "missing_reason": missing_reason,
        },
        "artifacts": {
            "full_metrics": full_metrics_path,
            "figures": {
                key: value for key, value in dict(figures or {}).items() if value
            }
        },
        "errors": errors or [],
    }
    return result


def member_case_ids(cluster_state: Mapping[str, Any]) -> set[str]:
    return {str(item) for item in list(cluster_state.get("member_ids", []) or [])}


def scoped_candidate_sets(
    scope: str,
    cluster_state: Mapping[str, Any],
    all_cluster_states: list[Mapping[str, Any]] | None,
) -> dict[str, set[str]]:
    current = {
        str(state.get("set_id") or state.get("cluster_id") or ""): member_case_ids(state)
        for state in list(all_cluster_states or [cluster_state])
        if str(state.get("set_id") or state.get("cluster_id") or "")
    }
    return current


def clinical_dict(patient_state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = dict(patient_state.get("inventory", {}) or {})
    return dict(
        patient_state.get("clinical", {}) or inventory.get("Clinical", {}) or {}
    )


def float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_stage(value: Any) -> str:
    text = str(value or "").strip()
    for stage in ("IV", "III", "II", "I"):
        if f"Stage {stage}" in text or text == stage:
            return stage
    return text


def first_diagnosis(clinical: Mapping[str, Any]) -> dict[str, Any]:
    diagnoses = [dict(item) for item in list(clinical.get("diagnoses", []) or [])]
    primary = [
        item
        for item in diagnoses
        if str(item.get("diagnosis_is_primary_disease", "")).lower() == "true"
    ]
    return dict((primary or diagnoses or [{}])[0])


def clinical_record(case_id: str, patient_state: Mapping[str, Any]) -> dict[str, Any]:
    clinical = clinical_dict(patient_state)
    diagnosis = first_diagnosis(clinical)
    demographic = dict(clinical.get("demographic", {}) or {})
    follow_ups = [dict(item) for item in list(clinical.get("follow_ups", []) or [])]
    days_to_death = float_or_none(demographic.get("days_to_death"))
    days_to_last_follow_up = float_or_none(diagnosis.get("days_to_last_follow_up"))
    if days_to_last_follow_up is None:
        follow_days = [
            float_or_none(item.get("days_to_follow_up")) for item in follow_ups
        ]
        follow_days = [item for item in follow_days if item is not None]
        days_to_last_follow_up = max(follow_days) if follow_days else None
    vital_status = str(demographic.get("vital_status", "") or "")
    os_event = vital_status.strip().lower() in {"dead", "deceased", "1", "true", "yes"}
    os_time = days_to_death if days_to_death is not None else days_to_last_follow_up
    inventory = dict(patient_state.get("inventory", {}) or {})
    ct_entries = [dict(item) for item in list(inventory.get("CT", []) or [])]
    manufacturer = str(
        (ct_entries[0] if ct_entries else {}).get("Manufacturer", "") or ""
    )
    return {
        "case_id": str(case_id),
        "os_time": os_time,
        "os_event": int(os_event) if os_time is not None else None,
        "age": float_or_none(demographic.get("age_at_index")),
        "gender": str(demographic.get("gender", "") or ""),
        "race": str(demographic.get("race", "") or ""),
        "vital_status": vital_status,
        "stage": str(diagnosis.get("ajcc_pathologic_stage", "") or ""),
        "stage_group": normalize_stage(diagnosis.get("ajcc_pathologic_stage", "")),
        "t_stage": str(diagnosis.get("ajcc_pathologic_t", "") or ""),
        "n_stage": str(diagnosis.get("ajcc_pathologic_n", "") or ""),
        "m_stage": str(diagnosis.get("ajcc_pathologic_m", "") or ""),
        "grade": str(diagnosis.get("tumor_grade", "") or ""),
        "year_of_diagnosis": float_or_none(diagnosis.get("year_of_diagnosis")),
        "primary_diagnosis": str(diagnosis.get("primary_diagnosis", "") or ""),
        "morphology": str(diagnosis.get("morphology", "") or ""),
        "manufacturer": manufacturer,
    }


def clinical_table(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(case_id): clinical_record(str(case_id), dict(patient_state))
        for case_id, patient_state in patient_states_by_id.items()
    }


def read_case_feature_table(path: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    if not path or not Path(path).exists():
        return [], {}
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            feature_names = [
                name for name in list(reader.fieldnames or []) if name != "case_id"
            ]
            table: dict[str, dict[str, float]] = {}
            for row in reader:
                case_id = str(row.get("case_id", "") or "")
                if not case_id:
                    continue
                table[case_id] = {}
                for feature_name in feature_names:
                    try:
                        value = row.get(feature_name, "")
                        table[case_id][feature_name] = (
                            float(value) if str(value).strip() else float("nan")
                        )
                    except (TypeError, ValueError):
                        table[case_id][feature_name] = float("nan")
        return feature_names, table
    except Exception:
        return [], {}


def read_gmt_gene_sets(path: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    gmt_path = Path(path).expanduser()
    if not path or not gmt_path.exists():
        return {}, {}
    pathway_to_genes: dict[str, list[str]] = {}
    gene_to_pathways: dict[str, list[str]] = {}
    with gmt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pathway = str(parts[0] or "").strip()
            genes = [str(item).strip() for item in parts[2:] if str(item).strip()]
            if not pathway or not genes:
                continue
            unique_genes = sorted(set(genes))
            pathway_to_genes[pathway] = unique_genes
            for gene in unique_genes:
                gene_to_pathways.setdefault(gene, []).append(pathway)
    return pathway_to_genes, gene_to_pathways


def feature_dataframe(path: str):
    import pandas as pd

    if not path or not Path(path).exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "case_id" not in frame.columns:
        return pd.DataFrame()
    for column in frame.columns:
        if column != "case_id":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index("case_id", drop=True)


def standardized_mean_difference(values_a, values_b) -> float | None:
    array_a = np.asarray(list(values_a), dtype=float)
    array_b = np.asarray(list(values_b), dtype=float)
    array_a = array_a[np.isfinite(array_a)]
    array_b = array_b[np.isfinite(array_b)]
    if len(array_a) < 2 or len(array_b) < 2:
        return None
    pooled_var = (
        (len(array_a) - 1) * float(np.var(array_a, ddof=1))
        + (len(array_b) - 1) * float(np.var(array_b, ddof=1))
    ) / float(len(array_a) + len(array_b) - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    if not math.isfinite(pooled_sd) or pooled_sd == 0:
        return None
    return float((np.mean(array_a) - np.mean(array_b)) / pooled_sd)


def cliffs_delta(values_a, values_b) -> float | None:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    return float((np.greater.outer(a, b).sum() - np.less.outer(a, b).sum()) / (len(a) * len(b)))


def epsilon_squared(groups: list[list[float]]) -> float | None:
    values = [np.asarray(group, dtype=float) for group in groups]
    values = [group[np.isfinite(group)] for group in values if len(group)]
    n = sum(len(group) for group in values)
    k = len(values)
    if n <= k or k < 2:
        return None
    from scipy.stats import kruskal

    statistic = float(kruskal(*values).statistic)
    return float(max(0.0, (statistic - k + 1.0) / (n - k)))


def bias_corrected_cramers_v(table: np.ndarray) -> float | None:
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or min(table.shape) < 2 or table.sum() <= 1:
        return None
    from scipy.stats import chi2_contingency

    chi2 = float(chi2_contingency(table, correction=False)[0])
    n = float(table.sum())
    phi2 = chi2 / n
    rows, cols = table.shape
    phi2corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1.0))
    rows_corr = rows - ((rows - 1) ** 2) / (n - 1.0)
    cols_corr = cols - ((cols - 1) ** 2) / (n - 1.0)
    denominator = min(cols_corr - 1.0, rows_corr - 1.0)
    return float(math.sqrt(phi2corr / denominator)) if denominator > 0 else None


def bh_fdr(p_values: list[float]) -> list[float]:
    from statsmodels.stats.multitest import multipletests

    if not p_values:
        return []
    values = [
        float(value) if value is not None and math.isfinite(float(value)) else 1.0
        for value in p_values
    ]
    return [float(value) for value in multipletests(values, method="fdr_bh")[1]]


def assign_groupwise_fdr(
    rows: list[dict], group_key: str, p_key: str, q_key: str
) -> None:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key, "")), []).append(row)
    for group_rows in groups.values():
        p_values = [row.get(p_key) for row in group_rows]
        for row, q_value in zip(group_rows, bh_fdr(p_values)):
            row[q_key] = float(q_value)


def fisher_exact_result(
    a: int, b: int, c: int, d: int
) -> tuple[float | None, float | None]:
    from scipy.stats import fisher_exact

    result = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float | None, list[float] | None]:
    cells = [float(a), float(b), float(c), float(d)]
    if any(value == 0 for value in cells):
        cells = [value + 0.5 for value in cells]
    aa, bb, cc, dd = cells
    odds = aa * dd / (bb * cc)
    import math
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    log_odds = math.log(odds)
    return odds, [math.exp(log_odds - 1.96 * se), math.exp(log_odds + 1.96 * se)]
