from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.io import ensure_dir, write_json
from utils.llm_utils import load_yaml_file
from utils.tool_utils import safe_identifier


DEFAULT_REVIEW_TOOLS = {
    "tool_stability_check": {
        "module": "tools.tool_stability_check",
        "function": "tool_stability_check",
        "evidence_blocks": ["set_reliability"],
    },
    "tool_survival_analysis": {
        "module": "tools.tool_survival_analysis",
        "function": "tool_survival_analysis",
        "evidence_blocks": ["clinical_context"],
    },
    "tool_mutation_enrichment": {
        "module": "tools.tool_mutation_enrichment",
        "function": "tool_mutation_enrichment",
        "evidence_blocks": ["biological_support"],
    },
    "tool_pathway_enrichment": {
        "module": "tools.tool_pathway_enrichment",
        "function": "tool_pathway_enrichment",
        "evidence_blocks": ["biological_support"],
    },
    "tool_confound_test": {
        "module": "tools.tool_confound_test",
        "function": "tool_confound_test",
        "evidence_blocks": ["confounder_exclusion"],
    },
    "tool_known_label_echo_test": {
        "module": "tools.tool_known_label_echo_test",
        "function": "tool_known_label_echo_test",
        "evidence_blocks": ["known_label_echo"],
    },
    "tool_multimodal_consistency_check": {
        "module": "tools.tool_multimodal_consistency_check",
        "function": "tool_multimodal_consistency_check",
        "evidence_blocks": ["multimodal_support"],
    },
}

DEFAULT_CONFOUNDER_FIELDS = {
    "demographic": ["gender", "race", "age_at_index"],
    "technical": [
        "ct_manufacturer",
        "year_of_diagnosis",
    ],
    "known_or_disease_label": ["stage_group", "stage", "grade", "primary_diagnosis", "morphology"],
}

MOFS_CLUSTER_COLORS = [
    "#119da4",
    "#ff6666",
    "#ffc857",
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#8ab17d",
    "#6d597a",
]


def mofs_cluster_color(index: int) -> str:
    return MOFS_CLUSTER_COLORS[int(index) % len(MOFS_CLUSTER_COLORS)]


def style_mofs_axes(ax, *, grid_axis: str = "both") -> None:
    ax.set_facecolor("#f3f6f6")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=8, colors="black")
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color="#cacfd2",
            linestyle="--",
            linewidth=0.6,
            alpha=0.8,
        )
        ax.set_axisbelow(True)


def subtype_review_dir(output_root: str, cluster_id: str) -> Path:
    return ensure_dir(
        Path(output_root) / "subtype_review" / safe_identifier(cluster_id)
    )


def subtype_review_figure_dir(output_root: str, cluster_id: str) -> Path:
    return ensure_dir(subtype_review_dir(output_root, cluster_id) / "figures")


def subtype_review_global_figure_dir(output_root: str) -> Path:
    return ensure_dir(Path(output_root) / "subtype_review" / "global" / "figures")


def subtype_review_config(config_dir: str) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser() if config_dir else Path("configs")
    try:
        return load_yaml_file(config_path / "subtype_review.yaml")
    except Exception:
        return {}


def subtype_review_tools_config(config_dir: str) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser() if config_dir else Path("configs")
    try:
        return load_yaml_file(config_path / "subtype_review_tools.yaml")
    except Exception:
        return {}


def subtype_review_tool_definitions(config_dir: str = "") -> dict[str, dict[str, Any]]:
    tools_config = subtype_review_tools_config(config_dir)
    raw_tools = dict(tools_config.get("tools", {}) or DEFAULT_REVIEW_TOOLS)
    definitions = {}
    for tool_name, raw_definition in raw_tools.items():
        definition = dict(raw_definition or {})
        definitions[str(tool_name)] = {
            "module": str(definition.get("module", "") or ""),
            "function": str(definition.get("function", "") or ""),
            "evidence_blocks": [
                str(item)
                for item in list(definition.get("evidence_blocks", []) or [])
                if str(item)
            ],
        }
    return definitions


def subtype_review_tool_imports(config_dir: str = "") -> dict[str, tuple[str, str]]:
    imports = {}
    for tool_name, definition in subtype_review_tool_definitions(config_dir).items():
        module_name = str(definition.get("module", "") or "")
        function_name = str(definition.get("function", "") or "")
        if module_name and function_name:
            imports[tool_name] = (module_name, function_name)
    return imports


def subtype_review_confounder_fields(config_dir: str = "") -> dict[str, set[str]]:
    tools_config = subtype_review_tools_config(config_dir)
    raw_fields = dict(tools_config.get("confounder_fields", {}) or DEFAULT_CONFOUNDER_FIELDS)
    return {
        "demographic": {str(item).lower() for item in list(raw_fields.get("demographic", []) or [])},
        "technical": {str(item).lower() for item in list(raw_fields.get("technical", []) or [])},
        "known_or_disease_label": {
            str(item).lower()
            for item in list(raw_fields.get("known_or_disease_label", []) or [])
        },
    }


def tool_parameters(config_dir: str, tool_key: str) -> dict[str, Any]:
    tools_config = subtype_review_tools_config(config_dir)
    return dict(tools_config.get(tool_key, {}) or {})


def figures_enabled(config_dir: str) -> bool:
    policy = dict(subtype_review_config(config_dir).get("artifact_policy", {}) or {})
    return bool(policy.get("save_figures", True))


def save_png_figure(
    output_root: str,
    cluster_id: str,
    filename: str,
    draw_function,
    *,
    config_dir: str = "",
) -> str:
    if not figures_enabled(config_dir):
        return ""
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz = dict(subtype_review_config(config_dir).get("visualization", {}) or {})
        dpi = int(viz.get("dpi", 180) or 180)
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
        draw_function(fig, ax)
        fig.tight_layout()
        path = subtype_review_figure_dir(output_root, cluster_id) / filename
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return str(path)
    except Exception:
        return ""


def save_global_png_figure(
    output_root: str,
    filename: str,
    draw_function,
    *,
    config_dir: str = "",
) -> str:
    if not figures_enabled(config_dir):
        return ""
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz = dict(subtype_review_config(config_dir).get("visualization", {}) or {})
        dpi = int(viz.get("dpi", 180) or 180)
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
        draw_function(fig, ax)
        fig.tight_layout()
        path = subtype_review_global_figure_dir(output_root) / filename
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return str(path)
    except Exception:
        return ""


def tool_result(
    *,
    tool_name: str,
    status: str,
    cluster_id: str,
    output_root: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    evidence_hints: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    missing_reason: str = "",
    artifact_key: str = "",
    errors: list[str] | None = None,
    support_level: str = "none",
    concern_level: str = "none",
    figures: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        "tool_name": tool_name,
        "status": status,
        "cluster_id": cluster_id,
        "results": {
            "summary": summary,
            "metrics": metrics or {},
            "support_level": support_level,
            "concern_level": concern_level,
            "evidence_hints": evidence_hints or [],
            "warnings": warnings or [],
            "missing_reason": missing_reason,
        },
        "artifacts": {
            "figures": {
                key: value for key, value in dict(figures or {}).items() if value
            }
        },
        "errors": errors or [],
    }
    return result


def member_case_ids(cluster_state: Mapping[str, Any]) -> set[str]:
    return {str(item) for item in list(cluster_state.get("member_ids", []) or [])}


def split_member_states(
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    members = member_case_ids(cluster_state)
    inside, outside = [], []
    for case_id, patient_state in patient_states_by_id.items():
        target = inside if str(case_id) in members else outside
        target.append(dict(patient_state))
    return inside, outside


def cluster_vs_rest_ids(
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    members = member_case_ids(cluster_state)
    member_ids = [
        str(case_id) for case_id in patient_states_by_id if str(case_id) in members
    ]
    rest_ids = [
        str(case_id) for case_id in patient_states_by_id if str(case_id) not in members
    ]
    return member_ids, rest_ids


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
                        table[case_id][feature_name] = float(value) if str(value).strip() else float("nan")
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


def bh_fdr(p_values: list[float]) -> list[float]:
    from statsmodels.stats.multitest import multipletests

    values = [
        float(value) if value is not None and math.isfinite(float(value)) else 1.0
        for value in p_values
    ]
    return [float(value) for value in multipletests(values, method="fdr_bh")[1]]


def fisher_exact_result(a: int, b: int, c: int, d: int) -> tuple[float | None, float | None]:
    from scipy.stats import fisher_exact

    result = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(result.statistic), float(result.pvalue)
