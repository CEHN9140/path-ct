from __future__ import annotations

import math
import numpy as np

from tools.subtype_review_common import (
    bh_fdr,
    enrichment_decision_metrics,
    feature_dataframe,
    member_case_ids,
    read_gmt_gene_sets,
    standardized_mean_difference,
    tool_parameters,
    tool_result,
)
from utils.visualization import configure_matplotlib


def candidate_set_members(all_cluster_states, fallback_cluster_state):
    states = list(all_cluster_states or [fallback_cluster_state])
    return {
        str(state.get("cluster_id", "") or ""): member_case_ids(state)
        for state in states
        if str(state.get("cluster_id", "") or "")
    }


def round_float(value, digits=6):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits)


def rna_feature_path(patient_states_by_id):
    for patient_state in patient_states_by_id.values():
        omics = dict(patient_state.get("omics_evidence", {}) or {})
        path = str(
            omics.get("rna_pathway_feature_path")
            or omics.get("rna_feature_path")
            or ""
        )
        if path:
            return path
    return ""


def empty_pathway_result(cluster_id, output_root, summary, missing_reason):
    return tool_result(
        tool_name="tool_pathway_enrichment",
        status="missing",
        cluster_id=cluster_id,
        output_root=output_root,
        summary=summary,
        metrics={"rna_pathway_enrichment": []},
        decision_metrics={"per_set_rna_pathway_enrichment": {}},
        missing_reason=missing_reason,
        support_level="none",
        concern_level="moderate",
        figures={},
    )


def ssgsea_scores(feature_frame, pathway_to_genes, min_size):
    configure_matplotlib()
    from gseapy import ssgsea

    result = ssgsea(
        data=feature_frame.transpose(),
        gene_sets=pathway_to_genes,
        outdir=None,
        no_plot=True,
        threads=1,
        min_size=min_size,
        verbose=False,
        seed=123,
    ).res2d
    return result.pivot(index="Name", columns="Term", values="NES")


def finite_scores(score_frame, case_ids, pathway):
    if not case_ids:
        return []
    return (
        score_frame.loc[[case_id for case_id in case_ids if case_id in score_frame.index], pathway]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tolist()
    )


def pathway_rows(score_frame, pathway_gene_counts, candidate_sets):
    from scipy.stats import mannwhitneyu

    all_case_ids = sorted({case_id for members in candidate_sets.values() for case_id in members})
    rows = []
    for candidate_set_id, members in candidate_sets.items():
        set_ids = [case_id for case_id in all_case_ids if case_id in members]
        rest_ids = [case_id for case_id in all_case_ids if case_id not in members]
        for pathway in score_frame.columns:
            set_values = finite_scores(score_frame, set_ids, pathway)
            rest_values = finite_scores(score_frame, rest_ids, pathway)
            available_n = len(set_values) + len(rest_values)
            p_value = None
            if set_values and rest_values:
                p_value = float(
                    mannwhitneyu(set_values, rest_values, alternative="two-sided").pvalue
                )
            set_mean = float(np.mean(set_values)) if set_values else None
            rest_mean = float(np.mean(rest_values)) if rest_values else None
            set_median = float(np.median(set_values)) if set_values else None
            rest_median = float(np.median(rest_values)) if rest_values else None
            smd = standardized_mean_difference(set_values, rest_values)
            rows.append(
                {
                    "candidate_set_id": candidate_set_id,
                    "pathway": pathway,
                    "pathway_gene_count": int(pathway_gene_counts.get(pathway, 0)),
                    "available_n": available_n,
                    "missing_n": len(all_case_ids) - available_n,
                    "set_available_n": len(set_values),
                    "rest_available_n": len(rest_values),
                    "set_mean_score": round_float(set_mean),
                    "rest_mean_score": round_float(rest_mean),
                    "set_median_score": round_float(set_median),
                    "rest_median_score": round_float(rest_median),
                    "delta_mean_score": round_float(
                        set_mean - rest_mean
                        if set_mean is not None and rest_mean is not None
                        else None
                    ),
                    "standardized_mean_difference": round_float(smd),
                    "mannwhitney_p_value": round_float(p_value),
                    "q_value": None,
                }
            )
    p_values = [
        1.0 if row["mannwhitney_p_value"] is None else float(row["mannwhitney_p_value"])
        for row in rows
    ]
    for row, q_value in zip(rows, bh_fdr(p_values)):
        row["q_value"] = round_float(q_value)
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row.get("q_value") is None else float(row["q_value"]),
            -abs(
                float(
                    row.get("standardized_mean_difference")
                    if row.get("standardized_mean_difference") is not None
                    else row.get("delta_mean_score")
                    or 0.0
                )
            ),
            str(row.get("pathway") or ""),
            str(row.get("candidate_set_id") or ""),
        ),
    )


def gene_differential_expression_rows(feature_frame, candidate_sets):
    from scipy.stats import mannwhitneyu

    all_case_ids = sorted({case_id for members in candidate_sets.values() for case_id in members})
    rows = []
    for candidate_set_id, members in candidate_sets.items():
        set_ids = [case_id for case_id in all_case_ids if case_id in members and case_id in feature_frame.index]
        rest_ids = [
            case_id
            for case_id in all_case_ids
            if case_id not in members and case_id in feature_frame.index
        ]
        for gene in feature_frame.columns:
            set_values = (
                feature_frame.loc[set_ids, gene]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tolist()
            )
            rest_values = (
                feature_frame.loc[rest_ids, gene]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tolist()
            )
            if not set_values or not rest_values:
                continue
            set_mean = float(np.mean(set_values))
            rest_mean = float(np.mean(rest_values))
            p_value = float(mannwhitneyu(set_values, rest_values, alternative="two-sided").pvalue)
            smd = standardized_mean_difference(set_values, rest_values)
            rows.append(
                {
                    "candidate_set_id": candidate_set_id,
                    "gene": gene,
                    "available_n": len(set_values) + len(rest_values),
                    "set_available_n": len(set_values),
                    "rest_available_n": len(rest_values),
                    "set_mean": round_float(set_mean),
                    "rest_mean": round_float(rest_mean),
                    "log2_fold_change": round_float(set_mean - rest_mean),
                    "standardized_mean_difference": round_float(smd),
                    "mannwhitney_p_value": round_float(p_value),
                    "q_value": None,
                }
            )
    p_values = [
        1.0 if row["mannwhitney_p_value"] is None else float(row["mannwhitney_p_value"])
        for row in rows
    ]
    for row, q_value in zip(rows, bh_fdr(p_values)):
        row["q_value"] = round_float(q_value)
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row.get("q_value") is None else float(row["q_value"]),
            -abs(
                float(
                    row.get("standardized_mean_difference")
                    if row.get("standardized_mean_difference") is not None
                    else row.get("log2_fold_change")
                    or 0.0
                )
            ),
            str(row.get("gene") or ""),
            str(row.get("candidate_set_id") or ""),
        ),
    )


def tool_pathway_enrichment(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    params = tool_parameters(config_dir, "rna")
    min_pathway_overlap = max(int(params.get("min_pathway_overlap", 15) or 15), 1)
    gmt_path = str(params.get("pathway_gene_sets_path", "") or "")
    feature_path = rna_feature_path(patient_states_by_id)
    feature_frame = feature_dataframe(feature_path)
    pathway_to_genes, _ = read_gmt_gene_sets(gmt_path)
    if feature_frame.empty or not pathway_to_genes:
        return empty_pathway_result(
            cluster_id,
            output_root,
            "RNA pathway enrichment inputs are unavailable.",
            "missing_rna_or_gene_sets",
        )

    feature_frame = feature_frame.loc[:, feature_frame.notna().all(axis=0)]
    filtered_pathways = {}
    pathway_gene_counts = {}
    for pathway, genes in pathway_to_genes.items():
        intersected_genes = [gene for gene in genes if gene in feature_frame.columns]
        if len(intersected_genes) >= min_pathway_overlap:
            filtered_pathways[pathway] = intersected_genes
            pathway_gene_counts[pathway] = len(intersected_genes)
    if feature_frame.empty or not filtered_pathways:
        return empty_pathway_result(
            cluster_id,
            output_root,
            "RNA pathway enrichment inputs are unavailable after feature intersection.",
            "missing_intersected_genes",
        )

    candidate_sets = candidate_set_members(all_cluster_states, cluster_state)
    if not candidate_sets:
        return empty_pathway_result(
            cluster_id,
            output_root,
            "Candidate set labels are unavailable.",
            "missing_candidate_sets",
        )

    score_frame = ssgsea_scores(
        feature_frame,
        filtered_pathways,
        min_pathway_overlap,
    )
    rows = pathway_rows(score_frame, pathway_gene_counts, candidate_sets)
    return tool_result(
        tool_name="tool_pathway_enrichment",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary="RNA ssGSEA pathway enrichment table was computed.",
        metrics={"rna_pathway_enrichment": rows},
        decision_metrics={
            "per_set_rna_pathway_enrichment": enrichment_decision_metrics(
                rows, "pathway"
            )
        },
        evidence_hints=[],
        support_level="informational",
        concern_level="none",
        figures={},
    )
