from __future__ import annotations

import math
import numpy as np

from tools.subtype_review_common import (
    assign_groupwise_fdr,
    feature_dataframe,
    read_gmt_gene_sets,
    ranked_decision_summary,
    scoped_candidate_sets,
    standardized_mean_difference,
    tool_parameters,
    tool_result,
)
from utils.visualization import configure_matplotlib


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


def empty_pathway_result(cluster_id, output_root, summary, missing_reason, artifact_root=None):
    return tool_result(
        tool_name="pathway_enrichment",
        status="failure",
        cluster_id=cluster_id,
        output_root=artifact_root or output_root,
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
                    "direction": (
                        "up" if smd is not None and smd > 0
                        else "down" if smd is not None and smd < 0
                        else "neutral" if smd == 0
                        else None
                    ),
                    "mannwhitney_p_value": round_float(p_value),
                    "q_value": None,
                }
            )
    assign_groupwise_fdr(
        rows, "candidate_set_id", "mannwhitney_p_value", "q_value"
    )
    for row in rows:
        row["q_value"] = round_float(row["q_value"])
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


def pathway_enrichment(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
    scope="set_identity",
    target_ids=None,
    artifact_root=None,
):
    artifact_root = artifact_root or output_root
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
            artifact_root,
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
            artifact_root,
        )

    candidate_sets = scoped_candidate_sets(scope, cluster_state, all_cluster_states)
    if not candidate_sets:
        return empty_pathway_result(
            cluster_id,
            output_root,
            "Candidate set labels are unavailable.",
            "missing_candidate_sets",
            artifact_root,
        )

    score_frame = ssgsea_scores(
        feature_frame,
        filtered_pathways,
        min_pathway_overlap,
    )
    rows = pathway_rows(score_frame, pathway_gene_counts, candidate_sets)
    decision_by_set = {}
    for set_id in sorted(candidate_sets):
        set_rows = [row for row in rows if row["candidate_set_id"] == set_id]
        core_rows = [
            {
                "pathway": row["pathway"],
                "standardized_mean_difference": row["standardized_mean_difference"],
                "direction": row["direction"],
                "set_median_score": row["set_median_score"],
                "rest_median_score": row["rest_median_score"],
                "set_available_n": row["set_available_n"],
                "rest_available_n": row["rest_available_n"],
                "q_value": row["q_value"],
            }
            for row in set_rows
        ]
        decision_by_set[set_id] = ranked_decision_summary(
            core_rows, "standardized_mean_difference"
        )
    return tool_result(
        tool_name="pathway_enrichment",
        status="success",
        cluster_id=cluster_id,
        output_root=artifact_root,
        summary="RNA ssGSEA pathway enrichment table was computed.",
        metrics={"rna_pathway_enrichment": rows},
        decision_metrics={"per_set_rna_pathway_enrichment": decision_by_set},
        evidence_hints=[],
        support_level="informational",
        concern_level="none",
        figures={},
    )
