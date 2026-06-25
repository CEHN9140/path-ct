from __future__ import annotations

import math

from tools.subtype_review_common import (
    bh_fdr,
    fisher_exact_result,
    member_case_ids,
    read_case_feature_table,
    read_gmt_gene_sets,
    tool_parameters,
    tool_result,
)

def wxs_feature_path(patient_states_by_id):
    for patient_state in patient_states_by_id.values():
        omics = dict(patient_state.get("omics_evidence", {}) or {})
        path = str(
            omics.get("wxs_full_feature_path", "")
            or omics.get("wxs_feature_path", "")
            or ""
        )
        if path:
            return path
    return ""


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


def mutation_values(table, case_ids, genes):
    values = []
    for case_id in case_ids:
        case_values = [
            float(table.get(case_id, {}).get(gene, float("nan"))) for gene in genes
        ]
        case_values = [value for value in case_values if math.isfinite(value)]
        if case_values:
            values.append(float(any(value > 0 for value in case_values)))
    return values


def enrichment_row_counts(values):
    altered = int(sum(value > 0 for value in values))
    total = len(values)
    frequency = altered / float(total) if total else None
    return altered, total, frequency


def apply_q_values(rows):
    p_values = [1.0 if row.get("p_value") is None else float(row["p_value"]) for row in rows]
    for row, q_value in zip(rows, bh_fdr(p_values)):
        row["q_value"] = round_float(q_value)
    return rows


def sort_enrichment_rows(rows, effect_key="delta_frequency"):
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row.get("q_value") is None else float(row["q_value"]),
            -abs(float(row.get(effect_key) or 0.0)),
            str(row.get("gene") or row.get("pathway") or ""),
            str(row.get("candidate_set_id") or ""),
        ),
    )


def wxs_gene_rows(feature_names, table, candidate_sets):
    all_case_ids = sorted({case_id for members in candidate_sets.values() for case_id in members})
    rows = []
    for candidate_set_id, members in candidate_sets.items():
        set_ids = [case_id for case_id in all_case_ids if case_id in members]
        rest_ids = [case_id for case_id in all_case_ids if case_id not in members]
        for gene in feature_names:
            set_values = mutation_values(table, set_ids, [gene])
            rest_values = mutation_values(table, rest_ids, [gene])
            set_mutated, set_total, set_frequency = enrichment_row_counts(set_values)
            rest_mutated, rest_total, rest_frequency = enrichment_row_counts(rest_values)
            available_n = set_total + rest_total
            odds_ratio, p_value = (None, None)
            if set_total and rest_total:
                odds_ratio, p_value = fisher_exact_result(
                    set_mutated,
                    set_total - set_mutated,
                    rest_mutated,
                    rest_total - rest_mutated,
                )
            rows.append(
                {
                    "candidate_set_id": candidate_set_id,
                    "gene": gene,
                    "available_n": available_n,
                    "missing_n": len(all_case_ids) - available_n,
                    "set_mutated_n": set_mutated,
                    "set_total_n": set_total,
                    "rest_mutated_n": rest_mutated,
                    "rest_total_n": rest_total,
                    "set_frequency": round_float(set_frequency),
                    "rest_frequency": round_float(rest_frequency),
                    "delta_frequency": round_float(
                        set_frequency - rest_frequency
                        if set_frequency is not None and rest_frequency is not None
                        else None
                    ),
                    "odds_ratio": round_float(odds_ratio),
                    "p_value": round_float(p_value),
                }
            )
    return sort_enrichment_rows(apply_q_values(rows))


def wxs_pathway_rows(
    feature_names,
    table,
    candidate_sets,
    gmt_path,
    gene_set_collection="",
    evidence_role="",
):
    pathway_to_genes, _ = read_gmt_gene_sets(gmt_path)
    feature_set = set(feature_names)
    all_case_ids = sorted({case_id for members in candidate_sets.values() for case_id in members})
    rows = []
    for candidate_set_id, members in candidate_sets.items():
        set_ids = [case_id for case_id in all_case_ids if case_id in members]
        rest_ids = [case_id for case_id in all_case_ids if case_id not in members]
        for pathway, genes in pathway_to_genes.items():
            pathway_genes = [gene for gene in genes if gene in feature_set]
            if not pathway_genes:
                continue
            set_values = mutation_values(table, set_ids, pathway_genes)
            rest_values = mutation_values(table, rest_ids, pathway_genes)
            set_hit, set_total, set_frequency = enrichment_row_counts(set_values)
            rest_hit, rest_total, rest_frequency = enrichment_row_counts(rest_values)
            available_n = set_total + rest_total
            odds_ratio, p_value = (None, None)
            if set_total and rest_total:
                odds_ratio, p_value = fisher_exact_result(
                    set_hit,
                    set_total - set_hit,
                    rest_hit,
                    rest_total - rest_hit,
                )
            rows.append(
                {
                    "candidate_set_id": candidate_set_id,
                    "pathway": pathway,
                    "gene_set_collection": gene_set_collection,
                    "evidence_role": evidence_role,
                    "pathway_gene_count": len(pathway_genes),
                    "available_n": available_n,
                    "missing_n": len(all_case_ids) - available_n,
                    "set_hit_n": set_hit,
                    "set_total_n": set_total,
                    "rest_hit_n": rest_hit,
                    "rest_total_n": rest_total,
                    "set_frequency": round_float(set_frequency),
                    "rest_frequency": round_float(rest_frequency),
                    "delta_frequency": round_float(
                        set_frequency - rest_frequency
                        if set_frequency is not None and rest_frequency is not None
                        else None
                    ),
                    "odds_ratio": round_float(odds_ratio),
                    "p_value": round_float(p_value),
                }
            )
    return sort_enrichment_rows(apply_q_values(rows))


def mutation_pathway_gene_set_specs(mutation_params):
    raw_specs = mutation_params.get("pathway_gene_sets", []) or []
    if not raw_specs and mutation_params.get("pathway_gene_sets_path"):
        raw_specs = [
            {
                "name": str(mutation_params.get("pathway_gene_sets_name", "mutation_pathway") or "mutation_pathway"),
                "role": str(mutation_params.get("pathway_gene_sets_role", "primary") or "primary"),
                "path": str(mutation_params.get("pathway_gene_sets_path", "") or ""),
            }
        ]
    specs = []
    if isinstance(raw_specs, dict):
        items = []
        for name, value in raw_specs.items():
            spec = dict(value or {}) if isinstance(value, dict) else {"path": str(value)}
            spec.setdefault("name", str(name))
            items.append(spec)
    else:
        items = list(raw_specs)
    for item in items:
        spec = dict(item or {}) if isinstance(item, dict) else {"path": str(item)}
        path = str(spec.get("path", "") or "")
        if path:
            specs.append(
                {
                    "name": str(spec.get("name", "") or "mutation_pathway"),
                    "role": str(spec.get("role", "") or "primary"),
                    "path": path,
                }
            )
    return specs


def empty_mutation_result(cluster_id, output_root, summary, missing_reason):
    return tool_result(
        tool_name="tool_mutation_enrichment",
        status="missing",
        cluster_id=cluster_id,
        output_root=output_root,
        summary=summary,
        metrics={"wxs_gene_enrichment": [], "wxs_pathway_enrichment": []},
        missing_reason=missing_reason,
        support_level="none",
        concern_level="moderate",
        figures={},
    )


def tool_mutation_enrichment(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
):
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    mutation_params = tool_parameters(config_dir, "mutation")
    feature_names, table = read_case_feature_table(
        wxs_feature_path(patient_states_by_id)
    )
    if not feature_names or not table:
        return empty_mutation_result(
            cluster_id,
            output_root,
            "WXS feature table is unavailable.",
            "missing_wxs_feature_table",
        )

    candidate_sets = candidate_set_members(all_cluster_states, cluster_state)
    if not candidate_sets:
        return empty_mutation_result(
            cluster_id,
            output_root,
            "Candidate set labels are unavailable.",
            "missing_candidate_sets",
        )

    gene_rows = wxs_gene_rows(feature_names, table, candidate_sets)
    pathway_rows = []
    warnings = []
    gene_set_specs = mutation_pathway_gene_set_specs(mutation_params)
    if not gene_set_specs:
        warnings.append("No mutation pathway gene-set files are configured.")
    for spec in gene_set_specs:
        rows = wxs_pathway_rows(
            feature_names,
            table,
            candidate_sets,
            spec["path"],
            gene_set_collection=spec["name"],
            evidence_role=spec["role"],
        )
        if rows:
            pathway_rows.extend(rows)
        else:
            warnings.append(f"No mutation pathway rows were computed for {spec['name']}: {spec['path']}")
    return tool_result(
        tool_name="tool_mutation_enrichment",
        status="success",
        cluster_id=cluster_id,
        output_root=output_root,
        summary="WXS gene and pathway enrichment tables were computed.",
        metrics={
            "wxs_gene_enrichment": gene_rows,
            "wxs_pathway_enrichment": pathway_rows,
        },
        evidence_hints=[],
        warnings=warnings,
        support_level="informational",
        concern_level="none",
        figures={},
    )
