from __future__ import annotations

from typing import Any


DIMENSION_GUIDES = {
    "biological_support": "Does the candidate exhibit a coherent and interpretable biological phenotype relative to the rest of the current partition?",
    "cross_modal_consistency": "Are memberships, internal structures, and boundaries coherently represented across the four-view patient geometry?",
    "confounder_exclusion": "Could measured technical, acquisition, or site factors plausibly account for an important part of the signal or geometry? Association is not causation, and nonsignificance does not establish absence of confounding.",
    "known_label_echo": "How does the partition relate to clinical stratification and established ccRCC taxonomies? TCGA m1-m4 and ClearCode34 overlap the RNA discovery view; this is taxonomy correspondence, not independent validation.",
}

METRIC_SEMANTICS = {
    ("biological_support", "rna_pathway_enrichment"): {
        "analysis_status": "Computational estimability only; not_estimable is neither support nor contradiction.",
        "set_n/rest_n": "Sample counts for the set-versus-rest contrast; they affect estimability and stability.",
        "ranking_method": "DESeq2 negative-binomial differential-expression Wald statistic; rankings use the unshrunken Wald statistic, not adjusted p-values or shrunken fold changes.",
        "finite_ranked_gene_n": "Number of finite Wald statistics entering the ranked enrichment analysis.",
        "nes": "Normalized pathway enrichment direction and strength relative to the target-versus-rest Wald-statistic ranking.",
        "fdr_q": "GSEA multiple-testing-adjusted enrichment evidence, not biological importance.",
        "direction": "Enrichment direction relative to the target-versus-rest Wald-statistic ranking.",
        "leading_edge_genes": "Genes contributing most to pathway enrichment; overlap can indicate shared programs.",
        "contrast_interpretation": "When the current partition contains exactly two sets, the two set-versus-rest contrasts are reciprocal views of the same comparison, not independent biological confirmations. When it contains more than two sets, the rest combines multiple candidate sets and may be heterogeneous.",
    },
    ("biological_support", "wxs_mutation_enrichment"): {
        "analysis_status": "Computational estimability only; not_estimable is neither support nor contradiction.",
        "tested_gene_n": "Number of genes in the full nonsynonymous interpretation matrix tested, including observed genes and configured drivers retained even when all patients are wild type.",
        "set_mutated_n/set_n and rest_mutated_n/rest_n": "Raw mutation prevalence and sample support in each comparison group.",
        "effect_status": "Effect estimability; not_estimable_no_events means neither group has a mutation event, so no odds ratio or interval is reported.",
        "odds_ratio": "Target-versus-rest mutation enrichment or depletion effect when estimable; values above one indicate enrichment in the target.",
        "odds_ratio_ci95": "95% log-odds interval using the same Haldane-Anscombe corrected estimator as the odds ratio when a table cell is zero; null when neither group has events.",
        "zero_cell_correction_applied": "A 0.5 Haldane-Anscombe correction was used for the odds ratio and interval; Fisher p-value still uses the original uncorrected table. No correction is used when both groups have zero events.",
        "p_value": "Two-sided Fisher exact evidence from the original 2x2 table.",
        "q_global": "BH-FDR across the complete tested-gene family, including all genes in the interpretation matrix.",
        "q_driver": "BH-FDR within the prespecified ccRCC driver panel; do not conflate it with q_global.",
        "driver_panel_member": "Post-hoc prespecified ccRCC biological annotation, not a discovery feature-selection criterion.",
        "exploratory_top_genes": "Limited display subset only; every interpretation-matrix gene participates in q_global, and the full table is available at artifact_paths.",
    },
    ("cross_modal_consistency", "affinity_geometry_concordance"): {
        "patient_n": "Number of patients contributing to the geometry comparison.",
        "grv": "Overall shared structure between two modality patient geometries computed from their native distance matrices; not mechanistic agreement or independent validation.",
        "bootstrap_ci95/bootstrap_valid_n": "Paired patient bootstrap uncertainty interval and number of nondegenerate replicates.",
        "permutation_p/permutations": "One-sided permutation evidence under exchangeable patient correspondence between the two geometries; not an action threshold.",
        "comparison": "Defines whether alignment uses current partition labels, one target versus the rest, or the selected candidate pair.",
        "current_membership_alignment.silhouette": "How well the already-defined current labels separate in a single modality's native distance geometry; no reclustering is performed.",
        "mean_within_distance/mean_between_distance": "Mean native distance among same-label or different-label patient pairs under the current membership comparison.",
        "two-set partition interpretation": "When the current partition contains exactly two sets, the two set-versus-rest contrasts are reciprocal descriptions of the same comparison and must not be treated as independent confirmations.",
        "multi-set partition interpretation": "When the current partition contains more than two sets, rest is a mixture of other candidate sets; interpret set-versus-rest alignment against that potentially heterogeneous background.",
    },
    ("cross_modal_consistency", "structural_diagnostics"): {
        "member_n": "Number of patients represented in the structural calculation.",
        "mean_within_affinity": "Average fused affinity among patients assigned to the same candidate set.",
        "mean_between_affinity": "Average fused affinity across the current pair boundary.",
        "screen_candidate_k": (
            "Algorithmic eigengap screening resolution. candidate_k=1 means the dominant gap is at the single-cluster resolution, so this screening calculation does not show a dominant multi-cluster subdivision signal. candidate_k>1 identifies a possible multi-cluster resolution for further interpretation, not evidence by itself that the set should be split."
        ),
        "candidate_eigengap/eigengaps": (
            "Magnitude and profile of spectral eigengaps across candidate resolutions. "
            "Interpret the dominant gap's location together with its magnitude. "
            "A large leading gap at k=1 describes dominance of the single-cluster "
            "resolution; do not describe it as evidence of internal multi-cluster "
            "substructure."
        ),
        "nearest_pair_targets/nearest_pair_affinities": "Relatively close candidate pairs and their between-set affinity; a pair-review entry point, not a merge conclusion.",
        "solutions[K].silhouette": "Relative separation of a feasible internal split in fused geometry.",
        "solutions[K].child_sizes": "Sizes and balance of children for a feasible split; feasibility is not scientific support.",
        "boundary_silhouette": "Separation of the current pair membership in fused affinity geometry.",
        "union_eigengap": "Spectral structure of the union of the two sets; it does not itself prescribe merging.",
    },
    ("confounder_exclusion", "confounder_association"): {
        "cramers_v": "Magnitude of categorical-factor association with candidate membership.",
        "epsilon_squared": "Effect magnitude for a continuous factor across candidate groups.",
        "p_value/q_value": "Statistical evidence before and after multiple-testing correction, not causal contribution.",
        "contingency": "Observed factor-level imbalance contributing to the association.",
        "n": "Number of observations contributing to the factor analysis.",
    },
    ("confounder_exclusion", "confounder_representation_effect"): {
        "permanova.r_squared": "Univariable distance-based location/centroid association between one technical factor and geometry; not independent causal variance explained, and factor R-squared values must not be added.",
        "permanova.permutation_p/permanova.q_value": "Permutation evidence for factor-associated geometry location differences.",
        "permdisp.f_statistic/permdisp.permutation_p/permdisp.q_value": "Whether technical-factor levels differ in multivariate dispersion; when dispersion differs, PERMANOVA may reflect both location and dispersion.",
        "distance_regression.r_squared": "Univariable association magnitude between a continuous acquisition factor and CT native-distance geometry.",
        "distance_regression.pseudo_f/permutation_p/q_value": "Distance-based regression statistic and permutation evidence, with its own BH family.",
        "modality": "Geometry used: fused for tissue source site and CT native distance for CT acquisition factors.",
        "n/levels/level_counts": "Sample count and factor-level coverage contributing to the analysis.",
        "interpretation_boundary": "A significant PERMANOVA does not establish technical artifact or causation; a nonsignificant PERMDISP does not establish absence of confounding.",
    },
    ("known_label_echo", "known_label_echo"): {
        "cramers_v/contingency": "Clinical-category association magnitude and the levels contributing to it.",
        "p_value/available_n/missing_n": "Association evidence and clinical-label coverage.",
        "ari/ami": "Agreement and chance-adjusted shared information between hard partitions.",
        "reference_n": "Patient coverage in the known-taxonomy comparison.",
    },
}

INTERPRETATION_REQUIREMENTS = {
    "biological_support": ["Integrate pathway or mutation direction, magnitude, coherence, sample availability, and multiplicity."],
    "cross_modal_consistency": ["Interpret GRV as global patient-geometry similarity, not as current-boundary support. Jointly inspect native-distance current-label alignment; high GRV alone does not show shared subtype boundaries, and low GRV may reflect complementary information rather than a faulty modality. No reclustering is done by this tool. Weak pair separation does not recommend merge, and feasible internal solutions do not recommend split. For set-level subdivision, interpret actual feasible k>=2 solutions and their measurements; partition-level screening does not provide those solutions."],
    "confounder_exclusion": ["Separate association from causation and discuss coverage and factor imbalance. Interpret PERMANOVA together with PERMDISP. A significant PERMANOVA is not proof of technical artifact; a nonsignificant PERMDISP is not proof of no confounding. PERMANOVA, PERMDISP, and continuous distance-regression p-values belong to separate BH families."],
    "known_label_echo": ["Describe taxonomy correspondence conservatively; do not call it external validation."],
}


STRUCTURAL_SCOPE_INTERPRETATIONS = {
    "partition": (
        "Partition structural diagnostics are triage evidence. The internal screen asks whether a current candidate "
        "shows an obvious subdivision signal, and nearest-pair affinity identifies boundaries for possible pair review. "
        "Neither result independently establishes retention, split, or merge."
    ),
    "set": (
        "For set scope, a single-cluster dominant resolution means no obvious internal multi-cluster subdivision signal. "
        "This is evidence against an immediate split hypothesis, not positive evidence that the set is independent from "
        "neighboring candidates. A resolution above one raises a possible subdivision hypothesis only; interpret actual "
        "feasible solutions, separation, and child sizes before considering split."
    ),
    "pair": (
        "For pair scope, current-boundary separation and union structure answer different questions: boundary measurements "
        "describe the current labels, while union structure describes internal subdivision of their union. A union "
        "dominated by the single-cluster resolution lacks a dominant internal subdivision signal. This is structurally "
        "compatible with treating the pair as one candidate and must never be interpreted as evidence against merge, but "
        "is not sufficient by itself to justify merge. A multi-cluster union suggests retained internal structure but "
        "does not establish that it corresponds to the current pair labels. A weak current boundary together with a "
        "single-cluster-dominated union is compatible with removing the boundary; a clearly represented boundary with "
        "meaningful union structure favors preserving distinct units. Conflicting boundary and union findings do not "
        "identify a simple merge solution. Integrate both with the other evidence."
    ),
}


def guidance_for(dimension: str, aspect: str, scope: str | None = None) -> dict[str, Any]:
    guidance = {
        "dimension": dimension,
        "aspect": aspect,
        "scientific_question": DIMENSION_GUIDES[dimension],
        "metric_semantics": METRIC_SEMANTICS.get((dimension, aspect), {}),
        "interpretation_requirements": INTERPRETATION_REQUIREMENTS[dimension],
    }
    if aspect == "structural_diagnostics" and scope is not None:
        guidance["scope_interpretation"] = STRUCTURAL_SCOPE_INTERPRETATIONS[scope]
    return guidance
