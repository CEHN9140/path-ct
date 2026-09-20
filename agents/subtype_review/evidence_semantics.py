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
        "finite_ranked_gene_n": "Number of finite gene statistics entering the ranked enrichment analysis.",
        "nes": "Normalized pathway enrichment direction and relative strength; interpret sign as set versus rest.",
        "fdr_q": "Multiple-testing-adjusted statistical evidence, not biological importance.",
        "direction": "Enrichment direction relative to the set-versus-rest ranking.",
        "leading_edge_genes": "Genes contributing most to pathway enrichment; overlap can indicate shared programs.",
    },
    ("biological_support", "wxs_mutation_enrichment"): {
        "analysis_status": "Computational estimability only; not_estimable is neither support nor contradiction.",
        "set_mutated_n/set_n and rest_mutated_n/rest_n": "Mutation prevalence and sample support in each comparison group.",
        "odds_ratio/odds_ratio_ci95": "Enrichment or depletion direction, magnitude, and interval precision.",
        "p_value/q_value": "Unadjusted and multiple-testing-adjusted evidence; neither alone establishes biological importance.",
        "driver_panel_member": "Post-hoc known-driver annotation, not a feature-selection criterion.",
    },
    ("cross_modal_consistency", "affinity_geometry_concordance"): {
        "patient_n": "Number of patients contributing to the geometry comparison.",
        "pairwise_affinity_geometry_grv": "Overall similarity between two modality patient affinity/distance geometries; not native-feature similarity, mechanistic agreement, or independent validation.",
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
        "r_squared": "Fraction of representation-geometry variation associated with the factor.",
        "permutation_p/q_value": "Permutation-based evidence before and after multiple-testing correction; not causality.",
        "modality": "Representation in which the factor effect was measured.",
        "n/levels": "Sample count and factor-level coverage contributing to the analysis.",
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
    "cross_modal_consistency": ["Distinguish geometry concordance from biological agreement. Weak pair separation does not recommend merge, and feasible internal solutions do not recommend split. For set-level subdivision, interpret actual feasible k>=2 solutions and their measurements; partition-level screening does not provide those solutions."],
    "confounder_exclusion": ["Separate association from causation and discuss coverage and factor imbalance."],
    "known_label_echo": ["Describe taxonomy correspondence conservatively; do not call it external validation."],
}


def guidance_for(dimension: str, aspect: str) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "aspect": aspect,
        "scientific_question": DIMENSION_GUIDES[dimension],
        "metric_semantics": METRIC_SEMANTICS.get((dimension, aspect), {}),
        "interpretation_requirements": INTERPRETATION_REQUIREMENTS[dimension],
    }
