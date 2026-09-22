from __future__ import annotations

from typing import Any


MEMBERSHIP_POSITIVE_CONCLUSION = (
    "Membership-role conclusion: the current exact-set membership is "
    "positively represented in the native multimodal patient geometries."
)
MEMBERSHIP_NOT_POSITIVE_CONCLUSION = (
    "Membership-role conclusion: the current exact-set membership is "
    "not positively represented in the native multimodal patient geometries."
)

ROLE_CONCLUSION_CONTRACTS = {
    ("affinity_geometry_concordance", "set"): {
        "required_conclusions": [
            MEMBERSHIP_POSITIVE_CONCLUSION,
            MEMBERSHIP_NOT_POSITIVE_CONCLUSION,
        ],
        "interpretation_rule": (
            "End dimension_interpretation with exactly one required conclusion. "
            "Use the positive conclusion only when the joint native-view synthesis "
            "affirmatively supports the current exact-set membership as an independent "
            "unit. Weak, limited, mixed, conflicting, non-separating, or materially "
            "contradicted evidence does not constitute positive membership support. "
            "Do not use modality vote counting, unanimity, majority rules, or fixed "
            "numeric thresholds."
        ),
    },
}


DIMENSION_GUIDES = {
    "biological_support": "Does the candidate exhibit a coherent and interpretable biological phenotype relative to the rest of the current partition?",
    "cross_modal_consistency": "Is the current membership or boundary represented in patient geometry, or does a decision-relevant internal subdivision support finer granularity?",
    "confounder_exclusion": "Could measured technical, acquisition, or site factors plausibly account for an important part of the signal or geometry? Association is not causation, and nonsignificance does not establish absence of confounding.",
    "known_label_echo": "How does the partition relate to clinical stratification and established ccRCC taxonomies? TCGA m1-m4 and ClearCode34 overlap the RNA discovery view; this is taxonomy correspondence, not independent validation.",
}

EVIDENCE_ROLE_CONTRACTS = {
    "biological_support": {
        "role": "Characterize whether the candidate has a coherent and interpretable biological identity or phenotype.",
        "request_focus": "Ask whether the candidate shows a coherent and interpretable biological phenotype or molecular program relative to the appropriate comparison population.",
        "does_not_establish": "Biological identity does not by itself establish that current membership, boundary, or granularity deserves independent retention.",
    },
    "cross_modal_consistency": {
        "role": "Evaluate whether current membership, boundary, internal structure, or granularity is represented in the multimodal patient geometry.",
        "request_focus": "Ask one question about either current membership or boundary representation in relevant patient geometries, or—when a split hypothesis could change the decision—internal subdivision. Do not combine membership representation and internal subdivision in one request.",
        "does_not_establish": "Structural or cross-modal evidence does not by itself establish biological meaning or clinical validity.",
    },
    "confounder_exclusion": {
        "role": "Evaluate whether measured technical, acquisition, site, or related factors remain plausible alternative explanations for the observed candidate structure.",
        "request_focus": "Ask whether measured technical, acquisition, or site factors remain plausible alternative explanations at the requested scope. Partition scope concerns the partition as a whole; set scope concerns the target candidate relative to the rest. Do not imply target-specific confounding from partition-level evidence. Do not imply covariate adjustment, causal control, or persistence after adjustment unless the supplied analysis supports it.",
        "does_not_establish": "Absence of a measured confounder association does not by itself positively establish biological identity or an independent boundary.",
    },
    "known_label_echo": {
        "role": "Describe correspondence with clinical stratification or previously reported ccRCC taxonomies.",
        "request_focus": "Ask only how the current partition corresponds to available clinical stratification or previously reported subtype taxonomies.",
        "does_not_establish": "Known-label correspondence or non-correspondence does not by itself establish novelty, validity, independence, or retention.",
    },
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
        "alignment_patient_n": "Number of patients contributing to the full-partition current-label alignment calculation.",
        "geometry_concordance_patient_n": "Number of patients contributing to partition-level GRV geometry concordance.",
        "grv": "Overall shared structure between two modality patient geometries computed from their native distance matrices; not mechanistic agreement or independent validation.",
        "bootstrap_ci95/bootstrap_valid_n": "Paired patient bootstrap uncertainty interval and number of nondegenerate replicates.",
        "permutation_p/permutations": "One-sided permutation evidence under exchangeable patient correspondence between the two geometries; not an action threshold.",
        "comparison": "Defines whether alignment uses full current partition labels or the selected candidate pair.",
        "native_view_membership_alignment": "Current-label representation measured independently in CT, WSI, RNA, and WXS native patient-distance geometries; synthesize the views jointly rather than using vote counting or unanimity.",
        "nearest_competing_set/mean_distance_to_each_other_set": "The competing current set with the smallest target-to-set native distance and the corresponding distances.",
        "mean_within_distance": "Mean per-patient average distance to the patient's own current label, excluding self, corresponding to the within-label quantity used by silhouette interpretation.",
        "mean_between_distance": "Mean per-patient distance to the nearest competing current label, corresponding to the b(i) quantity used by silhouette interpretation; for set scope this is not the pooled distance to all other candidates.",
    },
    ("cross_modal_consistency", "structural_diagnostics"): {
        "geometry_basis": "Candidate-generation consensus geometry used to construct the current candidate partition; it is not external validation or a stability score.",
        "member_n": "Number of patients represented in the structural calculation.",
        "mean_within_affinity": "Average candidate-generation consensus affinity among patients assigned to the same candidate set.",
        "mean_between_affinity": "Average candidate-generation consensus affinity across the current pair boundary.",
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
        "solutions[K].normalized_cut": "Graph normalized-cut cost for a feasible internal split; lower values indicate less cross-child edge mass relative to child volume.",
        "solutions[K].candidate_consensus_separation": "Separation of proposed split labels in the resampled consensus co-assignment geometry used for candidate generation.",
        "solutions[K].child_sizes": "Sizes and balance of children for a feasible split; feasibility is not scientific support.",
        "solutions[K].native_view_separation": "Standard native-distance silhouettes after projecting the proposed split labels into each source modality; these do not replace fused-graph evidence.",
        "current_boundary_normalized_cut": (
            "Normalized-cut cost of the current pair boundary. Lower cost means less "
            "cross-boundary affinity relative to side volume and therefore stronger graph "
            "separation of the current sides; higher cost means more cross-boundary "
            "connectivity and weaker graph support for preserving that boundary. "
            "This is not a threshold or standalone merge rule."
        ),
        "left_volume/right_volume": "Candidate-consensus graph volumes of the two current pair sides.",
        "independent_two_way_spectral_ari": "Agreement between the current pair labels and an independent two-way spectral partition of their union; it is boundary evidence, not an action rule.",
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
        "modality": "Geometry used: candidate-generation consensus for tissue source site and CT native distance for CT acquisition factors.",
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
    "cross_modal_consistency": ["For membership or pair-boundary representation, synthesize CT, WSI, RNA, and WXS native patient-geometry results jointly. Do not use modality vote counting, majority rules, unanimity, or fixed thresholds. Interpret direction, magnitude, negative fraction, within-versus-between distance contrast, nearest competing set, and material cross-modal disagreement. Strong support in one modality does not automatically override material disagreement in others, but disagreement does not automatically invalidate a candidate. State whether the overall native-view pattern provides coherent, weak, conflicting, or limited evidence without making accept, drop, split, or merge recommendations. Global GRV is only partition-level patient-geometry concordance. Internal-subdivision evidence informs split or granularity only; absence of subdivision is neither positive nor negative evidence about independence from neighboring candidates. For set-level subdivision, interpret actual feasible k>=2 solutions and their measurements; partition-level screening does not provide those solutions."],
    "confounder_exclusion": ["Separate association from causation and discuss coverage and factor imbalance. Respect evidence scope: partition-scope association describes a partition-wide technical concern and must not be presented as a candidate-specific confounder association; set-scope association can directly inform candidate-specific alternative explanations. Interpret PERMANOVA together with PERMDISP. A significant PERMANOVA is not proof of technical artifact; nonsignificance does not establish absence of confounding. PERMANOVA, PERMDISP, and continuous distance-regression p-values belong to separate BH families."],
    "known_label_echo": ["Describe taxonomy correspondence conservatively; do not call it external validation."],
}


SCOPE_INTERPRETATIONS = {
    ("structural_diagnostics", "partition"): (
        "Partition structural diagnostics are triage evidence. The internal screen asks whether a current candidate "
        "shows an obvious subdivision signal, and nearest-pair affinity identifies boundaries for possible pair review. "
        "Neither result independently establishes retention, split, or merge."
    ),
    ("structural_diagnostics", "set"): (
        "For set scope, a single-cluster dominant resolution means no obvious internal multi-cluster subdivision signal. "
        "This weighs against an immediate split hypothesis, but is neither positive nor negative evidence about whether "
        "the set is independent from neighboring candidates. A resolution above one raises a possible subdivision hypothesis only; interpret actual "
        "feasible solutions, separation, and child sizes before considering split."
    ),
    ("structural_diagnostics", "pair"): (
        "For pair scope, current-boundary evidence and union-structure evidence answer different questions and must not be conflated. "
        "The current-boundary measurements describe whether the present two-way assignment is represented in candidate-consensus geometry. "
        "Higher agreement with an independent two-way spectral partition and stronger within-side relative to between-side affinity support "
        "representation of the current boundary. The normalized-cut cost has the opposite direction: larger cross-boundary cut cost reflects "
        "more cross-boundary connectivity and must not be described as evidence for preserving distinct units. Union eigengap candidate_k=1 "
        "means that the union is dominated by the single-cluster resolution. This is merge-compatible evidence and must never be interpreted "
        "as evidence for preserving the current two-set boundary. candidate_k>1 indicates possible internal subdivision of the union, but it "
        "supports preserving the current boundary only when that subdivision is also meaningfully aligned with the current two-way structure. "
        "A represented current boundary together with a single-cluster-dominated union is conflicting structural evidence, not automatic evidence "
        "against merge. A weak or conflicting current boundary together with a single-cluster-dominated union is a coherent merge-compatible "
        "pattern, although it is not sufficient by itself to mandate merge. Do not recommend accept, drop, split, or merge from this report."
    ),
    ("affinity_geometry_concordance", "set"): (
        "For set scope, each native patient-distance geometry is evaluated under the full current partition labels. "
        "Interpret nearest_competing_set, mean_distance_to_each_other_set, within-versus-between distances, and "
        "silhouette summaries jointly; the native views are not votes. This evidence bears directly on membership "
        "representation but does not by itself determine accept or drop."
    ),
    ("affinity_geometry_concordance", "pair"): (
        "For pair scope, native patient-distance current-label alignments directly address whether the existing boundary between the two current "
        "candidate sets is represented. The four views provide joint support or disagreement, not votes. Little within-label versus between-label "
        "distinction provides limited geometric support for the current boundary; consistently represented separation "
        "provides positive evidence that the boundary is expressed in the available geometries. This evidence does not "
        "itself determine whether the pair should be merged; integrate it with biological, confounder, and other structural "
        "evidence."
    ),
    ("confounder_association", "partition"): (
        "For partition scope, this analysis tests whether the current multi-set partition as a whole is associated "
        "with measured technical, acquisition, or site factors. An association indicates a partition-wide technical "
        "concern and may motivate candidate-specific follow-up. It does not establish that every current candidate, "
        "or any particular candidate, has a candidate-specific confounder association. Do not promote partition-level "
        "association into a target-specific claim without set-scope evidence."
    ),
    ("confounder_association", "set"): (
        "For set scope, this analysis tests whether the target candidate's membership relative to the rest of the "
        "current partition is associated with measured technical, acquisition, or site factors. This evidence can "
        "directly inform whether those measured factors remain plausible candidate-specific alternative explanations "
        "for the current membership. Association is not causation, and nonsignificance does not establish absence "
        "of confounding."
    ),
}


def guidance_for(dimension: str, aspect: str, scope: str | None = None) -> dict[str, Any]:
    guidance = {
        "dimension": dimension,
        "aspect": aspect,
        "scientific_question": DIMENSION_GUIDES[dimension],
        "evidence_role": EVIDENCE_ROLE_CONTRACTS[dimension],
        "metric_semantics": METRIC_SEMANTICS.get((dimension, aspect), {}),
        "interpretation_requirements": INTERPRETATION_REQUIREMENTS[dimension],
    }
    scope_key = (aspect, scope)
    if scope_key in SCOPE_INTERPRETATIONS:
        guidance["scope_interpretation"] = SCOPE_INTERPRETATIONS[scope_key]
    role_contract = ROLE_CONCLUSION_CONTRACTS.get((aspect, scope))
    if role_contract is not None:
        guidance["role_conclusion_contract"] = role_contract
    return guidance
