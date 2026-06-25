# Verification Protocol Library

This protocol governs candidate subtype review. A candidate cluster is only a subtype hypothesis until it passes structured verification.

## Evidence Gaps

For each candidate set, track missing evidence in these blocks:

- `set_reliability`: stability evidence gap.
- `multimodal_support`: cross-modal consistency evidence gap.
- `clinical_context`: clinical association evidence gap.
- `biological_support`: molecular, pathway, mutation, or module support evidence gap.
- `confounder_exclusion`: center, scanner, protocol, batch, or site confounding evidence gap.
- `known_label_echo`: known stage, grade, histology, or driver-label echo evidence gap.

## Verification Vector

VerifierReview.verification_vector should summarize the available compact metrics as:

- `N_stability`: bootstrap, split consistency, seed stability, or equivalent set reliability.
- `A_crossmodal`: CT-pathology or other cross-modal agreement.
- `S_survival`: survival or clinical endpoint association, such as log-rank support or Cox gain.
- `M_molecular`: mutation, pathway, RNA, protein, or module enrichment support.
- `C_confound`: confounding penalty from center, scanner, protocol, batch, or site predictability.
- `E_echo`: known-label echo penalty from stage, grade, histology, or common driver labels.
- `U_uncertainty`: uncertainty from small sample size, missingness, variance, or borderline margins.

Use only metrics present in the compact evidence. Mark missing dimensions in evidence_gaps instead of inventing values.

## Acceptance Gate

Set `accept_ready=true` only when all of these are satisfied from compact metric evidence:

- Stability is adequate.
- Multimodal support is adequate when multimodal data are available.
- At least one clinical or biological support block is convincing.
- Confounder penalty is acceptable.
- Known-label echo risk is acceptable.
- Uncertainty is not dominated by small sample size, missingness, or borderline statistical margins.

Acceptance requires `confidence_level="high"`, explicit `reason_codes`, and valid `metric_refs`.

## Router Rules

RouterDecision chooses the next workflow action only after verifier evidence review:

- `continue_review`: choose this only when unexecuted tools can fill specific evidence_gaps and may change the final action.
- `drop`: choose this when evidence is insufficient, unrecoverable under the budget, dominated by confounding or known-label echo, or the set is clearly unstable without a useful merge/split path.
- `split`: choose this when available evidence or deterministic split_plan indicates internal multimodal or molecular heterogeneity.
- `merge`: choose this when the set is too small or unstable and a deterministic merge_plan identifies a neighboring candidate set.

Router must not output accept. Accept is controlled only by the verifier acceptance gate.

## Revision Rules

Reviser executes actions deterministically:

- `drop`: end current set review.
- `split`: create child sets from deterministic split_plan or computable evidence signals; never use LLM-invented member_ids.
- `merge`: create merged set from deterministic merge_plan and record absorbed_set_ids.

Split and merge generated sets must re-enter the review queue as new candidate sets.

## Failure Strategy

- Budget exhaustion without accept becomes `drop` with `insufficient_evidence`; this is not a high-confidence biological negative.
- LLM schema failures, router failures, tool-system failures, or parse failures become `review_unavailable`; keep these separate from evidence insufficiency.
