# Candidate subtype review protocol v2

The review uses four scientific dimensions only:

* `biological_support`
* `cross_modal_consistency`
* `confounder_exclusion`
* `known_label_echo`

Structure proposals are generated deterministically by Python. They are not a
scientific finding and never replace proposal-specific evidence. Evidence is
scoped to `set_identity`, `split_proposal`, `merge_proposal`, or
`partition`. Python derives mandatory evidence requests and exact evidence
signatures; proposal evidence must include `proposal_id`. Findings may cite
metrics only from that exact dimension, scope, signature, and proposal.

`p > 0.05`, `q > 0.05`, or an inconclusive biology result is not positive
evidence for Merge or Drop. Accept requires supporting cross-modal identity in
at least two original modalities, available set-level confounder evidence,
available partition-level known-label evidence, no corresponding conflict, and
no unresolved supported structural correction. Biology may be supporting,
mixed, or inconclusive.

Split requires an eligible generated proposal, support for the exact child
membership in at least two original modalities, molecular/biology support, and
no technical or known-label veto. Imaging-only Split requests more biology
evidence. Merge requires positive weak-boundary evidence in at least two
modalities and no biological distinction veto. Drop requires positive
confounder or near-identity known-label evidence; it never removes patients.

When all decision-relevant evidence is exhausted but no action is justified,
retain the partition and mark the review unresolved rather than forcing Drop.
