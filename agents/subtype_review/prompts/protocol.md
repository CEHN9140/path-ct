# Candidate subtype review protocol v2

The review uses four scientific dimensions only:

* `biological_support`
* `cross_modal_consistency`
* `confounder_exclusion`
* `known_label_echo`

Revision intents and exact candidates are generated deterministically by Python. They are not a fifth scientific dimension and never replace proposal-specific evidence. Evidence is scoped to `set_identity`, `split_proposal`, `merge_proposal`, or `partition`. Python derives mandatory requests and exact signatures; proposal evidence must include `proposal_id`. Findings may cite metrics only from that exact dimension, scope, signature, and proposal.

The first Router Split or Merge starts candidate generation only; it does not change membership. Python then exposes candidate-specific evidence requests. Python finishes all requestable evidence for every eligible candidate before declaring a revision ready. If at least one exact candidate is then fully supported, Router emits the same Split or Merge action and Reviser selects only its `plan_id`. Python validates, applies, and records provenance. Split intent has one target, Merge intent has two canonical targets, and neither carries `proposal_id`.

`p > 0.05`, `q > 0.05`, or an inconclusive biology result is not positive evidence for Merge or Drop. Accept requires supporting cross-modal identity, available set-level biological, confounder, and partition-level known-label evidence, no corresponding conflict, and no unresolved supported structural correction. Concordant identity requires at least two supporting original modalities. Complementary identity requires one supporting modality, one distinct moderate modality, and supporting biology. Modality-dominant identity is not sufficient for Accept. Biology may be supporting, mixed, or inconclusive, but conflicting set-level biology vetoes Accept.

After base evidence is complete, Python independently exposes all legal Accept, Drop, Split, Merge, and evidence-request actions. A positive within-set structural signal blocks Accept and Drop only for its exact target; it does not impose a global action order. Mixed confounder association is retained as a caveat, not an Accept veto. Split requires enough patients for two minimum-size children and positive within-set heterogeneity evidence. Merge requires a positive weak-boundary pair signal, not merely two non-Accept endpoints.

An exact Split candidate requires support for its child membership in at least two original modalities, molecular/biology support, and no technical or known-label veto. Imaging-only Split requests biology evidence. Merge requires positive weak-boundary evidence in at least two modalities and no biological distinction veto; a provisionally accepted endpoint remains eligible for this review. Python separates partition-level confounder association, singleton set-level association, and technical invalidation. Drop is permitted for exact technical invalidation or for insufficient identity support after every authorized Split/Merge intent involving that set has been exhausted. The recorded reason distinguishes these cases; insufficient-validation Drop is not a claim of biological invalidity. Drop never removes patients.

When all decision-relevant evidence is exhausted but no action is justified, retain the partition and mark the review unresolved rather than forcing Drop.
