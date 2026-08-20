# Candidate subtype review protocol v2

The review uses four scientific dimensions only:

* `biological_support`
* `cross_modal_consistency`
* `confounder_exclusion`
* `known_label_echo`

Revision intents and exact candidates are generated deterministically by Python. They are not a fifth scientific dimension and never replace proposal-specific evidence. Evidence is scoped to `set_identity`, `split_proposal`, `merge_proposal`, or `partition`. Python derives mandatory requests and exact signatures; proposal evidence must include `proposal_id`. Findings may cite metrics only from that exact dimension, scope, signature, and proposal.

The first Router Split or Merge starts candidate generation only; it does not change membership. Python then exposes candidate-specific evidence requests. Python finishes all requestable evidence for every eligible candidate before declaring a revision ready. If at least one exact candidate is then fully supported, Router emits the same Split or Merge action and Reviser selects only its `plan_id`. Python validates, applies, and records provenance. Split intent has one target, Merge intent has two canonical targets, and neither carries `proposal_id`.

`p > 0.05`, `q > 0.05`, or an inconclusive biology result is not positive evidence for Merge or Drop. Accept requires supporting cross-modal identity in at least two original modalities, available set-level confounder evidence, available partition-level known-label evidence, no corresponding conflict, and no unresolved supported structural correction. Biology may be supporting, mixed, or inconclusive.

After base evidence is complete, Python exhaustively offers every unblocked structural review allowed by the current partition. Split requires an active set large enough for two minimum-size children and no Drop-level confounder evidence. Merge covers every canonical pair of active sets for which neither set has Drop-level evidence. Accept is unavailable while any unblocked Split or Merge intent involving that set remains unreviewed. These initial intents only generate exact candidates; they are not evidence that membership should change.

An exact Split candidate requires support for its child membership in at least two original modalities, molecular/biology support, and no technical or known-label veto. Imaging-only Split requests biology evidence. Merge requires positive weak-boundary evidence in at least two modalities and no biological distinction veto. Python separates partition-level confounder association, singleton set-level association, and technical invalidation. Association can block Accept but cannot justify Drop; Drop requires deterministic technical invalidation for that exact singleton set. Proposal conflicts veto only their exact revision, while partition-level conflicts block taxonomy acceptance rather than dropping an individual set. Drop never removes patients.

When all decision-relevant evidence is exhausted but no action is justified, retain the partition and mark the review unresolved rather than forcing Drop.
