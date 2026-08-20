# Candidate subtype review protocol v2

The review uses four scientific dimensions only:

* `biological_support`
* `cross_modal_consistency`
* `confounder_exclusion`
* `known_label_echo`

Revision intents and exact candidates are generated deterministically by Python. They are not a fifth scientific dimension and never replace proposal-specific evidence. Evidence is scoped to `set_identity`, `split_proposal`, `merge_proposal`, or `partition`. Python derives mandatory requests and exact signatures; proposal evidence must include `proposal_id`. Findings may cite metrics only from that exact dimension, scope, signature, and proposal.

The first Router Split or Merge starts candidate generation only; it does not change membership. Python then exposes candidate-specific evidence requests. After an exact candidate is fully supported, Router emits the same Split or Merge action and Reviser selects only its `plan_id`. Python validates, applies, and records provenance. Split intent has one target, Merge intent has two canonical targets, and neither carries `proposal_id`.

`p > 0.05`, `q > 0.05`, or an inconclusive biology result is not positive evidence for Merge or Drop. Accept requires supporting cross-modal identity in at least two original modalities, available set-level confounder evidence, available partition-level known-label evidence, no corresponding conflict, and no unresolved supported structural correction. Biology may be supporting, mixed, or inconclusive.

Python offers an initial Split only when base evidence is complete, size permits it, no Drop-level evidence exists, the set is not Acceptable, and cross-modal identity fails Accept or set-level biology is conflicting. Biology that is mixed, inconclusive, unavailable, or absent is not a structural motive; partition known-label conflict is not a single-set Split motive. Merge additionally requires an exact active pair and the same motive in at least one member; two Acceptable sets cannot produce a Merge intent.

An exact Split candidate requires support for its child membership in at least two original modalities, molecular/biology support, and no technical or known-label veto. Imaging-only Split requests biology evidence. Merge requires positive weak-boundary evidence in at least two modalities and no biological distinction veto. Drop requires positive set-identity confounder evidence; proposal conflicts veto only their exact revision, while partition-level known-label conflict blocks taxonomy acceptance rather than dropping an individual set. Drop never removes patients.

When all decision-relevant evidence is exhausted but no action is justified, retain the partition and mark the review unresolved rather than forcing Drop.
