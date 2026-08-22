# Router

Read the current partition, Evidence Reports, raw-evidence inventory, and the four validation rules. Return one JSON object with an `actions` array. Every current set may occur in at most one action in this round.

Allowed actions:

- `need_more_evidence`: include one or more exact requests. This action must be selected alone. Request only dimensions whose current evidence is absent or insufficient.
- `accept`: the set has sufficient independent, interpretable evidence and no unresolved structural concern.
- `drop`: use only for a clearly invalid set or when relevant evidence is closed but the set does not meet Accept.
- `split`: exactly one target. Use only when the reports and structural metrics show positive within-set heterogeneity.
- `merge`: two or more targets. Use only when multiple independent modalities show a positive weak boundary.

Do not infer Split from weak evidence, and do not infer Merge merely because two sets are not acceptable. Do not call tools, compute metrics, edit membership, or write provenance. Python validates target validity, evidence-cache identity, action overlap, structural triggers, and plan execution.

Return exactly:

```json
{"actions":[{"action":"need_more_evidence","target_ids":["C1"],"requests":[{"dimension":"cross_modal_consistency","scope":"set_identity","target_ids":["C1"]}],"reason":"The current reports do not establish cross-modal identity."}]}
```
