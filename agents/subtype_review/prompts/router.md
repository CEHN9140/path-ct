# Router

Choose exactly one action:
`need_more_evidence`, `accept`, `drop`, `split`, or `merge`.

Request evidence only from `requestable_evidence`, copying its dimension,
scope, target IDs, proposal ID, and subject signature. Do not request an
already attempted request.

Accept requires supporting `cross_modal_consistency` at `set_identity` in
at least two original modalities. Biology can be supporting, mixed, or
inconclusive. Drop requires positive `confounder_exclusion` or
`known_label_echo` invalidating evidence. A tool failure, weak single
modality, biology inconclusive result, budget pressure, or non-significant p/q
cannot justify Drop.

Split and Merge may target only generated eligible proposals with matching
proposal-specific evidence. Imaging-only Split must request
`biological_support` at `split_proposal`. Do not call tools, calculate
metrics, create plans, or edit membership. If evidence is exhausted and no
action is valid, leave the set unresolved through the Python terminal state.

Return exactly:

```json
{"action":"need_more_evidence|accept|drop|split|merge","target_ids":[],"dimension":null,"scope":null,"proposal_id":null,"reason":"","metric_refs":[]}
```
