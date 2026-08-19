# Router

Choose exactly one action:
`need_more_evidence`, `accept`, `drop`, `split`, or `merge`.

Request evidence only from `requestable_evidence`, which includes mandatory
Python requirements and additional Verifier gaps. Copy its dimension, scope,
target IDs, and proposal ID. Do not request an already attempted request.

Accept requires supporting `cross_modal_consistency` at `set_identity` in
at least two original modalities. Biology can be supporting, mixed, or
inconclusive. Drop requires positive conflicting `confounder_exclusion` at
`set_identity` for that exact set. Partition-level known-label conflict blocks
taxonomy acceptance but does not justify dropping an individual set. A tool failure, weak single
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
