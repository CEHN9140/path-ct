# Router

Read the current partition and this round's complete Evidence Reports. Return one JSON object with an `actions` array.

Every current set must occur exactly once across all action targets. Merge may contain multiple targets, but no target may occur in another action. If any action is `need_more_evidence`, all other actions are tentative; Python will execute only the requests and revalidate the whole partition in a new round.

Allowed actions:

- `need_more_evidence`: request only a new analysis that can distinguish scientific actions. Do not repeat a successful request with the same partition, targets, dimension, scope, and analysis.
- `accept`: reliable cross-modal or complementary support, biology, no sufficient technical explanation, no known-label echo, and no positive structural signal.
- `drop`: explicit exclusion or complete relevant evidence that still fails Accept.
- `split`: only positive internal heterogeneity supported by structural diagnostics.
- `merge`: only positive weak-boundary evidence from multiple independent modalities.

Do not treat inability to Accept as evidence for Split or Merge. Do not call tools, compute metrics, edit membership, or write provenance.

Return exactly one valid JSON object:

```json
{"actions":[{"action":"drop","target_ids":["C1"],"reason":"The complete evidence does not support acceptance."}]}
```
