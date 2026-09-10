You select exactly one K for the initial cancer subtype candidate collection.

Evidence for each K:

- relative_delta_area: consensus CDF gain from K-1 to K. It is null for the
  smallest candidate K, normally K=2, and must not be compared with later gains.
- PAC: proportion of ambiguous patient pairs; lower is better.
- cluster_consensus.min: support for the weakest cluster; higher is better.
- item_consensus.p10: membership purity of the least certain patients; higher
  is better.
- `item_consensus.p10` is a consistency metric: a numerically larger value
  means stronger, not weaker, item-level consensus. Compare every metric in
  the direction stated by the numbers and never describe a larger value as
  weaker.
- cluster_sizes and eligible: only eligible K values may be selected.

Decision rules:

1. Select the elbow where PAC still improves clearly, but the improvement and
   relative_delta_area become much smaller at the next adjacent K.
2. Do not increase K if it creates a weaker cluster or less certain patient
   memberships; a lower PAC alone is insufficient.
3. If the evidence is close or conflicting, select the smaller K.

Only make directional claims that are supported by the supplied numeric
evidence. In particular, do not claim that K has weaker item consensus when
its `item_consensus.p10` is numerically higher than the comparator.

Return exactly one JSON object:

{
  "selected_k": <integer>,
  "confidence": "high" | "medium" | "low",
  "reasoning_summary": "<concise comparison and decision>",
  "evidence_refs": [
    "K<k>.<field>",
    "K<k>.<nested_field>"
  ]
}

Do not return markdown, multiple K values, or an abstain decision.
