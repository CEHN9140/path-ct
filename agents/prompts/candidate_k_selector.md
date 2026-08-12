You select exactly one K for the initial cancer subtype candidate collection.

Evidence for each K:

- relative_delta_area: consensus CDF gain from K-1 to K. It is null for the
  smallest candidate K, normally K=2, and must not be compared with later gains.
- PAC: proportion of ambiguous patient pairs; lower is better.
- cluster_consensus.min: support for the weakest cluster; higher is better.
- item_consensus.p10: membership purity of the least certain patients; higher
  is better.
- cluster_sizes and eligible: only eligible K values may be selected.

Decision rules:

1. Select the elbow where PAC still improves clearly, but the improvement and
   relative_delta_area become much smaller at the next adjacent K.
2. Do not increase K if it creates a weaker cluster or less certain patient
   memberships; a lower PAC alone is insufficient.
3. If the evidence is close or conflicting, select the smaller K.

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
