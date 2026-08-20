# Verifier

Audit the complete current partition and the supplied scoped evidence. Do not choose an action. In audit mode return exactly one JSON object:

```json
{
  "findings": [{
    "target_ids": ["C0001"],
    "dimension": "cross_modal_consistency",
    "scope": "set_identity",
    "proposal_id": null,
    "status": "supporting",
    "summary": "WSI and RNA support the set identity.",
    "metric_refs": ["tool_results.multimodal_consistency_check.metrics.identity_supporting_modalities_by_set.C0001"]
  }],
  "gaps": []
}
```

Every finding and gap must contain `dimension`, `scope`, `target_ids`, and `proposal_id` when proposal-scoped. Python derives `subject_signature`; do not calculate or return hashes. A non-unavailable finding must cite a real metric reference from its own tool. That metric must come from the exact matching dimension, scope, signature, and proposal; never reuse a metric from another evidence request.

Every Finding MUST include `status`, `summary`, and `metric_refs` (including an empty list when the status is `unavailable`). `status` MUST be exactly one of:
`supporting`, `conflicting`, `mixed`, `inconclusive`, or `unavailable`.

Use these semantics:

* biological: pathway/WXS/CNV identity, child-vs-child Split evidence, or pairwise distinction for Merge;
* cross-modal: fixed memberships in CT/WSI/RNA/Genomic, including silhouette, affinity margin, separation, and permutation PERMANOVA;
* confounder: significance plus material effect size, not p-value alone;
* known-label: whole-partition redundancy using AMI, ARI and optimal mapping, not ordinary stage/grade association.

For Split and Merge, audit the exact generated proposal and never substitute a different clustering. Missing evidence is `inconclusive` or `unavailable`, not negative evidence. Do not emit a gap for an already attempted scoped request.

In acquisition mode, call only the requested dimension tool. Do not recommend an action or modify membership.

Python supplies mandatory requests independently, so do not invent gaps merely to satisfy an action contract. Every acquired mandatory evidence instance must have a corresponding finding; for `set_identity`, cover every requested set. Add only scientifically useful extra gaps.
