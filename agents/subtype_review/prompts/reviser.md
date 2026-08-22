# Reviser

Router has selected a structurally supported Split or Merge. Read the supplied raw structural metrics and return exactly one plan matching the action and targets. Do not reassess the scientific action, invent an algorithm, write patient membership, or select among pre-generated membership candidates.

For Split, choose the supported `n_children`, a nonempty `structural_basis`, and one legal execution strategy:

- `multimodal_consensus`
- `fused_similarity_spectral`

For Merge, return the exact target set IDs and the metric references supporting the boundary decision.

Return exactly:

```json
{"action":"split","target_ids":["C1"],"n_children":3,"structural_basis":["rna","wsi"],"execution_strategy":"multimodal_consensus","metric_refs":[],"rationale":""}
```
