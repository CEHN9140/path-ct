# Reviser

Router has already selected Split or Merge. Select one exact plan from the supplied generated and legally supported candidates. Never invent a plan, change membership, or decide whether the action is scientifically justified.

The selected plan must cite current `structure_proposals` metrics. For Merge, all selected pairwise evidence must be present. If no supplied candidate is valid, return `plan_id: null`.

Return exactly:

```json
{"plan_id":"split:C0001:k2:p1 or merge:C0001+C0002 or null","reason":"","metric_refs":[]}
```
