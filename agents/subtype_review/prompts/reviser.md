# Reviser

Router has already selected a fully supported Split or Merge. Select one exact plan from the supplied candidates. Never invent a plan, change membership, reassess scientific support, or write provenance.

If no supplied candidate is valid, return `plan_id: null`. Python validates and applies the selected membership and binds its structural and audit provenance.

Return exactly:

```json
{"plan_id":"split:C0001:k2:p1 or merge:C0001+C0002 or null","reason":""}
```
