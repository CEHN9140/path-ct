# Candidate subtype verification protocol

This document is the only scientific rule source. Verifier audits or acquires evidence, Router chooses the next action, Reviser selects a legal structural plan, and Python checks interfaces and executes already-decided operations.

## Review object

The scientific object is the complete current partition of all patients. CT, WSI, RNA and WXS+CNV genomic evidence may contribute together; no fixed modality vote and no mechanical requirement for every modality or dimension applies. Missing or failed evidence is unknown, not negative evidence. Accept and Drop retain the set and all patients. Split and Merge use only exact plans supplied by structural evidence.

## Evidence audit

The allowed dimensions are `biological_support`, `cross_modal_consistency`, `confounder_exclusion`, `known_label_echo` and `structural_adequacy`. A finding uses `supporting`, `conflicting`, `mixed`, `inconclusive` or `unavailable`. Findings about set-specific admission evidence name the affected set identifiers; an empty target list is reserved for genuinely whole-partition evidence. A gap is reported only when missing information prevents discrimination among actions that remain plausible. Absence of a dimension is not itself a gap, and the audit must not create one gap per unrun capability. Once current findings distinguish a protocol-compliant action, do not request unrelated evidence merely to make the audit comprehensive. Conversely, an active set without sufficient admission evidence cannot silently lose its decision-relevant gap while an unattempted relevant dimension could still distinguish Accept from Drop.

Known-label echo is a whole-partition comparison with stage and grade. It is not required to be computed in every review, but an explicit conflicting echo blocks final completion. A Drop cannot remove patients or evade this audit.

## Evidence acquisition

Router may return `need_more_evidence` only for a dimension named in a current Verifier gap. Gap `target_ids` identify the sets affected by missing evidence; they do not constrain the computation scope. An empty Router `target_ids` list requests the validation capability once for the complete current partition and is valid even when the matching gap names affected sets. A nonempty Router target list must stay within the matching gap. Verifier then selects the mounted tool for that dimension and returns real tool calls. Python executes the calls on the complete partition and returns ToolMessages to Verifier for a fresh audit.

The same dimension is attempted at most once for the same patient partition, including failed or unavailable tool results. A Split or Merge creates a new patient partition and resets eligibility. Accept, Drop and renaming do not reset it. No tool is run for coverage, reassurance or budget consumption. A successful inventory entry with nonempty metrics is available evidence and must be audited from its supplied values; it must never be described as missing merely because raw clinical fields are not separately present in the prompt. If an attempted dimension remains scientifically inconclusive, report that factual finding and explain the residual uncertainty without requesting the same capability again.

## Router actions

Router chooses exactly one action per decision round: `need_more_evidence`, `accept`, `drop`, `split` or `merge`. `need_more_evidence` is resolved by the Verifier tool loop. Accept and Drop are provisional decisions that retain the set. Split and Merge are structural decisions and are passed to Reviser. Scientific actions may target only identifiers explicitly listed in `eligible_action_target_ids`; provisionally decided sets remain visible as partition context but are not eligible again unless a later conflicting audit reactivates them.

Accept requires coherent positive disease-related evidence and no unresolved supported structural correction. One isolated modality or absence of a contradiction is insufficient. Biological and structural questions are usually the most direct admission questions. Cross-modal evidence is acquired when existing identity or boundary evidence cannot distinguish the remaining actions. Confounder evidence is acquired when CT evidence is material to the contemplated decision. Known-label echo is acquired only when echo exclusion is decision-relevant to the current complete partition. These are relevance rules, not a mandatory sequence. Drop requires exhausted decision-changing evidence and structural alternatives while positive admission remains unsupported or materially contradicted. Tool failure, missing evidence and budget pressure are not Drop evidence.

The presence of a legal `split_candidates` entry proves only executability and never proves heterogeneity. Split separation is directional: larger positive `normalized_separation` and `minimum_child_separation` values mean stronger proposed child structure; lower values never support Split. A proposed Split has statistical support only when its `selection_adjusted_null.separation_gain_over_null` is positive and its multiplicity-adjusted `q_value` is at most 0.05, and at least two of CT, WSI, RNA and genomic have positive `separation_gain_over_null`, `q_value <= 0.05` and positive `minimum_child_separation`. Otherwise the plan remains legal but unsupported and cannot produce `structural_adequacy=conflicting` or a Router Split action. Original-modality confirmation cannot rescue a failed fused calibration, and fused evidence alone cannot rescue absent original-modality confirmation.

Split requires a conflicting structural finding identifying statistically supported within-set heterogeneity and an exact legal Split plan. Merge requires a conflicting structural finding identifying an insufficient boundary, an anchor-containing legal Merge plan and current pairwise evidence for every selected pair. A coherent distinct identity opposing Merge blocks it. When scientifically equivalent supported targets or plans remain, use the canonical order supplied in the request.

## Completion

The complete endpoint is reached only when every current set is provisionally accepted, no decision-relevant gap remains, no supported legal structural correction remains, and no explicit known-label echo conflict is present. This endpoint does not require all five dimensions or all tools. A final Drop, unresolved gap, unresolved supported structural correction or explicit echo conflict produces `final_validation_failed`.

## Python boundary

Python validates JSON types, identifiers, metric references, current partition signatures, tool reuse, budgets, plan identifiers, member completeness, mutual exclusivity and Merge pairwise evidence. Python does not choose targets, impose action priorities or reinterpret scientific findings.
