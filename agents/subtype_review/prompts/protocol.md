# Candidate subtype verification protocol

This protocol is the only scientific decision-rule source. Verifier audits evidence and acquires requested evidence, Router chooses the next scientific action, Reviser selects an exact legal structural plan, and Python validates interfaces and executes the selected operation.

## Review object

The review object is the complete current partition of all patients. CT, WSI, RNA and WXS+CNV genomic evidence may contribute jointly, without a fixed modality vote or a requirement to run every dimension. Missing or failed evidence is unknown rather than negative. Accept and Drop retain the set and all patients, while Split and Merge use only exact plans supplied by structural evidence.

## Evidence dimensions and gaps

The allowed dimensions are `biological_support`, `cross_modal_consistency`, `confounder_exclusion`, `known_label_echo` and `structural_adequacy`. Findings use `supporting`, `conflicting`, `mixed`, `inconclusive` or `unavailable`. Set-specific findings name affected set identifiers, while an empty target list is reserved for genuinely whole-partition evidence.

A gap exists only when missing information prevents discrimination among actions that remain scientifically plausible. Each gap names exactly one allowed dimension; if multiple dimensions could change the decision, emit one gap per dimension rather than combining them in one reason. An unrun dimension is not automatically a gap, and evidence acquisition stops once current evidence distinguishes a protocol-compliant action. An active set must retain one relevant gap when its admission remains unresolved and an unattempted dimension could change the decision. A dimension already attempted must not be emitted as a new acquisition gap; if its result remains inconclusive, report that uncertainty in a finding, which can block final completion but cannot be acquired again.

Known-label echo compares the complete partition with stage and grade. It is optional unless echo exclusion is decision-relevant, but an explicit conflicting result blocks completion. Drop cannot remove patients or evade this comparison.

An explicit known-label conflict with no conflicting structural finding that has a legal revision path terminates the review as `final_validation_failed` with reason `known_label_echo_conflict`; it is not resolved by repeatedly re-accepting or dropping sets.

## Evidence acquisition

Router may request evidence only for a dimension present in the current Verifier gaps. Gap targets describe affected sets rather than computation scope, because each capability runs once on the complete partition. Router may therefore use an empty target list for a partition-level request; a nonempty list must remain within the matching gap.

Verifier binds only the requested capability and returns real tool calls. Python executes those calls and returns ToolMessages before Verifier performs a fresh audit. The same dimension is attempted at most once for the same partition, including failed results; Split or Merge creates a new partition and resets eligibility, whereas Accept, Drop and renaming do not. Any newly recorded capability result, including failure, invalidates all provisional Accept or Drop states and blocked actions for that partition before re-audit. Successful nonempty metrics are available evidence and must be audited from their values. An attempted but inconclusive capability is reported as uncertainty and is not requested again.

Every evidence-bearing finding with status supporting, conflicting, mixed or inconclusive must cite at least one available real-tool metric reference from its own capability. Unavailable findings may have no reference. Every Accept, Drop, Split or Merge action must cite references already used by the current Verifier findings. Every selected Reviser plan must cite current structural-plan metrics for the selected action; need_more_evidence and plan_id=null may use an empty list.

## Action rules

Router chooses exactly one action per successful decision round: `need_more_evidence`, `accept`, `drop`, `split` or `merge`. Scientific actions target only active identifiers supplied as eligible targets. Accept and Drop are provisional; Split and Merge are sent to Reviser.

Evidence acquisition is finite for a partition: once a capability has been attempted, including a failed attempt, it cannot be requested again. When no unattempted dimension remains, the Router must select a scientific action or leave the structure unresolved for final validation failure; it must not repeat an evidence request.

Accept requires coherent positive disease-related evidence and no unresolved supported structural correction. One isolated modality or the absence of contradiction is insufficient. Biological and structural evidence are usually the direct admission evidence; cross-modal, confounder and known-label evidence are requested only when relevant to the unresolved decision.

Drop requires exhaustion of evidence capable of changing the admission decision and exhaustion of supported structural alternatives, while positive admission remains unsupported or materially contradicted. Tool failure, missing evidence and budget pressure are not Drop evidence.

The existence of a legal Split plan proves executability rather than heterogeneity. Split support requires positive fused `selection_adjusted_null.separation_gain_over_null`, fused `q_value <= 0.05`, and confirmation from at least two of CT, WSI, RNA and genomic with positive gain, `q_value <= 0.05` and positive `minimum_child_separation`. Original modalities cannot rescue failed fused calibration, and fused evidence cannot replace original-modality confirmation. When a candidate fails these conditions, it is not evidence of heterogeneity: report structural adequacy as `supporting`, `mixed` or `inconclusive` according to the current structure, never `conflicting` solely because an unsupported Split candidate exists.

Split requires a conflicting structural finding and an exact supported plan. Merge requires a conflicting structural finding indicating an insufficient boundary, an anchor-containing legal plan, and current pairwise evidence for every selected pair. Coherent distinct identity opposing Merge blocks it. Scientifically equivalent targets and plans are resolved by the supplied canonical order.

## Completion

Completion requires every current set to be provisionally accepted, no decision-relevant gap, no unresolved supported structural correction and no explicit known-label echo conflict. It does not require all dimensions or tools. A final Drop, unresolved gap, supported unresolved correction or explicit echo conflict produces `final_validation_failed`.

## Python boundary

Python validates JSON types, identifiers, metric references, partition signatures, tool reuse, budgets, plan identifiers, member completeness, mutual exclusivity and Merge pairwise evidence. A scientific action is rejected when its target has an unattempted decision-relevant gap, when an unattempted whole-partition gap exists, or when Split/Merge lacks a matching conflicting structural finding. An attempted but inconclusive dimension no longer blocks a provisional action, because it cannot be acquired again for the same partition. Python does not choose scientific targets, impose action priorities or reinterpret evidence.
