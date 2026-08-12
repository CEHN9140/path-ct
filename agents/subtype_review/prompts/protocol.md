# Candidate subtype validation protocol

## Shared invariants

- The complete current partition is the object under review.
- Accept and Drop are provisional states; both keep their members and may be reopened.
- Missing or failed evidence is unknown, never negative evidence.
- CT acquisition confounding can discount CT evidence only; it cannot explain RNA, WXS or WSI evidence.
- Four modalities are all used for discovery and audit, but no fixed modality vote is required.
- Split and Merge must use exact plans produced by the structural tool.
- A parent partition is not validated by evidence from hypothetical children.
- A complete result requires every non-retired set to be accepted.

## Five dimensions

- `biological_support`: coherent RNA/WXS mechanisms, not isolated features.
- `cross_modal_consistency`: compatible CT, WSI, RNA and WXS+CNV genomic evidence for current memberships.
- `confounder_exclusion`: whether CT support can be explained by acquisition metadata.
- `known_label_echo`: whether the complete partition reproduces stage/grade rather than crossing or refining them.
- `structural_adequacy`: internal integrity for Split and external distinctness for Merge.

## Actions

- Request evidence only when the missing dimension can distinguish remaining actions.
- Accept only when identity, independent support, distinctness and structural integrity are all supported.
- Split only when current-set internal heterogeneity is supported and an exact legal plan exists.
- Merge only when the selected combination lacks independent boundaries, has pairwise evidence for every pair, and no coherent mechanism opposes merging.
- Drop is conservative exclusion from the final reliable collection after available evidence and structural alternatives are exhausted; it does not prove biological nonexistence.
- A known-label echo blocks completion. It may be resolved only by evidence-based Split/Merge or independent identity; never optimize the echo metric directly.
