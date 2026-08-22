# Subtype review protocol

The workflow has exactly three Agents: Verifier, Router, and Reviser. Python is the deterministic runtime and is not an Agent.

The four validation dimensions are parallel:

- `biological_support`
- `cross_modal_consistency`
- `confounder_exclusion`
- `known_label_echo`

At the start of every Router round, Python recomputes the current partition's validation metrics. Verifier calls the requested `@tool`s and writes detailed Evidence Reports. The current round evidence is not reused to skip computation; previous evidence is retained only in provenance history.

Router output must cover every current set exactly once. `need_more_evidence` has priority: if present, every other action is tentative and only all new requests execute. The completed supplement is followed by a fresh full-partition validation and a new Router round.

Accept requires reliable cross-modal or complementary support, reasonable biology, no sufficient technical explanation, no simple known-label echo, and no positive internal heterogeneity or positive weak boundary. Split requires positive internal heterogeneity. Merge requires positive weak boundary from multiple independent modalities. Drop is either explicit exclusion or insufficient evidence after relevant evidence is complete and no structural action is supported.

Router success increments `round` exactly once. Verifier, tools, Reviser, and Python membership execution never increment it. `max_rounds=10` limits successful Router decisions; the tenth decision executes, but no eleventh Router call is allowed. A pending supplement or new partition at that point produces run-level `review_incomplete_due_to_round_budget`.

Split and Merge parents are retained as `superseded_by_split` or `superseded_by_merge` history objects. New current sets are fully revalidated and may be revised again. Normal completion requires every current set to be exactly `accept` or `drop`.
