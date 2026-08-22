# Subtype review protocol

The review has three Agents: Verifier, Router, and Reviser. Python is the deterministic runtime; it manages state, evidence signatures, tool binding, contract validation, and membership execution.

The four scientific dimensions are平级:

- `biological_support`
- `cross_modal_consistency`
- `confounder_exclusion`
- `known_label_echo`

There is no base-evidence tier. Router requests only the evidence needed by the current decision. Raw tool output is cached by dimension, scope, current partition, and target membership. A structural change invalidates current evidence but preserves it in history.

Verifier first acquires every requested tool in one turn, then writes detailed Evidence Reports. Reports contain observations, statistical interpretation, medical interpretation, limitations, and metric references. Verifier never selects an action.

Router gives each current set at most one action per round. `need_more_evidence` must be the only selected action in that round. Accept requires sufficient independent evidence, no strong technical explanation, no known-label near-identity, and no unresolved structural signal. Split requires positive internal heterogeneity. Merge requires positive weak-boundary evidence from multiple independent modalities. Drop is allowed for explicit technical invalidation or for closed relevant evidence that still fails the Accept rule.

Reviser runs only after Router selects Split or Merge. It reads raw structural metrics, returns one plan, and never writes patient membership. Python executes a fixed spectral strategy, verifies complete membership coverage, and creates a new partition. Superseded parents are retained as history and are not counted as Drop.

The current partition terminates only when every current set is Accept or Drop. Tool/API/data failures produce run-level `review_unavailable`. If the round budget ends before the evidence and actions close, the run is `review_incomplete_due_to_round_budget`; it is not a scientific negative result.
