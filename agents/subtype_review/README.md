# Subtype review

`graph.py` owns the three node implementations and graph wiring:

- `router_node`: build the decision payload, validate evidence/action coverage,
  retry invalid plans, record history and choose the next step.
- `verifier_node`: compute eligible tools, execute selected calls, validate reports,
  attach provenance and store reports under the current partition signature.
- `reviser_node`: select structural references, validate and fingerprint the plan,
  execute Split/Merge, update memberships and return to Router for reassessment.

Nodes return partial state updates without mutating the caller's state. Runtime
context holds models, the active tool registry and paths. Messages are replaced
per evidence round, not accumulated across partitions. Shared helpers are limited
to partition/evidence views, trace/failure handling and graph/output entry points.
Removed internal helpers are not retained as compatibility wrappers; replay
scripts and tests now call the nodes for execution and validation.

`llm.py` contains model construction, API settings, message conversion, report
summaries, payload serialization, JSON correction and usage/cache accounting.
System prompts, generation settings and bounded correction behavior are unchanged.
Structured API clients are reused. Missing required runtime fields fail explicitly;
optional scientific evidence still remains unassessed/unavailable when absent.

`schemas.py` defines state and response contracts. `tools.py` retains the scientific
tool registry, tool schemas and result compaction. `runner.py` loads configuration
and starts the graph. No separate state/evidence/revision/LLM-summary modules.

Regression tests use model/tool doubles: they verify contracts and control flow,
not scientific conclusions or live-provider behavior. They do not rerun experiments.
