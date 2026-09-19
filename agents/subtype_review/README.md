# Subtype review implementation

The graph still starts at Router and contains three agents:

```text
Router --EvidenceRequests--> Verifier (acquire tools -> audit reports) --> Router
Router --Split/Merge-------> Reviser (validate -> apply revision) -------> Router
Router --Accept/Drop-------> end
```

`graph.py` contains agent transitions, conditional edges and output serialization.
`state.py` owns partition identity, initialization, trace/failure bookkeeping and
round snapshots. `evidence.py` owns tool eligibility/execution, report provenance,
coverage and action validation. `revision.py` validates and applies structural
plans. The existing imports used by experiment runners remain available from
`graph.py`.

`ReviewState` and `ReviewControl` describe the mutable workflow data. Models,
tool registry, input/output paths and active modalities are supplied through
`ReviewContext`, not serialized into state. The graph adapter copies mutable
containers and returns each node's declared state updates. Transition helpers
remain callable directly by contract tests. Messages use overwrite semantics:
they belong to one evidence round and are reset before reacquisition/revision.
An append-only message reducer would change this behavior.

`llm.py` owns provider settings, payload serialization and usage tracking. Existing
temperature, generation limits, thinking settings and retry behavior are retained.
Each structured model lazily reuses one OpenAI-compatible client. Verifier
acquisition binds only eligible tools; audit/Router/Reviser retain JSON-object
transport followed by Pydantic validation. System prompt files are unchanged.
Tool results retain their call IDs; LangChain message objects are converted with
the library's OpenAI message converter when an API history is needed. Remote
audit retains the existing compact evidence payload and its provenance repair.

`tools.py` still contains the original tool registry, schemas and result compaction.
The unused `summarize_evidence` helper was removed; `llm_summary.py` retains the
report projection used by Router and existing replay scripts.

These boundaries follow the official [LangGraph state/context and node-update
model](https://docs.langchain.com/oss/python/langgraph/graph-api) and
[LangChain message/tool-call contracts](https://docs.langchain.com/oss/python/langchain/messages).
They do not change the scientific workflow or imply independent validation of its
scientific decisions. The regression suite uses deterministic model/tool doubles;
it does not call paid APIs or regenerate scientific experiment results.
