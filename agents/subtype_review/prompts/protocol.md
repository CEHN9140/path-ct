# Subtype review protocol

The Graph has exactly three Agents: Verifier, Router, and Reviser. Python owns runtime, validation, and deterministic membership execution.

\`ReviewState.partition\` is the only scientific object. It contains only the current sets. Historical parents are stored in \`history\` and child lineage; they are never retained in the current partition.

Before every Router decision, Python prepares a new round. Every default scientific tool runs again for the whole current partition, together with any valid extra tools requested by the preceding Router. Reports are per current set for the set dimensions and partition-level for global tools.

Router must cover every current set exactly once. Need Evidence has priority and makes all other actions tentative. Split and Merge require positive structural evidence. If no evidence or structural action supports acceptance, Router chooses Drop.

Reviser is called once for a RouterPlan with structural actions and returns one whole-partition RevisionPlan. Python validates and applies all plans atomically. A changed partition starts a new full round.

Only a successful Router schema and contract validation increments \`round\`. The tenth Router decision may complete Accept/Drop; evidence or a structural change on round ten ends as \`review_incomplete_due_to_round_budget\`.
