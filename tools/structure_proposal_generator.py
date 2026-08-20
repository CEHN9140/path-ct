from __future__ import annotations

from typing import Any, Mapping


def generate_structure_proposals(
    cluster_state: Mapping[str, Any],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str = "",
    all_cluster_states: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate deterministic legal Split/Merge proposals without scientific labels."""
    from tools.structural_adequacy import generate_structure_proposal_metrics

    raw = generate_structure_proposal_metrics(
        dict(cluster_state),
        {str(key): dict(value) for key, value in patient_states_by_id.items()},
        output_root,
        config_dir=config_dir,
        all_cluster_states=[dict(item) for item in list(all_cluster_states or [])],
    )
    metrics = dict(dict(raw.get("results", {}) or {}).get("metrics", {}) or {})
    splits = [dict(item) for item in list(metrics.get("split_candidates", []) or [])]
    merges = [dict(item) for item in list(metrics.get("merge_candidates", []) or [])]
    memberships = {
        str(item.get("set_id") or item.get("cluster_id")): sorted(str(x) for x in item.get("member_ids", []) or [])
        for item in list(all_cluster_states or [cluster_state])
    }
    for row in splits:
        row["proposal_id"] = str(row.get("plan_id", ""))
        row["parent_members"] = memberships.get(str(row.get("source_set_id", "")), [])
        row["eligible_for_review"] = (
            int(row.get("child_count", 0) or 0) == 2
            and float(
                dict(row.get("selection_adjusted_null", {}) or {}).get("q_value", 1)
                or 1
            )
            <= 0.05
            and float(
                dict(row.get("selection_adjusted_null", {}) or {}).get(
                    "separation_gain_over_null", 0
                )
                or 0
            )
            > 0
        )
    for row in merges:
        row["proposal_id"] = str(row.get("plan_id", ""))
        row["memberships"] = [memberships.get(str(set_id), []) for set_id in row.get("set_ids", [])]
        row["eligible_for_review"] = True
    return {
        "tool_name": "structure_proposal_generator",
        "status": str(raw.get("status", "failure")),
        "results": {
            "metrics": {
                "split_proposals": splits,
                "merge_proposals": merges,
            },
            "warnings": list(dict(raw.get("results", {}) or {}).get("warnings", []) or []),
            "missing_reason": str(dict(raw.get("results", {}) or {}).get("missing_reason", "") or ""),
        },
        "errors": list(raw.get("errors", []) or []),
    }
