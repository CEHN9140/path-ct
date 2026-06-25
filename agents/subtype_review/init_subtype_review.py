from __future__ import annotations

from typing import Any, Mapping

from utils.cluster_flow import initial_subtype_review_state


def init_subtype_review(state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    cluster = dict(inventory.get("cluster", {}) or {})
    inventory["cluster_state"] = (
        cluster
        if cluster.get("budget_state") or cluster.get("revision_history")
        else initial_subtype_review_state(cluster)
    )
    return {"inventory": inventory}
