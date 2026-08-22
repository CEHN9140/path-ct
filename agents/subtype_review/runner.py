from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agents.subtype_review.graph import build_review_graph, initial_review_state
from agents.subtype_review.llm import (
    LLMUsageTracker,
    build_default_reviser,
    build_default_router,
    build_default_verifier,
)
from utils.llm_utils import load_yaml_file


def run_subtype_review(
    candidate_clusters: list[dict[str, Any]],
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    data_root: str,
    config_dir: str,
    *,
    artifact_root: str | None = None,
) -> dict[str, Any]:
    review_config = load_yaml_file(Path(config_dir) / "subtype_review.yaml")
    budget = dict(review_config.get("budget", {}) or {})
    usage_tracker = LLMUsageTracker()
    runtime = {
        "patient_states_by_id": {
            str(key): dict(value) for key, value in patient_states_by_id.items()
        },
        "output_root": str(data_root),
        "data_root": str(data_root),
        "review_output_root": str(artifact_root or data_root),
        "config_dir": str(config_dir),
    }
    verifier_model = build_default_verifier(review_config, config_dir, usage_tracker=usage_tracker)
    reviser_model = build_default_reviser(review_config, config_dir, usage_tracker=usage_tracker)
    router_model = build_default_router(review_config, config_dir, usage_tracker=usage_tracker)
    graph = build_review_graph(
        verifier_model=verifier_model,
        router_model=router_model,
        reviser_model=reviser_model,
        runtime=runtime,
    )
    state = initial_review_state(candidate_clusters)
    state["control"]["max_rounds"] = int(budget.get("max_rounds", 10) or 10)
    state["control"]["max_failures"] = int(budget.get("max_failures", 3) or 3)
    final_state = graph.invoke(state, context=runtime)
    final_state["control"]["llm_usage"] = usage_tracker.snapshot()
    return final_state
