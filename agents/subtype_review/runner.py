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
    usage_tracker = LLMUsageTracker(
        max_llm_calls=int(budget.get("max_llm_calls", 12) or 12)
    )
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
    state["control"]["max_rounds"] = int(budget.get("max_rounds", 12) or 12)
    state["control"]["max_failures"] = int(budget.get("max_failures", 3) or 3)
    for key, default in {
        "max_split_depth": 1,
        "max_structural_changes": 2,
        "max_structural_reviews": 3,
        "router_contract_attempts": 2,
    }.items():
        state["control"][key] = int(budget.get(key, default) or default)
    cross_modal = dict(review_config.get("cross_modal", {}) or {})
    state["control"]["policy"] = {
        "accept_min_supporting_modalities": int(
            cross_modal.get("accept_min_supporting_modalities", 2) or 2
        ),
        "split_min_supporting_modalities": int(
            cross_modal.get("split_min_supporting_modalities", 2) or 2
        ),
        "merge_min_supporting_modalities": int(
            cross_modal.get("merge_min_supporting_modalities", 2) or 2
        ),
        "split_require_molecular_or_biology": bool(
            cross_modal.get("split_require_molecular_or_biology", True)
        ),
        "min_split_size": int(budget.get("min_split_size", 10) or 10),
    }
    final_state = graph.invoke(
        state,
        context=runtime,
        config={"recursion_limit": int(state["control"]["max_rounds"]) * 3 + 10},
    )
    final_state["control"]["llm_usage"] = usage_tracker.snapshot()
    return final_state
