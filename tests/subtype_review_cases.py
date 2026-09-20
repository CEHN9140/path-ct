"""Model doubles for testing contracts through the actual graph nodes."""

import copy
from types import SimpleNamespace

from agents.subtype_review.graph import router_node, verifier_node


def router_inputs(state, context):
    captured = {}
    router_node(state, {**context, "router_model": SimpleNamespace(
        invoke=lambda payload: captured.update(payload) or {},
    )})
    return captured


def check_router_plan(plan, state, context):
    reviewed = router_node(copy.deepcopy(state), {
        **context, "router_model": SimpleNamespace(invoke=lambda payload: plan.model_dump()),
    })
    if reviewed["control"]["error"]:
        raise ValueError(reviewed["control"]["error"])
    return reviewed


def acquire_evidence(state, message, context):
    state["control"]["next"] = "verifier_acquire"
    state.update(verifier_node(state, {
        **context, "verifier_model": SimpleNamespace(invoke=lambda payload: message),
    }))
    if state["control"]["error"]:
        raise ValueError(state["control"]["error"])
