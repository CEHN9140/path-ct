from agents.subtype_review.graph import initial_review_state, router_node
from agents.subtype_review.llm import LLMOutputLengthError


def plan(*actions):
    return {"actions": [
        {"action": action, "target_ids": targets, "tool_requests": [], "reason": "test"}
        for action, targets in actions
    ]}


def test_router_uses_three_calls_and_accepts_two_of_three_plan_consensus():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])

    class Router:
        def __init__(self):
            self.calls = []
            self.responses = iter([
                plan(("accept", ["C1"])),
                plan(("drop", ["C1"])),
                plan(("accept", ["C1"])),
            ])

        def invoke(self, payload):
            self.calls.append(payload)
            return next(self.responses)

    router = Router()
    router_node(state, {"router_model": router})

    assert len(router.calls) == 3
    assert router.calls[0] == router.calls[1] == router.calls[2]
    assert state["control"]["round"] == 1
    assert state["control"]["status"] == "complete"
    assert state["router_plan"]["actions"][0]["action"] == "accept"


def test_router_without_safe_consensus_is_incomplete():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
    ])

    class Router:
        def __init__(self):
            self.responses = iter([
                plan(("accept", ["C1"]), ("drop", ["C2"])),
                plan(("drop", ["C1"]), ("split", ["C2"])),
                plan(("split", ["C1"]), ("accept", ["C2"])),
            ])

        def invoke(self, payload):
            return next(self.responses)

    router_node(state, {"router_model": Router()})

    assert state["control"]["round"] == 0
    assert state["control"]["status"] == "review_incomplete_due_to_router_consensus"


def test_router_merge_consensus_keeps_the_exact_pair():
    state = initial_review_state([
        {"cluster_id": "C1", "member_ids": ["P1", "P2"]},
        {"cluster_id": "C2", "member_ids": ["P3", "P4"]},
    ])

    class Router:
        def __init__(self):
            self.responses = iter([
                plan(("merge", ["C1", "C2"])),
                plan(("merge", ["C2", "C1"])),
                plan(("accept", ["C1"]), ("accept", ["C2"])),
            ])

        def invoke(self, payload):
            return next(self.responses)

    router_node(state, {"router_model": Router()})

    assert state["control"]["round"] == 1
    assert state["router_plan"]["actions"] == [{
        "action": "merge",
        "target_ids": ["C1", "C2"],
        "tool_requests": [],
        "reason": "test",
    }]


def test_router_length_error_stops_the_ensemble_immediately():
    state = initial_review_state([{"cluster_id": "C1", "member_ids": ["P1", "P2"]}])

    class Router:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            raise LLMOutputLengthError("too long")

    router = Router()
    router_node(state, {"router_model": router})

    assert router.calls == 1
    assert state["control"]["status"] == "review_unavailable"
