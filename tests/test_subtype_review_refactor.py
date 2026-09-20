import copy
from types import SimpleNamespace

from agents.subtype_review.graph import build_review_graph, initial_review_state
from agents.subtype_review.llm import JsonStructuredModel
from agents.subtype_review.schemas import EvidenceReportBatch


class TerminalRouter:
    def invoke(self, payload):
        return {"actions": [{
            "action": "drop", "target_ids": [item["set_id"]],
            "decision_state": {
                "identity": "unassessed", "structure": "unassessed",
                "alternative_explanation": "unassessed", "uncertainty": "yes",
            },
        } for item in payload["partition"]["sets"]]}


def test_graph_does_not_mutate_caller_state():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    original = copy.deepcopy(state)
    result = build_review_graph().invoke(state, context={"router_model": TerminalRouter()})

    assert state == original
    assert result["control"]["status"] == "complete"
    assert result["history"]


def test_structured_model_reuses_its_openai_client(monkeypatch):
    clients, requests = [], []

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content='{"reports":[]}'),
        )])

    def client(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("openai.OpenAI", client)
    monkeypatch.setenv("TEST_REVIEW_KEY", "test")
    model = JsonStructuredModel({
        "api_key_env": "TEST_REVIEW_KEY", "base_url": "http://test", "model_name": "test-model",
        "temperature": 0.0, "max_new_tokens": 100,
    }, EvidenceReportBatch, "unchanged prompt")

    assert model.invoke({"mode": "audit"}) == {"reports": []}
    assert model.invoke({"mode": "audit"}) == {"reports": []}
    assert len(clients) == 1
    assert len(requests) == 2 and requests[0] == requests[1]
    assert requests[0]["max_tokens"] == 100
