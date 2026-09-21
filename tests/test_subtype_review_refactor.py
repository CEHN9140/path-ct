import copy
from types import SimpleNamespace

from agents.subtype_review.graph import build_review_graph, initial_review_state
from agents.subtype_review.llm import JsonStructuredModel
from agents.subtype_review.schemas import EvidenceReportBatch


class TerminalRouter:
    def invoke(self, payload):
        report_ref = payload["evidence_reports"][0]["report_ref"]
        return {"actions": [{
            "action": "drop", "target_ids": [item["set_id"]],
            "evidence_report_refs": [report_ref], "reason": "The evidence does not justify retention.",
        } for item in payload["partition"]["sets"]]}


def test_graph_does_not_mutate_caller_state():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    original = copy.deepcopy(state)

    class Verifier:
        def invoke(self, payload):
            return {"reports": [{
                **{key: required[key] for key in ("dimension", "aspect", "scope", "target_ids")},
                "observations": [],
                "dimension_interpretation": "The partition screen describes current structure.",
                "cross_evidence_context": "No prior report is relevant.",
                "limitations": [],
                "tool_refs": [],
                "metric_refs": [],
            } for required in payload["required_reports"]]}

    def structural_diagnostics(**kwargs):
        return {"status": "success", "metrics": {"partition": {
            "internal_structure": {"C1": {"candidate_k": 1}},
        }}}

    context = {
        "router_model": TerminalRouter(), "verifier_model": Verifier(),
        "tool_registry": {"structural_diagnostics": {
            "dimension": "cross_modal_consistency", "aspect": "structural_diagnostics",
            "scopes": ("partition",), "description": "Partition screen.",
            "function": structural_diagnostics,
        }},
        "patient_states_by_id": {"P1": {}, "P2": {}},
        "data_root": "/tmp", "config_dir": "configs",
    }
    result = build_review_graph().invoke(state, context=context)

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
