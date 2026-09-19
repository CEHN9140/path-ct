"""Regression checks for graph state ownership and API message transport."""

import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agents.subtype_review.graph import build_review_graph, initial_review_state, partition_signature
from agents.subtype_review.llm import JsonStructuredModel, LLMUsageTracker, message_history
from agents.subtype_review.schemas import EvidenceReportBatch
from agents.subtype_review.tools import TOOL_REGISTRY


class TerminalRouter:
    def invoke(self, payload):
        return {"actions": [{
            "action": "drop", "target_ids": [item["set_id"]],
            "decision_state": {
                "identity": "unassessed", "structure": "unassessed",
                "alternative_explanation": "unassessed", "uncertainty": "yes",
            },
        } for item in payload["partition"]["sets"]]}


def test_graph_owns_its_updates_without_mutating_the_callers_state():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    before = copy.deepcopy(state)
    updates = list(build_review_graph().stream(
        state, context={"router_model": TerminalRouter()}, stream_mode="updates",
    ))

    assert state == before
    router_update = updates[-1]["router"]
    assert router_update["control"]["status"] == "complete"
    assert router_update["history"]
    assert "partition" not in router_update
    assert "evidence_memory" not in router_update


def test_structured_model_reuses_client_with_identical_generation_parameters(monkeypatch):
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
    assert requests[0]["messages"][0]["content"] == "unchanged prompt\n\nReturn exactly one valid JSON object."


def test_assistant_tool_calls_use_api_format_and_match_tool_message_ids():
    messages = message_history({"message_history": [
        AIMessage(content="", tool_calls=[{
            "name": "pathway_enrichment", "args": {"target_ids": ["C1"]},
            "id": "call-1", "type": "tool_call",
        }]),
        ToolMessage(content='{"status":"success"}', tool_call_id="call-1"),
    ]})
    call = messages[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "pathway_enrichment"
    assert json.loads(call["function"]["arguments"]) == {"target_ids": ["C1"]}
    assert messages[1]["tool_call_id"] == call["id"]


def test_json_correction_retry_preserves_history_and_usage(monkeypatch):
    requests = []
    contents = iter(['{"invalid":true}', '{"reports":[]}'])

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return SimpleNamespace(
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=next(contents)))],
        )

    monkeypatch.setenv("TEST_REVIEW_KEY", "test")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    usage = LLMUsageTracker()
    model = JsonStructuredModel({
        "api_key_env": "TEST_REVIEW_KEY", "base_url": "http://test", "model_name": "test-model",
        "max_new_tokens": 100, "json_retries": 1,
    }, EvidenceReportBatch, "unchanged prompt", usage)
    assert model.invoke({"mode": "audit"}) == {"reports": []}
    assert requests[1]["messages"][:2] == requests[0]["messages"]
    assert requests[1]["messages"][2] == {"role": "assistant", "content": '{"invalid":true}'}
    assert requests[1]["messages"][3]["content"].startswith("Return corrected JSON only. Error: ValidationError:")
    assert usage.snapshot()["api_calls"] == 2
    assert usage.snapshot()["total_tokens"] == 30


@pytest.mark.parametrize("operation", ["split", "merge"])
def test_revision_reacquires_evidence_on_the_new_partition(operation, monkeypatch):
    state = initial_review_state([
        {"set_id": "C1", "member_ids": [f"P{i}" for i in range(6)]},
        {"set_id": "C2", "member_ids": [f"P{i}" for i in range(6, 10)]},
    ])
    signature = partition_signature(state["partition"]["sets"])
    state["round_evidence"] = [{
        "tool_name": "multimodal_consistency_check", "status": "success",
        "dimension": "cross_modal_consistency", "scope": "set_identity",
        "target_ids": ["C1", "C2"], "partition_signature": signature,
        "full_metrics": {"structural_characterization": {
            "internal_structure_by_set": {"C1": {}},
            "boundary_by_pair": {"C1+C2": {"targets": ["C1", "C2"]}},
        }},
    }]
    state["reports"] = [{
        "dimension": "cross_modal_consistency", "scope": "set_identity",
        "target_ids": [target], "tool_refs": ["multimodal_consistency_check"],
    } for target in ("C1", "C2")]
    before = copy.deepcopy(state)

    class Router(TerminalRouter):
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            plan = super().invoke(payload)
            if self.calls == 1:
                if operation == "merge":
                    plan["actions"] = plan["actions"][:1]
                    plan["actions"][0]["target_ids"] = ["C1", "C2"]
                plan["actions"][0]["action"] = operation
                plan["actions"][0]["decision_state"]["structure"] = "incompatible"
                return plan
            if self.calls == 2:
                assert payload["evidence_reports"] == []
                return {"evidence_requests": [{
                    "dimension": "cross_modal_consistency", "question": "Assess revised sets.",
                    "target_ids": [item["set_id"] for item in payload["partition"]["sets"]],
                }]}
            for action in plan["actions"]:
                action["action"] = "accept"
                action["decision_state"]["structure"] = "compatible"
            return plan

    class Reviser:
        def invoke(self, payload):
            if operation == "merge":
                return {"merge_plans": [{"target_ids": ["C1", "C2"]}]}
            return {"split_plans": [{
                "target_id": "C1", "n_children": 2, "structural_basis": ["fused"],
                "execution_strategy": "fused_similarity_spectral",
            }]}

    class Verifier:
        def invoke(self, payload):
            if payload["mode"] == "acquire":
                return AIMessage(content="", tool_calls=[{
                    "name": "multimodal_consistency_check", "id": "revision-call",
                    "args": {"target_ids": payload["evidence_requests"][0]["target_ids"]},
                }])
            return {"reports": payload["required_reports"]}

    def tool(*args, **kwargs):
        return {"status": "success", "results": {"decision_metrics": {
            "sets": {target: {"score": 0.7} for target in kwargs["target_ids"]},
        }}}

    monkeypatch.setattr("tools.cross_modal_structure.execute_split_membership",
                        lambda root, members, *args: [members[:3], members[3:]])
    router = Router()
    result = build_review_graph().invoke(state, context={
        "data_root": "/tmp", "router_model": router,
        "reviser_model": Reviser(), "verifier_model": Verifier(),
        "tool_registry": {"multimodal_consistency_check": {
            **TOOL_REGISTRY["multimodal_consistency_check"], "function": tool,
        }},
    })
    assert state == before
    assert result["control"]["status"] == "complete"
    assert router.calls == 3
    new_signature = partition_signature(result["partition"]["sets"])
    assert new_signature != signature
    assert result["evidence_memory"][new_signature] == result["reports"]
    assert all(row["partition_signature"] == new_signature for row in result["round_evidence"])
    assert {r["target_ids"][0] for r in result["reports"]} == {s["set_id"] for s in result["partition"]["sets"]}
    assert all(report["metric_refs"] for report in result["reports"])
    assert result["messages"][-1].tool_call_id == "revision-call"
