from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agents.subtype_review.llm import (
    JsonStructuredModel,
    LLMUsageTracker,
    LocalVerifierModel,
    VerifierChatModel,
    build_default_verifier,
    build_default_reviser,
    build_default_router,
)
from agents.subtype_review.schemas import EvidenceReportBatch, RevisionPlan, RouterPlan
from utils.llm_utils import LocalLLMClient, load_yaml_file, resolve_api_key


def test_subtype_review_config_declares_project_generation_limit():
    config = load_yaml_file("configs/subtype_review.yaml")
    assert config["llm"]["max_new_tokens"] == 32768


def test_local_openai_compatible_client_maps_project_limit_to_max_tokens(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=[]))]
            )

    monkeypatch.setattr("utils.llm_utils.local_llm_server_available", lambda url: True)
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    client = LocalLLMClient({
        "base_url": "http://local",
        "model_name": "hf-chat",
        "api_key": "test",
        "temperature": 0,
        "max_new_tokens": 32768,
    })
    client.chat([{"role": "user", "content": "test"}])
    assert captured["max_tokens"] == 32768


def test_openai_compatible_structured_request_maps_project_limit_to_max_tokens(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"actions":[{"action":"drop","target_ids":["C1"]}]}'))]
            )

    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    model = JsonStructuredModel(
        {
            "api_key_env": "TEST_KEY",
            "base_url": "http://test",
            "model_name": "deepseek-v4-flash",
            "temperature": 0,
            "max_new_tokens": 32768,
        },
        RouterPlan,
        "prompt",
    )
    model.invoke({})
    assert captured["max_tokens"] == 32768


def test_chatopenai_verifier_receives_project_limit_as_max_tokens(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=fake_chat_openai),
    )
    config = {
        "llm": {
            "api_key_env": "TEST_KEY",
            "base_url": "http://test",
            "model_name": "deepseek-v4-flash",
            "temperature": 0,
            "max_new_tokens": 32768,
        },
        "prompt_dir": "agents/subtype_review/prompts",
    }
    build_default_verifier(config, "/data/qijun/path-ct/configs")
    assert captured["max_tokens"] == 32768


class BoundModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.bind_calls = []
        self.requests = []

    def bind(self, **kwargs):
        self.bind_calls.append(kwargs)
        return self

    def bind_tools(self, tools, **kwargs):
        self.bind_calls.append({"tools": tools, **kwargs})
        return self

    def invoke(self, messages):
        self.requests.append(messages)
        return SimpleNamespace(content=next(self.responses), tool_calls=[])


def test_usage_tracker_is_statistics_only():
    tracker = LLMUsageTracker()
    for _ in range(20):
        tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}})
    assert tracker.snapshot() == {
        "api_calls": 20,
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_api_key_must_come_from_configured_environment(monkeypatch):
    monkeypatch.setenv("REVIEW_KEY", "secret")
    assert resolve_api_key({"api_key_env": "REVIEW_KEY"}) == "secret"
    with pytest.raises(ValueError):
        resolve_api_key({"api_key": "inline"})


def test_verifier_acquisition_binds_requested_real_tools_only():
    model = BoundModel(["unused"])
    first = SimpleNamespace(name="multimodal_consistency_check")
    second = SimpleNamespace(name="pathway_enrichment")
    verifier = VerifierChatModel(model, "prompt", [first, second])
    verifier.invoke({
        "mode": "acquire",
        "tool_requests": [
            {"tool_name": "multimodal_consistency_check"},
            {"tool_name": "pathway_enrichment"},
        ],
    })
    assert model.bind_calls[0]["tool_choice"] == "required"
    assert {tool.name for tool in model.bind_calls[0]["tools"]} == {
        "multimodal_consistency_check", "pathway_enrichment"
    }


def test_verifier_audit_accepts_report_batch_without_status():
    model = BoundModel([json.dumps({"reports": []})])
    verifier = VerifierChatModel(model, "prompt", [])
    response = verifier.invoke({"mode": "audit", "reports": []})
    assert json.loads(response.content) == {"reports": []}


def test_verifier_chat_audit_serializes_tool_history_without_dropping_evidence():
    model = BoundModel([json.dumps({"reports": []})])
    verifier = VerifierChatModel(model, "prompt", [])
    verifier.invoke({
        "mode": "audit",
        "round_evidence": [{"tool_name": "pathway_enrichment", "status": "success"}],
        "tool_messages": [
            AIMessage(content="", tool_calls=[{
                "name": "pathway_enrichment", "args": {}, "id": "call-1", "type": "tool_call"
            }]),
            ToolMessage(content='{"status":"success"}', tool_call_id="call-1"),
        ],
    })
    history = model.requests[0]
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "tool"
    request = json.loads(history[3]["content"])
    assert request["round_evidence"][0]["tool_name"] == "pathway_enrichment"
    assert "tool_messages" not in request


def test_local_verifier_audit_serializes_tool_history_without_dropping_evidence(monkeypatch):
    monkeypatch.setattr("agents.subtype_review.llm.resolve_api_key", lambda config: "test")
    monkeypatch.setattr("agents.subtype_review.llm.local_llm_server_available", lambda url: True)
    verifier = LocalVerifierModel(
        {
            "base_url": "http://local",
            "model_name": "local",
            "temperature": 0,
            "max_new_tokens": 256,
            "json_retries": 1,
        },
        "prompt",
        [],
    )
    requests = []

    def chat(messages, tools=None):
        requests.append(messages)
        return {"content": json.dumps({"reports": []})}

    verifier.client.chat = chat
    verifier.invoke({
        "mode": "audit",
        "round_evidence": [{"tool_name": "pathway_enrichment", "status": "success"}],
        "tool_messages": [
            AIMessage(content="assistant tool call"),
            ToolMessage(content="tool result", tool_call_id="call-1"),
        ],
    })
    history = requests[0]
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "tool"
    request = json.loads(history[3]["content"])
    assert request["round_evidence"][0]["status"] == "success"
    assert "tool_messages" not in request


def test_default_agent_prompts_use_new_contracts(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        "agents.subtype_review.llm.build_structured_model",
        lambda config, schema, prompt, usage_tracker=None: prompts.append((schema, prompt)) or prompt,
    )
    config = {"llm": {"model_name": "test"}, "prompt_dir": "agents/subtype_review/prompts"}
    build_default_router(config, "/data/qijun/path-ct/configs")
    build_default_reviser(config, "/data/qijun/path-ct/configs")
    assert prompts[0][0] is RouterPlan
    assert prompts[1][0] is RevisionPlan
    assert prompts[0][1].startswith("# Router")
    assert prompts[1][1].startswith("# Reviser")


def test_review_prompts_constrain_internal_evidence_and_medical_claims():
    router = (Path("agents/subtype_review/prompts/router.md")).read_text(encoding="utf-8")
    verifier = (Path("agents/subtype_review/prompts/verifier.md")).read_text(encoding="utf-8")

    assert "not independent validation" in router
    assert "one discovery modality" in router.lower()
    assert "set_available_n + rest_available_n" in verifier
    assert "do not call these metrics" in verifier.lower()
    assert "prognosis" in verifier
    assert "nonsignificant trends" in verifier
