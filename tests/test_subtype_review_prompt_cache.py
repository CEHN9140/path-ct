import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from agents.subtype_review.llm import JsonStructuredModel, LLMUsageTracker, VerifierChatModel
from agents.subtype_review.schemas import EvidenceReportBatch


def capture_requests(monkeypatch, reports=None):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content=json.dumps({"reports": reports or []})
            ))],
            usage={"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
        )

    monkeypatch.setattr("openai.OpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ))
    monkeypatch.setenv("TEST_CACHE_KEY", "not-a-real-key")
    model = JsonStructuredModel({
        "api_key_env": "TEST_CACHE_KEY", "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-v4-flash", "max_new_tokens": 32768,
    }, EvidenceReportBatch, "JSON audit")
    return model, requests


def audit_payload(run):
    evidence = {
        "tool_name": "multimodal_consistency_check", "dimension": "cross_modal_consistency",
        "scope": "set_identity", "target_ids": ["C1"], "status": "success",
        "metrics": {"effect": 0.125, "q": 0.3, "warnings": ["scientific limitation"]},
        "warnings": [], "missing_reason": "", "errors": [],
    }
    return {
        "mode": "audit", "partition": {"sets": [{"set_id": "C1", "member_n": 6}]},
        "required_reports": [{"dimension": "cross_modal_consistency", "target_ids": ["C1"]}],
        "round_evidence": [{**evidence, "metric_refs": ["very.long.metric.ref"] * 200,
                            "full_metrics": {"structural_characterization": {"ARI": 0.92}},
                            "artifact_paths": {"full": f"/output/run{run}/metrics.json"},
                            "partition_signature": "local-provenance"}],
        "prior_reports": [{"observations": [{"finding": "retain this"}],
                           "tool_refs": ["multimodal_consistency_check"], "metric_refs": ["ref"] * 200}],
        "round": 1,
    }


def test_audit_wire_deduplicates_without_losing_scientific_evidence(monkeypatch):
    audit, requests = capture_requests(monkeypatch)
    verifier = VerifierChatModel(None, audit, "prompt", [])
    original = audit_payload(1)
    before = copy.deepcopy(original)
    verifier.invoke(original)
    verifier.invoke(audit_payload(2))
    first = requests[0]["messages"][1]["content"]
    assert first == requests[1]["messages"][1]["content"]  # run paths/IDs cannot break prefixes
    payload = json.loads(first)
    assert len(payload["round_evidence"]) == 1
    row = payload["round_evidence"][0]
    assert row["metrics"] == original["round_evidence"][0]["metrics"]
    assert row["full_metrics"] == original["round_evidence"][0]["full_metrics"]
    assert not {"artifact_paths", "metric_refs", "partition_signature"} & row.keys()
    assert payload["prior_reports"][0]["tool_refs"] == ["multimodal_consistency_check"]
    assert "metric_refs" not in payload["prior_reports"][0]
    assert original == before  # local provenance remains intact
    assert len(requests) == 2  # repeats must still invoke the model independently


def test_audit_tool_order_does_not_change_wire_payload(monkeypatch):
    audit, requests = capture_requests(monkeypatch)
    verifier = VerifierChatModel(None, audit, "prompt", [])
    payload = audit_payload(1)
    payload["round_evidence"].append({
        "tool_name": "pathway_enrichment", "target_ids": ["C1"], "metrics": {"q": 0.4},
    })
    verifier.invoke(payload)
    payload["round_evidence"].reverse()
    verifier.invoke(payload)
    assert requests[0]["messages"] == requests[1]["messages"]


def test_structured_wire_is_stable_and_round_corrections_follow_evidence(monkeypatch):
    model, requests = capture_requests(monkeypatch)
    payload = {"round": 1, "validation_error": "old", "partition": {"b": 2, "a": 1},
               "round_evidence": [{"metrics": {"q": 0.1, "effect": 0.5}}]}
    model.invoke(payload)
    model.invoke({**payload, "partition": {"a": 1, "b": 2}})
    first = requests[0]["messages"][1]["content"]
    assert first == requests[1]["messages"][1]["content"]
    assert json.loads(first) == payload
    assert first.index('"round_evidence"') < first.index('"round"')
    assert first.index('"round_evidence"') < first.index('"validation_error"')


@pytest.mark.parametrize("response", [
    {"usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
               "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200}},
    SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=10, total_tokens=1010,
                                         prompt_cache_hit_tokens=800, prompt_cache_miss_tokens=200)),
    AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010},
              response_metadata={"token_usage": {"prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200}}),
    {"usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
               "prompt_tokens_details": {"cached_tokens": 800}}},
    {"usage_metadata": {"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010,
                        "input_token_details": {"cache_read": 800}}},
])
def test_cache_accounting_handles_sdk_and_langchain(response):
    tracker = LLMUsageTracker()
    tracker.before_request()
    tracker.record_response(response)
    stats = tracker.snapshot()
    assert stats["prompt_tokens"] == 1000  # no double counting normalized + raw usage
    assert stats["prompt_cache_hit_tokens"] == 800
    assert stats["prompt_cache_miss_tokens"] == 200
    assert stats["prompt_cache_hit_rate"] == 0.8


def test_unknown_cache_usage_is_not_zero_and_rate_uses_observed_requests():
    tracker = LLMUsageTracker()
    tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 9000, "completion_tokens": 10, "total_tokens": 9010}})
    assert tracker.snapshot()["prompt_cache_hit_tokens"] is None
    assert tracker.snapshot()["prompt_cache_hit_rate"] is None
    tracker.before_request()
    tracker.record_response({"usage": {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 0,
                                       "prompt_cache_miss_tokens": 1000}})
    stats = tracker.snapshot()
    assert stats["prompt_cache_hit_tokens"] == 0
    assert stats["prompt_cache_hit_rate"] == 0.0
    assert stats["cache_observed_requests"] == 1


def test_usage_is_persisted_per_request_without_prompt_content(tmp_path):
    path = tmp_path / "llm_requests.jsonl"
    tracker = LLMUsageTracker(path)
    tracker.before_request(role="verifier_audit", model="deepseek-v4-flash",
                           messages=[{"role": "user", "content": "private patient detail"}])
    tracker.record_response({"model": "returned-model-version", "usage": {
        "prompt_tokens": 1000, "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200}})
    row = json.loads(path.read_text())
    assert row["role"] == "verifier_audit"
    assert row["response_model"] == "returned-model-version"
    assert row["prompt_cache_hit_tokens"] == 800
    assert row["request_chars"] > 0
    assert "private patient detail" not in path.read_text()
