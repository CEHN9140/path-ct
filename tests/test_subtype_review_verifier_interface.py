import json

from langchain_core.messages import AIMessage, ToolMessage

from agents.subtype_review.graph import (
    compact_partition_for_llm,
    eligible_tools_for_requests,
    execute_tool_calls,
    initial_review_state,
    required_reports_for_round,
)
from agents.subtype_review.llm_summary import summarize_reports
from agents.subtype_review.tools import TOOL_REGISTRY


def runtime(registry=None):
    return {
        "tool_registry": registry or TOOL_REGISTRY,
        "patient_states_by_id": {}, "data_root": "/tmp",
        "config_dir": "configs", "artifact_root": "/tmp",
    }


def test_tool_message_contains_only_decision_payload():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    request = {"dimension": "biological_support", "target_ids": ["C1"], "question": "x"}
    state["control"].update({
        "pending_evidence_requests": [request],
        "eligible_tools": eligible_tools_for_requests(state, runtime(), [request]),
    })
    registry = {key: {**value} for key, value in TOOL_REGISTRY.items()}
    registry["pathway_enrichment"]["function"] = lambda *args, **kwargs: {
        "status": "success",
        "results": {"decision_metrics": {"C1": {"q": 0.01}}, "metrics": {"private": 1}},
        "artifacts": {"full": "/secret"},
    }
    execute_tool_calls(
        state,
        AIMessage(content="", tool_calls=[{
            "name": "pathway_enrichment", "args": {"target_ids": ["C1"]},
            "id": "call-1", "type": "tool_call",
        }]),
        runtime(registry),
    )
    payload = json.loads(state["messages"][-1].content)
    assert set(payload) == {
        "tool_name", "dimension", "scope", "target_ids", "status",
        "metrics", "warnings", "missing_reason", "errors",
    }
    assert payload["metrics"] == {"C1": {"q": 0.01}}


def test_router_report_summary_preserves_interpretation_and_excludes_raw_metrics():
    assert summarize_reports([{
        "tool_name": "pathway_enrichment", "dimension": "biological_support",
        "scope": "set_identity", "target_ids": ["C1"], "status": "success",
        "observations": [{"metric": "q", "finding": "small effect"}],
        "medical_interpretation": "uncertain identity",
        "metrics": {"private": 1}, "metric_refs": ["leaf.ref"],
    }]) == [{
        "dimension": "biological_support", "scope": "set_identity", "target_ids": ["C1"],
        "observations": [{"metric": "q", "finding": "small effect"}],
        "statistical_interpretation": "", "medical_interpretation": "uncertain identity",
        "limitations": [], "tool_refs": [],
    }]


def test_required_reports_follow_tools_actually_selected():
    state = initial_review_state([
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
    ])
    state["control"]["pending_evidence_requests"] = [{
        "dimension": "biological_support", "target_ids": ["C1", "C2"], "question": "x",
    }]
    state["round_evidence"] = [{
        "tool_name": "pathway_enrichment", "dimension": "biological_support",
        "scope": "set_identity", "target_ids": ["C1"], "status": "success",
    }]
    assert required_reports_for_round(state, runtime()) == [{
        "dimension": "biological_support", "scope": "set_identity",
        "target_ids": ["C1"], "tool_names": ["pathway_enrichment"],
    }]
    assert compact_partition_for_llm(state) == {
        "sets": [{"set_id": "C1", "member_n": 2}, {"set_id": "C2", "member_n": 2}]
    }
