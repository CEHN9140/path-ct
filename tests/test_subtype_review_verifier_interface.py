import json

from langchain_core.messages import AIMessage, ToolMessage

from agents.subtype_review.graph import (
    compact_partition_for_llm,
    execute_tool_calls,
    initial_review_state,
    metric_blocks_for_report,
    required_reports_for_round,
    validate_reports,
)
from agents.subtype_review.llm_summary import summarize_evidence
from agents.subtype_review.schemas import EvidenceReportBatch
from agents.subtype_review.tools import TOOL_REGISTRY


def registry_with_fake_tool(name, result):
    registry = {key: {**value} for key, value in TOOL_REGISTRY.items()}
    registry[name]["function"] = lambda *args, **kwargs: result
    return registry


def test_tool_message_contains_only_decision_payload():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1"]}])
    state["control"]["pending_tools"] = [{"tool_name": "pathway_enrichment", "target_ids": ["C1"]}]
    registry = registry_with_fake_tool(
        "pathway_enrichment",
        {
            "status": "success",
            "results": {
                "decision_metrics": {"per_set_rna_pathway_enrichment": {"C1": [{"q_value": 0.01}]}},
                "metrics": {"patient_level": ["large"]},
                "warnings": ["warning"],
            },
            "artifacts": {"full_metrics": "/secret/full.json"},
        },
    )
    execute_tool_calls(
        state,
        AIMessage(content="", tool_calls=[{
            "name": "pathway_enrichment", "args": {}, "id": "call-1", "type": "tool_call"
        }]),
        {
            "tool_registry": registry,
            "patient_states_by_id": {},
            "data_root": "/tmp",
            "config_dir": "configs",
            "artifact_root": "/tmp",
        },
    )
    payload = json.loads(state["messages"][-1].content)
    assert set(payload) == {
        "tool_name", "dimension", "scope", "target_ids", "status",
        "metrics", "warnings", "missing_reason", "errors",
    }
    assert "full_metrics" not in payload
    assert "artifact_paths" not in payload
    assert "metric_refs" not in payload
    assert payload["metrics"] == {"per_set_rna_pathway_enrichment": {"C1": [{"q_value": 0.01}]}}


def test_round_evidence_summary_is_metadata_only():
    summary = summarize_evidence([{
        "tool_name": "pathway_enrichment",
        "dimension": "biological_support",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "status": "success",
        "metrics": {"large": "payload"},
        "metric_refs": ["leaf.ref"],
    }])
    assert summary == [{
        "tool_name": "pathway_enrichment",
        "dimension": "biological_support",
        "scope": "set_identity",
        "target_ids": ["C1"],
        "status": "success",
        "warnings": [],
        "errors": [],
        "missing_reason": "",
    }]


def test_required_reports_and_partition_payload_are_compact():
    state = initial_review_state([
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3"]},
    ])
    state["control"]["pending_tools"] = [
        {"tool_name": "pathway_enrichment", "target_ids": ["C1", "C2"]},
        {"tool_name": "known_label_echo_test", "target_ids": []},
    ]
    required = required_reports_for_round(state, {"tool_registry": TOOL_REGISTRY})
    assert required == [
        {
            "dimension": "biological_support", "scope": "set_identity",
            "target_ids": ["C1"], "tool_names": ["pathway_enrichment"],
        },
        {
            "dimension": "biological_support", "scope": "set_identity",
            "target_ids": ["C2"], "tool_names": ["pathway_enrichment"],
        },
        {
            "dimension": "known_label_echo", "scope": "partition",
            "target_ids": [], "tool_names": ["known_label_echo_test"],
        },
    ]
    assert compact_partition_for_llm(state) == {
        "sets": [{"set_id": "C1", "member_n": 2}, {"set_id": "C2", "member_n": 1}]
    }


def test_python_attaches_target_grounded_block_metric_refs():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1"]}])
    state["control"]["pending_tools"] = [
        {"tool_name": "pathway_enrichment", "target_ids": ["C1"]},
        {"tool_name": "mutation_enrichment", "target_ids": ["C1"]},
        {"tool_name": "cnv_characterization", "target_ids": ["C1"]},
    ]
    state["round_evidence"] = [
        {
            "tool_name": "pathway_enrichment", "dimension": "biological_support",
            "scope": "set_identity", "target_ids": ["C1"],
            "metrics": {"per_set_rna_pathway_enrichment": {"C1": []}},
        },
        {
            "tool_name": "mutation_enrichment", "dimension": "biological_support",
            "scope": "set_identity", "target_ids": ["C1"],
            "metrics": {"per_set_wxs_feature_enrichment": {"C1": []}},
        },
        {
            "tool_name": "cnv_characterization", "dimension": "biological_support",
            "scope": "set_identity", "target_ids": ["C1"],
            "metrics": {"per_comparison_cnv": {"C1_vs_rest": []}},
        },
    ]
    batch = EvidenceReportBatch.model_validate({"reports": [{
        "dimension": "biological_support", "scope": "set_identity",
        "target_ids": ["C1"], "observations": [],
        "tool_refs": ["pathway_enrichment", "mutation_enrichment", "cnv_characterization"],
    }]})
    validate_reports(batch, state, {"tool_registry": TOOL_REGISTRY})
    assert batch.reports[0].metric_refs == [
        "tool_results.cnv_characterization.metrics.per_comparison_cnv.C1_vs_rest",
        "tool_results.mutation_enrichment.metrics.per_set_wxs_feature_enrichment.C1",
        "tool_results.pathway_enrichment.metrics.per_set_rna_pathway_enrichment.C1",
    ]
    assert all("C2" not in ref for ref in batch.reports[0].metric_refs)


def test_partition_metric_ref_is_partition_grounded():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1"]}])
    state["control"]["pending_tools"] = [{"tool_name": "known_label_echo_test", "target_ids": []}]
    state["round_evidence"] = [{
        "tool_name": "known_label_echo_test", "dimension": "known_label_echo",
        "scope": "partition", "target_ids": [],
        "metrics": {"known_label_structure_comparison": {"stage": {}}},
    }]
    report = EvidenceReportBatch.model_validate({"reports": [{
        "dimension": "known_label_echo", "scope": "partition", "target_ids": [],
        "observations": [], "tool_refs": ["known_label_echo_test"],
    }]}).reports[0]
    assert metric_blocks_for_report(report, state) == [
        "tool_results.known_label_echo_test.metrics.known_label_structure_comparison"
    ]
