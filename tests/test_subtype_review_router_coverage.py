from agents.subtype_review.graph import initial_review_state, partition_signature, router_node
from agents.subtype_review.schemas import EVIDENCE_DIMENSIONS
from agents.subtype_review.tools import TOOL_REGISTRY


def test_partition_reports_update_partition_coverage_without_nesting():
    state = initial_review_state([{"set_id": "C1", "member_ids": ["P1", "P2"]}])
    signature = partition_signature(state["partition"]["sets"])
    state["tool_evidence"].append({
        "tool_name": "structural_diagnostics", "scope": "partition",
        "target_ids": [], "partition_signature": signature,
    })
    state["reports"].append({
        "dimension": "cross_modal_consistency", "scope": "partition",
        "target_ids": [], "report_ref": "ER:partition",
    })
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [], "evidence_requests": [{
                "dimension": "biological_support", "scope": "set",
                "target_ids": ["C1"], "question": "Clarify C1 phenotype.",
            }]}

    runtime = {
        "router_model": Router(), "tool_registry": TOOL_REGISTRY,
        "patient_states_by_id": {}, "data_root": "/tmp",
        "artifact_root": "/tmp", "config_dir": "configs",
    }
    router_node(state, runtime)

    coverage = captured["evidence_coverage"]["partition"]
    assert coverage["cross_modal_consistency"] == "assessed"
    assert set(coverage) == set(EVIDENCE_DIMENSIONS)

