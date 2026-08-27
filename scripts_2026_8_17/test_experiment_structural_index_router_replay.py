from __future__ import annotations

from scripts_2026_8_17 import experiment_structural_index_router_replay as replay


def test_replay_payload_contains_structural_index_and_does_not_execute_actions():
    entry = {
        "round": 1,
        "partition": {"sets": [
            {"set_id": "C1", "member_ids": ["P1", "P2"], "revision_lineage": []},
            {"set_id": "C2", "member_ids": ["P3", "P4"], "revision_lineage": []},
        ]},
        "round_evidence": [{
            "tool_name": "multimodal_consistency_check",
            "full_metrics": {"structural_characterization": {
                "internal_structure_by_set": {
                    "C1": {"fused_binary_probe": {"median_silhouette": 0.2}},
                    "C2": {"fused_binary_probe": {"median_silhouette": 0.1}},
                },
                "boundary_by_pair": {
                    "C1+C2": {"targets": ["C1", "C2"], "fused": {
                        "pair_median_silhouette": 0.3,
                    }},
                },
            }},
        }],
        "evidence_reports": [],
        "router_plan": {"actions": [
            {"action": "accept", "target_ids": ["C1"], "support_sources": ["RNA", "CNV"], "corroboration_satisfied": True, "active_contradiction": False, "dominant_confounder": False, "tool_requests": [], "reason": "ok"},
            {"action": "accept", "target_ids": ["C2"], "support_sources": ["RNA", "CNV"], "corroboration_satisfied": True, "active_contradiction": False, "dominant_confounder": False, "tool_requests": [], "reason": "ok"},
        ]},
    }

    class Router:
        def __init__(self):
            self.payload = None

        def invoke(self, payload):
            self.payload = payload
            assert payload["available_extra_evidence"] == []
            assert payload["structural_index"]["per_set"]["C1"]["binary_probe"]["fused_silhouette"] == 0.2
            return {"actions": [
                {"action": "accept", "target_ids": ["C1"], "support_sources": ["RNA", "CNV"], "corroboration_satisfied": True, "active_contradiction": False, "dominant_confounder": False, "tool_requests": [], "reason": "ok"},
                {"action": "accept", "target_ids": ["C2"], "support_sources": ["RNA", "CNV"], "corroboration_satisfied": True, "active_contradiction": False, "dominant_confounder": False, "tool_requests": [], "reason": "ok"},
            ]}

    model = Router()
    payload, plan = replay.replay_entry(entry, model)
    assert payload["structural_index"]["boundary_by_pair"][0]["pair"] == ["C1", "C2"]
    assert plan["actions"][0]["action"] == "accept"
    assert model.payload is payload
