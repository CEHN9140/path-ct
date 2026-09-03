from __future__ import annotations

import json

from scripts_2026_8_31.analyze_router_repeatability import audit_router_repeatability


def test_router_repeatability_reports_action_flip_and_evidence_summary(tmp_path):
    for repeat, action in ((1, "drop"), (2, "accept"), (3, "drop")):
        run_root = tmp_path / f"run{repeat}" / "K3"
        run_root.mkdir(parents=True)
        (run_root / "run_metadata.json").write_text(json.dumps({
            "source_sha256": "source",
            "input_data_signature": "input",
            "review_signature": "review",
        }))
        (run_root / "final_review_summary.json").write_text(json.dumps({
            "status": "review_complete",
            "raw_control_status": "complete",
        }))
        (run_root / "review_history.json").write_text(json.dumps([{
            "partition": {"sets": [{"set_id": "C1", "member_ids": ["p1", "p2"]}]},
            "router_plan": {"actions": [{"action": action, "target_ids": ["C1"]}]},
            "round_evidence": [
                {"tool_name": "pathway_enrichment", "metrics": {"per_set_rna_pathway_enrichment": {
                    "C1": {"summary": {"significant_q05": 4}}
                }}},
                {"tool_name": "mutation_enrichment", "metrics": {"per_set_wxs_feature_enrichment": {
                    "C1": {"summary": {"significant_q05": 2}}
                }}},
                {"tool_name": "cnv_characterization", "metrics": {"per_comparison_cnv": {
                    "C1_vs_rest": {
                        "continuous": {"summary": {"significant_q05": 1}},
                        "gain_loss": {"summary": {"significant_q05": 3}},
                    }
                }}},
                {"tool_name": "multimodal_consistency_check", "artifact_paths": {"run": str(repeat)}, "metrics": {
                    "cross_modal_consistency": {"per_set": {"C1": {
                        "modality_support": {name: {"median_silhouette": 0.1} for name in ("ct", "wsi", "rna", "genomic")},
                        "internal_structure": {"fused_binary_probe": {"median_silhouette": 0.02, "resampling": {
                            "median_resample_ari": 0.8, "pac": 0.2,
                        }}},
                    }}}
                }},
                {"tool_name": "confound_test", "metrics": {"sets": {"C1": [
                    {"q_value": 0.01}, {"q_value": 0.2}
                ]}}},
            ],
        }]))

    summary = audit_router_repeatability(tmp_path, (3,), (1, 2, 3), tmp_path / "audit")
    row = summary["rows"][0]
    assert summary["complete_run_count"] == 3
    assert summary["discordant_set_count"] == 1
    assert row["decision_agreement_fraction"] == 2 / 3
    assert row["raw_evidence_equal"] is True
    evidence = (tmp_path / "audit" / "router_discordant_evidence.csv").read_text()
    assert "repeat1_rna_significant_count" in evidence
    assert "repeat1_cnv_significant_count" in evidence
