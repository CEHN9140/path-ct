from __future__ import annotations

import json

from scripts_2026_8_17.analyze_structural_action_opportunities import audit_experiment


def test_audit_exports_split_and_merge_metrics_with_report_mention(tmp_path):
    run_root = tmp_path / "run1" / "K2"
    run_root.mkdir(parents=True)
    history = [{
        "round": 1,
        "partition_signature": "p1",
        "partition": {"sets": [
            {"set_id": "C1", "member_ids": ["p1", "p2"]},
            {"set_id": "C2", "member_ids": ["p3", "p4"]},
        ]},
        "router_plan": {"actions": [
            {"action": "accept", "target_ids": ["C1"]},
            {"action": "drop", "target_ids": ["C2"]},
        ]},
        "evidence_reports": [{
            "dimension": "cross_modal_consistency",
            "scope": "set_identity",
            "target_ids": ["C1"],
            "observations": [{"metric": "boundary", "finding": "C1 versus C2 boundary"}],
        }],
        "round_evidence": [{
            "tool_name": "multimodal_consistency_check",
            "full_metrics": {"structural_characterization": {
                "internal_structure_by_set": {
                    "C1": {"member_n": 2, "fused_binary_probe": {
                        "child_sizes": [1, 1], "median_silhouette": 0.2,
                        "normalized_cut": 0.3, "resampling": {
                            "median_resample_ari": 0.8,
                            "consensus_separation": 0.7,
                            "pac": 0.1,
                            "degenerate_resample_fraction": 0.0,
                        }}, "probe_support_by_modality": {
                            name: {"median_silhouette": 0.1}
                            for name in ("ct", "wsi", "rna", "genomic")
                        }},
                    "C2": {"member_n": 2},
                },
                "boundary_by_pair": {
                    "C1+C2": {"fused": {
                        "pair_median_silhouette": 0.05,
                        "left_boundary_separation": 0.4,
                        "right_boundary_separation": 0.3,
                        "left_median_margin": 0.2,
                        "right_median_margin": 0.1,
                    }, "modalities": {
                        name: {"pair_median_silhouette": 0.01,
                               "left_boundary_separation": 0.2,
                               "right_boundary_separation": 0.1}
                        for name in ("ct", "wsi", "rna", "genomic")
                    }},
                },
            }},
        }],
    }]
    (run_root / "review_history.json").write_text(json.dumps(history), encoding="utf-8")
    (run_root / "final_review_summary.json").write_text(
        json.dumps({"status": "review_complete", "raw_control_status": "complete"}),
        encoding="utf-8",
    )

    summary = audit_experiment(tmp_path, [2], [1], tmp_path / "audit")

    split_rows = json.loads((tmp_path / "audit" / "split_opportunities.json").read_text())
    merge_rows = json.loads((tmp_path / "audit" / "merge_opportunities.json").read_text())
    assert split_rows[0]["fused_probe_silhouette"] == 0.2
    assert len(split_rows[0]["partition_id"]) == 12
    assert "partition_signature" not in split_rows[0]
    assert merge_rows[0]["pair_metrics_available_in_raw_artifact"] is True
    assert merge_rows[0]["pair_mentioned_in_verifier_report"] is True
    assert summary["audited_set_count"] == 2
    assert summary["audited_pair_count"] == 1
    assert summary["pair_metrics_by_k"]["2"]["fused_pair_silhouette"]["median"] == 0.05
