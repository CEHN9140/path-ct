from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.subtype_review_runtime import (
    attach_global_figures_to_reports,
    build_pipeline_output,
    build_review_global_figures,
    cluster_labels,
)


class SubtypeReviewFigureTest(unittest.TestCase):
    def test_build_review_global_figures_creates_core_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            output_root = root / "out"
            consensus_dir = output_root / "candidate_subtype" / "consensus_cluster"
            consensus_dir.mkdir(parents=True)
            np.save(
                consensus_dir / "best_consensus_matrix.npy",
                np.array(
                    [
                        [1, 0.9, 0.1, 0.1],
                        [0.9, 1, 0.2, 0.2],
                        [0.1, 0.2, 1, 0.8],
                        [0.1, 0.2, 0.8, 1],
                    ],
                    dtype=float,
                ),
            )
            (consensus_dir / "consensus_scatter_coordinates.json").write_text(
                '[{"case_id":"A","cluster_label":0,"x":0.0,"y":0.0},'
                '{"case_id":"B","cluster_label":0,"x":0.1,"y":0.0},'
                '{"case_id":"C","cluster_label":1,"x":1.0,"y":1.0},'
                '{"case_id":"D","cluster_label":1,"x":1.1,"y":1.0}]',
                encoding="utf-8",
            )
            rna_path = root / "rna.csv"
            wxs_path = root / "wxs.csv"
            with rna_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(
                    [
                        ["case_id", "HALLMARK_A", "HALLMARK_B", "HALLMARK_C"],
                        ["A", "3", "1", "2"],
                        ["B", "4", "1", "2"],
                        ["C", "1", "4", "2"],
                        ["D", "1", "5", "2"],
                    ]
                )
            with wxs_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(
                    [
                        ["case_id", "VHL", "PBRM1", "BAP1"],
                        ["A", "1", "0", "1"],
                        ["B", "1", "0", "0"],
                        ["C", "0", "1", "0"],
                        ["D", "0", "1", "0"],
                    ]
                )
            patient_states = [
                {
                    "case_id": "A",
                    "inventory": {
                        "Clinical": {
                            "diagnoses": [{"days_to_last_follow_up": 100}],
                            "demographic": {"vital_status": "Dead", "days_to_death": 40},
                        }
                    },
                    "ct_evidence": {"features": [1, 1]},
                    "wsi_evidence": {"features": [1, 0]},
                    "omics_evidence": {
                        "rna_features": [3, 1, 2],
                        "wxs_features": [1, 0, 1],
                        "rna_pathway_feature_path": str(rna_path),
                        "wxs_full_feature_path": str(wxs_path),
                    },
                },
                {
                    "case_id": "B",
                    "inventory": {
                        "Clinical": {
                            "diagnoses": [{"days_to_last_follow_up": 120}],
                            "demographic": {"vital_status": "Alive"},
                        }
                    },
                    "ct_evidence": {"features": [1.1, 1]},
                    "wsi_evidence": {"features": [1, 0.1]},
                    "omics_evidence": {
                        "rna_features": [4, 1, 2],
                        "wxs_features": [1, 0, 0],
                        "rna_pathway_feature_path": str(rna_path),
                        "wxs_full_feature_path": str(wxs_path),
                    },
                },
                {
                    "case_id": "C",
                    "inventory": {
                        "Clinical": {
                            "diagnoses": [{"days_to_last_follow_up": 130}],
                            "demographic": {"vital_status": "Alive"},
                        }
                    },
                    "ct_evidence": {"features": [4, 4]},
                    "wsi_evidence": {"features": [0, 1]},
                    "omics_evidence": {
                        "rna_features": [1, 4, 2],
                        "wxs_features": [0, 1, 0],
                        "rna_pathway_feature_path": str(rna_path),
                        "wxs_full_feature_path": str(wxs_path),
                    },
                },
                {
                    "case_id": "D",
                    "inventory": {
                        "Clinical": {
                            "diagnoses": [{"days_to_last_follow_up": 160}],
                            "demographic": {"vital_status": "Dead", "days_to_death": 140},
                        }
                    },
                    "ct_evidence": {"features": [4.1, 4]},
                    "wsi_evidence": {"features": [0.1, 1]},
                    "omics_evidence": {
                        "rna_features": [1, 5, 2],
                        "wxs_features": [0, 1, 0],
                        "rna_pathway_feature_path": str(rna_path),
                        "wxs_full_feature_path": str(wxs_path),
                    },
                },
            ]
            cluster_states = [
                {
                    "cluster_id": "C1",
                    "member_ids": ["A", "B"],
                    "report_draft": {
                        "final_evidence_matrix": {
                            "biological_support": {
                                "metrics": {
                                    "rna_pathway_enrichment": {"preview_rows": [
                                        {
                                            "candidate_set_id": "C1",
                                            "pathway": "HALLMARK_A",
                                            "delta_mean_score": 1.2,
                                            "standardized_mean_difference": 2.0,
                                            "q_value": 0.01,
                                        }
                                    ]},
                                    "wxs_gene_enrichment": {"preview_rows": [
                                        {
                                            "candidate_set_id": "C1",
                                            "gene": "VHL",
                                            "delta_frequency": 1.0,
                                            "q_value": 0.01,
                                        }
                                    ]},
                                    "wxs_pathway_enrichment": {"preview_rows": [
                                        {
                                            "candidate_set_id": "C1",
                                            "pathway": "KEGG_A",
                                            "gene_set_collection": "KEGG",
                                            "delta_frequency": 0.8,
                                            "q_value": 0.02,
                                            "evidence_role": "primary",
                                        },
                                        {
                                            "candidate_set_id": "C2",
                                            "pathway": "REACTOME_A",
                                            "gene_set_collection": "REACTOME",
                                            "delta_frequency": -0.7,
                                            "q_value": 0.03,
                                            "evidence_role": "validation",
                                        }
                                    ]},
                                }
                            }
                        }
                    },
                },
                {"cluster_id": "C2", "member_ids": ["C", "D"]},
            ]

            figures = build_review_global_figures(
                str(output_root), cluster_states, patient_states
            )

            self.assertIn("consensus_matrix_heatmap", figures)
            self.assertIn("integrated_snf_embedding_scatter", figures)
            self.assertIn("modality_embedding_4panel", figures)
            self.assertIn("rna_hallmark_ssgsea_bubble", figures)
            self.assertIn("rna_hallmark_ssgsea_heatmap", figures)
            self.assertIn("rna_subtype_signature_dotplot", figures)
            self.assertIn("rna_subtype_defining_hallmark_heatmap", figures)
            self.assertIn("rna_top_pathway_boxplots", figures)
            self.assertIn("subtype_evidence_score_panel", figures)
            self.assertIn("mutation_gene_oncoplot", figures)
            self.assertIn("mutation_kegg_pathway_bubble", figures)
            self.assertIn("mutation_reactome_pathway_bubble", figures)
            self.assertIn("mutation_pathway_bubble", figures)
            self.assertIn("survival_global_km", figures)
            self.assertNotIn("driver_oncoplot", figures)
            for path in figures.values():
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)
            cluster_dir = output_root / "subtype_review" / "C1" / "figures"
            self.assertTrue((cluster_dir / "cluster_member_consensus_support.png").exists())
            self.assertTrue((cluster_dir / "selected_pathway_boxplot.png").exists())
            attach_global_figures_to_reports(cluster_states, figures)
            report_figures = cluster_states[0]["report_draft"]["figures"]
            self.assertIn("consensus_matrix_heatmap", report_figures["set_reliability"])
            self.assertIn("modality_embedding_4panel", report_figures["multimodal_support"])
            self.assertIn("mutation_gene_oncoplot", report_figures["biological_support"])
            self.assertIn("rna_subtype_defining_hallmark_heatmap", report_figures["biological_support"])
            self.assertIn("subtype_evidence_score_panel", report_figures["biological_support"])
            self.assertIn("mutation_kegg_pathway_bubble", report_figures["biological_support"])
            self.assertIn("mutation_reactome_pathway_bubble", report_figures["biological_support"])
            output = build_pipeline_output(
                patient_states=patient_states,
                candidate_clusters=[],
                cluster_states=cluster_states,
                patient_store_paths={},
                cluster_store_paths={},
                graph_paths={},
                output_root=str(output_root),
            )
            summary_figures = output["final_review_summary"]["all_cluster_reports"][0]["figures"]
            self.assertIn("consensus_matrix_heatmap", summary_figures["set_reliability"])

    def test_build_review_global_figures_returns_empty_without_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            figures = build_review_global_figures(str(Path(temp_dir) / "out"), [], [])
            self.assertEqual(figures, {})

    def test_active_labels_skip_absorbed_split_and_drop_records(self) -> None:
        labels = cluster_labels(
            [
                {"cluster_id": "C1", "member_ids": ["A"], "final_action": "accept"},
                {"cluster_id": "C1_old", "member_ids": ["A"], "absorbed_into": "C1"},
                {"cluster_id": "C2", "member_ids": ["B"], "final_action": "split"},
                {"cluster_id": "C3", "member_ids": ["C"], "status": "drop"},
                {"cluster_id": "C2_S1", "member_ids": ["B"], "status": "under_review"},
            ]
        )

        self.assertEqual(labels, {"A": "C1", "B": "C2_S1"})

    def test_global_figure_refresh_removes_stale_known_pngs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            output_root = Path(temp_dir) / "out"
            figure_dir = output_root / "subtype_review" / "global" / "figures"
            figure_dir.mkdir(parents=True)
            stale_path = figure_dir / "mutation_reactome_pathway_bubble.png"
            stale_path.write_bytes(b"stale")

            figures = build_review_global_figures(
                str(output_root),
                [{"cluster_id": "C1", "member_ids": ["A"], "final_action": "accept"}],
                [{"case_id": "A"}],
            )

            self.assertEqual(figures, {})
            self.assertFalse(stale_path.exists())


if __name__ == "__main__":
    unittest.main()
