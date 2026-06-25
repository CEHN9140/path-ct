from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.tool_confound_test import tool_confound_test
from tools.tool_known_label_echo_test import tool_known_label_echo_test
from tools.tool_multimodal_consistency_check import tool_multimodal_consistency_check
from tools.tool_mutation_enrichment import tool_mutation_enrichment, wxs_feature_path
from tools.tool_pathway_enrichment import tool_pathway_enrichment
from tools.tool_stability_check import tool_stability_check
from tools.tool_survival_analysis import tool_survival_analysis
from tools.subtype_review_common import bh_fdr, standardized_mean_difference, tool_parameters
from utils.subtype_review_runtime import build_review_global_figures


def clinical(stage: str, grade: str, vital: str, death: int | None = None) -> dict:
    return {
        "diagnoses": [
            {
                "ajcc_pathologic_stage": stage,
                "ajcc_pathologic_t": "T1" if "I" in stage else "T3",
                "ajcc_pathologic_n": "N0",
                "ajcc_pathologic_m": "M0",
                "tumor_grade": grade,
                "days_to_last_follow_up": 120,
                "year_of_diagnosis": 2020,
            }
        ],
        "demographic": {
            "vital_status": vital,
            "days_to_death": death,
            "age_at_index": 60,
            "gender": "male",
            "race": "white",
        },
    }


def empty_clinical(vital: str = "Alive") -> dict:
    return {"diagnoses": [{}], "demographic": {"vital_status": vital, "age_at_index": 60}}


class SubtypeReviewToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir="/tmp")
        self.root = Path(self.temp.name)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        self.output_root = str(self.root / "out")
        self.hallmark_gmt_path = self.root / "hallmark.gmt"
        self.kegg_gmt_path = self.root / "kegg.gmt"
        self.reactome_gmt_path = self.root / "reactome.gmt"
        self.hallmark_gmt_path.write_text(
            "\n".join(
                [
                    "HALLMARK_G1_HIGH\tna\tG1\tG3",
                    "HALLMARK_G2_HIGH\tna\tG2\tG3",
                ]
            ),
            encoding="utf-8",
        )
        self.kegg_gmt_path.write_text(
            "KEGG_VHL_SIGNALING\tna\tVHL\tX\n",
            encoding="utf-8",
        )
        self.reactome_gmt_path.write_text(
            "REACTOME_PBRM1_COMPLEX\tna\tPBRM1\tX\n",
            encoding="utf-8",
        )
        (self.config_dir / "subtype_review.yaml").write_text(
            "\n".join(
                [
                    "artifact_policy:",
                    "  save_figures: true",
                    "visualization:",
                    "  dpi: 80",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (self.config_dir / "subtype_review_tools.yaml").write_text(
            "\n".join(
                [
                    "mutation:",
                    "  min_cluster_mutated_cases: 1",
                    "  pathway_gene_sets:",
                    f"    KEGG:\n      role: primary\n      path: {self.kegg_gmt_path}",
                    f"    REACTOME:\n      role: validation\n      path: {self.reactome_gmt_path}",
                    "rna:",
                    "  max_reported_features: 5",
                    "  min_pathway_overlap: 1",
                    f"  pathway_gene_sets_path: {self.hallmark_gmt_path}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self.wxs_path = self.root / "wxs.csv"
        self.wxs_full_path = self.root / "wxs_full.csv"
        self.rna_path = self.root / "rna.csv"
        with self.wxs_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(
                [
                    ["case_id", "VHL", "PBRM1", "X"],
                    ["A", "1", "0", "1"],
                    ["B", "1", "0", "0"],
                    ["C", "0", "1", "0"],
                    ["D", "0", "1", "0"],
                ]
            )
        with self.wxs_full_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(
                [
                    ["case_id", "VHL", "PBRM1", "X", "FULL_ONLY"],
                    ["A", "1", "0", "1", "1"],
                    ["B", "1", "0", "0", "1"],
                    ["C", "0", "1", "0", "0"],
                    ["D", "0", "1", "0", "0"],
                ]
            )
        with self.rna_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(
                [
                    ["case_id", "G1", "G2", "G3"],
                    ["A", "5", "1", "1"],
                    ["B", "6", "1", "1"],
                    ["C", "1", "5", "1"],
                    ["D", "1", "6", "1"],
                ]
            )
        self.states = {
            "A": self.state(clinical("Stage I", "G1", "Dead", 50), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1]),
            "B": self.state(clinical("Stage I", "G1", "Alive"), [1.1, 1], [0.9, 0], [6, 1, 1], [1, 0, 0]),
            "C": self.state(clinical("Stage III", "G3", "Alive"), [4, 4], [0, 1], [1, 5, 1], [0, 1, 0]),
            "D": self.state(clinical("Stage III", "G3", "Alive"), [4.1, 4], [0, 0.9], [1, 6, 1], [0, 1, 0]),
        }
        self.cluster = {
            "cluster_id": "C1",
            "member_ids": ["A", "B"],
            "consensus": {
                "matrix_path": str(self.root / "cons.npy"),
                "metadata_path": str(self.root / "meta.json"),
                "partition_count": 10,
            },
        }
        self.all_clusters = [self.cluster, {"cluster_id": "C2", "member_ids": ["C", "D"]}]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_subtype_review_tool_parameters_are_loaded_from_separate_yaml(self) -> None:
        self.assertNotIn("driver_genes", tool_parameters(str(self.config_dir), "mutation"))
        self.assertEqual(tool_parameters(str(self.config_dir), "rna")["max_reported_features"], 5)

    def state(self, clinical_payload, ct, wsi, rna, wxs, manufacturer="GE"):
        return {
            "inventory": {"Clinical": clinical_payload, "CT": [{"Manufacturer": manufacturer}]},
            "ct_evidence": {"features": ct},
            "wsi_evidence": {"features": wsi},
            "omics_evidence": {
                "rna_feature_path": str(self.rna_path),
                "wxs_feature_path": str(self.wxs_path),
                "wxs_full_feature_path": str(self.wxs_full_path),
                "rna_features": rna,
                "wxs_features": wxs,
            },
        }

    def assert_tool_shape(self, result):
        self.assertIn(result["status"], {"success", "warning", "missing", "failure"})
        self.assertIn("support_level", result["results"])
        self.assertIn("concern_level", result["results"])
        if importlib.util.find_spec("matplotlib"):
            for path in result["artifacts"].get("figures", {}).values():
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_validation_tools_return_structured_metrics(self):
        consensus = np.array([[1, 0.9, 0.1, 0.1], [0.9, 1, 0.2, 0.2], [0.1, 0.2, 1, 0.8], [0.1, 0.2, 0.8, 1]], dtype=float)
        np.save(self.root / "cons.npy", consensus)
        (self.root / "meta.json").write_text('{"patient_ids":["A","B","C","D"]}', encoding="utf-8")
        results = []
        for tool in [
            tool_stability_check,
            tool_multimodal_consistency_check,
            tool_survival_analysis,
            tool_mutation_enrichment,
            tool_pathway_enrichment,
            tool_confound_test,
        ]:
            result = tool(self.cluster, self.states, self.output_root, str(self.config_dir), self.all_clusters)
            self.assert_tool_shape(result)
            results.append(result)
        stability_metrics = results[0]["results"]["metrics"]
        self.assertEqual(
            set(stability_metrics),
            {"set_reliability_global_consensus", "set_reliability_set_consensus"},
        )
        self.assertEqual(results[0]["artifacts"]["figures"], {})
        self.assertEqual(
            stability_metrics["set_reliability_global_consensus"]["candidate_set_sizes"],
            {"C1": 2, "C2": 2},
        )
        self.assertEqual(
            stability_metrics["set_reliability_global_consensus"]["total_candidate_set_n"],
            4,
        )
        c1_reliability = stability_metrics["set_reliability_set_consensus"]["C1"]
        self.assertEqual(c1_reliability["set_n"], 2)
        self.assertEqual(c1_reliability["matrix_available_n"], 2)
        self.assertAlmostEqual(c1_reliability["within_consensus_mean"], 0.9)
        self.assertAlmostEqual(c1_reliability["outside_consensus_mean"], 0.15)
        self.assertEqual(c1_reliability["nearest_neighbor_candidate_set_id"], "C2")
        self.assertAlmostEqual(c1_reliability["nearest_other_consensus"], 0.15)
        self.assertAlmostEqual(c1_reliability["consensus_margin"], 0.75)
        self.assertIn("A", c1_reliability["per_member_consensus_support"])
        self.assertIn("A", c1_reliability["per_member_silhouette"])
        self.assertNotIn("cluster_consensus", stability_metrics)
        self.assertNotIn("within_consensus_mean", stability_metrics)
        self.assertNotIn("per_member_support", stability_metrics)
        multimodal_metrics = results[1]["results"]["metrics"]
        self.assertEqual(
            set(multimodal_metrics),
            {"crossmodal_global_alignment", "crossmodal_set_alignment"},
        )
        self.assertEqual(len(multimodal_metrics["crossmodal_global_alignment"]), 6)
        self.assertEqual(len(multimodal_metrics["crossmodal_set_alignment"]["C1"]), 6)
        first_pair = multimodal_metrics["crossmodal_global_alignment"][0]
        self.assertEqual(first_pair["scope"], "global")
        self.assertIsNone(first_pair["candidate_set_id"])
        self.assertIn("spearman_distance_correlation", first_pair)
        self.assertIn("mantel_p_value", first_pair)
        self.assertIn("cka_similarity_alignment", first_pair)
        self.assertNotIn("q_value", first_pair)
        self.assertEqual(results[1]["artifacts"]["figures"], {})
        self.assertNotIn("supported_modality_count", results[1]["results"]["metrics"])
        self.assertNotIn("contradictory_modality_count", results[1]["results"]["metrics"])
        self.assertNotIn("dominant_modality", results[1]["results"]["metrics"])
        self.assertNotIn("modality_metrics", results[1]["results"]["metrics"])
        survival_metrics = results[2]["results"]["metrics"]
        self.assertEqual(
            set(survival_metrics),
            {"survival_global_association", "survival_set_association"},
        )
        self.assertEqual(survival_metrics["survival_global_association"]["endpoint"], "OS")
        self.assertIn("logrank_p_value", survival_metrics["survival_global_association"])
        self.assertIn("hazard_ratio", survival_metrics["survival_set_association"]["C1"])
        self.assertEqual(results[2]["artifacts"]["figures"], {})
        self.assertEqual(wxs_feature_path(self.states), str(self.wxs_full_path))
        mutation_metrics = results[3]["results"]["metrics"]
        self.assertEqual(set(mutation_metrics), {"wxs_gene_enrichment", "wxs_pathway_enrichment"})
        self.assertEqual(results[3]["artifacts"]["figures"], {})
        self.assertEqual(len(mutation_metrics["wxs_gene_enrichment"]), 8)
        self.assertEqual(len(mutation_metrics["wxs_pathway_enrichment"]), 4)
        vhl_c1 = next(
            row
            for row in mutation_metrics["wxs_gene_enrichment"]
            if row["candidate_set_id"] == "C1" and row["gene"] == "VHL"
        )
        self.assertNotIn("driver_gene", vhl_c1)
        self.assertEqual(vhl_c1["set_mutated_n"], 2)
        self.assertEqual(vhl_c1["set_total_n"], 2)
        self.assertEqual(vhl_c1["rest_mutated_n"], 0)
        self.assertEqual(vhl_c1["rest_total_n"], 2)
        self.assertEqual(vhl_c1["set_frequency"], 1.0)
        self.assertGreater(vhl_c1["delta_frequency"], 0)
        pathway_c1 = next(
            row
            for row in mutation_metrics["wxs_pathway_enrichment"]
            if row["candidate_set_id"] == "C1" and row["pathway"] == "KEGG_VHL_SIGNALING"
        )
        self.assertEqual(pathway_c1["gene_set_collection"], "KEGG")
        self.assertEqual(pathway_c1["evidence_role"], "primary")
        self.assertEqual(pathway_c1["pathway_gene_count"], 2)
        self.assertEqual(pathway_c1["set_hit_n"], 2)
        self.assertIn("q_value", pathway_c1)
        reactome_c1 = next(
            row
            for row in mutation_metrics["wxs_pathway_enrichment"]
            if row["candidate_set_id"] == "C1" and row["pathway"] == "REACTOME_PBRM1_COMPLEX"
        )
        self.assertEqual(reactome_c1["gene_set_collection"], "REACTOME")
        self.assertEqual(reactome_c1["evidence_role"], "validation")
        self.assertNotIn("top_enriched_driver_genes", mutation_metrics)
        self.assertNotIn("top_mutation_pathways", mutation_metrics)
        self.assertNotIn("significant_driver_gene_count", mutation_metrics)
        self.assertNotIn("top_enriched_all_genes", results[3]["results"]["metrics"])
        self.assertNotIn("mutation_profile_similarity_by_cluster", results[3]["results"]["metrics"])
        pathway_metrics = results[4]["results"]["metrics"]
        self.assertEqual(set(pathway_metrics), {"rna_pathway_enrichment"})
        self.assertEqual(results[4]["artifacts"]["figures"], {})
        self.assertEqual(len(pathway_metrics["rna_pathway_enrichment"]), 4)
        rna_c1 = next(
            row
            for row in pathway_metrics["rna_pathway_enrichment"]
            if row["candidate_set_id"] == "C1" and row["pathway"] == "HALLMARK_G1_HIGH"
        )
        self.assertEqual(rna_c1["pathway_gene_count"], 2)
        self.assertEqual(rna_c1["set_available_n"], 2)
        self.assertEqual(rna_c1["rest_available_n"], 2)
        self.assertGreater(rna_c1["delta_mean_score"], 0)
        self.assertIn("standardized_mean_difference", rna_c1)
        self.assertIn("q_value", rna_c1)
        self.assertNotIn("top_rna_pathways", pathway_metrics)
        self.assertNotIn("significant_pathway_count", pathway_metrics)
        self.assertNotIn("signature_category", pathway_metrics)
        self.assertNotIn("pathway_profile_similarity_by_cluster", results[4]["results"]["metrics"])
        confound_metrics = results[5]["results"]["metrics"]
        self.assertEqual(
            set(confound_metrics),
            {"confounder_global_association", "confounder_set_association"},
        )
        self.assertEqual(
            set(confound_metrics["confounder_global_association"]),
            {"gender", "race", "ct_manufacturer", "age_at_index", "year_of_diagnosis"},
        )
        self.assertIn("contingency_table", confound_metrics["confounder_global_association"]["gender"])
        self.assertIn("cramers_v", confound_metrics["confounder_global_association"]["ct_manufacturer"])
        self.assertIn("max_pairwise_smd", confound_metrics["confounder_global_association"]["age_at_index"])
        self.assertIn("standardized_mean_difference", confound_metrics["confounder_set_association"]["C1"]["age_at_index"])
        manufacturer_rows = confound_metrics["confounder_set_association"]["C1"]["ct_manufacturer"]
        self.assertIn("GE", manufacturer_rows)
        self.assertIn("delta_fraction", manufacturer_rows["GE"])
        self.assertNotIn("dominant_level", manufacturer_rows["GE"])
        self.assertNotIn("available_fields", confound_metrics)
        self.assertNotIn("field_results", confound_metrics)
        self.assertNotIn("confounder_summary", confound_metrics)

        cluster = {**self.cluster, "validation_results": {item["tool_name"]: item for item in results}}
        echo = tool_known_label_echo_test(cluster, self.states, self.output_root, str(self.config_dir), self.all_clusters)
        self.assert_tool_shape(echo)
        metrics = echo["results"]["metrics"]
        self.assertEqual(
            set(metrics),
            {"known_label_global_association", "known_label_set_enrichment"},
        )
        self.assertEqual(set(metrics["known_label_global_association"]), {"stage", "grade"})
        stage = metrics["known_label_global_association"]["stage"]
        self.assertEqual(stage["available_n"], 4)
        self.assertEqual(stage["missing_n"], 0)
        self.assertIn("contingency_table", stage)
        self.assertIn("chi_square_p_value", stage)
        self.assertIn("low_expected_count", stage)
        self.assertIn("cramers_v", stage)
        self.assertNotIn("adjusted_rand_index", stage)
        self.assertNotIn("adjusted_mutual_info", stage)
        self.assertNotIn("normalized_mutual_info", stage)
        self.assertNotIn("global_purity", stage)
        set_stage = metrics["known_label_set_enrichment"]["C1"]["stage"]
        self.assertEqual(set_stage["candidate_set_id"], "C1")
        self.assertEqual(set_stage["dominant_level"], "I")
        self.assertEqual(set_stage["overlap_count"], 2)
        self.assertEqual(set_stage["set_fraction"], 1.0)
        self.assertNotIn("set_purity", set_stage)
        self.assertEqual(set_stage["level_recall"], 1.0)
        self.assertIn("q_value", set_stage)
        self.assertNotIn("known_label_global_redundancy", metrics)
        self.assertNotIn("known_label_set_redundancy", metrics)
        self.assertNotIn("known_label_associations", metrics)
        self.assertNotIn("per_label_enrichment", metrics)
        self.assertNotIn("clinical_echo_items", echo["results"]["metrics"])
        self.assertNotIn("max_clinical_echo_strength", echo["results"]["metrics"])

    def test_known_label_echo_distinguishes_repetition_subset_and_missing_labels(self) -> None:
        states = {
            "A": self.state(clinical("Stage I", "G1", "Dead"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1]),
            "B": self.state(clinical("Stage I", "G1", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1]),
            "C": self.state(clinical("Stage I", "G2", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1]),
            "D": self.state(clinical("Stage II", "G2", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1]),
            "E": self.state(clinical("Stage III", "G3", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1]),
            "F": self.state(empty_clinical(), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1]),
        }
        cluster_a = {"cluster_id": "C1", "member_ids": ["A", "B"]}
        cluster_b = {"cluster_id": "C2", "member_ids": ["C", "D", "E", "F"]}
        result = tool_known_label_echo_test(
            cluster_a,
            states,
            self.output_root,
            str(self.config_dir),
            [cluster_a, cluster_b],
        )
        metrics = result["results"]["metrics"]
        stage_global = metrics["known_label_global_association"]["stage"]
        self.assertEqual(stage_global["available_n"], 5)
        self.assertEqual(stage_global["missing_n"], 1)
        self.assertEqual(sum(sum(row.values()) for row in stage_global["contingency_table"].values()), 5)
        self.assertTrue(stage_global["low_expected_count"])
        self.assertIsNotNone(stage_global["chi_square_p_value"])
        c1_stage = metrics["known_label_set_enrichment"]["C1"]["stage"]
        self.assertEqual(c1_stage["dominant_level"], "I")
        self.assertEqual(c1_stage["set_fraction"], 1.0)
        self.assertEqual(c1_stage["level_recall"], 0.666667)
        c2_stage = metrics["known_label_set_enrichment"]["C2"]["stage"]
        self.assertLess(c2_stage["set_fraction"], 1.0)
        self.assertEqual(c2_stage["missing_n"], 1)

    def test_multimodal_outputs_only_crossmodal_alignment_tables(self):
        result = tool_multimodal_consistency_check(
            self.cluster,
            self.states,
            self.output_root,
            str(self.config_dir),
            self.all_clusters,
        )

        metrics = result["results"]["metrics"]
        self.assertEqual(
            set(metrics),
            {"crossmodal_global_alignment", "crossmodal_set_alignment"},
        )
        self.assertEqual(result["artifacts"]["figures"], {})
        self.assertNotIn("modality_coverage", metrics)
        self.assertNotIn("modality_separation", metrics)
        self.assertNotIn("global_single_modality_alignment", metrics)
        c1_rows = metrics["crossmodal_set_alignment"]["C1"]
        self.assertEqual(len(c1_rows), 6)
        self.assertTrue(all(row["available_n"] == 2 for row in c1_rows))
        self.assertTrue(all(row["missing_reason"] == "insufficient_common_cases" for row in c1_rows))

    def test_runtime_generates_review_global_figures(self):
        consensus = np.array(
            [
                [1, 0.9, 0.1, 0.1],
                [0.9, 1, 0.2, 0.2],
                [0.1, 0.2, 1, 0.8],
                [0.1, 0.2, 0.8, 1],
            ],
            dtype=float,
        )
        np.save(self.root / "cons.npy", consensus)
        (self.root / "meta.json").write_text(
            '{"patient_ids":["A","B","C","D"]}',
            encoding="utf-8",
        )
        consensus_dir = Path(self.output_root) / "candidate_subtype" / "consensus_cluster"
        consensus_dir.mkdir(parents=True)
        (consensus_dir / "consensus_scatter_coordinates.json").write_text(
            '[{"case_id":"A","cluster_label":0,"x":0.0,"y":0.0},'
            '{"case_id":"B","cluster_label":0,"x":0.1,"y":0.0},'
            '{"case_id":"C","cluster_label":1,"x":1.0,"y":1.0},'
            '{"case_id":"D","cluster_label":1,"x":1.1,"y":1.0}]',
            encoding="utf-8",
        )
        cluster_states = []
        for cluster in self.all_clusters:
            pathway_result = tool_pathway_enrichment(
                cluster,
                self.states,
                self.output_root,
                str(self.config_dir),
                self.all_clusters,
            )
            mutation_result = tool_mutation_enrichment(
                cluster,
                self.states,
                self.output_root,
                str(self.config_dir),
                self.all_clusters,
            )
            rows = pathway_result["results"]["metrics"].get("rna_pathway_enrichment", [])
            mutation_metrics = mutation_result["results"]["metrics"]
            cluster_states.append(
                {
                    **cluster,
                    "tool_results": [pathway_result, mutation_result],
                    "report_draft": {
                        "final_evidence_matrix": {
                            "biological_support": {
                                "metrics": {
                                    "rna_pathway_enrichment": rows,
                                    "wxs_gene_enrichment": mutation_metrics["wxs_gene_enrichment"],
                                    "wxs_pathway_enrichment": mutation_metrics["wxs_pathway_enrichment"],
                                }
                            }
                        }
                    },
                }
            )
        figures = build_review_global_figures(
            self.output_root,
            cluster_states,
            [
                {"case_id": case_id, **state}
                for case_id, state in self.states.items()
            ],
        )
        expected = {
            "consensus_matrix_heatmap",
            "integrated_snf_embedding_scatter",
            "modality_embedding_4panel",
            "rna_hallmark_ssgsea_bubble",
            "rna_hallmark_ssgsea_heatmap",
            "mutation_gene_oncoplot",
            "mutation_kegg_pathway_bubble",
            "mutation_pathway_bubble",
        }
        self.assertTrue(expected.issubset(figures))
        self.assertNotIn("driver_oncoplot", figures)
        for path in figures.values():
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)

    def test_biological_tools_keep_empty_tables_when_inputs_missing(self):
        states = {
            case_id: {"omics_evidence": {}}
            for case_id in ["A", "B", "C", "D"]
        }
        mutation = tool_mutation_enrichment(
            self.cluster,
            states,
            self.output_root,
            str(self.config_dir),
            self.all_clusters,
        )
        pathway = tool_pathway_enrichment(
            self.cluster,
            states,
            self.output_root,
            str(self.config_dir),
            self.all_clusters,
        )
        self.assertEqual(
            mutation["results"]["metrics"],
            {"wxs_gene_enrichment": [], "wxs_pathway_enrichment": []},
        )
        self.assertEqual(pathway["results"]["metrics"], {"rna_pathway_enrichment": []})
        self.assertEqual(mutation["artifacts"]["figures"], {})
        self.assertEqual(pathway["artifacts"]["figures"], {})

    def test_bh_fdr_matches_statsmodels(self):
        from statsmodels.stats.multitest import multipletests

        p_values = [0.01, 0.2, 0.03, 0.5]
        expected = multipletests(p_values, method="fdr_bh")[1]
        np.testing.assert_allclose(bh_fdr(p_values), expected)

    def test_standardized_mean_difference_uses_two_group_pooled_sd(self):
        smd = standardized_mean_difference([10, 12, 14], [1, 2, 3])
        pooled_sd = (((3 - 1) * 4 + (3 - 1) * 1) / 4) ** 0.5
        self.assertAlmostEqual(smd, (12 - 2) / pooled_sd)
        self.assertIsNone(standardized_mean_difference([1, 1], [1, 1]))

    def test_survival_reports_global_and_set_os_metrics_without_figures_or_q_values(self):
        result = tool_survival_analysis(
            self.cluster,
            self.states,
            self.output_root,
            str(self.config_dir),
            self.all_clusters,
        )
        metrics = result["results"]["metrics"]
        global_os = metrics["survival_global_association"]
        set_os = metrics["survival_set_association"]["C1"]
        self.assertEqual(global_os["endpoint"], "OS")
        self.assertEqual(global_os["available_n"], 4)
        self.assertEqual(global_os["missing_n"], 0)
        self.assertEqual(global_os["event_n"], 1)
        self.assertEqual(global_os["censored_n"], 3)
        self.assertEqual(global_os["per_set_n"], {"C1": 2, "C2": 2})
        self.assertEqual(global_os["per_set_event_n"], {"C1": 1, "C2": 0})
        self.assertIn("per_set_median_os_days", global_os)
        self.assertIn("logrank_p_value", global_os)
        self.assertEqual(set_os["candidate_set_id"], "C1")
        self.assertEqual(set_os["endpoint"], "OS")
        self.assertEqual(set_os["set_n"], 2)
        self.assertEqual(set_os["rest_n"], 2)
        self.assertEqual(set_os["set_event_n"], 1)
        self.assertEqual(set_os["rest_event_n"], 0)
        self.assertIn("hazard_ratio", set_os)
        self.assertIn("cox_p_value", set_os)
        self.assertIn("logrank_p_value", set_os)
        self.assertNotIn("q_value", set_os)
        self.assertNotIn("global_km", metrics)
        self.assertNotIn("cox_ph", metrics)
        self.assertNotIn("clinical_summary", metrics)
        self.assertEqual(result["artifacts"]["figures"], {})

    def test_survival_handles_censoring_missing_os_and_direction(self):
        states = {}

        def add_case(case_id, vital, death=None, follow=120):
            payload = clinical("Stage I", "G1", vital, death)
            payload["diagnoses"][0]["days_to_last_follow_up"] = follow
            states[case_id] = self.state(
                payload,
                [1, 1],
                [1, 0],
                [5, 1, 1],
                [1, 0, 1],
            )

        for case_id, death in [("A", 10), ("B", 40), ("C", 90)]:
            add_case(case_id, "Dead", death=death)
        for case_id, follow in [("D", 110), ("E", 160), ("F", 220)]:
            add_case(case_id, "Alive", follow=follow)
        for case_id, death in [("G", 70), ("H", 140), ("I", 210)]:
            add_case(case_id, "Dead", death=death)
        for case_id, follow in [("J", 220), ("K", 240), ("L", 260)]:
            add_case(case_id, "Alive", follow=follow)
        states["M"] = self.state(empty_clinical(), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1])
        cluster_a = {"cluster_id": "C1", "member_ids": ["A", "B", "C", "D", "E", "F"]}
        cluster_b = {"cluster_id": "C2", "member_ids": ["G", "H", "I", "J", "K", "L", "M"]}

        result = tool_survival_analysis(
            cluster_a,
            states,
            self.output_root,
            str(self.config_dir),
            [cluster_a, cluster_b],
        )
        metrics = result["results"]["metrics"]
        global_os = metrics["survival_global_association"]
        c1 = metrics["survival_set_association"]["C1"]

        self.assertEqual(global_os["available_n"], 12)
        self.assertEqual(global_os["missing_n"], 1)
        self.assertEqual(global_os["event_n"], 6)
        self.assertEqual(global_os["censored_n"], 6)
        self.assertGreater(c1["hazard_ratio"], 1.0)
        self.assertEqual(c1["direction"], "worse_survival_in_set")

    def test_multimodal_rna_uses_spearman_distance_for_crossmodal_pairs(self):
        result = tool_multimodal_consistency_check(
            self.cluster,
            self.states,
            self.output_root,
            str(self.config_dir),
            self.all_clusters,
        )
        rows = result["results"]["metrics"]["crossmodal_global_alignment"]
        rna_rows = [row for row in rows if "rna" in {row["modality_a"], row["modality_b"]}]
        self.assertEqual(len(rna_rows), 3)
        self.assertTrue(
            all(
                row["distance_metric_a"] == "spearman"
                or row["distance_metric_b"] == "spearman"
                for row in rna_rows
            )
        )

    def test_multimodal_alignment_detects_aligned_and_discordant_kernels(self):
        states = {}
        for case_id, angle in [
            ("A", 0.0),
            ("B", 0.1),
            ("C", 0.2),
            ("D", 1.2),
            ("E", 1.3),
            ("F", 1.4),
        ]:
            vector = [float(np.cos(angle)), float(np.sin(angle))]
            states[case_id] = self.state(
                clinical("Stage I", "G1", "Alive"),
                vector,
                vector,
                [vector[0], vector[1], 1.0],
                [1 if case_id in {"A", "C", "E"} else 0, 0],
            )
        cluster_a = {"cluster_id": "C1", "member_ids": ["A", "B", "C"]}
        cluster_b = {"cluster_id": "C2", "member_ids": ["D", "E", "F"]}

        result = tool_multimodal_consistency_check(
            cluster_a,
            states,
            self.output_root,
            str(self.config_dir),
            [cluster_a, cluster_b],
        )
        global_rows = result["results"]["metrics"]["crossmodal_global_alignment"]
        ct_wsi = next(
            row
            for row in global_rows
            if row["modality_a"] == "ct" and row["modality_b"] == "wsi"
        )
        ct_wxs = next(
            row
            for row in global_rows
            if row["modality_a"] == "ct" and row["modality_b"] == "wxs"
        )
        self.assertEqual(ct_wsi["available_n"], 6)
        self.assertGreater(ct_wsi["spearman_distance_correlation"], 0.9)
        self.assertGreater(ct_wsi["cka_similarity_alignment"], 0.9)
        self.assertLess(ct_wxs["cka_similarity_alignment"], ct_wsi["cka_similarity_alignment"])

    def test_multimodal_alignment_handles_constant_distance_matrix(self):
        states = {}
        for case_id in ["A", "B", "C"]:
            states[case_id] = self.state(
                clinical("Stage I", "G1", "Alive"),
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1, 0],
            )
        cluster = {"cluster_id": "C1", "member_ids": ["A", "B", "C"]}

        result = tool_multimodal_consistency_check(
            cluster,
            states,
            self.output_root,
            str(self.config_dir),
            [cluster],
        )
        row = next(
            item
            for item in result["results"]["metrics"]["crossmodal_global_alignment"]
            if item["modality_a"] == "ct" and item["modality_b"] == "wsi"
        )

        self.assertEqual(row["available_n"], 3)
        self.assertIsNone(row["spearman_distance_correlation"])
        self.assertIsNone(row["mantel_p_value"])
        self.assertIsNone(row["cka_similarity_alignment"])
        self.assertEqual(row["missing_reason"], "constant_distance_matrix")

    def test_confound_reports_only_full_global_and_set_metrics(self):
        result = tool_confound_test(
            self.cluster,
            self.states,
            self.output_root,
            str(self.config_dir),
            self.all_clusters,
        )
        metrics = result["results"]["metrics"]
        self.assertEqual(
            set(metrics["confounder_global_association"]),
            {"gender", "race", "ct_manufacturer", "age_at_index", "year_of_diagnosis"},
        )
        self.assertEqual(set(metrics["confounder_set_association"]), {"C1", "C2"})
        self.assertEqual(metrics["confounder_global_association"]["gender"]["available_n"], 4)
        self.assertEqual(metrics["confounder_global_association"]["gender"]["missing_n"], 0)
        self.assertIn("low_expected_count", metrics["confounder_global_association"]["race"])
        self.assertIn("q_value", metrics["confounder_global_association"]["age_at_index"])
        self.assertEqual(metrics["confounder_set_association"]["C1"]["age_at_index"]["direction"], None)
        manufacturer_rows = metrics["confounder_set_association"]["C1"]["ct_manufacturer"]
        self.assertEqual(set(manufacturer_rows), {"GE"})
        self.assertEqual(manufacturer_rows["GE"]["level"], "GE")
        self.assertIn("odds_ratio", manufacturer_rows["GE"])
        self.assertNotIn("dominant_level", manufacturer_rows["GE"])
        self.assertNotIn("stage_group", metrics["confounder_global_association"])
        self.assertNotIn("grade", metrics["confounder_global_association"])

    def test_confound_distinguishes_subset_numeric_and_missing_values(self):
        states = {
            "A": self.state(clinical("Stage I", "G1", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1], "GE"),
            "B": self.state(clinical("Stage I", "G1", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1], "GE"),
            "C": self.state(clinical("Stage II", "G2", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1], "SIEMENS"),
            "D": self.state(empty_clinical(), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1], "SIEMENS"),
            "E": self.state(clinical("Stage III", "G3", "Alive"), [1, 1], [1, 0], [5, 1, 1], [1, 0, 1], "PHILIPS"),
        }
        states["A"]["inventory"]["Clinical"]["demographic"]["age_at_index"] = 80
        states["B"]["inventory"]["Clinical"]["demographic"]["age_at_index"] = 82
        states["C"]["inventory"]["Clinical"]["demographic"]["age_at_index"] = 50
        states["D"]["inventory"]["Clinical"]["demographic"].pop("age_at_index", None)
        states["E"]["inventory"]["Clinical"]["demographic"]["age_at_index"] = 52
        cluster_a = {"cluster_id": "C1", "member_ids": ["A", "B"]}
        cluster_b = {"cluster_id": "C2", "member_ids": ["C", "D", "E"]}
        result = tool_confound_test(
            cluster_a,
            states,
            self.output_root,
            str(self.config_dir),
            [cluster_a, cluster_b],
        )
        metrics = result["results"]["metrics"]
        age_global = metrics["confounder_global_association"]["age_at_index"]
        self.assertEqual(age_global["available_n"], 4)
        self.assertEqual(age_global["missing_n"], 1)
        self.assertGreater(age_global["max_pairwise_smd"], 1.0)
        self.assertAlmostEqual(age_global["max_pairwise_smd"], 21.213203, places=5)
        c1_age = metrics["confounder_set_association"]["C1"]["age_at_index"]
        self.assertGreater(c1_age["standardized_mean_difference"], 1.0)
        self.assertAlmostEqual(c1_age["standardized_mean_difference"], 21.213203, places=5)
        self.assertEqual(c1_age["direction"], "higher_in_set")
        c1_manufacturer = metrics["confounder_set_association"]["C1"]["ct_manufacturer"]["GE"]
        self.assertEqual(c1_manufacturer["level"], "GE")
        self.assertEqual(c1_manufacturer["set_fraction"], 1.0)
        self.assertEqual(c1_manufacturer["rest_fraction"], 0.0)
        self.assertTrue(c1_manufacturer["sparse_level"])
        self.assertIn("SIEMENS", metrics["confounder_set_association"]["C1"]["ct_manufacturer"])
        self.assertIn("PHILIPS", metrics["confounder_set_association"]["C1"]["ct_manufacturer"])

    def test_stability_missing_consensus_keeps_fixed_metric_tables(self):
        cluster_a = {"cluster_id": "C1", "member_ids": ["A", "B"]}
        cluster_b = {"cluster_id": "C2", "member_ids": ["C", "D"]}
        result = tool_stability_check(
            cluster_a,
            self.states,
            self.output_root,
            str(self.config_dir),
            [cluster_a, cluster_b],
        )
        metrics = result["results"]["metrics"]
        self.assertEqual(
            set(metrics),
            {"set_reliability_global_consensus", "set_reliability_set_consensus"},
        )
        self.assertEqual(result["artifacts"]["figures"], {})
        self.assertEqual(metrics["set_reliability_global_consensus"]["candidate_set_count"], 2)
        self.assertEqual(metrics["set_reliability_global_consensus"]["matrix_available_n"], 0)
        self.assertEqual(metrics["set_reliability_global_consensus"]["matrix_missing_n"], 4)
        self.assertIsNotNone(metrics["set_reliability_global_consensus"]["missing_reason"])
        c1_row = metrics["set_reliability_set_consensus"]["C1"]
        self.assertEqual(c1_row["set_n"], 2)
        self.assertEqual(c1_row["matrix_available_n"], 0)
        self.assertEqual(c1_row["matrix_missing_n"], 2)
        self.assertIsNone(c1_row["within_consensus_mean"])
        self.assertEqual(c1_row["per_member_consensus_support"], {})
        self.assertIsNotNone(c1_row["missing_reason"])

    def test_invalid_feature_value_is_not_silently_zero_filled(self):
        with self.rna_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(
                [
                    ["case_id", "G1", "G2", "G3"],
                    ["A", "5", "bad", "1"],
                    ["B", "6", "1", "1"],
                    ["C", "1", "5", "1"],
                    ["D", "1", "6", "1"],
                ]
            )
        result = tool_pathway_enrichment(
            self.cluster,
            self.states,
            self.output_root,
            str(self.config_dir),
            self.all_clusters,
        )
        rows = result["results"]["metrics"]["rna_pathway_enrichment"]
        self.assertTrue(len(rows) >= 1)
        self.assertTrue(all("available_n" in row and "missing_n" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
