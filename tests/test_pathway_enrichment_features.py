from __future__ import annotations

import csv
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from tools.tool_pathway_enrichment import tool_pathway_enrichment


class PathwayEnrichmentFeatureTest(unittest.TestCase):
    def test_pathway_enrichment_prefers_pathway_feature_table(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            gmt_path = root / "hallmark.gmt"
            top_gene_path = root / "rna_top.csv"
            pathway_path = root / "rna_pathway.csv"
            output_root = str(root / "out")

            gmt_path.write_text(
                "HALLMARK_VHL\tna\tVHL\tPBRM1\n",
                encoding="utf-8",
            )
            (config_dir / "subtype_review.yaml").write_text(
                "artifact_policy:\n  save_figures: false\n",
                encoding="utf-8",
            )
            (config_dir / "subtype_review_tools.yaml").write_text(
                "\n".join(
                    [
                        "rna:",
                        "  max_reported_features: 5",
                        "  min_pathway_overlap: 2",
                        f"  pathway_gene_sets_path: {gmt_path}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with top_gene_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(
                    [
                        ["case_id", "G1", "G2"],
                        ["A", "5", "1"],
                        ["B", "6", "1"],
                        ["C", "1", "5"],
                        ["D", "1", "6"],
                    ]
                )
            with pathway_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(
                    [
                        ["case_id", "VHL", "PBRM1", "G1", "G2"],
                        ["A", "5", "4", "5", "1"],
                        ["B", "6", "5", "6", "1"],
                        ["C", "1", "1", "1", "5"],
                        ["D", "1", "1", "1", "6"],
                    ]
                )

            states = {
                case_id: {
                    "omics_evidence": {
                        "rna_feature_path": str(top_gene_path),
                        "rna_pathway_feature_path": str(pathway_path),
                    }
                }
                for case_id in ["A", "B", "C", "D"]
            }
            def fake_ssgsea(data, gene_sets, **kwargs):
                self.assertEqual(list(data.index), ["VHL", "PBRM1", "G1", "G2"])
                self.assertEqual(gene_sets, {"HALLMARK_VHL": ["PBRM1", "VHL"]})
                self.assertEqual(kwargs["min_size"], 2)
                return types.SimpleNamespace(
                    res2d=pd.DataFrame(
                        [
                            {"Name": "A", "Term": "HALLMARK_VHL", "NES": 2.0},
                            {"Name": "B", "Term": "HALLMARK_VHL", "NES": 2.2},
                            {"Name": "C", "Term": "HALLMARK_VHL", "NES": -1.0},
                            {"Name": "D", "Term": "HALLMARK_VHL", "NES": -1.1},
                        ]
                    )
                )

            with patch.dict(
                "sys.modules",
                {"gseapy": types.SimpleNamespace(ssgsea=fake_ssgsea)},
            ), patch(
                "tools.tool_pathway_enrichment.bh_fdr",
                side_effect=lambda values: values,
            ):
                result = tool_pathway_enrichment(
                    {"cluster_id": "C1", "member_ids": ["A", "B"]},
                    states,
                    output_root,
                    str(config_dir),
                    [
                        {"cluster_id": "C1", "member_ids": ["A", "B"]},
                        {"cluster_id": "C2", "member_ids": ["C", "D"]},
                    ],
                )

            metrics = result["results"]["metrics"]
            self.assertEqual(result["status"], "success")
            self.assertEqual(set(metrics), {"rna_pathway_enrichment"})
            rows = metrics["rna_pathway_enrichment"]
            self.assertEqual(len(rows), 2)
            c1_row = next(row for row in rows if row["candidate_set_id"] == "C1")
            self.assertEqual(c1_row["pathway"], "HALLMARK_VHL")
            self.assertEqual(c1_row["pathway_gene_count"], 2)
            self.assertGreater(c1_row["delta_mean_score"], 0)
            self.assertEqual(result["artifacts"]["figures"], {})


if __name__ == "__main__":
    unittest.main()
