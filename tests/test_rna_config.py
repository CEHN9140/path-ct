from __future__ import annotations

import tempfile
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools.rna import build_rna_cohort_cache, run_case_rna_features
from utils.omics_utils import build_cohort_signature, collect_case_file_paths

evidence_builder_module = importlib.import_module("agents.evidence_builder")


def write_rna_file(path: Path, values: dict[str, float]) -> None:
    rows = [
        ["meta"] * 9,
        ["gene_id", "gene_name", "gene_type", "unstranded", "stranded_first", "stranded_second", "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"],
        ["meta"] * 9,
        ["meta"] * 9,
        ["meta"] * 9,
        ["meta"] * 9,
    ]
    for gene_name, value in values.items():
        rows.append(["", gene_name, "", "", "", "", str(value), "", ""])
    path.write_text("\n".join("\t".join(row) for row in rows), encoding="utf-8")


class RnaConfigTest(unittest.TestCase):
    def test_top_gene_count_comes_from_config(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "rna.yaml").write_text("top_gene_count: 1\n", encoding="utf-8")
            case_a = root / "case_a.tsv"
            case_b = root / "case_b.tsv"
            write_rna_file(case_a, {"A": 1, "B": 10})
            write_rna_file(case_b, {"A": 9, "B": 11})
            cohort_cases = [
                {"Case_ID": "case_a", "RNA_Seq": [{"File Path": str(case_a)}]},
                {"Case_ID": "case_b", "RNA_Seq": [{"File Path": str(case_b)}]},
            ]

            cache = build_rna_cohort_cache(
                cohort_cases,
                output_root=str(output_root),
                config_dir=str(config_dir),
            )
            bundle = run_case_rna_features(
                case_id="case_a",
                cohort_cases=cohort_cases,
                output_root=str(output_root),
                config_dir=str(config_dir),
            )

            self.assertEqual(cache["manifest"]["top_gene_count"], 1)
            self.assertEqual(cache["manifest"]["selected_gene_count"], 1)
            self.assertEqual(bundle["payload"]["selected_gene_count"], 1)
            self.assertEqual(len(bundle["payload"]["feature_values"]), 1)
            pathway_features = pd.read_csv(cache["pathway_features_path"])
            self.assertEqual(
                pathway_features.columns.tolist(),
                ["case_id", "A", "B"],
            )
            self.assertEqual(
                bundle["tool_result"]["artifacts"]["pathway_features_path"],
                cache["pathway_features_path"],
            )

    def test_old_rna_snapshot_without_pathway_features_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "rna.yaml").write_text("top_gene_count: 1\n", encoding="utf-8")
            (config_dir / "wxs.yaml").write_text(
                "nonsynonymous_variants: [Missense_Mutation]\ncapture_size: 38.0\n",
                encoding="utf-8",
            )
            rna_file = root / "case_a.tsv"
            wxs_file = root / "case_a.maf.gz"
            write_rna_file(rna_file, {"A": 1, "B": 10})
            import gzip

            with gzip.open(wxs_file, "wt", encoding="utf-8") as handle:
                handle.write("Hugo_Symbol\tVariant_Classification\n")
            cohort_cases = [
                {"Case_ID": "case_a", "RNA_Seq": [{"File Path": str(rna_file)}], "WXS": [{"File Path": str(wxs_file)}]},
            ]
            signature = build_cohort_signature(
                collect_case_file_paths(cohort_cases, "RNA_Seq"),
                extra={
                    "top_gene_count": 1,
                    "modality": "RNA_Seq",
                    "feature_mode": "top_genes_only",
                },
            )
            old_feature_path = output_root / "rna" / "case_features.csv"
            old_top_path = output_root / "rna" / "top_genes.csv"
            old_manifest_path = output_root / "rna" / "manifest.json"
            old_feature_path.parent.mkdir(parents=True)
            old_feature_path.write_text("case_id,A\ncase_a,1\n", encoding="utf-8")
            old_top_path.write_text("gene_name,mad\nA,1\n", encoding="utf-8")
            old_manifest_path.write_text("{}", encoding="utf-8")
            (output_root / "rna" / "case_a.json").write_text(
                """{
  "tool_result": {
    "tool_name": "rna",
    "status": "success",
    "metrics": {},
    "artifacts": {
      "case_features_path": "%s",
      "top_genes_path": "%s",
      "manifest_path": "%s"
    },
    "provenance": {"signature": "%s"},
    "errors": []
  },
  "payload": {"feature_values": [1.0]}
}
"""
                % (old_feature_path, old_top_path, old_manifest_path, signature),
                encoding="utf-8",
            )
            state = {
                "case_id": "case_a",
                "qc": "success",
                "inventory": cohort_cases[0],
            }

            with patch.object(evidence_builder_module, "announce_tool_action") as announce:
                result = evidence_builder_module.build_evidence_states(
                    [state], output_root=str(output_root), config_dir=str(config_dir)
                )[0]

            messages = [call.args[2] for call in announce.call_args_list]
            self.assertIn("Call tool `rna`.", messages)
            self.assertIn(
                "rna_pathway_feature_path",
                result["omics_evidence"],
            )


if __name__ == "__main__":
    unittest.main()
