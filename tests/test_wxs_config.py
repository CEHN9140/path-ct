from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.wxs import build_wxs_cohort_cache, run_case_wxs_features


class WxsConfigTest(unittest.TestCase):
    def test_top_frequency_genes_are_used_for_fusion_and_all_genes_are_kept(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "wxs.yaml").write_text(
                "\n".join(
                    [
                        "nonsynonymous_variants:",
                        "  - Missense_Mutation",
                        "top_gene_count: 2",
                        "capture_size: 38.0",
                    ]
                ),
                encoding="utf-8",
            )
            case_a = root / "case_a.maf.gz"
            case_b = root / "case_b.maf.gz"
            case_c = root / "case_c.maf.gz"
            header = ["Hugo_Symbol", "Variant_Classification"]
            case_genes = {
                case_a: ["VHL", "PBRM1", "BAP1"],
                case_b: ["VHL", "PBRM1"],
                case_c: ["VHL"],
            }
            for path, genes in case_genes.items():
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write("\t".join(header) + "\n")
                    for gene in genes:
                        handle.write(f"{gene}\tMissense_Mutation\n")
            cohort_cases = [
                {"Case_ID": "case_a", "WXS": [{"File Path": str(case_a)}]},
                {"Case_ID": "case_b", "WXS": [{"File Path": str(case_b)}]},
                {"Case_ID": "case_c", "WXS": [{"File Path": str(case_c)}]},
            ]

            cache = build_wxs_cohort_cache(
                cohort_cases,
                output_root=str(output_root),
                config_dir=str(config_dir),
            )
            bundle = run_case_wxs_features(
                case_id="case_a",
                cohort_cases=cohort_cases,
                output_root=str(output_root),
                config_dir=str(config_dir),
            )
            case_features = pd.read_csv(cache["case_features_path"])
            all_features = pd.read_csv(cache["all_features_path"])

            self.assertEqual(cache["manifest"]["selected_gene_count"], 2)
            self.assertEqual(list(case_features.columns), ["case_id", "VHL", "PBRM1"])
            self.assertEqual(list(all_features.columns), ["case_id", "BAP1", "PBRM1", "VHL"])
            self.assertEqual(bundle["payload"]["selected_genes"], ["VHL", "PBRM1"])
            self.assertEqual(bundle["payload"]["feature_values"], [1, 1])
            self.assertEqual(
                bundle["tool_result"]["artifacts"]["all_features_path"],
                cache["all_features_path"],
            )


if __name__ == "__main__":
    unittest.main()
