from __future__ import annotations

import unittest
from pathlib import Path


class CtQcCohortQcTest(unittest.TestCase):
    def test_ct_qc_uses_internal_radqy_configuration(self) -> None:
        checked_paths = [
            Path("configs/ct_qc.yaml"),
            Path("tools/ct_qc.py"),
        ]
        forbidden_tokens = [
            "radqy" + "_bin",
            "/" + "data/qijun/RadQy",
            "module: " + "tools.radqy.backend.radqy",
            "function: " + "run_radqy_dataset",
        ]
        text_by_path = {
            path: path.read_text(encoding="utf-8")
            for path in checked_paths
        }
        for path, text in text_by_path.items():
            with self.subTest(path=str(path)):
                for token in forbidden_tokens:
                    self.assertNotIn(token, text)

    def test_totalsegmentator_matplotlib_cache_is_not_written_to_case_output(self) -> None:
        text = Path("tools/ct_qc.py").read_text(encoding="utf-8")
        self.assertIn('env["MPLCONFIGDIR"] = "/tmp/path_ct_matplotlib"', text)
        self.assertNotIn('env["MPLCONFIGDIR"] = str(output_dir.parent / "matplotlib")', text)

    def test_totalsegmentator_home_is_not_written_to_case_output(self) -> None:
        text = Path("tools/ct_qc.py").read_text(encoding="utf-8")
        self.assertIn('env["TOTALSEG_HOME_DIR"] = "/tmp/path_ct_totalseg_home"', text)
        self.assertNotIn('env["TOTALSEG_HOME_DIR"] = str(output_dir.parent / "totalseg_home")', text)


if __name__ == "__main__":
    unittest.main()
