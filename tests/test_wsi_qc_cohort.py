from __future__ import annotations

import tempfile
import unittest
import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from agents.inventory import inventory_case
from tools.wsi_qc import build_wsi_case_summary

quality_control_module = importlib.import_module("agents.quality_control")


class WsiQcCohortTest(unittest.TestCase):
    def test_selects_largest_usable_tissue_slide_before_artifact_filter(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            summary = build_wsi_case_summary(
                "case_1",
                [
                    {
                        "case_id": "case_1",
                        "slide_name": "low_artifact",
                        "slide_path": "/slides/low.svs",
                        "output_dir": "/out/low",
                        "tissue_pixels": 100,
                        "usable_tissue_pixels": 100,
                        "artifact_pixels": 1,
                        "artifact_rate": 0.01,
                        "tissue_rate": 0.99,
                    },
                    {
                        "case_id": "case_1",
                        "slide_name": "largest_tissue",
                        "slide_path": "/slides/large.svs",
                        "output_dir": "/out/large",
                        "tissue_pixels": 200,
                        "usable_tissue_pixels": 200,
                        "artifact_pixels": 50,
                        "artifact_rate": 0.20,
                        "tissue_rate": 0.80,
                    },
                ],
                output_root=temp_dir,
                max_artifact_rate=0.10,
                tissue_class_id=1,
                artifact_class_ids=[2],
                background_class_ids=[0],
            )

        selected = summary["selected_slide"]
        self.assertEqual(selected["selected_slide_name"], "largest_tissue")
        self.assertFalse(selected["passes_threshold"])
        self.assertFalse(summary["case_qc_passes_threshold"])

    def test_zero_usable_tissue_selected_slide_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            summary = build_wsi_case_summary(
                "case_1",
                [
                    {
                        "case_id": "case_1",
                        "slide_name": "blank",
                        "slide_path": "/slides/blank.svs",
                        "output_dir": "/out/blank",
                        "tissue_pixels": 0,
                        "usable_tissue_pixels": 0,
                        "artifact_pixels": 0,
                        "artifact_rate": 0.0,
                        "tissue_rate": 0.0,
                    }
                ],
                output_root=temp_dir,
                max_artifact_rate=0.10,
                tissue_class_id=1,
                artifact_class_ids=[2],
                background_class_ids=[0],
            )

        self.assertEqual(summary["selected_slide"]["selected_slide_name"], "blank")
        self.assertFalse(summary["selected_slide"]["passes_threshold"])
        self.assertFalse(summary["case_qc_passes_threshold"])

    def test_quality_control_cohort_updates_patient_states_from_worker_summaries(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "wsi_qc.yaml").write_text(
                """
python_executable: /grandqc/python
device: cpu
max_artifact_rate: 0.1
tissue_class_id: 1
artifact_class_ids: [2]
background_class_ids: [0]
tissue_detect: {}
artifact_seg: {}
""",
                encoding="utf-8",
            )
            states = [
                inventory_case(
                    {
                        "Case_ID": "pass_case",
                        "WSI": [{"File Path": "/slides/pass.svs"}],
                    }
                ),
                inventory_case(
                    {
                        "Case_ID": "fail_case",
                        "WSI": [{"File Path": "/slides/fail.svs"}],
                    }
                ),
                {
                    **inventory_case(
                        {
                            "Case_ID": "ct_fail_case",
                            "WSI": [{"File Path": "/slides/skip.svs"}],
                        }
                    ),
                    "qc": "fail",
                },
            ]

            worker_result = {
                "selection_summaries": {
                    "pass_case": {
                        "case_qc_passes_threshold": True,
                        "selected_slide": {"passes_threshold": True},
                    },
                    "fail_case": {
                        "case_qc_passes_threshold": False,
                        "selected_slide": {"passes_threshold": False},
                    },
                }
            }
            with patch.object(
                quality_control_module,
                "run_wsi_qc_cohort_tool",
                return_value=worker_result,
            ) as cohort_tool:
                updated = quality_control_module.run_wsi_qc_cohort(
                    states,
                    output_root=str(root / "output"),
                    config_dir=str(config_dir),
                )

            sent_cases = cohort_tool.call_args.args[0]
            self.assertEqual(
                [case["Case_ID"] for case in sent_cases],
                ["pass_case", "fail_case"],
            )
            self.assertEqual(
                [state["qc"] for state in updated],
                ["success", "fail", "fail"],
            )

    def test_quality_control_cohort_skips_tool_when_no_cases_pass_ct_qc(self) -> None:
        states = [
            {
                **inventory_case(
                    {
                        "Case_ID": "ct_fail_case",
                        "WSI": [{"File Path": "/slides/skip.svs"}],
                    }
                ),
                "qc": "fail",
            }
        ]
        with patch.object(
            quality_control_module,
            "run_wsi_qc_cohort_tool",
            side_effect=AssertionError("should not call WSI QC"),
        ):
            updated = quality_control_module.run_wsi_qc_cohort(
                states,
                output_root="/tmp",
                config_dir="/tmp",
            )

        self.assertEqual(updated, states)

    def test_worker_stderr_is_visible_in_interactive_terminal(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "wsi_qc.yaml").write_text(
                f"""
python_executable: {sys.executable}
device: cpu
max_artifact_rate: 0.1
tissue_class_id: 1
artifact_class_ids: [2]
background_class_ids: [0]
tissue_detect: {{}}
artifact_seg: {{}}
""",
                encoding="utf-8",
            )

            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"selection_summaries": {"case_1": {"case_qc_passes_threshold": false}}}',
                stderr="",
            )
            with patch("tools.wsi_qc.sys.stderr.isatty", return_value=True):
                with patch("tools.wsi_qc.subprocess.run", return_value=completed) as run:
                    import tools.wsi_qc as wsi_qc

                    wsi_qc.run_wsi_qc_cohort(
                        [{"Case_ID": "case_1", "WSI": [{"File Path": "/slides/a.svs"}]}],
                        output_root=str(root / "output"),
                        config_dir=str(config_dir),
                    )

            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertIsNone(kwargs["stderr"])
            self.assertNotIn("capture_output", kwargs)

    def test_run_wsi_qc_cohort_reuses_existing_selection_summary(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "wsi_qc.yaml").write_text(
                """
python_executable: /grandqc/python
device: cpu
max_artifact_rate: 0.1
tissue_class_id: 1
artifact_class_ids: [2]
background_class_ids: [0]
tissue_detect: {}
artifact_seg: {}
""",
                encoding="utf-8",
            )
            summary_dir = root / "output" / "wsi_qc" / "case_1"
            summary_dir.mkdir(parents=True)
            (summary_dir / "selection_summary.json").write_text(
                '{"case_id": "case_1", "case_qc_passes_threshold": true, "selected_slide": {"passes_threshold": true}}',
                encoding="utf-8",
            )

            import tools.wsi_qc as wsi_qc

            with patch("tools.wsi_qc.subprocess.run", side_effect=AssertionError("should reuse cached WSI QC")):
                result = wsi_qc.run_wsi_qc_cohort(
                    [{"Case_ID": "case_1", "WSI": [{"File Path": "/slides/a.svs"}]}],
                    output_root=str(root / "output"),
                    config_dir=str(config_dir),
                )

            self.assertTrue(
                result["selection_summaries"]["case_1"]["selected_slide"]["passes_threshold"]
            )

    def test_worker_reuses_existing_slide_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            slide_path = root / "slide.svs"
            slide_path.write_text("slide", encoding="utf-8")
            slide_dir = root / "output" / "wsi_qc" / "case_1" / "grandqc" / "slide"
            tissue_dir = slide_dir / "tis_det_mask"
            mask_dir = slide_dir / "mask_qc"
            tissue_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            from PIL import Image

            Image.new("L", (2, 2), color=1).save(tissue_dir / "slide_MASK.png")
            Image.new("L", (2, 2), color=1).save(mask_dir / "slide_mask.png")

            import tools.wsi_qc as wsi_qc

            config = {
                "device": "cpu",
                "max_artifact_rate": 0.1,
                "tissue_class_id": 1,
                "artifact_class_ids": [2],
                "background_class_ids": [0],
                "tissue_detect": {},
                "artifact_seg": {},
            }
            with patch(
                "tools.pathology_qc.wsi_tis_detect.wsi_tis_detect",
                side_effect=AssertionError("should reuse tissue detection output"),
            ):
                with patch(
                    "tools.pathology_qc.artifacts_seg.artifacts_seg",
                    side_effect=AssertionError("should reuse artifact output"),
                ):
                    result = wsi_qc.run_wsi_qc_cohort_worker(
                        [{"Case_ID": "case_1", "WSI": [{"File Path": str(slide_path)}]}],
                        output_root=str(root / "output"),
                        config_dir=str(root / "configs"),
                        tool_config=config,
                    )

            summary = result["selection_summaries"]["case_1"]
            self.assertTrue(summary["selected_slide"]["passes_threshold"])


if __name__ == "__main__":
    unittest.main()
