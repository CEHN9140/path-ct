from __future__ import annotations

import json
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

evidence_builder_module = importlib.import_module("agents.evidence_builder")


class CtTumorSegCacheTest(unittest.TestCase):
    def test_reuses_cached_mask_when_config_model_path_is_relative(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "ct_tumor_seg.yaml").write_text(
                "\n".join(
                    [
                        "backend: nnunetv2_modelfolder",
                        "model_folder: tools/kits23_trained_model",
                        "output_label: 2",
                        "device: cuda:0",
                        "mpl_config_dir: tools/nnUNet/.mplconfig",
                    ]
                ),
                encoding="utf-8",
            )

            case_id = "TCGA-B0-5709"
            mask_path = output_root / "ct_tumor_seg" / case_id / f"{case_id}_mask.nii.gz"
            mask_path.parent.mkdir(parents=True)
            mask_path.write_bytes(b"cached")
            (output_root / "ct_tumor_seg" / f"{case_id}.json").write_text(
                json.dumps(
                    {
                        "tool_result": {
                            "tool_name": "ct_tumor_seg",
                            "status": "success",
                            "metrics": {"returncode": 0},
                            "artifacts": {"segmentation_path": str(mask_path)},
                            "provenance": {
                                "backend": "nnunetv2_modelfolder",
                                "case_id": case_id,
                                "model_folder": str(Path("tools/kits23_trained_model").resolve()),
                                "output_label": 2,
                            },
                            "errors": [],
                        },
                        "payload": {},
                    }
                ),
                encoding="utf-8",
            )

            state = {
                "case_id": case_id,
                "qc": "success",
                "inventory": {"CT": [{"File Path": str(root / "ct.nii.gz")}]},
            }
            class FakeImage:
                dataobj = [1]

                class header:
                    @staticmethod
                    def get_zooms() -> tuple[float, float, float]:
                        return (1.0, 1.0, 1.0)

            with (
                patch.object(evidence_builder_module, "run_ct_tumor_seg", side_effect=AssertionError("should reuse cache")),
                patch("nibabel.load", return_value=FakeImage()),
            ):
                result = evidence_builder_module.ct_tumor_seg(state, output_root=str(output_root), config_dir=str(config_dir))

            self.assertEqual(result["qc"], "success")

    def test_reuses_cached_failure_without_running_segmentation(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "ct_tumor_seg.yaml").write_text(
                "\n".join(
                    [
                        "backend: nnunetv2_modelfolder",
                        "model_folder: tools/kits23_trained_model",
                        "output_label: 2",
                        "device: cuda:0",
                        "mpl_config_dir: tools/nnUNet/.mplconfig",
                    ]
                ),
                encoding="utf-8",
            )

            case_id = "TCGA-FAILED"
            snapshot_dir = output_root / "ct_tumor_seg"
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / f"{case_id}.json").write_text(
                json.dumps(
                    {
                        "tool_result": {
                            "tool_name": "ct_tumor_seg",
                            "status": "failure",
                            "metrics": {"returncode": 1},
                            "artifacts": {"saved_output_path": str(snapshot_dir / f"{case_id}.json")},
                            "provenance": {
                                "backend": "nnunetv2_modelfolder",
                                "case_id": case_id,
                                "model_folder": str(Path("tools/kits23_trained_model").resolve()),
                                "output_label": 2,
                            },
                            "errors": ["nnUNet prediction failed."],
                        },
                        "payload": {"case_id": case_id, "returncode": 1},
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "case_id": case_id,
                "qc": "success",
                "inventory": {"CT": [{"File Path": str(root / "ct.nii.gz")}]},
            }

            with patch.object(evidence_builder_module, "run_ct_tumor_seg", side_effect=AssertionError("should reuse cached failure")):
                result = evidence_builder_module.ct_tumor_seg(state, output_root=str(output_root), config_dir=str(config_dir))

            self.assertEqual(result["qc"], "fail")


if __name__ == "__main__":
    unittest.main()
