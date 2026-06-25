from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools.ct_qc import match_radqy_rows, prepare_ct_series, radqy_robust_z_flags, run_ct_qc_cohort
from tools.radqy.backend.radqy import top_folder_under_root


class CtQcCohortRadqyTest(unittest.TestCase):
    def test_radqy_robust_z_flags_series_with_two_bad_metrics(self) -> None:
        rows = pd.DataFrame(
            [
                {"series_id": "good_1", "CV": 10.0, "CJV": 0.10, "EFC": 0.20, "PSNR": 10.0},
                {"series_id": "good_2", "CV": 11.0, "CJV": 0.11, "EFC": 0.21, "PSNR": 9.8},
                {"series_id": "good_3", "CV": 9.0, "CJV": 0.09, "EFC": 0.19, "PSNR": 10.1},
                {"series_id": "bad", "CV": 80.0, "CJV": 1.20, "EFC": 0.22, "PSNR": 9.9},
            ]
        )
        result = radqy_robust_z_flags(
            rows,
            high_bad=["CV", "CJV", "EFC"],
            low_bad=["PSNR"],
            cutoff=3.0,
            fail_threshold=2,
        ).set_index("series_id")

        self.assertEqual(result.loc["bad", "iqm_bad_count"], 2)
        self.assertTrue(result.loc["bad", "radqy_iqm_pass"] is False)
        self.assertEqual(result.loc["bad", "iqm_bad_metrics"], ["CV", "CJV"])
        self.assertTrue(result.loc["good_1", "radqy_iqm_pass"])

    def test_run_ct_qc_cohort_removes_radqy_input_dir(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "ct_qc.yaml").write_text(
                """
dicom_prefilter: {}
dcm2niix: {}
nifti_qc: {}
totalsegmentator: {}
radqy:
  modality: CT
  sample_stride: 4
  middle_percent: 100
  segmenter: otsuhull
  save_images: false
  robust_z_cutoff: 3.0
  bad_count_fail_threshold: 2
  metrics:
    high_bad: [CV, CJV, EFC]
    low_bad: [PSNR]
""",
                encoding="utf-8",
            )
            with patch("tools.ct_qc.prepare_ct_series", return_value=([], str(output_root / "ct_qc" / "case_1" / "dicom_prefilter_report.csv"))):
                run_ct_qc_cohort([{"Case_ID": "case_1", "CT": []}], output_root=str(output_root), config_dir=str(config_dir))

            self.assertFalse((output_root / "ct_qc" / "radqy_input").exists())
            self.assertTrue((output_root / "ct_qc" / "radqy_results.tsv").exists())

    def test_run_ct_qc_cohort_uses_tmp_radqy_input_dir(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            source_dir = root / "source_dicom"
            config_dir.mkdir()
            source_dir.mkdir()
            (source_dir / "image.dcm").write_text("dicom", encoding="utf-8")
            (config_dir / "ct_qc.yaml").write_text(
                """
dicom_prefilter: {}
dcm2niix: {}
nifti_qc: {}
totalsegmentator: {}
radqy:
  modality: CT
  sample_stride: 4
  middle_percent: 100
  segmenter: otsuhull
  save_images: false
  robust_z_cutoff: 3.0
  bad_count_fail_threshold: 2
  metrics:
    high_bad: [CV, CJV, EFC]
    low_bad: [PSNR]
""",
                encoding="utf-8",
            )
            record = {
                "ct_id": "ct_1",
                "ct_path": str(root / "ct_1.nii.gz"),
                "ct_source_path": str(source_dir),
                "radqy_subject_id": "case_1__ct_1",
                "file_exists": True,
                "n_slices": 20,
                "slice_thickness_median": 2.0,
            }
            captured = {}

            def fake_radqy(input_dir, output_dir, **kwargs):
                captured["input_dir"] = Path(input_dir)
                pd.DataFrame(
                    [
                        {
                            "Participant (topfolder--subfolder--patient ID)": "case_1__ct_1",
                            "CV": 1.0,
                            "CJV": 1.0,
                            "EFC": 1.0,
                            "PSNR": 30.0,
                        }
                    ]
                ).to_csv(Path(output_dir) / kwargs["output_name"], sep="\t", index=False)

            with patch("tools.ct_qc.prepare_ct_series", return_value=([record], str(output_root / "ct_qc" / "case_1" / "dicom_prefilter_report.csv"))):
                with patch("tools.ct_qc.run_radqy_dataset", side_effect=fake_radqy):
                    run_ct_qc_cohort([{"Case_ID": "case_1", "CT": [str(source_dir)]}], output_root=str(output_root), config_dir=str(config_dir))

            self.assertTrue(str(captured["input_dir"]).startswith("/tmp/path_ct_radqy_input_"))
            self.assertNotEqual(captured["input_dir"], output_root / "ct_qc" / "radqy_input")
            self.assertFalse((output_root / "ct_qc" / "radqy_input").exists())

    def test_run_ct_qc_cohort_reuses_complete_radqy_results(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            source_dir = root / "source_dicom"
            ct_file = root / "ct_1.nii.gz"
            config_dir.mkdir()
            source_dir.mkdir()
            (source_dir / "image.dcm").write_text("dicom", encoding="utf-8")
            ct_file.write_text("nifti", encoding="utf-8")
            (config_dir / "ct_qc.yaml").write_text(
                """
dicom_prefilter: {}
dcm2niix: {}
nifti_qc: {}
totalsegmentator: {}
radqy:
  modality: CT
  sample_stride: 4
  middle_percent: 100
  segmenter: otsuhull
  save_images: false
  robust_z_cutoff: 3.0
  bad_count_fail_threshold: 2
  metrics:
    high_bad: [CV, CJV, EFC]
    low_bad: [PSNR]
""",
                encoding="utf-8",
            )
            results_tsv = output_root / "ct_qc" / "radqy_results.tsv"
            results_tsv.parent.mkdir(parents=True)
            results_tsv.write_text(
                "#settings: cached\n"
                "P#\tParticipant (topfolder--subfolder--patient ID)\tCV\tCJV\tEFC\tPSNR\n"
                "P1\tcase_1__ct_1--case_1__ct_1--case_1\t1.0\t1.0\t1.0\t30.0\n",
                encoding="utf-8",
            )
            record = {
                "case_id": "case_1",
                "ct_id": "ct_1",
                "ct_path": str(ct_file),
                "ct_source_path": str(source_dir),
                "radqy_subject_id": "case_1__ct_1",
                "file_exists": True,
                "n_slices": 20,
                "slice_thickness_median": 2.0,
            }

            with patch("tools.ct_qc.prepare_ct_series", return_value=([record], str(output_root / "ct_qc" / "case_1" / "dicom_prefilter_report.csv"))):
                with patch("tools.ct_qc.run_radqy_dataset", side_effect=AssertionError("should reuse cached RadQy results")):
                    result = run_ct_qc_cohort([{"Case_ID": "case_1", "CT": [str(source_dir)]}], output_root=str(output_root), config_dir=str(config_dir))

            self.assertTrue(results_tsv.exists())
            self.assertEqual(result["passed_case_ids"], ["case_1"])
            self.assertEqual(result["selection_summaries"]["case_1"]["selected_series"]["CV"], 1.0)

    def test_run_ct_qc_cohort_reruns_when_radqy_results_are_incomplete(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            source_dir = root / "source_dicom"
            ct_file = root / "ct_1.nii.gz"
            config_dir.mkdir()
            source_dir.mkdir()
            (source_dir / "image.dcm").write_text("dicom", encoding="utf-8")
            ct_file.write_text("nifti", encoding="utf-8")
            (config_dir / "ct_qc.yaml").write_text(
                """
dicom_prefilter: {}
dcm2niix: {}
nifti_qc: {}
totalsegmentator: {}
radqy:
  modality: CT
  sample_stride: 4
  middle_percent: 100
  segmenter: otsuhull
  save_images: false
  robust_z_cutoff: 3.0
  bad_count_fail_threshold: 2
  metrics:
    high_bad: [CV, CJV, EFC]
    low_bad: [PSNR]
""",
                encoding="utf-8",
            )
            results_tsv = output_root / "ct_qc" / "radqy_results.tsv"
            results_tsv.parent.mkdir(parents=True)
            results_tsv.write_text(
                "P#\tParticipant (topfolder--subfolder--patient ID)\tCV\tCJV\tEFC\tPSNR\n"
                "P1\tother_case__ct_1\t1.0\t1.0\t1.0\t30.0\n",
                encoding="utf-8",
            )
            record = {
                "case_id": "case_1",
                "ct_id": "ct_1",
                "ct_path": str(ct_file),
                "ct_source_path": str(source_dir),
                "radqy_subject_id": "case_1__ct_1",
                "file_exists": True,
                "n_slices": 20,
                "slice_thickness_median": 2.0,
            }

            def fake_radqy(input_dir, output_dir, **kwargs):
                pd.DataFrame(
                    [
                        {
                            "Participant (topfolder--subfolder--patient ID)": "case_1__ct_1",
                            "CV": 2.0,
                            "CJV": 1.0,
                            "EFC": 1.0,
                            "PSNR": 30.0,
                        }
                    ]
                ).to_csv(Path(output_dir) / kwargs["output_name"], sep="\t", index=False)

            with patch("tools.ct_qc.prepare_ct_series", return_value=([record], str(output_root / "ct_qc" / "case_1" / "dicom_prefilter_report.csv"))):
                with patch("tools.ct_qc.run_radqy_dataset", side_effect=fake_radqy) as radqy:
                    result = run_ct_qc_cohort([{"Case_ID": "case_1", "CT": [str(source_dir)]}], output_root=str(output_root), config_dir=str(config_dir))

            self.assertEqual(radqy.call_count, 1)
            self.assertEqual(result["selection_summaries"]["case_1"]["selected_series"]["CV"], 2.0)

    def test_match_radqy_rows_keeps_p_hash_header(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            results_tsv = root / "radqy_results.tsv"
            results_tsv.write_text(
                "#outdir: /tmp/out\n"
                "#settings: save_images=False\n"
                "\n"
                "P#\tParticipant (topfolder--subfolder--patient ID)\tCV\tCJV\tEFC\tPSNR\n"
                "P1\tcase_1__ct_1--case_1__ct_1--case_1\t1.0\t2.0\t3.0\t30.0\n",
                encoding="utf-8",
            )

            matched = match_radqy_rows(
                [{"case_id": "case_1", "radqy_subject_id": "case_1__ct_1"}],
                results_tsv,
            )

            self.assertEqual(float(matched[0]["CV"]), 1.0)
            self.assertEqual(float(matched[0]["PSNR"]), 30.0)
            self.assertEqual(matched[0]["series_id"], "case_1__ct_1")

    def test_prepare_ct_series_flushes_prefilter_report_after_each_series(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            dcm2nii_dir = root / "ct_qc" / "case_1" / "dcm2nii"
            source_1 = root / "source_1"
            source_2 = root / "source_2"
            source_1.mkdir(parents=True)
            source_2.mkdir(parents=True)

            with patch("tools.ct_qc.read_dicom_headers", side_effect=[[], RuntimeError("interrupted")]):
                with self.assertRaises(RuntimeError):
                    prepare_ct_series(
                        "case_1",
                        [{"File Path": str(source_1)}, {"File Path": str(source_2)}],
                        dcm2nii_dir,
                        {},
                        {},
                        {},
                        {},
                    )

            report_path = root / "ct_qc" / "case_1" / "dicom_prefilter_report.csv"
            self.assertTrue(report_path.exists())
            report = pd.read_csv(report_path)
            self.assertEqual(report["record_index"].tolist(), [1])
            self.assertEqual(report["source_path"].tolist(), [str(source_1)])

    def test_prepare_ct_series_reuses_existing_prefilter_report_row(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            dcm2nii_dir = root / "ct_qc" / "case_1" / "dcm2nii"
            source_1 = root / "source_1"
            source_1.mkdir(parents=True)
            report_path = root / "ct_qc" / "case_1" / "dicom_prefilter_report.csv"
            report_path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "case_id": "case_1",
                        "record_index": 1,
                        "source_path": str(source_1),
                        "prefilter_pass": False,
                        "prefilter_reasons": "cached",
                    }
                ]
            ).to_csv(report_path, index=False)

            with patch("tools.ct_qc.read_dicom_headers", side_effect=AssertionError("should reuse cached row")):
                records, path = prepare_ct_series(
                    "case_1",
                    [{"File Path": str(source_1)}],
                    dcm2nii_dir,
                    {},
                    {},
                    {},
                    {},
                )

            self.assertEqual(records, [])
            self.assertEqual(path, str(report_path))
            report = pd.read_csv(report_path)
            self.assertEqual(report["prefilter_reasons"].tolist(), ["cached"])

    def test_run_ct_qc_cohort_keeps_existing_dcm2nii_dir_for_resume(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            dcm2nii_dir = output_root / "ct_qc" / "case_1" / "dcm2nii"
            marker = dcm2nii_dir / "already_done.nii.gz"
            config_dir.mkdir()
            dcm2nii_dir.mkdir(parents=True)
            marker.write_text("existing", encoding="utf-8")
            (config_dir / "ct_qc.yaml").write_text(
                """
dicom_prefilter: {}
dcm2niix: {}
nifti_qc: {}
totalsegmentator: {}
radqy:
  modality: CT
  sample_stride: 4
  middle_percent: 100
  segmenter: otsuhull
  save_images: false
  robust_z_cutoff: 3.0
  bad_count_fail_threshold: 2
  metrics:
    high_bad: [CV, CJV, EFC]
    low_bad: [PSNR]
""",
                encoding="utf-8",
            )

            with patch("tools.ct_qc.prepare_ct_series", return_value=([], str(output_root / "ct_qc" / "case_1" / "dicom_prefilter_report.csv"))):
                run_ct_qc_cohort([{"Case_ID": "case_1", "CT": []}], output_root=str(output_root), config_dir=str(config_dir))

            self.assertTrue(marker.exists())

    def test_radqy_top_folder_uses_symlink_path_inside_input_root(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            input_root = root / "radqy_input"
            subject_dir = input_root / "case_1"
            source_dir = root / "source"
            subject_dir.mkdir(parents=True)
            source_dir.mkdir()
            source_file = source_dir / "1.dcm"
            source_file.write_bytes(b"DICM")
            link_path = subject_dir / "1.dcm"
            link_path.symlink_to(source_file)

            self.assertEqual(top_folder_under_root(input_root, link_path), "case_1")


if __name__ == "__main__":
    unittest.main()
