from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main as pipeline_main


class PipelineEvidenceCheckpointTest(unittest.TestCase):
    def test_run_pipeline_resumes_from_evidence_ready_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "snf.yaml").write_text("neighbor_count: 20\n", encoding="utf-8")
            cases = [{"Case_ID": "case_1"}]
            patient_states = [
                {
                    "case_id": "case_1",
                    "qc": "success",
                    "ct_evidence": {"features": [1.0]},
                    "wsi_evidence": {"features": [1.0]},
                    "omics_evidence": {},
                }
            ]
            pipeline_main.save_evidence_checkpoint(
                str(output_root),
                cases,
                str(config_dir),
                patient_states,
            )

            args = SimpleNamespace(output_root=str(output_root), config_dir=str(config_dir))
            candidate_result = {
                "patient_states": patient_states,
                "patient_states_by_id": {"case_1": patient_states[0]},
                "candidate_clusters": [],
                "patient_store_paths": {},
                "candidate_clusters_path": {},
            }
            final_output = {"status": "ok"}

            with (
                patch.object(pipeline_main, "inventory_case", side_effect=AssertionError("should skip inventory/evidence")),
                patch.object(pipeline_main, "run_ct_qc_cohort", side_effect=AssertionError("should skip cohort CT QC")),
                patch.object(pipeline_main, "run_wsi_qc_cohort", side_effect=AssertionError("should skip WSI QC")),
                patch.object(pipeline_main, "evidence_builder", side_effect=AssertionError("should skip evidence builder")),
                patch.object(pipeline_main, "build_evidence_states", side_effect=AssertionError("should skip omics evidence builder")),
                patch.object(pipeline_main, "build_subtype_review_graph", return_value=object()),
                patch.object(pipeline_main, "save_graph_pngs", return_value={}),
                patch.object(pipeline_main, "candidate_proposer", return_value=candidate_result),
                patch.object(pipeline_main, "subtype_review_agent", return_value={"inventory": {**candidate_result, "cluster_states": [], "cluster_store_paths": {}}}),
                patch.object(pipeline_main, "build_pipeline_output", return_value=final_output),
                patch.object(pipeline_main, "save_final_output", return_value=str(output_root / "final_output.json")),
            ):
                result = pipeline_main.run_pipeline(args, cases)

            self.assertEqual(result, final_output)

    def test_run_pipeline_uses_cohort_ct_qc_before_wsi_qc(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            for name in pipeline_main.EVIDENCE_CHECKPOINT_CONFIGS:
                (config_dir / name).write_text("{}\n", encoding="utf-8")
            cases = [{"Case_ID": "case_1", "CT": [{"File Path": "/ct"}], "WSI": [{"File Path": "/wsi"}]}]
            args = SimpleNamespace(output_root=str(output_root), config_dir=str(config_dir))
            call_order = []

            def cohort_qc(patient_states, *, output_root, config_dir):
                call_order.append("ct")
                return [{**dict(item), "qc": "success"} for item in patient_states]

            def wsi_qc(patient_states, *, output_root, config_dir):
                call_order.append("wsi")
                return [{**dict(item), "qc": "success"} for item in patient_states]

            candidate_result = {
                "patient_states": [{"case_id": "case_1", "qc": "success"}],
                "patient_states_by_id": {"case_1": {"case_id": "case_1", "qc": "success"}},
                "candidate_clusters": [],
                "patient_store_paths": {},
                "candidate_clusters_path": {},
            }

            with (
                patch.object(pipeline_main, "run_ct_qc_cohort", side_effect=cohort_qc),
                patch.object(pipeline_main, "run_wsi_qc_cohort", side_effect=wsi_qc),
                patch.object(pipeline_main, "evidence_builder", side_effect=lambda state, **kwargs: dict(state)),
                patch.object(pipeline_main, "build_evidence_states", side_effect=lambda states, **kwargs: list(states)),
                patch.object(pipeline_main, "build_subtype_review_graph", return_value=object()),
                patch.object(pipeline_main, "save_graph_pngs", return_value={}),
                patch.object(pipeline_main, "candidate_proposer", return_value=candidate_result),
                patch.object(pipeline_main, "subtype_review_agent", return_value={"inventory": {**candidate_result, "cluster_states": [], "cluster_store_paths": {}}}),
                patch.object(pipeline_main, "build_pipeline_output", return_value={"status": "ok"}),
                patch.object(pipeline_main, "save_final_output", return_value=str(output_root / "final_output.json")),
            ):
                pipeline_main.run_pipeline(args, cases)

            self.assertEqual(call_order[:2], ["ct", "wsi"])

    def test_evidence_checkpoint_ignores_candidate_clustering_config_changes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "snf.yaml").write_text("neighbor_count: 20\n", encoding="utf-8")
            candidate_config_path = config_dir / "candidate_clustering.yaml"
            candidate_config_path.write_text("repeat_count: 50\n", encoding="utf-8")
            cases = [{"Case_ID": "case_1"}]
            patient_states = [{"case_id": "case_1", "qc": "success"}]

            pipeline_main.save_evidence_checkpoint(
                str(output_root),
                cases,
                str(config_dir),
                patient_states,
            )
            self.assertEqual(
                pipeline_main.load_evidence_checkpoint(str(output_root), cases, str(config_dir)),
                patient_states,
            )

            candidate_config_path.write_text("repeat_count: 10\n", encoding="utf-8")
            self.assertEqual(
                pipeline_main.load_evidence_checkpoint(str(output_root), cases, str(config_dir)),
                patient_states,
            )

    def test_evidence_checkpoint_invalidates_when_evidence_config_changes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            config_path = config_dir / "ct_qc.yaml"
            config_path.write_text("enabled: true\n", encoding="utf-8")
            cases = [{"Case_ID": "case_1"}]
            patient_states = [{"case_id": "case_1", "qc": "success"}]

            pipeline_main.save_evidence_checkpoint(
                str(output_root),
                cases,
                str(config_dir),
                patient_states,
            )
            config_path.write_text("enabled: false\n", encoding="utf-8")
            self.assertIsNone(
                pipeline_main.load_evidence_checkpoint(str(output_root), cases, str(config_dir))
            )


if __name__ == "__main__":
    unittest.main()
