from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_subtype_review_from_candidate as review_runner


class RunSubtypeReviewFromCandidateTest(unittest.TestCase):
    def test_loads_existing_checkpoint_and_candidate_clusters(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            source_root = root / "output_kirc_raw"
            review_root = root / "review_output"
            config_dir = root / "configs"
            config_dir.mkdir()
            checkpoint_dir = source_root / "storage" / "pipeline_checkpoints"
            candidate_dir = source_root / "candidate_subtype"
            checkpoint_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            (checkpoint_dir / "evidence_ready.json").write_text(
                json.dumps(
                    {
                        "stage": "evidence_ready",
                        "patient_states": [{"case_id": "TCGA-1", "qc": "success"}],
                    }
                ),
                encoding="utf-8",
            )
            (candidate_dir / "candidate_clusters.json").write_text(
                json.dumps(
                    [
                        {
                            "cluster_id": "C0001",
                            "member_ids": ["TCGA-1"],
                            "consensus": {
                                "matrix_path": "/data/qijun/path-ct/output_kirc/candidate_subtype/consensus_cluster/best_consensus_matrix.npy",
                                "metadata_path": "/data/qijun/path-ct/output_kirc/candidate_subtype/consensus_cluster/consensus_metadata.json",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            captured = {}

            def fake_subtype_review_agent(state, graph, output_root, config_path):
                inventory = dict(state["inventory"])
                captured["graph"] = graph
                captured["output_root"] = output_root
                captured["config_dir"] = config_path
                captured["cluster"] = dict(inventory["candidate_clusters"][0])
                return {
                    "inventory": {
                        **inventory,
                        "cluster_states": [],
                        "patient_store_paths": {},
                        "cluster_store_paths": {},
                    }
                }

            with (
                patch.object(review_runner, "build_subtype_review_graph", return_value="fake_graph"),
                patch.object(review_runner, "subtype_review_agent", side_effect=fake_subtype_review_agent),
            ):
                final_output = review_runner.run_subtype_review_from_candidate(
                    str(source_root),
                    str(review_root),
                    str(config_dir),
                )

            self.assertEqual(captured["graph"], "fake_graph")
            self.assertEqual(captured["output_root"], str(review_root.resolve()))
            self.assertEqual(captured["config_dir"], str(config_dir.resolve()))
            self.assertEqual(captured["cluster"]["member_ids"], ["TCGA-1"])
            self.assertEqual(
                captured["cluster"]["consensus"]["matrix_path"],
                str(source_root.resolve() / "candidate_subtype" / "consensus_cluster" / "best_consensus_matrix.npy"),
            )
            self.assertEqual(final_output["stage"], "logic_v_sub_v2_coordinator")
            self.assertTrue((review_root / "storage" / "reports" / "final_output.json").exists())

    def test_falls_back_to_per_cluster_json_files(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            source_root = Path(temp_dir) / "output_kirc_raw"
            candidate_dir = source_root / "candidate_subtype"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "C0002.json").write_text(
                json.dumps({"cluster_id": "C0002", "member_ids": ["B"]}),
                encoding="utf-8",
            )
            (candidate_dir / "C0001.json").write_text(
                json.dumps({"cluster_id": "C0001", "member_ids": ["A"]}),
                encoding="utf-8",
            )

            clusters, path = review_runner.load_candidate_clusters(source_root)

            self.assertEqual([cluster["cluster_id"] for cluster in clusters], ["C0001", "C0002"])
            self.assertEqual(path, str(candidate_dir))


if __name__ == "__main__":
    unittest.main()
