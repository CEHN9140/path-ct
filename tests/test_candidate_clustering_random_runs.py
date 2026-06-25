from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeKMedoids:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.calls.append(self.kwargs)

    def fit_predict(self, values):
        matrix = np.asarray(values, dtype=float)
        medoid_count = int(self.kwargs["n_clusters"])
        medoids = np.arange(medoid_count)
        return np.argmin(matrix[:, medoids], axis=1)


class FakeAgglomerativeClustering:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.calls.append(self.kwargs)

    def fit_predict(self, values):
        matrix = np.asarray(values, dtype=float)
        n_clusters = int(self.kwargs["n_clusters"])
        return np.arange(matrix.shape[0], dtype=int) % n_clusters


class FakeSpectralClustering(FakeAgglomerativeClustering):
    pass


class FakeMDS:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.calls.append({"kwargs": self.kwargs})

    def fit_transform(self, values):
        matrix = np.asarray(values, dtype=float)
        self.calls[-1]["matrix"] = matrix
        return np.column_stack(
            [
                np.arange(matrix.shape[0], dtype=float),
                matrix.sum(axis=1),
            ]
        )


class CandidateClusteringRandomRunsTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeKMedoids.calls = []
        FakeAgglomerativeClustering.calls = []
        FakeSpectralClustering.calls = []
        FakeMDS.calls = []
        sklearn_extra = types.ModuleType("sklearn_extra")
        sklearn_extra_cluster = types.ModuleType("sklearn_extra.cluster")
        sklearn_extra_cluster.KMedoids = FakeKMedoids
        sklearn = types.ModuleType("sklearn")
        sklearn_cluster = types.ModuleType("sklearn.cluster")
        sklearn_cluster.AgglomerativeClustering = FakeAgglomerativeClustering
        sklearn_cluster.SpectralClustering = FakeSpectralClustering
        sklearn_manifold = types.ModuleType("sklearn.manifold")
        sklearn_manifold.MDS = FakeMDS
        sklearn.cluster = sklearn_cluster
        sklearn.manifold = sklearn_manifold
        self.cluster_module_patch = patch.dict(
            sys.modules,
            {
                "sklearn": sklearn,
                "sklearn.cluster": sklearn_cluster,
                "sklearn.manifold": sklearn_manifold,
                "sklearn_extra": sklearn_extra,
                "sklearn_extra.cluster": sklearn_extra_cluster,
            },
        )
        self.cluster_module_patch.start()

    def tearDown(self) -> None:
        self.cluster_module_patch.stop()

    def test_scatter_coordinates_use_snf_similarity_with_sklearn_mds(self) -> None:
        from utils.cluster_flow import consensus_scatter_coordinates

        snf_matrix = np.asarray(
            [
                [1.0, 0.8, 0.1],
                [0.8, 1.0, 0.3],
                [0.1, 0.3, 1.0],
            ],
            dtype=float,
        )
        coordinates = consensus_scatter_coordinates(snf_matrix)

        self.assertEqual(coordinates.shape, (3, 2))
        self.assertEqual(FakeMDS.calls[-1]["kwargs"]["dissimilarity"], "precomputed")
        np.testing.assert_allclose(
            FakeMDS.calls[-1]["matrix"],
            np.asarray(
                [
                    [0.0, 0.2, 0.9],
                    [0.2, 0.0, 0.7],
                    [0.9, 0.7, 0.0],
                ],
                dtype=float,
            ),
        )

    def test_consensus_metric_helpers_calculate_cdf_delta_and_pac(self) -> None:
        candidate_proposer_module = importlib.import_module("agents.candidate_proposer")
        consensus_values = np.asarray([0.0, 0.2, 1.0], dtype=float)

        thresholds, cdf_values, cdf_area = (
            candidate_proposer_module.calculate_consensus_cdf(
                consensus_values,
                thresholds=np.asarray([0.0, 0.5, 1.0], dtype=float),
            )
        )
        delta_area, relative_delta_area = candidate_proposer_module.calculate_delta_area(
            cdf_area,
            previous_cdf_area=0.5,
        )
        pac = candidate_proposer_module.calculate_pac(
            consensus_values,
            lower=0.1,
            upper=0.9,
        )

        np.testing.assert_allclose(thresholds, np.asarray([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(cdf_values, np.asarray([1 / 3, 2 / 3, 1.0]))
        self.assertAlmostEqual(cdf_area, 2 / 3)
        self.assertAlmostEqual(delta_area, 1 / 6)
        self.assertAlmostEqual(relative_delta_area, 1 / 3)
        self.assertAlmostEqual(pac, 1 / 3)

    def test_consensus_metric_helpers_do_not_clip_inputs(self) -> None:
        candidate_proposer_module = importlib.import_module("agents.candidate_proposer")
        consensus_values = np.asarray([-0.2, 0.5, 1.2], dtype=float)

        _, cdf_values, cdf_area = candidate_proposer_module.calculate_consensus_cdf(
            consensus_values,
            thresholds=np.asarray([0.0, 1.0], dtype=float),
        )
        pac = candidate_proposer_module.calculate_pac(
            consensus_values,
            lower=-0.1,
            upper=1.1,
        )

        np.testing.assert_allclose(cdf_values, np.asarray([1 / 3, 2 / 3]))
        self.assertAlmostEqual(cdf_area, 0.5)
        self.assertAlmostEqual(pac, 1 / 3)

    def test_runs_snf_algorithms_from_config_with_random_k_and_seed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "repeat_count: 2",
                        "max_clusters: 4",
                        "consensus_linkage: average",
                        "pac_lower: 0.1",
                        "pac_upper: 0.9",
                        "selection_method: pac_elbow",
                        "pac_gain_threshold: 0.05",
                        "algorithms:",
                        "  hierarchical:",
                        "    linkage_options: [average, complete]",
                        "  spectral:",
                        "    assign_labels_options: [kmeans, discretize, cluster_qr]",
                        "  pam:",
                        "    method: pam",
                        "    init_options: [k-medoids++, random, heuristic]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(6)
            ]
            z_snf = np.full((6, 6), 0.2, dtype=float)
            np.fill_diagonal(z_snf, 1.0)
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": z_snf},
            ):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

            records_path = output_root / "candidate_subtype" / "partition_records.json"
            records = json.loads(records_path.read_text(encoding="utf-8"))
            consensus_dir = output_root / "candidate_subtype" / "consensus_cluster"
            self.assertEqual(len(records), 18)
            counts = {algorithm: 0 for algorithm in ["hierarchical", "spectral", "pam"]}
            counts_by_k = {2: 0, 3: 0, 4: 0}
            for record in records:
                counts[record["algorithm"]] += 1
                counts_by_k[record["n_clusters"]] += 1
                self.assertIn(record["n_clusters"], {2, 3, 4})
                self.assertIsInstance(record["seed"], int)
                self.assertIn("run_index", record)
                self.assertIsInstance(record.get("algorithm_config"), dict)
                if record["algorithm"] == "hierarchical":
                    self.assertIn(record["algorithm_config"]["linkage"], {"average", "complete"})
                    self.assertNotIn("linkage_options", record["algorithm_config"])
                if record["algorithm"] == "spectral":
                    self.assertIn(
                        record["algorithm_config"]["assign_labels"],
                        {"kmeans", "discretize", "cluster_qr"},
                    )
                    self.assertNotIn("assign_labels_options", record["algorithm_config"])
                if record["algorithm"] == "pam":
                    self.assertEqual(record["algorithm_config"]["method"], "pam")
                    self.assertIn(
                        record["algorithm_config"]["init"],
                        {"k-medoids++", "random", "heuristic"},
                    )
                    self.assertNotIn("init_options", record["algorithm_config"])
                self.assertEqual(len(record.get("labels", {})), 6)
            self.assertEqual(counts, {"hierarchical": 6, "spectral": 6, "pam": 6})
            self.assertEqual(counts_by_k, {2: 6, 3: 6, 4: 6})
            self.assertEqual(len(FakeKMedoids.calls), 6)
            self.assertTrue(
                all(call["metric"] == "precomputed" for call in FakeKMedoids.calls)
            )
            self.assertTrue(all("max_iter" not in call for call in FakeKMedoids.calls))
            self.assertTrue(
                all("n_init" not in call for call in FakeSpectralClustering.calls)
            )
            self.assertTrue((consensus_dir / "consensus_scatter.png").exists())
            self.assertEqual(
                (consensus_dir / "consensus_scatter.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            self.assertFalse((output_root / "storage" / "feature_store").exists())
            self.assertFalse((output_root / "snf").exists())
            scatter_points = json.loads((consensus_dir / "consensus_scatter_coordinates.json").read_text(encoding="utf-8"))
            self.assertEqual(len(scatter_points), 6)
            self.assertTrue({"case_id", "cluster_label", "x", "y"}.issubset(scatter_points[0]))
            metadata = json.loads((consensus_dir / "consensus_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["scatter_ellipse_count"], metadata["best_n_clusters"])
            self.assertTrue(metadata["scatter_path"].endswith(".png"))
            self.assertTrue(metadata["cdf_plot_path"].endswith(".png"))
            self.assertTrue(metadata["delta_area_plot_path"].endswith(".png"))
            self.assertTrue(metadata["pac_plot_path"].endswith(".png"))
            expected_snf_distance = 1.0 - z_snf
            np.fill_diagonal(expected_snf_distance, 0.0)
            np.testing.assert_allclose(FakeMDS.calls[-1]["matrix"], expected_snf_distance)

    def test_only_runs_algorithms_present_in_config(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "repeat_count: 2",
                        "max_clusters: 4",
                        "consensus_linkage: average",
                        "pac_lower: 0.1",
                        "pac_upper: 0.9",
                        "selection_method: pac_elbow",
                        "pac_gain_threshold: 0.05",
                        "algorithms:",
                        "  pam:",
                        "    method: pam",
                        "    init_options: [k-medoids++, random, heuristic]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(6)
            ]
            z_snf = np.eye(6, dtype=float)
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": z_snf},
            ):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

            records = json.loads((output_root / "candidate_subtype" / "partition_records.json").read_text(encoding="utf-8"))
            self.assertEqual(len(records), 6)
            self.assertEqual({record["algorithm"] for record in records}, {"pam"})

    def test_candidate_clustering_requires_configured_repeat_count(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "max_clusters: 4",
                        "consensus_linkage: average",
                        "algorithms:",
                        "  pam:",
                        "    method: pam",
                        "    init_options: [k-medoids++, random, heuristic]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(6)
            ]
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": np.eye(6, dtype=float)},
            ), self.assertRaises(KeyError):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

    def test_consensus_skips_k_without_valid_partitions(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "repeat_count: 2",
                        "max_clusters: 4",
                        "consensus_linkage: average",
                        "pac_lower: 0.1",
                        "pac_upper: 0.9",
                        "selection_method: pac_elbow",
                        "pac_gain_threshold: 0.05",
                        "algorithms:",
                        "  hierarchical:",
                        "    linkage_options: [average]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(6)
            ]
            z_snf = np.eye(6, dtype=float)
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")
            original_canonical_partition = candidate_proposer_module.canonical_partition

            class FakeRng:
                def __init__(self):
                    self.k_values = [4, 2]

                def choice(self, values):
                    return list(values)[0]

                def integers(self, low, high=None):
                    return self.k_values.pop(0) if high == 5 else 123

            def fake_canonical_partition(labels):
                values = tuple(int(item) for item in np.asarray(labels, dtype=int))
                return (0,) * len(values) if len(set(values)) == 4 else original_canonical_partition(labels)

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": z_snf},
            ), patch.object(
                candidate_proposer_module.np.random,
                "default_rng",
                return_value=FakeRng(),
            ), patch.object(
                candidate_proposer_module,
                "canonical_partition",
                side_effect=fake_canonical_partition,
            ):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

            consensus_dir = output_root / "candidate_subtype" / "consensus_cluster"
            metadata = json.loads((consensus_dir / "consensus_metadata.json").read_text(encoding="utf-8"))
            result_k_values = {
                item["n_clusters"] for item in metadata["consensus_results"]
            }
            self.assertNotIn(4, result_k_values)
            best_result = next(
                item
                for item in metadata["consensus_results"]
                if item["n_clusters"] == metadata["best_n_clusters"]
            )
            clusters = json.loads(
                (output_root / "candidate_subtype" / "candidate_clusters.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(clusters)
            self.assertEqual(
                clusters[0]["consensus"]["partition_count"],
                best_result["partition_count"],
            )
            self.assertFalse((consensus_dir / "matrices" / "consensus_matrix_K4.npy").exists())
            self.assertFalse(
                (consensus_dir / "visualizations" / "consensus_heatmap_K4.png").exists()
            )

    def test_partition_is_invalid_when_actual_cluster_count_differs_from_target_k(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "repeat_count: 1",
                        "max_clusters: 4",
                        "consensus_linkage: average",
                        "pac_lower: 0.1",
                        "pac_upper: 0.9",
                        "selection_method: pac_elbow",
                        "pac_gain_threshold: 0.05",
                        "algorithms:",
                        "  hierarchical:",
                        "    linkage_options: [average]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(6)
            ]
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")
            original_canonical_partition = candidate_proposer_module.canonical_partition

            def fake_canonical_partition(labels):
                values = tuple(int(item) for item in np.asarray(labels, dtype=int))
                return original_canonical_partition(
                    np.asarray([index % 2 for index in range(len(values))], dtype=int)
                ) if len(set(values)) == 4 else original_canonical_partition(labels)

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": np.eye(6, dtype=float)},
            ), patch.object(
                candidate_proposer_module,
                "canonical_partition",
                side_effect=fake_canonical_partition,
            ):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

            records = json.loads(
                (output_root / "candidate_subtype" / "partition_records.json").read_text(
                    encoding="utf-8"
                )
            )
            k4_records = [record for record in records if record["n_clusters"] == 4]
            self.assertTrue(k4_records)
            self.assertTrue(all(not record["valid"] for record in k4_records))
            self.assertEqual(
                {record["skip_reason"] for record in k4_records},
                {"cluster_count_mismatch"},
            )

    def test_selects_best_k_by_pac_elbow(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "repeat_count: 1",
                        "max_clusters: 4",
                        "consensus_linkage: average",
                        "pac_lower: 0.1",
                        "pac_upper: 0.9",
                        "selection_method: pac_elbow",
                        "pac_gain_threshold: 0.05",
                        "algorithms:",
                        "  hierarchical:",
                        "    linkage_options: [average]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(6)
            ]
            matrices = []
            for off_diagonal_value in [0.5, 0.0, 0.3]:
                matrix = np.full((6, 6), off_diagonal_value, dtype=float)
                np.fill_diagonal(matrix, 1.0)
                matrices.append(matrix)
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": np.eye(6, dtype=float)},
            ), patch.object(
                candidate_proposer_module,
                "consensus_matrix_from_partitions",
                side_effect=matrices,
            ), patch.object(
                candidate_proposer_module,
                "calculate_pac",
                side_effect=[0.9, 0.5, 0.48],
            ):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

            consensus_dir = output_root / "candidate_subtype" / "consensus_cluster"
            metadata = json.loads(
                (consensus_dir / "consensus_metadata.json").read_text(encoding="utf-8")
            )
            clusters = json.loads(
                (output_root / "candidate_subtype" / "candidate_clusters.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["best_n_clusters"], 3)
            self.assertEqual(metadata["selection_metric"], "pac_elbow")
            self.assertEqual(metadata["best_pac"], 0.5)
            self.assertEqual(metadata["pac_gain_threshold"], 0.05)
            self.assertAlmostEqual(metadata["pac_gain"], 0.4)
            self.assertEqual(metadata["next_n_clusters"], 4)
            self.assertAlmostEqual(metadata["next_pac_gain"], 0.02)
            self.assertEqual(metadata["pac_lower"], 0.1)
            self.assertEqual(metadata["pac_upper"], 0.9)
            self.assertTrue((consensus_dir / "pac_plot.png").exists())
            self.assertTrue(clusters)
            self.assertEqual(
                clusters[0]["consensus"]["selection_metric"],
                "pac_elbow",
            )
            self.assertEqual(clusters[0]["consensus"]["best_n_clusters"], 3)

    def test_pac_elbow_uses_smallest_gain_when_threshold_is_not_crossed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "repeat_count: 1",
                        "max_clusters: 4",
                        "consensus_linkage: average",
                        "pac_lower: 0.1",
                        "pac_upper: 0.9",
                        "selection_method: pac_elbow",
                        "pac_gain_threshold: 0.1",
                        "algorithms:",
                        "  hierarchical:",
                        "    linkage_options: [average]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(6)
            ]
            matrices = []
            for off_diagonal_value in [0.5, 0.6, 0.0]:
                matrix = np.full((6, 6), off_diagonal_value, dtype=float)
                np.fill_diagonal(matrix, 1.0)
                matrices.append(matrix)
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": np.eye(6, dtype=float)},
            ), patch.object(
                candidate_proposer_module,
                "consensus_matrix_from_partitions",
                side_effect=matrices,
            ), patch.object(
                candidate_proposer_module,
                "calculate_pac",
                side_effect=[0.9, 0.5, 0.2],
            ):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

            consensus_dir = output_root / "candidate_subtype" / "consensus_cluster"
            metadata = json.loads(
                (consensus_dir / "consensus_metadata.json").read_text(encoding="utf-8")
            )
            clusters = json.loads(
                (output_root / "candidate_subtype" / "candidate_clusters.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["best_n_clusters"], 3)
            self.assertEqual(metadata["selection_metric"], "pac_elbow_smallest_gain")
            self.assertAlmostEqual(metadata["pac_gain"], 0.4)
            self.assertEqual(metadata["next_n_clusters"], 4)
            self.assertAlmostEqual(metadata["next_pac_gain"], 0.3)
            self.assertTrue(clusters)

    def test_normalizes_snf_matrix_before_clustering(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            output_root = root / "output"
            config_dir.mkdir()
            (config_dir / "candidate_clustering.yaml").write_text(
                "\n".join(
                    [
                        "repeat_count: 1",
                        "max_clusters: 2",
                        "consensus_linkage: average",
                        "pac_lower: 0.1",
                        "pac_upper: 0.9",
                        "selection_method: pac_elbow",
                        "pac_gain_threshold: 0.05",
                        "algorithms:",
                        "  hierarchical:",
                        "    linkage_options: [average, complete]",
                        "  spectral:",
                        "    assign_labels_options: [kmeans, discretize, cluster_qr]",
                    ]
                ),
                encoding="utf-8",
            )
            patient_states = [
                {"case_id": f"case_{index}", "qc": "success"}
                for index in range(4)
            ]
            z_snf = np.array(
                [
                    [1.0, 1.4, -0.2, 0.2],
                    [0.4, 1.0, 0.8, 0.3],
                    [0.6, 0.1, 1.0, 2.0],
                    [0.2, 0.3, 0.4, 1.0],
                ],
                dtype=float,
            )
            expected_w = np.asarray(z_snf, dtype=float)
            expected_w = (expected_w + expected_w.T) / 2.0
            expected_w = np.clip(expected_w, 0.0, 1.0)
            np.fill_diagonal(expected_w, 1.0)
            expected_distance = 1.0 - expected_w
            np.fill_diagonal(expected_distance, 0.0)
            captured = {}
            candidate_proposer_module = importlib.import_module("agents.candidate_proposer")

            class FakeAgglomerativeClustering:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def fit_predict(self, values):
                    if "hierarchical_distance" not in captured:
                        captured["hierarchical_distance"] = np.asarray(values, dtype=float)
                    return np.array([0, 0, 1, 1], dtype=int)

            class FakeSpectralClustering:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def fit_predict(self, values):
                    captured["spectral_affinity"] = np.asarray(values, dtype=float)
                    return np.array([0, 0, 1, 1], dtype=int)

            with patch.object(
                candidate_proposer_module,
                "build_feature_store_payload",
                return_value={"z_snf": z_snf},
            ), patch(
                "sklearn.cluster.AgglomerativeClustering",
                FakeAgglomerativeClustering,
            ), patch("sklearn.cluster.SpectralClustering", FakeSpectralClustering):
                candidate_proposer_module.candidate_proposer(
                    patient_states,
                    output_root=str(output_root),
                    config_dir=str(config_dir),
                )

            np.testing.assert_allclose(captured["hierarchical_distance"], expected_distance)
            np.testing.assert_allclose(captured["spectral_affinity"], expected_w)


if __name__ == "__main__":
    unittest.main()
