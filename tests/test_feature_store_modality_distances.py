from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import distance as scipy_distance

from agents.candidate_proposer import build_feature_store_payload, snf_fuse


class FeatureStoreModalityDistanceTest(unittest.TestCase):
    def test_snf_uses_modality_specific_distances(self) -> None:
        seen_metrics: list[str] = []
        real_cdist = scipy_distance.cdist

        def spy_cdist(xa, xb, metric="euclidean", *args, **kwargs):
            seen_metrics.append(str(metric))
            return real_cdist(xa, xb, metric=metric, *args, **kwargs)

        patient_states = [
            {
                "case_id": "case_1",
                "qc": "success",
                "ct_evidence": {"features": [1.0, 4.0]},
                "wsi_evidence": {"features": [3.0, 4.0]},
                "omics_evidence": {
                    "rna_features": [1.0, 2.0, 3.0],
                    "wxs_features": [1, 0, 1],
                },
            },
            {
                "case_id": "case_2",
                "qc": "success",
                "ct_evidence": {"features": [2.0, 5.0]},
                "wsi_evidence": {"features": [4.0, 3.0]},
                "omics_evidence": {
                    "rna_features": [3.0, 1.0, 2.0],
                    "wxs_features": [1, 1, 0],
                },
            },
            {
                "case_id": "case_3",
                "qc": "success",
                "ct_evidence": {"features": [3.0, 6.0]},
                "wsi_evidence": {"features": [1.0, 5.0]},
                "omics_evidence": {
                    "rna_features": [2.0, 3.0, 1.0],
                    "wxs_features": [0, 1, 1],
                },
            },
        ]

        with patch("scipy.spatial.distance.cdist", side_effect=spy_cdist):
            payload = build_feature_store_payload(patient_states)

        self.assertEqual(seen_metrics, ["euclidean", "cosine", "correlation", "jaccard"])
        self.assertEqual(payload["snf_config"]["modality_metrics"]["ct"], "euclidean")
        self.assertEqual(payload["snf_config"]["modality_metrics"]["wsi"], "cosine")
        self.assertEqual(payload["snf_config"]["modality_metrics"]["rna"], "spearman")
        self.assertEqual(payload["snf_config"]["modality_metrics"]["wxs"], "jaccard")
        self.assertEqual(payload["z_snf"].shape, (3, 3))
        self.assertTrue(np.isfinite(payload["z_snf"]).all())

    def test_wxs_clustering_features_are_kept_after_upstream_top_gene_selection(self) -> None:
        patient_states = [
            {
                "case_id": "case_1",
                "qc": "success",
                "ct_evidence": {"features": [1.0, 4.0]},
                "wsi_evidence": {"features": [3.0, 4.0]},
                "omics_evidence": {
                    "rna_features": [1.0, 2.0, 3.0],
                    "wxs_features": [1, 1, 1],
                },
            },
            {
                "case_id": "case_2",
                "qc": "success",
                "ct_evidence": {"features": [2.0, 5.0]},
                "wsi_evidence": {"features": [4.0, 3.0]},
                "omics_evidence": {
                    "rna_features": [3.0, 1.0, 2.0],
                    "wxs_features": [1, 1, 0],
                },
            },
            {
                "case_id": "case_3",
                "qc": "success",
                "ct_evidence": {"features": [3.0, 6.0]},
                "wsi_evidence": {"features": [1.0, 5.0]},
                "omics_evidence": {
                    "rna_features": [2.0, 3.0, 1.0],
                    "wxs_features": [1, 0, 0],
                },
            },
        ]

        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "snf.yaml").write_text(
                "\n".join(
                    [
                        "neighbor_count: 20",
                        "iterations: 20",
                        "mu: 0.5",
                        "alpha: 1.0",
                        "ct_low_variance_threshold: 1.0e-8",
                        "ct_high_correlation_threshold: 0.95",
                    ]
                ),
                encoding="utf-8",
            )

            payload = build_feature_store_payload(
                patient_states,
                config_dir=str(config_dir),
            )

        self.assertEqual(payload["z_wxs"].shape, (3, 3))
        np.testing.assert_array_equal(
            payload["z_wxs"],
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        )
        self.assertEqual(payload["snf_config"]["wxs_feature_mode"], "top_frequency_gene_binary")

    def test_snf_output_is_normalized_similarity_matrix(self) -> None:
        raw_network = np.array(
            [
                [2.0, 0.5, 1.5],
                [0.2, 3.0, 0.8],
                [1.0, 0.4, 4.0],
            ],
            dtype=float,
        )

        with patch("snf.compute.affinity_matrix", return_value=raw_network):
            z_snf = snf_fuse(
                [np.array([[0.0], [1.0], [2.0]], dtype=float)],
                k=20,
                iterations=20,
                mu=0.5,
                alpha=1.0,
            )

        self.assertEqual(z_snf.shape, (3, 3))
        self.assertTrue(np.isfinite(z_snf).all())
        self.assertGreaterEqual(float(z_snf.min()), 0.0)
        self.assertLessEqual(float(z_snf.max()), 1.0)
        np.testing.assert_allclose(z_snf, z_snf.T)
        np.testing.assert_allclose(np.diag(z_snf), np.ones(3))

    def test_invalid_distances_are_not_treated_as_identical_cases(self) -> None:
        captured_distances: list[np.ndarray] = []

        def capture_affinity(distance_matrix, *args, **kwargs):
            captured_distances.append(np.asarray(distance_matrix, dtype=float).copy())
            return np.eye(3, dtype=float)

        with (
            patch(
                "scipy.spatial.distance.cdist",
                return_value=np.array(
                    [
                        [0.0, np.nan, 0.4],
                        [np.nan, 0.0, np.inf],
                        [0.4, np.inf, 0.0],
                    ],
                    dtype=float,
                ),
            ),
            patch("snf.compute.affinity_matrix", side_effect=capture_affinity),
        ):
            snf_fuse(
                [np.array([[0.0], [1.0], [2.0]], dtype=float)],
                k=20,
                iterations=20,
                mu=0.5,
                alpha=1.0,
            )

        distance_matrix = captured_distances[0]
        self.assertEqual(float(distance_matrix[0, 0]), 0.0)
        self.assertGreater(float(distance_matrix[0, 1]), 0.0)
        self.assertGreater(float(distance_matrix[1, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
