from __future__ import annotations

import unittest

import numpy as np

from agents.candidate_proposer import build_feature_store_payload


class FeatureStoreRnaStandardizationTest(unittest.TestCase):
    def test_rna_features_are_column_z_scored_in_feature_store(self) -> None:
        payload = build_feature_store_payload(
            [
                {
                    "case_id": "case_1",
                    "qc": "success",
                    "omics_evidence": {"rna_features": [1.0, 5.0]},
                },
                {
                    "case_id": "case_2",
                    "qc": "success",
                    "omics_evidence": {"rna_features": [3.0, 5.0]},
                },
                {
                    "case_id": "case_3",
                    "qc": "success",
                    "omics_evidence": {"rna_features": [5.0, 5.0]},
                },
            ]
        )

        expected_first_column = np.asarray([-1.224744871, 0.0, 1.224744871])
        np.testing.assert_allclose(payload["z_rna"][:, 0], expected_first_column)
        np.testing.assert_allclose(payload["z_rna"][:, 1], [0.0, 0.0, 0.0])
        self.assertNotIn("z_joint", payload)


if __name__ == "__main__":
    unittest.main()
