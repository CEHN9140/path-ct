from __future__ import annotations

import unittest

import numpy as np

from agents.candidate_proposer import build_feature_store_payload


class FeatureStoreCtStandardizationTest(unittest.TestCase):
    def test_ct_features_are_column_z_scored_in_feature_store(self) -> None:
        payload = build_feature_store_payload(
            [
                {
                    "case_id": "case_1",
                    "qc": "success",
                    "ct_evidence": {"features": [1.0, 5.0]},
                },
                {
                    "case_id": "case_2",
                    "qc": "success",
                    "ct_evidence": {"features": [3.0, 5.0]},
                },
                {
                    "case_id": "case_3",
                    "qc": "success",
                    "ct_evidence": {"features": [5.0, 5.0]},
                },
            ]
        )

        expected_first_column = np.asarray([-1.224744871, 0.0, 1.224744871])
        np.testing.assert_allclose(payload["z_ct"][:, 0], expected_first_column)
        self.assertEqual(payload["z_ct"].shape, (3, 1))
        self.assertEqual(payload["ct_feature_names"], ["ct_0000"])
        self.assertNotIn("z_joint", payload)


if __name__ == "__main__":
    unittest.main()
