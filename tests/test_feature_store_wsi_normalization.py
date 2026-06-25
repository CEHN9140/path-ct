from __future__ import annotations

import unittest

import numpy as np

from agents.candidate_proposer import build_feature_store_payload


class FeatureStoreWsiNormalizationTest(unittest.TestCase):
    def test_wsi_features_are_l2_normalized_in_feature_store(self) -> None:
        payload = build_feature_store_payload(
            [
                {
                    "case_id": "case_1",
                    "qc": "success",
                    "wsi_evidence": {"features": [3.0, 4.0]},
                },
                {
                    "case_id": "case_2",
                    "qc": "success",
                    "wsi_evidence": {"features": [0.0, 0.0]},
                },
            ]
        )

        np.testing.assert_allclose(payload["z_wsi"][0], [0.6, 0.8])
        np.testing.assert_allclose(payload["z_wsi"][1], [0.0, 0.0])
        self.assertNotIn("z_joint", payload)


if __name__ == "__main__":
    unittest.main()
