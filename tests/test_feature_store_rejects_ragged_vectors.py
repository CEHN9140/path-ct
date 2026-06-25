from __future__ import annotations

import unittest

from agents.candidate_proposer import build_feature_store_payload


class FeatureStoreRejectsRaggedVectorsTest(unittest.TestCase):
    def test_ragged_modality_vectors_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_feature_store_payload(
                [
                    {
                        "case_id": "case_1",
                        "qc": "success",
                        "ct_evidence": {"features": [1.0, 2.0]},
                    },
                    {
                        "case_id": "case_2",
                        "qc": "success",
                        "ct_evidence": {"features": [3.0]},
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
