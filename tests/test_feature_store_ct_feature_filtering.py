from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agents.candidate_proposer import build_feature_store_payload


class FeatureStoreCtFeatureFilteringTest(unittest.TestCase):
    def test_ct_removes_low_variance_and_highly_correlated_features_before_zscore(self) -> None:
        payload = build_feature_store_payload(
            [
                {
                    "case_id": "case_1",
                    "qc": "success",
                    "ct_evidence": {"features": [1.0, 2.0, 10.0, 5.0]},
                },
                {
                    "case_id": "case_2",
                    "qc": "success",
                    "ct_evidence": {"features": [2.0, 4.0, 11.0, 5.0]},
                },
                {
                    "case_id": "case_3",
                    "qc": "success",
                    "ct_evidence": {"features": [3.0, 6.0, 9.0, 5.0]},
                },
                {
                    "case_id": "case_4",
                    "qc": "success",
                    "ct_evidence": {"features": [4.0, 8.0, 12.0, 5.0]},
                },
            ]
        )

        self.assertEqual(payload["ct_feature_names"], ["ct_0000", "ct_0002"])
        self.assertEqual(payload["z_ct"].shape, (4, 2))
        np.testing.assert_allclose(
            np.mean(payload["z_ct"], axis=0), [0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(np.std(payload["z_ct"], axis=0), [1.0, 1.0])

    def test_ct_feature_names_follow_radiomics_json_keys_after_filtering(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            paths = []
            values_by_case = [
                {"shape_volume": 1.0, "shape_volume_copy": 2.0, "firstorder_mean": 10.0, "constant": 5.0},
                {"shape_volume": 2.0, "shape_volume_copy": 4.0, "firstorder_mean": 11.0, "constant": 5.0},
                {"shape_volume": 3.0, "shape_volume_copy": 6.0, "firstorder_mean": 9.0, "constant": 5.0},
                {"shape_volume": 4.0, "shape_volume_copy": 8.0, "firstorder_mean": 12.0, "constant": 5.0},
            ]
            for index, values in enumerate(values_by_case, 1):
                path = root / f"case_{index}.json"
                path.write_text(json.dumps(values), encoding="utf-8")
                paths.append(path)

            payload = build_feature_store_payload(
                [
                    {
                        "case_id": f"case_{index}",
                        "qc": "success",
                        "ct_evidence": {"feature_path": str(path)},
                    }
                    for index, path in enumerate(paths, 1)
                ]
            )

        self.assertEqual(payload["ct_feature_names"], ["shape_volume", "firstorder_mean"])


if __name__ == "__main__":
    unittest.main()
