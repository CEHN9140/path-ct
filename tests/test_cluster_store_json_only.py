from __future__ import annotations

import tempfile
import unittest

from utils.cluster_store import save_candidate_clusters


class ClusterStoreJsonOnlyTest(unittest.TestCase):
    def test_save_candidate_clusters_returns_json_paths_only(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            paths = save_candidate_clusters(
                temp_dir,
                [{"cluster_id": "C0001", "member_ids": ["case_1", "case_2"]}],
            )

            self.assertNotIn("visualization_path", paths)


if __name__ == "__main__":
    unittest.main()
