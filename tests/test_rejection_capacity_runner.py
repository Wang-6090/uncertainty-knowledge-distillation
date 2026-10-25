import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_rejection_capacity_pair import default_split_path


class RejectionCapacityRunnerTest(unittest.TestCase):
    def test_default_split_path_tracks_requested_seed(self):
        self.assertEqual(
            default_split_path(2099).as_posix(),
            "splits_cifar100_semantic_hard_seed2099/"
            "cifar100_60_40_semantic_hard.json",
        )


if __name__ == "__main__":
    unittest.main()
