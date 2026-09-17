from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.losses import supervised_contrastive_loss
from novel_discovery.pipeline import calibration_diagnostics, fit_score_normalization, run_discovery
from train import parse_args


class CommandLineTest(unittest.TestCase):
    def test_discover_arguments_stay_in_sync_with_runtime(self):
        args = parse_args(
            [
                "discover",
                "--open-val-ratio", "0.2",
                "--temperature-calibration",
                "--calibration-bins", "10",
                "--auto-calibrate-score",
            ]
        )

        self.assertEqual(args.command, "discover")
        self.assertEqual(args.open_val_ratio, 0.2)
        self.assertTrue(args.temperature_calibration)
        self.assertEqual(args.calibration_bins, 10)
        self.assertTrue(args.auto_calibrate_score)


class LossBehaviorTest(unittest.TestCase):
    def test_supervised_contrastive_loss_ignores_anchors_without_positives(self):
        features = torch.eye(4)
        labels = torch.tensor([0, 1, 2, 3])

        loss = supervised_contrastive_loss(features, labels)

        self.assertEqual(loss.item(), 0.0)

    def test_supervised_contrastive_loss_uses_positive_pairs(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])

        loss = supervised_contrastive_loss(features, labels)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)


class ScoreBehaviorTest(unittest.TestCase):
    def test_score_normalization_does_not_require_prototypes_for_entropy_stats(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 0.3]),
            "features": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        }

        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)

        self.assertIn("entropy", stats)
        self.assertNotIn("proto_dist", stats)


class CalibrationTest(unittest.TestCase):
    def test_calibration_diagnostics_reports_ece(self):
        result = calibration_diagnostics(
            np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]),
            np.array([0, 1, 1]),
            num_bins=5,
        )
        self.assertGreaterEqual(result["ece"], 0.0)
        self.assertEqual(sum(item["count"] for item in result["bins"]), 3)


class DiscoveryReportTest(unittest.TestCase):
    def test_run_discovery_reports_candidate_pool_purity(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 2.0, 2.2, 2.1, 0.15], dtype=float),
            "epistemic": np.zeros(6, dtype=float),
            "aleatoric": np.zeros(6, dtype=float),
            "expected_entropy": np.zeros(6, dtype=float),
            "head_uncertainty": np.zeros(6, dtype=float),
            "probs": np.array(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.5, 0.5],
                    [0.4, 0.6],
                    [0.45, 0.55],
                    [0.7, 0.3],
                ],
                dtype=float,
            ),
            "logits": np.array(
                [
                    [2.0, 0.0],
                    [1.5, 0.0],
                    [0.1, 0.2],
                    [0.2, 0.1],
                    [0.0, 0.3],
                    [1.0, 0.2],
                ],
                dtype=float,
            ),
            "features": np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.8, 0.2],
                ],
                dtype=np.float32,
            ),
            "projections": np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.8, 0.2],
                ],
                dtype=np.float32,
            ),
            "labels": np.array([0, 1, -1, -1, -1, 1]),
            "raw_labels": np.array([0, 1, 10, 11, 10, 1]),
            "is_known": np.array([1, 1, 0, 0, 0, 1]),
        }

        report, _, _, _ = run_discovery(
            outputs,
            threshold=1.0,
            num_novel=2,
            score_mode="entropy_only",
            cluster_k="oracle",
        )

        self.assertEqual(report["cluster_candidate_count"], 3)
        self.assertEqual(report["cluster_true_unknown_count"], 3)
        self.assertEqual(report["cluster_false_reject_count"], 0)
        self.assertTrue(math.isclose(report["cluster_candidate_purity"], 1.0))
        self.assertIn("cluster_all_unknown_oracle_nmi", report)
        self.assertIn("cluster_candidate_unknown_oracle_nmi", report)
        self.assertEqual(report["cluster_k"], 2)
        self.assertIn("unknown_reject_rate", report)


if __name__ == "__main__":
    unittest.main()
