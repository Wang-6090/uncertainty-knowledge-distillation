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
from novel_discovery.pipeline import (
    calibrate_class_thresholds,
    evaluate_cluster_candidates,
    fit_score_normalization,
    purify_candidate_mask,
    run_discovery,
)


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


class DiscoveryReportTest(unittest.TestCase):
    def test_class_conditional_thresholds_fall_back_for_small_classes(self):
        scores_known = np.array([1.0, 1.1, 1.2, 3.0, 3.1, 3.2])
        thresholds = calibrate_class_thresholds(
            scores_known=scores_known,
            predicted_classes=np.array([0, 0, 0, 1, 1, 1]),
            num_classes=3,
            percentile=90.0,
            min_samples=3,
        )

        self.assertGreater(thresholds[1], thresholds[0])
        self.assertEqual(thresholds[2], np.percentile(scores_known, 90.0))

    def test_candidate_purification_keeps_only_high_uncertainty_rejections(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 0.9, 1.0, 1.1]),
            "epistemic": np.zeros(5),
            "head_uncertainty": np.array([0.1, 0.2, 0.4, 0.8, 0.9]),
            "probs": np.array(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.6, 0.4],
                    [0.4, 0.6],
                    [0.5, 0.5],
                ]
            ),
        }
        mask, report = purify_candidate_mask(
            outputs,
            np.array([False, False, True, True, True]),
            mode="head_uncertainty",
            keep_ratio=2 / 3,
        )

        self.assertEqual(report["before_count"], 3)
        self.assertEqual(report["after_count"], 2)
        self.assertTrue(np.array_equal(mask, np.array([False, False, False, True, True])))

    def test_candidate_purification_can_rank_by_open_score(self):
        outputs = {
            "entropy": np.zeros(5),
            "epistemic": np.zeros(5),
            "head_uncertainty": np.zeros(5),
            "probs": np.full((5, 2), 0.5),
        }
        mask, report = purify_candidate_mask(
            outputs,
            np.array([True, True, True, False, False]),
            mode="open_score",
            keep_ratio=2 / 3,
            open_score=np.array([0.1, 0.9, 0.8, 0.7, 0.6]),
        )

        self.assertEqual(report["before_count"], 3)
        self.assertEqual(report["after_count"], 2)
        self.assertTrue(np.array_equal(mask, np.array([False, True, True, False, False])))

    def test_auto_k_search_reaches_protocol_maximum(self):
        rng = np.random.default_rng(7)
        centers = np.array(
            [
                [-6.0, -6.0],
                [-6.0, 6.0],
                [6.0, -6.0],
                [6.0, 6.0],
            ],
            dtype=np.float32,
        )
        features = np.concatenate(
            [center + 0.15 * rng.normal(size=(12, 2)) for center in centers],
            axis=0,
        ).astype(np.float32)

        selected, diagnostics = evaluate_cluster_candidates(
            features,
            max_clusters=4,
            selection="silhouette",
            stability_repeats=2,
        )

        self.assertEqual(selected, 4)
        self.assertEqual(diagnostics[-1]["k"], 4)

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


if __name__ == "__main__":
    unittest.main()
