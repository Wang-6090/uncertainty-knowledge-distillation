from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.losses import (
    discovery_unknown_loss,
    energy_margin_loss,
    per_sample_energy_margin_loss,
    proxy_contrastive_loss,
    supervised_contrastive_loss,
    weighted_energy_margin_loss,
)
from novel_discovery.pipeline import (
    calibration_diagnostics,
    compute_mahalanobis_distance,
    compute_open_score,
    compute_discovery_candidate_weights,
    filter_discovery_candidates_by_neighbors,
    fit_score_normalization,
    run_discovery,
    select_discovery_candidates,
    update_ema_model,
)
from train import parse_args, scheduled_weight


class CommandLineTest(unittest.TestCase):
    def test_discover_arguments_stay_in_sync_with_runtime(self):
        args = parse_args(
            [
                "discover",
                "--open-val-ratio", "0.2",
                "--temperature-calibration",
                "--calibration-bins", "10",
                "--auto-calibrate-score",
                "--score-mode", "odin_msp",
                "--odin-epsilon", "0.001",
                "--odin-temperature", "1000",
            ]
        )

        self.assertEqual(args.command, "discover")
        self.assertEqual(args.open_val_ratio, 0.2)
        self.assertTrue(args.temperature_calibration)
        self.assertEqual(args.calibration_bins, 10)
        self.assertTrue(args.auto_calibrate_score)
        self.assertEqual(args.score_mode, "odin_msp")
        self.assertAlmostEqual(args.odin_epsilon, 0.001)
        self.assertAlmostEqual(args.odin_temperature, 1000.0)

    def test_train_student_proxy_arguments_are_available(self):
        args = parse_args(
            [
                "train_student",
                "--alpha-proxy", "0.2",
                "--proxy-temperature", "0.07",
                "--discovery-pool",
                "--discovery-pool-mode", "mixed",
                "--alpha-discovery-selective-energy", "0.1",
                "--discovery-select-ratio", "0.25",
                "--discovery-select-mode", "consensus",
                "--discovery-selection-model", "ema",
                "--discovery-ema-decay", "0.95",
                "--discovery-neighbor-filter",
                "--discovery-neighbor-k", "3",
                "--discovery-neighbor-min-votes", "2",
                "--discovery-soft-weighting",
                "--discovery-neighbor-temperature", "0.3",
                "--discovery-selective-warmup-epochs", "1",
                "--discovery-selective-ramp-epochs", "2",
            ]
        )

        self.assertEqual(args.command, "train_student")
        self.assertAlmostEqual(args.alpha_proxy, 0.2)
        self.assertAlmostEqual(args.proxy_temperature, 0.07)
        self.assertEqual(args.discovery_pool_mode, "mixed")
        self.assertAlmostEqual(args.alpha_discovery_selective_energy, 0.1)
        self.assertAlmostEqual(args.discovery_select_ratio, 0.25)
        self.assertEqual(args.discovery_select_mode, "consensus")
        self.assertEqual(args.discovery_selection_model, "ema")
        self.assertAlmostEqual(args.discovery_ema_decay, 0.95)
        self.assertTrue(args.discovery_neighbor_filter)
        self.assertEqual(args.discovery_neighbor_k, 3)
        self.assertEqual(args.discovery_neighbor_min_votes, 2)
        self.assertTrue(args.discovery_soft_weighting)
        self.assertAlmostEqual(args.discovery_neighbor_temperature, 0.3)
        self.assertEqual(args.discovery_selective_warmup_epochs, 1)
        self.assertEqual(args.discovery_selective_ramp_epochs, 2)

    def test_selective_discovery_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)
        self.assertEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=0), 1.0)


class LossBehaviorTest(unittest.TestCase):
    def test_energy_margin_loss_is_finite_and_has_gradient(self):
        known = torch.tensor([[3.0, 0.0], [2.0, 0.0]], requires_grad=True)
        outlier = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)

        loss = energy_margin_loss(known, outlier, margin=1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(known.grad)

    def test_weighted_energy_margin_loss_has_per_sample_behavior(self):
        known = torch.tensor([[3.0, 0.0], [2.0, 0.0]], requires_grad=True)
        outlier = torch.tensor([[0.0, 0.0], [1.0, 0.0]], requires_grad=True)
        per_sample = per_sample_energy_margin_loss(known, outlier, margin=1.0)
        weighted = weighted_energy_margin_loss(
            known,
            outlier,
            outlier_weight=torch.tensor([1.0, 0.0]),
            margin=1.0,
        )

        self.assertEqual(per_sample.numel(), 2)
        self.assertAlmostEqual(weighted.item(), per_sample[0].item(), places=6)
        weighted.backward()
        self.assertIsNotNone(outlier.grad)

    def test_weighted_losses_return_zero_for_zero_weights(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        uncertainty = torch.zeros(2, requires_grad=True)
        loss = discovery_unknown_loss(
            logits,
            uncertainty,
            sample_weight=torch.zeros(2),
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

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

    def test_proxy_contrastive_loss_works_without_batch_positive_pairs(self):
        features = torch.eye(3, requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        proxies = torch.eye(3, requires_grad=True)

        loss = proxy_contrastive_loss(features, labels, proxies, temperature=0.1)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(features.grad)

    def test_update_ema_model_smooths_parameters(self):
        source = torch.nn.Linear(1, 1, bias=False)
        ema = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            source.weight.fill_(2.0)
            ema.weight.fill_(0.0)

        update_ema_model(ema, source, decay=0.9)

        self.assertAlmostEqual(ema.weight.item(), 0.2, places=6)
        self.assertFalse(ema.training)


class ScoreBehaviorTest(unittest.TestCase):
    def test_score_normalization_does_not_require_prototypes_for_entropy_stats(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 0.3]),
            "features": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        }

        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)

        self.assertIn("entropy", stats)
        self.assertNotIn("proto_dist", stats)

    def test_classwise_novel_mass_score_uses_known_class_statistics(self):
        outputs = {
            "entropy": np.zeros(4),
            "epistemic": np.zeros(4),
            "aleatoric": np.zeros(4),
            "features": np.arange(4, dtype=np.float32).reshape(4, 1),
            "logits": np.array(
                [[4.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 4.0]],
                dtype=np.float32,
            ),
            "unified_novel_mass": np.array([0.1, 0.3, 0.2, 0.4]),
        }
        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)
        score, _ = compute_open_score(
            outputs,
            score_mode="classwise_unified_novel_mass",
            normalization=stats,
        )

        self.assertIn("unified_novel_mass_classwise", stats)
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[1], score[0])

    def test_odin_score_requires_extracted_odin_values(self):
        outputs = {
            "entropy": np.array([0.1, 0.2]),
            "epistemic": np.array([0.0, 0.0]),
            "aleatoric": np.array([0.0, 0.0]),
            "probs": np.array([[0.9, 0.1], [0.6, 0.4]]),
            "features": np.array([[0.0], [1.0]], dtype=np.float32),
        }

        with self.assertRaises(ValueError):
            compute_open_score(outputs, score_mode="odin_msp")

        outputs["odin_msp"] = np.array([0.05, 0.4])
        score, _ = compute_open_score(outputs, score_mode="odin_msp")

        np.testing.assert_allclose(score, np.array([0.05, 0.4]))

    def test_shared_mahalanobis_uses_precision_matrix(self):
        features = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        stats = {
            "means": np.array([[0.0, 0.0]], dtype=np.float32),
            "variances": np.array([[1.0, 4.0]], dtype=np.float32),
            "precision": np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float32),
        }

        diag = compute_mahalanobis_distance(features, stats, covariance="diag")
        shared = compute_mahalanobis_distance(features, stats, covariance="shared")

        np.testing.assert_allclose(diag, np.array([0.5, 0.5]))
        np.testing.assert_allclose(shared, np.array([1.0, 1.0]))

    def test_non_mahalanobis_score_skips_mahalanobis_computation(self):
        outputs = {
            "entropy": np.array([0.1, 0.2]),
            "epistemic": np.zeros(2),
            "aleatoric": np.zeros(2),
            "probs": np.array([[0.9, 0.1], [0.6, 0.4]]),
            "features": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        }
        # Deliberately incompatible precision shape: non-Mahalanobis scores
        # should not touch this statistic.
        stats = {
            "means": np.zeros((1, 2), dtype=np.float32),
            "variances": np.ones((1, 2), dtype=np.float32),
            "precision": np.zeros((3, 3), dtype=np.float32),
        }

        score, _ = compute_open_score(
            outputs,
            score_mode="entropy_only",
            gaussian_stats=stats,
        )

        np.testing.assert_allclose(score, outputs["entropy"])

    def test_select_discovery_candidates_picks_high_entropy_samples(self):
        logits = torch.tensor(
            [
                [5.0, 0.0],
                [0.0, 0.0],
                [4.0, 0.0],
                [0.1, 0.1],
            ]
        )

        mask = select_discovery_candidates(logits, ratio=0.5, mode="entropy")

        self.assertEqual(mask.sum().item(), 2)
        self.assertTrue(mask[1].item())
        self.assertTrue(mask[3].item())

    def test_consensus_selection_requires_multiple_risk_signals(self):
        logits = torch.tensor(
            [
                [5.0, 0.0],
                [0.0, 0.0],
                [4.0, 0.0],
                [0.1, 0.1],
                [2.0, 2.0],
                [3.5, 0.0],
                [0.2, 0.2],
                [6.0, 0.0],
            ]
        )
        uncertainty = torch.tensor([0.0, 0.9, 0.1, 0.8, 0.95, 0.1, 0.85, 0.0])

        mask = select_discovery_candidates(logits, uncertainty, ratio=0.25, mode="consensus")

        self.assertLessEqual(mask.sum().item(), 2)
        self.assertTrue(mask[1].item() or mask[3].item() or mask[4].item() or mask[6].item())
        self.assertFalse(mask[0].item())

    def test_neighbor_filter_keeps_candidates_with_selected_neighbors(self):
        mask = torch.tensor([True, True, False, True, False])
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )

        filtered, agreement = filter_discovery_candidates_by_neighbors(
            mask,
            features,
            k=2,
            min_votes=1,
        )

        self.assertTrue(filtered[0].item())
        self.assertTrue(filtered[1].item())
        self.assertFalse(filtered[3].item())
        self.assertGreater(agreement, 0.0)

    def test_soft_candidate_weights_are_bounded_and_nonuniform(self):
        logits = torch.tensor(
            [
                [5.0, 0.0],
                [0.0, 0.0],
                [4.0, 0.0],
                [0.1, 0.1],
            ]
        )
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        mask, weights, _ = compute_discovery_candidate_weights(
            logits,
            features=features,
            ratio=0.5,
            mode="entropy",
            neighbor_k=2,
        )

        self.assertEqual(mask.sum().item(), 2)
        self.assertTrue(torch.all(weights >= 0.0))
        self.assertTrue(torch.all(weights <= 1.0))
        self.assertTrue(torch.all(weights[~mask] == 0.0))
        self.assertGreater(weights[mask].max().item(), weights[mask].min().item())


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
            "odin_msp": np.zeros(6, dtype=float),
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

        report, _, _, detail = run_discovery(
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
        self.assertIsNotNone(detail["pred_cluster"])
        self.assertTrue(np.all(detail["pred_cluster"][2:5] >= 0))

    def test_run_discovery_can_skip_clustering(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 2.0, 2.2]),
            "epistemic": np.zeros(4),
            "aleatoric": np.zeros(4),
            "probs": np.array([[0.9, 0.1], [0.8, 0.2], [0.5, 0.5], [0.4, 0.6]]),
            "logits": np.array([[2.0, 0.0], [1.5, 0.0], [0.1, 0.2], [0.2, 0.1]]),
            "features": np.eye(4, dtype=np.float32),
            "projections": np.eye(4, dtype=np.float32),
            "labels": np.array([0, 1, -1, -1]),
            "raw_labels": np.array([0, 1, 10, 11]),
            "is_known": np.array([1, 1, 0, 0]),
        }
        report, _, _, detail = run_discovery(
            outputs,
            threshold=1.0,
            num_novel=2,
            score_mode="entropy_only",
            enable_clustering=False,
        )

        self.assertTrue(report["clustering_skipped"])
        self.assertIsNone(detail["pred_cluster"])
        self.assertNotIn("cluster_candidate_count", report)


if __name__ == "__main__":
    unittest.main()
