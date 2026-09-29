from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.losses import (
    discovery_unknown_loss,
    angular_margin_loss,
    proxy_anchor_loss,
    reciprocal_point_loss,
    known_pseudo_label_consistency_loss,
    energy_margin_loss,
    per_sample_energy_margin_loss,
    proxy_contrastive_loss,
    supervised_contrastive_loss,
    weighted_energy_margin_loss,
    uncertainty_weights,
    unknown_feature_margin_loss,
    unknown_feature_separation_loss,
    uncertainty_separation_loss,
    outlier_exposure_uniform_loss,
    prototype_repulsion_loss,
)
from novel_discovery.data import (
    build_data_bundle,
    build_open_validation_and_discovery,
    make_class_split,
    limit_dataset,
    split_known_indices,
    split_known_for_discovery,
    _known_labels,
    validate_class_split,
)
from scripts.make_cifar100_splits import build_protocols
from novel_discovery.pipeline import (
    calibration_diagnostics,
    attach_knn_distances,
    attach_vim_residual,
    compute_mahalanobis_distance,
    compute_gaussian_nll,
    compute_openmax_score,
    compute_reciprocal_score,
    compute_relative_mahalanobis_distance,
    calibrate_coverage_threshold,
    compute_open_score,
    compute_discovery_candidate_weights,
    evaluate_cluster_candidates,
    filter_discovery_candidates_by_neighbors,
    fit_score_normalization,
    run_discovery,
    calibrate_class_thresholds,
    calibrate_open_threshold,
    purify_candidate_mask,
    candidate_purification_score,
    evaluate_candidate_purification,
    select_discovery_candidates,
    select_joint_candidates,
    compute_joint_candidate_weights,
    update_ema_model,
    synthesize_virtual_outliers,
    synthesize_gaussian_virtual_outliers,
    fit_feature_rejector,
    fit_virtual_outlier_rejector,
    fit_known_support_rejector,
    attach_known_support_score,
    fit_pu_feature_rejector,
    fit_nnpu_feature_rejector,
    attach_feature_rejector_score,
)
from train import parse_args, scheduled_weight


class CommandLineTest(unittest.TestCase):
    def test_feature_rejector_is_separate_score(self):
        known = {"features": np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1]])}
        unknown = {"features": np.asarray([[-1.0, 0.0], [-0.9, -0.1], [-1.0, -0.1]])}
        rejector = fit_feature_rejector(known, unknown, max_samples=10, seed=0)
        outputs = {
            "features": np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
            "entropy": np.zeros(2),
            "epistemic": np.zeros(2),
            "aleatoric": np.zeros(2),
        }
        attach_feature_rejector_score(outputs, rejector)
        self.assertEqual(outputs["feature_rejector_score"].shape, (2,))
        self.assertLess(outputs["feature_rejector_score"][0], outputs["feature_rejector_score"][1])
        score, _ = compute_open_score(outputs, score_mode="feature_rejector")
        np.testing.assert_allclose(score, outputs["feature_rejector_score"])

    def test_support_augmented_rejector_requires_and_uses_support_score(self):
        known = {
            "features": np.asarray([[1.0, 0.0], [0.9, 0.1]]),
            "logits": np.zeros((2, 2)),
            "probs": np.full((2, 2), 0.5),
            "entropy": np.zeros(2),
            "head_uncertainty": np.zeros(2),
            "known_support_score": np.asarray([0.1, 0.2]),
        }
        unknown = {key: value.copy() for key, value in known.items()}
        unknown["features"] = np.asarray([[-1.0, 0.0], [-0.9, -0.1]])
        unknown["known_support_score"] = np.asarray([2.0, 2.1])
        rejector = fit_feature_rejector(
            known, unknown, feature_mode="support_augmented", seed=0
        )
        outputs = {key: value[:1].copy() for key, value in unknown.items()}
        attach_feature_rejector_score(outputs, rejector, feature_mode="support_augmented")
        self.assertIn("feature_rejector_score", outputs)

    def test_soft_pu_feature_rejector_accepts_unlabeled_pool(self):
        known = {"features": np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1]])}
        unlabeled = {"features": np.asarray([[-1.0, 0.0], [-0.9, -0.1], [0.8, 0.2]])}
        rejector = fit_pu_feature_rejector(known, unlabeled, max_samples=10, iterations=2, seed=0)
        self.assertEqual(rejector.coef_.shape, (1, 2))

    def test_nnpu_feature_rejector_returns_unknown_score(self):
        known = {"features": np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1]])}
        unlabeled = {
            "features": np.asarray(
                [[-1.0, 0.0], [-0.9, -0.1], [0.8, 0.2], [0.7, 0.3]]
            )
        }
        rejector = fit_nnpu_feature_rejector(
            known, unlabeled, known_prior=0.5, iterations=30, seed=0
        )
        known_score = float(rejector.decision_function(np.asarray([[1.0, 0.0]]))[0])
        unknown_score = float(rejector.decision_function(np.asarray([[-1.0, 0.0]]))[0])
        self.assertTrue(np.isfinite([known_score, unknown_score]).all())
        self.assertGreater(unknown_score, known_score)

    def test_virtual_outlier_rejector_uses_known_features_only(self):
        rng = np.random.default_rng(0)
        features = np.concatenate(
            [rng.normal(loc=-1.0, scale=0.1, size=(8, 3)),
             rng.normal(loc=1.0, scale=0.1, size=(8, 3))], axis=0
        ).astype(np.float32)
        outputs = {"features": features, "labels": np.array([0] * 8 + [1] * 8)}
        rejector = fit_virtual_outlier_rejector(outputs, max_samples=8, seed=0)
        self.assertTrue(hasattr(rejector, "predict_proba"))

    def test_known_support_detector_scores_far_features_higher(self):
        known = {
            "features": np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            "labels": np.asarray([0, 0, 1, 1]),
        }
        support = fit_known_support_rejector(known, quantile=0.95)
        outputs = {"features": np.asarray([[1.0, 0.0], [-1.0, 0.0]])}
        attach_known_support_score(outputs, support)
        self.assertGreater(outputs["known_support_score"][1], outputs["known_support_score"][0])

    def test_class_split_validation_rejects_overlap_and_wrong_count(self):
        with self.assertRaises(ValueError):
            validate_class_split([0, 1, 2], [0, 1], [1, 2], num_known=2)
        with self.assertRaises(ValueError):
            validate_class_split([0, 1, 2], [0], [1], num_known=1)

    def test_class_split_file_is_validated_before_loading(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "split.json"
            path.write_text(
                '{"known_classes": [0, 1], "novel_classes": [1, 2]}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                make_class_split([0, 1, 2], 2, 0, str(path))

    def test_stratified_known_split_keeps_each_class_in_train_and_val(self):
        labels = [label for label in range(4) for _ in range(10)]
        train_indices, val_indices = split_known_indices(
            len(labels), val_ratio=0.2, seed=7, labels=labels
        )
        train_labels = {labels[index] for index in train_indices}
        val_labels = {labels[index] for index in val_indices}
        self.assertEqual(train_labels, set(range(4)))
        self.assertEqual(val_labels, set(range(4)))
        self.assertEqual(len(train_indices) + len(val_indices), len(labels))

    def test_default_known_split_remains_random_compatible(self):
        random_split = split_known_indices(20, val_ratio=0.2, seed=7)
        explicit_random = split_known_indices(20, val_ratio=0.2, seed=7, labels=None)
        self.assertEqual(random_split, explicit_random)

    def test_stratified_limit_keeps_each_class_when_budget_allows(self):
        dataset = torch.utils.data.TensorDataset(
            torch.zeros(12, 1),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        )
        limited = limit_dataset(
            dataset,
            max_items=8,
            seed=9,
            labels=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        )
        selected_labels = {int(limited[index][1]) for index in range(len(limited))}
        self.assertEqual(selected_labels, {0, 1, 2, 3})
        self.assertEqual(len(limited), 8)

    def test_stratified_split_handles_subset_without_changing_dataset_contract(self):
        base = torch.utils.data.TensorDataset(
            torch.arange(20).float().unsqueeze(1),
            torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5),
        )
        subset = torch.utils.data.Subset(base, list(range(16)))
        self.assertEqual(_known_labels(subset), [0] * 5 + [1] * 5 + [2] * 5 + [3])
        train, val = split_known_indices(
            len(subset), val_ratio=0.25, seed=4,
            labels=[int(subset[index][1]) for index in range(len(subset))],
        )
        self.assertEqual(len(train) + len(val), len(subset))
        self.assertTrue(set(int(subset[index][1]) for index in val))

    def test_known_split_mode_is_exposed_on_all_commands(self):
        args = parse_args(["train_teacher", "--known-split-mode", "stratified"])
        self.assertEqual(args.known_split_mode, "stratified")

    def test_cifar_protocols_have_expected_semantic_structure(self):
        fine_names = [f"fine_{i}" for i in range(100)]
        coarse_names = [f"coarse_{i}" for i in range(20)]
        fine_to_coarse = {fine_id: fine_id // 5 for fine_id in range(100)}
        protocols = build_protocols(fine_names, coarse_names, fine_to_coarse, seed=42)

        self.assertEqual(set(protocols), {"random", "semantic_hard", "semantic_isolated"})
        for protocol in protocols.values():
            self.assertEqual(len(protocol["known_classes"]), 60)
            self.assertEqual(len(protocol["novel_classes"]), 40)
            self.assertEqual(
                set(protocol["known_classes"]) | set(protocol["novel_classes"]),
                set(range(100)),
            )
        hard_counts = {
            coarse_id: sum(
                fine_to_coarse[fine_id] == coarse_id
                for fine_id in protocols["semantic_hard"]["known_classes"]
            )
            for coarse_id in range(20)
        }
        self.assertEqual(set(hard_counts.values()), {3})
        isolated_counts = {
            coarse_id: sum(
                fine_to_coarse[fine_id] == coarse_id
                for fine_id in protocols["semantic_isolated"]["known_classes"]
            )
            for coarse_id in range(20)
        }
        self.assertEqual(set(isolated_counts.values()), {0, 5})

    def test_teacher_proxy_anchor_arguments_are_available(self):
        args = parse_args(["train_teacher"])
        self.assertEqual(args.alpha_proxy_anchor, 0.0)
        self.assertEqual(args.proxy_anchor_alpha, 32.0)
        self.assertEqual(args.proxy_anchor_margin, 0.1)

    def test_imagefolder_requires_separate_train_and_test_roots(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                build_data_bundle("imagefolder", root, 1, 0, 32)

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
                "--max-auto-clusters", "48",
                "--threshold-policy", "open_known_coverage",
                "--react-percentile", "99.5",
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
        self.assertEqual(args.max_auto_clusters, 48)
        self.assertEqual(args.threshold_policy, "open_known_coverage")
        self.assertAlmostEqual(args.react_percentile, 99.5)

    def test_open_validation_is_reserved_from_train_splits_not_test(self):
        def make_dataset(start, count, known):
            ids = torch.arange(start, start + count)
            labels = torch.zeros(count, dtype=torch.long) if known else torch.full((count,), -1)
            known_flags = torch.ones(count, dtype=torch.long) if known else torch.zeros(count, dtype=torch.long)
            return torch.utils.data.TensorDataset(
                torch.zeros(count, 3, 2, 2), labels, ids, known_flags, ids
            )

        known_train = make_dataset(0, 20, known=True)
        known_val = make_dataset(100, 20, known=True)
        unknown_pool = make_dataset(200, 40, known=False)
        untouched_test = make_dataset(300, 30, known=False)
        val_cal, open_val, discovery = build_open_validation_and_discovery(
            known_train,
            known_val,
            unknown_pool,
            open_val_ratio=0.25,
            seed=7,
            discovery_pool_mode="unknown",
        )

        ids = lambda dataset: {int(dataset[index][4]) for index in range(len(dataset))}
        cal_ids = ids(val_cal)
        open_ids = ids(open_val)
        discovery_ids = ids(discovery)
        test_ids = ids(untouched_test)

        self.assertTrue(cal_ids.isdisjoint(open_ids))
        self.assertTrue(open_ids.isdisjoint(discovery_ids))
        self.assertTrue(open_ids.isdisjoint(test_ids))
        self.assertTrue(discovery_ids.isdisjoint(test_ids))
        self.assertTrue(all(100 <= sample_id < 120 for sample_id in cal_ids))
        self.assertTrue(any(100 <= sample_id < 120 for sample_id in open_ids))
        self.assertTrue(any(200 <= sample_id < 240 for sample_id in open_ids))

    def test_mixed_discovery_reserves_disjoint_known_samples(self):
        def make_dataset(start, count, known):
            ids = torch.arange(start, start + count)
            labels = torch.zeros(count, dtype=torch.long) if known else torch.full((count,), -1)
            known_flags = torch.ones(count, dtype=torch.long) if known else torch.zeros(count, dtype=torch.long)
            return torch.utils.data.TensorDataset(
                torch.zeros(count, 3, 2, 2), labels, ids, known_flags, ids
            )

        known = make_dataset(0, 20, known=True)
        unknown = make_dataset(100, 12, known=False)
        supervised, known_pool = split_known_for_discovery(known, 0.25, seed=11)
        _, _, mixed = build_open_validation_and_discovery(
            supervised,
            make_dataset(200, 4, known=True),
            unknown,
            open_val_ratio=0.0,
            seed=11,
            discovery_pool_mode="mixed",
            known_discovery_pool=known_pool,
        )
        ids = lambda dataset: {int(dataset[index][4]) for index in range(len(dataset))}
        supervised_ids = ids(supervised)
        known_pool_ids = ids(known_pool)
        mixed_ids = ids(mixed)
        self.assertTrue(supervised_ids.isdisjoint(known_pool_ids))
        self.assertTrue(known_pool_ids.issubset(mixed_ids))
        self.assertTrue({100, 101}.issubset(mixed_ids))

    def test_mixed_discovery_rejects_reusing_supervised_known_samples(self):
        dataset = torch.utils.data.TensorDataset(
            torch.zeros(4, 3, 2, 2),
            torch.zeros(4, dtype=torch.long),
            torch.arange(4),
            torch.ones(4, dtype=torch.long),
            torch.arange(4),
        )
        with self.assertRaises(ValueError):
            build_open_validation_and_discovery(
                dataset,
                dataset,
                dataset,
                open_val_ratio=0.0,
                seed=1,
                discovery_pool_mode="mixed",
            )

    def test_train_student_proxy_arguments_are_available(self):
        args = parse_args(
            [
                "train_student",
                "--alpha-proxy", "0.2",
                "--proxy-temperature", "0.07",
                "--uncertainty-weight-min", "0.5",
                "--uncertainty-weight-max", "1.0",
                "--discovery-pool",
                "--discovery-pool-mode", "mixed",
                "--rejector-strict-mixed",
                "--alpha-discovery-selective-energy", "0.1",
                "--alpha-discovery-uniform", "0.07",
                "--discovery-uniform-warmup-epochs", "1",
                "--discovery-uniform-ramp-epochs", "2",
                "--discovery-select-ratio", "0.25",
                "--discovery-select-mode", "consensus",
                "--discovery-selection-model", "ema",
                "--discovery-ema-decay", "0.95",
                "--discovery-neighbor-filter",
                "--discovery-neighbor-k", "3",
                "--discovery-neighbor-min-votes", "2",
                "--discovery-soft-weighting",
                "--discovery-neighbor-temperature", "0.3",
                "--alpha-discovery-feature-margin", "0.05",
                "--discovery-feature-margin", "0.2",
                "--discovery-selective-warmup-epochs", "1",
            "--discovery-selective-ramp-epochs", "2",
                "--uncertainty-separation-warmup-epochs", "1",
                "--uncertainty-separation-ramp-epochs", "2",
                "--outlier-dataset", "cifar10",
                "--outlier-data-root", ".\\data",
                "--alpha-outlier-feature-margin", "0.05",
                "--outlier-feature-margin", "0.15",
            ]
        )

        self.assertEqual(args.command, "train_student")
        self.assertAlmostEqual(args.alpha_proxy, 0.2)
        self.assertAlmostEqual(args.proxy_temperature, 0.07)
        self.assertAlmostEqual(args.uncertainty_weight_min, 0.5)
        self.assertAlmostEqual(args.uncertainty_weight_max, 1.0)
        self.assertEqual(args.discovery_pool_mode, "mixed")
        self.assertTrue(args.rejector_strict_mixed)
        self.assertAlmostEqual(args.alpha_discovery_selective_energy, 0.1)
        self.assertAlmostEqual(args.alpha_discovery_uniform, 0.07)
        self.assertEqual(args.discovery_uniform_warmup_epochs, 1)
        self.assertEqual(args.discovery_uniform_ramp_epochs, 2)
        self.assertAlmostEqual(args.discovery_select_ratio, 0.25)
        self.assertEqual(args.discovery_select_mode, "consensus")
        self.assertEqual(args.discovery_selection_model, "ema")
        self.assertAlmostEqual(args.discovery_ema_decay, 0.95)
        self.assertTrue(args.discovery_neighbor_filter)
        self.assertEqual(args.discovery_neighbor_k, 3)
        self.assertEqual(args.discovery_neighbor_min_votes, 2)
        self.assertTrue(args.discovery_soft_weighting)
        self.assertAlmostEqual(args.discovery_neighbor_temperature, 0.3)
        self.assertAlmostEqual(args.alpha_discovery_feature_margin, 0.05)
        self.assertAlmostEqual(args.discovery_feature_margin, 0.2)
        self.assertEqual(args.outlier_dataset, "cifar10")
        self.assertEqual(args.outlier_data_root, ".\\data")
        self.assertAlmostEqual(args.alpha_outlier_feature_margin, 0.05)
        self.assertAlmostEqual(args.outlier_feature_margin, 0.15)
        self.assertEqual(args.discovery_selective_warmup_epochs, 1)
        self.assertEqual(args.discovery_selective_ramp_epochs, 2)
        self.assertEqual(args.uncertainty_separation_warmup_epochs, 1)
        self.assertEqual(args.uncertainty_separation_ramp_epochs, 2)

    def test_selective_discovery_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)
        self.assertEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=0), 1.0)

    def test_uncertainty_separation_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)

    def test_discovery_uniform_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)


class LossBehaviorTest(unittest.TestCase):
    def test_angular_margin_loss_is_finite_and_has_gradient(self):
        features = torch.randn(6, 4, requires_grad=True)
        weights = torch.randn(3, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = angular_margin_loss(features, labels, weights, margin=0.2, scale=16.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(weights.grad)

    def test_prototype_repulsion_loss_is_finite_and_has_gradient(self):
        prototypes = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], requires_grad=True
        )
        loss = prototype_repulsion_loss(prototypes, similarity_margin=0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(prototypes.grad)

    def test_openmax_score_is_finite_and_increases_for_far_features(self):
        stats = {
            "means": np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32),
            "weibull_shapes": np.array([2.0, 2.0], dtype=np.float32),
            "weibull_locations": np.array([0.0, 0.0], dtype=np.float32),
            "weibull_scales": np.array([1.0, 1.0], dtype=np.float32),
        }
        near = compute_openmax_score(np.array([[0.1, 0.1]], dtype=np.float32), stats)
        far = compute_openmax_score(np.array([[10.0, 10.0]], dtype=np.float32), stats)
        self.assertTrue(np.isfinite(near).all())
        self.assertTrue(np.isfinite(far).all())
        self.assertGreater(float(far[0]), float(near[0]))

    def test_proxy_anchor_loss_is_finite_and_has_gradient(self):
        features = torch.randn(8, 4, requires_grad=True)
        proxies = torch.randn(4, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        loss = proxy_anchor_loss(features, labels, proxies)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(proxies.grad)

    def test_reciprocal_point_loss_and_score_are_finite_and_differentiable(self):
        known = torch.randn(6, 4, requires_grad=True)
        unknown = torch.randn(5, 4, requires_grad=True)
        prototypes = torch.randn(3, 4, requires_grad=True)
        points = torch.nn.Parameter(torch.randn(2, 4))
        loss = reciprocal_point_loss(known, unknown, prototypes, points)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(points.grad)
        scores = compute_reciprocal_score(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        self.assertTrue(np.allclose(scores, [1.0, 0.0]))

    def test_known_pseudo_label_consistency_uses_only_selected_samples(self):
        first = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0]], requires_grad=True
        )
        second = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0]], requires_grad=True
        )
        labels = torch.tensor([0, 1])
        selected = torch.tensor([True, False])
        loss = known_pseudo_label_consistency_loss(first, second, labels, selected)
        expected = 0.5 * (
            torch.nn.functional.cross_entropy(first[:1], labels[:1])
            + torch.nn.functional.cross_entropy(second[:1], labels[:1])
        )
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)
        self.assertTrue(torch.allclose(first.grad[1], torch.zeros_like(first.grad[1])))
        self.assertTrue(torch.allclose(second.grad[1], torch.zeros_like(second.grad[1])))

    def test_uncertainty_weights_can_be_clipped(self):
        values = uncertainty_weights(
            torch.tensor([0.0, 2.0]),
            uncertainty_weight_min=0.5,
            uncertainty_weight_max=0.75,
        )
        self.assertTrue(torch.allclose(values, torch.tensor([0.75, 0.5])))

    def test_energy_margin_loss_is_finite_and_has_gradient(self):
        known = torch.tensor([[3.0, 0.0], [2.0, 0.0]], requires_grad=True)
        outlier = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)

        loss = energy_margin_loss(known, outlier, margin=1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(known.grad)

    def test_vos_virtual_outliers_are_finite_and_outside_known_proxy_cone(self):
        torch.manual_seed(7)
        features = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]], requires_grad=True
        )
        labels = torch.tensor([0, 1])
        proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        virtual = synthesize_virtual_outliers(
            features, labels, proxies, tail_scale=2.0, noise_scale=0.0
        )
        self.assertEqual(tuple(virtual.shape), tuple(features.shape))
        self.assertTrue(torch.isfinite(virtual).all())
        cosine = torch.nn.functional.cosine_similarity(
            virtual.detach(), proxies.detach()[labels], dim=-1
        )
        self.assertTrue(torch.all(cosine < 0.8))
        virtual.sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(proxies.grad)

    def test_gaussian_vos_uses_class_statistics_and_keeps_shape(self):
        torch.manual_seed(11)
        labels = torch.tensor([0, 1, 0, 1])
        means = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        variances = torch.tensor([[0.25, 0.25], [0.04, 0.04]])
        virtual = synthesize_gaussian_virtual_outliers(
            labels, means, variances, tail_scale=2.0
        )
        self.assertEqual(tuple(virtual.shape), (4, 2))
        self.assertTrue(torch.isfinite(virtual).all())
        self.assertGreater(torch.norm(virtual[1] - means[1]).item(), 0.0)

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

    def test_unknown_feature_margin_pushes_features_from_known_prototypes(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
        )
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = unknown_feature_margin_loss(features, prototypes, similarity_margin=0.2)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)

        far_features = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
        far_loss = unknown_feature_margin_loss(far_features, prototypes, similarity_margin=0.2)
        self.assertAlmostEqual(far_loss.item(), 0.0, places=6)

    def test_uncertainty_separation_loss_has_known_unknown_direction(self):
        known = torch.tensor([0.1, 0.2], requires_grad=True)
        unknown = torch.tensor([0.8, 0.9], requires_grad=True)
        loss = uncertainty_separation_loss(known, unknown)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(known.grad.mean().item(), 0.0)
        self.assertLess(unknown.grad.mean().item(), 0.0)

    def test_unknown_feature_separation_is_not_scaled_by_known_batch_size(self):
        unknown = torch.tensor([[1.0, 0.0]])
        one_known = torch.tensor([[0.5, 0.8660254]])
        repeated_known = one_known.repeat(64, 1)
        one = unknown_feature_separation_loss(
            one_known, unknown, similarity_margin=0.0, temperature=0.1
        )
        repeated = unknown_feature_separation_loss(
            repeated_known, unknown, similarity_margin=0.0, temperature=0.1
        )
        self.assertAlmostEqual(one.item(), repeated.item(), places=5)

    def test_outlier_exposure_uniform_loss_prefers_uniform_logits(self):
        uniform = torch.zeros(2, 3, requires_grad=True)
        concentrated = torch.tensor([[5.0, 0.0, 0.0]], requires_grad=True)
        uniform_loss = outlier_exposure_uniform_loss(uniform)
        concentrated_loss = outlier_exposure_uniform_loss(concentrated)
        self.assertLess(uniform_loss.item(), concentrated_loss.item())
        concentrated_loss.backward()
        self.assertTrue(torch.isfinite(concentrated.grad).all())

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
    def test_relative_mahalanobis_uses_class_minus_background_density(self):
        stats = {
            "means": np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32),
            "variances": np.ones((2, 2), dtype=np.float32),
            "precision": np.eye(2, dtype=np.float32),
            "global_mean": np.array([2.0, 0.0], dtype=np.float32),
            "global_precision": np.eye(2, dtype=np.float32),
        }
        scores = compute_relative_mahalanobis_distance(
            np.array([[0.1, 0.0], [2.0, 0.0]], dtype=np.float32), stats
        )
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(scores[1], scores[0])

    def test_score_normalization_does_not_require_prototypes_for_entropy_stats(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 0.3]),
            "features": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        }

        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)

        self.assertIn("entropy", stats)
        self.assertNotIn("proto_dist", stats)

    def test_normalized_full_requires_known_validation_stats_and_is_finite(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 0.4, 0.8]),
            "epistemic": np.array([0.01, 0.02, 0.04, 0.08]),
            "aleatoric": np.array([0.2, 0.3, 0.5, 0.9]),
            "features": np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        }
        prototypes = np.array([[0.0], [2.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_open_score(outputs, prototypes=prototypes, score_mode="normalized_full")

        stats = fit_score_normalization(outputs, prototypes=prototypes, gaussian_stats=None)
        score, proto_dist = compute_open_score(
            outputs,
            prototypes=prototypes,
            score_mode="normalized_full",
            normalization=stats,
        )

        self.assertIn("epistemic", stats)
        self.assertIn("aleatoric", stats)
        self.assertIn("proto_dist", stats)
        self.assertEqual(score.shape, (4,))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.isfinite(proto_dist).all())

    def test_knn_ood_uses_nearest_known_projection_distances(self):
        outputs = {"features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
        attach_knn_distances(
            outputs,
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            k=1,
            feature_key="features",
        )
        np.testing.assert_allclose(outputs["knn_distance"], np.array([0.0, 0.0]), atol=1e-6)

        outputs["entropy"] = np.array([0.1, 0.5])
        outputs["epistemic"] = np.zeros(2)
        outputs["aleatoric"] = np.zeros(2)
        outputs["probs"] = np.array([[0.9, 0.1], [0.5, 0.5]])
        outputs["features"] = np.zeros((2, 2), dtype=np.float32)
        score, _ = compute_open_score(outputs, score_mode="knn_distance")
        np.testing.assert_allclose(score, outputs["knn_distance"])

    def test_gaussian_nll_prefers_features_near_known_class_distribution(self):
        stats = {
            "means": np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32),
            "variances": np.ones((2, 2), dtype=np.float32),
        }
        nll = compute_gaussian_nll(
            np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32), stats
        )
        self.assertLess(nll[0], nll[1])

    def test_logit_scores_have_expected_unknown_direction(self):
        outputs = {
            "logits": np.array([[5.0, 1.0, 0.0], [1.0, 1.1, 1.0]], dtype=np.float32),
            "entropy": np.zeros(2),
            "epistemic": np.zeros(2),
            "aleatoric": np.zeros(2),
        }
        max_logit, _ = compute_open_score(outputs, score_mode="max_logit")
        margin, _ = compute_open_score(outputs, score_mode="logit_margin")
        self.assertLess(max_logit[0], max_logit[1])
        self.assertLess(margin[0], margin[1])

    def test_uncertainty_scores_are_available_for_open_detection(self):
        outputs = {
            "logits": np.array([[5.0, 1.0], [1.0, 1.1]], dtype=np.float32),
            "entropy": np.array([0.1, 0.6]),
            "epistemic": np.zeros(2),
            "aleatoric": np.zeros(2),
            "head_uncertainty": np.array([0.1, 0.8]),
        }
        score, _ = compute_open_score(outputs, score_mode="head_uncertainty")
        np.testing.assert_allclose(score, outputs["head_uncertainty"])
        combined, _ = compute_open_score(outputs, score_mode="margin_uncertainty")
        self.assertGreater(combined[1], combined[0])

    def test_vim_residual_is_zero_inside_known_subspace(self):
        outputs = {"features": np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)}
        stats = {
            "center": np.zeros(2, dtype=np.float32),
            "components": np.array([[1.0, 0.0]], dtype=np.float32),
        }
        attach_vim_residual(outputs, stats)
        self.assertAlmostEqual(outputs["vim_residual"][0], 0.0, places=6)
        self.assertAlmostEqual(outputs["vim_residual"][1], 2.0, places=6)

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

    def test_uncertainty_aware_novel_mass_score_uses_known_statistics(self):
        outputs = {
            "entropy": np.array([0.1, 0.4, 0.8]),
            "epistemic": np.zeros(3),
            "aleatoric": np.zeros(3),
            "features": np.arange(3, dtype=np.float32).reshape(3, 1),
            "logits": np.array([[4.0, 0.0], [2.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            "unified_novel_mass": np.array([0.1, 0.2, 0.3]),
        }
        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)
        score, _ = compute_open_score(
            outputs,
            score_mode="normalized_unified_novel_mass_entropy",
            normalization=stats,
        )
        self.assertIn("unified_novel_mass", stats)
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[2], score[0])

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

    def test_chunked_mahalanobis_matches_full_batch_for_both_covariances(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(23, 7)).astype(np.float32)
        means = rng.normal(size=(5, 7)).astype(np.float32)
        variances = np.exp(rng.normal(size=(5, 7))).astype(np.float32)
        matrix = rng.normal(size=(7, 7))
        precision = (matrix.T @ matrix + np.eye(7)).astype(np.float32)
        stats = {"means": means, "variances": variances, "precision": precision}

        for covariance in ("diag", "shared"):
            full_batch = compute_mahalanobis_distance(
                features, stats, covariance=covariance, chunk_size=len(features)
            )
            chunked = compute_mahalanobis_distance(
                features, stats, covariance=covariance, chunk_size=4
            )
            np.testing.assert_allclose(chunked, full_batch, rtol=1e-6, atol=1e-6)

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

    def test_joint_novel_mass_selection_prefers_novel_subspace(self):
        known_logits = torch.tensor(
            [[5.0, 0.0], [5.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )
        novel_logits = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [4.0, 4.0]]
        )
        mask = select_joint_candidates(
            known_logits, novel_logits, ratio=0.5, mode="novel_mass"
        )
        self.assertEqual(mask.sum().item(), 2)
        self.assertTrue(mask[2].item())
        self.assertTrue(mask[3].item())

    def test_joint_novel_mass_weights_require_novel_logits(self):
        with self.assertRaises(ValueError):
            compute_joint_candidate_weights(
                torch.zeros(4, 2), ratio=0.5, mode="novel_mass"
            )

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


class AutoClusterSelectionTest(unittest.TestCase):
    def test_auto_k_search_respects_bounds_above_twenty(self):
        rng = np.random.default_rng(123)
        centers = rng.normal(size=(22, 4)) * 5.0
        features = np.concatenate(
            [center + rng.normal(scale=0.05, size=(4, 4)) for center in centers],
            axis=0,
        ).astype(np.float32)
        _, diagnostics = evaluate_cluster_candidates(
            features,
            max_clusters=22,
            selection="silhouette",
            n_init=1,
            stability_repeats=1,
        )
        self.assertEqual(diagnostics[-1]["k"], 22)

    def test_auto_k_gmm_bic_returns_model_selection_diagnostics(self):
        rng = np.random.default_rng(321)
        features = np.concatenate(
            [rng.normal(loc=center, scale=0.2, size=(30, 3)) for center in (-3.0, 3.0)],
            axis=0,
        ).astype(np.float32)
        selected, diagnostics = evaluate_cluster_candidates(
            features,
            max_clusters=4,
            selection="gmm_bic",
            n_init=1,
            stability_repeats=1,
        )
        self.assertGreaterEqual(selected, 2)
        self.assertTrue(all("gmm_bic" in row for row in diagnostics))


class CalibrationTest(unittest.TestCase):
    def test_open_threshold_maximizes_unknown_rejection_at_known_coverage(self):
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.4, 0.5, 0.9])
        is_known = np.array([True, True, True, True, False, False, False, False])
        threshold, report = calibrate_open_threshold(
            scores, is_known, "known_coverage", target_known_coverage=0.75
        )
        self.assertAlmostEqual(report["known_accept_rate"], 0.75)
        self.assertAlmostEqual(report["unknown_reject_rate"], 0.25)
        self.assertAlmostEqual(report["target_known_coverage"], 0.75)
        self.assertAlmostEqual(threshold, 0.7)

    def test_open_threshold_balances_known_acceptance_and_unknown_rejection(self):
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        is_known = np.array([True, True, True, False, False, False])
        threshold, report = calibrate_open_threshold(scores, is_known, "balanced_accuracy")
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLess(threshold, 0.8)
        self.assertAlmostEqual(report["known_accept_rate"], 1.0)
        self.assertAlmostEqual(report["unknown_reject_rate"], 1.0)

    def test_known_coverage_threshold_uses_known_scores_only(self):
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        threshold, report = calibrate_coverage_threshold(scores, target_known_coverage=0.8)
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLess(threshold, 0.9)
        self.assertAlmostEqual(report["target_known_coverage"], 0.8)
        self.assertGreaterEqual(report["known_accept_rate"], 0.8)

    def test_known_coverage_threshold_rejects_invalid_target(self):
        with self.assertRaises(ValueError):
            calibrate_coverage_threshold(np.array([0.1, 0.2]), target_known_coverage=0.0)

    def test_class_conditional_thresholds_fallback_for_small_classes(self):
        thresholds = calibrate_class_thresholds(
            np.array([1.0, 2.0, 10.0, 12.0]),
            np.array([0, 0, 1, 1]),
            num_classes=3,
            percentile=50.0,
            min_samples=3,
        )
        self.assertTrue(np.allclose(thresholds, np.array([6.0, 6.0, 6.0])))

    def test_calibration_diagnostics_reports_ece(self):
        result = calibration_diagnostics(
            np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]),
            np.array([0, 1, 1]),
            num_bins=5,
        )

        self.assertGreaterEqual(result["ece"], 0.0)
        self.assertEqual(sum(item["count"] for item in result["bins"]), 3)

    def test_react_energy_score_requires_and_uses_react_output(self):
        outputs = {
            "entropy": np.array([0.1, 0.2]),
            "epistemic": np.zeros(2),
            "aleatoric": np.zeros(2),
            "probs": np.array([[0.9, 0.1], [0.6, 0.4]]),
            "features": np.zeros((2, 2), dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            compute_open_score(outputs, score_mode="react_energy")
        outputs["react_energy"] = np.array([1.2, 2.3])
        score, _ = compute_open_score(outputs, score_mode="react_energy")
        np.testing.assert_allclose(score, outputs["react_energy"])


class DiscoveryReportTest(unittest.TestCase):
    def test_labeled_knn_support_downranks_known_class_consensus(self):
        outputs = {
            "features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        }
        bank = {
            "features": np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32),
            "labels": np.array([0, 0, 1]),
        }
        attach_knn_distances(outputs, bank, k=2)
        self.assertGreater(outputs["knn_class_support"][0], outputs["knn_class_support"][1])
        score = candidate_purification_score(outputs, "knn_support")
        self.assertGreater(score[1], score[0])

    def test_open_validation_purification_metrics(self):
        outputs = {"is_known": np.array([True, False, False, True])}
        report = evaluate_candidate_purification(
            outputs,
            np.array([True, True, True, False]),
            np.array([False, True, True, False]),
        )
        self.assertAlmostEqual(report["before"]["known_contamination"], 1 / 3)
        self.assertAlmostEqual(report["after"]["unknown_precision"], 1.0)

    def test_candidate_purification_keeps_highest_risk_only(self):
        outputs = {
            "entropy": np.array([0.1, 0.5, 0.9, 0.2]),
            "epistemic": np.zeros(4),
            "head_uncertainty": np.zeros(4),
            "probs": np.full((4, 2), 0.5),
        }
        mask, report = purify_candidate_mask(
            outputs,
            np.array([False, True, True, True]),
            mode="entropy",
            keep_ratio=2 / 3,
        )
        self.assertEqual(mask.sum(), 2)
        self.assertTrue(mask[2])
        self.assertEqual(report["removed_count"], 1)

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
