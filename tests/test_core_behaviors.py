from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.losses import (
    discovery_unknown_loss,
    angular_margin_loss,
    proxy_anchor_loss,
    hard_proxy_margin_loss,
    reciprocal_point_loss,
    known_pseudo_label_consistency_loss,
    energy_margin_loss,
    per_sample_energy_margin_loss,
    proxy_contrastive_loss,
    supervised_contrastive_loss,
    weighted_energy_margin_loss,
    uncertainty_weights,
    unknown_feature_margin_loss,
    pu_unknown_feature_margin_loss,
    unknown_feature_separation_loss,
    unknown_feature_boundary_loss,
    knn_support_boundary_loss,
    objectosphere_loss,
    uncertainty_separation_loss,
    nnpu_known_uncertainty_loss,
    outlier_exposure_uniform_loss,
    prototype_repulsion_loss,
    supervised_center_loss,
    supervised_radius_loss,
    supervised_center_margin_loss,
)
from novel_discovery.data import (
    build_data_bundle,
    build_open_validation_and_discovery,
    dataset_source_indices,
    make_class_split,
    limit_dataset,
    split_known_indices,
    split_known_for_discovery,
    split_validation_for_calibration,
    OpenSetImageFolder,
    _known_labels,
    _known_flags,
    known_proportion,
    unknown_subset,
    validate_class_split,
)
from scripts.make_cifar100_splits import build_protocols
from scripts.analyze_feature_overlap import _class_centroids, _distribution_summary
from novel_discovery.pipeline import (
    calibration_diagnostics,
    attach_knn_distances,
    estimate_classwise_knn_radii,
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
    collect_classwise_knn_support,
    train_one_epoch_student,
    teacher_kd_uncertainty,
    weighted_discovery_view_loss,
    synthesize_virtual_outliers,
    synthesize_gaussian_virtual_outliers,
    fit_feature_rejector,
    fit_virtual_outlier_rejector,
    fit_known_support_rejector,
    attach_known_support_score,
    fit_known_conformal_support_rejector,
    attach_known_conformal_support_score,
    fit_pu_feature_rejector,
    fit_nnpu_feature_rejector,
    attach_feature_rejector_score,
    build_rejector_features,
    power_novelty_weights,
    cross_view_min_uncertainty,
)
from train import (
    initialize_novel_head_kmeans,
    load_checkpoint,
    parse_args,
    scheduled_weight,
    save_checkpoint,
    save_command_config,
)
from novel_discovery.models import build_model, freeze_batchnorm_stats


class CommandLineTest(unittest.TestCase):
    def test_rejection_branch_has_separate_outputs_and_gradients(self):
        model = build_model(
            3,
            backbone="resnet18",
            pretrained=False,
            rejection_feature_dim=16,
        )
        outputs = model(torch.randn(4, 3, 32, 32))
        self.assertEqual(tuple(outputs["logits"].shape), (4, 3))
        self.assertEqual(tuple(outputs["rejection_features"].shape), (4, 16))
        self.assertEqual(tuple(outputs["rejection_logits"].shape), (4, 3))
        loss = outputs["rejection_logits"].mean() + outputs["uncertainty"].mean()
        loss.backward()
        self.assertIsNotNone(model.rejection_projector[0].weight.grad)
        self.assertIsNotNone(model.encoder.features[0].weight.grad)
        self.assertIsNone(model.classifier.weight.grad)

    def test_rejection_embedding_mode_uses_independent_features(self):
        outputs = {
            "features": np.zeros((2, 4), dtype=np.float32),
            "rejection_features": np.asarray(
                [[3.0, 0.0], [0.0, 4.0]], dtype=np.float32
            ),
        }
        result = build_rejector_features(outputs, feature_mode="rejection_embedding")
        np.testing.assert_allclose(
            result, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )

    def test_rejection_outlier_uniform_option_is_exposed(self):
        args = parse_args(
            [
                "train_student",
                "--dataset",
                "toy",
                "--alpha-outlier-rejection-uniform",
                "0.05",
            ]
        )
        self.assertAlmostEqual(args.alpha_outlier_rejection_uniform, 0.05)

    def test_model_seed_is_independent_from_data_split_seed(self):
        defaults = parse_args(["train_student", "--seed", "42"])
        seeded = parse_args(
            ["train_student", "--seed", "42", "--model-seed", "43"]
        )
        self.assertEqual(defaults.seed, 42)
        self.assertIsNone(defaults.model_seed)
        self.assertEqual(seeded.seed, 42)
        self.assertEqual(seeded.model_seed, 43)

    def test_rejection_uncertainty_augmented_mode_keeps_rejection_base(self):
        outputs = {
            "features": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "rejection_features": np.asarray(
                [[3.0, 0.0], [0.0, 4.0]], dtype=np.float32
            ),
            "logits": np.zeros((2, 2), dtype=np.float32),
            "probs": np.full((2, 2), 0.5, dtype=np.float32),
            "entropy": np.zeros(2, dtype=np.float32),
            "head_uncertainty": np.zeros(2, dtype=np.float32),
            "epistemic": np.zeros(2, dtype=np.float32),
            "expected_entropy": np.zeros(2, dtype=np.float32),
            "aleatoric": np.zeros(2, dtype=np.float32),
        }
        result = build_rejector_features(
            outputs, feature_mode="rejection_uncertainty_augmented"
        )
        self.assertEqual(result.shape, (2, 2 + 2 + 4 + 3))
        np.testing.assert_allclose(result[:, :2], np.eye(2, dtype=np.float32))

    def test_rejection_logit_augmented_mode_uses_rejection_logits(self):
        outputs = {
            "features": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "rejection_features": np.asarray(
                [[3.0, 0.0], [0.0, 4.0]], dtype=np.float32
            ),
            "logits": np.zeros((2, 2), dtype=np.float32),
            "probs": np.full((2, 2), 0.5, dtype=np.float32),
            "entropy": np.zeros(2, dtype=np.float32),
            "head_uncertainty": np.zeros(2, dtype=np.float32),
            "rejection_logits": np.asarray(
                [[2.0, 0.0], [0.0, 2.0]], dtype=np.float32
            ),
        }
        result = build_rejector_features(
            outputs, feature_mode="rejection_logit_augmented"
        )
        # rejection base (2) + classifier summary (2+1+1+1+1) +
        # rejection logits and their entropy/max-probability/margin (2+3).
        self.assertEqual(result.shape, (2, 2 + 6 + 5))
        np.testing.assert_allclose(result[:, :2], np.eye(2, dtype=np.float32))
        np.testing.assert_allclose(result[:, 2 + 6 : 2 + 6 + 2], outputs["rejection_logits"])
        self.assertAlmostEqual(float(result[0, -1]), float(result[1, -1]), places=6)

    def test_rejection_support_logit_mode_requires_support_score(self):
        outputs = {
            "features": np.zeros((1, 2), dtype=np.float32),
            "rejection_features": np.ones((1, 2), dtype=np.float32),
            "logits": np.zeros((1, 2), dtype=np.float32),
            "probs": np.full((1, 2), 0.5, dtype=np.float32),
            "entropy": np.zeros(1, dtype=np.float32),
            "head_uncertainty": np.zeros(1, dtype=np.float32),
            "rejection_logits": np.zeros((1, 2), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "known_support_score"):
            build_rejector_features(
                outputs, feature_mode="rejection_support_logit_augmented"
            )

    def test_rejection_support_modes_share_the_same_base_and_support_feature(self):
        outputs = {
            "features": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "rejection_features": np.asarray([[0.0, 2.0]], dtype=np.float32),
            "logits": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "probs": np.asarray([[0.75, 0.25]], dtype=np.float32),
            "entropy": np.asarray([0.5], dtype=np.float32),
            "head_uncertainty": np.asarray([0.1], dtype=np.float32),
            "rejection_logits": np.asarray([[2.0, 0.0]], dtype=np.float32),
            "known_support_score": np.asarray([0.3], dtype=np.float32),
        }
        base = build_rejector_features(
            outputs, feature_mode="rejection_support_augmented"
        )
        with_logits = build_rejector_features(
            outputs, feature_mode="rejection_support_logit_augmented"
        )
        np.testing.assert_allclose(base[:, :8], with_logits[:, :8])
        self.assertEqual(base.shape[1], 2 + 2 + 4 + 1)
        self.assertEqual(with_logits.shape[1], base.shape[1] + 2 + 3)
        self.assertAlmostEqual(float(base[0, -1]), 0.3)
        self.assertAlmostEqual(float(with_logits[0, -1]), 0.3)

    def test_rejection_logit_mode_requires_extracted_logits(self):
        outputs = {
            "features": np.zeros((1, 2), dtype=np.float32),
            "rejection_features": np.ones((1, 2), dtype=np.float32),
            "logits": np.zeros((1, 2), dtype=np.float32),
            "probs": np.full((1, 2), 0.5, dtype=np.float32),
            "entropy": np.zeros(1, dtype=np.float32),
            "head_uncertainty": np.zeros(1, dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "rejection_logits"):
            build_rejector_features(outputs, feature_mode="rejection_logit_augmented")

    def test_parser_accepts_rejection_logit_modes(self):
        for mode in (
            "rejection_support_augmented",
            "rejection_logit_augmented",
            "rejection_support_logit_augmented",
        ):
            args = parse_args(
                ["discover", "--dataset", "toy", "--rejector-feature-mode", mode]
            )
            self.assertEqual(args.rejector_feature_mode, mode)

    def test_prototype_augmented_rejector_features_append_classwise_similarity(self):
        outputs = {
            "features": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "logits": np.zeros((2, 2), dtype=np.float32),
            "probs": np.full((2, 2), 0.5, dtype=np.float32),
            "entropy": np.zeros(2, dtype=np.float32),
            "head_uncertainty": np.zeros(2, dtype=np.float32),
            "prototype_similarity": np.asarray(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
        }
        result = build_rejector_features(outputs, feature_mode="prototype_augmented")
        self.assertEqual(result.shape, (2, 2 + 2 + 4 + 2))
        np.testing.assert_allclose(result[:, -2:], outputs["prototype_similarity"])

    def test_checkpoint_rejects_mismatched_class_split(self):
        with tempfile.TemporaryDirectory() as root:
            model = build_model(2, backbone="resnet18", pretrained=False)
            path = Path(root, "model.pt")
            save_checkpoint(
                model,
                path,
                extra={"known_classes": [4, 9], "novel_classes": [1, 2, 3]},
            )
            with self.assertRaisesRegex(ValueError, "known_classes"):
                load_checkpoint(
                    build_model(2, backbone="resnet18", pretrained=False),
                    path,
                    torch.device("cpu"),
                    expected_known_classes=[0, 1],
                    expected_novel_classes=[2, 3, 4],
                )

    def test_command_configs_survive_legacy_config_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            save_command_config(root, "train_student", {"command": "train_student", "epochs": 3})
            save_command_config(root, "discover", {"command": "discover", "score_mode": "energy"})
            train_config = Path(root, "train_student_config.json")
            discover_config = Path(root, "discover_config.json")
            legacy_config = Path(root, "config.json")
            self.assertEqual(json.loads(train_config.read_text(encoding="utf-8"))["epochs"], 3)
            self.assertEqual(json.loads(discover_config.read_text(encoding="utf-8"))["score_mode"], "energy")
            self.assertEqual(json.loads(legacy_config.read_text(encoding="utf-8"))["command"], "discover")

    def test_teacher_is_valid_discovery_selection_model(self):
        args = parse_args(
            [
                "train_student",
                "--dataset",
                "toy",
                "--discovery-selection-model",
                "teacher",
            ]
        )
        self.assertEqual(args.discovery_selection_model, "teacher")

    def test_teacher_can_supply_frozen_feature_margin_weights(self):
        args = parse_args(
            [
                "train_student",
                "--dataset",
                "toy",
                "--discovery-selection-model",
                "teacher",
                "--discovery-feature-margin-weight-source",
                "ema_product",
            ]
        )
        self.assertEqual(args.discovery_selection_model, "teacher")
        self.assertEqual(args.discovery_feature_margin_weight_source, "ema_product")

    def test_joint_prototype_refresh_is_opt_in(self):
        args = parse_args(["train_student", "--dataset", "toy"])
        self.assertEqual(args.joint_prototype_refresh_epochs, 0)
        args = parse_args(
            [
                "train_student",
                "--dataset",
                "toy",
                "--joint-prototype-refresh-epochs",
                "2",
            ]
        )
        self.assertEqual(args.joint_prototype_refresh_epochs, 2)

    def test_feature_overlap_diagnostics_separate_easy_known_and_unknown(self):
        known = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
        unknown = np.asarray([[-1.0, 0.0], [-0.9, -0.1]], dtype=np.float32)
        result = _distribution_summary(1.0 - known[:, 0], 1.0 - unknown[:, 0])
        self.assertGreater(result["unknownness_auroc"], 0.99)
        self.assertLess(result["histogram_overlap_0_to_1"], 0.5)

    def test_discovery_detail_keeps_feature_norm_when_available(self):
        detail = {"score": np.asarray([1.0, 2.0]), "feature_norm": np.asarray([3.0, 4.0])}
        self.assertEqual(detail["feature_norm"].tolist(), [3.0, 4.0])

    def test_empirical_centroid_builder_skips_unobserved_classes(self):
        features = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
        labels = np.asarray([0, 0, 2])
        centers, classes = _class_centroids(features, labels, num_classes=4)
        self.assertEqual(classes, [0, 2])
        self.assertEqual(centers.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-6)

    def test_unknown_feature_boundary_loss_is_finite_and_differentiable(self):
        known = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], requires_grad=False)
        labels = torch.tensor([0, 0, 1])
        unknown = torch.tensor([[0.8, 0.2], [-0.2, 0.9]], requires_grad=True)
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)
        loss = unknown_feature_boundary_loss(
            known, labels, unknown, prototypes, similarity_margin=0.2
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(unknown.grad)
        self.assertTrue(torch.isfinite(unknown.grad).all())

    def test_soft_unknown_feature_separation_accepts_sample_weights(self):
        known = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        unknown = torch.tensor([[0.8, 0.2], [-0.2, 0.9]], requires_grad=True)
        weights = torch.tensor([1.0, 0.5])
        loss = unknown_feature_separation_loss(
            known,
            unknown,
            similarity_margin=0.0,
            temperature=0.1,
            sample_weight=weights,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(unknown.grad)
        self.assertTrue(torch.isfinite(unknown.grad).all())

    def test_knn_support_boundary_loss_pushes_samples_outside_known_support(self):
        bank = [
            torch.tensor([[1.0, 0.0], [0.99, 0.1]]),
            torch.tensor([[0.0, 1.0], [0.1, 0.99]]),
            torch.empty(0, 0),
        ]
        radii = torch.tensor([0.1, 0.1, 0.0])
        near = torch.tensor([[1.0, 0.0]], requires_grad=True)
        far = torch.tensor([[-1.0, -1.0]], requires_grad=True)
        near_loss = knn_support_boundary_loss(near, bank, radii, k=2, margin=0.05)
        far_loss = knn_support_boundary_loss(far, bank, radii, k=2, margin=0.05)
        self.assertGreater(float(near_loss), float(far_loss))
        self.assertEqual(float(far_loss), 0.0)
        near_loss.backward()
        self.assertIsNotNone(near.grad)
        self.assertTrue(torch.isfinite(near.grad).all())

    def test_knn_support_boundary_loss_accepts_continuous_sample_weights(self):
        bank = [torch.tensor([[1.0, 0.0], [0.98, 0.02]])]
        radii = torch.tensor([0.2])
        features = torch.tensor([[1.0, 0.0], [0.98, 0.02]], requires_grad=True)
        weights = torch.tensor([1.0, 0.0])
        loss = knn_support_boundary_loss(
            features,
            bank,
            radii,
            k=2,
            margin=0.05,
            sample_weights=weights,
        )
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(features.grad)

    def test_classwise_knn_support_bank_uses_known_labels_and_restores_mode(self):
        class IdentityFeatureModel(torch.nn.Module):
            def forward(self, images):
                return {"features": images}

        images = torch.tensor(
            [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99], [-1.0, 0.0]]
        )
        labels = torch.tensor([0, 0, 1, 1, -1])
        raw = labels.clone()
        is_known = (labels >= 0).long()
        indices = torch.arange(len(labels))
        dataset = torch.utils.data.TensorDataset(images, labels, raw, is_known, indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=3)
        model = IdentityFeatureModel().train()
        bank = collect_classwise_knn_support(
            model, loader, torch.device("cpu"), num_classes=3, k=1, quantile=0.95
        )
        self.assertTrue(model.training)
        self.assertEqual(bank["counts"].tolist(), [2, 2, 0])
        self.assertEqual(len(bank["features_by_class"]), 3)
        self.assertEqual(float(bank["radii"][2]), 0.0)

    def test_freeze_batchnorm_stats_keeps_affine_parameters_trainable(self):
        model = build_model(3, backbone="resnet18", pretrained=False)
        model.train()
        freeze_batchnorm_stats(model)
        batchnorms = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        self.assertTrue(batchnorms)
        self.assertTrue(all(not module.training for module in batchnorms))
        self.assertTrue(all(module.weight.requires_grad and module.bias.requires_grad for module in batchnorms))

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

    def test_nnpu_feature_rejector_supports_corrected_nu_risk(self):
        known = {"features": np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1]])}
        unlabeled = {
            "features": np.asarray(
                [[-1.0, 0.0], [-0.9, -0.1], [0.8, 0.2], [0.7, 0.3]]
            )
        }
        corrected = fit_nnpu_feature_rejector(
            known,
            unlabeled,
            known_prior=0.5,
            iterations=30,
            seed=0,
            risk_mode="nu_corrected",
        )
        score = corrected.decision_function(np.asarray([[-1.0, 0.0], [1.0, 0.0]]))
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[0], score[1])

    def test_nnpu_score_margin_regularizer_is_optional_and_finite(self):
        known = {"features": np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1]])}
        unlabeled = {
            "features": np.asarray(
                [[-1.0, 0.0], [-0.9, -0.1], [0.8, 0.2], [0.7, 0.3]]
            )
        }
        rejector = fit_nnpu_feature_rejector(
            known,
            unlabeled,
            known_prior=0.5,
            iterations=20,
            seed=0,
            risk_mode="nu_corrected",
            score_margin_weight=0.1,
            score_margin=0.2,
        )
        scores = rejector.decision_function(np.asarray([[1.0, 0.0], [-1.0, 0.0]]))
        self.assertTrue(np.isfinite(scores).all())

    def test_nnpu_feature_rejector_supports_stratified_known_sampling(self):
        known = {
            "features": np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]] * 3,
                dtype=np.float32,
            ),
            "labels": np.asarray([0, 0, 1, 1] * 3),
        }
        unlabeled = {
            "features": np.asarray(
                [[-1.0, 0.0], [-0.9, -0.1], [0.0, -1.0], [0.1, -0.9]],
                dtype=np.float32,
            )
        }
        rejector = fit_nnpu_feature_rejector(
            known,
            unlabeled,
            known_prior=0.5,
            max_samples=4,
            iterations=20,
            seed=0,
            risk_mode="nu_corrected",
            stratified_known=True,
        )
        scores = rejector.decision_function(np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32))
        self.assertTrue(np.isfinite(scores).all())

    def test_nnpu_feature_rejector_rejects_unknown_risk_mode(self):
        known = {"features": np.asarray([[1.0, 0.0], [0.9, 0.1]])}
        unlabeled = {"features": np.asarray([[-1.0, 0.0], [-0.9, -0.1]])}
        with self.assertRaisesRegex(ValueError, "risk_mode"):
            fit_nnpu_feature_rejector(known, unlabeled, risk_mode="invalid")

    def test_nnpu_mlp_rejector_returns_finite_unknown_scores(self):
        known = {"features": np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1]])}
        unlabeled = {
            "features": np.asarray(
                [[-1.0, 0.0], [-0.9, -0.1], [0.8, 0.2], [0.7, 0.3]]
            )
        }
        rejector = fit_nnpu_feature_rejector(
            known,
            unlabeled,
            known_prior=0.5,
            iterations=30,
            seed=0,
            risk_mode="nu_corrected",
            model_type="mlp",
        )
        scores = rejector.decision_function(
            np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        )
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(rejector.predict_proba(np.asarray([[0.0, 0.0]])).shape, (1, 2))

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

    def test_known_conformal_support_scores_far_features_higher(self):
        known = {
            "features": np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
            ),
            "labels": np.asarray([0, 0, 1, 1]),
        }
        calibration = {
            "features": np.asarray(
                [[0.98, 0.02], [0.88, 0.12], [0.02, 0.98], [0.12, 0.88]],
                dtype=np.float32,
            )
        }
        model = fit_known_conformal_support_rejector(known, calibration, quantile=0.95)
        outputs = {"features": np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)}
        attach_known_conformal_support_score(outputs, model)
        self.assertTrue(np.isfinite(outputs["known_conformal_score"]).all())
        self.assertGreater(outputs["known_conformal_score"][1], outputs["known_conformal_score"][0])

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

    def test_calibration_split_is_disjoint_complete_and_stratified(self):
        labels = torch.tensor([label for label in range(4) for _ in range(20)])
        dataset = torch.utils.data.TensorDataset(torch.arange(80).float().unsqueeze(1), labels)
        selection, calibration = split_validation_for_calibration(
            dataset, calibration_ratio=0.4, seed=19, mode="stratified"
        )
        selection_indices = set(selection.indices)
        calibration_indices = set(calibration.indices)
        self.assertFalse(selection_indices & calibration_indices)
        self.assertEqual(selection_indices | calibration_indices, set(range(len(dataset))))
        self.assertEqual(len(calibration), 32)
        self.assertEqual(
            {int(dataset[index][1]) for index in selection_indices}, set(range(4))
        )
        self.assertEqual(
            {int(dataset[index][1]) for index in calibration_indices}, set(range(4))
        )

    def test_dataset_source_indices_resolves_nested_subsets_and_concat(self):
        base = torch.utils.data.TensorDataset(torch.arange(12).float().unsqueeze(1))
        nested = torch.utils.data.Subset(
            torch.utils.data.Subset(base, [3, 7, 9, 11]), [1, 3]
        )
        self.assertEqual(dataset_source_indices(nested), [7, 11])
        combined = torch.utils.data.ConcatDataset(
            [nested, torch.utils.data.Subset(base, [0, 2])]
        )
        self.assertEqual(dataset_source_indices(combined), [7, 11, 2, 4])

    def test_calibration_overlap_control_matches_size_and_selection(self):
        labels = torch.tensor([label for label in range(4) for _ in range(20)])
        dataset = torch.utils.data.TensorDataset(torch.arange(80).float().unsqueeze(1), labels)
        disjoint_selection, disjoint_calibration = split_validation_for_calibration(
            dataset, calibration_ratio=0.4, seed=19, mode="stratified"
        )
        overlap_selection, overlap_calibration = split_validation_for_calibration(
            dataset,
            calibration_ratio=0.4,
            seed=19,
            mode="stratified",
            overlap_control=True,
        )
        self.assertEqual(disjoint_selection.indices, overlap_selection.indices)
        self.assertEqual(len(disjoint_calibration), len(overlap_calibration))
        self.assertEqual(len(overlap_calibration), 32)
        self.assertTrue(set(overlap_calibration.indices) <= set(overlap_selection.indices))
        self.assertFalse(
            set(disjoint_calibration.indices) & set(disjoint_selection.indices)
        )
        with self.assertRaisesRegex(ValueError, "calibration_ratio <= 0.5"):
            split_validation_for_calibration(
                dataset,
                calibration_ratio=0.6,
                seed=19,
                mode="stratified",
                overlap_control=True,
            )

    def test_zero_calibration_ratio_preserves_legacy_dataset(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(8, 1), torch.arange(8) % 2)
        selection, calibration = split_validation_for_calibration(
            dataset, calibration_ratio=0.0, seed=3
        )
        self.assertIs(selection, dataset)
        self.assertIsNone(calibration)

    def test_calibration_ratio_is_available_for_training_and_discovery(self):
        train_args = parse_args(["train_student", "--calibration-ratio", "0.5"])
        discover_args = parse_args(
            ["discover", "--calibration-ratio", "0.5", "--calibration-overlap-control"]
        )
        self.assertEqual(train_args.calibration_ratio, 0.5)
        self.assertEqual(discover_args.calibration_ratio, 0.5)
        self.assertTrue(discover_args.calibration_overlap_control)

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

    def test_imagefolder_maps_known_classes_by_name_and_tracks_mixed_prior(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            for class_name, color in (("class_a", (255, 0, 0)), ("class_b", (0, 255, 0))):
                class_dir = root_path / class_name
                class_dir.mkdir()
                Image.new("RGB", (8, 8), color).save(class_dir / "sample.png")

            # ImageFolder's raw integer order is alphabetical (a=0, b=1),
            # while the model's known-label order here deliberately contains
            # only class_b. The mapping must use class names, not raw ids.
            open_dataset = OpenSetImageFolder(root, ["class_b"], include_unknown=True)
            unknown_item = open_dataset[0]
            known_item = open_dataset[1]
            self.assertEqual((unknown_item[1], unknown_item[2], unknown_item[3]), (-1, 0, 0))
            self.assertEqual((known_item[1], known_item[2], known_item[3]), (0, 1, 1))
            self.assertEqual(_known_flags(open_dataset), [False, True])
            self.assertAlmostEqual(known_proportion(open_dataset), 0.5)
            self.assertEqual(len(unknown_subset(open_dataset)), 1)

            known_only = OpenSetImageFolder(root, ["class_b"], include_unknown=False)
            self.assertEqual(len(known_only), 1)
            self.assertEqual(_known_labels(known_only), [0])

    def test_novelty_weight_power_sharpens_without_reordering(self):
        weights = torch.tensor([0.0, 0.25, 0.5, 1.0])
        torch.testing.assert_close(power_novelty_weights(weights), weights)
        sharpened = power_novelty_weights(weights, power=4.0)
        torch.testing.assert_close(sharpened, torch.tensor([0.0, 0.00390625, 0.0625, 1.0]))
        self.assertTrue(torch.all(sharpened[1:] <= weights[1:]))
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            power_novelty_weights(weights, power=0.0)

    def test_cross_view_min_uncertainty_requires_both_views_to_agree(self):
        first = torch.tensor([0.9, 0.2, 0.7])
        second = torch.tensor([0.3, 0.8, 0.6])
        weights = cross_view_min_uncertainty(first, second)
        torch.testing.assert_close(weights, torch.tensor([0.3, 0.2, 0.6]))
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            cross_view_min_uncertainty(first, second[:2])

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

    def test_matched_pool_protocol_holds_supervision_and_pool_size_constant(self):
        def make_dataset(start, count, known):
            ids = torch.arange(start, start + count)
            labels = torch.zeros(count, dtype=torch.long) if known else torch.full((count,), -1)
            flags = torch.ones(count, dtype=torch.long) if known else torch.zeros(count, dtype=torch.long)
            return torch.utils.data.TensorDataset(
                torch.zeros(count, 3, 2, 2), labels, ids, flags, ids
            )

        known = make_dataset(0, 40, known=True)
        unknown = make_dataset(100, 80, known=False)
        supervised, reserved_known = split_known_for_discovery(
            known, 0.25, seed=11
        )
        pure = build_open_validation_and_discovery(
            supervised,
            make_dataset(300, 10, known=True),
            unknown,
            open_val_ratio=0.0,
            seed=23,
            discovery_pool_mode="unknown",
            known_discovery_pool=reserved_known,
            matched_discovery_pool=True,
            matched_pool_size=20,
            matched_known_prior=0.2,
        )[2]
        mixed = build_open_validation_and_discovery(
            supervised,
            make_dataset(300, 10, known=True),
            unknown,
            open_val_ratio=0.0,
            seed=23,
            discovery_pool_mode="mixed",
            known_discovery_pool=reserved_known,
            matched_discovery_pool=True,
            matched_pool_size=20,
            matched_known_prior=0.2,
        )[2]

        ids = lambda dataset: {int(dataset[index][4]) for index in range(len(dataset))}
        train_ids = ids(supervised)
        pure_unknown_ids = ids(pure)
        mixed_ids = ids(mixed)
        mixed_known_ids = {
            int(mixed[index][4]) for index in range(len(mixed)) if int(mixed[index][3]) == 1
        }
        mixed_unknown_ids = mixed_ids - mixed_known_ids

        self.assertEqual(len(pure), 20)
        self.assertEqual(len(mixed), 20)
        self.assertEqual(len(supervised), 30)
        self.assertEqual(known_proportion(pure), 0.0)
        self.assertAlmostEqual(known_proportion(mixed), 0.2)
        self.assertTrue(train_ids.isdisjoint(mixed_ids))
        self.assertTrue(mixed_unknown_ids.issubset(pure_unknown_ids))

    def test_matched_pool_requires_enough_reserved_known_and_unknown_samples(self):
        def make_dataset(start, count, known):
            ids = torch.arange(start, start + count)
            labels = torch.zeros(count, dtype=torch.long) if known else torch.full((count,), -1)
            flags = torch.ones(count, dtype=torch.long) if known else torch.zeros(count, dtype=torch.long)
            return torch.utils.data.TensorDataset(
                torch.zeros(count, 3, 2, 2), labels, ids, flags, ids
            )

        with self.assertRaisesRegex(ValueError, "Not enough reserved known"):
            build_open_validation_and_discovery(
                make_dataset(0, 8, known=True),
                make_dataset(20, 2, known=True),
                make_dataset(40, 10, known=False),
                open_val_ratio=0.0,
                seed=1,
                discovery_pool_mode="mixed",
                matched_discovery_pool=True,
                matched_pool_size=10,
                matched_known_prior=0.5,
            )

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
                "--discovery-feature-margin-weight-power", "4.0",
                "--discovery-feature-margin-weight-source", "cross_view_min_uncertainty",
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
        self.assertAlmostEqual(args.discovery_feature_margin_weight_power, 4.0)
        self.assertEqual(
            args.discovery_feature_margin_weight_source,
            "cross_view_min_uncertainty",
        )
        self.assertEqual(
            args.discovery_uncertainty_feature_margin_mode,
            "soft_weighted",
        )
        self.assertTrue(args.rejector_strict_mixed)
        self.assertAlmostEqual(args.alpha_discovery_selective_energy, 0.1)
        self.assertAlmostEqual(args.alpha_discovery_uniform, 0.07)
        self.assertEqual(args.discovery_uniform_warmup_epochs, 1)
        self.assertEqual(args.discovery_uniform_ramp_epochs, 2)
        self.assertAlmostEqual(args.discovery_select_ratio, 0.25)

    def test_matched_discovery_pool_protocol_arguments_are_available(self):
        args = parse_args(
            [
                "train_student",
                "--matched-discovery-pool",
                "--matched-pool-size", "5400",
                "--matched-known-prior", "0.2",
                "--mixed-known-pool-ratio", "0.05",
            ]
        )
        self.assertTrue(args.matched_discovery_pool)
        self.assertEqual(args.matched_pool_size, 5400)
        self.assertAlmostEqual(args.matched_known_prior, 0.2)
        self.assertAlmostEqual(args.mixed_known_pool_ratio, 0.05)

    def test_knn_boundary_options_are_exposed_with_safe_defaults(self):
        defaults = parse_args(["train_student"])
        self.assertEqual(defaults.alpha_discovery_knn_boundary, 0.0)
        self.assertEqual(defaults.discovery_knn_k, 5)
        args = parse_args(
            [
                "train_student",
                "--alpha-discovery-knn-boundary", "0.1",
                "--discovery-knn-k", "3",
                "--discovery-knn-quantile", "0.9",
                "--discovery-knn-margin", "0.03",
            ]
        )
        self.assertAlmostEqual(args.alpha_discovery_knn_boundary, 0.1)
        self.assertEqual(args.discovery_knn_k, 3)
        self.assertAlmostEqual(args.discovery_knn_quantile, 0.9)
        self.assertAlmostEqual(args.discovery_knn_margin, 0.03)

    def test_knn_boundary_can_use_candidate_gate_on_mixed_pool(self):
        defaults = parse_args(["train_student"])
        self.assertFalse(defaults.discovery_feature_candidate_gating)
        self.assertFalse(defaults.discovery_cross_view_gating)
        self.assertEqual(defaults.discovery_knn_warmup_epochs, 0)
        self.assertEqual(defaults.discovery_knn_ramp_epochs, 0)
        self.assertFalse(defaults.discovery_knn_soft_weighting)
        args = parse_args(
            [
                "train_student",
                "--discovery-pool-mode", "mixed",
                "--discovery-feature-candidate-gating",
                "--discovery-cross-view-gating",
                "--alpha-discovery-knn-boundary", "0.1",
                "--discovery-knn-warmup-epochs", "2",
                "--discovery-knn-ramp-epochs", "2",
            ]
        )
        self.assertEqual(args.discovery_pool_mode, "mixed")
        self.assertTrue(args.discovery_feature_candidate_gating)
        self.assertTrue(args.discovery_cross_view_gating)
        self.assertAlmostEqual(args.alpha_discovery_knn_boundary, 0.1)
        self.assertEqual(args.discovery_knn_warmup_epochs, 2)
        self.assertEqual(args.discovery_knn_ramp_epochs, 2)

        soft_args = parse_args(
            [
                "train_student",
                "--dataset",
                "toy",
                "--discovery-knn-soft-weighting",
            ]
        )
        self.assertTrue(soft_args.discovery_knn_soft_weighting)

    def test_selective_discovery_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)
        self.assertEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=0), 1.0)

    def test_uncertainty_separation_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)

    def test_nnpu_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)

    def test_discovery_uniform_weight_schedule(self):
        self.assertEqual(scheduled_weight(0, warmup_epochs=1, ramp_epochs=2), 0.0)
        self.assertAlmostEqual(scheduled_weight(1, warmup_epochs=1, ramp_epochs=2), 0.5)
        self.assertEqual(scheduled_weight(2, warmup_epochs=1, ramp_epochs=2), 1.0)


class LossBehaviorTest(unittest.TestCase):
    def test_discovery_view_loss_requires_its_explicit_weight(self):
        first = torch.randn(4, 8, requires_grad=True)
        second = torch.randn(4, 8, requires_grad=True)
        disabled = weighted_discovery_view_loss(first, second, alpha=0.0)
        self.assertEqual(float(disabled), 0.0)
        enabled = weighted_discovery_view_loss(first, second, alpha=0.1)
        self.assertGreater(float(enabled), 0.0)
        enabled.backward()
        self.assertIsNotNone(first.grad)

    def test_angular_margin_loss_is_finite_and_has_gradient(self):
        features = torch.randn(6, 4, requires_grad=True)
        weights = torch.randn(3, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = angular_margin_loss(features, labels, weights, margin=0.2, scale=16.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(weights.grad)

    def test_angular_margin_loss_supports_a_correct_sample_mask(self):
        features = torch.randn(6, 4, requires_grad=True)
        weights = torch.randn(3, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        mask = torch.tensor([True, False, True, False, True, False])
        loss = angular_margin_loss(
            features,
            labels,
            weights,
            margin=0.2,
            scale=16.0,
            sample_mask=mask,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(weights.grad)


class UncertaintyKDTargetTest(unittest.TestCase):
    def test_mc_epistemic_target_is_normalized_and_uses_mean_logits(self):
        class DummyTeacher(torch.nn.Module):
            def forward(self, images):
                logits = torch.zeros(images.size(0), 4)
                return {
                    "logits": logits,
                    "uncertainty": torch.full((images.size(0),), 0.25),
                    "features": torch.zeros(images.size(0), 3),
                    "proj": torch.zeros(images.size(0), 2),
                }

            def mc_predict(self, images, mc_samples=4):
                self.last_samples = mc_samples
                return {
                    "mean_logits": torch.full((images.size(0), 4), 0.5),
                    "epistemic": torch.full((images.size(0),), 0.7),
                    "predictive_entropy": torch.full((images.size(0),), 1.2),
                }

        teacher = DummyTeacher()
        outputs, logits, uncertainty = teacher_kd_uncertainty(
            teacher, torch.randn(3, 2), source="mc_epistemic", mc_samples=5
        )
        self.assertEqual(outputs["logits"].shape, (3, 4))
        self.assertTrue(torch.allclose(logits, torch.full((3, 4), 0.5)))
        self.assertTrue(torch.all((uncertainty >= 0.0) & (uncertainty <= 1.0)))
        self.assertEqual(teacher.last_samples, 5)

    def test_mc_target_rejects_single_sample(self):
        class DummyTeacher(torch.nn.Module):
            def forward(self, images):
                return {
                    "logits": torch.zeros(images.size(0), 2),
                    "uncertainty": torch.zeros(images.size(0)),
                }

        with self.assertRaises(ValueError):
            teacher_kd_uncertainty(
                DummyTeacher(), torch.randn(2, 1), source="mc_epistemic", mc_samples=1
            )

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

    def test_hard_proxy_margin_loss_is_finite_and_has_gradient(self):
        features = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        proxies = torch.randn(4, 16, requires_grad=True)
        loss = hard_proxy_margin_loss(features, labels, proxies, margin=0.2)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(proxies.grad)
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

    def test_pu_unknown_feature_margin_corrects_known_contamination(self):
        known = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
        prototypes = torch.tensor([[1.0, 0.0]])
        loss = pu_unknown_feature_margin_loss(
            known,
            mixed,
            prototypes,
            known_prior=0.5,
            similarity_margin=0.2,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(mixed.grad)
        self.assertIsNone(known.grad)

        active_mixed = torch.tensor([[0.8, 0.6]], requires_grad=True)
        active_loss = pu_unknown_feature_margin_loss(
            known.detach(),
            active_mixed,
            prototypes,
            known_prior=0.2,
            similarity_margin=0.2,
        )
        self.assertGreater(active_loss.item(), 0.0)
        active_loss.backward()
        self.assertGreater(active_mixed.grad.norm().item(), 0.0)

    def test_objectosphere_loss_pushes_unknown_norm_down_and_has_gradients(self):
        known = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        unknown = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        loss = objectosphere_loss(known, unknown, known_radius=2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(unknown.grad)
        self.assertGreater(unknown.grad.norm().item(), 0.0)

    def test_uncertainty_separation_loss_has_known_unknown_direction(self):
        known = torch.tensor([0.1, 0.2], requires_grad=True)
        unknown = torch.tensor([0.8, 0.9], requires_grad=True)
        loss = uncertainty_separation_loss(known, unknown)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(known.grad.mean().item(), 0.0)
        self.assertLess(unknown.grad.mean().item(), 0.0)

    def test_nnpu_known_uncertainty_loss_is_finite_nonnegative_and_differentiable(self):
        known_uncertainty = torch.tensor([0.1, 0.25, 0.2], requires_grad=True)
        unlabeled_uncertainty = torch.tensor([0.35, 0.75, 0.6], requires_grad=True)
        loss = nnpu_known_uncertainty_loss(
            known_uncertainty, unlabeled_uncertainty, known_prior=0.2
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(known_uncertainty.grad)
        self.assertIsNotNone(unlabeled_uncertainty.grad)
        self.assertTrue(torch.isfinite(known_uncertainty.grad).all())
        self.assertTrue(torch.isfinite(unlabeled_uncertainty.grad).all())

    def test_nnpu_known_uncertainty_loss_rejects_invalid_prior(self):
        with self.assertRaises(ValueError):
            nnpu_known_uncertainty_loss(torch.tensor([0.2]), torch.tensor([0.4]), 1.0)

    def test_nnpu_corrected_risk_matches_mixed_pool_decomposition(self):
        known_uncertainty = torch.tensor([0.2, 0.4], requires_grad=True)
        unlabeled_uncertainty = torch.tensor([0.3, 0.7], requires_grad=True)
        prior = 0.25
        known_u = known_uncertainty.clamp(1e-6, 1.0 - 1e-6)
        unlabeled_u = unlabeled_uncertainty.clamp(1e-6, 1.0 - 1e-6)
        known_logit = torch.log1p(-known_u) - torch.log(known_u)
        unlabeled_logit = torch.log1p(-unlabeled_u) - torch.log(unlabeled_u)
        expected = (
            prior * F.softplus(-known_logit).mean()
            + F.softplus(-unlabeled_logit).mean()
            - prior * F.softplus(known_logit).mean()
        )
        actual = nnpu_known_uncertainty_loss(
            known_uncertainty,
            unlabeled_uncertainty,
            known_prior=prior,
            risk_mode="nu_corrected",
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_nnpu_negative_risk_uses_gradient_correction_not_clamp(self):
        known_uncertainty = torch.tensor([0.001, 0.002], requires_grad=True)
        unlabeled_uncertainty = torch.tensor([0.998, 0.999], requires_grad=True)
        loss = nnpu_known_uncertainty_loss(
            known_uncertainty, unlabeled_uncertainty, known_prior=0.5
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        # Gradient ascent on estimated negative risk moves the negative-risk
        # estimate back toward zero; for novelty probability u this increases
        # d(loss)/du in this deliberately negative-risk example.
        self.assertGreater(unlabeled_uncertainty.grad.mean().item(), 0.0)

    def test_known_proportion_tracks_subset_and_mixed_pool_composition(self):
        class FlagDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.flags = [1, 1, 0, 0]

            def __len__(self):
                return len(self.flags)

            def __getitem__(self, index):
                return torch.zeros(1), 0 if self.flags[index] else -1, 0, self.flags[index], index

        dataset = FlagDataset()
        self.assertAlmostEqual(known_proportion(dataset), 0.5)
        subset = torch.utils.data.Subset(dataset, [0, 1, 2])
        self.assertAlmostEqual(known_proportion(subset), 2.0 / 3.0)


class DiscoveryTrainingDispatchTest(unittest.TestCase):
    def test_uncertainty_separation_alone_consumes_discovery_batches(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3, 2)
                self.uncertainty_head = torch.nn.Linear(3, 1)
                self.projector = torch.nn.Linear(3, 4)
                self.dropout_p = 0.0

            def forward(self, images, stochastic=False):
                features = images.mean(dim=(2, 3))
                logits = self.classifier(features)
                uncertainty = torch.sigmoid(self.uncertainty_head(features)).squeeze(-1)
                projection = torch.nn.functional.normalize(
                    self.projector(features), dim=-1
                )
                return {
                    "logits": logits,
                    "features": features,
                    "proj": projection,
                    "uncertainty": uncertainty,
                }

        torch.manual_seed(17)
        student = TinyModel()
        teacher = TinyModel()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        known_images = torch.rand(2, 3, 8, 8)
        labels = torch.tensor([0, 1])
        known_loader = [
            (known_images, labels, labels, torch.ones(2, dtype=torch.bool), torch.arange(2))
        ]
        discovery_loader = [
            (torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8))
        ]

        stats = train_one_epoch_student(
            student,
            teacher,
            known_loader,
            optimizer,
            torch.device("cpu"),
            discovery_loader=discovery_loader,
            alpha_discovery_uncertainty_separation=0.1,
            alpha_discovery=0.0,
            alpha_discovery_unknown=0.0,
            alpha_discovery_energy=0.0,
            alpha_discovery_uniform=0.0,
            alpha_discovery_feature_margin=0.0,
            alpha_discovery_feature_separation=0.0,
            alpha_discovery_boundary=0.0,
            alpha_discovery_knn_boundary=0.0,
            alpha_discovery_selective_unknown=0.0,
            alpha_discovery_selective_energy=0.0,
            alpha_reciprocal=0.0,
            alpha_joint_discovery=0.0,
        )

        self.assertGreater(stats["discovery_uncertainty_separation"], 0.0)

    def test_rejection_feature_margin_is_active_without_separation_loss(self):
        class TinyRejectionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3, 2)
                self.rejection_projector = torch.nn.Linear(3, 3)
                self.rejection_classifier = torch.nn.Linear(3, 2)
                self.rejection_uncertainty_head = torch.nn.Linear(3, 1)
                self.projector = torch.nn.Linear(3, 4)
                self.dropout_p = 0.0

            def forward(self, images, stochastic=False):
                features = images.mean(dim=(2, 3))
                rejection_features = self.rejection_projector(features)
                return {
                    "logits": self.classifier(features),
                    "features": features,
                    "rejection_features": rejection_features,
                    "rejection_logits": self.rejection_classifier(rejection_features),
                    "proj": torch.nn.functional.normalize(self.projector(features), dim=-1),
                    "uncertainty": torch.sigmoid(
                        self.rejection_uncertainty_head(rejection_features)
                    ).squeeze(-1),
                }

            def rejection_forward(self, features):
                rejection_features = self.rejection_projector(features)
                return (
                    rejection_features,
                    self.rejection_classifier(rejection_features),
                    torch.sigmoid(
                        self.rejection_uncertainty_head(rejection_features)
                    ).squeeze(-1),
                )

        torch.manual_seed(29)
        student = TinyRejectionModel()
        teacher = TinyRejectionModel()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        known_images = torch.rand(3, 3, 8, 8)
        labels = torch.tensor([0, 1, 0])
        known_loader = [
            (known_images, labels, labels, torch.ones(3, dtype=torch.bool), torch.arange(3))
        ]
        mixed_loader = [(torch.rand(3, 3, 8, 8), torch.rand(3, 3, 8, 8))]

        stats = train_one_epoch_student(
            student,
            teacher,
            known_loader,
            optimizer,
            torch.device("cpu"),
            discovery_loader=mixed_loader,
            discovery_pool_mode="mixed",
            alpha_discovery_rejection_feature_margin=0.1,
            discovery_rejection_feature_margin=0.0,
            alpha_discovery_rejection_feature_separation=0.0,
            alpha_discovery=0.0,
            alpha_discovery_unknown=0.0,
            alpha_discovery_energy=0.0,
            alpha_discovery_uniform=0.0,
            alpha_discovery_feature_margin=0.0,
            alpha_discovery_feature_separation=0.0,
            alpha_discovery_boundary=0.0,
            alpha_discovery_knn_boundary=0.0,
            alpha_discovery_uncertainty_separation=0.0,
            alpha_discovery_uncertainty_pu=0.0,
            alpha_discovery_selective_unknown=0.0,
            alpha_discovery_selective_energy=0.0,
            alpha_reciprocal=0.0,
            alpha_joint_discovery=0.0,
        )

        self.assertGreater(stats["discovery_rejection_feature_margin"], 0.0)

    def test_rejection_branch_distillation_is_active(self):
        class TinyRejectionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3, 3)
                self.classifier = torch.nn.Linear(3, 2)
                self.rejection_projector = torch.nn.Linear(3, 3)
                self.rejection_classifier = torch.nn.Linear(3, 2)
                self.rejection_uncertainty_head = torch.nn.Linear(3, 1)
                self.projector = torch.nn.Linear(3, 4)
                self.dropout_p = 0.0

            def forward(self, images, stochastic=False):
                features = self.encoder(images.mean(dim=(2, 3)))
                rejection_features = self.rejection_projector(features)
                return {
                    "logits": self.classifier(features),
                    "features": features,
                    "rejection_features": rejection_features,
                    "rejection_logits": self.rejection_classifier(rejection_features),
                    "proj": torch.nn.functional.normalize(self.projector(features), dim=-1),
                    "uncertainty": torch.sigmoid(
                        self.rejection_uncertainty_head(rejection_features)
                    ).squeeze(-1),
                }

            def rejection_forward(self, features):
                rejection_features = self.rejection_projector(features)
                return (
                    rejection_features,
                    self.rejection_classifier(rejection_features),
                    torch.sigmoid(
                        self.rejection_uncertainty_head(rejection_features)
                    ).squeeze(-1),
                )

        torch.manual_seed(31)
        student = TinyRejectionModel()
        teacher = TinyRejectionModel()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        images = torch.rand(4, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])
        loader = [(images, labels, labels, torch.ones(4, dtype=torch.bool), torch.arange(4))]
        stats = train_one_epoch_student(
            student,
            teacher,
            loader,
            optimizer,
            torch.device("cpu"),
            alpha_rejection_kd=0.5,
            alpha_rejection_feat_kd=0.5,
            alpha_kd=0.0,
            alpha_feat_kd=0.0,
            alpha_supcon=0.0,
        )
        self.assertGreater(stats["rejection_kd"], 0.0)
        self.assertGreater(stats["rejection_feat_kd"], 0.0)
        self.assertEqual(stats["discovery_rejection_feature_separation"], 0.0)

    def test_mixed_pool_pu_objective_consumes_unlabeled_batches(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3, 2)
                self.uncertainty_head = torch.nn.Linear(3, 1)
                self.projector = torch.nn.Linear(3, 4)
                self.dropout_p = 0.0

            def forward(self, images, stochastic=False):
                features = images.mean(dim=(2, 3))
                return {
                    "logits": self.classifier(features),
                    "features": features,
                    "proj": torch.nn.functional.normalize(self.projector(features), dim=-1),
                    "uncertainty": torch.sigmoid(self.uncertainty_head(features)).squeeze(-1),
                }

        torch.manual_seed(23)
        student = TinyModel()
        teacher = TinyModel()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        known_images = torch.rand(2, 3, 8, 8)
        labels = torch.tensor([0, 1])
        known_loader = [
            (known_images, labels, labels, torch.ones(2, dtype=torch.bool), torch.arange(2))
        ]
        mixed_unlabeled_loader = [
            (torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8))
        ]
        stats = train_one_epoch_student(
            student,
            teacher,
            known_loader,
            optimizer,
            torch.device("cpu"),
            discovery_loader=mixed_unlabeled_loader,
            alpha_discovery_uncertainty_pu=0.1,
            discovery_uncertainty_known_prior=0.2,
            alpha_discovery=0.0,
            alpha_discovery_unknown=0.0,
            alpha_discovery_energy=0.0,
            alpha_discovery_uniform=0.0,
            alpha_discovery_feature_margin=0.0,
            alpha_discovery_feature_separation=0.0,
            alpha_discovery_boundary=0.0,
            alpha_discovery_knn_boundary=0.0,
            alpha_discovery_uncertainty_separation=0.0,
            alpha_discovery_selective_unknown=0.0,
            alpha_discovery_selective_energy=0.0,
            alpha_reciprocal=0.0,
            alpha_joint_discovery=0.0,
        )
        self.assertGreater(stats["discovery_uncertainty_pu"], 0.0)
        self.assertIn("discovery_uncertainty_feature_margin_weight_mean", stats)
        self.assertEqual(stats["discovery_uncertainty_feature_margin_weight_mean"], 0.0)

    def test_candidate_gated_knn_boundary_consumes_mixed_unlabeled_batches(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3, 2)
                self.uncertainty_head = torch.nn.Linear(3, 1)
                self.projector = torch.nn.Linear(3, 4)
                self.dropout_p = 0.0

            def forward(self, images, stochastic=False):
                features = images.mean(dim=(2, 3))
                return {
                    "logits": self.classifier(features),
                    "features": features,
                    "proj": torch.nn.functional.normalize(self.projector(features), dim=-1),
                    "uncertainty": torch.sigmoid(self.uncertainty_head(features)).squeeze(-1),
                }

        torch.manual_seed(37)
        student = TinyModel()
        teacher = TinyModel()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        known_images = torch.rand(4, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])
        known_loader = [
            (known_images, labels, labels, torch.ones(4, dtype=torch.bool), torch.arange(4))
        ]
        mixed_loader = [
            (torch.rand(4, 3, 8, 8), torch.rand(4, 3, 8, 8))
        ]
        support = {
            "features_by_class": [torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]),
                                  torch.tensor([[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]])],
            "radii": torch.tensor([0.5, 0.5]),
        }
        stats = train_one_epoch_student(
            student,
            teacher,
            known_loader,
            optimizer,
            torch.device("cpu"),
            discovery_loader=mixed_loader,
            discovery_pool_mode="mixed",
            discovery_feature_candidate_gating=True,
            discovery_select_ratio=0.5,
            discovery_select_mode="consensus",
            alpha_discovery_knn_boundary=0.1,
            discovery_knn_support=support,
            alpha_discovery=0.0,
            alpha_discovery_unknown=0.0,
            alpha_discovery_energy=0.0,
            alpha_discovery_uniform=0.0,
            alpha_discovery_feature_margin=0.0,
            alpha_discovery_feature_separation=0.0,
            alpha_discovery_boundary=0.0,
            alpha_discovery_uncertainty_separation=0.0,
            alpha_discovery_uncertainty_pu=0.0,
            alpha_discovery_selective_unknown=0.0,
            alpha_discovery_selective_energy=0.0,
            alpha_discovery_objectosphere=0.1,
            objectosphere_known_radius=2.0,
            alpha_reciprocal=0.0,
            alpha_joint_discovery=0.0,
        )
        self.assertGreater(stats["discovery_knn_boundary"], 0.0)
        self.assertGreater(stats["discovery_objectosphere"], 0.0)

    def test_soft_weighted_knn_boundary_records_feature_candidate_diagnostics(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3, 2)
                self.uncertainty_head = torch.nn.Linear(3, 1)
                self.projector = torch.nn.Linear(3, 4)
                self.dropout_p = 0.0

            def forward(self, images, stochastic=False):
                features = images.mean(dim=(2, 3))
                return {
                    "logits": self.classifier(features),
                    "features": features,
                    "proj": torch.nn.functional.normalize(self.projector(features), dim=-1),
                    "uncertainty": torch.sigmoid(self.uncertainty_head(features)).squeeze(-1),
                }

        torch.manual_seed(38)
        student = TinyModel()
        teacher = TinyModel()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        known_images = torch.rand(4, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])
        known_loader = [(known_images, labels, labels, torch.ones(4, dtype=torch.bool), torch.arange(4))]
        mixed_loader = [(torch.rand(4, 3, 8, 8), torch.rand(4, 3, 8, 8))]
        support = {
            "features_by_class": [
                torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]),
                torch.tensor([[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]]),
            ],
            "radii": torch.tensor([0.5, 0.5]),
        }
        stats = train_one_epoch_student(
            student,
            teacher,
            known_loader,
            optimizer,
            torch.device("cpu"),
            discovery_loader=mixed_loader,
            discovery_pool_mode="mixed",
            discovery_feature_candidate_gating=True,
            discovery_select_ratio=0.5,
            discovery_select_mode="entropy",
            discovery_knn_soft_weighting=True,
            alpha_discovery_knn_boundary=0.1,
            discovery_knn_support=support,
            alpha_discovery=0.0,
            alpha_discovery_unknown=0.0,
            alpha_discovery_energy=0.0,
            alpha_discovery_uniform=0.0,
            alpha_discovery_feature_margin=0.0,
            alpha_discovery_feature_separation=0.0,
            alpha_discovery_boundary=0.0,
            alpha_discovery_uncertainty_separation=0.0,
            alpha_discovery_uncertainty_pu=0.0,
            alpha_discovery_selective_unknown=0.0,
            alpha_discovery_selective_energy=0.0,
            alpha_reciprocal=0.0,
            alpha_joint_discovery=0.0,
        )
        self.assertGreater(stats["discovery_knn_boundary"], 0.0)
        self.assertGreater(stats["feature_candidate_ratio"], 0.0)
        self.assertGreater(stats["feature_candidate_weight_mean"], 0.0)

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

    def test_supervised_center_loss_compacts_same_class_features(self):
        close = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
            requires_grad=True,
        )
        spread = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1])
        close_loss = supervised_center_loss(close, labels)
        spread_loss = supervised_center_loss(spread, labels)
        close_loss.backward()
        self.assertLess(close_loss.item(), spread_loss.item())
        self.assertTrue(torch.isfinite(close.grad).all())

    def test_supervised_radius_loss_only_penalizes_outliers(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_radius_loss(features, labels, radius=0.01)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(
            supervised_radius_loss(features.detach(), labels, radius=1.0).item(),
            0.0,
        )

    def test_supervised_center_margin_loss_is_finite_and_differentiable(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_center_margin_loss(features, labels, margin=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

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

    def test_feature_rejector_fusion_uses_validation_normalization(self):
        outputs = {
            "entropy": np.array([0.1, 0.2, 0.4, 0.8]),
            "epistemic": np.zeros(4),
            "aleatoric": np.zeros(4),
            "head_uncertainty": np.zeros(4),
            "features": np.array(
                [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
                dtype=np.float32,
            ),
            "logits": np.array(
                [[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]],
                dtype=np.float32,
            ),
            "probs": np.array(
                [[0.95, 0.05], [0.95, 0.05], [0.05, 0.95], [0.05, 0.95]],
                dtype=np.float32,
            ),
            "labels": np.array([0, 0, 1, 1]),
            "feature_rejector_score": np.array([0.1, 0.2, 1.0, 1.2]),
            "known_support_score": np.array([0.8, 0.9, 1.4, 1.6]),
        }
        prototypes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        normalization = fit_score_normalization(
            outputs, prototypes=prototypes, gaussian_stats=None
        )
        score, _ = compute_open_score(
            outputs,
            prototypes=prototypes,
            score_mode="feature_rejector_fusion",
            normalization=normalization,
            fusion_base_score="normalized_entropy_proto",
            fusion_rejector_weight=0.75,
        )
        self.assertEqual(score.shape, (4,))
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[-1], score[0])

        missing_support = dict(outputs)
        missing_support.pop("known_support_score")
        with self.assertRaises(ValueError):
            compute_open_score(
                missing_support,
                prototypes=prototypes,
                score_mode="feature_rejector_fusion",
                normalization=normalization,
                fusion_base_score="normalized_entropy_proto",
                fusion_rejector_weight=0.75,
            )

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

    def test_knn_distance_can_be_conditioned_on_predicted_class(self):
        outputs = {
            "features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "logits": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        }
        bank = {
            "features": np.array(
                [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]],
                dtype=np.float32,
            ),
            "labels": np.array([0, 0, 1, 1], dtype=np.int64),
        }
        attach_knn_distances(outputs, bank, k=1, feature_key="features")
        np.testing.assert_allclose(
            outputs["knn_predicted_class_distance"], np.array([0.0, 0.0]), atol=1e-6
        )

    def test_knn_class_radii_exclude_self_and_relative_distance_is_scale_adjusted(self):
        bank = {
            "features": np.array(
                [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]],
                dtype=np.float32,
            ),
            "labels": np.array([0, 0, 1, 1], dtype=np.int64),
        }
        bank["class_support_radii"] = estimate_classwise_knn_radii(
            bank["features"], bank["labels"], k=1, quantile=0.95
        )
        outputs = {
            "features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "logits": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        }
        attach_knn_distances(outputs, bank, k=1, feature_key="features")
        self.assertEqual(len(bank["class_support_radii"]), 2)
        self.assertTrue(np.isfinite(outputs["knn_predicted_class_relative_distance"]).all())
        self.assertTrue(np.isfinite(outputs["knn_predicted_class_support"]).all())
        self.assertTrue(((outputs["knn_predicted_class_support"] >= 0.0) &
                         (outputs["knn_predicted_class_support"] <= 1.0)).all())
        self.assertTrue(np.isfinite(outputs["knn_min_class_distance"]).all())
        self.assertTrue(np.isfinite(outputs["knn_min_class_relative_distance"]).all())

    def test_min_class_knn_distance_checks_support_beyond_predicted_class(self):
        bank = {
            "features": np.array(
                [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99]],
                dtype=np.float32,
            ),
            "labels": np.array([0, 0, 1, 1], dtype=np.int64),
        }
        bank["class_support_radii"] = estimate_classwise_knn_radii(
            bank["features"], bank["labels"], k=1, quantile=0.95
        )
        outputs = {
            # The classifier predicts class 0, but the feature is close to class 1.
            "features": np.array([[0.0, 1.0]], dtype=np.float32),
            "logits": np.array([[4.0, 0.0]], dtype=np.float32),
        }
        attach_knn_distances(outputs, bank, k=1, feature_key="features")
        self.assertGreater(outputs["knn_predicted_class_distance"][0], 0.5)
        self.assertLess(outputs["knn_min_class_distance"][0], 1e-5)
        self.assertLess(
            outputs["knn_min_class_relative_distance"][0],
            outputs["knn_predicted_class_relative_distance"][0],
        )

    def test_min_class_relative_knn_score_uses_known_validation_statistics(self):
        outputs = {
            "features": np.zeros((3, 2), dtype=np.float32),
            "entropy": np.array([0.1, 0.2, 0.3]),
            "epistemic": np.zeros(3),
            "aleatoric": np.zeros(3),
            "knn_min_class_relative_distance": np.array([1.0, 2.0, 3.0]),
        }
        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)
        score, _ = compute_open_score(
            outputs,
            score_mode="normalized_entropy_min_class_relative_knn",
            normalization=stats,
        )
        self.assertTrue(np.isfinite(score).all())

    def test_min_class_raw_knn_score_uses_known_validation_statistics(self):
        outputs = {
            "features": np.zeros((3, 2), dtype=np.float32),
            "entropy": np.array([0.1, 0.2, 0.3]),
            "epistemic": np.zeros(3),
            "aleatoric": np.zeros(3),
            "knn_min_class_distance": np.array([1.0, 2.0, 3.0]),
        }
        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)
        score, _ = compute_open_score(
            outputs,
            score_mode="normalized_entropy_min_class_knn",
            normalization=stats,
        )
        self.assertTrue(np.isfinite(score).all())

    def test_uncertainty_corrected_min_class_knn_score_is_finite(self):
        outputs = {
            "features": np.zeros((3, 2), dtype=np.float32),
            "entropy": np.array([0.1, 0.2, 0.3]),
            "head_uncertainty": np.array([0.2, 0.4, 0.6]),
            "epistemic": np.zeros(3),
            "aleatoric": np.zeros(3),
            "knn_min_class_distance": np.array([1.0, 2.0, 3.0]),
        }
        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)
        score, _ = compute_open_score(
            outputs,
            score_mode="normalized_entropy_uncertainty_min_class_knn",
            normalization=stats,
        )
        self.assertTrue(np.isfinite(score).all())

    def test_classwise_predicted_knn_normalization_uses_known_validation_classes(self):
        outputs = {
            "features": np.zeros((4, 2), dtype=np.float32),
            "logits": np.array(
                [[3, 0], [3, 0], [0, 3], [0, 3]], dtype=np.float32
            ),
            "labels": np.array([0, 0, 1, 1]),
            "entropy": np.array([0.0, 1.0, 0.0, 1.0]),
            "epistemic": np.zeros(4),
            "aleatoric": np.zeros(4),
            "knn_predicted_class_distance": np.array([1.0, 3.0, 10.0, 14.0]),
        }
        stats = fit_score_normalization(outputs, prototypes=None, gaussian_stats=None)
        classwise = stats["knn_predicted_class_distance_classwise"]
        self.assertEqual(classwise["count"], [2, 2])
        score, _ = compute_open_score(
            outputs,
            score_mode="normalized_entropy_predicted_class_knn",
            normalization=stats,
        )
        self.assertTrue(np.isfinite(score).all())

    def test_three_signal_score_adds_known_validation_standardized_components(self):
        outputs = {
            "features": np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "entropy": np.array([1.0, 3.0], dtype=np.float32),
            "epistemic": np.zeros(2, dtype=np.float32),
            "aleatoric": np.zeros(2, dtype=np.float32),
            "knn_distance": np.array([2.0, 6.0], dtype=np.float32),
        }
        gaussian_stats = {
            "means": np.array([[0.0, 0.0]], dtype=np.float32),
            "variances": np.ones((1, 2), dtype=np.float32),
        }
        normalization = {
            "entropy": {"mean": 1.0, "std": 1.0},
            "mahalanobis": {"mean": 1.0, "std": 1.0},
            "knn_distance": {"mean": 2.0, "std": 2.0},
        }
        score, _ = compute_open_score(
            outputs,
            score_mode="normalized_entropy_mahalanobis_knn",
            normalization=normalization,
            gaussian_stats=gaussian_stats,
        )
        # Entropy z-scores [0, 2], Mahalanobis z-scores [-1, 1],
        # and kNN-distance z-scores [0, 2].
        np.testing.assert_allclose(score, np.array([-1.0, 5.0]), atol=1e-6)

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

    def test_distance_consensus_selection_uses_known_prototypes(self):
        logits = torch.tensor(
            [[5.0, 0.0], [4.0, 0.0], [0.1, 0.1], [3.0, 0.0]]
        )
        features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [-1.0, 0.0]]
        )
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mask = select_discovery_candidates(
            logits,
            ratio=0.25,
            mode="prototype_distance",
            features=features,
            prototypes=prototypes,
        )
        self.assertEqual(mask.sum().item(), 1)
        self.assertTrue(mask[3].item())

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


class PrototypeRefreshTest(unittest.TestCase):
    def test_kmeans_refresh_completes_with_single_init(self):
        class DummyStudent(torch.nn.Module):
            def forward(self, images):
                return {
                    "features": images,
                    "logits": torch.zeros(images.size(0), 2),
                    "uncertainty": torch.zeros(images.size(0)),
                }

        class DummyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prototypes = torch.nn.Parameter(torch.zeros(3, 2))

        generator = torch.Generator().manual_seed(7)
        features = torch.randn(12, 2, generator=generator)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, torch.zeros(12, dtype=torch.long)),
            batch_size=4,
        )
        head = DummyHead()
        stats = initialize_novel_head_kmeans(
            head,
            DummyStudent(),
            loader,
            torch.device("cpu"),
            num_novel=3,
            seed=7,
            n_init=1,
        )
        self.assertEqual(stats["samples"], 12)
        self.assertTrue(torch.isfinite(head.prototypes).all())
        self.assertEqual(tuple(head.prototypes.shape), (3, 2))


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
    def test_pu_unknown_feature_margin_has_nonzero_loss_and_gradient(self):
        known_features = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
        unlabeled_features = torch.tensor(
            [[1.0, 0.5]], dtype=torch.float32, requires_grad=True
        )
        known_prototypes = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        loss = pu_unknown_feature_margin_loss(
            known_features,
            unlabeled_features,
            known_prototypes,
            known_prior=0.5,
            similarity_margin=0.2,
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(unlabeled_features.grad)
        self.assertGreater(unlabeled_features.grad.norm().item(), 0.0)

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

    def test_run_discovery_separates_classifier_accuracy_from_rejection(self):
        outputs = {
            "entropy": np.array([0.1, 2.0, 0.2]),
            "epistemic": np.zeros(3),
            "aleatoric": np.zeros(3),
            "probs": np.array([[0.9, 0.1], [0.6, 0.4], [0.8, 0.2]]),
            "logits": np.array([[2.0, 0.0], [1.2, 0.0], [1.5, 0.0]]),
            "features": np.eye(3, dtype=np.float32),
            "projections": np.eye(3, dtype=np.float32),
            "labels": np.array([0, 0, 1]),
            "raw_labels": np.array([0, 0, 1]),
            "is_known": np.array([1, 1, 1]),
        }
        report, _, _, _ = run_discovery(
            outputs,
            threshold=1.0,
            num_novel=1,
            score_mode="entropy_only",
            enable_clustering=False,
        )

        # The second known sample is rejected but its class prediction is
        # correct.  Rejection must not lower ordinary known-class accuracy.
        self.assertEqual(report["known_class_correct_all_known"], 2)
        self.assertAlmostEqual(report["known_class_accuracy_all_known"], 2 / 3)
        self.assertAlmostEqual(report["accepted_correct_fraction_of_all_known"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
