from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.losses import (
    class_holdout_support_margin,
    class_center_feature_distillation_loss,
    centered_feature_distillation_loss,
    distillation_loss,
    feature_distillation_loss,
    quality_aware_distillation_loss,
    semantic_representation_distillation,
    selective_distillation_loss,
    teacher_quality_weights,
    uncertainty_decoupled_distillation_loss,
)
from novel_discovery.uncertainty_kd import (
    feature_distillation_loss as module_feature_distillation_loss,
    kl_distillation_loss,
    per_sample_kl_distillation_loss,
    quality_aware_kl_loss,
    teacher_quality_weights as module_teacher_quality_weights,
    uncertainty_calibration_loss,
    uncertainty_weights,
)
from scripts.run_uncertainty_kd_core_pair import (
    discovery_score_mode_for_profile,
    make_split,
)


class UncertaintyKnowledgeDistillationTest(unittest.TestCase):
    def test_class_holdout_support_margin_updates_student_only(self):
        torch.manual_seed(7)
        features = torch.randn(12, 5, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        loss, active_fraction = class_holdout_support_margin(
            features, labels, holdout_fraction=0.25, margin=0.2
        )
        self.assertGreater(active_fraction, 0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(features.grad.abs().sum().item(), 0.0)

    def test_class_holdout_support_margin_is_zero_when_batch_has_too_few_classes(self):
        features = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss, active_fraction = class_holdout_support_margin(features, labels)
        self.assertEqual(active_fraction, 0.0)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_class_holdout_support_margin_validates_inputs(self):
        with self.assertRaises(ValueError):
            class_holdout_support_margin(torch.zeros(2, 3, 1), torch.zeros(2, dtype=torch.long))
        with self.assertRaises(ValueError):
            class_holdout_support_margin(torch.zeros(2, 3), torch.zeros(3, dtype=torch.long))
        with self.assertRaises(ValueError):
            class_holdout_support_margin(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), holdout_fraction=1.0)

    def test_dry_run_split_generation_does_not_write_files(self):
        expected_split = {
            "protocol": "semantic_hard",
            "metadata": {"seed": 2170},
            "known_classes": list(range(60)),
            "novel_classes": list(range(60, 100)),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            split_dir = Path(temporary_dir) / "seed2170"
            with (
                patch(
                    "scripts.run_uncertainty_kd_core_pair.read_cifar100_taxonomy",
                    return_value=([], [], {}),
                ),
                patch(
                    "scripts.run_uncertainty_kd_core_pair.build_protocols",
                    return_value={"semantic_hard": expected_split},
                ),
            ):
                split_path, split = make_split(
                    2170, split_dir, Path(temporary_dir), persist=False
                )
            self.assertEqual(split, expected_split)
            self.assertFalse(split_path.exists())
            self.assertFalse(split_dir.exists())

    def test_srd_kl_matches_teacher_critic_formula_and_updates_student_only(self):
        features = torch.tensor(
            [[0.2, 0.8, -0.1], [0.7, -0.2, 0.3]], requires_grad=True
        )
        critic_weight = torch.tensor(
            [[0.4, -0.1, 0.2], [-0.3, 0.5, 0.1]], requires_grad=True
        )
        critic_bias = torch.tensor([0.1, -0.2], requires_grad=True)
        teacher_logits = torch.tensor(
            [[1.1, -0.4], [-0.2, 0.9]], requires_grad=True
        )
        temperature = 1.7
        cross_logits = F.linear(features, critic_weight.detach(), critic_bias.detach())
        expected = F.kl_div(
            F.log_softmax(cross_logits / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2

        observed = semantic_representation_distillation(
            features,
            teacher_logits,
            critic_weight,
            critic_bias,
            temperature=temperature,
        )
        self.assertTrue(torch.allclose(observed, expected))
        observed.backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(features.grad.abs().sum().item(), 0.0)
        self.assertIsNone(critic_weight.grad)
        self.assertIsNone(critic_bias.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_srd_rejects_incompatible_feature_and_class_dimensions(self):
        with self.assertRaises(ValueError):
            semantic_representation_distillation(
                torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(3, 5)
            )
        with self.assertRaises(ValueError):
            semantic_representation_distillation(
                torch.zeros(2, 4), torch.zeros(2, 2), torch.zeros(3, 4)
            )

    def test_selective_kd_keeps_only_teacher_correct_rows(self):
        student = torch.tensor(
            [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0]], requires_grad=True
        )
        teacher = torch.tensor(
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], requires_grad=True
        )
        labels = torch.tensor([0, 2])
        loss, fraction = selective_distillation_loss(student, teacher, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(fraction.item(), 0.5)
        loss.backward()
        self.assertGreater(student.grad.abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)

    def test_selective_kd_returns_finite_zero_when_teacher_is_always_wrong(self):
        student = torch.randn(2, 3, requires_grad=True)
        teacher = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True)
        loss, fraction = selective_distillation_loss(
            student, teacher, torch.tensor([0, 0])
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(fraction.item(), 0.0)
        loss.backward()
        self.assertEqual(student.grad.abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)

    def test_isolated_centered_feature_profile_uses_max_softmax_discovery(self):
        self.assertEqual(
            discovery_score_mode_for_profile("ce_centered_feature_kd"),
            "max_softmax",
        )
        self.assertEqual(
            discovery_score_mode_for_profile("ce_only"),
            "max_softmax",
        )
        self.assertEqual(
            discovery_score_mode_for_profile("current_auxiliary"),
            "feature_rejector",
        )

    def test_centered_feature_kd_matches_within_class_geometry_and_updates_student_only(self):
        student = torch.tensor(
            [[1.0, 0.0], [0.2, 0.9], [0.0, 1.0], [0.8, 0.3]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1])
        loss = centered_feature_distillation_loss(student, teacher, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)

    def test_centered_feature_kd_skips_singletons_and_invalid_rows(self):
        student = torch.randn(3, 4, requires_grad=True)
        teacher = torch.randn(3, 4, requires_grad=True)
        labels = torch.tensor([0, 1, -1])
        loss = centered_feature_distillation_loss(student, teacher, labels)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertEqual(student.grad.abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)

    def test_centered_feature_kd_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            centered_feature_distillation_loss(
                torch.zeros(2, 3), torch.zeros(2, 4), torch.tensor([0, 0])
            )

    def test_class_center_feature_kd_updates_student_only(self):
        student = torch.tensor(
            [[1.0, 0.0], [0.2, 0.9], [0.0, 1.0], [0.8, 0.3]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]],
            requires_grad=True,
        )
        loss = class_center_feature_distillation_loss(
            student, teacher, torch.tensor([0, 0, 1, 1])
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)

    def test_quality_weights_use_true_class_confidence_and_are_detached(self):
        teacher_logits = torch.tensor(
            [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0]], requires_grad=True
        )
        labels = torch.tensor([0, 2])
        weights = module_teacher_quality_weights(
            teacher_logits, labels, floor=0.25, mode="mean_normalized"
        )
        expected_quality = torch.softmax(teacher_logits.detach(), dim=1)[
            torch.arange(2), labels
        ].clamp_min(0.25)
        expected = expected_quality / expected_quality.mean()
        self.assertTrue(torch.allclose(weights, expected))
        self.assertFalse(weights.requires_grad)
        self.assertTrue(torch.allclose(weights, teacher_quality_weights(
            teacher_logits, labels, floor=0.25
        )))

    def test_quality_aware_kl_matches_weighted_per_sample_formula(self):
        student = torch.tensor([[1.0, 0.0, -0.2], [0.1, 0.9, 0.0]], requires_grad=True)
        teacher = torch.tensor([[0.8, 0.2, -0.1], [0.0, 1.0, 0.2]])
        labels = torch.tensor([0, 2])
        per_sample = per_sample_kl_distillation_loss(student, teacher, temperature=1.7)
        weights = module_teacher_quality_weights(
            teacher, labels, floor=0.25, power=1.0, mode="mean_normalized"
        )
        expected = (per_sample * weights).mean()
        observed = quality_aware_kl_loss(
            student, teacher, labels, temperature=1.7, quality_floor=0.25
        )
        self.assertTrue(torch.allclose(observed, expected))
        observed.backward()
        self.assertIsNotNone(student.grad)

    def test_quality_aware_kl_rejects_invalid_labels_and_detaches_teacher(self):
        student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[0.8, 0.2], [0.0, 1.0]], requires_grad=True)
        with self.assertRaises(ValueError):
            quality_aware_distillation_loss(student, teacher, torch.tensor([0, 2]))
        loss = quality_aware_distillation_loss(student, teacher, torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0.0)

    def test_standard_kl_only_backpropagates_to_student(self):
        student_logits = torch.tensor([[1.0, 0.0], [0.1, 0.9]], requires_grad=True)
        teacher_logits = torch.tensor([[0.8, 0.2], [0.0, 1.0]], requires_grad=True)

        loss = kl_distillation_loss(
            student_logits,
            teacher_logits,
            uncertainty_weighted=False,
            temperature=2.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_uncertainty_weights_are_detached_normalized_and_clipped(self):
        uncertainty = torch.tensor([0.0, 1.0, 3.0], requires_grad=True)

        normalized = uncertainty_weights(uncertainty, mode="mean_normalized")
        clipped = uncertainty_weights(
            uncertainty,
            mode="mean_normalized",
            uncertainty_weight_min=0.5,
            uncertainty_weight_max=1.5,
        )

        self.assertAlmostEqual(normalized.mean().item(), 1.0, places=6)
        self.assertTrue(torch.all(clipped >= 0.5))
        self.assertTrue(torch.all(clipped <= 1.5))
        self.assertFalse(clipped.requires_grad)
        self.assertIsNone(uncertainty.grad)

    def test_weighted_kl_matches_manual_per_sample_average(self):
        student_logits = torch.tensor([[1.0, 0.0], [0.1, 0.9]], requires_grad=True)
        teacher_logits = torch.tensor([[0.8, 0.2], [0.0, 1.0]])
        teacher_uncertainty = torch.tensor([0.0, 1.0])

        per_sample = per_sample_kl_distillation_loss(student_logits, teacher_logits)
        expected = (per_sample * uncertainty_weights(teacher_uncertainty)).mean()
        observed = kl_distillation_loss(
            student_logits,
            teacher_logits,
            teacher_uncertainty=teacher_uncertainty,
        )

        self.assertTrue(torch.allclose(observed, expected))

    def test_losses_module_uses_same_kl_implementation(self):
        student_logits = torch.tensor([[1.0, 0.0], [0.1, 0.9]])
        teacher_logits = torch.tensor([[0.8, 0.2], [0.0, 1.0]])
        uncertainty = torch.tensor([0.2, 0.7])

        module_loss = kl_distillation_loss(
            student_logits,
            teacher_logits,
            teacher_uncertainty=uncertainty,
            temperature=2.0,
        )
        public_loss = distillation_loss(
            student_logits,
            teacher_logits,
            teacher_uncertainty=uncertainty,
            temperature=2.0,
        )
        self.assertTrue(torch.allclose(public_loss, module_loss))

    def test_feature_loss_public_entry_uses_module(self):
        student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        uncertainty = torch.tensor([0.0, 1.0])

        self.assertTrue(
            torch.allclose(
                feature_distillation_loss(student, teacher, uncertainty),
                module_feature_distillation_loss(student, teacher, uncertainty),
            )
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            kl_distillation_loss(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                teacher_uncertainty=torch.zeros(2, 1),
            )

    def test_empty_batch_returns_differentiable_zero_loss(self):
        student_logits = torch.empty(0, 3, requires_grad=True)
        teacher_logits = torch.empty(0, 3)

        loss = kl_distillation_loss(student_logits, teacher_logits)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(student_logits.grad)

    def test_uncertainty_decoupled_kd_is_finite_and_updates_student_only(self):
        student_logits = torch.tensor(
            [[1.0, 0.2, -0.1], [0.0, 1.1, 0.3]], requires_grad=True
        )
        teacher_logits = torch.tensor(
            [[0.7, 0.4, -0.2], [0.1, 0.8, 0.5]], requires_grad=True
        )
        labels = torch.tensor([0, 1])
        uncertainty = torch.tensor([0.1, 0.8], requires_grad=True)

        loss = uncertainty_decoupled_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            uncertainty,
            temperature=2.0,
            uncertainty_weight_mode="mean_normalized",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNone(teacher_logits.grad)
        self.assertIsNone(uncertainty.grad)
        self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)

    def test_uncertainty_decoupled_kd_rejects_invalid_labels(self):
        with self.assertRaises(ValueError):
            uncertainty_decoupled_distillation_loss(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.tensor([0, 3]),
                torch.zeros(2),
            )

    def test_uncertainty_decoupled_kd_matches_explicit_formula(self):
        student = torch.tensor([[1.2, 0.1, -0.4], [0.2, 0.9, -0.1]])
        teacher = torch.tensor([[0.8, 0.5, -0.2], [-0.1, 1.1, 0.3]])
        labels = torch.tensor([0, 1])
        uncertainty = torch.tensor([0.2, 0.7])
        temperature = 1.7

        student_prob = F.softmax(student / temperature, dim=1)
        teacher_prob = F.softmax(teacher / temperature, dim=1)
        weights = uncertainty_weights(uncertainty, mode="mean_normalized")
        per_sample = []
        for index, label in enumerate(labels.tolist()):
            student_target_rest = torch.stack(
                (student_prob[index, label], 1.0 - student_prob[index, label])
            )
            teacher_target_rest = torch.stack(
                (teacher_prob[index, label], 1.0 - teacher_prob[index, label])
            )
            target_kl = F.kl_div(
                student_target_rest.log(), teacher_target_rest, reduction="sum"
            )
            mask = torch.arange(student.size(1)) != label
            student_non_target = student_prob[index, mask]
            teacher_non_target = teacher_prob[index, mask]
            student_non_target = student_non_target / student_non_target.sum()
            teacher_non_target = teacher_non_target / teacher_non_target.sum()
            non_target_kl = F.kl_div(
                student_non_target.log(), teacher_non_target, reduction="sum"
            )
            per_sample.append(target_kl + weights[index] * non_target_kl)
        expected = torch.stack(per_sample).mean() * (temperature**2)
        actual = uncertainty_decoupled_distillation_loss(
            student,
            teacher,
            labels,
            uncertainty,
            temperature=temperature,
            uncertainty_weight_mode="mean_normalized",
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_feature_distillation_is_weighted_and_differentiable(self):
        student_projection = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        teacher_projection = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        teacher_uncertainty = torch.tensor([0.0, 1.0], requires_grad=True)

        loss = module_feature_distillation_loss(
            student_projection,
            teacher_projection,
            teacher_uncertainty=teacher_uncertainty,
            uncertainty_weight_mode="mean_normalized",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student_projection.grad)
        self.assertIsNone(teacher_projection.grad)
        self.assertIsNone(teacher_uncertainty.grad)

    def test_uncertainty_calibration_targets_do_not_push_classifier_logits(self):
        uncertainty = torch.tensor([0.2, 0.8], requires_grad=True)
        logits = torch.tensor([[3.0, 0.0], [0.1, 0.2]], requires_grad=True)
        labels = torch.tensor([0, 0])

        loss = uncertainty_calibration_loss(
            uncertainty,
            logits,
            labels,
            target_mode="classification_error",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(uncertainty.grad)
        self.assertIsNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
