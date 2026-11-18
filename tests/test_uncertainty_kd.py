from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.losses import (
    distillation_loss,
    feature_distillation_loss,
    uncertainty_decoupled_distillation_loss,
)
from novel_discovery.uncertainty_kd import (
    feature_distillation_loss as module_feature_distillation_loss,
    kl_distillation_loss,
    per_sample_kl_distillation_loss,
    uncertainty_calibration_loss,
    uncertainty_weights,
)


class UncertaintyKnowledgeDistillationTest(unittest.TestCase):
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
