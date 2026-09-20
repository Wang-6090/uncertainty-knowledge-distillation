from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.uncertainty_kd import (
    feature_distillation_loss,
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
            clamp_min=0.5,
            clamp_max=1.5,
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
        weights = uncertainty_weights(teacher_uncertainty)
        expected = (per_sample * weights).mean()
        observed = kl_distillation_loss(
            student_logits,
            teacher_logits,
            teacher_uncertainty=teacher_uncertainty,
        )

        self.assertTrue(torch.allclose(observed, expected))

    def test_empty_batch_returns_differentiable_zero_loss(self):
        student_logits = torch.empty(0, 3, requires_grad=True)
        teacher_logits = torch.empty(0, 3)

        loss = kl_distillation_loss(student_logits, teacher_logits)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(student_logits.grad)

    def test_feature_distillation_is_weighted_and_differentiable(self):
        student_projection = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        teacher_projection = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        teacher_uncertainty = torch.tensor([0.0, 1.0], requires_grad=True)

        loss = feature_distillation_loss(
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
