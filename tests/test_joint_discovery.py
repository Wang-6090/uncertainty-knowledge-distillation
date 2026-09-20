from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.joint_discovery import (
    NovelPrototypeHead,
    balanced_assignment_loss,
    balanced_assignments,
    joint_discovery_loss,
    neighbor_consistency_loss,
    novel_consistency_loss,
)


class JointDiscoveryTest(unittest.TestCase):
    def test_novel_prototype_head_returns_expected_logits(self):
        head = NovelPrototypeHead(feature_dim=8, num_novel=4)
        features = torch.randn(6, 8, requires_grad=True)
        logits = head(features)

        self.assertEqual(tuple(logits.shape), (6, 4))
        logits.mean().backward()
        self.assertIsNotNone(head.prototypes.grad)
        self.assertIsNotNone(features.grad)

    def test_balanced_assignments_are_row_normalized(self):
        logits = torch.randn(8, 4)
        assignments = balanced_assignments(logits)

        self.assertTrue(torch.allclose(assignments.sum(dim=1), torch.ones(8), atol=1e-4))
        self.assertTrue(torch.isfinite(assignments).all())

    def test_joint_loss_is_finite_and_differentiable(self):
        first_features = torch.randn(8, 6, requires_grad=True)
        second_features = torch.randn(8, 6, requires_grad=True)
        first_logits = torch.randn(8, 4, requires_grad=True)
        second_logits = torch.randn(8, 4, requires_grad=True)

        losses = joint_discovery_loss(
            first_features,
            second_features,
            first_logits,
            second_logits,
            confidence_threshold=0.2,
            neighbor_k=3,
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(first_logits.grad)
        self.assertIsNotNone(second_logits.grad)

    def test_empty_and_small_batches_are_safe(self):
        empty = torch.empty(0, 4)
        self.assertEqual(novel_consistency_loss(empty, empty).item(), 0.0)
        self.assertEqual(balanced_assignment_loss(empty).item(), 0.0)
        self.assertEqual(neighbor_consistency_loss(torch.empty(1, 6), torch.empty(1, 4)).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
