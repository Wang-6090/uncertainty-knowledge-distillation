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
    combine_known_novel_logits,
    joint_discovery_loss,
    information_maximization_loss,
    neighbor_consistency_loss,
    novel_consistency_loss,
    prototype_pseudo_label_loss,
)
from novel_discovery.pipeline import select_joint_candidates


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

    def test_information_maximization_is_finite_and_differentiable(self):
        logits = torch.randn(8, 4, requires_grad=True)
        loss = information_maximization_loss(logits)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_unified_logits_append_novel_classes(self):
        known = torch.randn(3, 5)
        novel = torch.randn(3, 4)
        unified = combine_known_novel_logits(known, novel)
        self.assertEqual(tuple(unified.shape), (3, 9))

    def test_weighted_joint_loss_is_finite(self):
        first_features = torch.randn(6, 5, requires_grad=True)
        second_features = torch.randn(6, 5, requires_grad=True)
        first_logits = torch.randn(6, 4, requires_grad=True)
        second_logits = torch.randn(6, 4, requires_grad=True)
        weights = torch.linspace(0.1, 1.0, steps=6)
        losses = joint_discovery_loss(
            first_features,
            second_features,
            first_logits,
            second_logits,
            confidence_threshold=0.0,
            sample_weights=weights,
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(first_logits.grad)

    def test_hard_prototype_pseudo_loss_is_finite_and_differentiable(self):
        first = torch.randn(8, 4, requires_grad=True)
        second = torch.randn(8, 4, requires_grad=True)
        loss = prototype_pseudo_label_loss(first, second, confidence_threshold=0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)

    def test_prototype_distance_candidate_selection(self):
        known_logits = torch.randn(8, 3)
        novel_logits = torch.randn(8, 2)
        features = torch.randn(8, 5)
        prototypes = torch.randn(3, 5)
        mask = select_joint_candidates(
            known_logits,
            novel_logits,
            features=features,
            known_prototypes=prototypes,
            ratio=0.25,
            mode="prototype_distance",
        )
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(int(mask.sum()), 2)

    def test_distance_consensus_candidate_selection(self):
        mask = select_joint_candidates(
            torch.randn(8, 3),
            torch.randn(8, 2),
            uncertainty=torch.rand(8),
            features=torch.randn(8, 5),
            known_prototypes=torch.randn(3, 5),
            ratio=0.25,
            mode="distance_consensus",
        )
        self.assertEqual(int(mask.sum()), 2)


if __name__ == "__main__":
    unittest.main()
