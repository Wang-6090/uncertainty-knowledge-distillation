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
    memory_neighbor_consistency_loss,
    information_maximization_loss,
    known_residual_weights,
    neighbor_novel_support_weights,
    novel_mass_weights,
    bounded_novel_mass_weights,
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

    def test_zero_weight_samples_do_not_change_active_sinkhorn_targets(self):
        active_logits = torch.tensor([[4.0, 0.0, -1.0], [0.0, 3.0, -1.0]])
        weights = torch.tensor([1.0, 1.0, 0.0])
        first = balanced_assignments(
            torch.cat([active_logits, torch.tensor([[10.0, -10.0, -10.0]])]),
            sample_weights=weights,
            iterations=5,
        )
        second = balanced_assignments(
            torch.cat([active_logits, torch.tensor([[-10.0, 10.0, -10.0]])]),
            sample_weights=weights,
            iterations=5,
        )
        self.assertTrue(torch.allclose(first[:2], second[:2], atol=1e-6))
        self.assertTrue(torch.allclose(first.sum(dim=-1), torch.ones(3), atol=1e-5))

    def test_weighted_sinkhorn_validates_sample_weight_shape(self):
        with self.assertRaises(ValueError):
            balanced_assignments(torch.randn(4, 3), sample_weights=torch.ones(3))

    def test_sinkhorn_is_finite_for_low_temperature_and_large_logits(self):
        logits = torch.tensor([[100.0, 80.0, -50.0], [-80.0, 100.0, 50.0]])
        weights = torch.tensor([1.0, 0.2])
        for sample_weights in (None, weights):
            assignments = balanced_assignments(
                logits, temperature=0.01, iterations=5, sample_weights=sample_weights
            )
            self.assertTrue(torch.isfinite(assignments).all())
            self.assertTrue(torch.allclose(
                assignments.sum(dim=-1), torch.ones(2), atol=1e-5
            ))

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

    def test_memory_neighbor_consistency_is_finite_and_differentiable(self):
        features = torch.randn(4, 6, requires_grad=True)
        logits = torch.randn(4, 3, requires_grad=True)
        memory_features = torch.randn(8, 6)
        memory_logits = torch.randn(8, 3)
        loss = memory_neighbor_consistency_loss(
            features, logits, memory_features, memory_logits, k=3
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # Neighbor selection is intentionally detached; only current logits
        # receive gradients from this consistency target.
        self.assertIsNone(features.grad)
        self.assertIsNotNone(logits.grad)

    def test_memory_neighbor_consistency_empty_bank_is_zero(self):
        loss = memory_neighbor_consistency_loss(
            torch.randn(2, 4),
            torch.randn(2, 3),
            torch.empty(0, 4),
            torch.empty(0, 3),
        )
        self.assertEqual(float(loss), 0.0)

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

    def test_known_residual_weights_downweight_confident_known_samples(self):
        logits = torch.tensor([[8.0, 0.0], [0.0, 0.0]])
        weights = known_residual_weights(logits)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertTrue(torch.all((weights >= 0.0) & (weights <= 1.0)))

    def test_novel_mass_weights_compare_known_and_novel_evidence(self):
        known = torch.tensor([[8.0, 0.0], [0.0, 0.0]])
        novel = torch.tensor([[0.0, 0.0], [8.0, 8.0]])
        weights = novel_mass_weights(known, novel)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertTrue(torch.all((weights >= 0.0) & (weights <= 1.0)))

    def test_bounded_novel_mass_weights_preserve_relative_evidence(self):
        masses = torch.tensor([0.0, 0.5, 1.0])
        agreement = torch.tensor([0.0, 0.5, 1.0])
        weights = bounded_novel_mass_weights(masses, agreement, floor=0.05)
        self.assertTrue(torch.allclose(weights, torch.tensor([0.05, 0.2875, 1.0])))
        self.assertGreaterEqual(float(weights.min()), 0.05)
        with self.assertRaises(ValueError):
            bounded_novel_mass_weights(masses, floor=1.0)

    def test_neighbor_support_downweights_isolated_novel_mass(self):
        features = torch.tensor(
            [
                [1.0, 0.0], [0.99, 0.01], [0.98, 0.02],
                [-1.0, 0.0], [-0.99, 0.01], [-0.98, -0.01],
            ]
        )
        masses = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
        weights = neighbor_novel_support_weights(features, masses, k=2)
        self.assertGreater(float(weights[:3].mean()), float(weights[3]))
        self.assertTrue(torch.isfinite(weights).all())

    def test_neighbor_support_handles_empty_and_singleton_batches(self):
        empty = neighbor_novel_support_weights(torch.empty(0, 4), torch.empty(0), k=3)
        single = neighbor_novel_support_weights(torch.ones(1, 4), torch.tensor([0.7]), k=3)
        self.assertEqual(empty.numel(), 0)
        self.assertTrue(torch.allclose(single, torch.tensor([0.7])))

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

    def test_unweighted_sinkhorn_keeps_loss_weights_but_disables_weighted_targets(self):
        first_features = torch.randn(5, 4, requires_grad=True)
        second_features = torch.randn(5, 4, requires_grad=True)
        first_logits = torch.tensor(
            [[4.0, 0.0, -1.0], [0.0, 3.0, -1.0], [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True
        )
        second_logits = (first_logits.detach() + 0.2 * torch.randn(5, 3)).requires_grad_()
        weights = torch.tensor([1.0, 0.8, 0.3, 0.1, 0.0])
        losses = joint_discovery_loss(
            first_features,
            second_features,
            first_logits,
            second_logits,
            confidence_threshold=0.0,
            sample_weights=weights,
            weighted_assignments=False,
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
