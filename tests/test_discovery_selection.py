import unittest

import numpy as np

from novel_discovery.discovery_selection import (
    EMASelector,
    auto_k,
    evaluate_selection,
    filter_by_neighbor_agreement,
    knn_agreement,
    relation_affinity,
    risk_score,
    select_candidates,
)


class DiscoverySelectionTest(unittest.TestCase):
    def test_empty_and_zero_ratio(self):
        mask, weights = select_candidates([], ratio=0.5)
        self.assertEqual(mask.size, 0)
        mask, weights = select_candidates([0.1, 0.2], ratio=0)
        self.assertFalse(mask.any())
        self.assertTrue(np.all(weights == 0))

    def test_soft_selection_is_bounded(self):
        mask, weights = select_candidates([0.1, 0.9, 0.2], ratio=0.5, soft=True, weight_floor=0.2)
        self.assertEqual(mask.sum(), 2)
        self.assertGreaterEqual(weights[mask].min(), 0.2)
        self.assertLessEqual(weights.max(), 1.0)

    def test_knn_clamps_k_larger_than_batch(self):
        agreement = knn_agreement([True, False], np.eye(2), k=20)
        np.testing.assert_allclose(agreement, [0.0, 1.0])

    def test_auto_k_handles_tiny_pool(self):
        self.assertEqual(auto_k(np.empty((0, 2)))[0], None)
        self.assertEqual(auto_k(np.ones((2, 2)))[0], 1)

    def test_consensus_score(self):
        score = risk_score([0.0, 1.0], msp=[0.0, 1.0], mode="consensus")
        np.testing.assert_allclose(score, [0.0, 1.0])

    def test_ema_selector_smooths_values(self):
        ema = EMASelector(decay=0.9)
        np.testing.assert_allclose(ema.update([0.0, 0.0]), [0.0, 0.0])
        np.testing.assert_allclose(ema.update([1.0, 1.0]), [0.1, 0.1])

    def test_neighbor_gate_and_evaluation_report(self):
        features = np.array([[1., 0.], [1., .1], [0., 1.], [0., .9]])
        mask = np.array([True, True, False, False])
        filtered = filter_by_neighbor_agreement(mask, features, k=3, min_agreement=0.3)
        self.assertTrue(filtered[:2].all())
        report = evaluate_selection([False, False, True, True], [0.1, 0.2, 0.8, 0.9], mask)
        self.assertAlmostEqual(report["candidate_purity"], 0.0)
        self.assertAlmostEqual(report["unknown_reject_rate"], 0.0)

    def test_relation_affinity_is_row_normalized_and_bounded(self):
        affinity = relation_affinity(np.eye(4), k=10)
        self.assertEqual(affinity.shape, (4, 4))
        np.testing.assert_allclose(affinity.sum(axis=1), np.ones(4))
        self.assertTrue(np.all(np.diag(affinity) == 0.0))


if __name__ == "__main__":
    unittest.main()
