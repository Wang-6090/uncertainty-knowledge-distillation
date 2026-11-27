import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_teacher_ce_rank_complementarity import summarize_ranks, true_class_ranks


def test_true_class_rank_and_probability_are_exact_on_tie_free_logits():
    logits = np.array([[0.1, 0.9, 0.2], [2.0, 1.0, 0.0]], dtype=np.float64)
    labels = np.array([0, 2])

    ranks, probabilities = true_class_ranks(logits, labels)

    assert ranks.tolist() == [3, 3]
    assert probabilities[0] == pytest.approx(np.exp(0.1) / np.exp(logits[0]).sum())
    assert probabilities[1] == pytest.approx(np.exp(0.0) / np.exp(logits[1]).sum())


def test_rank_summary_reports_reciprocal_and_top_k_rates():
    summary = summarize_ranks(np.array([1, 3, 6]), np.array([0.7, 0.2, 0.1]))

    assert summary["mean_true_class_rank"] == pytest.approx(10 / 3)
    assert summary["mean_reciprocal_rank"] == pytest.approx((1 + 1 / 3 + 1 / 6) / 3)
    assert summary["top1_rate"] == pytest.approx(1 / 3)
    assert summary["top5_rate"] == pytest.approx(2 / 3)
    assert summary["top10_rate"] == 1.0


@pytest.mark.parametrize(
    ("logits", "labels"),
    [
        (np.zeros((2, 3)), np.array([0])),
        (np.array([[np.nan, 0.0]]), np.array([0])),
        (np.array([[0.0, 1.0]]), np.array([2])),
    ],
)
def test_true_class_rank_rejects_invalid_inputs(logits, labels):
    with pytest.raises(ValueError):
        true_class_ranks(logits, labels)
