import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_teacher_uncertainty_alignment import (
    classwise_alignment,
    minibatch_weight_diagnostics,
)


def test_classwise_alignment_separates_within_class_and_between_class_signals():
    values = {
        "labels": np.repeat(np.arange(3), 4),
        "uncertainty": np.asarray(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5, 0.6]
        ),
        "confidence": np.asarray(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.7, 0.6, 0.5, 0.4]
        ),
        "error": np.asarray(
            [False, False, False, True, False, True, True, True, False, False, True, True]
        ),
    }

    result = classwise_alignment(values, [12, 28, 49])

    assert result["class_count"] == 3
    assert result["within_class_error_uncertainty_valid_class_count"] == 3
    assert result["within_class_error_uncertainty_macro_auroc"] == 1.0
    assert result["spearman_class_error_rate_vs_mean_uncertainty"] == 1.0
    assert [row["raw_class_id"] for row in result["classes"]] == [12, 28, 49]
    assert [row["teacher_error_rate"] for row in result["classes"]] == [0.25, 0.75, 0.5]


def test_classwise_alignment_handles_classes_without_both_error_outcomes():
    values = {
        "labels": np.asarray([0, 0, 1, 1]),
        "uncertainty": np.asarray([0.1, 0.2, 0.7, 0.8]),
        "confidence": np.asarray([0.9, 0.8, 0.3, 0.2]),
        "error": np.asarray([False, False, True, True]),
    }

    result = classwise_alignment(values)

    assert result["class_count"] == 2
    assert result["within_class_error_uncertainty_valid_class_count"] == 0
    assert result["within_class_error_uncertainty_macro_auroc"] is None


def test_class_balanced_weights_remove_between_class_only_variation():
    result = minibatch_weight_diagnostics(
        labels=np.asarray([0, 0, 1, 1]),
        uncertainty=np.asarray([0.1, 0.1, 0.8, 0.8]),
        batch_size=4,
        repeats=1,
        seed=3,
    )

    assert result["global_uncertainty"]["mean_batch_weight_std"] > 0.0
    assert result["within_class_uncertainty"]["mean_batch_weight_std"] == 0.0
    assert np.isclose(result["within_class_uncertainty"]["mean_batch_ess_fraction"], 1.0)
    assert result["singleton_class_exposure_fraction"] == 0.0


def test_class_balanced_weights_keep_within_class_signal_and_count_singletons():
    result = minibatch_weight_diagnostics(
        labels=np.asarray([0, 0, 1]),
        uncertainty=np.asarray([0.1, 0.9, 0.4]),
        batch_size=3,
        repeats=1,
        seed=3,
    )

    assert result["within_class_uncertainty"]["mean_batch_weight_std"] > 0.0
    assert result["within_class_uncertainty"]["mean_batch_ess_fraction"] < 1.0
    assert result["singleton_class_exposure_fraction"] == 1 / 3
