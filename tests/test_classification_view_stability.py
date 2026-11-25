import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_classification_view_stability import summarize_views


def test_view_summary_detects_prediction_flip_and_groups_correctness():
    values = {
        "features_a": np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "features_b": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        "logits_a": np.asarray([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]]),
        "logits_b": np.asarray([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]),
        "labels": np.asarray([0, 0, 1]),
    }

    result = summarize_views(values)

    assert result["samples"] == 3
    assert result["prediction_flip_rate"] == 1 / 3
    assert result["groups"]["both_views_correct"]["count"] == 2
    assert result["groups"]["at_least_one_view_wrong"]["count"] == 1
    assert result["groups"]["both_views_correct"]["feature_cosine_mean"] == 1.0


def test_view_summary_handles_all_wrong_group_without_nan():
    values = {
        "features_a": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "features_b": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "logits_a": np.asarray([[3.0, 0.0], [0.0, 3.0]]),
        "logits_b": np.asarray([[3.0, 0.0], [0.0, 3.0]]),
        "labels": np.asarray([1, 0]),
    }

    result = summarize_views(values)

    assert result["groups"]["both_views_correct"]["count"] == 0
    assert result["groups"]["both_views_correct"]["feature_cosine_mean"] is None
    assert np.isfinite(result["groups"]["at_least_one_view_wrong"]["logit_js_mean"])
