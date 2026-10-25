import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_shared_rejection_fusion_fresh import evaluate_pair


METRICS = {
    "auroc": 0.8,
    "aupr": 0.7,
    "fpr95": 0.6,
    "oscr": 0.59,
    "unknown_reject_rate": 0.25,
    "known_accept_rate": 0.95,
    "known_class_accuracy_after_accept": 0.72,
}


def _write_report(path, score_mode, **protocol_changes):
    protocol = {
        "active": True,
        "score_mode": score_mode,
        "discovery_pool_mode": "mixed",
        "training": "nnpu",
        "pool_selection": "all",
        "strict_mixed": False,
        "hard_labels_used": False,
        "interpretation": "unlabeled_mixed_pu",
        "test_labels_used_for_training_or_calibration": False,
        **protocol_changes,
    }
    report = {"threshold_policy": "known_val_known_coverage", **METRICS}
    report["rejector_protocol"] = protocol
    path.write_text(json.dumps(report), encoding="utf-8")


def test_fusion_pair_requires_identical_protocol_except_score_mode(tmp_path):
    control = tmp_path / "control.json"
    treatment = tmp_path / "treatment.json"
    _write_report(control, "feature_rejector")
    _write_report(treatment, "feature_rejector_uncertainty_fusion")

    result = evaluate_pair(1, control, treatment)

    assert result["protocol_match_except_score_mode"] is True
    assert result["threshold_policy_match"] is True


def test_fusion_pair_rejects_hidden_protocol_difference(tmp_path):
    control = tmp_path / "control.json"
    treatment = tmp_path / "treatment.json"
    _write_report(control, "feature_rejector")
    _write_report(
        treatment,
        "feature_rejector_uncertainty_fusion",
        training="soft_pu",
    )

    with pytest.raises(ValueError, match="protocols differ"):
        evaluate_pair(1, control, treatment)
