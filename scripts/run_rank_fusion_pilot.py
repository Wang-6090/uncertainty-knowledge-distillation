"""Offline pilot for a continuous rank calibrated uncertainty fusion.

This script consumes frozen discovery artifacts only.  It never fits on test
labels: the two empirical CDFs and the 95% operating threshold are fitted
from the saved known validation arrays.  Test labels are used only to report
the final evaluation metrics.

The pilot is intentionally limited to ranking and candidate selection.  A
full discovery run is justified only if this stage removes the conjunction
calibration failure without losing the raw detector's ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.metrics import (
    compute_aupr,
    compute_auroc,
    compute_fpr95,
    compute_oscr,
    open_set_confusion,
)


SEEDS = (42, 123, 3407)
ARMS = ("A", "B", "C")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def empirical_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values to their mid-rank in a known-only reference sample."""
    values = np.asarray(values, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if values.size == 0 or reference.size == 0:
        raise ValueError("rank calibration requires non-empty arrays")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("rank calibration requires finite arrays")
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left.astype(float) + right.astype(float)) / (2.0 * len(ordered))


def evaluate(score: np.ndarray, detail: dict[str, Any], threshold: float) -> dict[str, Any]:
    score = np.asarray(score, dtype=float).reshape(-1)
    known = np.asarray(detail["true_known"], dtype=bool)
    labels = np.asarray(detail["true_label"], dtype=int)
    predicted = np.asarray(detail["pred_class"], dtype=int)
    if len(score) != len(known):
        raise ValueError("score and discovery detail lengths differ")
    predicted_known = score <= float(threshold)
    open_labels = (~known).astype(int)
    class_correct = predicted == labels
    result = {
        "auroc": compute_auroc(open_labels, score),
        "aupr": compute_aupr(open_labels, score),
        "fpr95": compute_fpr95(open_labels, score),
        "oscr": compute_oscr(known, predicted_known, class_correct, score),
        "known_class_accuracy_all_known": float(
            np.mean(class_correct[known]) if np.any(known) else np.nan
        ),
        "candidate_count": int(np.sum(~predicted_known)),
        "candidate_unknown_recall": float(
            np.mean((~predicted_known)[~known]) if np.any(~known) else np.nan
        ),
        "candidate_purity": float(
            np.mean((~known)[~predicted_known]) if np.any(~predicted_known) else np.nan
        ),
        "threshold": float(threshold),
    }
    result.update(open_set_confusion(known, predicted_known))
    return result


def run_one(run_dir: Path) -> dict[str, Any]:
    detail = _read_json(run_dir / "discovery_detail.json")
    calibration = np.load(run_dir / "calibration_scores.npz")
    rejector = np.load(run_dir / "feature_rejector_scores.npz")
    validation_feature = calibration["validation_feature_rejector_score"]
    validation_uncertainty = calibration["validation_head_uncertainty"]
    test_feature = rejector["test_feature_rejector_score"]
    test_uncertainty = np.asarray(detail["head_uncertainty"], dtype=float)

    validation_score = 0.5 * empirical_rank(
        validation_feature, validation_feature
    ) + 0.5 * empirical_rank(validation_uncertainty, validation_uncertainty)
    test_score = 0.5 * empirical_rank(
        test_feature, validation_feature
    ) + 0.5 * empirical_rank(test_uncertainty, validation_uncertainty)
    threshold = float(np.quantile(validation_score, 0.95, method="linear"))
    result = evaluate(test_score, detail, threshold)
    result["calibration_known_accept_rate"] = float(np.mean(validation_score <= threshold))
    result["calibration_known_coverage_gap"] = result["calibration_known_accept_rate"] - 0.95
    result["reference_protocol"] = {
        "calibration_source": "known_validation_only",
        "test_labels_used_for_score_or_threshold": False,
        "feature_component": "empirical_midrank(feature_rejector_score)",
        "uncertainty_component": "empirical_midrank(raw_head_uncertainty)",
        "combination": "0.5 * feature_rank + 0.5 * uncertainty_rank",
        "threshold_policy": "known_validation_95_percent_coverage",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="formal_conjunction_audit_20261121",
        help="Frozen audit tag whose run directories are used as input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis" / "rank_fusion_pilot_20261121.json",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = ROOT / "runs" / f"{args.tag}_s{seed}_{arm}_feature_rejector"
            if not run_dir.is_dir():
                raise FileNotFoundError(run_dir)
            results[f"{arm}:s{seed}"] = run_one(run_dir)

    payload = {
        "protocol": "continuous_rank_fusion_frozen_pilot_20261121",
        "question": "Can rank-preserving fusion avoid the isotonic conjunction collapse?",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
