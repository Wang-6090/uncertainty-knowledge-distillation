"""Audit validation-calibrated, threshold-local uncertainty re-ranking.

The formal rejector score is kept unchanged away from a narrow band around its
known-validation threshold.  Inside the band, head uncertainty is added as a
small, validation-calibrated residual.  Band width and residual weight are
fixed before reading test labels.  This script evaluates detection and
candidate-pool quality only; it does not train or alter any checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2121, 2124, 2125, 2127)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile_threshold(scores: np.ndarray, target: float) -> float:
    # Higher score means more likely unknown.  This is the same orientation as
    # the formal feature-rejector score.
    return float(np.quantile(scores, target, method="linear"))


def binary_metrics(scores: np.ndarray, known_mask: np.ndarray, threshold: float) -> dict:
    unknown_mask = ~known_mask
    pred_unknown = scores > threshold
    return {
        "auroc": float(roc_auc_score(unknown_mask.astype(int), scores)),
        "aupr": float(average_precision_score(unknown_mask.astype(int), scores)),
        "threshold": float(threshold),
        "known_accept_rate": float((~pred_unknown[known_mask]).mean()),
        "unknown_reject_rate": float(pred_unknown[unknown_mask].mean()),
        "unknown_false_accept": int((~pred_unknown[unknown_mask]).sum()),
        "known_rejected": int(pred_unknown[known_mask].sum()),
        "candidate_count": int(pred_unknown.sum()),
        "candidate_unknown_count": int((pred_unknown & unknown_mask).sum()),
        "candidate_coverage": float((pred_unknown & unknown_mask).sum() / unknown_mask.sum()),
        "candidate_purity": float((pred_unknown & unknown_mask).sum() / max(1, pred_unknown.sum())),
    }


def evaluate_case(seed: int, band_multiplier: float, alpha: float) -> dict:
    run = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control_discover"
    export = ROOT / "runs" / f"rejection_boundary_calibration_s{seed}" / "calibration_scores.npz"
    if not export.is_file():
        raise FileNotFoundError(export)
    detail = read_json(run / "discovery_detail.json")
    report = read_json(run / "discovery_report.json")
    with np.load(export, allow_pickle=False) as payload:
        val_score = np.asarray(payload["score"], dtype=np.float64)
        val_unc = np.asarray(payload["validation_head_uncertainty"], dtype=np.float64)
        baseline_threshold = float(payload["calibrated_threshold"])

    test_score = np.asarray(detail["score"], dtype=np.float64)
    test_unc = np.asarray(detail["head_uncertainty"], dtype=np.float64)
    known_mask = np.asarray(detail["true_known"], dtype=bool)
    if not (val_score.shape == val_unc.shape and test_score.shape == test_unc.shape):
        raise ValueError(f"unaligned arrays for seed={seed}")
    if not np.isfinite(val_unc).all() or not np.isfinite(test_unc).all():
        raise ValueError(f"non-finite uncertainty for seed={seed}")

    unc_mean = float(val_unc.mean())
    unc_std = max(float(val_unc.std(ddof=1)), 1e-6)
    val_z = (val_unc - unc_mean) / unc_std
    test_z = (test_unc - unc_mean) / unc_std
    score_std = max(float(val_score.std(ddof=1)), 1e-6)
    band = band_multiplier * score_std

    val_boundary = np.abs(val_score - baseline_threshold) <= band
    test_boundary = np.abs(test_score - baseline_threshold) <= band
    val_modified = val_score.copy()
    test_modified = test_score.copy()
    val_modified[val_boundary] += alpha * score_std * val_z[val_boundary]
    test_modified[test_boundary] += alpha * score_std * test_z[test_boundary]
    modified_threshold = percentile_threshold(val_modified, 0.95)

    baseline = binary_metrics(test_score, known_mask, baseline_threshold)
    modified = binary_metrics(test_modified, known_mask, modified_threshold)
    baseline.update({"source_report_auroc": float(report["auroc"]), "source_report_unknown_reject_rate": float(report["unknown_reject_rate"])})
    delta = {
        "auroc": modified["auroc"] - baseline["auroc"],
        "aupr": modified["aupr"] - baseline["aupr"],
        "unknown_reject_rate": modified["unknown_reject_rate"] - baseline["unknown_reject_rate"],
        "known_accept_rate": modified["known_accept_rate"] - baseline["known_accept_rate"],
        "unknown_false_accept": modified["unknown_false_accept"] - baseline["unknown_false_accept"],
        "candidate_purity": modified["candidate_purity"] - baseline["candidate_purity"],
    }
    changed = (test_score > baseline_threshold) != (test_modified > modified_threshold)
    return {
        "seed": seed,
        "band_multiplier": band_multiplier,
        "alpha": alpha,
        "calibration": {
            "baseline_threshold": baseline_threshold,
            "modified_threshold": modified_threshold,
            "validation_sample_count": int(len(val_score)),
            "validation_boundary_count": int(val_boundary.sum()),
            "test_boundary_count": int(test_boundary.sum()),
            "validation_score_std": score_std,
            "uncertainty_mean": unc_mean,
            "uncertainty_std": unc_std,
        },
        "baseline": baseline,
        "modified": modified,
        "delta_modified_minus_baseline": delta,
        "changed_test_decisions": int(changed.sum()),
        "changed_test_decision_rate": float(changed.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bands", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "rejection_uncertainty_band_4split.json")
    args = parser.parse_args()

    rows = []
    for band in args.bands:
        for alpha in args.alphas:
            for seed in args.seeds:
                rows.append(evaluate_case(seed, band, alpha))
    result = {
        "question": "Does validation-calibrated uncertainty help only at the formal rejection boundary?",
        "protocol": "frozen formal checkpoints; band and alpha fixed from known validation only; test labels used for reporting only",
        "seeds": list(args.seeds),
        "bands_in_validation_score_std": list(args.bands),
        "alphas_in_validation_score_std": list(args.alphas),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
