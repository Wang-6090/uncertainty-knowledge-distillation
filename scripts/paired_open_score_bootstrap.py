"""Retrospective paired bootstrap for two open-set test-score runs.

Test labels are used only for post-hoc confidence intervals. This tool must not
be used to choose a model, score, or hyperparameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def read_run(path: Path):
    detail = json.loads((path / "discovery_detail.json").read_text(encoding="utf-8"))
    known = np.asarray(detail["true_known"], dtype=bool)
    scores = np.asarray(detail["score"], dtype=np.float64)
    labels = np.asarray(detail["raw_labels"], dtype=np.int64)
    true_label = np.asarray(detail["true_label"], dtype=np.int64)
    pred = np.asarray(detail["pred_class"], dtype=np.int64)
    return known, scores, labels, true_label, pred


def metrics(known, scores, true_label, pred):
    threshold = float(np.quantile(scores[known], 0.95, method="higher"))
    accepted = scores <= threshold
    return {
        "auroc": float(roc_auc_score(~known, scores)),
        "exact95_known_threshold": threshold,
        "known_acceptance": float(accepted[known].mean()),
        "unknown_rejection_at_exact95_known_coverage": float((~accepted[~known]).mean()),
        "accepted_known_classification_accuracy": float((pred[known & accepted] == true_label[known & accepted]).mean()),
        "all_known_classification_accuracy": float((pred[known] == true_label[known]).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    base = read_run(args.baseline)
    treatment = read_run(args.treatment)
    for name, left, right in zip(("true_known", "raw_labels", "true_label"), (base[0], base[2], base[3]), (treatment[0], treatment[2], treatment[3])):
        if not np.array_equal(left, right):
            raise ValueError(f"runs do not have identical ordered {name}")

    known, base_scores, raw_labels, true_label, base_pred = base
    _, treatment_scores, _, _, treatment_pred = treatment
    if base_scores.shape != treatment_scores.shape:
        raise ValueError("score arrays must have identical shapes")
    base_metrics = metrics(known, base_scores, true_label, base_pred)
    treatment_metrics = metrics(known, treatment_scores, true_label, treatment_pred)

    known_indices = np.flatnonzero(known)
    unknown_indices = np.flatnonzero(~known)
    rng = np.random.default_rng(args.seed)
    differences = {"auroc": [], "unknown_rejection_at_exact95_known_coverage": []}
    for _ in range(args.bootstrap):
        sample = np.concatenate(
            (
                rng.choice(known_indices, size=len(known_indices), replace=True),
                rng.choice(unknown_indices, size=len(unknown_indices), replace=True),
            )
        )
        sampled_known = known[sample]
        b = metrics(sampled_known, base_scores[sample], true_label[sample], base_pred[sample])
        t = metrics(sampled_known, treatment_scores[sample], true_label[sample], treatment_pred[sample])
        for key in differences:
            differences[key].append(t[key] - b[key])

    intervals = {
        key: {
            "mean_difference_treatment_minus_baseline": float(np.mean(values)),
            "percentile_95_ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "fraction_bootstraps_treatment_better": float(np.mean(np.asarray(values) > 0)),
        }
        for key, values in differences.items()
    }
    result = {
        "warning": "Retrospective test-label diagnostic only; not for model or hyperparameter selection.",
        "baseline": base_metrics,
        "treatment": treatment_metrics,
        "paired_bootstrap": {"replicates": args.bootstrap, "seed": args.seed, "intervals": intervals},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
