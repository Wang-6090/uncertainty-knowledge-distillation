"""Audit detector ranking and fixed-known-coverage operating points.

This is a post-hoc diagnostic. Test labels are used only to stratify and
report results at fixed known-coverage operating points; they are never used
to fit a detector, select a model, or choose a production threshold.

Example:
    python scripts/audit_detector_operating_points.py `
      --run baseline=runs/..._detect `
      --run rejector=runs/..._rejector `
      --run fusion=runs/..._fusion075 `
      --output analysis/operating_points.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.metrics import compute_aupr, compute_auroc, compute_fpr95, compute_oscr
from novel_discovery.utils import save_json


def _score_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
    }


def _histogram_overlap(left: np.ndarray, right: np.ndarray) -> float:
    lower = float(min(left.min(), right.min()))
    upper = float(max(left.max(), right.max()))
    if upper - lower < 1e-12:
        return 1.0
    bins = np.linspace(lower, upper, 51)
    left_hist = np.histogram(left, bins=bins)[0].astype(float)
    right_hist = np.histogram(right, bins=bins)[0].astype(float)
    left_hist /= max(float(left_hist.sum()), 1.0)
    right_hist /= max(float(right_hist.sum()), 1.0)
    return float(np.minimum(left_hist, right_hist).sum())


def _operating_point(
    scores: np.ndarray,
    true_known: np.ndarray,
    pred_class: np.ndarray,
    true_label: np.ndarray,
    target_coverage: float,
) -> dict[str, float | int]:
    known = np.asarray(true_known).astype(bool)
    scores = np.asarray(scores, dtype=float)
    # Lower score means more likely known throughout the project.
    threshold = float(np.quantile(scores[known], target_coverage))
    accepted = scores <= threshold
    known_accept = accepted & known
    unknown_reject = (~accepted) & (~known)
    known_count = max(int(known.sum()), 1)
    unknown_count = max(int((~known).sum()), 1)
    correct = np.asarray(pred_class) == np.asarray(true_label)
    return {
        "target_known_coverage": float(target_coverage),
        "diagnostic_threshold_from_test_known": threshold,
        "known_accept_rate": float(known_accept.sum() / known_count),
        "unknown_reject_rate": float(unknown_reject.sum() / unknown_count),
        "known_rejected": int((known & ~accepted).sum()),
        "unknown_false_accept": int(((~known) & accepted).sum()),
        "known_class_correct_all_known": int((correct & known).sum()),
        "known_class_accuracy_all_known": float((correct & known).sum() / known_count),
        "accepted_correct_fraction_of_all_known": float(
            (correct & known_accept).sum() / known_count
        ),
        "known_class_accuracy_after_accept": float(
            (correct & known_accept).sum() / max(int(known_accept.sum()), 1)
        ),
        "oscr": compute_oscr(known, accepted, correct, scores),
    }


def _load_run(path: Path) -> dict:
    detail_path = path / "discovery_detail.json"
    if not detail_path.exists():
        raise FileNotFoundError(f"missing {detail_path}")
    with detail_path.open("r", encoding="utf-8") as handle:
        detail = json.load(handle)
    scores = np.asarray(detail["score"], dtype=float)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    pred_class = np.asarray(detail["pred_class"])
    true_label = np.asarray(detail["true_label"])
    if not (len(scores) == len(true_known) == len(pred_class) == len(true_label)):
        raise ValueError(f"inconsistent array lengths in {detail_path}")
    unknown_label = (~true_known).astype(int)
    report_path = path / "discovery_report.json"
    report = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    return {
        "path": str(path),
        "score_mode": detail.get("score_mode"),
        "scores": scores,
        "true_known": true_known,
        "pred_class": pred_class,
        "true_label": true_label,
        "unknown_label": unknown_label,
        "report": report,
    }


def audit(runs: list[tuple[str, Path]], coverages: list[float]) -> dict:
    output = {"protocol": {
        "note": (
            "Post-hoc test-label diagnostic. Test labels determine only the "
            "reported operating points; they are not used for fitting or selection."
        ),
        "score_direction": "lower scores are accepted as known",
        "coverages": coverages,
    }, "runs": {}}
    for label, path in runs:
        item = _load_run(path)
        scores = item["scores"]
        known = item["true_known"]
        unknown = ~known
        result = {
            "path": item["path"],
            "score_mode": item["score_mode"],
            "reported_metrics": {
                key: item["report"].get(key)
                for key in (
                    "auroc", "aupr", "fpr95", "oscr", "known_accept_rate",
                    "unknown_reject_rate", "known_class_accuracy_all_known",
                )
                if key in item["report"]
            },
            "metric_consistency": {
                "recomputed_known_class_accuracy_all_known": float(
                    (item["pred_class"] == item["true_label"])[known].mean()
                ),
                "recomputed_accepted_correct_fraction_of_all_known": float(
                    ((item["pred_class"] == item["true_label"]) & known & (scores <= np.quantile(scores[known], 0.95))).sum()
                    / max(int(known.sum()), 1)
                ),
                "historical_report_field_warning": (
                    "Older reports may have stored accepted-correct/all-known under "
                    "known_class_accuracy_all_known; rerun discover after the metric fix."
                ),
            },
            "ranking": {
                "auroc": compute_auroc(item["unknown_label"], scores),
                "aupr": compute_aupr(item["unknown_label"], scores),
                "fpr95": compute_fpr95(item["unknown_label"], scores),
                "known_score": _score_summary(scores[known]),
                "unknown_score": _score_summary(scores[unknown]),
                "histogram_overlap_0_to_1": _histogram_overlap(scores[known], scores[unknown]),
            },
            "operating_points": [
                _operating_point(
                    scores,
                    known,
                    item["pred_class"],
                    item["true_label"],
                    coverage,
                )
                for coverage in coverages
            ],
        }
        output["runs"][label] = result
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="A saved discover run; repeat for each detector to compare.",
    )
    parser.add_argument(
        "--coverage",
        action="append",
        type=float,
        default=None,
        help="Target known coverage; repeatable. Defaults to 0.90, 0.95, 0.97.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(args):
    runs = []
    for value in args.run:
        if "=" not in value:
            raise ValueError(f"--run must use LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label:
            raise ValueError("detector label cannot be empty")
        runs.append((label, Path(raw_path)))
    coverages = args.coverage or [0.90, 0.95, 0.97]
    if any(not 0.0 < coverage < 1.0 for coverage in coverages):
        raise ValueError("coverage values must be in (0, 1)")
    result = audit(runs, coverages)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved audit to {args.output}")


if __name__ == "__main__":
    main(parse_args())
