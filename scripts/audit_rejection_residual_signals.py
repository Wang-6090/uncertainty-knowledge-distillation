"""Audit whether saved signals add information beyond the formal rejector.

This is a frozen diagnostic.  It never fits weights or thresholds and never
uses test labels for model selection.  Test known/unknown labels are used only
to report AUROC and error-cohort statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2121, 2124, 2125, 2127)

SIGNALS = {
    "feature_rejector_score": ("higher", "formal rejector score"),
    "entropy": ("higher", "predictive entropy"),
    "head_uncertainty": ("higher", "uncertainty head"),
    "epistemic": ("higher", "epistemic uncertainty"),
    "expected_entropy": ("higher", "expected entropy"),
    "knn_predicted_class_distance": ("higher", "predicted-class KNN distance"),
    "knn_predicted_class_relative_distance": ("higher", "predicted-class relative KNN distance"),
    "knn_predicted_class_support": ("lower", "predicted-class neighborhood support"),
    "knn_min_class_distance": ("higher", "nearest-class KNN distance"),
    "knn_min_class_relative_distance": ("higher", "nearest-class relative KNN distance"),
    "known_support_score": ("higher", "known-support score"),
    "known_conformal_score": ("higher", "known conformal score"),
    "proto_dist": ("higher", "prototype distance"),
    "mahalanobis": ("higher", "Mahalanobis distance"),
    "vim_residual": ("higher", "ViM residual"),
}


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, score))


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y).size < 2:
        return None
    return float(average_precision_score(y, score))


def _stats(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "count": int(len(y)),
        "auroc": _safe_auc(y, score),
        "aupr": _safe_ap(y, score),
    }


def audit_split(path: Path, threshold_path: Path) -> dict:
    detail = json.loads(path.read_text(encoding="utf-8"))
    calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = float(calibration["calibrated_threshold"])
    formal = np.asarray(detail["feature_rejector_score"], dtype=np.float64)
    y = 1 - np.asarray(detail["true_known"], dtype=np.int8)
    accepted = formal <= threshold
    rejected = ~accepted
    unknown = y == 1
    known = ~unknown
    unknown_false_accept = unknown & accepted
    known_true_accept = known & accepted
    unknown_true_reject = unknown & rejected
    known_false_reject = known & rejected

    signals = {}
    for key, (direction, description) in SIGNALS.items():
        raw = np.asarray(detail[key], dtype=np.float64)
        score = raw if direction == "higher" else -raw
        signals[key] = {
            "description": description,
            "direction": direction,
            "all": _stats(y, score),
            "formal_accepted_cohort": _stats(
                y[accepted], score[accepted]
            ),
            "formal_rejected_cohort": _stats(
                y[rejected], score[rejected]
            ),
            "mean_unknown_false_accept": float(score[unknown_false_accept].mean())
            if unknown_false_accept.any() else None,
            "mean_known_true_accept": float(score[known_true_accept].mean())
            if known_true_accept.any() else None,
            "mean_unknown_true_reject": float(score[unknown_true_reject].mean())
            if unknown_true_reject.any() else None,
            "mean_known_false_reject": float(score[known_false_reject].mean())
            if known_false_reject.any() else None,
            "pearson_with_formal": float(np.corrcoef(formal, score)[0, 1])
            if np.std(score) > 1e-12 else None,
        }

    return {
        "protocol": {
            "detail": str(path),
            "threshold_report": str(threshold_path),
            "threshold": threshold,
            "test_labels_used_only_for_final_metrics": True,
            "weights_fitted": False,
            "threshold_changed": False,
        },
        "cohorts": {
            "total": int(len(y)),
            "known": int(known.sum()),
            "unknown": int(unknown.sum()),
            "formal_accepted": int(accepted.sum()),
            "formal_rejected": int(rejected.sum()),
            "unknown_false_accept": int(unknown_false_accept.sum()),
            "known_true_accept": int(known_true_accept.sum()),
            "unknown_true_reject": int(unknown_true_reject.sum()),
            "known_false_reject": int(known_false_reject.sum()),
        },
        "signals": signals,
    }


def _mean_rows(reports: dict) -> dict:
    keys = next(iter(reports.values()))["signals"]
    result = {}
    for key in keys:
        rows = [reports[name]["signals"][key] for name in reports]
        result[key] = {
            "description": rows[0]["description"],
            "direction": rows[0]["direction"],
            "all_auroc": float(np.mean([r["all"]["auroc"] for r in rows])),
            "accepted_cohort_auroc": float(np.mean([
                r["formal_accepted_cohort"]["auroc"] for r in rows
                if r["formal_accepted_cohort"]["auroc"] is not None
            ])),
            "rejected_cohort_auroc": float(np.mean([
                r["formal_rejected_cohort"]["auroc"] for r in rows
                if r["formal_rejected_cohort"]["auroc"] is not None
            ])),
            "mean_pearson_with_formal": float(np.mean([
                r["pearson_with_formal"] for r in rows
                if r["pearson_with_formal"] is not None
            ])),
        }
    return result


def render_markdown(payload: dict) -> str:
    lines = [
        "# Residual rejection-signal audit",
        "",
        "Frozen diagnostic on the four formal semantic-hard discovery runs.",
        "The formal threshold and score remain unchanged. Test labels are used only for final cohort metrics; no score weights or thresholds are fitted.",
        "",
        "## Cohort interpretation",
        "",
        "The main diagnostic is the formal-accepted cohort: samples that the current rejector would accept as known. Its AUROC asks whether an existing signal can still separate unknown false accepts from known true accepts.",
        "",
        "## Four-split signal summary",
        "",
        "| signal | all AUROC | accepted-cohort AUROC | rejected-cohort AUROC | mean Pearson with formal score |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in sorted(payload["mean_signals"].items(), key=lambda item: -item[1]["accepted_cohort_auroc"]):
        lines.append(
            f"| {key} | {row['all_auroc']:.4f} | {row['accepted_cohort_auroc']:.4f} | {row['rejected_cohort_auroc']:.4f} | {row['mean_pearson_with_formal']:.4f} |"
        )
    lines.extend(["", "## Per-split formal cohorts", "", "| split | threshold | unknown false accepts | known true accepts | unknown true rejects | known false rejects |", "|---|---:|---:|---:|---:|---:|"])
    for name, report in payload["reports"].items():
        c = report["cohorts"]
        lines.append(
            f"| {name} | {report['protocol']['threshold']:.4f} | {c['unknown_false_accept']} | {c['known_true_accept']} | {c['unknown_true_reject']} | {c['known_false_reject']} |"
        )
    lines.extend(["", "## Decision rule for the next experiment", "", "A signal is eligible for a fixed threshold-band post-processing pilot only if its accepted-cohort AUROC is consistently above 0.5 in all four splits and its direction is coherent. This audit alone does not promote a method; it only identifies whether a pre-specified, calibration-only pilot is justified.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(ROOT / "runs"))
    parser.add_argument("--output", default=str(ROOT / "analysis" / "rejection_residual_signal_audit_4split.json"))
    parser.add_argument("--markdown", default=str(ROOT / "analysis" / "rejection_residual_signal_audit_4split.md"))
    args = parser.parse_args()
    run_root = Path(args.run_root)
    reports = {}
    for seed in SEEDS:
        detail = run_root / f"rejection_ce_decay_full_s{seed}_static_control_discover" / "discovery_detail.json"
        threshold = run_root / f"rejection_ce_decay_full_s{seed}_static_control_discover" / "calibration_report.json"
        print(f"[audit] s{seed}", flush=True)
        reports[f"s{seed}"] = audit_split(detail, threshold)
    payload = {"reports": reports, "mean_signals": _mean_rows(reports)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(render_markdown(payload), encoding="utf-8")
    print(f"saved {output}")
    print(f"saved {args.markdown}")


if __name__ == "__main__":
    main()
