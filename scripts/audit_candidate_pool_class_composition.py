"""Audit classwise candidate-pool composition for a frozen paired comparison.

This is a retrospective diagnostic. Test labels are used only after the runs
finish to describe unknown-class coverage and candidate-pool composition. They
are never used for training, checkpoint selection, threshold calibration,
candidate selection, or method choice.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PAIRS = {
    2121: (
        "rejection_ce_decay_full_s2121_static_control_discover",
        "rejection_uncertainty_band_full_s2121",
    ),
    2122: (
        "rejection_ce_decay_full_s2122_static_control_discover",
        "rejection_uncertainty_band_full_s2122",
    ),
    2123: (
        "rejection_ce_decay_full_s2123_static_control_discover",
        "rejection_uncertainty_band_full_s2123",
    ),
    2124: (
        "rejection_ce_decay_full_s2124_static_control_discover",
        "rejection_uncertainty_band_full_s2124",
    ),
    2125: (
        "rejection_ce_decay_full_s2125_static_control_discover",
        "rejection_uncertainty_band_full_s2125",
    ),
    2127: (
        "rejection_ce_decay_full_s2127_static_control_discover",
        "rejection_uncertainty_band_full_s2127",
    ),
}


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(float(value) for value in values)
    position = (len(xs) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return xs[low]
    return xs[low] + (xs[high] - xs[low]) * (position - low)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    center = mean(values)
    return math.sqrt(mean([(value - center) ** 2 for value in values]))


def gini(values: list[int]) -> float:
    xs = sorted(float(value) for value in values if value >= 0)
    total = sum(xs)
    if not xs or total == 0:
        return 0.0
    n = len(xs)
    return sum((2 * index - n - 1) * value for index, value in enumerate(xs, 1)) / (
        n * total
    )


def load_run(runs_dir: Path, directory: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = runs_dir / directory
    detail = json.loads((root / "discovery_detail.json").read_text(encoding="utf-8"))
    report = json.loads((root / "discovery_report.json").read_text(encoding="utf-8"))
    return detail, report


def audit(detail: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    raw_labels = [int(value) for value in detail["raw_labels"]]
    true_known = [bool(value) for value in detail["true_known"]]
    pred_known = [bool(value) for value in detail["pred_known"]]
    scores = [float(value) for value in detail["score"]]
    threshold = float(report["score_distribution"]["threshold"])

    # The discovery evaluator marks pred_known=0 when the score is above the
    # calibrated threshold. Keep this check in the audit so a malformed or
    # mismatched output cannot silently be treated as a candidate pool.
    threshold_mask_mismatch = sum(
        int((score <= threshold) != predicted_known)
        for score, predicted_known in zip(scores, pred_known)
    )

    unknown_labels = sorted(
        {label for label, known in zip(raw_labels, true_known) if not known}
    )
    rows: list[dict[str, Any]] = []
    for label in unknown_labels:
        indices = [index for index, raw_label in enumerate(raw_labels) if raw_label == label]
        unknown_indices = [index for index in indices if not true_known[index]]
        rejected = sum(not pred_known[index] for index in unknown_indices)
        rows.append(
            {
                "label": label,
                "count": len(unknown_indices),
                "candidate_count": rejected,
                "coverage": rejected / len(unknown_indices),
                "score_q25": quantile([scores[index] for index in unknown_indices], 0.25),
                "score_median": quantile([scores[index] for index in unknown_indices], 0.50),
                "score_q75": quantile([scores[index] for index in unknown_indices], 0.75),
            }
        )

    coverages = [row["coverage"] for row in rows]
    candidate_counts = [row["candidate_count"] for row in rows]
    candidate_mask = [not value for value in pred_known]
    unknown_mask = [not value for value in true_known]
    unknown_candidates = sum(candidate and unknown for candidate, unknown in zip(candidate_mask, unknown_mask))
    known_candidates = sum(candidate and not unknown for candidate, unknown in zip(candidate_mask, unknown_mask))
    candidate_total = sum(candidate_mask)
    total_unknown = sum(unknown_mask)
    top_counts = sorted(candidate_counts, reverse=True)

    return {
        "threshold": threshold,
        "threshold_mask_mismatch": threshold_mask_mismatch,
        "unknown_count": total_unknown,
        "unknown_rejected": unknown_candidates,
        "unknown_rejection_rate": unknown_candidates / total_unknown,
        "candidate_count": candidate_total,
        "known_contamination_count": known_candidates,
        "candidate_purity": unknown_candidates / candidate_total if candidate_total else float("nan"),
        "unknown_candidate_recall": unknown_candidates / total_unknown,
        "class_count": len(rows),
        "classwise_coverage": {
            "min": min(coverages),
            "q25": quantile(coverages, 0.25),
            "median": quantile(coverages, 0.50),
            "q75": quantile(coverages, 0.75),
            "max": max(coverages),
            "std": std(coverages),
        },
        "classes_below_10pct": sum(value < 0.10 for value in coverages),
        "classes_below_20pct": sum(value < 0.20 for value in coverages),
        "classes_with_zero_candidates": sum(value == 0 for value in candidate_counts),
        "candidate_count_distribution": {
            "min": min(candidate_counts),
            "q25": quantile(candidate_counts, 0.25),
            "median": quantile(candidate_counts, 0.50),
            "q75": quantile(candidate_counts, 0.75),
            "max": max(candidate_counts),
            "gini": gini(candidate_counts),
            "top5_mass": sum(top_counts[:5]) / candidate_total if candidate_total else float("nan"),
            "top10_mass": sum(top_counts[:10]) / candidate_total if candidate_total else float("nan"),
        },
        "cluster_k": int(detail["cluster_k"]),
        "true_k": int(report.get("cluster_true_k", len(unknown_labels))),
        "cluster_k_abs_error": abs(int(detail["cluster_k"]) - int(report.get("cluster_true_k", len(unknown_labels)))),
        "auroc": float(report["auroc"]),
        "aupr": float(report["aupr"]),
        "known_accept_rate": float(report["known_accept_rate"]),
        "cluster_acc": float(report["cluster_acc"]),
        "cluster_nmi": float(report["cluster_nmi"]),
        "cluster_ari": float(report["cluster_ari"]),
        "classes": rows,
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {row["label"]: row for row in left["classes"]}
    right_rows = {row["label"]: row for row in right["classes"]}
    labels = sorted(set(left_rows) & set(right_rows))
    deltas = [right_rows[label]["coverage"] - left_rows[label]["coverage"] for label in labels]
    return {
        "labels_compared": len(labels),
        "mean_coverage_delta": mean(deltas),
        "median_coverage_delta": quantile(deltas, 0.50),
        "min_coverage_delta": min(deltas),
        "max_coverage_delta": max(deltas),
        "positive_classes": sum(delta > 0 for delta in deltas),
        "negative_classes": sum(delta < 0 for delta in deltas),
        "unchanged_classes": sum(delta == 0 for delta in deltas),
        "classes": [
            {
                "label": label,
                "baseline_coverage": left_rows[label]["coverage"],
                "uncertainty_coverage": right_rows[label]["coverage"],
                "coverage_delta": right_rows[label]["coverage"] - left_rows[label]["coverage"],
                "baseline_candidate_count": left_rows[label]["candidate_count"],
                "uncertainty_candidate_count": right_rows[label]["candidate_count"],
            }
            for label in labels
        ],
    }


def build_report(results: dict[str, Any]) -> str:
    lines = [
        "# Candidate-pool class-composition audit for uncertainty-band",
        "",
        "This is a post-hoc audit of six frozen paired runs. Test labels are used only after evaluation to describe unknown-class coverage and candidate-pool composition. They are not used for training, checkpoint selection, threshold calibration, candidate selection, or method choice.",
        "",
        "The audit separates three questions: whether unknown samples are rejected, whether the rejected pool is contaminated by known samples, and whether the rejected unknown samples are distributed across classes. This distinction is needed because higher pool purity does not by itself imply better new-class clustering.",
        "",
        "| split | arm | AUROC | unknown rejection | candidate purity | classwise coverage min/median/max | <20% classes | top-10 candidate mass | auto K (true K) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, arms in results.items():
        for arm_name in ("baseline", "uncertainty_band"):
            current = arms[arm_name]
            coverage = current["classwise_coverage"]
            concentration = current["candidate_count_distribution"]
            lines.append(
                f"| {seed} | {arm_name} | {current['auroc']:.4f} | "
                f"{current['unknown_rejection_rate']:.3%} | {current['candidate_purity']:.3%} | "
                f"{coverage['min']:.3%} / {coverage['median']:.3%} / {coverage['max']:.3%} | "
                f"{current['classes_below_20pct']} | {concentration['top10_mass']:.3%} | "
                f"{current['cluster_k']} ({current['true_k']}) |"
            )
        delta = arms["comparison"]
        lines.append(
            f"| {seed} | uncertainty - baseline | — | {arms['uncertainty_band']['unknown_rejection_rate'] - arms['baseline']['unknown_rejection_rate']:+.3%} | "
            f"{arms['uncertainty_band']['candidate_purity'] - arms['baseline']['candidate_purity']:+.3%} | "
            f"{delta['min_coverage_delta']:+.3%} / {delta['median_coverage_delta']:+.3%} / {delta['max_coverage_delta']:+.3%} | "
            f"{delta['positive_classes']} positive / {delta['negative_classes']} negative | "
            f"{arms['uncertainty_band']['candidate_count_distribution']['top10_mass'] - arms['baseline']['candidate_count_distribution']['top10_mass']:+.3%} | "
            f"{arms['uncertainty_band']['cluster_k_abs_error'] - arms['baseline']['cluster_k_abs_error']:+d} |"
        )

    all_baseline = [arms["baseline"] for arms in results.values()]
    all_uncertainty = [arms["uncertainty_band"] for arms in results.values()]
    mean_purity_delta = mean([u["candidate_purity"] - b["candidate_purity"] for b, u in zip(all_baseline, all_uncertainty)])
    mean_unknown_delta = mean([u["unknown_rejection_rate"] - b["unknown_rejection_rate"] for b, u in zip(all_baseline, all_uncertainty)])
    mean_median_coverage_delta = mean([
        u["classwise_coverage"]["median"] - b["classwise_coverage"]["median"]
        for b, u in zip(all_baseline, all_uncertainty)
    ])
    mean_k_error_delta = mean([u["cluster_k_abs_error"] - b["cluster_k_abs_error"] for b, u in zip(all_baseline, all_uncertainty)])
    lines += [
        "",
        "## Paired six-split summary",
        "",
        f"Across the six paired splits, the mean change in unknown rejection rate is {mean_unknown_delta:+.3%}, the mean change in candidate purity is {mean_purity_delta:+.3%}, and the mean change in the median per-class unknown coverage is {mean_median_coverage_delta:+.3%}. The mean change in automatic-K absolute error is {mean_k_error_delta:+.2f}.",
        "",
        "The classwise comparison is descriptive. It does not justify class-specific thresholds or weighting from test labels. If a future method targets class coverage, its rule must be derived from known-only validation information and evaluated with the same fixed operating point. The present audit is sufficient to decide whether a new experiment should target candidate quantity, classwise coverage, or downstream representation.",
        "",
        "Machine-readable output: `candidate_pool_class_composition_6split.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for seed, (baseline_dir, uncertainty_dir) in PAIRS.items():
        baseline_detail, baseline_report = load_run(args.runs_dir, baseline_dir)
        uncertainty_detail, uncertainty_report = load_run(args.runs_dir, uncertainty_dir)
        baseline = audit(baseline_detail, baseline_report)
        uncertainty = audit(uncertainty_detail, uncertainty_report)
        results[str(seed)] = {
            "baseline_run": baseline_dir,
            "uncertainty_run": uncertainty_dir,
            "baseline": baseline,
            "uncertainty_band": uncertainty,
            "comparison": compare(baseline, uncertainty),
        }

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.analysis_dir / "candidate_pool_class_composition_6split.json"
    out_md = args.analysis_dir / "candidate_pool_class_composition_6split.md"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(build_report(results), encoding="utf-8")
    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()
