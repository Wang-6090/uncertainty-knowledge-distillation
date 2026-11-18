"""Post-hoc attribution of candidate-pool concentration in uncertainty-band.

This analysis reads the frozen six-split class-composition audit. Test labels
are used only after both paired runs have finished, to describe classwise
coverage and candidate-pool composition. No result here is used to train,
calibrate, select a checkpoint, choose a method, or set a new threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    center = mean(values)
    return math.sqrt(mean([(value - center) ** 2 for value in values]))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x_mean, y_mean = mean(xs), mean(ys)
    dx = [value - x_mean for value in xs]
    dy = [value - y_mean for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            result[order[position]] = average_rank
        index = end + 1
    return result


def correlation(xs: list[float], ys: list[float]) -> dict[str, float]:
    return {
        "pearson": pearson(xs, ys),
        "spearman": pearson(ranks(xs), ranks(ys)),
    }


def analyze_split(split: str, arms: dict[str, Any]) -> dict[str, Any]:
    baseline = arms["baseline"]
    treatment = arms["uncertainty_band"]
    baseline_rows = {row["label"]: row for row in baseline["classes"]}
    treatment_rows = {row["label"]: row for row in treatment["classes"]}
    labels = sorted(set(baseline_rows) & set(treatment_rows))

    baseline_coverage = [baseline_rows[label]["coverage"] for label in labels]
    coverage_delta = [
        treatment_rows[label]["coverage"] - baseline_rows[label]["coverage"]
        for label in labels
    ]
    baseline_counts = [baseline_rows[label]["candidate_count"] for label in labels]
    count_delta = [
        treatment_rows[label]["candidate_count"] - baseline_rows[label]["candidate_count"]
        for label in labels
    ]

    baseline_top10 = set(
        sorted(labels, key=lambda label: baseline_rows[label]["candidate_count"], reverse=True)[:10]
    )
    positive_gain = [max(delta, 0) for delta in count_delta]
    positive_gain_total = sum(positive_gain)
    top10_positive_gain = sum(
        gain for label, gain in zip(labels, positive_gain) if label in baseline_top10
    )

    return {
        "split": split,
        "labels_compared": len(labels),
        "coverage_delta_correlation": correlation(baseline_coverage, coverage_delta),
        "candidate_count_delta_correlation": correlation(baseline_counts, count_delta),
        "positive_class_count": sum(delta > 0 for delta in count_delta),
        "negative_class_count": sum(delta < 0 for delta in count_delta),
        "unchanged_class_count": sum(delta == 0 for delta in count_delta),
        "total_candidate_delta": sum(count_delta),
        "positive_candidate_gain": positive_gain_total,
        "negative_candidate_loss": sum(-min(delta, 0) for delta in count_delta),
        "positive_gain_share_in_baseline_top10": (
            top10_positive_gain / positive_gain_total if positive_gain_total else float("nan")
        ),
        "baseline_top10_labels": sorted(baseline_top10),
        "delta_top10_mass": (
            treatment["candidate_count_distribution"]["top10_mass"]
            - baseline["candidate_count_distribution"]["top10_mass"]
        ),
        "delta_gini": (
            treatment["candidate_count_distribution"]["gini"]
            - baseline["candidate_count_distribution"]["gini"]
        ),
        "delta_candidate_purity": treatment["candidate_purity"] - baseline["candidate_purity"],
        "delta_cluster_acc": treatment["cluster_acc"] - baseline["cluster_acc"],
        "delta_cluster_nmi": treatment["cluster_nmi"] - baseline["cluster_nmi"],
        "delta_cluster_ari": treatment["cluster_ari"] - baseline["cluster_ari"],
        "delta_k_abs_error": treatment["cluster_k_abs_error"] - baseline["cluster_k_abs_error"],
        "baseline_classes_below_20pct": baseline["classes_below_20pct"],
        "treatment_classes_below_20pct": treatment["classes_below_20pct"],
    }


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [row[key] for row in rows]
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def build_report(summary: dict[str, Any]) -> str:
    rows = summary["splits"]
    lines = [
        "# Post-hoc attribution of uncertainty-band candidate-pool changes",
        "",
        "This report is a retrospective description of six frozen paired runs. Test labels are used only after evaluation to describe unknown-class coverage and candidate-pool composition. They are not used for training, checkpoint selection, threshold calibration, candidate selection, or method choice.",
        "",
        "The analysis asks whether the treatment's candidate gains are distributed across unknown classes or concentrated in classes that already had high baseline coverage. It also records whether concentration changes move together with purity and clustering metrics. With six splits, these are descriptive relationships rather than confirmatory statistical tests.",
        "",
        "| split | coverage vs delta Spearman | baseline-count vs delta Spearman | positive/negative classes | gain share in baseline top-10 | top-10 mass change | Gini change | purity change | NMI change | K-error change |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['coverage_delta_correlation']['spearman']:+.3f} | "
            f"{row['candidate_count_delta_correlation']['spearman']:+.3f} | "
            f"{row['positive_class_count']}/{row['negative_class_count']} | "
            f"{row['positive_gain_share_in_baseline_top10']:.1%} | "
            f"{row['delta_top10_mass']:+.3%} | {row['delta_gini']:+.3f} | "
            f"{row['delta_candidate_purity']:+.3%} | {row['delta_cluster_nmi']:+.4f} | "
            f"{row['delta_k_abs_error']:+d} |"
        )

    correlations = summary["cross_split_correlations"]
    mean_gain_share = mean(finite_values(rows, "positive_gain_share_in_baseline_top10"))
    mean_positive = mean(finite_values(rows, "positive_class_count"))
    mean_negative = mean(finite_values(rows, "negative_class_count"))
    lines += [
        "",
        "## Findings",
        "",
        f"Across the six splits, the treatment added candidates to an average of {mean_positive:.1f} classes and removed candidates from {mean_negative:.1f} classes. The positive candidate gain assigned to the ten classes with the highest baseline candidate counts was {mean_gain_share:.1%} on average. This is a concentration diagnostic: it does not mean those classes are semantically special, and it cannot justify test-label-based class weighting.",
        "",
        "The classwise correlations are mixed rather than uniformly positive. Therefore the present evidence does not establish that uncertainty-band systematically favors already well-covered classes. The stronger and more repeatable observation is that the treatment changes class composition while improving mean rejection and purity only modestly; the number of low-coverage classes can increase in some splits even when purity improves.",
        "",
        f"Across splits, the correlation between top-10 mass change and purity change is {correlations['top10_mass_vs_purity']['pearson']:+.3f}, between top-10 mass change and NMI change is {correlations['top10_mass_vs_nmi']['pearson']:+.3f}, and between top-10 mass change and K-error improvement is {correlations['top10_mass_vs_k_improvement']['pearson']:+.3f}. These correlations use only six paired points, so they are evidence for follow-up questions rather than a method-selection rule.",
        "",
        "## Decision",
        "",
        "The result is sufficient to reject a broad claim that uncertainty-band provides a stable class-coverage improvement. It is not sufficient to justify another threshold or fusion-weight scan. A new experiment is warranted only if it tests a pre-specified, known-validation-only control for candidate distribution or coverage. The next candidate should change one mechanism at a time and keep the split, teacher/student checkpoints, rejector, calibration target, discovery pool, and clustering representation fixed.",
        "",
        "The new rule must be defined without unknown test labels. Test labels may be used after completion to report coverage by class, purity, and clustering changes. If no validation-only rule can be specified before seeing the paired test outputs, no new experiment should be started from this audit.",
        "",
        "Machine-readable output: `candidate_pool_posthoc_attribution_6split.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("analysis/candidate_pool_class_composition_6split.json"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [analyze_split(split, arms) for split, arms in sorted(data.items())]

    def cross_split_correlation(x_key: str, y_key: str) -> dict[str, float]:
        xs = [row[x_key] for row in rows]
        ys = [row[y_key] for row in rows]
        return correlation(xs, ys)

    summary = {
        "input": str(args.input),
        "split_count": len(rows),
        "splits": rows,
        "cross_split_correlations": {
            "top10_mass_vs_purity": cross_split_correlation("delta_top10_mass", "delta_candidate_purity"),
            "top10_mass_vs_nmi": cross_split_correlation("delta_top10_mass", "delta_cluster_nmi"),
            "top10_mass_vs_k_improvement": cross_split_correlation("delta_top10_mass", "delta_k_abs_error"),
        },
    }
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "candidate_pool_posthoc_attribution_6split.json"
    md_path = args.analysis_dir / "candidate_pool_posthoc_attribution_6split.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_report(summary), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
