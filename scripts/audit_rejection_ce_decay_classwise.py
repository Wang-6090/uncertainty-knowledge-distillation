"""Audit classwise unknown coverage for the frozen late-decay pairs.

This is a post-hoc diagnostic only. Test labels are used to describe error
concentration after the runs are complete; they are never used to train,
calibrate, select, or modify a method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PAIRS = {
    2121: ("static_control", "late_decay"),
    2124: ("static_control", "late_decay"),
    2125: ("static_control", "late_decay"),
    2127: ("static_control", "late_decay"),
}


def load_run(runs_dir: Path, seed: int, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = runs_dir / f"rejection_ce_decay_full_s{seed}_{arm}_discover"
    detail = json.loads((root / "discovery_detail.json").read_text(encoding="utf-8"))
    report = json.loads((root / "discovery_report.json").read_text(encoding="utf-8"))
    return detail, report


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def gini(values: list[float]) -> float:
    xs = sorted(float(v) for v in values if v >= 0)
    total = sum(xs)
    if not xs or total == 0:
        return 0.0
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, start=1)) / (n * total)


def audit(detail: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    score = detail["score"]
    true_known = detail["true_known"]
    # `true_label` is the remapped known/unknown task label used by the
    # evaluator. `raw_labels` retains the original CIFAR-100 fine class and
    # is required for a classwise unknown audit.
    labels = detail["raw_labels"]
    threshold = float(report["score_distribution"]["threshold"])
    novel_labels = sorted({int(y) for y, k in zip(labels, true_known) if not int(k)})

    rows = []
    for label in novel_labels:
        indices = [i for i, y in enumerate(labels) if int(y) == label]
        rejected = sum(float(score[i]) >= threshold for i in indices)
        rows.append(
            {
                "label": label,
                "count": len(indices),
                "rejected": rejected,
                "rejection_rate": rejected / len(indices),
                "accepted": len(indices) - rejected,
                "score_q25": quantile([float(score[i]) for i in indices], 0.25),
                "score_median": quantile([float(score[i]) for i in indices], 0.50),
            }
        )

    rates = [row["rejection_rate"] for row in rows]
    candidate_counts = [row["rejected"] for row in rows]
    total_unknown = sum(row["count"] for row in rows)
    total_rejected = sum(candidate_counts)
    return {
        "threshold": threshold,
        "unknown_count": total_unknown,
        "unknown_rejected": total_rejected,
        "unknown_rejection_rate": total_rejected / total_unknown,
        "class_count": len(rows),
        "classwise_rejection_rate": {
            "min": min(rates),
            "q25": quantile(rates, 0.25),
            "median": quantile(rates, 0.50),
            "q75": quantile(rates, 0.75),
            "max": max(rates),
            "std": math.sqrt(safe_mean([(r - safe_mean(rates)) ** 2 for r in rates])),
        },
        "candidate_count_distribution": {
            "min": min(candidate_counts),
            "q25": quantile(candidate_counts, 0.25),
            "median": quantile(candidate_counts, 0.50),
            "q75": quantile(candidate_counts, 0.75),
            "max": max(candidate_counts),
            "gini": gini(candidate_counts),
        },
        "classes_with_zero_rejection": sum(x == 0 for x in candidate_counts),
        "classes_with_full_rejection": sum(x == rows[i]["count"] for i, x in enumerate(candidate_counts)),
        "classes": rows,
    }


def build_report(results: dict[str, Any]) -> str:
    lines = [
        "# Late-decay classwise unknown-coverage audit",
        "",
        "This is a post-hoc diagnostic of the four frozen paired runs. Test labels are used only to describe classwise error concentration after evaluation; no label enters training, checkpoint selection, threshold calibration, candidate selection, or method choice.",
        "",
        "The key question is whether the larger late-decay candidate pool is broadly distributed across unknown classes or concentrated in a small subset. Concentration would provide a concrete explanation for a purer candidate pool that nevertheless has worse clustering geometry.",
        "",
        "| split | arm | unknown rejection | classwise rate min/median/max | zero-rejection classes | candidate count min/median/max | candidate-count Gini |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed, arms in results.items():
        for arm in arms:
            r = arms[arm]
            cr = r["classwise_rejection_rate"]
            cc = r["candidate_count_distribution"]
            lines.append(
                f"| {seed} | {arm} | {r['unknown_rejection_rate']:.3%} | "
                f"{cr['min']:.3%} / {cr['median']:.3%} / {cr['max']:.3%} | "
                f"{r['classes_with_zero_rejection']} | "
                f"{cc['min']:.0f} / {cc['median']:.1f} / {cc['max']:.0f} | {cc['gini']:.3f} |"
            )
        static_rows = {row["label"]: row for row in arms["static_control"]["classes"]}
        late_rows = {row["label"]: row for row in arms["late_decay"]["classes"]}
        deltas = [
            late_rows[label]["rejection_rate"] - static_rows[label]["rejection_rate"]
            for label in static_rows
        ]
        lines.append(
            f"| {seed} | late - static classwise delta | {safe_mean(deltas):+.3%} mean | "
            f"{min(deltas):+.3%} / {quantile(deltas, 0.50):+.3%} / {max(deltas):+.3%} | "
            f"{sum(delta > 0 for delta in deltas)} positive / {sum(delta == 0 for delta in deltas)} zero | — | — |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "This audit is descriptive. It does not justify class-specific weighting from test labels. If late-decay increases candidate-count concentration or leaves a large class tail with little coverage, a follow-up must first define a label-free, validation-derived coverage control. If classwise coverage is broadly improved while clustering still declines, the next experiment should target candidate representation stability rather than candidate quantity.",
        "",
        "Machine-readable output: `rejection_ce_decay_classwise_audit_4split.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for seed, arms in PAIRS.items():
        results[str(seed)] = {}
        for arm in arms:
            detail, report = load_run(args.runs_dir, seed, arm)
            results[str(seed)][arm] = audit(detail, report)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.analysis_dir / "rejection_ce_decay_classwise_audit_4split.json"
    out_md = args.analysis_dir / "rejection_ce_decay_classwise_audit_4split_20261110.md"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(build_report(results), encoding="utf-8")
    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()
