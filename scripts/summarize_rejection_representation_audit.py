#!/usr/bin/env python3
"""Summarize frozen shared/rejection representation mechanism audits."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("representation audit must be a non-empty object")
    for name, report in data.items():
        protocol = report["protocol"]
        if protocol["deterministic_eval"] is not True:
            raise ValueError(f"{name}: non-deterministic evaluation")
        if protocol["unknown_labels_used_only_for_posthoc_diagnostics"] is not True:
            raise ValueError(f"{name}: unknown labels were not posthoc-only")
        if protocol["known_class_labels_used_only_for_posthoc_diagnostics"] is not True:
            raise ValueError(f"{name}: known labels were not posthoc-only")
    return data


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "shared_participation": lambda r: r["effective_rank"]["shared_participation"],
        "rejection_participation": lambda r: r["effective_rank"]["rejection_participation"],
        "shared_rank95": lambda r: r["effective_rank"]["shared_rank95"],
        "rejection_rank95": lambda r: r["effective_rank"]["rejection_rank95"],
        "neighbor_jaccard_at_10": lambda r: r["geometry"]["neighbor_jaccard_at_10"],
        "shared_to_rejection_cross_r2": lambda r: r["geometry"]["shared_to_rejection_cross_r2"],
        "rejection_to_shared_cross_r2": lambda r: r["geometry"]["rejection_to_shared_cross_r2"],
        "shared_binary_probe_auc": lambda r: r["posthoc_binary_probe_auc"]["shared_features"],
        "rejection_binary_probe_auc": lambda r: r["posthoc_binary_probe_auc"]["rejection_features"],
        "shared_logit_probe_auc": lambda r: r["posthoc_binary_probe_auc"]["shared_logits"],
    }
    rows = []
    for name, report in data.items():
        rows.append({"name": name, "seed": report["protocol"]["seed"], **{key: fn(report) for key, fn in metrics.items()}})
    summary = {
        key: {"mean": mean([row[key] for row in rows]), "std": std([row[key] for row in rows])}
        for key in metrics
    }
    return {
        "protocol": "rejection_representation_audit_s2084_s2086_20261024",
        "rows": rows,
        "summary": summary,
        "decision": "mechanism_signal_but_low_rank_coupled_no_training_change",
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Shared/rejection representation audit summary",
        "",
        "This is a deterministic post-hoc audit of frozen shared-backbone",
        "students on seeds 2084–2086. Known/unknown labels are used only for",
        "descriptive probes; no checkpoint, threshold, selector, or training",
        "method is chosen from these results.",
        "",
        "| seed | shared participation | rejection participation | neighbor Jaccard@10 | shared→rejection R² | rejection→shared R² | shared probe AUROC | rejection probe AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['seed']} | {row['shared_participation']:.3f} | {row['rejection_participation']:.3f} | {row['neighbor_jaccard_at_10']:.3f} | {row['shared_to_rejection_cross_r2']:.3f} | {row['rejection_to_shared_cross_r2']:.3f} | {row['shared_binary_probe_auc']:.3f} | {row['rejection_binary_probe_auc']:.3f} |"
        )
    lines += [
        "",
        "## Mean mechanism statistics",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for key, values in report["summary"].items():
        lines.append(f"| {key} | {values['mean']:.6f} | {values['std']:.6f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The rejection branch is not a duplicate of the shared representation:",
        "neighbor overlap is about 0.28 and the rejection probe separates the",
        "frozen test known/unknown cohorts better than the shared feature probe on",
        "all three splits. However, rejection participation is only about 15.1",
        "versus 37.5 for the shared feature, and shared-to-rejection cross-R² is",
        "about 0.85. The branch therefore contains a useful residual signal but",
        "remains low-rank and strongly coupled to the shared representation.",
        "",
        "This is mechanism evidence, not a new-class discovery improvement. The",
        "same checkpoints' rejection-support candidate selector did not establish",
        "a stable class-tail coverage gain in the separate candidate audit. Do not",
        "change the training objective or default score from this audit alone.",
        "",
        "Next valid training step: one pre-registered mechanism that explicitly",
        "raises rejection representation rank/known-class retention while keeping",
        "the current shared-backbone branch and all validation-only calibration",
        "rules fixed. Do not simultaneously add VOS, extra PU, uncertainty-KD, or",
        "selector fusion.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("analysis/rejection_representation_audit_s2084_s2086.json"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    report = summarize(load(args.input))
    json_path = args.analysis_dir / "rejection_representation_audit_s2084_s2086_summary.json"
    markdown_path = args.analysis_dir / "rejection_representation_audit_s2084_s2086_20261024.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
