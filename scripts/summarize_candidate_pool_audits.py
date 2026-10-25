#!/usr/bin/env python3
"""Summarize frozen label-free candidate-selector audits across fresh splits.

The script reports every selector in a fixed alphabetical order. Pool labels
are read only from retrospective audit fields and never used to rank a method,
select a candidate rule, or launch training.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    protocol = report["protocol"]
    if protocol["labels_used_only_for_audit"] is not True:
        raise ValueError(f"{path}: pool labels were not audit-only")
    if protocol["mc_pool_labels_used_for_selection"] is not False:
        raise ValueError(f"{path}: MC selector used pool labels")
    if protocol["validation_known_labels_used_only_for_calibration"] is not True:
        raise ValueError(f"{path}: validation protocol mismatch")
    if protocol["pool_size"] != 5400 or protocol["selector_ratio"] != 0.25:
        raise ValueError(f"{path}: selector budget mismatch")
    return report


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(paths: list[Path]) -> dict[str, Any]:
    reports = [load(path) for path in paths]
    selectors = sorted(reports[0]["candidate_audit"])
    if any(sorted(report["candidate_audit"]) != selectors for report in reports[1:]):
        raise ValueError("selector sets differ across audit reports")

    rows: list[dict[str, Any]] = []
    for selector in selectors:
        values = []
        for report in reports:
            aggregate = report["candidate_audit"][selector]
            classwise = report["classwise_candidate_audit"][selector][
                "novel_recall_summary"
            ]
            values.append(
                {
                    "known_contamination": aggregate["known_contamination"],
                    "unknown_recall": aggregate["unknown_recall"],
                    "novel_recall_min": classwise["min"],
                    "novel_recall_median": classwise["median"],
                }
            )
        row = {
            "selector": selector,
            "per_split": values,
            "mean": {
                key: mean([value[key] for value in values])
                for key in values[0]
            },
            "std": {
                key: std([value[key] for value in values])
                for key in values[0]
            },
        }
        rows.append(row)
    return {
        "protocol": "semantic_hard_candidate_pool_audit_20261024",
        "splits": [report["protocol"]["seed"] for report in reports],
        "source_reports": [str(path) for path in paths],
        "selector_budget": {"pool_size": 5400, "ratio": 0.25},
        "rows": rows,
        "decision": "descriptive_only_no_selector_selected",
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Semantic-hard candidate selector audit summary",
        "",
        "This is a retrospective, label-free selector audit across frozen",
        "shared-backbone checkpoints. Every selector uses the same 5,400-sample",
        "pool and 25% budget. Pool labels are used only after selection to",
        "measure contamination and novel-class recall; no selector is promoted",
        "or used to start training from this report.",
        "",
        "| selector | mean contamination | mean unknown recall | mean min novel recall | mean median novel recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        values = row["mean"]
        lines.append(
            f"| {row['selector']} | {values['known_contamination']:.4f} | {values['unknown_recall']:.4f} | {values['novel_recall_min']:.4f} | {values['novel_recall_median']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The table is descriptive only. Selector differences are candidate-pool",
        "tradeoffs, not model-performance gains: a lower contamination value",
        "does not by itself establish better unknown coverage, representation",
        "quality, or end-to-end discovery. Because the same retrospective labels",
        "were used to report every row, no row is used to choose a training",
        "target, loss, or default selector.",
        "",
        "The next training experiment, if resumed, must pre-register one selector",
        "or use a selector-independent objective on a new split before seeing its",
        "pool labels. No candidate selector is changed by this audit.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    paths = [
        Path("analysis/semantic_hard_candidate_pool_audit_s2084_shared.json"),
        Path("analysis/semantic_hard_candidate_pool_audit_s2085_shared.json"),
        Path("analysis/semantic_hard_candidate_pool_audit_s2086_shared.json"),
    ]
    report = summarize(paths)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "semantic_hard_candidate_pool_audit_s2084_s2086_summary.json"
    markdown_path = args.analysis_dir / "semantic_hard_candidate_pool_audit_s2084_s2086_20261024.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
