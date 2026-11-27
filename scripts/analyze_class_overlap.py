"""Summarize known/unknown score overlap by true CIFAR class.

This is a retrospective diagnostic only.  It must not be used to choose a
threshold or a training configuration because it reads test labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("discover_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    detail = json.loads((args.discover_dir / "discovery_detail.json").read_text(encoding="utf-8"))
    report = json.loads((args.discover_dir / "discovery_report.json").read_text(encoding="utf-8"))
    score = np.asarray(detail["score"], dtype=np.float64)
    # ``true_label`` is the open-set evaluation label and maps every unknown
    # sample to -1.  ``raw_labels`` retains the dataset class id and is the
    # correct field for per-unknown-class diagnostics.
    raw_label = np.asarray(detail["raw_labels"], dtype=np.int64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    threshold = float(report["score_distribution"]["threshold"])
    rows = []
    for label in np.unique(raw_label):
        mask = raw_label == label
        rows.append(
            {
                "label": int(label),
                "known": bool(true_known[mask].all()),
                "count": int(mask.sum()),
                "score_mean": float(score[mask].mean()),
                "score_median": float(np.median(score[mask])),
                "score_q05": float(np.quantile(score[mask], 0.05)),
                "score_q95": float(np.quantile(score[mask], 0.95)),
                "reject_rate_at_report_threshold": float((score[mask] > threshold).mean()),
            }
        )
    known = [row for row in rows if row["known"]]
    unknown = [row for row in rows if not row["known"]]
    unknown.sort(key=lambda row: row["reject_rate_at_report_threshold"])
    result = {
        "warning": "Retrospective test-label diagnostic only; not for model or threshold selection.",
        "discover_dir": str(args.discover_dir),
        "threshold": threshold,
        "known_class_summary": known,
        "hardest_unknown_classes": unknown,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
