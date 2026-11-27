"""Describe unknown false accepts by CIFAR-100 fine class and predicted class.

This is a post-hoc test-set audit only. Labels must not be used to train,
calibrate, select checkpoints, or choose a method.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def analyze(detail_path: Path, split_path: Path) -> dict:
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    required = {"raw_labels", "true_known", "pred_known", "pred_class"}
    missing = required.difference(detail)
    if missing:
        raise ValueError(f"Discovery detail missing required fields: {sorted(missing)}")

    raw_labels = np.asarray(detail["raw_labels"], dtype=np.int64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    pred_known = np.asarray(detail["pred_known"], dtype=bool)
    pred_class = np.asarray(detail["pred_class"], dtype=np.int64)
    if not (len(raw_labels) == len(true_known) == len(pred_known) == len(pred_class)):
        raise ValueError("Discovery detail fields have inconsistent lengths")

    known_classes = {int(value) for value in split["known_classes"]}
    novel_classes = {int(value) for value in split["novel_classes"]}
    if known_classes & novel_classes or known_classes | novel_classes != set(range(100)):
        raise ValueError("Split must partition CIFAR-100 labels 0..99")

    observed_known = np.isin(raw_labels, list(known_classes))
    if not np.array_equal(observed_known, true_known):
        raise ValueError("raw_labels and true_known disagree with the split metadata")

    unknown_mask = ~true_known
    false_accept_mask = unknown_mask & pred_known
    class_counts = Counter(map(int, raw_labels[unknown_mask]))
    false_accept_counts = Counter(map(int, raw_labels[false_accept_mask]))
    per_novel_class = []
    for class_id in sorted(novel_classes):
        total = class_counts[class_id]
        false_accepts = false_accept_counts[class_id]
        per_novel_class.append(
            {
                "class_id": class_id,
                "count": total,
                "false_accepts": false_accepts,
                "false_accept_rate": false_accepts / total if total else None,
            }
        )

    accepted_prediction_counts = Counter(map(int, pred_class[false_accept_mask]))
    false_accept_total = int(false_accept_mask.sum())
    rates = [row["false_accept_rate"] for row in per_novel_class if row["false_accept_rate"] is not None]
    return {
        "protocol": split.get("protocol"),
        "detail_path": str(detail_path),
        "split_path": str(split_path),
        "test_labels_used_for_training_selection_or_calibration": False,
        "purpose": "descriptive post-hoc test-set error analysis only",
        "unknown_count": int(unknown_mask.sum()),
        "unknown_false_accept_count": false_accept_total,
        "unknown_false_accept_rate": (
            false_accept_total / int(unknown_mask.sum()) if unknown_mask.any() else None
        ),
        "novel_classes_with_false_accepts": sum(row["false_accepts"] > 0 for row in per_novel_class),
        "novel_class_count": len(novel_classes),
        "per_novel_class_false_accept_rate": {
            "min": min(rates) if rates else None,
            "median": float(np.median(rates)) if rates else None,
            "max": max(rates) if rates else None,
        },
        "novel_classes": per_novel_class,
        "false_accepts_by_predicted_known_class": [
            {"class_id": class_id, "count": count}
            for class_id, count in accepted_prediction_counts.most_common()
        ],
        "top_5_predicted_class_share": (
            sum(count for _, count in accepted_prediction_counts.most_common(5))
            / false_accept_total
            if false_accept_total
            else None
        ),
        "top_10_predicted_class_share": (
            sum(count for _, count in accepted_prediction_counts.most_common(10))
            / false_accept_total
            if false_accept_total
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(args.detail, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
