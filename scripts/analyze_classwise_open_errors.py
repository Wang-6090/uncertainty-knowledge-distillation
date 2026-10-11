"""Summarize per-class open-set errors for a fixed paired detector comparison.

Test labels are used only for retrospective grouping and never for fitting,
threshold selection, or model selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def _read(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _threshold_by_sample(thresholds, predicted_classes: np.ndarray, count: int):
    values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim == 0:
        return np.full(count, float(values), dtype=np.float64)
    if values.ndim == 1 and values.size:
        predicted_classes = np.asarray(predicted_classes, dtype=np.int64)
        return values[np.clip(predicted_classes, 0, values.size - 1)]
    raise ValueError("calibrated_threshold must be a scalar or non-empty vector")


def _summarize(
    run_dir: Path,
    known_classes: list[int],
    novel_classes: list[int],
    class_names: list[str],
):
    detail = _read(run_dir / "discovery_detail.json")
    calibration = _read(run_dir / "calibration_report.json")
    scores = np.asarray(detail["score"], dtype=np.float64)
    raw_labels = np.asarray(detail["raw_labels"], dtype=np.int64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    pred_class = np.asarray(detail["pred_class"], dtype=np.int64)
    true_label = np.asarray(detail["true_label"], dtype=np.int64)
    threshold_values = np.asarray(calibration["calibrated_threshold"], dtype=np.float64)
    per_sample_threshold = _threshold_by_sample(threshold_values, pred_class, len(scores))
    accepted = scores <= per_sample_threshold

    known_rows = []
    for class_index, raw_label in enumerate(known_classes):
        mask = true_known & (raw_labels == raw_label)
        if not mask.any():
            continue
        known_rows.append(
            {
                "raw_label": int(raw_label),
                "count": int(mask.sum()),
                "accept_rate": float(accepted[mask].mean()),
                "classification_accuracy": float((pred_class[mask] == true_label[mask]).mean()),
                "false_reject_count": int((~accepted[mask]).sum()),
            }
        )

    novel_rows = []
    for raw_label in novel_classes:
        mask = (~true_known) & (raw_labels == raw_label)
        if not mask.any():
            continue
        accepted_predicted_classes = Counter(
            int(value) for value in pred_class[mask & accepted]
        )
        novel_rows.append(
            {
                "raw_label": int(raw_label),
                "class_name": class_names[raw_label],
                "count": int(mask.sum()),
                "false_accept_rate": float(accepted[mask].mean()),
                "correct_reject_count": int((~accepted[mask]).sum()),
                "score_median": float(np.median(scores[mask])),
                "score_q90": float(np.quantile(scores[mask], 0.90)),
                "false_accept_predicted_known_classes": [
                    {
                        "known_index": known_index,
                        "raw_label": int(known_classes[known_index]),
                        "class_name": class_names[known_classes[known_index]],
                        "count": int(count),
                    }
                    for known_index, count in accepted_predicted_classes.most_common()
                ],
            }
        )
    novel_rows.sort(key=lambda row: (-row["false_accept_rate"], -row["count"]))
    return {
        "run": str(run_dir),
        "score_mode": detail["score_mode"],
        "threshold": threshold_values.tolist()
        if threshold_values.ndim
        else float(threshold_values),
        "test_known_accept_rate": float(accepted[true_known].mean()),
        "test_unknown_false_accept_rate": float(accepted[~true_known].mean()),
        "known_classes": known_rows,
        "novel_classes_worst_first": novel_rows,
    }


def analyze(args):
    split = _read(args.split)
    known_classes = [int(value) for value in split["known_classes"]]
    novel_classes = [int(value) for value in split["novel_classes"]]
    class_names = [str(value) for value in split["fine_class_names"]]
    baseline = _summarize(args.baseline_dir, known_classes, novel_classes, class_names)
    candidate = _summarize(args.candidate_dir, known_classes, novel_classes, class_names)
    baseline_novel = {
        row["raw_label"]: row for row in baseline["novel_classes_worst_first"]
    }
    candidate_novel = {
        row["raw_label"]: row for row in candidate["novel_classes_worst_first"]
    }
    novel_delta = []
    for raw_label in novel_classes:
        if raw_label not in baseline_novel or raw_label not in candidate_novel:
            continue
        base = baseline_novel[raw_label]
        variant = candidate_novel[raw_label]
        novel_delta.append(
            {
                "raw_label": int(raw_label),
                "count": base["count"],
                "baseline_false_accept_rate": base["false_accept_rate"],
                "candidate_false_accept_rate": variant["false_accept_rate"],
                "change": variant["false_accept_rate"] - base["false_accept_rate"],
            }
        )
    novel_delta.sort(key=lambda row: (-row["candidate_false_accept_rate"], -row["count"]))
    result = {
        "protocol_note": (
            "Retrospective per-class test analysis only; test labels were not used "
            "for training, threshold calibration, or model selection."
        ),
        "baseline": baseline,
        "candidate": candidate,
        "novel_false_accept_rate_by_raw_class": novel_delta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "known_accept={:.4f}, unknown_false_accept={:.4f}, novel_classes={}".format(
            candidate["test_known_accept_rate"],
            candidate["test_unknown_false_accept_rate"],
            len(candidate["novel_classes_worst_first"]),
        )
    )
    print(f"saved: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
