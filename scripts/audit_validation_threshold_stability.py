"""Estimate known-only threshold variability from exported calibration scores.

This audit uses only known-validation scores. It reports bootstrap threshold
variation and out-of-bag known coverage; it does not estimate unknown rejection
or claim cross-split/domain-shift robustness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _summary(values: list[float]) -> dict:
    return {
        "valid_repeats": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        "q025": float(np.quantile(values, 0.025)) if values else None,
        "q975": float(np.quantile(values, 0.975)) if values else None,
    }


def audit(
    scores: np.ndarray,
    target: float,
    repeats: int,
    seed: int,
    predicted_classes: np.ndarray | None = None,
    classwise_min_samples: int = 5,
) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite 1-D array with at least two values")
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if predicted_classes is not None:
        predicted_classes = np.asarray(predicted_classes, dtype=np.int64)
        if predicted_classes.shape != scores.shape or np.any(predicted_classes < 0):
            raise ValueError("predicted_classes must be non-negative and match scores")
        if classwise_min_samples <= 0:
            raise ValueError("classwise_min_samples must be positive")

    rng = np.random.default_rng(seed)
    thresholds = np.empty(repeats, dtype=np.float64)
    oob_coverages = []
    classwise_oob_coverages = []
    n = scores.size
    for repeat in range(repeats):
        sampled = rng.integers(0, n, size=n)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[sampled] = True
        thresholds[repeat] = np.quantile(scores[sampled], target, method="linear")
        oob = scores[~in_bag]
        if oob.size:
            oob_coverages.append(float(np.mean(oob <= thresholds[repeat])))
            if predicted_classes is not None:
                global_threshold = float(np.percentile(scores[sampled], 95.0))
                class_thresholds = np.full(
                    int(predicted_classes.max()) + 1, global_threshold, dtype=np.float64
                )
                for class_id in range(len(class_thresholds)):
                    class_scores = scores[sampled][
                        predicted_classes[sampled] == class_id
                    ]
                    if len(class_scores) >= classwise_min_samples:
                        class_thresholds[class_id] = np.percentile(class_scores, 95.0)
                oob_classes = predicted_classes[~in_bag]
                classwise_oob_coverages.append(
                    float(np.mean(oob <= class_thresholds[oob_classes]))
                )

    full_threshold = float(np.quantile(scores, target, method="linear"))
    result = {
        "sample_count": int(n),
        "target_known_coverage": float(target),
        "full_sample_threshold": full_threshold,
        "full_sample_acceptance": float(np.mean(scores <= full_threshold)),
        "bootstrap_repeats": int(repeats),
        "bootstrap_threshold": {
            "mean": float(np.mean(thresholds)),
            "std": float(np.std(thresholds, ddof=1)) if repeats > 1 else 0.0,
            "q025": float(np.quantile(thresholds, 0.025)),
            "median": float(np.quantile(thresholds, 0.5)),
            "q975": float(np.quantile(thresholds, 0.975)),
        },
        "out_of_bag_known_acceptance": _summary(oob_coverages),
        "interpretation_limit": (
            "Known-validation finite-sample calibration stability only; does not "
            "measure unknown rejection or cross-split/domain shift."
        ),
    }
    if predicted_classes is not None:
        classwise_thresholds = np.full(
            int(predicted_classes.max()) + 1,
            float(np.percentile(scores, 95.0)),
            dtype=np.float64,
        )
        for class_id in range(len(classwise_thresholds)):
            class_scores = scores[predicted_classes == class_id]
            if len(class_scores) >= classwise_min_samples:
                classwise_thresholds[class_id] = np.percentile(class_scores, 95.0)
        result["class_conditional_q95"] = {
            "minimum_calibration_samples_per_predicted_class": int(classwise_min_samples),
            "full_sample_acceptance": float(
                np.mean(scores <= classwise_thresholds[predicted_classes])
            ),
            "out_of_bag_known_acceptance": _summary(classwise_oob_coverages),
            "interpretation_limit": (
                "Validation-only OOB diagnostic of the existing class-conditional policy; "
                "not test coverage or unknown-rejection evidence."
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="calibration_scores.npz from discover")
    parser.add_argument("--compare", type=Path, help="optional second aligned validation-score archive")
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classwise-min-samples",
        type=int,
        default=5,
        help=(
            "Minimum in-bag known-validation samples for a predicted-class q95; "
            "smaller bins fall back to the global q95."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.scores, allow_pickle=False) as payload:
        required = {"score", "predicted_class", "source_index", "calibrated_threshold"}
        if not required.issubset(payload.files):
            raise ValueError(f"score archive is missing arrays: {sorted(required - set(payload.files))}")
        scores = payload["score"]
        source_index = payload["source_index"]
        if len(source_index) != len(scores) or len(np.unique(source_index)) != len(source_index):
            raise ValueError("source indices must uniquely identify each exported score")
        predicted_classes = payload["predicted_class"]
        result = {
            "input": str(args.scores),
            "sample_index_sha256_matches_export_metadata": None,
            "audit": audit(
                scores,
                args.target,
                args.repeats,
                args.seed,
                predicted_classes=predicted_classes,
                classwise_min_samples=args.classwise_min_samples,
            ),
        }
    metadata_path = args.scores.with_name("calibration_report.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("score_export", {}).get("source_index_sha256")
        actual = hashlib.sha256(
            ",".join(map(str, source_index.tolist())).encode("utf-8")
        ).hexdigest()
        result["sample_index_sha256_matches_export_metadata"] = actual == expected
        result["metadata"] = metadata.get("score_export", {})
    if args.compare:
        with np.load(args.compare, allow_pickle=False) as other:
            other_scores = np.asarray(other["score"], dtype=np.float64)
            other_indices = np.asarray(other["source_index"], dtype=np.int64)
        if not np.array_equal(source_index, other_indices):
            raise ValueError("paired archives must contain the same source indices in the same order")
        rng = np.random.default_rng(args.seed)
        threshold_deltas = np.empty(args.repeats, dtype=np.float64)
        for repeat in range(args.repeats):
            sampled = rng.integers(0, len(scores), size=len(scores))
            threshold_deltas[repeat] = (
                np.quantile(other_scores[sampled], args.target, method="linear")
                - np.quantile(scores[sampled], args.target, method="linear")
            )
        result["paired_comparison"] = {
            "comparison_input": str(args.compare),
            "threshold_delta_definition": "comparison minus primary",
            "full_sample_threshold_delta": float(
                np.quantile(other_scores, args.target, method="linear")
                - np.quantile(scores, args.target, method="linear")
            ),
            "paired_bootstrap_threshold_delta": {
                "mean": float(np.mean(threshold_deltas)),
                "std": float(np.std(threshold_deltas, ddof=1)) if args.repeats > 1 else 0.0,
                "q025": float(np.quantile(threshold_deltas, 0.025)),
                "median": float(np.quantile(threshold_deltas, 0.5)),
                "q975": float(np.quantile(threshold_deltas, 0.975)),
            },
            "interpretation_limit": "Paired calibration-threshold variability only; not a test-coverage or unknown-rejection comparison.",
        }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
