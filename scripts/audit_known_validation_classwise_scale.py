"""Audit classwise rejection-score scale using known validation data only.

The audit restores labels for the frozen known calibration subset from the
recorded split protocol. It never reads test labels, mixed-pool labels, or
unknown labels. It describes whether one global rejection threshold is applied
to known predicted classes with materially different score scales. It does not
change a run, fit a calibration rule, or select a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _load_known_validation(run_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config_path = run_dir / "discover_config.json"
    if not config_path.is_file():
        config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = _resolve(config["split_path"])
    data_root = _resolve(config.get("data_root", "data"))
    bundle = build_data_bundle(
        "cifar100",
        str(data_root),
        int(config["num_known"]),
        int(config["seed"]),
        int(config["image_size"]),
        split_path=str(split_path),
        discovery_pool_mode=str(config.get("discovery_pool_mode", "mixed")),
        mixed_known_pool_ratio=float(config.get("mixed_known_pool_ratio", 0.2)),
        matched_discovery_pool=bool(config.get("matched_discovery_pool", True)),
        matched_pool_size=int(config.get("matched_pool_size", 5400)),
        matched_known_prior=float(config.get("matched_known_prior", 0.2)),
        calibration_ratio=float(config.get("calibration_ratio", 0.25)),
        transform_profile=str(config.get("transform_profile", "cifar")),
    )
    if bundle.calibration is None:
        raise RuntimeError(f"missing known calibration subset for {run_dir}")

    score_path = run_dir / "feature_rejector_scores.npz"
    input_path = run_dir / "feature_rejector_inputs.npz"
    with np.load(score_path, allow_pickle=False) as scores, np.load(
        input_path, allow_pickle=False
    ) as inputs:
        required_scores = {"validation_feature_rejector_score", "validation_source_index"}
        required_inputs = {"validation_logits", "validation_source_index"}
        if not required_scores.issubset(scores.files):
            raise ValueError(f"{score_path} is missing {sorted(required_scores - set(scores.files))}")
        if not required_inputs.issubset(inputs.files):
            raise ValueError(f"{input_path} is missing {sorted(required_inputs - set(inputs.files))}")
        score = np.asarray(scores["validation_feature_rejector_score"], dtype=np.float64).reshape(-1)
        source_index = np.asarray(scores["validation_source_index"], dtype=np.int64).reshape(-1)
        input_source_index = np.asarray(inputs["validation_source_index"], dtype=np.int64).reshape(-1)
        logits = np.asarray(inputs["validation_logits"], dtype=np.float64)
    if not np.array_equal(source_index, input_source_index):
        raise RuntimeError(f"score/input validation source indices differ for {run_dir}")
    restored_index = np.asarray(
        [int(bundle.calibration[i][4]) for i in range(len(bundle.calibration))],
        dtype=np.int64,
    )
    if not np.array_equal(source_index, restored_index):
        raise RuntimeError(f"exported validation source indices differ from protocol for {run_dir}")
    labels = np.asarray([int(bundle.calibration[i][1]) for i in range(len(bundle.calibration))], dtype=np.int64)
    predicted = logits.argmax(axis=1).astype(np.int64)
    if len(score) != len(labels) or len(predicted) != len(labels):
        raise RuntimeError(f"validation arrays are not aligned for {run_dir}")
    calibration = json.loads((run_dir / "calibration_report.json").read_text(encoding="utf-8"))
    threshold = float(calibration["calibrated_threshold"])
    return config, score, predicted, labels, np.full(len(score), threshold, dtype=np.float64)


def _summarize(seed: int, arm: str, run_dir: Path) -> dict[str, Any]:
    config, score, predicted, labels, thresholds = _load_known_validation(run_dir)
    accepted = score <= thresholds
    rows = []
    for class_id in sorted(np.unique(predicted).tolist()):
        mask = predicted == class_id
        class_scores = score[mask]
        rows.append(
            {
                "predicted_class": int(class_id),
                "count": int(mask.sum()),
                "accept_rate": float(accepted[mask].mean()),
                "classification_accuracy": float((predicted[mask] == labels[mask]).mean()),
                "score_median": float(np.median(class_scores)),
                "score_q90": float(np.quantile(class_scores, 0.90)),
                "score_q95": float(np.quantile(class_scores, 0.95)),
                "score_q99": float(np.quantile(class_scores, 0.99)),
                "q95_threshold_gap": float(thresholds[mask][0] - np.quantile(class_scores, 0.95)),
            }
        )
    q95 = np.asarray([row["score_q95"] for row in rows], dtype=np.float64)
    accept_rates = np.asarray([row["accept_rate"] for row in rows], dtype=np.float64)
    counts = np.asarray([row["count"] for row in rows], dtype=np.int64)
    return {
        "seed": int(seed),
        "arm": arm,
        "run_dir": str(run_dir),
        "protocol": {
            "known_validation_only": True,
            "unknown_or_test_labels_used": False,
            "threshold_changed": False,
            "model_changed": False,
            "score_source": "frozen feature_rejector_scores.npz",
        },
        "sample_count": int(len(score)),
        "predicted_class_count": int(len(rows)),
        "global_threshold": float(thresholds[0]),
        "global_accept_rate": float(accepted.mean()),
        "known_classification_accuracy": float((predicted == labels).mean()),
        "class_count_min_max": [int(counts.min()), int(counts.max())],
        "classwise_accept_rate": {
            "min": float(accept_rates.min()),
            "median": float(np.median(accept_rates)),
            "max": float(accept_rates.max()),
            "std": float(accept_rates.std(ddof=1)),
            "range": float(accept_rates.max() - accept_rates.min()),
        },
        "classwise_score_q95": {
            "min": float(q95.min()),
            "median": float(np.median(q95)),
            "max": float(q95.max()),
            "std": float(q95.std(ddof=1)),
            "range": float(q95.max() - q95.min()),
        },
        "classes": rows,
        "source_config_seed": int(config["seed"]),
    }


def _rankdata(values: list[float]) -> np.ndarray:
    """Return average ranks without requiring a scipy dependency."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return float("nan")
    ranks_a = _rankdata(values_a)
    ranks_b = _rankdata(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(centered_a, centered_b) / denominator)


def _cross_split_consistency(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare classwise q95 rankings across different seeds within each arm."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        by_arm.setdefault(str(report["arm"]), []).append(report)
    rows = []
    for arm, arm_reports in sorted(by_arm.items()):
        arm_reports = sorted(arm_reports, key=lambda report: int(report["seed"]))
        for index, report_a in enumerate(arm_reports):
            classes_a = {
                row["predicted_class"]: row for row in report_a["classes"]
            }
            for report_b in arm_reports[index + 1 :]:
                classes_b = {
                    row["predicted_class"]: row for row in report_b["classes"]
                }
                common_classes = sorted(set(classes_a) & set(classes_b))
                q95_a = [classes_a[class_id]["score_q95"] for class_id in common_classes]
                q95_b = [classes_b[class_id]["score_q95"] for class_id in common_classes]
                rows.append(
                    {
                        "arm": arm,
                        "seed_a": int(report_a["seed"]),
                        "seed_b": int(report_b["seed"]),
                        "common_class_count": len(common_classes),
                        "q95_spearman": _spearman(q95_a, q95_b),
                        "mean_absolute_q95_delta": float(
                            np.mean(np.abs(np.asarray(q95_a) - np.asarray(q95_b)))
                        ),
                    }
                )
    return rows


def render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Known-validation classwise rejection-score scale audit",
        "",
        "This is a read-only audit of frozen discovery runs. Labels are restored only for the known calibration subset to describe known-class correctness and acceptance. No unknown/test labels, training, threshold fitting, or model selection are used.",
        "",
        "| seed | arm | global accept | class q95 min/max | q95 std | class accept min/max | accept-rate std |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        q95 = report["classwise_score_q95"]
        acc = report["classwise_accept_rate"]
        lines.append(
            f"| {report['seed']} | {report['arm']} | {report['global_accept_rate']:.4f} | {q95['min']:.3f}/{q95['max']:.3f} | {q95['std']:.3f} | {acc['min']:.3f}/{acc['max']:.3f} | {acc['std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The q95 and acceptance ranges are descriptive evidence about the existing global threshold. They do not justify a class-conditional deployment rule; such a rule would require a separately pre-registered, validation-only comparison with out-of-sample calibration checks.",
            "",
            "## Cross-split q95 rank consistency",
            "",
            "| arm | seed pair | common classes | Spearman q95 rank | mean absolute q95 delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in _cross_split_consistency(reports):
        lines.append(
            f"| {row['arm']} | {row['seed_a']} / {row['seed_b']} | {row['common_class_count']} | {row['q95_spearman']:.4f} | {row['mean_absolute_q95_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Cross-split rank correlations compare the same raw CIFAR class IDs in known validation outputs. They are descriptive only and do not select a threshold, model, or training rule.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("SEED", "ARM", "RUN_DIR"),
        required=True,
        help="one frozen discovery run: seed arm run_dir; repeat for each arm",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for seed_text, arm, run_text in args.run:
        seed = int(seed_text)
        run_dir = _resolve(run_text)
        print(f"[audit] seed={seed} arm={arm}", flush=True)
        reports.append(_summarize(seed, arm, run_dir))
    payload = {
        "question": "Do frozen known-validation rejection scores have stable classwise scale heterogeneity across splits and arms?",
        "reports": reports,
        "cross_split_q95_rank_consistency": _cross_split_consistency(reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(reports), encoding="utf-8")
    print(f"saved {args.output}")
    print(f"saved {args.markdown}")


if __name__ == "__main__":
    main()
