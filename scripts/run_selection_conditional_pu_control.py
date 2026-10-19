"""Run a matched candidate-gated PU-kNN selection-conditional control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


def load_args(path: Path, command: str, work_dir: Path, seed: int, epochs: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.epochs = int(epochs)
    args.work_dir = str(work_dir)
    return args


def run_arm(source_train, source_discover, seed, epochs, conditional, tag):
    label = "conditional" if conditional else "historical"
    work_dir = ROOT / "runs" / f"{tag}_{label}"
    discover_dir = ROOT / "runs" / f"{tag}_{label}_discover"
    checkpoint = work_dir / "student.pt"
    if not checkpoint.is_file():
        train_args = load_args(source_train, "train_student", work_dir, seed, epochs)
        train_args.student_ckpt = str(checkpoint)
        train_args.alpha_discovery_knn_boundary = 0.1
        train_args.alpha_discovery_rejection_knn_boundary = 0.0
        train_args.alpha_discovery_rejection_boundary = 0.0
        train_args.alpha_discovery_boundary = 0.0
        train_args.discovery_feature_candidate_gating = True
        train_args.discovery_cross_view_gating = False
        train_args.discovery_feature_candidate_space = "shared_boundary"
        train_args.discovery_pool_mode = "mixed"
        train_args.discovery_knn_risk = "hard_nnpu"
        train_args.discovery_knn_known_prior = 0.2
        train_args.discovery_knn_selection_conditional = bool(conditional)
        fit_student(train_args)
    report_path = discover_dir / "discovery_report.json"
    if not report_path.is_file():
        discover_args = load_args(source_discover, "discover", discover_dir, seed, epochs)
        discover_args.student_ckpt = str(checkpoint)
        discover_args.discovery_feature_candidate_gating = True
        discover_args.discovery_cross_view_gating = False
        discover_args.discovery_feature_candidate_space = "shared_boundary"
        discover_args.discovery_pool_mode = "mixed"
        discover_args.alpha_discovery_knn_boundary = 0.1
        discover_args.alpha_discovery_rejection_knn_boundary = 0.0
        discover_args.alpha_discovery_rejection_boundary = 0.0
        discover_args.alpha_discovery_boundary = 0.0
        discover(discover_args)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    history = json.loads((work_dir / "train_history.json").read_text(encoding="utf-8"))
    last = history[-1].get("train", {}) if history else {}
    distribution = report.get("score_distribution", {})
    return {
        "label": label,
        "work_dir": str(work_dir),
        "metrics": {
            key: report.get(key)
            for key in (
                "auroc", "aupr", "fpr95", "oscr", "ks",
                "known_accept_rate", "unknown_reject_rate",
                "accepted_known_accuracy", "score_overlap",
            )
        },
        "derived_metrics": {
            "unknown_reject_rate": report.get("unknown_reject_rate", distribution.get("unknown_above_threshold")),
            "accepted_known_accuracy": report.get(
                "accepted_known_accuracy", report.get("known_class_accuracy_after_accept")
            ),
            "score_overlap": report.get(
                "score_overlap", distribution.get("overlap_unknown_below_known_q95")
            ),
            "known_above_threshold": distribution.get("known_above_threshold"),
            "unknown_above_threshold": distribution.get("unknown_above_threshold"),
            "mean_gap_unknown_minus_known": distribution.get("mean_gap_unknown_minus_known"),
            "median_gap_unknown_minus_known": distribution.get("median_gap_unknown_minus_known"),
            "ks_distance": distribution.get("ks_distance"),
        },
        "last_train": {
            key: last.get(key)
            for key in (
                "discovery_knn_boundary",
                "feature_candidate_ratio",
                "discovery_knn_mixed_selection_rate",
                "discovery_knn_known_selection_rate",
                "discovery_knn_effective_prior",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-discover", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--tag", default="selection_conditional_pu_s44_e4")
    args = parser.parse_args()
    results = [
        run_arm(args.source_train, args.source_discover, args.seed, args.epochs, flag, args.tag)
        for flag in (False, True)
    ]
    output = ROOT / "analysis" / f"{args.tag}_summary.json"
    payload = {
        "protocol": {
            **vars(args),
            "source_train": str(args.source_train),
            "source_discover": str(args.source_discover),
        },
        "arms": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
