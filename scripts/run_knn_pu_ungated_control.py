"""Compare naive and NNPU kNN losses on the unfiltered mixed pool.

This removes candidate selection from both arms so the configured mixed-pool
known prior and the known-risk correction refer to the same sampling frame.
The ``naive_mixed`` arm deliberately ignores contamination; only
``discovery_knn_risk`` differs between it and ``hard_nnpu``.
"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-discover", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--known-prior", type=float, default=0.2)
    parser.add_argument("--output-prefix", default="knn_pu_ungated_s44_e4")
    args = parser.parse_args()

    train_source = args.source_train.resolve()
    discover_source = args.source_discover.resolve()
    arms = ("naive_mixed", "hard_nnpu")
    summary = {
        "protocol": {
            "source_train": str(train_source),
            "source_discover": str(discover_source),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "known_prior": float(args.known_prior),
            "candidate_gating": False,
            "cross_view_gating": False,
            "single_changed_factor": "discovery_knn_risk",
            "test_labels_used_for_training_or_selection": False,
        },
        "arms": {},
    }

    for risk in arms:
        work_dir = ROOT / "runs" / f"{args.output_prefix}_{risk}"
        discover_dir = ROOT / "runs" / f"{args.output_prefix}_{risk}_discover"
        student_ckpt = work_dir / "student.pt"
        report_path = discover_dir / "discovery_report.json"

        if not student_ckpt.is_file():
            train_args = load_args(
                train_source, "train_student", work_dir, args.seed, args.epochs
            )
            train_args.student_ckpt = str(student_ckpt)
            train_args.alpha_discovery_knn_boundary = 0.1
            train_args.alpha_discovery_rejection_knn_boundary = 0.0
            train_args.alpha_discovery_rejection_boundary = 0.0
            train_args.alpha_discovery_boundary = 0.0
            train_args.discovery_feature_candidate_gating = False
            train_args.discovery_cross_view_gating = False
            train_args.discovery_feature_candidate_space = "shared"
            train_args.discovery_knn_risk = risk
            train_args.discovery_knn_known_prior = float(args.known_prior)
            fit_student(train_args)

        if not report_path.is_file():
            discover_args = load_args(
                discover_source, "discover", discover_dir, args.seed, args.epochs
            )
            discover_args.student_ckpt = str(student_ckpt)
            discover(discover_args)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        history_path = work_dir / "train_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        last_train = history[-1].get("train", {}) if history else {}
        config = json.loads((work_dir / "train_student_config.json").read_text(encoding="utf-8"))
        summary["arms"][risk] = {
            "work_dir": str(work_dir),
            "config": {
                key: config.get(key)
                for key in (
                    "discovery_feature_candidate_gating",
                    "discovery_cross_view_gating",
                    "discovery_knn_risk",
                    "discovery_knn_known_prior",
                    "alpha_discovery_knn_boundary",
                    "discovery_pool_mode",
                    "matched_pool_size",
                )
            },
            "last_epoch": {
                key: last_train.get(key)
                for key in (
                    "discovery_knn_boundary",
                    "feature_candidate_ratio",
                    "feature_candidate_known_fraction",
                )
            },
            "metrics": {
                key: report.get(key)
                for key in (
                    "auroc",
                    "aupr",
                    "fpr95",
                    "oscr",
                    "known_accept_rate",
                    "unknown_reject_rate",
                    "known_rejected",
                    "unknown_false_accept",
                )
            },
        }

    output = ROOT / "analysis" / f"{args.output_prefix}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
