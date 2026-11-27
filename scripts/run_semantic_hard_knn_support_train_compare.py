"""Compare training-time class-conditional kNN support-boundary loss on semantic-hard.

One shared semantic-hard teacher and one fixed student seed are used for a
matched baseline/treatment pair. The sole training factor is whether the
shared-feature discovery kNN support-boundary loss is active.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


def load_args(path: Path, command: str, seed: int, epochs: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.epochs = int(epochs)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-discover", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha-knn", type=float, default=0.1)
    parser.add_argument(
        "--cross-view-aggregation",
        choices=["intersection", "union", "union_budgeted"],
        default="intersection",
        help="Candidate aggregation used in both matched arms.",
    )
    parser.add_argument("--tag", default="semantic_hard_knn_support_train_s2029_e4")
    args = parser.parse_args()

    sources = (args.source_train.resolve(), args.source_discover.resolve(), args.teacher_ckpt.resolve())
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("Source configs and a split-compatible teacher checkpoint are required")

    split_path = ROOT / "splits_cifar100_protocols" / "cifar100_60_40_semantic_hard.json"
    arms = {"baseline": 0.0, "knn_support_boundary": float(args.alpha_knn)}
    output_dirs = {
        name: (
            ROOT / "runs" / f"{args.tag}_{name}",
            ROOT / "runs" / f"{args.tag}_{name}_discover",
        )
        for name in arms
    }
    for train_dir, discover_dir in output_dirs.values():
        if train_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"Refusing to overwrite outputs: {train_dir}, {discover_dir}")

    summary = {
        "question": "Does training-time class-conditional kNN support-boundary loss improve semantic-hard open-set separation?",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "single_changed_factor": "alpha_discovery_knn_boundary (0 vs configured treatment weight)",
        "treatment_weight": float(args.alpha_knn),
        "risk_mode": "pure",
        "candidate_gate": (
            "shared_boundary with cross-view gating and "
            f"{args.cross_view_aggregation} aggregation"
        ),
        "split": str(split_path),
        "teacher_checkpoint": str(args.teacher_ckpt.resolve()),
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single-seed exploratory training-objective comparison",
        "arms": {},
    }

    for name, alpha in arms.items():
        train_dir, discover_dir = output_dirs[name]
        train_args = load_args(args.source_train, "train_student", args.seed, args.epochs)
        train_args.split_path = str(split_path)
        train_args.work_dir = str(train_dir)
        train_args.teacher_ckpt = str(args.teacher_ckpt.resolve())
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.alpha_discovery_knn_boundary = alpha
        train_args.alpha_discovery_rejection_knn_boundary = 0.0
        train_args.alpha_discovery_rejection_boundary = 0.0
        train_args.alpha_discovery_boundary = 0.0
        train_args.discovery_knn_risk = "pure"
        train_args.discovery_knn_selection_conditional = False
        train_args.discovery_feature_candidate_gating = True
        train_args.discovery_cross_view_gating = True
        train_args.discovery_cross_view_aggregation = args.cross_view_aggregation
        train_args.discovery_feature_candidate_space = "shared_boundary"
        print(f"[train] arm={name} alpha_knn={alpha} output={train_dir}", flush=True)
        fit_student(train_args)

        discover_args = load_args(args.source_discover, "discover", args.seed, args.epochs)
        discover_args.split_path = str(split_path)
        discover_args.work_dir = str(discover_dir)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.skip_clustering = True
        discover_args.score_mode = "feature_rejector"
        discover_args.rejector_pool_selection = "all"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        print(f"[discover] arm={name} output={discover_dir}", flush=True)
        discover(discover_args)

        report = json.loads((discover_dir / "discovery_report.json").read_text(encoding="utf-8"))
        saved_train = json.loads((train_dir / "train_student_config.json").read_text(encoding="utf-8"))
        saved_discover = json.loads((discover_dir / "discover_config.json").read_text(encoding="utf-8"))
        metric_keys = (
            "auroc", "aupr", "fpr95", "oscr", "known_accept_rate",
            "unknown_reject_rate", "known_class_accuracy_after_accept",
        )
        summary["arms"][name] = {
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "train_config": {
                key: saved_train.get(key)
                for key in (
                    "seed", "epochs", "teacher_ckpt", "split_path",
                    "alpha_discovery_knn_boundary",
                    "alpha_discovery_rejection_knn_boundary",
                    "alpha_discovery_rejection_boundary", "discovery_knn_risk",
                    "discovery_feature_candidate_gating", "discovery_cross_view_gating",
                    "discovery_cross_view_aggregation",
                    "discovery_feature_candidate_space",
                )
            },
            "discover_config": {
                key: saved_discover.get(key)
                for key in (
                    "seed", "student_ckpt", "split_path", "score_mode",
                    "rejector_pool_selection", "threshold_policy", "target_known_coverage",
                    "matched_pool_size", "matched_known_prior",
                )
            },
            "metrics": {key: report.get(key) for key in metric_keys},
            "score_distribution": report.get("score_distribution", {}),
        }

    summary_path = ROOT / "analysis" / f"{args.tag}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
