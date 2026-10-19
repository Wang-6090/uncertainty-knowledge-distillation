"""Compare a full-pool nnPU kNN support-boundary objective on semantic-hard.

The treatment removes the low-recall candidate gate and applies the existing
soft-nnPU correction to the complete mixed discovery pool.  The baseline and
treatment use identical settings; only ``alpha_discovery_knn_boundary`` is
changed.  This isolates whether candidate recall, rather than the boundary
objective itself, was the limiting factor.
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
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha-knn", type=float, default=0.1)
    parser.add_argument("--known-prior", type=float, default=0.2)
    parser.add_argument("--softness", type=float, default=0.05)
    parser.add_argument("--tag", default="semantic_hard_knn_pu_full_pool_s2030_e4")
    args = parser.parse_args()

    sources = (
        args.source_train.resolve(),
        args.source_discover.resolve(),
        args.teacher_ckpt.resolve(),
    )
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("Source configs and a split-compatible teacher checkpoint are required")

    split_path = ROOT / "splits_cifar100_protocols" / "cifar100_60_40_semantic_hard.json"
    arms = {"baseline": 0.0, "knn_pu_full_pool": float(args.alpha_knn)}
    output_dirs = {
        name: (
            ROOT / "runs" / f"{args.tag}_{name}",
            ROOT / "runs" / f"{args.tag}_{name}_discover",
        )
        for name in arms
    }
    completed_arms = {}
    for name, (train_dir, discover_dir) in output_dirs.items():
        if not train_dir.exists() and not discover_dir.exists():
            completed_arms[name] = False
            continue
        required = (
            train_dir / "student.pt",
            train_dir / "train_student_config.json",
            discover_dir / "discovery_report.json",
            discover_dir / "discover_config.json",
        )
        if all(path.is_file() for path in required):
            completed_arms[name] = True
            continue
        raise FileExistsError(
            f"Refusing to overwrite incomplete outputs for arm={name}: "
            f"{train_dir}, {discover_dir}"
        )

    summary = {
        "question": "Does full-pool soft-nnPU kNN support-boundary training improve semantic-hard separation when candidate recall is not a gate?",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "single_changed_factor": "alpha_discovery_knn_boundary (0 vs configured treatment weight)",
        "treatment_weight": float(args.alpha_knn),
        "risk_mode": "soft_nnpu",
        "candidate_gate": "disabled; complete matched mixed discovery pool",
        "known_prior": float(args.known_prior),
        "softness": float(args.softness),
        "split": str(split_path),
        "teacher_checkpoint": str(args.teacher_ckpt.resolve()),
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single-seed exploratory training-objective comparison",
        "arms": {},
    }

    for name, alpha in arms.items():
        train_dir, discover_dir = output_dirs[name]
        if completed_arms[name]:
            print(f"[reuse] arm={name} complete outputs already exist", flush=True)
        else:
            train_args = load_args(args.source_train, "train_student", args.seed, args.epochs)
            train_args.split_path = str(split_path)
            train_args.work_dir = str(train_dir)
            train_args.teacher_ckpt = str(args.teacher_ckpt.resolve())
            train_args.student_ckpt = str(train_dir / "student.pt")
            train_args.alpha_discovery_knn_boundary = alpha
            train_args.alpha_discovery_rejection_knn_boundary = 0.0
            train_args.alpha_discovery_rejection_boundary = 0.0
            train_args.alpha_discovery_boundary = 0.0
            train_args.discovery_knn_risk = "soft_nnpu"
            train_args.discovery_knn_known_prior = float(args.known_prior)
            train_args.discovery_knn_softness = float(args.softness)
            train_args.discovery_knn_selection_conditional = False
            train_args.discovery_feature_candidate_gating = False
            train_args.discovery_cross_view_gating = False
            train_args.discovery_feature_candidate_space = "shared"
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
                    "discovery_knn_known_prior", "discovery_knn_softness",
                    "discovery_feature_candidate_gating", "discovery_cross_view_gating",
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
