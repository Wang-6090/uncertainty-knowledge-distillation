"""Train one fresh semantic-hard model and compare frozen rejectors.

The student, split, and discovery pool are shared by both detector arms. The
only detector factor is all-pool fitting versus prototype-boundary weighted
pool fitting. This is an exploratory seed replication; it is not a clean
confirmatory test because the candidate rule was prioritized after earlier
exploratory audits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


METRIC_KEYS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "known_accept_rate",
    "unknown_reject_rate",
    "known_rejected",
    "unknown_false_accept",
    "known_class_accuracy_after_accept",
    "calibration_known_accept_rate",
)


def load_config(path: Path, command: str, seed: int, epochs: int):
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
    parser.add_argument("--source-teacher", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-discover", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--student-epochs", type=int, default=4)
    parser.add_argument("--output-prefix", default="semantic_hard_rejector_seed2026")
    args = parser.parse_args()

    sources = (args.source_teacher.resolve(), args.source_train.resolve(), args.source_discover.resolve())
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("All source config paths must exist")

    teacher_dir = ROOT / "runs" / f"{args.output_prefix}_teacher"
    student_dir = ROOT / "runs" / f"{args.output_prefix}_student"
    discover_dirs = {
        "all_pool": ROOT / "runs" / f"{args.output_prefix}_all_pool",
        "prototype_boundary_weighted": ROOT / "runs" / f"{args.output_prefix}_prototype_boundary_weighted",
    }
    outputs = (teacher_dir, student_dir, *discover_dirs.values())
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    split_path = ROOT / "splits_cifar100_protocols" / "cifar100_60_40_semantic_hard.json"
    teacher_args = load_config(sources[0], "train_teacher", args.seed, args.teacher_epochs)
    teacher_args.split_path = str(split_path)
    teacher_args.work_dir = str(teacher_dir)
    teacher_args.teacher_ckpt = str(teacher_dir / "teacher.pt")
    print(f"[teacher] seed={args.seed} output={teacher_dir}", flush=True)
    fit_teacher(teacher_args)

    train_args = load_config(sources[1], "train_student", args.seed, args.student_epochs)
    train_args.split_path = str(split_path)
    train_args.work_dir = str(student_dir)
    train_args.teacher_ckpt = str(teacher_dir / "teacher.pt")
    train_args.student_ckpt = str(student_dir / "student.pt")
    print(f"[student] seed={args.seed} output={student_dir}", flush=True)
    fit_student(train_args)

    arms = {
        "all_pool": {"rejector_pool_selection": "all"},
        "prototype_boundary_weighted": {
            "rejector_pool_selection": "weighted_boundary",
            "rejector_candidate_mode": "prototype_boundary",
            "rejector_candidate_ratio": 0.25,
            "rejector_candidate_weight": 2.0,
        },
    }
    summary = {
        "question": "Does prototype-boundary weighted rejector fitting improve semantic-hard detection on a fresh training seed?",
        "seed": int(args.seed),
        "teacher_epochs": int(args.teacher_epochs),
        "student_epochs": int(args.student_epochs),
        "single_detector_factor": "rejector pool treatment (all vs weighted prototype-boundary candidates)",
        "candidate_ratio": 0.25,
        "candidate_weight": 2.0,
        "threshold_calibration": "known validation only, target 95% known coverage",
        "test_labels_used_for_training_calibration_or_method_selection": False,
        "evidence_level": "exploratory seed replication; candidate rule was prioritized after prior exploratory work",
        "teacher_dir": str(teacher_dir),
        "student_dir": str(student_dir),
        "arms": {},
    }
    for name, overrides in arms.items():
        run_args = load_config(sources[2], "discover", args.seed, args.student_epochs)
        run_args.split_path = str(split_path)
        run_args.work_dir = str(discover_dirs[name])
        run_args.student_ckpt = str(student_dir / "student.pt")
        run_args.score_mode = "feature_rejector"
        run_args.rejector_pool_selection = overrides["rejector_pool_selection"]
        if name == "prototype_boundary_weighted":
            run_args.rejector_candidate_mode = overrides["rejector_candidate_mode"]
            run_args.rejector_candidate_ratio = overrides["rejector_candidate_ratio"]
            run_args.rejector_candidate_weight = overrides["rejector_candidate_weight"]
        run_args.skip_clustering = True
        print(f"[discover] arm={name} output={discover_dirs[name]}", flush=True)
        discover(run_args)
        report_path = discover_dirs[name] / "discovery_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = json.loads((discover_dirs[name] / "discover_config.json").read_text(encoding="utf-8"))
        summary["arms"][name] = {
            "work_dir": str(discover_dirs[name]),
            "config": {
                key: config.get(key)
                for key in (
                    "seed", "student_ckpt", "discovery_pool_mode", "matched_pool_size",
                    "matched_known_prior", "score_mode", "rejector_training",
                    "rejector_pool_selection", "rejector_candidate_mode",
                    "rejector_candidate_ratio", "rejector_candidate_weight",
                    "rejector_feature_mode", "threshold_policy", "target_known_coverage",
                )
            },
            "metrics": {key: report.get(key) for key in METRIC_KEYS},
            "score_distribution": report.get("score_distribution", {}),
        }

    output_path = ROOT / "analysis" / f"{args.output_prefix}_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
