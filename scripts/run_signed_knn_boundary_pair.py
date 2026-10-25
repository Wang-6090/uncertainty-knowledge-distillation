"""Compare outward PU and signed-support PU kNN boundary objectives.

Both arms train on one fresh semantic-hard split with the same teacher,
student recipe, candidate event, and validation-only calibration.  The sole
algorithmic factor is ``discovery_knn_boundary_objective``:

* ``outward``: the existing PU kNN risk, which only pushes inferred unknown
  risk outside known support;
* ``signed_support``: the new opt-in objective, which additionally pulls
  labelled known samples back inside their own class-conditional support.

No mixed-pool or test labels are used for training, selection, calibration,
or method choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_cifar100_splits import build_protocols, read_cifar100_taxonomy  # noqa: E402
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_args(path: Path, command: str, work_dir: Path, seed: int, device: str):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    args.device = device
    return args


def make_split(seed: int, split_path: Path, data_root: Path) -> Path:
    if split_path.exists():
        split = read_json(split_path)
        if split.get("metadata", {}).get("seed") != seed:
            raise ValueError(f"existing split seed does not match: {split_path}")
        if split.get("protocol") != "semantic_hard":
            raise ValueError(f"existing split is not semantic_hard: {split_path}")
        return split_path
    fine_names, coarse_names, mapping = read_cifar100_taxonomy(data_root)
    split = build_protocols(fine_names, coarse_names, mapping, seed)["semantic_hard"]
    write_json(split_path, split)
    return split_path


def assert_same_configs(control_dir: Path, treatment_dir: Path) -> None:
    control = read_json(control_dir / "train_student_config.json")
    treatment = read_json(treatment_dir / "train_student_config.json")
    excluded = {
        "work_dir",
        "student_ckpt",
        "discovery_knn_boundary_objective",
        "best_epoch",
        "best_val_known_acc",
        "best_checkpoint_selection_score",
        "best_open_checkpoint_stats",
        "prototype_init_stats",
        "prototype_refresh_history",
    }
    differences = {
        key: (control.get(key), treatment.get(key))
        for key in sorted(set(control) | set(treatment))
        if key not in excluded and control.get(key) != treatment.get(key)
    }
    if differences:
        raise ValueError(f"paired train configs differ beyond the objective: {differences}")
    if control.get("discovery_knn_boundary_objective") != "outward":
        raise ValueError("control did not save the outward objective")
    if treatment.get("discovery_knn_boundary_objective") != "signed_support":
        raise ValueError("treatment did not save the signed-support objective")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2091)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--source-teacher-config",
        type=Path,
        default=Path("runs/rejection_encoder_init_fresh_s2083_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--source-student-config",
        type=Path,
        default=Path("runs/rejection_encoder_init_fresh_s2083_shared_backbone/train_student_config.json"),
    )
    parser.add_argument(
        "--source-discover-config",
        type=Path,
        default=Path("runs/rejection_encoder_init_fresh_s2083_shared_backbone_discover/discover_config.json"),
    )
    parser.add_argument("--alpha-knn", type=float, default=0.1)
    parser.add_argument("--tag", default="signed_knn_boundary_pair")
    options = parser.parse_args()
    if options.alpha_knn <= 0.0:
        raise ValueError("--alpha-knn must be positive")

    split_arg = options.split_path or Path(
        f"splits_cifar100_semantic_hard_seed{options.seed}/"
        "cifar100_60_40_semantic_hard.json"
    )
    split_path = make_split(options.seed, resolve(split_arg), resolve(options.data_root))
    teacher_source = resolve(options.source_teacher_config)
    student_source = resolve(options.source_student_config)
    discover_source = resolve(options.source_discover_config)
    for path in (teacher_source, student_source, discover_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    teacher_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_teacher"
    arm_dirs = {
        "outward": ROOT / "runs" / f"{options.tag}_s{options.seed}_outward",
        "signed_support": ROOT / "runs" / f"{options.tag}_s{options.seed}_signed_support",
    }
    discover_dirs = {
        name: ROOT / "runs" / f"{options.tag}_s{options.seed}_{name}_discover"
        for name in arm_dirs
    }
    for path in (teacher_dir, *arm_dirs.values(), *discover_dirs.values()):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    teacher_ckpt = teacher_dir / "teacher.pt"
    teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, options.seed, options.device)
    teacher_args.split_path = str(split_path)
    teacher_args.epochs = int(options.epochs)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    reports = {}
    for objective, train_dir in arm_dirs.items():
        train_args = load_args(student_source, "train_student", train_dir, options.seed, options.device)
        train_args.split_path = str(split_path)
        train_args.epochs = int(options.epochs)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.alpha_discovery_knn_boundary = float(options.alpha_knn)
        train_args.alpha_discovery_rejection_knn_boundary = 0.0
        train_args.alpha_discovery_rejection_boundary = 0.0
        train_args.alpha_discovery_boundary = 0.0
        train_args.discovery_knn_risk = "hard_nnpu"
        train_args.discovery_knn_boundary_objective = objective
        train_args.discovery_knn_selection_conditional = True
        train_args.discovery_single_view_candidate_gating = True
        train_args.discovery_cross_view_gating = False
        train_args.discovery_feature_candidate_gating = True
        train_args.discovery_feature_candidate_space = "shared_boundary"
        fit_student(train_args)

        discover_dir = discover_dirs[objective]
        discover_args = load_args(discover_source, "discover", discover_dir, options.seed, options.device)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.open_val_ratio = 0.0
        discover_args.skip_clustering = True
        discover(discover_args)
        report_path = discover_dir / "discovery_report.json"
        reports[objective] = {
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "train_config": read_json(train_dir / "train_student_config.json"),
            "discover_config": read_json(discover_dir / "discover_config.json"),
            "metrics": read_json(report_path),
        }

    assert_same_configs(arm_dirs["outward"], arm_dirs["signed_support"])
    metric_keys = (
        "auroc", "aupr", "fpr95", "oscr", "unknown_reject_rate",
        "known_accept_rate", "known_class_accuracy_after_accept",
    )
    result = {
        "protocol": f"signed_knn_boundary_pair_s{options.seed}_20261024",
        "question": "Does signed class-conditional support supervision improve semantic-hard open-set separation?",
        "seed": int(options.seed),
        "epochs": int(options.epochs),
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "discovery_knn_boundary_objective",
        "control": "outward",
        "treatment": "signed_support",
        "treatment_weight": float(options.alpha_knn),
        "candidate_protocol": "selection-conditional single-view shared-boundary gate",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single-seed exploratory training-objective comparison",
        "arms": {
            name: {
                "train_dir": value["train_dir"],
                "discover_dir": value["discover_dir"],
                "train_config": {
                    key: value["train_config"].get(key)
                    for key in (
                        "seed", "epochs", "split_path", "alpha_discovery_knn_boundary",
                        "discovery_knn_risk", "discovery_knn_boundary_objective",
                        "discovery_knn_selection_conditional",
                        "discovery_single_view_candidate_gating",
                        "discovery_cross_view_gating", "discovery_feature_candidate_space",
                    )
                },
                "discover_config": {
                    key: value["discover_config"].get(key)
                    for key in (
                        "split_path", "score_mode", "threshold_policy",
                        "target_known_coverage", "open_val_ratio",
                    )
                },
                "metrics": {key: value["metrics"].get(key) for key in metric_keys},
                "score_distribution": value["metrics"].get("score_distribution", {}),
            }
            for name, value in reports.items()
        },
    }
    result_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
