"""Run a paired screening for episode-conditioned pseudo-novel rejection.

The control and treatment use one freshly generated semantic-hard split, one
teacher checkpoint, the same shared-backbone student initialization, optimizer
budget, discovery pool, checkpoint rule and known-validation calibration. The
only student-training difference is the fixed episode loss coefficient.

This runner refuses to overwrite outputs and records the materialized config
comparison. It is intended for a smoke or screening run; it never selects the
coefficient from test metrics.
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

from scripts.make_cifar100_splits import (  # noqa: E402
    build_protocols,
    read_cifar100_taxonomy,
)
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    values = read_json(path)
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def ensure_split(seed: int, split_path: Path, data_root: Path) -> Path:
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


def materialized_differences(control, treatment, ignored: set[str]) -> dict[str, Any]:
    return {
        key: [getattr(control, key), getattr(treatment, key)]
        for key in vars(control)
        if key not in ignored and getattr(control, key) != getattr(treatment, key)
    }


def validate_sources(teacher: dict[str, Any], student: dict[str, Any], discover_config: dict[str, Any]) -> None:
    if teacher.get("dataset") != "cifar100" or student.get("dataset") != "cifar100":
        raise ValueError("episode pair requires CIFAR-100 source configs")
    if student.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student must use shared_backbone rejection initialization")
    if not bool(student.get("independent_rejection_encoder")):
        raise ValueError("source student must enable the independent rejection encoder")
    if int(student.get("rejection_feature_dim", 0) or 0) <= 0:
        raise ValueError("source student must have a positive rejection feature dimension")
    if student.get("alpha_vos", 0.0) != 0.0 or student.get("alpha_vos_rejection", 0.0) != 0.0:
        raise ValueError("source student must keep VOS disabled for this screening")
    if discover_config.get("threshold_policy") != "known_coverage":
        raise ValueError("source discovery config must use known_coverage")
    if float(discover_config.get("target_known_coverage", 0.0)) != 0.95:
        raise ValueError("source discovery config must target 95% known coverage")
    if discover_config.get("rejector_training") != "nnpu":
        raise ValueError("source discovery config must use NNPU rejector training")
    if not bool(discover_config.get("matched_discovery_pool")):
        raise ValueError("source discovery config must use matched discovery pools")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2150)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--source-teacher-config", type=Path, default=Path(
        "runs/rejection_encoder_init_fresh_s2083_teacher/train_teacher_config.json"
    ))
    parser.add_argument("--source-student-config", type=Path, default=Path(
        "runs/rejection_encoder_init_fresh_s2083_shared_backbone/train_student_config.json"
    ))
    parser.add_argument("--source-discover-config", type=Path, default=Path(
        "runs/rejection_encoder_init_fresh_s2083_shared_backbone_discover/discover_config.json"
    ))
    parser.add_argument("--tag", default="episode_pseudonovel_screening")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--limit-discovery", type=int, default=None)
    parser.add_argument("--episode-alpha", type=float, default=0.1)
    options = parser.parse_args()

    if options.episode_alpha <= 0.0:
        raise ValueError("episode-alpha must be positive for the treatment arm")
    split_arg = options.split_path or Path(
        f"splits/semantic_hard/seed{options.seed}/cifar100_60_40_semantic_hard.json"
    )
    data_root = resolve(options.data_root)
    split_path = ensure_split(options.seed, resolve(split_arg), data_root)
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("semantic-hard split must contain 60 known and 40 novel classes")

    teacher_source = resolve(options.source_teacher_config)
    student_source = resolve(options.source_student_config)
    discover_source = resolve(options.source_discover_config)
    for path in (teacher_source, student_source, discover_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    teacher_values = read_json(teacher_source)
    student_values = read_json(student_source)
    discover_values = read_json(discover_source)
    validate_sources(teacher_values, student_values, discover_values)

    prefix = f"{options.tag}_s{options.seed}"
    teacher_dir = ROOT / "runs" / f"{prefix}_teacher"
    control_dir = ROOT / "runs" / f"{prefix}_control"
    treatment_dir = ROOT / "runs" / f"{prefix}_treatment"
    discover_control_dir = ROOT / "runs" / f"{prefix}_control_discover"
    discover_treatment_dir = ROOT / "runs" / f"{prefix}_treatment_discover"
    all_dirs = (teacher_dir, control_dir, treatment_dir, discover_control_dir, discover_treatment_dir)
    for path in all_dirs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    teacher_ckpt = teacher_dir / "teacher.pt"
    teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    teacher_args.device = options.device
    if options.epochs is not None:
        teacher_args.epochs = options.epochs
    for key, value in (("limit_train", options.limit_train), ("limit_val", options.limit_val), ("limit_test", options.limit_test)):
        if value is not None:
            setattr(teacher_args, key, value)
    fit_teacher(teacher_args)

    arms = {}
    for name, arm_dir, coefficient in (
        ("control", control_dir, 0.0),
        ("treatment", treatment_dir, float(options.episode_alpha)),
    ):
        args = load_args(student_source, "train_student", arm_dir, options.seed)
        args.split_path = str(split_path)
        args.teacher_ckpt = str(teacher_ckpt)
        args.student_ckpt = str(arm_dir / "student.pt")
        args.device = options.device
        args.alpha_rejection_episode_pseudonovel = coefficient
        if options.epochs is not None:
            args.epochs = options.epochs
        for key, value in (("limit_train", options.limit_train), ("limit_val", options.limit_val), ("limit_test", options.limit_test), ("limit_discovery", options.limit_discovery)):
            if value is not None:
                setattr(args, key, value)
        arms[name] = args

    differences = materialized_differences(
        arms["control"], arms["treatment"], {"work_dir", "student_ckpt", "alpha_rejection_episode_pseudonovel"}
    )
    if differences:
        raise ValueError(f"unexpected control/treatment config differences: {differences}")

    for args in arms.values():
        if options.limit_discovery is not None:
            if not bool(args.matched_discovery_pool):
                args.limit_discovery = options.limit_discovery
            else:
                # build_data_bundle requires limit_discovery=0 or exactly the
                # matched pool size.  A smoke budget therefore shrinks the
                # matched pool identically in both arms.
                args.matched_pool_size = options.limit_discovery
                args.limit_discovery = 0
        fit_student(args)

    reports = {}
    for name, student_args, discover_dir in (
        ("control", arms["control"], discover_control_dir),
        ("treatment", arms["treatment"], discover_treatment_dir),
    ):
        args = load_args(discover_source, "discover", discover_dir, options.seed)
        args.split_path = str(split_path)
        args.student_ckpt = student_args.student_ckpt
        args.device = options.device
        args.open_val_ratio = 0.0
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.save_calibration_scores = True
        if options.limit_val is not None:
            args.limit_val = options.limit_val
        if options.limit_test is not None:
            args.limit_test = options.limit_test
        if options.limit_discovery is not None:
            if bool(args.matched_discovery_pool):
                args.matched_pool_size = options.limit_discovery
                args.limit_discovery = 0
            else:
                args.limit_discovery = options.limit_discovery
        discover(args)
        reports[name] = read_json(discover_dir / "discovery_report.json")

    result = {
        "question": "Does episode-conditioned pseudo-novel rejection improve the shared-backbone pipeline?",
        "seed": options.seed,
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "control": "episode loss disabled",
        "treatment": f"episode loss coefficient {options.episode_alpha}",
        "single_changed_factor": "alpha_rejection_episode_pseudonovel",
        "open_val_ratio": 0.0,
        "target_known_coverage": 0.95,
        "test_labels_used_for_training_selection_or_calibration": False,
        "training_runs": {name: str(args.work_dir) for name, args in arms.items()},
        "discovery_runs": {
            "control": str(discover_control_dir),
            "treatment": str(discover_treatment_dir),
        },
        "materialized_training_differences": {
            "alpha_rejection_episode_pseudonovel": [0.0, float(options.episode_alpha)]
        },
        "reports": {
            name: {
                key: report.get(key)
                for key in (
                    "auroc",
                    "aupr",
                    "fpr95",
                    "oscr",
                    "known_class_accuracy_all_known",
                    "known_accept_rate",
                    "unknown_reject_rate",
                    "cluster_acc",
                    "cluster_nmi",
                    "cluster_ari",
                )
            }
            for name, report in reports.items()
        },
    }
    result_path = ROOT / "analysis" / f"{prefix}.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
