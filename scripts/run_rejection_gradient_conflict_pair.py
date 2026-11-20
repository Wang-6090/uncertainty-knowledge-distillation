"""Run a paired screening for rejection gradient conflict routing.

The two student arms share one fresh semantic-hard split, one split-matched
teacher, initialization, optimizer budget, discovery pool and evaluation
protocol. The only training difference is whether auxiliary gradients in the
independent rejection branch are projected away from the known-support
gradient when the two objectives conflict.

The runner refuses to overwrite outputs and supports a preflight-only mode so
the materialized configuration can be audited before starting training.
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

from scripts.run_episode_pseudonovel_pair import (  # noqa: E402
    ensure_split,
    load_args,
    materialized_differences,
    read_json,
    resolve,
    validate_report,
    validate_sources,
    write_json,
)
from train import discover, fit_student, fit_teacher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2152)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument(
        "--source-teacher-config",
        type=Path,
        default=Path(
            "runs/rejection_encoder_init_fresh_s2083_teacher/train_teacher_config.json"
        ),
    )
    parser.add_argument(
        "--source-student-config",
        type=Path,
        default=Path(
            "runs/rejection_encoder_init_fresh_s2083_shared_backbone/train_student_config.json"
        ),
    )
    parser.add_argument(
        "--source-discover-config",
        type=Path,
        default=Path(
            "runs/rejection_encoder_init_fresh_s2083_shared_backbone_discover/discover_config.json"
        ),
    )
    parser.add_argument("--tag", default="rejection_gradient_conflict_screening")
    parser.add_argument(
        "--treatment-mode",
        choices=("project_auxiliary", "project_auxiliary_componentwise"),
        default="project_auxiliary_componentwise",
        help="Opt-in routing rule used by the treatment arm.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--limit-discovery", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    options = parser.parse_args()

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
    if student_values.get("alpha_rejection_episode_pseudonovel", 0.0) != 0.0:
        raise ValueError("source student must keep episode pseudo-novel rejection disabled")

    prefix = f"{options.tag}_s{options.seed}"
    teacher_dir = ROOT / "runs" / f"{prefix}_teacher"
    control_dir = ROOT / "runs" / f"{prefix}_control"
    treatment_dir = ROOT / "runs" / f"{prefix}_treatment"
    discover_control_dir = ROOT / "runs" / f"{prefix}_control_discover"
    discover_treatment_dir = ROOT / "runs" / f"{prefix}_treatment_discover"
    all_dirs = (
        teacher_dir,
        control_dir,
        treatment_dir,
        discover_control_dir,
        discover_treatment_dir,
    )
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
    for key, value in (
        ("limit_train", options.limit_train),
        ("limit_val", options.limit_val),
        ("limit_test", options.limit_test),
    ):
        if value is not None:
            setattr(teacher_args, key, value)

    arms = {}
    for name, arm_dir, mode in (
        ("control", control_dir, "off"),
        ("treatment", treatment_dir, options.treatment_mode),
    ):
        args = load_args(student_source, "train_student", arm_dir, options.seed)
        args.split_path = str(split_path)
        args.teacher_ckpt = str(teacher_ckpt)
        args.student_ckpt = str(arm_dir / "student.pt")
        args.device = options.device
        args.alpha_rejection_episode_pseudonovel = 0.0
        args.rejection_gradient_conflict_mode = mode
        if options.epochs is not None:
            args.epochs = options.epochs
        for key, value in (
            ("limit_train", options.limit_train),
            ("limit_val", options.limit_val),
            ("limit_test", options.limit_test),
            ("limit_discovery", options.limit_discovery),
        ):
            if value is not None:
                setattr(args, key, value)
        arms[name] = args

    differences = materialized_differences(
        arms["control"],
        arms["treatment"],
        {
            "work_dir",
            "student_ckpt",
            "rejection_gradient_conflict_mode",
        },
    )
    if differences:
        raise ValueError(f"unexpected control/treatment config differences: {differences}")
    if not arms["control"].independent_rejection_encoder:
        raise ValueError("paired screening requires an independent rejection encoder")
    if arms["control"].alpha_rejection_episode_pseudonovel != 0.0:
        raise ValueError("both arms must disable episode pseudo-novel rejection")

    preflight = {
        "seed": options.seed,
        "split_path": str(split_path),
        "teacher_source": str(teacher_source),
        "student_source": str(student_source),
        "discover_source": str(discover_source),
        "control_mode": arms["control"].rejection_gradient_conflict_mode,
        "treatment_mode": arms["treatment"].rejection_gradient_conflict_mode,
        "materialized_training_differences": {
            "rejection_gradient_conflict_mode": [
                "off",
                options.treatment_mode,
            ]
        },
        "output_directories": {name: str(path) for name, path in (
            ("teacher", teacher_dir),
            ("control", control_dir),
            ("treatment", treatment_dir),
            ("control_discover", discover_control_dir),
            ("treatment_discover", discover_treatment_dir),
        )},
    }
    if options.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return

    fit_teacher(teacher_args)
    for args in arms.values():
        if options.limit_discovery is not None and args.matched_discovery_pool:
            args.matched_pool_size = options.limit_discovery
            args.limit_discovery = 0
        fit_student(args)

    reports: dict[str, dict[str, Any]] = {}
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
        args.skip_clustering = False
        args.cluster_k = "auto"
        if options.limit_val is not None:
            args.limit_val = options.limit_val
        if options.limit_test is not None:
            args.limit_test = options.limit_test
        if options.limit_discovery is not None and args.matched_discovery_pool:
            args.matched_pool_size = options.limit_discovery
            args.limit_discovery = 0
        discover(args)
        reports[name] = validate_report(discover_dir / "discovery_report.json")

    result = {
        **preflight,
        "question": "Does rejection gradient conflict routing improve semantic-hard discovery?",
        "open_val_ratio": 0.0,
        "target_known_coverage": 0.95,
        "test_labels_used_for_training_selection_or_calibration": False,
        "training_runs": {name: str(args.work_dir) for name, args in arms.items()},
        "discovery_runs": {
            "control": str(discover_control_dir),
            "treatment": str(discover_treatment_dir),
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
