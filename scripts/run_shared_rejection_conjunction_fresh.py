"""Run the pre-registered shared-backbone uncertainty-conjunction ablation.

One teacher and one shared-backbone student are trained for a fresh
semantic-hard split. The same frozen student is evaluated with the existing
fixed validation-calibrated uncertainty fusion (control) and the fixed
conservative uncertainty conjunction (treatment). No test or mixed-pool labels
are used for training, calibration, thresholding, or method selection.
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


EXCLUDED_DISCOVERY_CONFIG_KEYS = {"score_mode", "work_dir"}


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


def validate_source_protocol(student: dict[str, Any], discover_config: dict[str, Any]) -> None:
    expected_student = {
        "independent_rejection_encoder_init": "shared_backbone",
        "independent_rejection_input_norm": "none",
        "alpha_vos": 0.0,
        "alpha_vos_rejection": 0.0,
        "alpha_discovery_uncertainty_feature_margin": 0.0,
    }
    for key, expected in expected_student.items():
        if student.get(key, 0.0 if isinstance(expected, float) else None) != expected:
            raise ValueError(f"student source has unexpected {key}: {student.get(key)!r}")
    expected_discovery = {
        "discovery_pool_mode": "mixed",
        "matched_discovery_pool": True,
        "rejector_pool_selection": "all",
        "rejector_training": "nnpu",
        "rejector_strict_mixed": False,
    }
    for key, expected in expected_discovery.items():
        if discover_config.get(key) != expected:
            raise ValueError(f"discovery source has unexpected {key}: {discover_config.get(key)!r}")
    if float(student.get("alpha_kd", 0.0)) <= 0.0 or student.get("kd_mode") != "uncertainty":
        raise ValueError("fresh conjunction protocol requires the registered uncertainty-KD student")


def assert_paired_discovery_configs(control_dir: Path, treatment_dir: Path) -> None:
    control = read_json(control_dir / "discover_config.json")
    treatment = read_json(treatment_dir / "discover_config.json")
    if control.get("score_mode") != "feature_rejector_uncertainty_fusion":
        raise ValueError("control discovery config has the wrong score mode")
    if treatment.get("score_mode") != "feature_rejector_uncertainty_conjunction":
        raise ValueError("treatment discovery config has the wrong score mode")
    keys = set(control) | set(treatment)
    differences = {
        key: (control.get(key), treatment.get(key))
        for key in sorted(keys - EXCLUDED_DISCOVERY_CONFIG_KEYS)
        if control.get(key) != treatment.get(key)
    }
    if differences:
        raise ValueError(f"control/treatment discovery configs differ beyond score_mode: {differences}")


def validate_report(path: Path, expected_score_mode: str) -> dict[str, Any]:
    report = read_json(path)
    protocol = report.get("rejector_protocol", {})
    if protocol.get("score_mode") != expected_score_mode:
        raise ValueError(f"{path}: unexpected score mode in report")
    if protocol.get("test_labels_used_for_training_or_calibration") is not False:
        raise ValueError(f"{path}: test labels were used for training/calibration")
    if protocol.get("hard_labels_used") is not False:
        raise ValueError(f"{path}: mixed-pool hard labels were used")
    if report.get("threshold_policy") != "known_val_known_coverage":
        raise ValueError(f"{path}: threshold is not known-validation coverage")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2090)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
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
    parser.add_argument("--tag", default="shared_rejection_uncertainty_conjunction_fresh")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    options = parser.parse_args()

    split_arg = options.split_path or Path(
        f"splits_cifar100_semantic_hard_seed{options.seed}/"
        "cifar100_60_40_semantic_hard.json"
    )
    split_path = make_split(options.seed, resolve(split_arg), resolve(options.data_root))
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("semantic-hard split must contain 60 known and 40 novel classes")

    teacher_source = resolve(options.source_teacher_config)
    student_source = resolve(options.source_student_config)
    discover_source = resolve(options.source_discover_config)
    for path in (teacher_source, student_source, discover_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_student = read_json(student_source)
    source_discover = read_json(discover_source)
    validate_source_protocol(source_student, source_discover)

    teacher_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_teacher"
    train_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_baseline"
    control_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_fusion"
    treatment_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_conjunction"
    all_dirs = (teacher_dir, train_dir, control_dir, treatment_dir)
    for path in all_dirs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    teacher_ckpt = teacher_dir / "teacher.pt"
    teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.device = options.device
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    student_ckpt = train_dir / "student.pt"
    train_args = load_args(student_source, "train_student", train_dir, options.seed)
    train_args.split_path = str(split_path)
    train_args.device = options.device
    train_args.teacher_ckpt = str(teacher_ckpt)
    train_args.independent_rejection_encoder = True
    train_args.independent_rejection_encoder_init = "shared_backbone"
    train_args.independent_rejection_input_norm = "none"
    train_args.alpha_discovery_uncertainty_feature_margin = 0.0
    train_args.student_ckpt = str(student_ckpt)
    fit_student(train_args)

    def run_discovery_arm(work_dir: Path, score_mode: str) -> None:
        args = load_args(discover_source, "discover", work_dir, options.seed)
        args.split_path = str(split_path)
        args.device = options.device
        args.independent_rejection_encoder = True
        args.independent_rejection_encoder_init = "shared_backbone"
        args.independent_rejection_input_norm = "none"
        args.student_ckpt = str(student_ckpt)
        args.score_mode = score_mode
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.save_calibration_scores = True
        discover(args)

    run_discovery_arm(control_dir, "feature_rejector_uncertainty_fusion")
    run_discovery_arm(treatment_dir, "feature_rejector_uncertainty_conjunction")
    assert_paired_discovery_configs(control_dir, treatment_dir)
    control = validate_report(control_dir / "discovery_report.json", "feature_rejector_uncertainty_fusion")
    treatment = validate_report(treatment_dir / "discovery_report.json", "feature_rejector_uncertainty_conjunction")

    metrics = (
        "auroc",
        "aupr",
        "fpr95",
        "oscr",
        "unknown_reject_rate",
        "known_accept_rate",
        "known_class_accuracy_after_accept",
    )
    delta = {key: float(treatment[key]) - float(control[key]) for key in metrics}
    result = {
        "protocol": "shared_rejection_uncertainty_conjunction_s2090_20261024",
        "question": "Does fixed uncertainty conjunction improve fixed validation-calibrated uncertainty fusion?",
        "seed": options.seed,
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "control": "feature_rejector_uncertainty_fusion",
        "treatment": "feature_rejector_uncertainty_conjunction",
        "treatment_rule": "min(z_feature_rejector, z_calibrated_uncertainty_error)",
        "calibration": "known_validation_only",
        "test_labels_used_for_training_selection_or_calibration": False,
        "control_discovery_run": str(control_dir),
        "treatment_discovery_run": str(treatment_dir),
        "control_metrics": {key: control[key] for key in metrics},
        "treatment_metrics": {key: treatment[key] for key in metrics},
        "delta_treatment_minus_control": delta,
        "joint_criterion": {
            "unknown_reject_rate_gain_at_least": 0.02,
            "fpr95_delta_at_most": 0.01,
            "oscr_delta_at_least": -0.005,
            "accepted_known_accuracy_delta_at_least": -0.005,
        },
        "joint_criterion_pass": (
            delta["unknown_reject_rate"] >= 0.02
            and delta["fpr95"] <= 0.01
            and delta["oscr"] >= -0.005
            and delta["known_class_accuracy_after_accept"] >= -0.005
        ),
    }
    result_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
