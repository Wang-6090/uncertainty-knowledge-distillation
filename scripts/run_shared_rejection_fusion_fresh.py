"""Train one shared-backbone fresh split and audit fixed uncertainty fusion.

This runner trains only the current shared-backbone mainline. It then compares
the existing feature-rejector score with the fixed validation-calibrated
uncertainty fusion on the same frozen checkpoint. No test or mixed-pool labels
are used for training, checkpoint selection, calibration, score selection, or
method choice.
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
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2086)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
    )
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
    parser.add_argument("--tag", default="shared_rejection_fusion_fresh")
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

    teacher_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_teacher"
    train_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_baseline"
    discover_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_baseline_discover"
    fusion_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_baseline_uncertainty_fusion"
    for path in (teacher_dir, train_dir, discover_dir, fusion_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    source_student = read_json(student_source)
    if source_student.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student config must use shared_backbone")
    if source_student.get("independent_rejection_input_norm", "none") != "none":
        raise ValueError("source student config must use the registered none input norm")
    if source_student.get("alpha_vos", 0.0) != 0.0 or source_student.get("alpha_vos_rejection", 0.0) != 0.0:
        raise ValueError("source student config must have VOS disabled")
    if source_student.get("alpha_discovery_uncertainty_feature_margin", 0.0) != 0.0:
        raise ValueError("source student config must disable training-time uncertainty feature-margin")

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
    train_args.alpha_discovery_uncertainty_feature_margin = 0.0
    train_args.student_ckpt = str(student_ckpt)
    fit_student(train_args)

    discover_args = load_args(discover_source, "discover", discover_dir, options.seed)
    discover_args.split_path = str(split_path)
    discover_args.device = options.device
    discover_args.independent_rejection_encoder = True
    discover_args.independent_rejection_encoder_init = "shared_backbone"
    discover_args.student_ckpt = str(student_ckpt)
    discover_args.score_mode = "feature_rejector"
    discover_args.threshold_policy = "known_coverage"
    discover_args.target_known_coverage = 0.95
    discover_args.open_val_ratio = 0.0
    discover_args.cluster_k = "auto"
    discover_args.skip_clustering = False
    discover(discover_args)

    fusion_args = load_args(discover_source, "discover", fusion_dir, options.seed)
    fusion_args.split_path = str(split_path)
    fusion_args.device = options.device
    fusion_args.independent_rejection_encoder = True
    fusion_args.independent_rejection_encoder_init = "shared_backbone"
    fusion_args.student_ckpt = str(student_ckpt)
    fusion_args.score_mode = "feature_rejector_uncertainty_fusion"
    fusion_args.threshold_policy = "known_coverage"
    fusion_args.target_known_coverage = 0.95
    fusion_args.cluster_k = "auto"
    fusion_args.skip_clustering = False
    fusion_args.open_val_ratio = 0.0
    discover(fusion_args)

    result = {
        "question": "Does fixed validation-calibrated uncertainty fusion improve the shared-backbone fresh mainline?",
        "seed": options.seed,
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "control": "feature_rejector",
        "treatment": "feature_rejector_uncertainty_fusion",
        "fusion_rule": "fixed_0.5_score_0.5_calibrated_uncertainty",
        "calibration": "known_validation_only",
        "test_labels_used_for_training_selection_or_calibration": False,
        "training_run": str(train_dir),
        "control_discovery_run": str(discover_dir),
        "fusion_discovery_run": str(fusion_dir),
        "success_criterion": {
            "unknown_rejection_gain_at_least": 0.02,
            "fpr95_change_at_most": 0.01,
            "oscr_drop_at_most": 0.005,
            "accepted_known_accuracy_drop_at_most": 0.005,
            "no_weight_tuning": True,
        },
    }
    result_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
