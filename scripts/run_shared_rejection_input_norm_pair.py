"""Run a preregistered shared-rejection input-normalization pair.

The pair keeps the copied pretrained ResNet-18 rejection encoder fixed and
changes only the normalization immediately before the rejection projector:
``none`` versus ``layernorm``.  Training, checkpoint selection, and threshold
calibration use no test labels.  The runner refuses to overwrite artifacts and
records the materialized configuration comparison for auditability.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2087)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("splits_cifar100_semantic_hard_seed2087/cifar100_60_40_semantic_hard.json"),
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
    parser.add_argument("--tag", default="shared_rejection_input_norm")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    options = parser.parse_args()

    split_path = ensure_split(options.seed, resolve(options.split_path), resolve(options.data_root))
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
    teacher_ckpt = teacher_dir / "teacher.pt"
    arm_names = ("control_none", "treatment_layernorm")
    arm_dirs = {
        name: ROOT / "runs" / f"{options.tag}_s{options.seed}_{name}"
        for name in arm_names
    }
    discover_dirs = {
        name: ROOT / "runs" / f"{options.tag}_s{options.seed}_{name}_discover"
        for name in arm_names
    }
    for path in (*arm_dirs.values(), *discover_dirs.values()):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if teacher_dir.exists() and not teacher_ckpt.is_file():
        raise FileExistsError(
            f"teacher output exists without a complete checkpoint: {teacher_dir}"
        )

    source_student = read_json(student_source)
    if source_student.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student config must use shared_backbone")
    if source_student.get("alpha_vos", 0.0) != 0.0 or source_student.get("alpha_vos_rejection", 0.0) != 0.0:
        raise ValueError("source student config must have VOS disabled")

    if not teacher_ckpt.is_file():
        teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, options.seed)
        teacher_args.split_path = str(split_path)
        teacher_args.teacher_ckpt = str(teacher_ckpt)
        teacher_args.device = options.device
        fit_teacher(teacher_args)

    materialized = {}
    for arm_name, norm_mode in (("control_none", "none"), ("treatment_layernorm", "layernorm")):
        train_args = load_args(student_source, "train_student", arm_dirs[arm_name], options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.independent_rejection_encoder = True
        train_args.independent_rejection_encoder_init = "shared_backbone"
        train_args.independent_rejection_input_norm = norm_mode
        train_args.device = options.device
        train_args.student_ckpt = str(arm_dirs[arm_name] / "student.pt")

        control_args = load_args(student_source, "train_student", arm_dirs["control_none"], options.seed)
        control_args.split_path = str(split_path)
        control_args.teacher_ckpt = str(teacher_ckpt)
        control_args.independent_rejection_encoder = True
        control_args.independent_rejection_encoder_init = "shared_backbone"
        control_args.independent_rejection_input_norm = "none"
        control_args.device = options.device
        control_args.student_ckpt = str(arm_dirs["control_none"] / "student.pt")
        differences = materialized_differences(
            control_args,
            train_args,
            {"work_dir", "student_ckpt", "independent_rejection_input_norm"},
        )
        if differences:
            raise ValueError(f"unexpected student config differences for {arm_name}: {differences}")
        materialized[arm_name] = {
            "input_norm": norm_mode,
            "differences_from_control": differences,
        }
        fit_student(train_args)

        discover_args = load_args(discover_source, "discover", discover_dirs[arm_name], options.seed)
        discover_args.split_path = str(split_path)
        discover_args.independent_rejection_encoder = True
        discover_args.independent_rejection_encoder_init = "shared_backbone"
        discover_args.independent_rejection_input_norm = norm_mode
        discover_args.device = options.device
        discover_args.student_ckpt = train_args.student_ckpt
        discover(discover_args)

    result = {
        "question": "Does LayerNorm improve rank and known-class retention of the shared rejection branch?",
        "seed": options.seed,
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "control": "shared_backbone + no input normalization",
        "treatment": "shared_backbone + LayerNorm before rejection projector",
        "single_changed_factor": "independent_rejection_input_norm",
        "test_labels_used_for_training_selection_or_calibration": False,
        "training_runs": {name: str(path) for name, path in arm_dirs.items()},
        "discovery_runs": {name: str(path) for name, path in discover_dirs.items()},
        "arms": materialized,
    }
    result_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
