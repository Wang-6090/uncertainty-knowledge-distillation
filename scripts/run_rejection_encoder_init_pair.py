"""Compare scratch CNN vs copied pretrained backbone for an independent rejector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("splits_cifar100_semantic_hard_seed2042/cifar100_60_40_semantic_hard.json"),
    )
    parser.add_argument(
        "--source-teacher-config",
        type=Path,
        default=Path("runs/semantic_hard_rejector_seed2028_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--source-student-config",
        type=Path,
        default=Path("runs/semantic_hard_vos_boundary_s2040_e4_control/train_student_config.json"),
    )
    parser.add_argument(
        "--source-discover-config",
        type=Path,
        default=Path("runs/semantic_hard_vos_boundary_s2040_e4_control_discover/discover_config.json"),
    )
    parser.add_argument("--tag", default="rejection_encoder_init")
    options = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else (ROOT / p).resolve()
    split_path = resolve(options.split_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed:
        raise ValueError("split metadata seed does not match requested seed")

    teacher_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    if teacher_dir.exists():
        raise FileExistsError(f"refusing to overwrite {teacher_dir}")
    teacher_args = load_args(resolve(options.source_teacher_config), "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    base_config = json.loads(resolve(options.source_student_config).read_text(encoding="utf-8"))
    if base_config.get("alpha_vos", 0.0) != 0.0 or base_config.get("alpha_vos_rejection", 0.0) != 0.0:
        raise ValueError("source student config must have VOS disabled")
    discovery_source = resolve(options.source_discover_config)
    for init_mode in ("scratch_cnn", "shared_backbone"):
        run_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_{init_mode}"
        discover_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_{init_mode}_discover"
        if run_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for {init_mode}")
        train_args = load_args(resolve(options.source_student_config), "train_student", run_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.independent_rejection_encoder = True
        train_args.independent_rejection_encoder_init = init_mode
        train_args.student_ckpt = str(run_dir / "student.pt")
        fit_student(train_args)

        discover_args = load_args(discovery_source, "discover", discover_dir, options.seed)
        discover_args.split_path = str(split_path)
        discover_args.independent_rejection_encoder = True
        discover_args.independent_rejection_encoder_init = init_mode
        discover_args.student_ckpt = train_args.student_ckpt
        discover(discover_args)

    print(json.dumps({"seed": options.seed, "split": str(split_path),
                      "teacher": str(teacher_ckpt),
                      "treatments": ["scratch_cnn", "shared_backbone"]}, indent=2))


if __name__ == "__main__":
    main()
