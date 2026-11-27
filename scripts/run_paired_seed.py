"""Run a reproducible baseline/treatment paired seed from saved run configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train import discover, fit_student, fit_teacher, parse_args


def load_config(
    path: Path,
    command: str,
    work_dir: Path,
    seed: int,
    teacher_ckpt: Path | None = None,
    student_ckpt: Path | None = None,
):
    config = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.work_dir = str(work_dir)
    if teacher_ckpt is not None:
        args.teacher_ckpt = str(teacher_ckpt)
    if student_ckpt is not None:
        args.student_ckpt = str(student_ckpt)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seed", type=int, default=43)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--restore-seed43-teacher",
        action="store_true",
        help="Rebuild the overwritten seed-43 teacher from the seed-42 training config.",
    )
    parser.add_argument(
        "--restore-seed43-students",
        action="store_true",
        help="Rebuild seed-43 baseline/treatment students from the seed-42 recipes.",
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Reuse the teacher checkpoint already present in the target seed directory.",
    )
    args = parser.parse_args()

    runs = args.runs_dir
    source = args.source_seed
    seed = args.seed
    if args.restore_seed43_teacher:
        seed = 43
        source = 42
    if args.restore_seed43_students:
        seed = 43
        source = 42
    teacher_dir = runs / f"semantic_isolated_cal40_s{seed}_teacher"

    if not args.skip_teacher and not args.restore_seed43_students:
        fit_teacher(
            load_config(
                runs / f"semantic_isolated_cal40_s{source}_teacher" / "train_teacher_config.json",
                "train_teacher",
                teacher_dir,
                seed,
                teacher_ckpt=teacher_dir / "teacher.pt",
            )
        )
    if args.restore_seed43_teacher:
        print("seed 43 teacher checkpoint rebuilt from the seed-42 training recipe.")
        return
    if args.restore_seed43_students:
        for arm in ("baseline", "treatment"):
            arm_dir = runs / f"semantic_isolated_cal40_s{seed}_{arm}"
            fit_student(
                load_config(
                    runs / f"semantic_isolated_cal40_s{source}_{arm}" / "train_student_config.json",
                    "train_student",
                    arm_dir,
                    seed,
                    teacher_dir / "teacher.pt",
                    student_ckpt=arm_dir / "student.pt",
                )
            )
        print("seed 43 baseline/treatment student checkpoints rebuilt from seed-42 recipes.")
        return

    for arm in ("baseline", "treatment"):
        arm_dir = runs / f"semantic_isolated_cal40_s{seed}_{arm}"
        fit_student(
            load_config(
                runs / f"semantic_isolated_cal40_s{source}_{arm}" / "train_student_config.json",
                "train_student",
                arm_dir,
                seed,
                teacher_dir / "teacher.pt",
                student_ckpt=arm_dir / "student.pt",
            )
        )

    for arm in ("baseline", "treatment"):
        arm_dir = runs / f"semantic_isolated_cal40_s{seed}_{arm}"
        discover(
            load_config(
                runs / f"semantic_isolated_cal40_s{source}_{arm}_detect" / "discover_config.json",
                "discover",
                runs / f"semantic_isolated_cal40_s{seed}_{arm}_detect",
                seed,
                student_ckpt=arm_dir / "student.pt",
            )
        )

    print(f"seed {seed} paired training and discovery complete.")


if __name__ == "__main__":
    main()
