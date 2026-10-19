"""Train a split-matched teacher and run a fixed CE-balance pair on a new split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import fit_teacher, parse_args  # noqa: E402


def load_config(path: Path, command: str, seed: int, epochs: int, work_dir: Path):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.epochs = epochs
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument(
        "--split-path",
        default="splits_cifar100_semantic_hard_seed2040/cifar100_60_40_semantic_hard.json",
    )
    parser.add_argument(
        "--source-teacher",
        default="runs/semantic_hard_rejector_seed2028_teacher/train_teacher_config.json",
    )
    parser.add_argument(
        "--source-train",
        default="runs/semantic_hard_knn_support_train_s2029_e4_budgeted_baseline/train_student_config.json",
    )
    parser.add_argument(
        "--source-discover",
        default="runs/semantic_hard_knn_support_train_s2029_e4_budgeted_baseline_discover/discover_config.json",
    )
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--student-epochs", type=int, default=4)
    parser.add_argument("--tag", default="independent_hard_split")
    args = parser.parse_args()

    split_path = Path(args.split_path)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    split_path = split_path.resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    teacher_dir = ROOT / "runs" / f"semantic_hard_ce_balance_s{args.seed}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    if teacher_dir.exists():
        raise FileExistsError(f"Refusing to overwrite teacher output: {teacher_dir}")

    teacher_args = load_config(
        ROOT / args.source_teacher,
        "train_teacher",
        args.seed,
        args.teacher_epochs,
        teacher_dir,
    )
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    print(f"[teacher] seed={args.seed} split={split_path}", flush=True)
    fit_teacher(teacher_args)

    runner = ROOT / "scripts" / "run_rejection_ce_balance_pilot.py"
    for weight in (1.0, 0.25):
        command = [
            sys.executable,
            str(runner),
            "--source-dir",
            str((ROOT / args.source_train).parent),
            "--seed",
            str(args.seed),
            "--config-seed",
            "2029",
            "--epochs",
            str(args.student_epochs),
            "--ce-weight",
            str(weight),
            "--tag",
            args.tag,
            "--split-path",
            str(split_path),
            "--teacher-ckpt",
            str(teacher_ckpt),
        ]
        print(f"[student+discovery] CE={weight}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
