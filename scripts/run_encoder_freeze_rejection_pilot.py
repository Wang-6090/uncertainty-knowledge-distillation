"""Paired pilot: fine-tune versus freeze the pretrained encoder.

The two arms reuse the same seed, teacher, data protocol, rejection branch,
losses, and detector. BatchNorm running statistics are frozen in both arms;
only the encoder learning-rate scale changes (0.1 versus 0.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args


def load_args(config_path: Path, command: str, work_dir: Path, seed: int):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    baseline_dir = ROOT / f"runs/rejection_supcon_s{args.seed}_baseline"
    train_config = baseline_dir / "train_student_config.json"
    discover_config = (
        ROOT / f"runs/rejection_supcon_s{args.seed}_baseline_discover/discover_config.json"
    )
    if not train_config.is_file() or not discover_config.is_file():
        raise FileNotFoundError("Audited matched baseline configs are required")
    source = json.loads(train_config.read_text(encoding="utf-8"))
    teacher = Path(source["teacher_ckpt"])
    if not teacher.is_absolute():
        teacher = ROOT / teacher
    if not teacher.is_file():
        raise FileNotFoundError(teacher)

    for arm, scale in (("finetune_lr01", 0.1), ("frozen_encoder", 0.0)):
        student_dir = ROOT / f"runs/encoder_freeze_rejection_s{args.seed}_{arm}"
        discover_dir = ROOT / f"runs/encoder_freeze_rejection_s{args.seed}_{arm}_discover"
        student_ckpt = student_dir / "student.pt"
        report = discover_dir / "discovery_report.json"
        if report.is_file() and student_ckpt.is_file():
            print(f"Reusing completed arm: {report}")
            continue

        if not student_ckpt.is_file():
            train_args = load_args(train_config, "train_student", student_dir, args.seed)
            train_args.teacher_ckpt = str(teacher)
            train_args.student_ckpt = str(student_ckpt)
            train_args.encoder_lr_scale = scale
            train_args.freeze_bn_stats = True
            fit_student(train_args)

        discover_args = load_args(
            discover_config, "discover", discover_dir, args.seed
        )
        discover_args.student_ckpt = str(student_ckpt)
        discover(discover_args)


if __name__ == "__main__":
    main()
