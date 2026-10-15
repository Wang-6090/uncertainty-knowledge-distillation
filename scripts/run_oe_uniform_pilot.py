"""Run a matched CIFAR-10 Outlier Exposure uniform-loss pilot."""

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
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--oe-weight", type=float, default=0.1)
    args = parser.parse_args()

    baseline_dir = ROOT / f"runs/nnpu_margin_pair_s{args.seed}_baseline"
    source_config = baseline_dir / "train_student_config.json"
    discovery_source = ROOT / f"runs/nnpu_margin_pair_s{args.seed}_baseline_discover"
    discovery_config = discovery_source / "discover_config.json"
    if not source_config.is_file() or not discovery_config.is_file():
        raise FileNotFoundError("Matched nnPU baseline configs are required")

    source = json.loads(source_config.read_text(encoding="utf-8"))
    teacher_ckpt = Path(source["teacher_ckpt"])
    if not teacher_ckpt.is_absolute():
        teacher_ckpt = ROOT / teacher_ckpt
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(teacher_ckpt)

    tag = f"nnpu_oe_uniform_pilot_s{args.seed}"
    student_dir = ROOT / "runs" / tag
    discover_dir = ROOT / "runs" / f"{tag}_discover"
    train_args = load_args(source_config, "train_student", student_dir, args.seed)
    train_args.teacher_ckpt = str(teacher_ckpt)
    train_args.student_ckpt = str(student_dir / "student.pt")
    train_args.outlier_dataset = "cifar10"
    train_args.outlier_data_root = str(ROOT / "data")
    train_args.outlier_download = False
    train_args.alpha_outlier_uniform = float(args.oe_weight)
    fit_student(train_args)

    discover_args = load_args(discovery_config, "discover", discover_dir, args.seed)
    discover_args.student_ckpt = train_args.student_ckpt
    discover(discover_args)


if __name__ == "__main__":
    main()
