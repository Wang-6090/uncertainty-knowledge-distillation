"""Paired pilot for a class-conditional PU boundary on rejection features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs and write an isolated audit run.",
    )
    args = parser.parse_args()

    source_train = ROOT / f"runs/encoder_freeze_rejection_s{args.seed}_finetune_lr01/train_student_config.json"
    source_discover = ROOT / f"runs/encoder_freeze_rejection_s{args.seed}_finetune_lr01_discover/discover_config.json"
    baseline_student = ROOT / f"runs/encoder_freeze_rejection_s{args.seed}_finetune_lr01/student.pt"
    if not source_train.is_file() or not source_discover.is_file() or not baseline_student.is_file():
        raise FileNotFoundError("audited encoder fine-tuning pilot artifacts are required")

    suffix = ""
    if args.epochs is not None:
        suffix = f"_e{args.epochs}"
    treatment_dir = ROOT / f"runs/rejection_boundary_s{args.seed}_w{str(args.weight).replace('.', 'p')}{suffix}"
    treatment_student = treatment_dir / "student.pt"
    treatment_discover = treatment_dir.parent / f"{treatment_dir.name}_discover"

    if not treatment_student.is_file():
        train_args = load_args(source_train, "train_student", treatment_dir, args.seed)
        train_args.student_ckpt = str(treatment_student)
        train_args.alpha_discovery_rejection_boundary = float(args.weight)
        train_args.discovery_boundary_mode = "pu"
        train_args.alpha_discovery_rejection_pu_feature_margin = 0.0
        if args.epochs is not None:
            train_args.epochs = int(args.epochs)
        fit_student(train_args)

    if not (treatment_discover / "discovery_report.json").is_file():
        discover_args = load_args(source_discover, "discover", treatment_discover, args.seed)
        discover_args.student_ckpt = str(treatment_student)
        discover(discover_args)


if __name__ == "__main__":
    main()
