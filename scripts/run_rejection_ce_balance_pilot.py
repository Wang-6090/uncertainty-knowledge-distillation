"""Controlled pilot for the known-CE weight on the independent rejection branch.

The script reuses the audited encoder-freeze/rejection configuration and changes
only ``alpha_rejection_known_ce`` for the treatment arm.  Existing control
artifacts may be reused; discovery uses the same saved configuration and does
not select a checkpoint from test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


def load_args(path: Path, command: str, work_dir: Path, seed: int, epochs: int):
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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--config-seed",
        type=int,
        default=123,
        help="Seed whose audited configuration is reused; runtime seed remains --seed.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--decay-start-epoch", type=int, default=0)
    parser.add_argument("--decay-epochs", type=int, default=0)
    parser.add_argument("--final-scale", type=float, default=1.0)
    parser.add_argument(
        "--source-dir",
        default="",
        help=(
            "Existing audited run directory whose train_student_config.json "
            "and sibling discover directory provide the fixed protocol."
        ),
    )
    parser.add_argument("--tag", default="", help="Optional output-name suffix.")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train the treatment but skip discovery.",
    )
    args = parser.parse_args()

    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.is_absolute():
            source_dir = ROOT / source_dir
        source_train = source_dir / "train_student_config.json"
        source_discover = source_dir.parent / f"{source_dir.name}_discover" / "discover_config.json"
    else:
        source_train = ROOT / (
            f"runs/encoder_freeze_rejection_s{args.config_seed}_finetune_lr01/"
            "train_student_config.json"
        )
        source_discover = ROOT / (
            f"runs/encoder_freeze_rejection_s{args.config_seed}_finetune_lr01_discover/"
            "discover_config.json"
        )
    if not source_train.is_file() or not source_discover.is_file():
        raise FileNotFoundError(
            "audited encoder-freeze rejection artifacts are required: "
            f"{source_train} and {source_discover}"
        )

    weight_tag = str(args.ce_weight).replace(".", "p")
    suffix = f"s{args.seed}_ce{weight_tag}"
    if args.decay_start_epoch or args.decay_epochs or args.final_scale != 1.0:
        suffix += f"_decay{args.decay_start_epoch}x{args.decay_epochs}to{str(args.final_scale).replace('.', 'p')}"
    if args.tag:
        suffix += f"_{args.tag}"
    treatment_dir = ROOT / f"runs/rejection_ce_balance_{suffix}"
    treatment_discover_dir = ROOT / f"{treatment_dir}_discover"

    student_ckpt = treatment_dir / "student.pt"
    if not student_ckpt.is_file():
        train_args = load_args(source_train, "train_student", treatment_dir, args.seed, args.epochs)
        train_args.student_ckpt = str(student_ckpt)
        train_args.alpha_rejection_known_ce = float(args.ce_weight)
        train_args.rejection_known_ce_decay_start_epoch = int(args.decay_start_epoch)
        train_args.rejection_known_ce_decay_epochs = int(args.decay_epochs)
        train_args.rejection_known_ce_final_scale = float(args.final_scale)
        fit_student(train_args)

    if args.train_only:
        return

    report = treatment_discover_dir / "discovery_report.json"
    if not report.is_file():
        discover_args = load_args(
            source_discover,
            "discover",
            treatment_discover_dir,
            args.seed,
            args.epochs,
        )
        discover_args.student_ckpt = str(student_ckpt)
        discover(discover_args)


if __name__ == "__main__":
    main()
