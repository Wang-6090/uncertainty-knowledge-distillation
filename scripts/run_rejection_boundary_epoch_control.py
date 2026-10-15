"""Equal-epoch control for the rejection-boundary pilot.

This is an audit script: it trains baseline and treatment with the same seed,
configuration, and fixed epoch count, then runs the same discovery protocol.
It does not use test labels for selection.
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
        help="Seed whose audited config is reused; runtime seed remains --seed.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument(
        "--risk-mode",
        choices=["hard_nnpu", "soft_nnpu"],
        default="hard_nnpu",
        help="PU correction used by the rejection-boundary loss.",
    )
    parser.add_argument("--boundary-warmup-epochs", type=int, default=0)
    parser.add_argument("--boundary-ramp-epochs", type=int, default=0)
    parser.add_argument(
        "--boundary-stop-epoch",
        type=int,
        default=0,
        help="Disable the boundary loss after this 1-based epoch; 0 keeps it active.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for output directories so multiple schedules can coexist.",
    )
    parser.add_argument(
        "--treatment-only",
        action="store_true",
        help="Reuse an existing equal-epoch baseline and only train/evaluate treatment.",
    )
    args = parser.parse_args()

    source_train = ROOT / f"runs/encoder_freeze_rejection_s{args.config_seed}_finetune_lr01/train_student_config.json"
    source_discover = ROOT / f"runs/encoder_freeze_rejection_s{args.config_seed}_finetune_lr01_discover/discover_config.json"
    if not source_train.is_file() or not source_discover.is_file():
        raise FileNotFoundError("audited encoder fine-tuning pilot artifacts are required")

    tag = f"s{args.seed}_e{args.epochs}"
    if args.risk_mode != "hard_nnpu":
        tag += f"_{args.risk_mode}"
    if args.tag:
        tag += f"_{args.tag}"
    baseline_dir = ROOT / f"runs/rejection_boundary_epoch_control_{tag}_baseline"
    treatment_dir = ROOT / f"runs/rejection_boundary_epoch_control_{tag}_w{str(args.weight).replace('.', 'p')}"
    baseline_discover_dir = ROOT / f"{baseline_dir}_discover"
    treatment_discover_dir = ROOT / f"{treatment_dir}_discover"

    if not args.treatment_only and not (baseline_dir / "student.pt").is_file():
        train_args = load_args(source_train, "train_student", baseline_dir, args.seed, args.epochs)
        train_args.student_ckpt = str(baseline_dir / "student.pt")
        train_args.alpha_discovery_rejection_boundary = 0.0
        fit_student(train_args)
    if not (treatment_dir / "student.pt").is_file():
        train_args = load_args(source_train, "train_student", treatment_dir, args.seed, args.epochs)
        train_args.student_ckpt = str(treatment_dir / "student.pt")
        train_args.alpha_discovery_rejection_boundary = float(args.weight)
        train_args.discovery_rejection_boundary_warmup_epochs = int(
            args.boundary_warmup_epochs
        )
        train_args.discovery_rejection_boundary_ramp_epochs = int(
            args.boundary_ramp_epochs
        )
        train_args.discovery_rejection_boundary_stop_epoch = int(
            args.boundary_stop_epoch
        )
        train_args.discovery_boundary_mode = "pu"
        train_args.discovery_boundary_pu_risk = args.risk_mode
        train_args.alpha_discovery_rejection_pu_feature_margin = 0.0
        fit_student(train_args)

    arms = [(treatment_dir, treatment_discover_dir)] if args.treatment_only else [
        (baseline_dir, baseline_discover_dir),
        (treatment_dir, treatment_discover_dir),
    ]
    for student_dir, discover_dir in arms:
        report = discover_dir / "discovery_report.json"
        if report.is_file():
            continue
        discover_args = load_args(source_discover, "discover", discover_dir, args.seed, args.epochs)
        discover_args.student_ckpt = str(student_dir / "student.pt")
        discover(discover_args)


if __name__ == "__main__":
    main()
