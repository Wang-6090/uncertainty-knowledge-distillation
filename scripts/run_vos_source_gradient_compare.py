"""Paired full-recipe comparison of detached vs source-gradient VOS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


def load_args(path: Path, command: str, work_dir: Path, seed: int, epochs: int):
    config = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.epochs = int(epochs)
    args.work_dir = str(work_dir)
    return args


def run_arm(
    source_train: Path,
    source_discover: Path,
    seed: int,
    epochs: int,
    tag: str,
    source_gradient: bool,
) -> None:
    suffix = "source_gradient" if source_gradient else "detached"
    student_dir = ROOT / "runs" / f"{tag}_{suffix}"
    discover_dir = ROOT / "runs" / f"{tag}_{suffix}_discover"
    student_ckpt = student_dir / "student.pt"
    train_args = load_args(source_train, "train_student", student_dir, seed, epochs)
    train_args.student_ckpt = str(student_ckpt)
    train_args.limit_train = 0
    train_args.audit_loss_gradients = True
    train_args.alpha_vos = 0.01
    train_args.alpha_vos_rejection = 0.0
    train_args.vos_mode = "interclass_mix"
    train_args.vos_interclass_pairing = "nearest_class"
    train_args.vos_boundary_fraction = 0.25
    train_args.vos_source_gradient = bool(source_gradient)
    fit_student(train_args)

    discover_args = load_args(
        source_discover, "discover", discover_dir, seed, epochs
    )
    discover_args.student_ckpt = str(student_ckpt)
    discover(discover_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-train",
        type=Path,
        default=ROOT
        / "runs/vos_boundary_replicate_s123_e4_interclass_mix/train_student_config.json",
    )
    parser.add_argument(
        "--source-discover",
        type=Path,
        default=ROOT
        / "runs/vos_boundary_replicate_s123_e4_interclass_mix_discover/discover_config.json",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--tag", default="vos_source_gradient_compare_s123_e4_20261018"
    )
    args = parser.parse_args()
    for config in (args.source_train, args.source_discover):
        if not config.is_file():
            raise FileNotFoundError(config)
    for source_gradient in (False, True):
        run_arm(
            args.source_train,
            args.source_discover,
            args.seed,
            args.epochs,
            args.tag,
            source_gradient,
        )


if __name__ == "__main__":
    main()
