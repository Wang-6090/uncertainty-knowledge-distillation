"""Matched control for boundary-focused inter-class virtual outliers.

The student recipe, teacher, split, pool, epoch budget and detector are copied
from the audited equal-epoch configuration.  The only training factor is the
optional VOS term: disabled for the control and ``interclass_mix`` for the
treatment.
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
    mode: str,
    alpha: float,
    tag: str,
    pairing_strategy: str,
    boundary_fraction: float,
    source_gradient: bool,
    alpha_rejection: float,
):
    student_dir = ROOT / "runs" / f"{tag}_{mode}"
    discover_dir = ROOT / "runs" / f"{tag}_{mode}_discover"
    student_ckpt = student_dir / "student.pt"
    if not student_ckpt.is_file():
        train_args = load_args(source_train, "train_student", student_dir, seed, epochs)
        train_args.student_ckpt = str(student_ckpt)
        train_args.alpha_vos = float(alpha)
        train_args.alpha_vos_rejection = float(alpha_rejection)
        train_args.vos_mode = (
            mode
            if mode != "control"
            else ("interclass_mix" if float(alpha) > 0.0 else "gaussian")
        )
        train_args.vos_interclass_pairing = pairing_strategy
        train_args.vos_boundary_fraction = float(boundary_fraction)
        train_args.vos_source_gradient = bool(source_gradient)
        fit_student(train_args)
    if not (discover_dir / "discovery_report.json").is_file():
        discover_args = load_args(source_discover, "discover", discover_dir, seed, epochs)
        discover_args.student_ckpt = str(student_ckpt)
        discover(discover_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--source-train",
        type=Path,
        default=ROOT / "runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_w0p1/train_student_config.json",
    )
    parser.add_argument(
        "--source-discover",
        type=Path,
        default=ROOT / "runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_w0p1_discover/discover_config.json",
    )
    parser.add_argument("--tag", default="interclass_vos_control_s44_e4")
    parser.add_argument("--pairing-strategy", choices=["random", "nearest_class"], default="random")
    parser.add_argument("--boundary-fraction", type=float, default=1.0)
    parser.add_argument(
        "--alpha-vos-rejection",
        type=float,
        default=0.0,
        help="Direct VOS uncertainty-head weight for the treatment arm.",
    )
    parser.add_argument(
        "--control-alpha-vos",
        type=float,
        default=0.0,
        help="VOS weight for the control arm (allows matched VOS ablations).",
    )
    parser.add_argument(
        "--control-pairing-strategy",
        choices=["random", "nearest_class"],
        default="random",
    )
    parser.add_argument("--control-boundary-fraction", type=float, default=1.0)
    parser.add_argument("--control-alpha-vos-rejection", type=float, default=0.0)
    parser.add_argument(
        "--source-gradient",
        action="store_true",
        help="Legacy ablation: allow VOS to backpropagate through source features.",
    )
    parser.add_argument(
        "--only-treatment",
        action="store_true",
        help="Run only the interclass VOS treatment arm; reuse an existing control.",
    )
    args = parser.parse_args()
    if not args.source_train.is_file() or not args.source_discover.is_file():
        raise FileNotFoundError("audited source configs are required")
    if not args.only_treatment:
        run_arm(
            args.source_train, args.source_discover, args.seed, args.epochs,
            "control", args.control_alpha_vos, args.tag,
            args.control_pairing_strategy, args.control_boundary_fraction,
            args.source_gradient, args.control_alpha_vos_rejection,
        )
    run_arm(
        args.source_train, args.source_discover, args.seed, args.epochs,
        "interclass_mix", 0.01, args.tag, args.pairing_strategy,
        args.boundary_fraction, args.source_gradient, args.alpha_vos_rejection,
    )


if __name__ == "__main__":
    main()
