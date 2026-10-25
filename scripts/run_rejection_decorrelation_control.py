"""Paired semantic-hard training test for rejection-feature decorrelation.

The only training variable is ``alpha_rejection_decorrelation``.  Existing
VOS-control configs provide the matched teacher, split, optimizer, detector,
and validation protocol.  The control checkpoint is reused; the treatment is
trained from the same deterministic seed and source configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def run_case(seed: int, alpha: float, source_train: Path, source_discover: Path, tag: str):
    source_train_values = json.loads(source_train.read_text(encoding="utf-8"))
    source_discover_values = json.loads(source_discover.read_text(encoding="utf-8"))
    control_dir = ROOT / "runs" / f"{tag}_control"
    treatment_dir = ROOT / "runs" / f"{tag}_decorrelation"
    control_discover_dir = ROOT / "runs" / f"{tag}_control_discover"
    treatment_discover_dir = ROOT / "runs" / f"{tag}_decorrelation_discover"

    # Reuse the already completed matched control only if its source protocol
    # agrees with the requested config; it is never modified in place.
    control_ckpt = control_dir / "student.pt"
    if not control_ckpt.is_file():
        args = load_args(source_train, "train_student", control_dir, seed)
        args.alpha_rejection_decorrelation = 0.0
        args.student_ckpt = str(control_ckpt)
        fit_student(args)
    if not (control_discover_dir / "discovery_report.json").is_file():
        args = load_args(source_discover, "discover", control_discover_dir, seed)
        args.student_ckpt = str(control_ckpt)
        discover(args)

    treatment_ckpt = treatment_dir / "student.pt"
    if not treatment_ckpt.is_file():
        args = load_args(source_train, "train_student", treatment_dir, seed)
        args.alpha_rejection_decorrelation = float(alpha)
        args.student_ckpt = str(treatment_ckpt)
        fit_student(args)
    if not (treatment_discover_dir / "discovery_report.json").is_file():
        args = load_args(source_discover, "discover", treatment_discover_dir, seed)
        args.student_ckpt = str(treatment_ckpt)
        discover(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-discover", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    run_case(args.seed, args.alpha, args.source_train, args.source_discover, args.tag)


if __name__ == "__main__":
    main()
