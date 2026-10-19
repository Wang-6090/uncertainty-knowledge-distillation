"""Paired control for the feature-rejector nnPU risk normalization.

The student checkpoint and discovery protocol are held fixed.  The only
changed factor is ``rejector_nnpu_risk``: ``legacy`` retains the historical
positive-risk normalization by ``1 - known_prior`` while ``nu_corrected``
uses the same class terms without that extra scaling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


def load_args(path: Path, work_dir: Path, seed: int, risk_mode: str):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args(["discover"])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    args.rejector_nnpu_risk = risk_mode
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path(
            "runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_w0p1_discover/discover_config.json"
        ),
    )
    parser.add_argument(
        "--student-ckpt",
        type=Path,
        default=Path(
            "runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_w0p1/student.pt"
        ),
    )
    parser.add_argument(
        "--tag", default="nnpu_formula_control_s44"
    )
    args = parser.parse_args()
    source_config = args.source_config
    student_ckpt = args.student_ckpt
    if not source_config.is_file():
        raise FileNotFoundError(f"missing source discovery config: {source_config}")
    if not student_ckpt.is_file():
        raise FileNotFoundError(f"missing student checkpoint: {student_ckpt}")

    for risk_mode in ("legacy", "nu_corrected"):
        work_dir = ROOT / "runs" / f"{args.tag}_{risk_mode}"
        report = work_dir / "open_set_error_analysis.json"
        if report.is_file():
            print(f"skip existing {report}")
            continue
        discover_args = load_args(source_config, work_dir, args.seed, risk_mode)
        discover_args.student_ckpt = str(student_ckpt)
        discover_args.rejector_nnpu_risk = risk_mode
        discover(discover_args)


if __name__ == "__main__":
    main()
