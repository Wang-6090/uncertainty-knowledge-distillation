"""Recheck balanced versus ordinary full-support nnPU rejector features.

This is a post-hoc paired control: the student checkpoint, discovery pool,
nnPU risk, support size, calibration and all other discovery settings are
copied from one audited configuration.  The only changed factor is
``rejector_feature_mode``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


def load_args(path: Path, work_dir: Path, seed: int, feature_mode: str):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args(["discover"])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    args.rejector_training = "nnpu"
    args.rejector_nnpu_risk = "nu_corrected"
    args.rejector_nnpu_model = "linear"
    args.rejector_max_samples = 0
    args.rejector_stratified_known = True
    args.rejector_feature_mode = feature_mode
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path(
            "runs/rejection_boundary_epoch_control_s44_e4_baseline_discover/discover_config.json"
        ),
    )
    parser.add_argument(
        "--student-ckpt",
        type=Path,
        default=Path(
            "runs/rejection_boundary_epoch_control_s44_e4_baseline/student.pt"
        ),
    )
    parser.add_argument("--tag", default="nnpu_balanced_support_recheck_s44")
    args = parser.parse_args()
    if not args.source_config.is_file():
        raise FileNotFoundError(f"missing source discovery config: {args.source_config}")
    if not args.student_ckpt.is_file():
        raise FileNotFoundError(f"missing student checkpoint: {args.student_ckpt}")

    for feature_mode, label in (
        ("support_augmented", "ordinary"),
        ("rejection_balanced_support_augmented", "balanced"),
    ):
        work_dir = ROOT / "runs" / f"{args.tag}_{label}"
        report = work_dir / "discovery_report.json"
        if report.is_file():
            print(f"skip existing {report}")
            continue
        discover_args = load_args(args.source_config, work_dir, args.seed, feature_mode)
        discover_args.student_ckpt = str(args.student_ckpt)
        discover(discover_args)


if __name__ == "__main__":
    main()
