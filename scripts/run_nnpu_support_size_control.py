"""Compare capped stratified support with the full known support set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


def load_args(path: Path, work_dir: Path, seed: int, max_samples: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args(["discover"])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    args.rejector_nnpu_risk = "nu_corrected"
    args.rejector_stratified_known = True
    args.rejector_max_samples = int(max_samples)
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
    parser.add_argument("--tag", default="nnpu_support_size_control_s44")
    args = parser.parse_args()
    if not args.source_config.is_file():
        raise FileNotFoundError(f"missing source discovery config: {args.source_config}")
    if not args.student_ckpt.is_file():
        raise FileNotFoundError(f"missing student checkpoint: {args.student_ckpt}")
    for max_samples, label in ((5000, "capped5000"), (0, "full")):
        work_dir = ROOT / "runs" / f"{args.tag}_{label}"
        report = work_dir / "discovery_report.json"
        if report.is_file():
            print(f"skip existing {report}")
            continue
        discover_args = load_args(args.source_config, work_dir, args.seed, max_samples)
        discover_args.student_ckpt = str(args.student_ckpt)
        discover(discover_args)


if __name__ == "__main__":
    main()
