"""Export known-validation signals for the four formal semantic-hard splits.

This re-evaluates frozen checkpoints only.  It does not train a model and does
not use unknown/test labels for calibration.  The exported arrays are used by
the boundary-only uncertainty audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


SEEDS = (2121, 2124, 2125, 2127)


def resolve_split(values: dict, seed: int) -> Path:
    candidate = Path(str(values.get("split_path", "")))
    if candidate.is_file():
        return candidate
    fallback = ROOT / "splits" / "semantic_hard" / f"seed{seed}" / "cifar100_60_40_semantic_hard.json"
    if not fallback.is_file():
        raise FileNotFoundError(f"missing split file: {candidate} and {fallback}")
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    for seed in args.seeds:
        train_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control"
        source_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control_discover"
        source_config = source_dir / "discover_config.json"
        checkpoint = train_dir / "student.pt"
        if not source_config.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"missing frozen model/config for seed={seed}")

        output_dir = ROOT / "runs" / f"rejection_boundary_calibration_s{seed}"
        score_path = output_dir / "calibration_scores.npz"
        if score_path.is_file():
            print(f"[exists] {score_path}", flush=True)
            continue
        if output_dir.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output_dir}")

        values = json.loads(source_config.read_text(encoding="utf-8"))
        run_args = parse_args(["discover"])
        for key, value in values.items():
            if hasattr(run_args, key):
                setattr(run_args, key, value)
        run_args.device = args.device
        run_args.seed = seed
        run_args.split_path = str(resolve_split(values, seed))
        run_args.work_dir = str(output_dir)
        run_args.student_ckpt = str(checkpoint)
        run_args.score_mode = "feature_rejector"
        run_args.threshold_policy = "known_coverage"
        run_args.target_known_coverage = 0.95
        run_args.skip_clustering = True
        run_args.save_calibration_scores = True
        print(f"[frozen discover] seed={seed}", flush=True)
        discover(run_args)
        if not score_path.is_file():
            raise RuntimeError(f"calibration score export was not created: {score_path}")
        print(f"[saved] {score_path}", flush=True)


if __name__ == "__main__":
    main()
