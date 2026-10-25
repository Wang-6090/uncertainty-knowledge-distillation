"""Re-evaluate frozen CE-decay models and export known-validation scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2063])
    args = parser.parse_args()

    for seed in args.seeds:
        prefix = f"rejection_ce_decay_s{seed}_replication"
        for arm in ("static_control", "late_decay"):
            train_dir = ROOT / "runs" / f"{prefix}_{arm}"
            discover_config = train_dir.with_name(train_dir.name + "_discover") / "discover_config.json"
            checkpoint = train_dir / "student.pt"
            if not discover_config.is_file() or not checkpoint.is_file():
                raise FileNotFoundError(f"missing frozen model/config for seed={seed}, arm={arm}")
            output_dir = ROOT / "runs" / f"{prefix}_{arm}_validation_score_export"
            if output_dir.exists():
                raise FileExistsError(f"refusing to overwrite {output_dir}")

            values = json.loads(discover_config.read_text(encoding="utf-8"))
            run_args = parse_args(["discover"])
            for key, value in values.items():
                if hasattr(run_args, key):
                    setattr(run_args, key, value)
            run_args.work_dir = str(output_dir)
            run_args.student_ckpt = str(checkpoint)
            run_args.score_mode = "feature_rejector"
            run_args.threshold_policy = "known_coverage"
            run_args.target_known_coverage = 0.95
            run_args.skip_clustering = True
            run_args.save_calibration_scores = True
            print(f"[frozen discover] seed={seed} arm={arm}", flush=True)
            discover(run_args)
            export_path = output_dir / "calibration_scores.npz"
            if not export_path.is_file():
                raise RuntimeError(f"calibration score export was not created: {export_path}")
            print(f"[saved known-validation scores] {export_path}", flush=True)


if __name__ == "__main__":
    main()
