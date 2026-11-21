"""Run a fixed boundary-uncertainty audit on frozen seed-2160 CE-decay arms.

No training is performed. The script reloads each arm's saved discovery config
and student checkpoint, changes only the fixed boundary correction, and runs the
full discovery evaluation for both arms.
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


SEED = 2160
TAG = "rejection_ce_decay_s2160_20261121"
ARMS = ("static_control", "late_decay")
BAND = 1.0
ALPHA = 0.5


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for arm in ARMS:
        train_dir = ROOT / "runs" / f"{TAG}_{arm}"
        source_dir = ROOT / "runs" / f"{TAG}_{arm}_discover"
        source_config = source_dir / "discover_config.json"
        checkpoint = train_dir / "student.pt"
        if not source_config.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"missing frozen input for arm={arm}")

        output_dir = ROOT / "runs" / f"{TAG}_{arm}_boundary_band1_alpha05"
        required = (
            output_dir / "discovery_report.json",
            output_dir / "discovery_detail.json",
            output_dir / "calibration_report.json",
        )
        if all(path.is_file() for path in required):
            print(f"[exists] arm={arm} output={output_dir}", flush=True)
            continue
        if output_dir.exists() and not args.overwrite:
            raise FileExistsError(f"refusing incomplete output: {output_dir}")

        values = load_json(source_config)
        run_args = parse_args(["discover"])
        for key, value in values.items():
            if hasattr(run_args, key):
                setattr(run_args, key, value)
        run_args.device = args.device
        run_args.seed = SEED
        run_args.work_dir = str(output_dir)
        run_args.student_ckpt = str(checkpoint)
        run_args.boundary_uncertainty_band_multiplier = BAND
        run_args.boundary_uncertainty_alpha = ALPHA
        run_args.skip_clustering = False
        run_args.save_calibration_scores = True

        print(f"[boundary audit] arm={arm} band={BAND} alpha={ALPHA}", flush=True)
        discover(run_args)
        for path in required:
            if not path.is_file():
                raise RuntimeError(f"expected output was not created: {path}")
        print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
