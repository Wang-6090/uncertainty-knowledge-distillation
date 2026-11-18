"""Run the boundary-only uncertainty correction through full discovery.

This is a frozen-checkpoint experiment.  The correction is calibrated from the
known validation split only, then passed through rejection, candidate selection,
PCA and KMeans.  Unknown labels are read only from the resulting reports.
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
    fallback = (
        ROOT
        / "splits"
        / "semantic_hard"
        / f"seed{seed}"
        / "cifar100_60_40_semantic_hard.json"
    )
    if not fallback.is_file():
        raise FileNotFoundError(f"missing split file: {candidate} and {fallback}")
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--band", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix appended to each split output directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        train_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control"
        source_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control_discover"
        source_config = source_dir / "discover_config.json"
        checkpoint = train_dir / "student.pt"
        if not source_config.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"missing frozen model/config for seed={seed}")

        output_dir = ROOT / "runs" / f"rejection_uncertainty_band_full_s{seed}{args.output_suffix}"
        report_path = output_dir / "discovery_report.json"
        detail_path = output_dir / "discovery_detail.json"
        calibration_path = output_dir / "calibration_report.json"
        if report_path.is_file() and detail_path.is_file() and calibration_path.is_file():
            print(f"[exists] seed={seed} output={output_dir}", flush=True)
            continue
        if output_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to reuse incomplete output: {output_dir}; use --overwrite only after inspection"
            )

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
        run_args.boundary_uncertainty_band_multiplier = float(args.band)
        run_args.boundary_uncertainty_alpha = float(args.alpha)
        run_args.skip_clustering = False
        run_args.save_calibration_scores = True

        print(
            f"[full discover] seed={seed} band={args.band} alpha={args.alpha}",
            flush=True,
        )
        discover(run_args)
        for required in (report_path, detail_path, calibration_path):
            if not required.is_file():
                raise RuntimeError(f"expected output was not created: {required}")
        print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
