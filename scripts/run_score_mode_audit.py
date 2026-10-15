"""Run a post-hoc score-mode audit on one fixed student checkpoint.

Every arm reuses the same discovery configuration, checkpoint, seed, pool and
known-coverage calibration.  This is a diagnostic comparison of score
construction only; test labels are not used by the runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-discover", type=Path, required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument(
        "--score-modes",
        nargs="+",
        default=[
            "feature_rejector",
            "known_support",
            "known_conformal",
            "normalized_entropy_proto",
            "feature_rejector_fusion",
        ],
    )
    parser.add_argument(
        "--fusion-base-score",
        default="normalized_entropy_proto",
    )
    parser.add_argument("--fusion-rejector-weight", type=float, default=0.5)
    args = parser.parse_args()

    source = json.loads(args.source_discover.read_text(encoding="utf-8"))
    for mode in args.score_modes:
        work_dir = ROOT / "runs" / f"{args.output_prefix}_{mode}"
        report = work_dir / "discovery_report.json"
        if report.is_file():
            print(f"reuse {report}")
            continue
        run_args = parse_args(["discover"])
        for key, value in source.items():
            if hasattr(run_args, key):
                setattr(run_args, key, value)
        run_args.student_ckpt = str(args.student_ckpt)
        run_args.work_dir = str(work_dir)
        run_args.score_mode = mode
        run_args.auto_calibrate_score = False
        run_args.skip_clustering = True
        run_args.fusion_base_score = args.fusion_base_score
        run_args.fusion_rejector_weight = args.fusion_rejector_weight
        # The parser validates rejector-specific options even for support or
        # prototype-only scores.  Make that irrelevant inherited option
        # explicit so the score-mode comparison remains single-factor.
        if mode not in {"feature_rejector", "feature_rejector_fusion", "virtual_rejector"}:
            run_args.rejector_feature_mode = "embedding"
        discover(run_args)


if __name__ == "__main__":
    main()
