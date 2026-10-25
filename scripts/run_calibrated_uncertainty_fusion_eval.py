"""Evaluate validation-calibrated uncertainty fusion on frozen students.

This is an evaluation-only mechanism check. It never retrains a student. The
uncertainty-to-error map is fitted from known validation labels inside
``train.discover`` and is then applied to validation, discovery-pool, and test
outputs without using pool/test labels for selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make direct ``python scripts/<name>.py`` execution behave like the other
# repository runners without requiring PYTHONPATH to be set by the caller.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import discover, parse_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation directory: {args.output_dir}")
    config = json.loads(args.source_config.read_text(encoding="utf-8"))
    run_args = parse_args(["discover"])
    for key, value in config.items():
        if hasattr(run_args, key):
            setattr(run_args, key, value)
    run_args.command = "discover"
    run_args.student_ckpt = str(args.student_ckpt.resolve())
    run_args.work_dir = str(args.output_dir.resolve())
    run_args.score_mode = "feature_rejector_uncertainty_fusion"
    run_args.threshold_policy = "known_coverage"
    run_args.target_known_coverage = 0.95
    run_args.cluster_k = "auto"
    run_args.skip_clustering = False
    run_args.open_val_ratio = 0.0
    discover(run_args)


if __name__ == "__main__":
    main()
