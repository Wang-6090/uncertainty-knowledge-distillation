"""Compare global support and predicted-class support rejector features.

All discovery, nnPU, calibration, and checkpoint settings are copied from one
source config. Only ``rejector_feature_mode`` changes.
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
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    source = json.loads(args.source_discover.read_text(encoding="utf-8"))
    modes = ("support_augmented", "predicted_support_augmented")
    summary = {
        "protocol": {
            "source_discover": str(args.source_discover),
            "student_ckpt": str(args.student_ckpt),
            "single_changed_factor": "rejector_feature_mode",
            "labels_used_for_feature_construction": False,
        },
        "arms": {},
    }
    for mode in modes:
        work_dir = ROOT / "runs" / f"{args.output_prefix}_{mode}"
        report_path = work_dir / "discovery_report.json"
        if not report_path.is_file():
            run_args = parse_args(["discover"])
            for key, value in source.items():
                if hasattr(run_args, key):
                    setattr(run_args, key, value)
            run_args.student_ckpt = str(args.student_ckpt)
            run_args.work_dir = str(work_dir)
            run_args.score_mode = "feature_rejector"
            run_args.rejector_pool_selection = "all"
            run_args.rejector_feature_mode = mode
            run_args.skip_clustering = True
            discover(run_args)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = json.loads((work_dir / "discover_config.json").read_text(encoding="utf-8"))
        summary["arms"][mode] = {
            "work_dir": str(work_dir),
            "config": {
                key: config.get(key)
                for key in (
                    "student_ckpt", "discovery_pool_mode", "matched_pool_size",
                    "rejector_pool_selection", "rejector_training", "rejector_known_prior",
                    "rejector_nnpu_risk", "rejector_feature_mode", "rejector_mc_samples",
                    "threshold_policy", "target_known_coverage",
                )
            },
            "metrics": {
                key: report.get(key)
                for key in (
                    "auroc", "aupr", "fpr95", "oscr", "known_accept_rate",
                    "unknown_reject_rate", "known_rejected", "unknown_false_accept",
                )
            },
        }
    output = ROOT / "analysis" / f"{args.output_prefix}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
