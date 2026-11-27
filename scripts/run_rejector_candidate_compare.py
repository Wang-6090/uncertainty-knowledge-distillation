"""Compare rejector candidate mining on one frozen student checkpoint.

Only ``rejector_candidate_mode`` changes between arms.  The student checkpoint,
mixed discovery pool, PU prior, rejector feature mode, calibration policy and
test split are copied from the source discover configuration.
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
    parser.add_argument(
        "--modes", nargs="+", default=["entropy_uncertainty", "prototype_boundary"]
    )
    args = parser.parse_args()
    source = json.loads(args.source_discover.read_text(encoding="utf-8"))
    summary = {"protocol": {}, "arms": {}}
    for mode in args.modes:
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
            run_args.rejector_pool_selection = "high_risk"
            run_args.rejector_candidate_mode = mode
            run_args.skip_clustering = True
            discover(run_args)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = json.loads((work_dir / "discover_config.json").read_text(encoding="utf-8"))
        summary["arms"][mode] = {
            "work_dir": str(work_dir),
            "config": {
                "student_ckpt": config.get("student_ckpt"),
                "rejector_pool_selection": config.get("rejector_pool_selection"),
                "rejector_candidate_mode": config.get("rejector_candidate_mode"),
                "rejector_candidate_ratio": config.get("rejector_candidate_ratio"),
                "rejector_training": config.get("rejector_training"),
                "rejector_known_prior": config.get("rejector_known_prior"),
                "rejector_feature_mode": config.get("rejector_feature_mode"),
                "threshold_policy": config.get("threshold_policy"),
                "target_known_coverage": config.get("target_known_coverage"),
            },
            "metrics": {
                key: report.get(key)
                for key in (
                    "auroc", "aupr", "fpr95", "known_accept_rate",
                    "unknown_reject_rate", "known_rejected", "unknown_false_accept",
                )
            },
        }
        summary["protocol"] = {
            "source_discover": str(args.source_discover),
            "student_ckpt": str(args.student_ckpt),
            "labels_used_for_selection": False,
            "single_changed_factor": "rejector_candidate_mode",
        }
    output = ROOT / "analysis" / f"{args.output_prefix}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
