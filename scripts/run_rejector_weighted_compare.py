"""Compare all-pool, soft boundary weighting, and hard candidate mining.

The frozen student, mixed discovery pool, nnPU prior, calibration protocol,
and test split are copied from one discover configuration.  Only the
rejector-pool treatment changes between arms.
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
    parser.add_argument("--candidate-weight", type=float, default=2.0)
    args = parser.parse_args()

    source = json.loads(args.source_discover.read_text(encoding="utf-8"))
    arms = {
        "all": {"rejector_pool_selection": "all"},
        "weighted_boundary": {
            "rejector_pool_selection": "weighted_boundary",
            "rejector_candidate_mode": "prototype_boundary",
            "rejector_candidate_weight": float(args.candidate_weight),
        },
        "high_risk": {
            "rejector_pool_selection": "high_risk",
            "rejector_candidate_mode": "prototype_boundary",
        },
    }
    summary = {
        "protocol": {
            "source_discover": str(args.source_discover),
            "student_ckpt": str(args.student_ckpt),
            "single_changed_factor": "rejector_pool_selection",
            "candidate_weight": float(args.candidate_weight),
            "labels_used_for_selection": False,
        },
        "arms": {},
    }
    for arm, overrides in arms.items():
        work_dir = ROOT / "runs" / f"{args.output_prefix}_{arm}"
        report_path = work_dir / "discovery_report.json"
        if not report_path.is_file():
            run_args = parse_args(["discover"])
            for key, value in source.items():
                if hasattr(run_args, key):
                    setattr(run_args, key, value)
            run_args.student_ckpt = str(args.student_ckpt)
            run_args.work_dir = str(work_dir)
            run_args.score_mode = "feature_rejector"
            run_args.rejector_training = "nnpu"
            run_args.rejector_pool_selection = overrides["rejector_pool_selection"]
            run_args.rejector_candidate_mode = overrides.get(
                "rejector_candidate_mode", "prototype_boundary"
            )
            run_args.rejector_candidate_weight = overrides.get(
                "rejector_candidate_weight", float(args.candidate_weight)
            )
            run_args.skip_clustering = True
            discover(run_args)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = json.loads((work_dir / "discover_config.json").read_text(encoding="utf-8"))
        summary["arms"][arm] = {
            "work_dir": str(work_dir),
            "config": {
                key: config.get(key)
                for key in (
                    "student_ckpt",
                    "discovery_pool_mode",
                    "matched_discovery_pool",
                    "matched_pool_size",
                    "score_mode",
                    "rejector_pool_selection",
                    "rejector_candidate_mode",
                    "rejector_candidate_ratio",
                    "rejector_candidate_weight",
                    "rejector_training",
                    "rejector_known_prior",
                    "rejector_nnpu_risk",
                    "rejector_feature_mode",
                    "threshold_policy",
                    "target_known_coverage",
                )
            },
            "metrics": {
                key: report.get(key)
                for key in (
                    "auroc",
                    "aupr",
                    "fpr95",
                    "known_accept_rate",
                    "unknown_reject_rate",
                    "known_rejected",
                    "unknown_false_accept",
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
