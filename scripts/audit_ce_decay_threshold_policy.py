"""Audit known-validation threshold transfer on frozen CE-decay pairs.

Models and scores stay fixed. The only comparison is a global 95%-coverage
known-validation threshold versus the existing class-conditional 95th-percentile
known-only policy. Test labels are consumed only by the final discover report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


def load_args(config_path: Path, work_dir: Path):
    values = json.loads(config_path.read_text(encoding="utf-8"))
    args = parse_args(["discover"])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.work_dir = str(work_dir)
    args.skip_clustering = True
    args.score_mode = "feature_rejector"
    args.target_known_coverage = 0.95
    return args


def main() -> None:
    seeds = (2054, 2055)
    arms = ("static_control", "late_decay")
    policies = ("known_coverage", "class_conditional")
    reports = {}
    for seed in seeds:
        for arm in arms:
            run_prefix = (
                "rejection_ce_decay_s2055_replication"
                if seed == 2055
                else f"rejection_ce_decay_s{seed}"
            )
            train_dir = ROOT / "runs" / f"{run_prefix}_{arm}"
            discover_config = train_dir.with_name(train_dir.name + "_discover") / "discover_config.json"
            student_ckpt = train_dir / "student.pt"
            if not discover_config.is_file() or not student_ckpt.is_file():
                raise FileNotFoundError(f"missing frozen model/config for seed={seed}, arm={arm}")
            for policy in policies:
                tag = f"rejection_ce_threshold_s{seed}_{arm}_{policy}"
                output_dir = ROOT / "runs" / tag
                if output_dir.exists():
                    raise FileExistsError(f"refusing to overwrite {output_dir}")
                args = load_args(discover_config, output_dir)
                args.student_ckpt = str(student_ckpt)
                args.threshold_policy = policy
                args.threshold_percentile = 95.0
                args.threshold_min_class_samples = 5
                print(f"[discover] seed={seed} arm={arm} policy={policy}", flush=True)
                discover(args)
                result_path = output_dir / "discovery_report.json"
                reports[f"s{seed}_{arm}_{policy}"] = {
                    "seed": seed,
                    "arm": arm,
                    "policy": policy,
                    "discover_dir": str(output_dir),
                    "metrics": json.loads(result_path.read_text(encoding="utf-8")),
                    "calibration": json.loads(
                        (output_dir / "calibration_report.json").read_text(encoding="utf-8")
                    ).get("threshold_policy"),
                }
    result = {
        "question": "Does known-only class-conditional calibration reduce threshold transfer variability on frozen CE-decay models?",
        "seeds": list(seeds),
        "models_or_scores_changed": False,
        "calibration_uses_unknown_or_test_labels": False,
        "class_conditional": "known validation 95th percentile by predicted class; minimum 5, global fallback",
        "global": "known validation threshold targeting 95% coverage",
        "test_labels_used_for_final_reporting_only": True,
        "arms": reports,
    }
    path = ROOT / "analysis" / "rejection_ce_threshold_policy_s2054_s2055.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
