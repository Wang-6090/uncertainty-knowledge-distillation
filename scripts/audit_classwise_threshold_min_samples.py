"""Known-validation-only sensitivity audit for classwise q95 bin support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_validation_threshold_stability import audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_root", type=Path, help="runs/ directory containing score exports")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2061, 2062, 2063])
    parser.add_argument("--minimums", type=int, nargs="+", default=[5, 10, 20, 30, 50])
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or any(value <= 0 for value in args.minimums):
        raise ValueError("repeats and minimum sample counts must be positive")

    rows = []
    for seed in args.seeds:
        for arm, run_name in (
            ("static", "static_control"),
            ("late", "late_decay"),
        ):
            archive = (
                args.score_root
                / f"rejection_ce_decay_s{seed}_replication_{run_name}_validation_score_export"
                / "calibration_scores.npz"
            )
            with np.load(archive, allow_pickle=False) as payload:
                scores = payload["score"]
                predicted_class = payload["predicted_class"]
            for minimum in args.minimums:
                result = audit(
                    scores,
                    target=0.95,
                    repeats=args.repeats,
                    seed=seed * 100 + minimum,
                    predicted_classes=predicted_class,
                    classwise_min_samples=minimum,
                )["class_conditional_q95"]
                oob = result["out_of_bag_known_acceptance"]
                class_counts = np.bincount(predicted_class)
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "sample_count": int(len(scores)),
                        "minimum_per_predicted_class": minimum,
                        "eligible_predicted_classes": int(np.sum(class_counts >= minimum)),
                        "calibration_samples_in_eligible_classes": int(
                            class_counts[class_counts >= minimum].sum()
                        ),
                        "full_sample_acceptance": result["full_sample_acceptance"],
                        "oob_known_acceptance": oob["mean"],
                        "oob_q025": oob["q025"],
                        "oob_q975": oob["q975"],
                    }
                )
    rendered = json.dumps({"repeats": args.repeats, "rows": rows}, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
