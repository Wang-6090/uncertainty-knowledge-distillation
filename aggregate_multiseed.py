from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_METRICS = [
    "auroc", "aupr", "fpr95", "oscr", "known_class_accuracy_all_known",
    "unknown_reject_rate", "cluster_acc", "cluster_nmi", "cluster_ari",
    "cluster_candidate_purity", "cluster_all_unknown_oracle_nmi",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated discovery runs as mean +/- sample std.")
    parser.add_argument("--root", default="./runs")
    parser.add_argument("--runs", nargs="+", required=True, help="Discovery run directories or names relative to --root.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--out", default="./analysis/multiseed_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    reports = []
    for run in args.runs:
        path = Path(run)
        path = path if path.is_absolute() else root / path
        with (path / "discovery_report.json").open(encoding="utf-8") as handle:
            reports.append(json.load(handle))
    summary = {"runs": args.runs, "num_seeds": len(reports), "metrics": {}}
    for metric in args.metrics:
        values = np.asarray([report.get(metric, float("nan")) for report in reports], dtype=float)
        values = values[np.isfinite(values)]
        summary["metrics"][metric] = {
            "count": int(len(values)),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else float("nan"),
            "values": values.tolist(),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    for metric, item in summary["metrics"].items():
        print(f"{metric}: {item['mean']:.4f} +/- {item['std']:.4f} (n={item['count']})")


if __name__ == "__main__":
    main()
