from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_METRICS = [
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "known_class_accuracy_all_known",
    "unknown_reject_rate",
    "cluster_candidate_purity",
    "cluster_acc",
    "cluster_nmi",
    "cluster_ari",
    "cluster_all_unknown_oracle_nmi",
]


def load_report(root: Path, run: str) -> dict:
    path = Path(run)
    path = path if path.is_absolute() else root / path
    with (path / "discovery_report.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "values": []}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": array.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated discovery runs as mean/std.")
    parser.add_argument("--root", default="./runs")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--out", default="./analysis/multiseed_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    reports = [load_report(root, run) for run in args.runs]
    summary = {
        "runs": args.runs,
        "num_runs": len(reports),
        "metrics": {
            metric: summarize([report.get(metric, float("nan")) for report in reports])
            for metric in args.metrics
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, allow_nan=True, indent=2)

    for metric, item in summary["metrics"].items():
        print(f"{metric}: {item['mean']:.4f} +/- {item['std']:.4f} (n={item['count']})")


if __name__ == "__main__":
    main()
