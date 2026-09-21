from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "auroc", "aupr", "fpr95", "oscr", "known_class_accuracy_all_known",
    "known_accept_rate", "unknown_reject_rate", "cluster_candidate_purity",
    "cluster_candidate_unknown_recall", "cluster_nmi", "cluster_ari",
    "cluster_unknown_only_nmi", "cluster_unknown_only_ari", "silhouette",
)


def mean_std(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {"mean": float(x.mean()) if x.size else None,
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0 if x.size else None,
            "count": int(x.size)}


def load_report(path: Path):
    report = json.loads((path / "discovery_report.json").read_text(encoding="utf-8"))
    chosen = report.get("cluster_k")
    report["silhouette"] = next((d.get("silhouette") for d in report.get("cluster_diagnostics", [])
                                  if d.get("k") == chosen), None)
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare hard and soft discovery selection runs.")
    parser.add_argument("--root", default="./runs")
    parser.add_argument("--prefix", default="ds2_cifar")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out", default="./analysis/discovery_selection_cifar.json")
    args = parser.parse_args()
    root = Path(args.root)
    result = {"protocol": {"seeds": args.seeds, "methods": ["hard", "soft"]}, "methods": {}}
    for method in ("hard", "soft"):
        reports = [load_report(root / f"{args.prefix}_s{seed}_{method}_detect_auto") for seed in args.seeds]
        result["methods"][method] = {
            "runs": [f"{args.prefix}_s{seed}_{method}_detect_auto" for seed in args.seeds],
            "metrics": {key: mean_std([r.get(key) for r in reports]) for key in METRICS},
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for method, data in result["methods"].items():
        print(method)
        for key, value in data["metrics"].items():
            print(f"  {key}: {value['mean']} +/- {value['std']} (n={value['count']})")


if __name__ == "__main__":
    main()
