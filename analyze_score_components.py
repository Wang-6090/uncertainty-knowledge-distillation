from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def summarize(scores, mask):
    scores = np.asarray(scores, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    known = scores[mask]
    unknown = scores[~mask]
    return {
        "known_mean": float(np.mean(known)) if known.size else float("nan"),
        "unknown_mean": float(np.mean(unknown)) if unknown.size else float("nan"),
        "known_median": float(np.median(known)) if known.size else float("nan"),
        "unknown_median": float(np.median(unknown)) if unknown.size else float("nan"),
        "known_p75": float(np.percentile(known, 75)) if known.size else float("nan"),
        "unknown_p75": float(np.percentile(unknown, 75)) if unknown.size else float("nan"),
    }


def analyze_run(run_dir: Path):
    detail = load_json(run_dir / "discovery_detail.json")
    true_known = np.asarray(detail["true_known"], dtype=int).astype(bool)
    components = {
        "combined_score": detail["score"],
        "entropy": detail.get("entropy"),
        "epistemic": detail.get("epistemic"),
        "aleatoric": detail.get("aleatoric"),
        "proto_dist": detail.get("proto_dist"),
    }
    out = {"run": run_dir.name, "n": int(len(true_known)), "components": {}}
    for name, values in components.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        out["components"][name] = {
            "auc": safe_auc(~true_known, values),
            "known": summarize(values, true_known),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["cifar_exp2", "cifar_ablation_kd0", "cifar_ablation_unc0", "cifar_ablation_base"],
    )
    parser.add_argument("--root", default="./runs")
    parser.add_argument("--out", default="./analysis/score_components.json")
    args = parser.parse_args()

    root = Path(args.root)
    results = [analyze_run(root / run) for run in args.runs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in results:
        print(item["run"])
        for name, metrics in item["components"].items():
            known = metrics["known"]
            print(
                f"  {name}: auc={metrics['auc']:.4f}, known_mean={known['known_mean']:.4f}, unknown_mean={known['unknown_mean']:.4f}"
            )

    print(out_path)


if __name__ == "__main__":
    main()
