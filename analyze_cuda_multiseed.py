from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRIC_KEYS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "known_class_accuracy_all_known",
    "known_accept_rate",
    "unknown_reject_rate",
    "cluster_acc",
    "cluster_nmi",
    "cluster_ari",
    "cluster_candidate_purity_before_purification",
    "cluster_candidate_purity",
    "cluster_candidate_unknown_recall",
    "cluster_candidate_count",
    "cluster_candidate_removed_count",
)

RUN_NAME_RE = re.compile(
    r"^revised_cuda_ms_s(?P<seed>\d+)_(?P<method>.+?)(?:_v\d+)?_detect_(?P<cluster_k>oracle|auto)$"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "count": int(values.size),
    }


def collect_runs(root: Path, pattern: str):
    items = []
    for run_dir in sorted(root.glob(pattern)):
        report_path = run_dir / "discovery_report.json"
        if not run_dir.is_dir() or not report_path.exists():
            continue
        match = RUN_NAME_RE.match(run_dir.name)
        if match is None:
            print(f"skip unrecognized run name: {run_dir.name}")
            continue
        detail_path = run_dir / "discovery_detail.json"
        detail = load_json(detail_path) if detail_path.exists() else {}
        config_path = run_dir / "config.json"
        items.append(
            {
                "run_dir": str(run_dir),
                "seed": int(match.group("seed")),
                "method": match.group("method"),
                "cluster_k_mode": match.group("cluster_k"),
                "cluster_k": detail.get("cluster_k"),
                "config": load_json(config_path) if config_path.exists() else {},
                "metrics": load_json(report_path),
            }
        )
    return items


def aggregate(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[(item["method"], item["cluster_k_mode"])].append(item)
    summary = []
    for (method, k_mode), rows in sorted(grouped.items()):
        metrics = {
            key: mean_std([row["metrics"].get(key) for row in rows])
            for key in METRIC_KEYS
        }
        metrics["cluster_k"] = mean_std([row.get("cluster_k") for row in rows])
        summary.append(
            {
                "method": method,
                "cluster_k_mode": k_mode,
                "seeds": sorted(row["seed"] for row in rows),
                "runs": [row["run_dir"] for row in rows],
                "metrics": metrics,
            }
        )
    return summary


def fmt(item):
    if item["mean"] is None:
        return "-"
    return f"{item['mean']:.4f} +/- {item['std']:.4f}"


def make_markdown(summary):
    lines = [
        "# CUDA Multi-seed Ablation Summary",
        "",
        "Results are mean +/- sample standard deviation across completed seeds.",
        "",
        "| method | K protocol | seeds | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | cluster ACC | NMI | ARI | purity | unknown recall | candidate count | removed | estimated K |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        m = row["metrics"]
        lines.append(
            f"| {row['method']} | {row['cluster_k_mode']} | "
            f"{', '.join(map(str, row['seeds']))} | {fmt(m['auroc'])} | "
            f"{fmt(m['aupr'])} | {fmt(m['fpr95'])} | {fmt(m['oscr'])} | "
            f"{fmt(m['known_class_accuracy_all_known'])} | "
            f"{fmt(m['known_accept_rate'])} | {fmt(m['unknown_reject_rate'])} | "
            f"{fmt(m['cluster_acc'])} | {fmt(m['cluster_nmi'])} | "
            f"{fmt(m['cluster_ari'])} | "
            f"{fmt(m['cluster_candidate_purity_before_purification'])} -> "
            f"{fmt(m['cluster_candidate_purity'])} | "
            f"{fmt(m['cluster_candidate_unknown_recall'])} | "
            f"{fmt(m['cluster_candidate_count'])} | "
            f"{fmt(m['cluster_candidate_removed_count'])} | "
            f"{fmt(m['cluster_k'])} |"
        )
    lines.extend(
        [
            "",
            "Compare B_standard_kd vs C_uncertainty_kd to test uncertainty-aware KD under the fixed protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./runs")
    parser.add_argument("--glob", default="revised_cuda_ms_s*_detect_*")
    parser.add_argument("--out-dir", default="./analysis/revised_cuda_multiseed")
    args = parser.parse_args()

    items = collect_runs(Path(args.root), args.glob)
    if not items:
        raise SystemExit(f"No completed runs found under {args.root} matching {args.glob}")
    summary = aggregate(items)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multiseed_summary.json").write_text(
        json.dumps({"runs": items, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "multiseed_summary.md").write_text(
        make_markdown(summary), encoding="utf-8"
    )
    print(out_dir / "multiseed_summary.md")
    print(out_dir / "multiseed_summary.json")
    print(f"completed runs: {len(items)}; groups: {len(summary)}")


if __name__ == "__main__":
    main()
