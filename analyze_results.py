from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torchvision import datasets


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split(path: Path):
    data = torch.load(path, map_location="cpu")
    known = [int(x) for x in data["known_classes"]]
    novel = [int(x) for x in data["novel_classes"]]
    return known, novel


def load_class_names(config: dict):
    if str(config.get("dataset", "")).lower() != "cifar100":
        return None
    try:
        root = config.get("data_root", "./data")
        return list(datasets.CIFAR100(root=root, train=True, download=False).classes)
    except Exception:
        return None


def safe_mean(values):
    values = list(values)
    if not values:
        return float("nan")
    return float(mean(values))


def safe_std(values):
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(values.std(ddof=0))


def percentile(values, p):
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, p))


def analyze_run(run_dir: Path):
    report = load_json(run_dir / "discovery_report.json")
    detail = load_json(run_dir / "discovery_detail.json")
    config_path = run_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    class_names = load_class_names(config)
    split_path = run_dir / "split.pt"
    known_classes, novel_classes = ([], [])
    if split_path.exists():
        known_classes, novel_classes = load_split(split_path)

    score = np.asarray(detail["score"], dtype=float)
    pred_known = np.asarray(detail["pred_known"], dtype=int).astype(bool)
    true_known = np.asarray(detail["true_known"], dtype=int).astype(bool)
    pred_class = np.asarray(detail["pred_class"], dtype=int)
    raw_labels = np.asarray(detail["raw_labels"], dtype=int)
    pred_cluster = detail.get("pred_cluster")
    pred_cluster = None if pred_cluster is None else np.asarray(pred_cluster, dtype=int)

    known_to_idx = {cls: idx for idx, cls in enumerate(known_classes)}

    known_score = score[true_known]
    unknown_score = score[~true_known]
    known_accept = pred_known[true_known]
    unknown_reject = ~pred_known[~true_known]

    known_total = int(true_known.sum())
    unknown_total = int((~true_known).sum())
    known_rejected = int((true_known & ~pred_known).sum())
    unknown_false_accept = int((~true_known & pred_known).sum())

    known_raw = raw_labels[true_known]
    unknown_raw = raw_labels[~true_known]

    known_class_correct = 0
    known_class_total = 0
    known_by_class = defaultdict(lambda: {"total": 0, "correct": 0, "rejected": 0})
    for rl, pk, pc in zip(raw_labels[true_known], pred_known[true_known], pred_class[true_known]):
        known_class_total += 1
        item = known_by_class[int(rl)]
        item["total"] += 1
        if pk:
            pred_idx = int(pc)
            true_idx = known_to_idx.get(int(rl), None)
            if true_idx is not None and pred_idx == true_idx:
                known_class_correct += 1
                item["correct"] += 1
        else:
            item["rejected"] += 1

    novel_by_class = defaultdict(lambda: {"total": 0, "false_accept": 0, "correct_reject": 0})
    for rl, pk in zip(raw_labels[~true_known], pred_known[~true_known]):
        item = novel_by_class[int(rl)]
        item["total"] += 1
        if pk:
            item["false_accept"] += 1
        else:
            item["correct_reject"] += 1

    top_known_reject = sorted(
        (
            {
                "raw_label": cls,
                "name": class_names[cls] if class_names and 0 <= cls < len(class_names) else None,
                "count": stats["total"],
                "accuracy": stats["correct"] / max(stats["total"], 1),
                "reject_rate": stats["rejected"] / max(stats["total"], 1),
            }
            for cls, stats in known_by_class.items()
        ),
        key=lambda x: (x["reject_rate"], -x["count"]),
        reverse=True,
    )[:8]

    top_novel_false_accept = sorted(
        (
            {
                "raw_label": cls,
                "name": class_names[cls] if class_names and 0 <= cls < len(class_names) else None,
                "count": stats["total"],
                "false_accept_rate": stats["false_accept"] / max(stats["total"], 1),
                "correct_reject_rate": stats["correct_reject"] / max(stats["total"], 1),
            }
            for cls, stats in novel_by_class.items()
        ),
        key=lambda x: (x["false_accept_rate"], -x["count"]),
        reverse=True,
    )[:8]

    score_summary = {
        "known": {
            "count": known_total,
            "mean": safe_mean(known_score),
            "std": safe_std(known_score),
            "p25": percentile(known_score, 25),
            "median": percentile(known_score, 50),
            "p75": percentile(known_score, 75),
        },
        "unknown": {
            "count": unknown_total,
            "mean": safe_mean(unknown_score),
            "std": safe_std(unknown_score),
            "p25": percentile(unknown_score, 25),
            "median": percentile(unknown_score, 50),
            "p75": percentile(unknown_score, 75),
        },
    }

    result = {
        "run_dir": str(run_dir),
        "score_mode": detail.get("score_mode"),
        "config": config,
        "metrics": report,
        "score_summary": score_summary,
        "class_summary": {
            "known_total": known_total,
            "unknown_total": unknown_total,
            "known_rejected": known_rejected,
            "unknown_false_accept": unknown_false_accept,
            "known_class_correct": known_class_correct,
            "known_class_accuracy_after_accept": report.get("known_class_accuracy_after_accept"),
            "known_class_accuracy_all_known": report.get("known_class_accuracy_all_known"),
        },
        "top_known_reject_classes": top_known_reject,
        "top_novel_false_accept_classes": top_novel_false_accept,
        "known_classes": known_classes,
        "novel_classes": novel_classes,
    }

    if pred_cluster is not None:
        result["pred_cluster_summary"] = {
            "non_negative": int((pred_cluster >= 0).sum()),
            "unique_clusters": int(len(set(pred_cluster[pred_cluster >= 0].tolist()))),
        }

    return result


def format_run_md(item):
    m = item["metrics"]
    s = item["score_summary"]
    c = item["class_summary"]
    lines = []
    title = Path(item["run_dir"]).name
    if item.get("score_mode"):
        title += f" [{item['score_mode']}]"
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"- AUROC: {m['auroc']:.4f}")
    lines.append(f"- FPR95: {m['fpr95']:.4f}")
    lines.append(f"- known accept rate: {m['known_accept_rate']:.4f}")
    lines.append(f"- unknown reject rate: {m['unknown_reject_rate']:.4f}")
    lines.append(f"- known class acc after accept: {m['known_class_accuracy_after_accept']:.4f}")
    lines.append(f"- known class acc all known: {m['known_class_accuracy_all_known']:.4f}")
    lines.append(f"- cluster NMI: {m['cluster_nmi']:.4f}")
    lines.append(f"- cluster ARI: {m['cluster_ari']:.4f}")
    lines.append("")
    lines.append("- Score distribution:")
    lines.append(
        f"  - known mean {s['known']['mean']:.4f}, median {s['known']['median']:.4f}, p75 {s['known']['p75']:.4f}"
    )
    lines.append(
        f"  - unknown mean {s['unknown']['mean']:.4f}, median {s['unknown']['median']:.4f}, p75 {s['unknown']['p75']:.4f}"
    )
    lines.append("")
    lines.append("- Biggest known-class rejection rates:")
    for row in item["top_known_reject_classes"][:5]:
        name = f" ({row['name']})" if row.get("name") else ""
        lines.append(
            f"  - class {row['raw_label']}{name}: reject_rate={row['reject_rate']:.3f}, accuracy={row['accuracy']:.3f}, count={row['count']}"
        )
    lines.append("")
    lines.append("- Biggest novel-class false-accept rates:")
    for row in item["top_novel_false_accept_classes"][:5]:
        name = f" ({row['name']})" if row.get("name") else ""
        lines.append(
            f"  - class {row['raw_label']}{name}: false_accept_rate={row['false_accept_rate']:.3f}, correct_reject_rate={row['correct_reject_rate']:.3f}, count={row['count']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["cifar_exp2", "cifar_ablation_kd0", "cifar_ablation_unc0", "cifar_ablation_base"],
    )
    parser.add_argument("--root", default="./runs")
    parser.add_argument("--out-dir", default="./analysis")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for run in args.runs:
        run_dir = root / run
        items.append(analyze_run(run_dir))

    md_lines = ["# Open-set Error Analysis", ""]
    md_lines.append("## Cross-run comparison")
    md_lines.append("")
    md_lines.append("| run | AUROC | FPR95 | known acc all known | unknown reject rate | known accept rate |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for item in items:
        m = item["metrics"]
        run_name = Path(item["run_dir"]).name
        if item.get("score_mode"):
            run_name += f" [{item['score_mode']}]"
        md_lines.append(
            f"| {run_name} | {m['auroc']:.4f} | {m['fpr95']:.4f} | {m['known_class_accuracy_all_known']:.4f} | {m['unknown_reject_rate']:.4f} | {m['known_accept_rate']:.4f} |"
        )
    md_lines.append("")
    for item in items:
        md_lines.append(format_run_md(item))
        md_lines.append("")

    summary_path = out_dir / "open_set_error_analysis.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_path = out_dir / "open_set_error_analysis.json"
    json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_path)
    print(json_path)
    for item in items:
        m = item["metrics"]
        print(
            f"{Path(item['run_dir']).name}: AUROC={m['auroc']:.4f}, FPR95={m['fpr95']:.4f}, known_acc={m['known_class_accuracy_all_known']:.4f}, unknown_reject={m['unknown_reject_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
