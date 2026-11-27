"""Aggregate the team-B (unknown risk + candidate screening) runs into a report.

Reads ``runs/team_b_rejection/s<seed>_<mode>_discover`` directories produced by
``scripts/run_team_b_rejection.py`` and writes

* ``metrics.json`` + ``summary.md`` inside every discover run directory
  (the "统一记录格式" required by docs/parallel_work_division.md 第9节), and
* ``summary.json`` + ``summary.md`` in the analysis directory, containing the
  per-seed table, the mean/std per rejector configuration, the paired
  scratch_cnn vs shared_backbone contrast, and the cross-seed direction check.

The script never re-runs a model and never touches unknown labels except through
the already-written discover artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RUN_PATTERN = re.compile(r"^s(?P<seed>-?\d+)_(?P<mode>[a-z0-9_]+)_discover$")

METRIC_KEYS = [
    ("auroc", "AUROC"),
    ("aupr", "AUPR"),
    ("fpr95", "FPR95"),
    ("oscr", "OSCR"),
    ("known_accept_rate", "已知接受率"),
    ("unknown_reject_rate", "未知拒识率"),
    ("pool_recall", "候选池未知召回"),
    ("pool_purity", "候选池纯度"),
    ("cluster_acc", "候选池聚类 ACC"),
    ("cluster_nmi", "候选池聚类 NMI"),
    ("cluster_ari", "候选池聚类 ARI"),
]

TABLE_COLUMNS = [
    ("auroc", "AUROC"),
    ("aupr", "AUPR"),
    ("fpr95", "FPR95"),
    ("oscr", "OSCR"),
    ("known_accept_rate", "已知接受"),
    ("unknown_reject_rate", "未知拒识"),
    ("candidates", "候选数"),
    ("pool_purity", "池纯度"),
    ("pool_recall", "池召回"),
    ("pollution", "已知污染"),
    ("cluster_acc", "聚类ACC"),
    ("cluster_nmi", "聚类NMI"),
    ("cluster_ari", "聚类ARI"),
]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None and not _isnan(value)]
    if not clean:
        return None, None
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return mean, std


def _isnan(value) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def load_run(discover_dir: Path, seed: int, mode: str) -> dict:
    report = _read_json(discover_dir / "discovery_report.json")
    calibration = _read_json(discover_dir / "calibration_report.json")
    detail = _read_json(discover_dir / "discovery_detail.json")
    config = _read_json(discover_dir / "discover_config.json")

    # Candidate-pool rates are recomputed from the per-sample arrays so that
    # "unknown recall" means "share of all unknown test samples that reach the
    # pool", not the post-purification retention reported by the pipeline.
    pred_known = np.asarray(detail["pred_known"], dtype=bool)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    candidate = ~pred_known
    unknown_total = int((~true_known).sum())
    candidate_true_unknown = int((candidate & ~true_known).sum())
    candidate_known = int((candidate & true_known).sum())
    pool_size = int(candidate.sum())
    pool_purity = float(candidate_true_unknown / pool_size) if pool_size else None
    pool_recall = (
        float(candidate_true_unknown / unknown_total) if unknown_total else None
    )

    threshold = calibration.get("calibrated_threshold")
    threshold_value = (
        float(np.asarray(threshold, dtype=np.float64))
        if np.asarray(threshold).ndim == 0
        else None
    )
    return {
        "seed": seed,
        "mode": mode,
        "run_dir": str(discover_dir),
        "run_name": discover_dir.name,
        "threshold": threshold_value,
        "threshold_policy": calibration.get("threshold_policy", {}).get("type"),
        "calibration_known_accept_rate": calibration.get("threshold_policy", {}).get(
            "known_accept_rate"
        ),
        "score_mode": detail.get("score_mode")
        or calibration.get("selected", {}).get("score_mode"),
        "auroc": report.get("auroc"),
        "aupr": report.get("aupr"),
        "fpr95": report.get("fpr95"),
        "oscr": report.get("oscr"),
        "known_accept_rate": report.get("known_accept_rate"),
        "unknown_reject_rate": report.get("unknown_reject_rate"),
        "known_accuracy_all_known": report.get("known_class_accuracy_all_known"),
        "known_accuracy_after_accept": report.get(
            "known_class_accuracy_after_accept"
        ),
        "candidates": pool_size,
        "candidates_before_purification": report.get(
            "cluster_candidate_count_before_purification"
        ),
        "candidate_true_unknown": candidate_true_unknown,
        "candidate_false_reject": candidate_known,
        "pool_purity": pool_purity,
        "pool_recall": pool_recall,
        "pollution": (
            float(candidate_known / pool_size) if pool_size else None
        ),
        "cluster_acc": report.get("cluster_unknown_only_acc", report.get("cluster_acc")),
        "cluster_nmi": report.get("cluster_unknown_only_nmi", report.get("cluster_nmi")),
        "cluster_ari": report.get("cluster_unknown_only_ari", report.get("cluster_ari")),
        "cluster_k": report.get("cluster_k"),
        "cluster_true_k": report.get("cluster_true_k"),
        "cluster_feature": report.get("cluster_feature"),
        "cluster_method": report.get("cluster_method"),
        "test_sample_count": int(len(detail.get("score", []))),
        "discover_config": config,
    }


def discover_runs(work_root: Path) -> list[dict]:
    runs = []
    for path in sorted(work_root.iterdir()):
        if not path.is_dir():
            continue
        match = RUN_PATTERN.match(path.name)
        if not match:
            continue
        runs.append(load_run(path, int(match.group("seed")), match.group("mode")))
    return runs


def write_run_records(run: dict) -> None:
    """Write the per-run metrics.json / summary.md required by the protocol."""
    run_dir = Path(run["run_dir"])
    metrics = {
        "run": run["run_name"],
        "seed": run["seed"],
        "rejector_configuration": run["mode"],
        "protocol": {
            "score_mode": run["score_mode"],
            "threshold_policy": run["threshold_policy"],
            "threshold": run["threshold"],
            "cluster_feature": run["cluster_feature"],
            "cluster_method": run["cluster_method"],
            "cluster_k": run["cluster_k"],
            "test_sample_count": run["test_sample_count"],
        },
        "metrics": {
            "auroc": run["auroc"],
            "aupr": run["aupr"],
            "fpr95": run["fpr95"],
            "oscr": run["oscr"],
            "known_accept_rate": run["known_accept_rate"],
            "unknown_reject_rate": run["unknown_reject_rate"],
            "known_accuracy_all_known": run["known_accuracy_all_known"],
            "candidate_count": run["candidates"],
            "candidate_true_unknown": run["candidate_true_unknown"],
            "candidate_false_reject": run["candidate_false_reject"],
            "candidate_purity": run["pool_purity"],
            "candidate_unknown_recall": run["pool_recall"],
            "cluster_acc_unknown_only": run["cluster_acc"],
            "cluster_nmi_unknown_only": run["cluster_nmi"],
            "cluster_ari_unknown_only": run["cluster_ari"],
        },
        "interpretation_limit": (
            "Unknown labels are used only for retrospective evaluation; the threshold "
            "is calibrated on known validation data."
        ),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = f"""# {run['run_name']}

## 1. 本次只改变了什么

本 run 属于李柯颖线（未知风险估计与候选筛选）。固定项：学生模型、数据集划分、
`{run['score_mode']}` 风险分数、已知验证集阈值策略 `{run['threshold_policy']}`、
聚类设置 `{run['cluster_feature']}` + `{run['cluster_method']}`。
本 run 的可变项只有随机种子（`{run['seed']}`）与拒识器配置（`{run['mode']}`）。

## 2. 与哪个正式基线比较

- 同一 seed 的 `scratch_cnn` 与 `shared_backbone` 互为对照；
- seed 之间的同一配置用于检查阈值与候选标记的稳定性。

## 3. 指标

| 指标 | 数值 |
| --- | ---: |
| 阈值 | {_fmt(run['threshold'], 6)} |
| AUROC | {_fmt(run['auroc'])} |
| AUPR | {_fmt(run['aupr'])} |
| FPR95 | {_fmt(run['fpr95'])} |
| OSCR | {_fmt(run['oscr'])} |
| 已知接受率 | {_fmt(run['known_accept_rate'])} |
| 未知拒识率 | {_fmt(run['unknown_reject_rate'])} |
| 候选数 | {_fmt(run['candidates'])} |
| 候选池纯度 | {_fmt(run['pool_purity'])} |
| 候选池未知召回 | {_fmt(run['pool_recall'])} |
| 候选池聚类 ACC/NMI/ARI | {_fmt(run['cluster_acc'])} / {_fmt(run['cluster_nmi'])} / {_fmt(run['cluster_ari'])} |

## 4. 三个正式种子的方向一致性

见分析目录下的聚合报告；单 run 报告不做方向性判断。

## 5. 结论

待由 `scripts/summarize_team_b_rejection.py` 生成的多 seed 报告填写。
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def render_markdown(runs: list[dict], paired: list[dict], protocol: dict) -> str:
    modes = sorted({run["mode"] for run in runs})
    seeds = sorted({run["seed"] for run in runs})

    lines = [
        "# 李柯颖线：未知风险估计与候选筛选（多 seed 汇总）",
        "",
        "> 未知标签只用于事后评测，未参与训练、阈值校准或模型选择。",
        "",
        "## 1. 固定协议",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
    ]
    for key, value in protocol.items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 2. 逐 seed 结果",
        "",
        "| seed | 拒识器 | 阈值 | AUROC | AUPR | FPR95 | OSCR | 已知接受 | 未知拒识 | 候选数 | 池纯度 | 池召回 | 已知污染 | 聚类ACC | 聚类NMI | 聚类ARI |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted(runs, key=lambda item: (item["seed"], item["mode"])):
        lines.append(
            "| {seed} | {mode} | {threshold} | {auroc} | {aupr} | {fpr95} | {oscr} | "
            "{known} | {unknown} | {cand} | {purity} | {recall} | {poll} | {acc} | {nmi} | {ari} |".format(
                seed=run["seed"],
                mode=run["mode"],
                threshold=_fmt(run["threshold"], 6),
                auroc=_fmt(run["auroc"]),
                aupr=_fmt(run["aupr"]),
                fpr95=_fmt(run["fpr95"]),
                oscr=_fmt(run["oscr"]),
                known=_fmt(run["known_accept_rate"]),
                unknown=_fmt(run["unknown_reject_rate"]),
                cand=_fmt(run["candidates"]),
                purity=_fmt(run["pool_purity"]),
                recall=_fmt(run["pool_recall"]),
                poll=_fmt(run["pollution"]),
                acc=_fmt(run["cluster_acc"]),
                nmi=_fmt(run["cluster_nmi"]),
                ari=_fmt(run["cluster_ari"]),
            )
        )

    lines += ["", "## 3. 各拒识器配置的均值 ± 标准差", ""]
    header = "| 指标 | " + " | ".join(modes) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in modes) + " |"
    lines += [header, separator]
    for key, label in METRIC_KEYS:
        cells = []
        for mode in modes:
            values = [run[key] for run in runs if run["mode"] == mode]
            mean, std = _mean_std(values)
            cells.append("-" if mean is None else f"{mean:.4f} ± {std:.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    if paired:
        lines += [
            "",
            "## 4. 成对对照：shared_backbone − scratch_cnn",
            "",
            "| seed | AUROC 差 | FPR95 差 | OSCR 差 | 池纯度差 | 池召回差 | 候选数差 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in paired:
            delta = row["delta"]
            lines.append(
                f"| {row['seed']} | {_fmt(delta.get('auroc'))} | {_fmt(delta.get('fpr95'))} | "
                f"{_fmt(delta.get('oscr'))} | {_fmt(delta.get('pool_purity'))} | "
                f"{_fmt(delta.get('pool_recall'))} | {_fmt(delta.get('candidates'))} |"
            )
        lines += ["", "方向一致性（shared 优于 scratch 的 seed 数 / 参与 seed 数）：", ""]
        lines.append("| 指标 | 优 | 平 | 劣 | 判断 |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for key, label in (
            ("auroc", "AUROC ↑"),
            ("aupr", "AUPR ↑"),
            ("fpr95", "FPR95 ↓"),
            ("oscr", "OSCR ↑"),
            ("pool_purity", "池纯度 ↑"),
            ("pool_recall", "池召回 ↑"),
        ):
            better = equal = worse = 0
            for row in paired:
                value = row["delta"].get(key)
                if value is None:
                    continue
                better_sign = -1 if key in {"fpr95"} else 1
                scaled = value * better_sign
                if abs(scaled) < 1e-12:
                    equal += 1
                elif scaled > 0:
                    better += 1
                else:
                    worse += 1
            total = better + equal + worse
            if total == 0:
                verdict = "无数据"
            elif better == total and better >= 2:
                verdict = "同向改善"
            elif worse == total:
                verdict = "同向变差"
            elif better > worse:
                verdict = "局部信号（不一致）"
            else:
                verdict = "无稳定方向"
            lines.append(
                f"| {label} | {better} | {equal} | {worse} | {verdict} |"
            )

    lines += [
        "",
        "## 5. 判读规则",
        "",
        "- 只有在三个正式种子上同向、且改善不是靠牺牲已知污染率或候选纯度换来的，才写成稳定收益；",
        "- 单个种子或单个指标改善一律写成局部信号；",
        "- 阈值只能由已知验证集确定，禁止用测试集未知标签挑选阈值。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("runs/team_b_rejection"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None,
                        help="split file, only used for the protocol table")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    runs = discover_runs(args.work_root)
    if not runs:
        raise SystemExit(f"no s<seed>_<mode>_discover directories under {args.work_root}")

    for run in runs:
        write_run_records(run)

    paired = []
    seeds = sorted({run["seed"] for run in runs})
    for seed in seeds:
        scratch = next(
            (run for run in runs if run["seed"] == seed and run["mode"] == "scratch_cnn"),
            None,
        )
        shared = next(
            (run for run in runs if run["seed"] == seed and run["mode"] == "shared_backbone"),
            None,
        )
        if scratch is None or shared is None:
            continue
        delta = {}
        for key in ("auroc", "aupr", "fpr95", "oscr", "pool_purity", "pool_recall",
                    "pollution", "candidates", "cluster_acc"):
            left = shared.get(key)
            right = scratch.get(key)
            delta[key] = (
                None if left is None or right is None else float(left) - float(right)
            )
        paired.append({"seed": seed, "delta": delta})

    first = runs[0]
    protocol = {
        "风险分数": first["score_mode"],
        "阈值策略": first["threshold_policy"],
        "聚类特征 / 方法 / K": f"{first['cluster_feature']} / {first['cluster_method']} / {first['cluster_k']}",
        "正式种子": ", ".join(str(seed) for seed in seeds),
        "每次运行测试样本数": first["test_sample_count"],
    }
    if args.split is not None:
        protocol["类别划分文件"] = str(args.split)

    output_dir = args.output_dir or (Path("analysis/team_b_rejection"))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": protocol,
        "runs": runs,
        "paired_shared_minus_scratch": paired,
        "note": args.note
        or "未知标签只用于事后评测，未参与训练、阈值校准或模型选择。",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_markdown(runs, paired, protocol), encoding="utf-8"
    )
    print(f"runs: {len(runs)} -> {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
