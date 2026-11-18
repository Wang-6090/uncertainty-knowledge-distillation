"""Audit rejection-state mechanisms across six frozen paired runs.

This is a post-hoc descriptive audit. Test labels are read only after the
paired runs have completed; they are not used for training, checkpoint
selection, threshold calibration, method selection, or rule selection.

The audit uses within-run quantile bands of distance from the run's own
calibrated threshold because the baseline and uncertainty-band scores are on
different scales. It also reports candidate-pool concentration and the score
and uncertainty profiles of the four rejection states.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PAIRS = {
    "2121": {
        "baseline": "runs/rejection_ce_decay_full_s2121_static_control_discover",
        "treatment": "runs/rejection_uncertainty_band_full_s2121",
    },
    "2122": {
        "baseline": "runs/rejection_ce_decay_full_s2122_static_control_discover",
        "treatment": "runs/rejection_uncertainty_band_full_s2122",
    },
    "2123": {
        "baseline": "runs/rejection_ce_decay_full_s2123_static_control_discover",
        "treatment": "runs/rejection_uncertainty_band_full_s2123",
    },
    "2124": {
        "baseline": "runs/rejection_ce_decay_full_s2124_static_control_discover",
        "treatment": "runs/rejection_uncertainty_band_full_s2124",
    },
    "2125": {
        "baseline": "runs/rejection_ce_decay_full_s2125_static_control_discover",
        "treatment": "runs/rejection_uncertainty_band_full_s2125",
    },
    "2127": {
        "baseline": "runs/rejection_ce_decay_full_s2127_static_control_discover",
        "treatment": "runs/rejection_uncertainty_band_full_s2127",
    },
}

UNCERTAINTY_FIELDS = (
    "entropy",
    "epistemic",
    "aleatoric",
    "expected_entropy",
    "head_uncertainty",
)
BAND_NAMES = ("q0_10", "q10_25", "q25_50", "q50_75", "q75_100")
BAND_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def gini(values: list[int]) -> float | None:
    if not values:
        return None
    x = np.sort(np.asarray(values, dtype=float))
    total = float(x.sum())
    if total <= 0:
        return 0.0
    index = np.arange(1, x.size + 1, dtype=float)
    return float((2 * np.sum(index * x) / (x.size * total)) - (x.size + 1) / x.size)


def entropy_from_counts(values: list[int]) -> float | None:
    total = sum(values)
    if total <= 0:
        return None
    probs = np.asarray(values, dtype=float) / total
    return float(-(probs * np.log(probs)).sum())


def load_run(root: Path) -> dict[str, Any]:
    report = json.loads((root / "discovery_report.json").read_text(encoding="utf-8"))
    detail = json.loads((root / "discovery_detail.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "calibration_report.json").read_text(encoding="utf-8"))

    threshold = float(report["score_distribution"]["threshold"])
    score = np.asarray(detail["score"], dtype=float)
    true_known = np.asarray(detail["true_known"], dtype=int)
    pred_known = np.asarray(detail["pred_known"], dtype=int)
    raw_labels = np.asarray(detail["raw_labels"], dtype=int)
    pred_class = np.asarray(detail["pred_class"], dtype=int)
    if not (len(score) == len(true_known) == len(pred_known) == len(raw_labels) == len(pred_class)):
        raise ValueError(f"length mismatch in {root}")

    boundary = calibration.get("boundary_uncertainty") or {}
    score_std = finite_float(boundary.get("score_std"))
    return {
        "root": str(root),
        "report": report,
        "detail": detail,
        "calibration": calibration,
        "threshold": threshold,
        "score": score,
        "distance": np.abs(score - threshold),
        "true_known": true_known,
        "pred_known": pred_known,
        "raw_labels": raw_labels,
        "pred_class": pred_class,
        "score_std_from_calibration": score_std,
    }


def band_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    distance = run["distance"]
    edges = np.quantile(distance, BAND_QUANTILES)
    true_known = run["true_known"]
    pred_known = run["pred_known"]
    rows = []
    for index, name in enumerate(BAND_NAMES):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == len(BAND_NAMES) - 1:
            mask = (distance >= lower) & (distance <= upper)
        else:
            mask = (distance >= lower) & (distance < upper)
        candidates = pred_known[mask] == 0
        known = true_known[mask] == 1
        unknown = ~known
        candidate_count = int(candidates.sum())
        known_candidates = int((candidates & known).sum())
        unknown_candidates = int((candidates & unknown).sum())
        rows.append(
            {
                "band": name,
                "distance_lower": lower,
                "distance_upper": upper,
                "sample_count": int(mask.sum()),
                "candidate_count": candidate_count,
                "unknown_count": int(unknown.sum()),
                "known_count": int(known.sum()),
                "unknown_reject_rate": safe_rate(unknown_candidates, int(unknown.sum())),
                "known_contamination_rate": safe_rate(known_candidates, candidate_count),
                "candidate_purity": safe_rate(unknown_candidates, candidate_count),
                "candidate_rate": safe_rate(candidate_count, int(mask.sum())),
            }
        )
    return rows


def state_rows(run: dict[str, Any]) -> dict[str, Any]:
    true_known = run["true_known"]
    pred_known = run["pred_known"]
    states = {
        "accepted_known": (true_known == 1) & (pred_known == 1),
        "rejected_known": (true_known == 1) & (pred_known == 0),
        "accepted_unknown": (true_known == 0) & (pred_known == 1),
        "rejected_unknown": (true_known == 0) & (pred_known == 0),
    }
    result = {}
    for name, mask in states.items():
        row = {
            "count": int(mask.sum()),
            "score": summarize(run["score"][mask]),
            "distance": summarize(run["distance"][mask]),
        }
        for field in UNCERTAINTY_FIELDS:
            if field in run["detail"]:
                row[field] = summarize(np.asarray(run["detail"][field], dtype=float)[mask])
        result[name] = row
    return result


def candidate_concentration(run: dict[str, Any]) -> dict[str, Any]:
    candidate = run["pred_known"] == 0
    unknown_candidate = candidate & (run["true_known"] == 0)
    known_candidate = candidate & (run["true_known"] == 1)
    result = {}
    for name, mask, label_key in (
        ("all_candidates", candidate, "pred_class"),
        ("unknown_candidates", unknown_candidate, "raw_labels"),
        ("known_contamination", known_candidate, "raw_labels"),
    ):
        labels = run[label_key][mask]
        counts = Counter(int(label) for label in labels.tolist())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        values = [count for _, count in ordered]
        total = int(labels.size)
        top5 = int(sum(count for _, count in ordered[:5]))
        top10 = int(sum(count for _, count in ordered[:10]))
        result[name] = {
            "count": total,
            "distinct_labels": len(ordered),
            "top5_share": safe_rate(top5, total),
            "top10_share": safe_rate(top10, total),
            "label_entropy": entropy_from_counts(values),
            "label_gini": gini(values),
            "top10": [{"label": label, "count": count} for label, count in ordered[:10]],
        }

    unknown_labels = set(run["raw_labels"][run["true_known"] == 0].tolist())
    unknown_candidate_labels = set(run["raw_labels"][unknown_candidate].tolist())
    result["unknown_class_coverage"] = {
        "test_unknown_classes": len(unknown_labels),
        "candidate_unknown_classes": len(unknown_candidate_labels),
        "coverage": safe_rate(len(unknown_candidate_labels), len(unknown_labels)),
    }
    return result


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    report = run["report"]
    calibration = run["calibration"]
    return {
        "root": run["root"],
        "threshold": run["threshold"],
        "calibration_score_std": run["score_std_from_calibration"],
        "threshold_policy": calibration.get("threshold_policy"),
        "boundary_uncertainty": calibration.get("boundary_uncertainty"),
        "metrics": {
            key: report.get(key)
            for key in (
                "auroc",
                "aupr",
                "fpr95",
                "oscr",
                "known_accept_rate",
                "unknown_reject_rate",
                "cluster_acc",
                "cluster_nmi",
                "cluster_ari",
            )
        },
        "distance_bands": band_rows(run),
        "states": state_rows(run),
        "candidate_concentration": candidate_concentration(run),
    }


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def paired_delta(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for key in baseline["metrics"]:
        metrics[key] = delta(baseline["metrics"][key], treatment["metrics"][key])
    b_cov = baseline["candidate_concentration"]["unknown_class_coverage"]["coverage"]
    t_cov = treatment["candidate_concentration"]["unknown_class_coverage"]["coverage"]
    b_candidates = baseline["candidate_concentration"]["all_candidates"]["count"]
    t_candidates = treatment["candidate_concentration"]["all_candidates"]["count"]
    b_unknown_candidates = baseline["candidate_concentration"]["unknown_candidates"]["count"]
    t_unknown_candidates = treatment["candidate_concentration"]["unknown_candidates"]["count"]
    return {
        "metrics_treatment_minus_baseline": metrics,
        "candidate_unknown_class_coverage_delta": delta(b_cov, t_cov),
        "candidate_count_delta": t_candidates - b_candidates,
        "candidate_purity_delta": delta(
            safe_rate(b_unknown_candidates, b_candidates),
            safe_rate(t_unknown_candidates, t_candidates),
        ),
    }


def finite_list(rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-split rejection mechanism audit",
        "",
        "本报告对六组已冻结的 baseline/treatment 配对结果进行事后机制分析，不启动训练。测试标签只用于实验完成后的描述性统计，不参与训练、checkpoint 选择、阈值校准、候选规则选择或方法选择。",
        "",
        "由于 baseline 与 uncertainty-band 的分数尺度不同，边界分析使用每个 run 自己的校准阈值，并按样本到该阈值的距离分为五个分位区间。区间只用于描述样本状态如何随边界距离变化，不构成新的阈值策略。",
        "",
        "## 审计问题",
        "",
        "1. 距离拒识阈值远近是否对应稳定的未知覆盖率、已知污染率和候选池纯度变化？",
        "2. 四种拒识状态（已知接受、已知误拒、未知误接受、未知正确拒识）的分数和不确定性是否存在跨划分稳定差异？",
        "3. 候选池是否被少数预测类别或真实类别集中占据，且这种集中是否跨划分稳定？",
        "",
        "## 运行级结果",
        "",
        "| split | arm | AUROC | unknown reject | candidate count | candidate purity | candidate top-10 predicted-class share | unknown class coverage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, pair in report["runs"].items():
        for arm in ("baseline", "treatment"):
            row = pair[arm]
            metrics = row["metrics"]
            conc = row["candidate_concentration"]
            candidate_purity = safe_rate(
                conc["unknown_candidates"]["count"], conc["all_candidates"]["count"]
            )
            lines.append(
                f"| {split} | {arm} | {metrics['auroc']:.4f} | {metrics['unknown_reject_rate']:.2%} | "
                f"{conc['all_candidates']['count']} | {candidate_purity:.2%} | "
                f"{conc['all_candidates']['top10_share']:.2%} | "
                f"{conc['unknown_class_coverage']['coverage']:.2%} |"
            )

    lines += [
        "",
        "candidate purity 定义为候选池中真实未知样本的比例，即 `unknown_candidates / all_candidates`；它与未知类别拒识率不同，前者描述候选池是否干净，后者描述未知样本被拒识的覆盖程度。候选 top-10 predicted-class share 用于描述候选样本是否集中在少数预测类别。运行级表格只用于定位数据范围，不能把一次 paired change 解释为稳定改进。",
        "",
        "## 阈值距离分层",
        "",
        "| split | arm | band | distance range | candidate rate | unknown reject rate | known contamination | candidate purity |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, pair in report["runs"].items():
        for arm in ("baseline", "treatment"):
            for row in pair[arm]["distance_bands"]:
                def pct(value: float | None) -> str:
                    return "NA" if value is None else f"{value:.2%}"
                lines.append(
                    f"| {split} | {arm} | {row['band']} | {row['distance_lower']:.3f}–{row['distance_upper']:.3f} | "
                    f"{pct(row['candidate_rate'])} | {pct(row['unknown_reject_rate'])} | "
                    f"{pct(row['known_contamination_rate'])} | {pct(row['candidate_purity'])} |"
                )

    lines += [
        "",
        "分层结果应重点看形状是否一致：如果离阈值越近，已知污染率稳定升高、候选纯度稳定下降，才说明边界样本确实是主要误差来源；如果不同 split 的方向不一致，就不能据此设计新的边界规则。",
        "",
        "## 跨划分汇总",
        "",
    ]
    pairs = []
    for split, pair in report["runs"].items():
        pairs.append(pair["delta"])
    for metric in ("auroc", "unknown_reject_rate", "cluster_acc", "cluster_nmi", "cluster_ari"):
        values = [x["metrics_treatment_minus_baseline"].get(metric) for x in pairs]
        values = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
        if values:
            lines.append(f"- treatment 相对 baseline 的 `{metric}` 变化：均值 `{np.mean(values):+.6f}`，正向 split `{sum(x > 0 for x in values)}/{len(values)}`，范围 `[{min(values):+.6f}, {max(values):+.6f}]`。")
    coverage_values = [x["candidate_unknown_class_coverage_delta"] for x in pairs]
    coverage_values = [float(x) for x in coverage_values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if coverage_values:
        lines.append(f"- 候选未知类别覆盖率变化：均值 `{np.mean(coverage_values):+.4f}`，正向 split `{sum(x > 0 for x in coverage_values)}/{len(coverage_values)}`，范围 `[{min(coverage_values):+.4f}, {max(coverage_values):+.4f}]`。")

    lines += [
        "",
        "## 拒识状态与不确定性",
        "",
        "下表给出六组 run 中四种状态的平均样本数、分数距离和主要不确定性均值。状态定义为：已知接受、已知误拒、未知误接受、未知正确拒识。",
        "",
        "| arm | state | mean count | mean distance | entropy | epistemic | head uncertainty |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("baseline", "treatment"):
        for state in ("accepted_known", "rejected_known", "accepted_unknown", "rejected_unknown"):
            rows = [pair[arm]["states"][state] for pair in report["runs"].values()]
            def avg(path: tuple[str, ...]) -> str:
                vals = finite_list(rows, path)
                return "NA" if not vals else f"{np.mean(vals):.4f}"
            lines.append(
                f"| {arm} | {state} | {avg(('count',))} | {avg(('distance','mean'))} | "
                f"{avg(('entropy','mean'))} | {avg(('epistemic','mean'))} | {avg(('head_uncertainty','mean'))} |"
            )

    lines += [
        "",
        "## 候选池集中度",
        "",
        "候选池集中度使用候选样本的预测已知类别（`pred_class`）和事后真实类别（`raw_labels`）分别统计。真实类别统计只用于解释误差来源，不能用于提出测试集类别加权规则。",
        "",
        "| split | arm | all-candidate top-10 predicted-class share | unknown-candidate top-10 true-class share | unknown class coverage | known contamination top-10 true-class share |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for split, pair in report["runs"].items():
        for arm in ("baseline", "treatment"):
            conc = pair[arm]["candidate_concentration"]
            lines.append(
                f"| {split} | {arm} | {conc['all_candidates']['top10_share']:.2%} | "
                f"{conc['unknown_candidates']['top10_share']:.2%} | {conc['unknown_class_coverage']['coverage']:.2%} | "
                f"{conc['known_contamination']['top10_share']:.2%} |"
            )

    lines += [
        "",
        "## 判断",
        "",
        "本审计只有在三个条件同时满足时才支持继续做新实验：阈值距离分层在多数 split 呈现同方向变化；四种状态的不确定性差异稳定；候选池集中度对应一个能够仅由已知训练/验证数据预先定义的控制机制。若这些条件不满足，就只保留诊断结果，不再扫描阈值、融合权重或测试集类别权重。",
        "",
        "机器可读结果：`analysis/cross_split_rejection_mechanisms_20261114.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "description": "Post-hoc audit of six frozen paired rejection runs.",
        "test_labels_used_only_for_posthoc_description": True,
        "pairs": PAIRS,
        "runs": {},
    }
    for split, paths in PAIRS.items():
        baseline = run_summary(load_run(Path(paths["baseline"])))
        treatment = run_summary(load_run(Path(paths["treatment"])))
        report["runs"][split] = {
            "baseline": baseline,
            "treatment": treatment,
            "delta": paired_delta(baseline, treatment),
        }

    json_path = args.output_dir / "cross_split_rejection_mechanisms_20261114.json"
    md_path = args.output_dir / "cross_split_rejection_mechanisms_20261114.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
