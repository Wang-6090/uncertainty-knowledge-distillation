"""Attribute fixed uncertainty-fusion changes to ranking and calibration effects.

This is a retrospective, evaluation-only audit of frozen paired discovery runs.
It does not train a model, select a checkpoint, tune a fusion weight, or use
test labels for calibration or method selection. Test labels are used only to
describe the completed predictions after the runs have finished.

The audit keeps two questions separate:

* ranking: does the fusion score reorder known and unknown samples? AUROC is
  threshold independent, so its paired change is the direct ranking evidence;
* operating point: after each score is calibrated independently to 95% known
  validation coverage, which test samples change accept/reject state, and how
  close are those samples to their own calibrated threshold?

The absolute control and fusion thresholds are intentionally not compared.
Their score scales differ. Standardized margins are computed within each
score mode using that mode's known-validation score distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PAIRS: dict[int, tuple[str, str, str | None, str | None]] = {
    2084: (
        "runs/shared_rejection_uncertainty_margin_s2084_baseline_discover",
        "runs/shared_rejection_uncertainty_margin_s2084_baseline_uncertainty_fusion",
        "runs/shared_rejection_fusion_calibration_s2084_control",
        "runs/shared_rejection_fusion_calibration_s2084_fusion",
    ),
    2085: (
        "runs/shared_rejection_uncertainty_margin_s2085_baseline_discover",
        "runs/shared_rejection_uncertainty_margin_s2085_baseline_uncertainty_fusion",
        "runs/shared_rejection_fusion_calibration_s2085_control",
        "runs/shared_rejection_fusion_calibration_s2085_fusion",
    ),
    2086: (
        "runs/shared_rejection_fusion_fresh_s2086_baseline_discover",
        "runs/shared_rejection_fusion_fresh_s2086_baseline_uncertainty_fusion",
        "runs/shared_rejection_fusion_calibration_s2086_control",
        "runs/shared_rejection_fusion_calibration_s2086_fusion",
    ),
    2089: (
        "runs/shared_rejection_fusion_fresh_s2089_baseline_discover",
        "runs/shared_rejection_fusion_fresh_s2089_baseline_uncertainty_fusion",
        "runs/shared_rejection_fusion_calibration_s2089_control",
        "runs/shared_rejection_fusion_calibration_s2089_fusion",
    ),
    2092: (
        "runs/shared_rejection_fusion_fresh_s2092_baseline_discover",
        "runs/shared_rejection_fusion_fresh_s2092_baseline_uncertainty_fusion",
        "runs/shared_rejection_fusion_boundary_export_s2092_control",
        "runs/shared_rejection_fusion_boundary_export_s2092_fusion",
    ),
    2098: (
        "runs/shared_rejection_fusion_fresh_s2098_baseline_discover",
        "runs/shared_rejection_fusion_fresh_s2098_baseline_uncertainty_fusion",
        "runs/shared_rejection_fusion_boundary_export_s2098_control",
        "runs/shared_rejection_fusion_boundary_export_s2098_fusion",
    ),
    2103: (
        "runs/shared_rejection_fusion_fresh_s2103_baseline_discover",
        "runs/shared_rejection_fusion_fresh_s2103_baseline_uncertainty_fusion",
        None,
        None,
    ),
}

METRICS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "unknown_reject_rate",
    "known_accept_rate",
    "known_class_accuracy_after_accept",
)
BANDS = (0.25, 0.5, 1.0, 2.0)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def as_array(detail: dict[str, Any], key: str, dtype: Any = np.float64) -> np.ndarray:
    if key not in detail:
        raise KeyError(f"discovery_detail.json is missing {key!r}")
    return np.asarray(detail[key], dtype=dtype)


def sha256_ints(values: np.ndarray) -> str:
    payload = ",".join(str(int(value)) for value in values.tolist()).encode()
    return hashlib.sha256(payload).hexdigest()


def average_rank(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without requiring scipy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_rank(left)
    right_rank = average_rank(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    return float(left_centered.dot(right_centered) / denominator) if denominator else 1.0


def load_calibration(path: Path) -> dict[str, Any]:
    score_path = path / "calibration_scores.npz"
    report_path = path / "calibration_report.json"
    if not score_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"incomplete calibration export: {path}")
    with np.load(score_path, allow_pickle=False) as payload:
        required = {"score", "source_index", "calibrated_threshold"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{score_path} missing arrays: {sorted(missing)}")
        scores = np.asarray(payload["score"], dtype=np.float64)
        source_index = np.asarray(payload["source_index"], dtype=np.int64)
        threshold = float(np.asarray(payload["calibrated_threshold"]).item())
    report = read_json(report_path)
    score_export = report.get("score_export", {})
    if score_export.get("contains_test_samples") is not False:
        raise ValueError(f"{report_path}: calibration export is not known-validation-only")
    if score_export.get("labels_exported") is True:
        raise ValueError(f"{report_path}: labels were exported in calibration metadata")
    if len(scores) != len(source_index) or len(np.unique(source_index)) != len(source_index):
        raise ValueError(f"{score_path}: score/source-index arrays are not aligned")
    if score_export.get("threshold_policy") not in {"known_val_known_coverage", "known_coverage"}:
        raise ValueError(f"{report_path}: unexpected threshold policy")
    return {
        "scores": scores,
        "source_index": source_index,
        "threshold": threshold,
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "report": report,
    }


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def boundary_band_stats(
    margin: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    count = int(mask.sum())
    return {
        "count": count,
        **{
            f"within_{str(band).replace('.', 'p')}sigma": rate(
                int(np.logical_and(mask, margin <= band).sum()), count
            )
            for band in BANDS
        },
    }


def summarize_state(
    true_known: np.ndarray,
    pred_known: np.ndarray,
) -> dict[str, Any]:
    known = true_known.astype(bool)
    unknown = ~known
    accepted = pred_known.astype(bool)
    rejected = ~accepted
    candidate_count = int(rejected.sum())
    known_rejected = int(np.logical_and(known, rejected).sum())
    unknown_rejected = int(np.logical_and(unknown, rejected).sum())
    return {
        "sample_count": int(len(true_known)),
        "known_count": int(known.sum()),
        "unknown_count": int(unknown.sum()),
        "candidate_count": candidate_count,
        "known_rejected_count": known_rejected,
        "unknown_rejected_count": unknown_rejected,
        "unknown_coverage": rate(unknown_rejected, int(unknown.sum())),
        "known_false_rejection_rate": rate(known_rejected, int(known.sum())),
        "candidate_pool_known_contamination": rate(known_rejected, candidate_count),
        "candidate_pool_unknown_purity": rate(unknown_rejected, candidate_count),
        "known_acceptance": rate(int(np.logical_and(known, accepted).sum()), int(known.sum())),
    }


def evaluate_pair(seed: int, paths: tuple[str, str, str | None, str | None]) -> dict[str, Any]:
    control_dir, fusion_dir, control_cal_dir, fusion_cal_dir = paths
    control_root = resolve(control_dir)
    fusion_root = resolve(fusion_dir)
    control_detail = read_json(control_root / "discovery_detail.json")
    fusion_detail = read_json(fusion_root / "discovery_detail.json")
    control_report = read_json(control_root / "discovery_report.json")
    fusion_report = read_json(fusion_root / "discovery_report.json")

    control_raw = as_array(control_detail, "raw_labels", np.int64)
    fusion_raw = as_array(fusion_detail, "raw_labels", np.int64)
    control_true_known = as_array(control_detail, "true_known", np.int8).astype(bool)
    fusion_true_known = as_array(fusion_detail, "true_known", np.int8).astype(bool)
    control_pred_known = as_array(control_detail, "pred_known", np.int8).astype(bool)
    fusion_pred_known = as_array(fusion_detail, "pred_known", np.int8).astype(bool)
    control_score = as_array(control_detail, "score")
    fusion_score = as_array(fusion_detail, "score")
    if control_detail.get("score_mode") != "feature_rejector":
        raise ValueError(f"{seed}: unexpected control score mode")
    if fusion_detail.get("score_mode") != "feature_rejector_uncertainty_fusion":
        raise ValueError(f"{seed}: unexpected fusion score mode")
    arrays_match = {
        "length": len(control_score) == len(fusion_score),
        "raw_labels": np.array_equal(control_raw, fusion_raw),
        "true_known": np.array_equal(control_true_known, fusion_true_known),
        "pred_class": np.array_equal(
            as_array(control_detail, "pred_class", np.int64),
            as_array(fusion_detail, "pred_class", np.int64),
        ),
        "feature_rejector_score": np.allclose(
            as_array(control_detail, "feature_rejector_score"),
            as_array(fusion_detail, "feature_rejector_score"),
            atol=1e-6,
            rtol=1e-6,
        ),
        "head_uncertainty": np.allclose(
            as_array(control_detail, "head_uncertainty"),
            as_array(fusion_detail, "head_uncertainty"),
            atol=1e-6,
            rtol=1e-6,
        ),
        "entropy": np.allclose(
            as_array(control_detail, "entropy"),
            as_array(fusion_detail, "entropy"),
            atol=1e-6,
            rtol=1e-6,
        ),
    }
    # The feature-rejector score must be identical when the two runs are to be
    # interpreted as a strict component comparison.  Earlier paired runs may
    # have used different open-validation settings, which can refit the
    # rejector inputs even when the final test samples are aligned.  Keep those
    # runs for descriptive metric/state comparisons, but never use them for
    # boundary attribution or call them strict component pairs.
    core_alignment = all(
        arrays_match[key]
        for key in (
            "length",
            "raw_labels",
            "true_known",
            "pred_class",
            "head_uncertainty",
            "entropy",
        )
    )
    if not core_alignment:
        raise ValueError(f"{seed}: paired discovery arrays are not aligned: {arrays_match}")
    strict_component_paired = bool(arrays_match["feature_rejector_score"])

    control_state = summarize_state(control_true_known, control_pred_known)
    fusion_state = summarize_state(fusion_true_known, fusion_pred_known)
    control_accept_fusion_reject = np.logical_and(control_pred_known, ~fusion_pred_known)
    control_reject_fusion_accept = np.logical_and(~control_pred_known, fusion_pred_known)
    any_flip = np.logical_xor(control_pred_known, fusion_pred_known)

    control_cal = None
    fusion_cal = None
    margin_result: dict[str, Any] | None = None
    if control_cal_dir and fusion_cal_dir and strict_component_paired:
        control_cal = load_calibration(resolve(control_cal_dir))
        fusion_cal = load_calibration(resolve(fusion_cal_dir))
        if not np.array_equal(control_cal["source_index"], fusion_cal["source_index"]):
            raise ValueError(f"{seed}: calibration source indices are not paired")
        control_margin = np.abs((control_score - control_cal["threshold"]) / control_cal["std"])
        fusion_margin = np.abs((fusion_score - fusion_cal["threshold"]) / fusion_cal["std"])
        margin_result = {
            "control": {
                "threshold": control_cal["threshold"],
                "calibration_mean": control_cal["mean"],
                "calibration_std": control_cal["std"],
                "threshold_standardized": (control_cal["threshold"] - control_cal["mean"]) / control_cal["std"],
                "all_test_margin": boundary_band_stats(control_margin, np.ones(len(control_margin), dtype=bool)),
                "control_accept_fusion_reject_margin": boundary_band_stats(control_margin, control_accept_fusion_reject),
                "control_reject_fusion_accept_margin": boundary_band_stats(control_margin, control_reject_fusion_accept),
            },
            "fusion": {
                "threshold": fusion_cal["threshold"],
                "calibration_mean": fusion_cal["mean"],
                "calibration_std": fusion_cal["std"],
                "threshold_standardized": (fusion_cal["threshold"] - fusion_cal["mean"]) / fusion_cal["std"],
                "all_test_margin": boundary_band_stats(fusion_margin, np.ones(len(fusion_margin), dtype=bool)),
                "control_accept_fusion_reject_margin": boundary_band_stats(fusion_margin, control_accept_fusion_reject),
                "control_reject_fusion_accept_margin": boundary_band_stats(fusion_margin, control_reject_fusion_accept),
            },
            "both_modes_margin": {
                "any_flip": {
                    "count": int(any_flip.sum()),
                    "within_0p25sigma_both": rate(int(np.logical_and(any_flip, np.logical_and(control_margin <= 0.25, fusion_margin <= 0.25)).sum()), int(any_flip.sum())),
                    "within_0p5sigma_both": rate(int(np.logical_and(any_flip, np.logical_and(control_margin <= 0.5, fusion_margin <= 0.5)).sum()), int(any_flip.sum())),
                    "within_1p0sigma_both": rate(int(np.logical_and(any_flip, np.logical_and(control_margin <= 1.0, fusion_margin <= 1.0)).sum()), int(any_flip.sum())),
                    "within_2p0sigma_both": rate(int(np.logical_and(any_flip, np.logical_and(control_margin <= 2.0, fusion_margin <= 2.0)).sum()), int(any_flip.sum())),
                },
            },
        }

    metrics = {
        "control": {metric: float(control_report[metric]) for metric in METRICS},
        "fusion": {metric: float(fusion_report[metric]) for metric in METRICS},
    }
    metrics["delta_fusion_minus_control"] = {
        metric: metrics["fusion"][metric] - metrics["control"][metric]
        for metric in METRICS
    }
    result = {
        "seed": seed,
        "control_path": str(control_root),
        "fusion_path": str(fusion_root),
        "pair_protocol": {
            "control_score_mode": "feature_rejector",
            "fusion_score_mode": "feature_rejector_uncertainty_fusion",
            "control_threshold_policy": control_report.get("threshold_policy"),
            "fusion_threshold_policy": fusion_report.get("threshold_policy"),
            "test_labels_used_for_training_or_calibration": control_report.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration"),
            "hard_labels_used": control_report.get("rejector_protocol", {}).get("hard_labels_used"),
        },
        "alignment_audit": {
            **arrays_match,
            "strict_component_paired": strict_component_paired,
            "raw_label_sha256": sha256_ints(control_raw),
            "interpretation": (
                "strict component pair: feature_rejector_score is identical"
                if strict_component_paired
                else "descriptive pair only: feature_rejector_score differs, so fusion-only attribution is not valid"
            ),
        },
        "control_state": control_state,
        "fusion_state": fusion_state,
        "state_delta_fusion_minus_control": {
            key: fusion_state[key] - control_state[key]
            for key in (
                "candidate_count",
                "known_rejected_count",
                "unknown_rejected_count",
                "unknown_coverage",
                "known_false_rejection_rate",
                "candidate_pool_known_contamination",
                "candidate_pool_unknown_purity",
                "known_acceptance",
            )
        },
        "state_flips": {
            "control_accept_fusion_reject_count": int(control_accept_fusion_reject.sum()),
            "control_reject_fusion_accept_count": int(control_reject_fusion_accept.sum()),
            "any_flip_count": int(any_flip.sum()),
            "control_accept_fusion_reject_known_count": int(np.logical_and(control_accept_fusion_reject, control_true_known).sum()),
            "control_accept_fusion_reject_unknown_count": int(np.logical_and(control_accept_fusion_reject, ~control_true_known).sum()),
            "control_reject_fusion_accept_known_count": int(np.logical_and(control_reject_fusion_accept, control_true_known).sum()),
            "control_reject_fusion_accept_unknown_count": int(np.logical_and(control_reject_fusion_accept, ~control_true_known).sum()),
        },
        "rank_audit": {
            "score_spearman": spearman_correlation(control_score, fusion_score),
            "control_score_std": float(control_score.std(ddof=1)),
            "fusion_score_std": float(fusion_score.std(ddof=1)),
        },
        "metrics": metrics,
        "margin_audit": margin_result,
        "calibration_export_available": margin_result is not None,
        "strict_component_paired": strict_component_paired,
    }
    return result


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_deltas = {
        metric: {
            "mean": mean([row["metrics"]["delta_fusion_minus_control"][metric] for row in results]),
            "std": float(np.std([row["metrics"]["delta_fusion_minus_control"][metric] for row in results], ddof=1)),
            "positive_count": int(sum(row["metrics"]["delta_fusion_minus_control"][metric] > 0 for row in results)),
        }
        for metric in METRICS
    }
    state_keys = (
        "candidate_count",
        "known_rejected_count",
        "unknown_rejected_count",
        "unknown_coverage",
        "known_false_rejection_rate",
        "candidate_pool_known_contamination",
        "candidate_pool_unknown_purity",
        "known_acceptance",
    )
    state_deltas = {
        key: {
            "mean": mean([row["state_delta_fusion_minus_control"][key] for row in results]),
            "std": float(np.std([row["state_delta_fusion_minus_control"][key] for row in results], ddof=1)),
        }
        for key in state_keys
    }
    flips = {
        key: {
            "mean": mean([row["state_flips"][key] for row in results]),
            "total": int(sum(row["state_flips"][key] for row in results)),
        }
        for key in (
            "control_accept_fusion_reject_count",
            "control_reject_fusion_accept_count",
            "any_flip_count",
            "control_accept_fusion_reject_known_count",
            "control_accept_fusion_reject_unknown_count",
            "control_reject_fusion_accept_known_count",
            "control_reject_fusion_accept_unknown_count",
        )
    }
    margin_results = [row for row in results if row["margin_audit"] is not None]
    return {
        "pair_count": len(results),
        "strict_component_pair_count": int(sum(row["strict_component_paired"] for row in results)),
        "descriptive_only_pair_count": int(sum(not row["strict_component_paired"] for row in results)),
        "margin_pair_count": len(margin_results),
        "metric_deltas": metric_deltas,
        "state_deltas": state_deltas,
        "state_flips": flips,
        "mean_score_spearman": mean([row["rank_audit"]["score_spearman"] for row in results]),
        "mean_test_score_std": {
            "control": mean([row["rank_audit"]["control_score_std"] for row in results]),
            "fusion": mean([row["rank_audit"]["fusion_score_std"] for row in results]),
        },
        "interpretation": {
            "ranking_evidence": "AUROC is threshold independent; its paired change is the direct evidence that score ordering changed.",
            "operating_point_evidence": "State flips and test known acceptance are measured after each mode is separately calibrated to 95% known validation coverage.",
        "threshold_comparison_rule": "Absolute control/fusion thresholds are not compared because the two score modes have different scales; standardized margins are computed within each mode only for strict component pairs.",
        "strict_pair_rule": "A pair is strict only when the aligned test feature_rejector_score arrays are identical; otherwise metrics and state flips are descriptive and cannot be attributed to uncertainty fusion alone.",
            "2103_limit": "Seed 2103 is included in state and metric comparisons but has no calibration score export, so it is excluded from standardized-margin analysis.",
        },
    }


def fmt(value: float, digits: int = 4) -> str:
    return "—" if not math.isfinite(value) else f"{value:+.{digits}f}"


def markdown(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Uncertainty-fusion boundary attribution audit (2084–2103)",
        "",
        "本记录是对已完成 paired discovery 结果的回顾性审计。没有重新训练学生模型、重新选择 checkpoint、扫描融合权重或用测试标签校准阈值；测试标签只用于实验结束后的状态翻转、候选池组成和指标描述。",
        "",
        "## 审计目的和判别规则",
        "",
        "审计把融合后的变化拆成两类证据。AUROC 与阈值无关，因此它的变化直接反映风险排序是否改变；未知覆盖率、已知接受率和候选池污染率则是在各自 score mode 使用已知验证集 95% coverage 阈值后的工作点结果。control 与 fusion 的阈值处于不同分数尺度，不能直接比较绝对值；阈值附近分析使用各自校准分布的标准差归一化。",
        "",
        "## 成对数据和状态翻转",
        "",
        "| seed | 配对性质 | AUROC Δ | 未知覆盖率 Δ | 已知接受率 Δ | 候选池已知污染率 Δ | accept→reject | reject→accept | Spearman | 标准化 margin |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        delta = row["metrics"]["delta_fusion_minus_control"]
        state_delta = row["state_delta_fusion_minus_control"]
        flips = row["state_flips"]
        lines.append(
            f"| {row['seed']} | {'严格' if row['strict_component_paired'] else '描述性'} | {fmt(delta['auroc'])} | {fmt(state_delta['unknown_coverage'])} | "
            f"{fmt(state_delta['known_acceptance'])} | {fmt(state_delta['candidate_pool_known_contamination'])} | "
            f"{flips['control_accept_fusion_reject_count']} | {flips['control_reject_fusion_accept_count']} | "
            f"{row['rank_audit']['score_spearman']:.4f} | "
            f"{'有' if row['calibration_export_available'] else '无'} |"
        )
    lines += [
        "",
        "### 七组汇总",
        "",
        f"严格组件配对 {summary['strict_component_pair_count']}/7 组；描述性配对 {summary['descriptive_only_pair_count']}/7 组。描述性配对仍可报告最终指标和状态翻转，但不能把变化单独归因于 uncertainty fusion。",
        "",
        "| 指标 | 平均变化（fusion−control） | 标准差 | 7组中上升组数 |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        stats = summary["metric_deltas"][metric]
        lines.append(f"| {metric} | {fmt(stats['mean'], 6)} | {stats['std']:.6f} | {stats['positive_count']}/7 |")
    lines += [
        "",
        "| 候选池/状态量 | 平均变化（fusion−control） | 标准差 |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("unknown_coverage", "未知覆盖率"),
        ("known_false_rejection_rate", "已知误拒率"),
        ("candidate_pool_known_contamination", "候选池已知污染率"),
        ("candidate_pool_unknown_purity", "候选池未知纯度"),
        ("known_acceptance", "已知接受率"),
    ):
        stats = summary["state_deltas"][key]
        lines.append(f"| {label} | {fmt(stats['mean'], 6)} | {stats['std']:.6f} |")
    lines += [
        "",
        "## 边界距离分析（仅含严格配对且有校准导出的实验）",
        "",
        "这里的 margin 是 `|test_score−本模式阈值| / 本模式校准分数标准差`。它只表示样本距离各自工作点有多近，不能把 control 的阈值绝对值和 fusion 的阈值绝对值放在一起比较。",
        "",
        "| seed | 任一状态翻转数 | 翻转样本同时落在两模式 0.25σ 内 | 0.5σ 内 | 1σ 内 | 2σ 内 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        margin = row["margin_audit"]
        if margin is None:
            continue
        both = margin["both_modes_margin"]["any_flip"]
        lines.append(
            f"| {row['seed']} | {both['count']} | {both['within_0p25sigma_both']:.3%} | "
            f"{both['within_0p5sigma_both']:.3%} | {both['within_1p0sigma_both']:.3%} | "
            f"{both['within_2p0sigma_both']:.3%} |"
        )
    lines += [
        "",
        "## 归因判断",
        "",
        f"1. 在全部七组描述性结果中，AUROC 有 {summary['metric_deltas']['auroc']['positive_count']}/7 组上升，平均变化为 {summary['metric_deltas']['auroc']['mean']:+.6f}。AUROC 不依赖阈值，因此这些结果显示两次运行的风险排序不同；其中前三组还混有拒识器输入协议差异，不能把全部排序变化归因于 uncertainty fusion。",
        f"2. 工作点层面也发生了样本重新分配：7组平均未知覆盖率变化为 {summary['state_deltas']['unknown_coverage']['mean']:+.6f}，平均已知接受率变化为 {summary['state_deltas']['known_acceptance']['mean']:+.6f}；状态翻转既包括 control 接受而 fusion 拒识，也包括相反方向，说明收益不是单向扩大候选池。",
        "3. 边界距离分析只对严格配对且有校准导出的组执行。如果翻转主要集中在各自工作点附近，说明阈值校准会把排序差异放大为候选池变化；阈值绝对值本身仍不能解释为独立收益。",
        "4. 因此当前证据应分层表述：严格配对组支持继续检验 fusion 对风险排序和边界样本的影响；前三组只能作为描述性结果，不能用于纯组件归因。现有证据仍不足以把 fusion 改为默认配置。",
        "",
        "## 限制和后续决定",
        "",
        "2084–2086 已纳入状态翻转和最终指标比较，但因 feature_rejector_score 不一致，只作描述性参考；2089、2092、2098 可用于严格组件归因，其中 2103 虽为严格配对但没有 calibration score export，只能做排序、状态翻转和最终指标分析。该审计只能解释当前 CIFAR-100 semantic-hard 配置下的成对结果，不能证明跨数据集、跨域或更换类别划分后的稳定性。基于已有证据，不继续扫描融合权重，也不重复训练；fusion 仅保留为显式 opt-in 候选，后续若要继续，应先提出新的、可证伪的机制假设。",
        "",
        "## 可复现入口",
        "",
        "`python scripts/audit_uncertainty_fusion_boundary_attribution.py`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    results = [evaluate_pair(seed, paths) for seed, paths in PAIRS.items()]
    summary = build_summary(results)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "uncertainty_fusion_boundary_attribution_2084_2103_20261114.json"
    markdown_path = output_dir / "uncertainty_fusion_boundary_attribution_2084_2103_20261114.md"
    payload = {
        "protocol": "retrospective_paired_boundary_attribution_20261114",
        "results": results,
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(results, summary), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
