"""Paired calibration/test bootstrap audit for frozen s2131 boundary runs.

This is a post-hoc uncertainty analysis of saved outputs. It does not train,
choose settings, or access unknown labels for calibration. Bootstrap intervals
are conditional on this single semantic-hard class split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "runs/boundary_uncertainty_s2131_control"
TREATMENT = ROOT / "runs/boundary_uncertainty_s2131_band1_alpha05"
OUT_JSON = ROOT / "analysis/boundary_uncertainty_s2131_bootstrap.json"
OUT_MD = ROOT / "analysis/boundary_uncertainty_s2131_bootstrap.md"
REPLICATES = 2000
BOOTSTRAP_SEED = 213115
TARGET_COVERAGE = 0.95
ALPHA = 0.5
BOUNDARY_MULTIPLIER = 1.0


def read_run(path: Path) -> dict:
    detail = json.loads((path / "discovery_detail.json").read_text(encoding="utf-8"))
    calibration = np.load(path / "calibration_scores.npz")
    report = json.loads((path / "calibration_report.json").read_text(encoding="utf-8"))
    return {
        "detail": detail,
        "calibration": {key: calibration[key] for key in calibration.files},
        "report": report,
    }


def boundary_state(raw_cal: np.ndarray, unc_cal: np.ndarray) -> tuple[float, float, float, float]:
    base_threshold = float(np.quantile(raw_cal, TARGET_COVERAGE, method="linear"))
    score_std = max(float(np.std(raw_cal, ddof=1)), 1e-6)
    unc_mean = float(np.mean(unc_cal))
    unc_std = max(float(np.std(unc_cal, ddof=1)), 1e-6)
    return base_threshold, score_std, unc_mean, unc_std


def apply_correction(
    raw: np.ndarray,
    uncertainty: np.ndarray,
    state: tuple[float, float, float, float],
) -> np.ndarray:
    base_threshold, score_std, unc_mean, unc_std = state
    mask = np.abs(raw - base_threshold) <= BOUNDARY_MULTIPLIER * score_std
    result = raw.copy()
    result[mask] += ALPHA * score_std * (uncertainty[mask] - unc_mean) / unc_std
    return result


def operating_metrics(scores: np.ndarray, threshold: float, known: np.ndarray) -> dict:
    reject = scores > threshold
    candidate_count = int(reject.sum())
    return {
        "known_accept_rate": float(np.mean(~reject[known])),
        "unknown_reject_rate": float(np.mean(reject[~known])),
        "candidate_count": candidate_count,
        "candidate_purity": float(np.mean(~known[reject])) if candidate_count else None,
    }


def pairwise_auc_component(
    unknown_scores: tuple[np.ndarray, np.ndarray],
    known_scores: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Compute unknown-vs-known AUC within one stratum.

    Each argument contains (control_scores, treatment_scores). The AUC is the
    fraction of unknown-known pairs correctly ordered, with ties counted as
    one half, matching the global AUROC definition.
    """
    unknown_control, unknown_treatment = unknown_scores
    known_control, known_treatment = known_scores
    if unknown_control.shape != unknown_treatment.shape:
        raise ValueError("unknown control/treatment score arrays are not aligned")
    if known_control.shape != known_treatment.shape:
        raise ValueError("known control/treatment score arrays are not aligned")
    unknown_count = int(unknown_control.size)
    known_count = int(known_control.size)
    pair_count = unknown_count * known_count
    if pair_count == 0:
        return {
            "unknown_count": unknown_count,
            "known_count": known_count,
            "pair_count": 0,
            "auc_control": None,
            "auc_treatment": None,
            "auc_delta": None,
        }
    labels = np.concatenate((
        np.ones(unknown_count, dtype=np.int8),
        np.zeros(known_count, dtype=np.int8),
    ))
    control_scores = np.concatenate((unknown_control, known_control))
    treatment_scores = np.concatenate((unknown_treatment, known_treatment))
    auc_control = float(roc_auc_score(labels, control_scores))
    auc_treatment = float(roc_auc_score(labels, treatment_scores))
    return {
        "unknown_count": unknown_count,
        "known_count": known_count,
        "pair_count": pair_count,
        "auc_control": auc_control,
        "auc_treatment": auc_treatment,
        "auc_delta": auc_treatment - auc_control,
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95_percentile": [float(x) for x in np.quantile(array, [0.025, 0.975])],
    }


def main() -> None:
    control = read_run(CONTROL)
    treatment = read_run(TREATMENT)
    dc, dt = control["detail"], treatment["detail"]
    known = np.asarray(dc["true_known"], dtype=bool)
    unknown_indicator = ~known
    raw_test = np.asarray(dc["raw_score"], dtype=np.float64)
    corrected_saved = np.asarray(dt["score"], dtype=np.float64)
    uncertainty_test = np.asarray(dt["head_uncertainty"], dtype=np.float64)
    raw_cal = np.asarray(control["calibration"]["raw_score"], dtype=np.float64)
    unc_cal = np.asarray(control["calibration"]["validation_head_uncertainty"], dtype=np.float64)
    raw_cal_t = np.asarray(treatment["calibration"]["raw_score"], dtype=np.float64)
    unc_cal_t = np.asarray(treatment["calibration"]["validation_head_uncertainty"], dtype=np.float64)

    if not np.array_equal(known, np.asarray(dt["true_known"], dtype=bool)):
        raise RuntimeError("paired test labels are not aligned")
    if not np.array_equal(raw_cal, raw_cal_t) or not np.array_equal(unc_cal, unc_cal_t):
        raise RuntimeError("paired calibration arrays differ")
    if not np.array_equal(raw_test, np.asarray(dt["raw_score"], dtype=np.float64)):
        raise RuntimeError("paired raw test scores differ")

    state = boundary_state(raw_cal, unc_cal)
    corrected_cal = apply_correction(raw_cal, unc_cal, state)
    corrected_test = apply_correction(raw_test, uncertainty_test, state)
    threshold_control = float(np.quantile(raw_cal, TARGET_COVERAGE, method="linear"))
    threshold_treatment = float(np.quantile(corrected_cal, TARGET_COVERAGE, method="linear"))
    saved_control_threshold = float(control["report"]["calibrated_threshold"])
    saved_treatment_threshold = float(treatment["report"]["calibrated_threshold"])
    if not np.isclose(threshold_control, saved_control_threshold, atol=1e-8):
        raise RuntimeError("recomputed control calibration threshold differs")
    if not np.isclose(threshold_treatment, saved_treatment_threshold, atol=1e-8):
        raise RuntimeError("recomputed treatment calibration threshold differs")
    max_saved_score_error = float(np.max(np.abs(corrected_test - corrected_saved)))
    if max_saved_score_error > 1e-8:
        raise RuntimeError(f"recomputed adjusted test score differs: {max_saved_score_error}")

    # Cross the score transformation with both operating thresholds. This
    # isolates ranking/score effects from recalibration effects.
    cross = {
        "raw_score_at_control_threshold": operating_metrics(raw_test, threshold_control, known),
        "adjusted_score_at_control_threshold": operating_metrics(corrected_test, threshold_control, known),
        "raw_score_at_treatment_threshold": operating_metrics(raw_test, threshold_treatment, known),
        "adjusted_score_at_treatment_threshold": operating_metrics(corrected_test, threshold_treatment, known),
    }
    rank = {
        "auroc_control_raw": float(roc_auc_score(unknown_indicator, raw_test)),
        "auroc_treatment_adjusted": float(roc_auc_score(unknown_indicator, corrected_test)),
        "auroc_delta": float(roc_auc_score(unknown_indicator, corrected_test) - roc_auc_score(unknown_indicator, raw_test)),
        "aupr_control_raw": float(average_precision_score(unknown_indicator, raw_test)),
        "aupr_treatment_adjusted": float(average_precision_score(unknown_indicator, corrected_test)),
        "aupr_delta": float(average_precision_score(unknown_indicator, corrected_test) - average_precision_score(unknown_indicator, raw_test)),
    }

    test_boundary = np.abs(raw_test - state[0]) <= BOUNDARY_MULTIPLIER * state[1]
    known_boundary_count = int(np.sum(known & test_boundary))
    unknown_boundary_count = int(np.sum(unknown_indicator & test_boundary))
    pair_strata = {
        "both_in_boundary": (unknown_indicator & test_boundary, known & test_boundary),
        "unknown_in_known_out": (unknown_indicator & test_boundary, known & ~test_boundary),
        "unknown_out_known_in": (unknown_indicator & ~test_boundary, known & test_boundary),
        "both_outside_boundary": (unknown_indicator & ~test_boundary, known & ~test_boundary),
    }
    pairwise_decomposition = {}
    total_pairs = int(np.sum(unknown_indicator) * np.sum(known))
    for name, (unknown_group, known_group) in pair_strata.items():
        pairwise_decomposition[name] = pairwise_auc_component(
            (raw_test[unknown_group], corrected_test[unknown_group]),
            (raw_test[known_group], corrected_test[known_group]),
        )
        pairwise_decomposition[name]["pair_weight"] = (
            pairwise_decomposition[name]["pair_count"] / total_pairs
        )
        pairwise_decomposition[name]["weighted_auc_delta"] = (
            pairwise_decomposition[name]["pair_weight"]
            * (pairwise_decomposition[name]["auc_delta"] or 0.0)
        )
    pairwise_decomposition["weighted_auc_control"] = float(sum(
        item["pair_weight"] * (item["auc_control"] or 0.0)
        for item in pairwise_decomposition.values() if isinstance(item, dict) and "pair_weight" in item
    ))
    pairwise_decomposition["weighted_auc_treatment"] = float(sum(
        item["pair_weight"] * (item["auc_treatment"] or 0.0)
        for item in pairwise_decomposition.values() if isinstance(item, dict) and "pair_weight" in item
    ))
    pairwise_decomposition["weighted_auc_delta"] = float(sum(
        item["weighted_auc_delta"]
        for item in pairwise_decomposition.values() if isinstance(item, dict) and "weighted_auc_delta" in item
    ))
    pair_weight_sum = float(sum(
        item["pair_weight"]
        for item in pairwise_decomposition.values() if isinstance(item, dict) and "pair_weight" in item
    ))
    if not np.isclose(pair_weight_sum, 1.0, atol=1e-12):
        raise RuntimeError(f"pair-stratum weights do not sum to one: {pair_weight_sum}")
    pairwise_decomposition["full_auc_reconstruction_error"] = float(
        pairwise_decomposition["weighted_auc_delta"] - rank["auroc_delta"]
    )
    if not np.isclose(pairwise_decomposition["full_auc_reconstruction_error"], 0.0, atol=1e-12):
        raise RuntimeError(
            "pair-stratum weighted AUROC delta does not reconstruct the global delta: "
            f"{pairwise_decomposition['full_auc_reconstruction_error']}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    known_indices = np.flatnonzero(known)
    unknown_indices = np.flatnonzero(~known)
    calibration_size = len(raw_cal)
    storage = {
        "auroc_control": [], "auroc_treatment": [], "auroc_delta": [],
        "aupr_control": [], "aupr_treatment": [], "aupr_delta": [],
        "threshold_control": [], "threshold_treatment": [],
    }
    scenarios = tuple(cross)
    operating_storage = {
        scenario: {key: [] for key in ("known_accept_rate", "unknown_reject_rate", "candidate_count", "candidate_purity")}
        for scenario in scenarios
    }

    for _ in range(REPLICATES):
        # Stratified paired test bootstrap preserves the observed 60/40 test
        # composition and uses identical resampled examples for both methods.
        test_indices = np.concatenate((
            rng.choice(known_indices, size=len(known_indices), replace=True),
            rng.choice(unknown_indices, size=len(unknown_indices), replace=True),
        ))
        y_boot = unknown_indicator[test_indices]
        raw_boot = raw_test[test_indices]
        unc_boot = uncertainty_test[test_indices]

        # Resample the shared known-only calibration set and refit the exact
        # score statistics and 95% known-coverage thresholds for each arm.
        cal_indices = rng.choice(calibration_size, size=calibration_size, replace=True)
        raw_cal_boot = raw_cal[cal_indices]
        unc_cal_boot = unc_cal[cal_indices]
        state_boot = boundary_state(raw_cal_boot, unc_cal_boot)
        adjusted_cal_boot = apply_correction(raw_cal_boot, unc_cal_boot, state_boot)
        t0 = state_boot[0]
        t1 = float(np.quantile(adjusted_cal_boot, TARGET_COVERAGE, method="linear"))
        adjusted_test_boot = apply_correction(raw_boot, unc_boot, state_boot)

        auc0 = float(roc_auc_score(y_boot, raw_boot))
        auc1 = float(roc_auc_score(y_boot, adjusted_test_boot))
        ap0 = float(average_precision_score(y_boot, raw_boot))
        ap1 = float(average_precision_score(y_boot, adjusted_test_boot))
        storage["auroc_control"].append(auc0)
        storage["auroc_treatment"].append(auc1)
        storage["auroc_delta"].append(auc1 - auc0)
        storage["aupr_control"].append(ap0)
        storage["aupr_treatment"].append(ap1)
        storage["aupr_delta"].append(ap1 - ap0)
        storage["threshold_control"].append(t0)
        storage["threshold_treatment"].append(t1)

        boot_known = known[test_indices]
        for scenario in scenarios:
            if scenario == "raw_score_at_control_threshold":
                scores, threshold = raw_boot, t0
            elif scenario == "adjusted_score_at_control_threshold":
                scores, threshold = adjusted_test_boot, t0
            elif scenario == "raw_score_at_treatment_threshold":
                scores, threshold = raw_boot, t1
            else:
                scores, threshold = adjusted_test_boot, t1
            values = operating_metrics(scores, threshold, boot_known)
            for key, value in values.items():
                operating_storage[scenario][key].append(float(value))

    boot_summary = {key: summarize(values) for key, values in storage.items()}
    operating_boot_summary = {
        scenario: {key: summarize(values) for key, values in metrics.items()}
        for scenario, metrics in operating_storage.items()
    }

    payload = {
        "protocol": {
            "split_id": "semantic-hard-60-40-s2131",
            "replicates": REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "test_resampling": "paired stratified bootstrap within known/unknown status",
            "calibration_resampling": "paired bootstrap of the same known-only validation calibration examples; recompute boundary state and thresholds per replicate",
            "target_known_coverage": TARGET_COVERAGE,
            "boundary_band_multiplier": BOUNDARY_MULTIPLIER,
            "alpha": ALPHA,
            "unknown_labels_used_for_calibration": False,
            "interpretation": "Intervals describe sampling/calibration uncertainty conditional on this single class split, not between-split variability.",
        },
        "reconstruction_checks": {
            "control_threshold_recomputed": threshold_control,
            "control_threshold_saved": saved_control_threshold,
            "treatment_threshold_recomputed": threshold_treatment,
            "treatment_threshold_saved": saved_treatment_threshold,
            "max_abs_adjusted_test_score_reconstruction_error": max_saved_score_error,
            "test_sample_count": int(len(known)),
            "known_test_count": int(known.sum()),
            "unknown_test_count": int((~known).sum()),
            "calibration_count": int(calibration_size),
            "calibration_boundary_count": int(np.sum(np.abs(raw_cal - threshold_control) <= BOUNDARY_MULTIPLIER * state[1])),
            "test_boundary_count": int(np.sum(np.abs(raw_test - threshold_control) <= BOUNDARY_MULTIPLIER * state[1])),
            "test_boundary_rate": float(np.mean(np.abs(raw_test - threshold_control) <= BOUNDARY_MULTIPLIER * state[1])),
            "test_adjustment_quantiles": [float(x) for x in np.quantile(corrected_test - raw_test, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])],
        },
        "rank_metrics_point_estimate": rank,
        "boundary_scope_and_pairwise_auc_decomposition": {
            "test_boundary_count": int(np.sum(test_boundary)),
            "test_boundary_rate": float(np.mean(test_boundary)),
            "known_boundary_count": known_boundary_count,
            "known_boundary_rate": float(known_boundary_count / np.sum(known)),
            "unknown_boundary_count": unknown_boundary_count,
            "unknown_boundary_rate": float(unknown_boundary_count / np.sum(unknown_indicator)),
            "total_unknown_known_pairs": total_pairs,
            "strata": pairwise_decomposition,
        },
        "threshold_cross_decomposition_point_estimate": cross,
        "bootstrap_summary": boot_summary,
        "threshold_cross_decomposition_bootstrap": operating_boot_summary,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(metric: dict, scale: float = 1.0) -> str:
        lo, hi = metric["ci95_percentile"]
        return f"{metric['median'] * scale:.4f} [{lo * scale:.4f}, {hi * scale:.4f}]"

    md = [
        "# s2131 边界不确定性配对 bootstrap 与阈值归因",
        "",
        f"- 重复次数：{REPLICATES}；随机种子：{BOOTSTRAP_SEED}。",
        "- 每次重复均在已知/未知测试样本内分层、有放回地抽样；控制与处理使用相同的抽样索引。",
        "- 同时对 725 个已知校准样本配对重抽样，并按原实现重新估计边界统计量和 95% 已知覆盖阈值。未知标签不进入校准。",
        "- 区间仅表示**该 semantic-hard 类别划分内**测试抽样与校准样本有限带来的不确定性，不代表跨类别划分波动或总体泛化区间。",
        "",
        "## 重建核验",
        "",
        f"重算控制/处理阈值分别为 `{threshold_control:.6f}` / `{threshold_treatment:.6f}`；处理组测试修正分数最大重建误差为 `{max_saved_score_error:.3g}`。测试边界带覆盖 `{payload['reconstruction_checks']['test_boundary_count']}/10000` 个样本（`{payload['reconstruction_checks']['test_boundary_rate']:.2%}`），说明该配置实际影响范围并不局限于很窄的一小撮样本。",
        "",
        "## 排序指标（bootstrap 中同时重抽校准集）",
        "",
        "区间格式为 bootstrap 中位数 [95% 百分位区间]；差值按每次配对重复计算。",
        "",
        "| 指标 | 修正关闭 | 修正开启 | 配对差值 |",
        "|---|---:|---:|---:|",
        f"| AUROC | {fmt(boot_summary['auroc_control'])} | {fmt(boot_summary['auroc_treatment'])} | {fmt(boot_summary['auroc_delta'])} |",
        f"| AUPR | {fmt(boot_summary['aupr_control'])} | {fmt(boot_summary['aupr_treatment'])} | {fmt(boot_summary['aupr_delta'])} |",
        "",
        "## 工作点归因",
        "",
        "测试指标在同一组重抽样上比较；四种组合分别为原始/调整分数与控制/处理阈值。表中为中位数 [95% 百分位区间]。候选数按每次 bootstrap 样本计数。",
        "",
        "| 分数与阈值 | 已知接受率 | 未知拒识率 | 候选数 | 候选纯度 |",
        "|---|---:|---:|---:|---:|",
    ]
    scenario_names = {
        "raw_score_at_control_threshold": "原始分数 + 控制阈值",
        "adjusted_score_at_control_threshold": "调整分数 + 控制阈值",
        "raw_score_at_treatment_threshold": "原始分数 + 处理阈值",
        "adjusted_score_at_treatment_threshold": "调整分数 + 处理阈值（完整处理）",
    }
    for key, label in scenario_names.items():
        stats = operating_boot_summary[key]
        md.append(
            f"| {label} | {fmt(stats['known_accept_rate'], 100)}% | "
            f"{fmt(stats['unknown_reject_rate'], 100)}% | "
            f"{fmt(stats['candidate_count'])} | {fmt(stats['candidate_purity'], 100)}% |"
        )
    boundary_diag = payload["boundary_scope_and_pairwise_auc_decomposition"]
    md.extend([
        "",
        "## 修正覆盖范围与 AUROC 配对分解",
        "",
        f"按控制校准集的阈值和分数标准差定义边界带后，共有 `{boundary_diag['test_boundary_count']}/10000` 个测试样本受分数修正影响（已知样本 `{boundary_diag['known_boundary_count']}/{int(np.sum(known))}`，未知样本 `{boundary_diag['unknown_boundary_count']}/{int(np.sum(unknown_indicator))}`）。因此这里的边界带覆盖面较宽，解释时不能称为只修正少量临界样本。",
        "",
        "下表把全部未知—已知样本对按双方是否位于边界带分层。每个 AUC 是该组样本对中未知分数高于已知分数的比例（平分计半）；加权差值之和应还原总体 AUROC 差值。该分解用于说明观测到的排序变化来自哪些样本对，不是新的独立验证。",
        "",
        "| 未知/已知样本对分层 | 未知数 | 已知数 | 样本对占比 | 原始 AUC | 修正 AUC | 对总体 AUROC 差值的贡献 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    stratum_names = {
        "both_in_boundary": "双方均在边界带",
        "unknown_in_known_out": "未知在带内、已知在带外",
        "unknown_out_known_in": "未知在带外、已知在带内",
        "both_outside_boundary": "双方均在带外",
    }
    for key, label in stratum_names.items():
        item = boundary_diag["strata"][key]
        md.append(
            f"| {label} | {item['unknown_count']} | {item['known_count']} | "
            f"{item['pair_weight']:.2%} | {item['auc_control']:.6f} | "
            f"{item['auc_treatment']:.6f} | {item['weighted_auc_delta']:+.6f} |"
        )
    md.extend([
        "",
        f"加权分解重建 AUROC 差值 `{boundary_diag['strata']['weighted_auc_delta']:+.6f}`；与总体直接计算值 `{rank['auroc_delta']:+.6f}` 的误差为 `{boundary_diag['strata']['full_auc_reconstruction_error']:.3g}`。",
    ])
    md.extend([
        "",
        "## 当前解释",
        "",
        "边界修正提升了全测试集的排序指标，但它与重新校准阈值是两个不同作用：若调整分数仍用控制阈值，拒识会明显增加，同时也会误拒更多已知样本；重新校准后已知接受率回到目标附近，未知拒识收益随之收窄。交叉表用于量化该权衡，而不是把固定阈值下的拒识率单独当作完整收益。",
        "",
        "因此，这组结果支持“边界不确定性会改变局部排序，并且最终工作点高度依赖匹配校准”这一机制性观察；它不证明新类别划分间稳定有效。当前不进行参数扫描，也不增加训练实验。若需继续，优先检验一个新的、事先固定配置的独立类别划分；否则本轮到此停止。",
        "",
        "## 复核输入",
        "",
        "- 控制运行：`runs/boundary_uncertainty_s2131_control/`",
        "- 处理运行：`runs/boundary_uncertainty_s2131_band1_alpha05/`",
        "- 配对复核：`analysis/boundary_uncertainty_s2131_paired_recheck.md`",
        "- 分析脚本：`scripts/audit_boundary_uncertainty_s2131_bootstrap.py`",
    ])
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({"rank_point": rank, "bootstrap": boot_summary, "cross_point": cross}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
