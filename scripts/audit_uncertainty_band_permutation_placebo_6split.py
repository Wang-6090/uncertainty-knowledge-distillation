"""Compare boundary uncertainty alignment with score-local permutation placebos.

This post-hoc audit reads six frozen paired runs. It does not train, tune,
recalibrate, or use test labels to construct permutation strata. Within the
validation-defined test boundary, validation uncertainty values are permuted
among samples in the same raw-score decile. The observed AUROC change is then
compared with the resulting placebo distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SPLIT_IDS = (2121, 2122, 2123, 2124, 2125, 2127)
TARGET_COVERAGE = 0.95
ALPHA = 0.5
BAND_MULTIPLIER = 1.0
SCORE_BINS = 10
PERMUTATIONS = 2000
RANDOM_SEED = 20261115
OUT_JSON = ROOT / "analysis" / "uncertainty_band_permutation_placebo_6split.json"
OUT_MD = ROOT / "analysis" / "uncertainty_band_permutation_placebo_6split.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(name: str, actual: float, expected: float, atol: float = 1e-8) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} mismatch: actual={actual}, expected={expected}")


def load_split(split_id: int) -> dict:
    baseline_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{split_id}_static_control_discover"
    treatment_dir = ROOT / "runs" / f"rejection_uncertainty_band_full_s{split_id}"
    paths = {
        "baseline_detail": baseline_dir / "discovery_detail.json",
        "baseline_report": baseline_dir / "calibration_report.json",
        "treatment_detail": treatment_dir / "discovery_detail.json",
        "treatment_report": treatment_dir / "calibration_report.json",
        "calibration": treatment_dir / "calibration_scores.npz",
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs:\n" + "\n".join(missing))

    baseline = read_json(paths["baseline_detail"])
    treatment = read_json(paths["treatment_detail"])
    baseline_report = read_json(paths["baseline_report"])
    treatment_report = read_json(paths["treatment_report"])
    boundary_report = treatment_report["boundary_uncertainty"]
    if not boundary_report.get("enabled"):
        raise RuntimeError(f"boundary correction is disabled for split={split_id}")
    require_close("alpha", float(boundary_report["alpha"]), ALPHA)
    require_close("band multiplier", float(boundary_report["band_multiplier"]), BAND_MULTIPLIER)

    with np.load(paths["calibration"], allow_pickle=False) as cal:
        raw_cal = np.asarray(cal["raw_score"], dtype=np.float64).reshape(-1)
        uncertainty_cal = np.asarray(cal["validation_head_uncertainty"], dtype=np.float64).reshape(-1)
        adjusted_cal_saved = np.asarray(cal["modified_score"], dtype=np.float64).reshape(-1)
        boundary_cal_saved = np.asarray(cal["boundary_mask"], dtype=bool).reshape(-1)
        threshold_cal_saved = float(np.asarray(cal["calibrated_threshold"]).reshape(-1)[0])

    threshold = float(baseline_report["calibrated_threshold"])
    require_close("baseline threshold", threshold, float(np.quantile(raw_cal, TARGET_COVERAGE, method="linear")))
    require_close("reported boundary baseline threshold", float(boundary_report["baseline_threshold"]), threshold)
    score_std = max(float(np.std(raw_cal, ddof=1)), 1e-6)
    uncertainty_mean = float(np.mean(uncertainty_cal))
    uncertainty_std = max(float(np.std(uncertainty_cal, ddof=1)), 1e-6)

    boundary_cal = np.abs(raw_cal - threshold) <= BAND_MULTIPLIER * score_std
    adjusted_cal = raw_cal.copy()
    adjusted_cal[boundary_cal] += ALPHA * score_std * (
        uncertainty_cal[boundary_cal] - uncertainty_mean
    ) / uncertainty_std
    if not np.array_equal(boundary_cal, boundary_cal_saved):
        raise RuntimeError(f"calibration boundary mask mismatch for split={split_id}")
    require_close(
        "reconstructed calibration scores",
        float(np.max(np.abs(adjusted_cal - adjusted_cal_saved))),
        0.0,
    )
    require_close(
        "reconstructed calibrated threshold",
        threshold_cal_saved,
        float(np.quantile(adjusted_cal, TARGET_COVERAGE, method="linear")),
    )

    raw = np.asarray(treatment["raw_score"], dtype=np.float64).reshape(-1)
    baseline_score = np.asarray(baseline["score"], dtype=np.float64).reshape(-1)
    adjusted_saved = np.asarray(treatment["score"], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(treatment["head_uncertainty"], dtype=np.float64).reshape(-1)
    known = np.asarray(baseline["true_known"], dtype=bool).reshape(-1)
    known_treatment = np.asarray(treatment["true_known"], dtype=bool).reshape(-1)
    if not all(
        array.shape == raw.shape
        for array in (baseline_score, adjusted_saved, uncertainty, known, known_treatment)
    ):
        raise RuntimeError(f"test arrays are not aligned for split={split_id}")
    if not np.array_equal(known, known_treatment):
        raise RuntimeError(f"known/unknown masks differ for split={split_id}")
    if not np.array_equal(raw, baseline_score):
        raise RuntimeError(f"baseline and treatment raw scores differ for split={split_id}")

    boundary = np.abs(raw - threshold) <= BAND_MULTIPLIER * score_std
    adjusted = raw.copy()
    adjusted[boundary] += ALPHA * score_std * (
        uncertainty[boundary] - uncertainty_mean
    ) / uncertainty_std
    require_close(
        "reconstructed treatment scores",
        float(np.max(np.abs(adjusted - adjusted_saved))),
        0.0,
    )
    if int(boundary.sum()) != int(boundary_report["test_boundary_count"]):
        raise RuntimeError(f"reported test boundary count differs for split={split_id}")
    if not all(np.all(np.isfinite(array)) for array in (raw, uncertainty, adjusted)):
        raise RuntimeError(f"non-finite test values for split={split_id}")

    return {
        "split_id": split_id,
        "raw": raw,
        "uncertainty": uncertainty,
        "known": known,
        "boundary": boundary,
        "score_std": score_std,
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_std": uncertainty_std,
        "observed_adjusted": adjusted_saved,
    }


def score_local_bins(
    raw_boundary: np.ndarray, bin_count: int = SCORE_BINS
) -> tuple[np.ndarray, list[np.ndarray]]:
    # Bin edges depend only on raw scores, never on test labels or uncertainty.
    edges = np.quantile(raw_boundary, np.linspace(0.0, 1.0, bin_count + 1))
    bin_ids = np.searchsorted(edges[1:-1], raw_boundary, side="right")
    groups = [np.flatnonzero(bin_ids == idx) for idx in range(bin_count)]
    groups = [group for group in groups if group.size]
    if len(groups) < 2:
        raise RuntimeError("too few distinct raw-score bins for a local permutation")
    return edges, groups


def evaluate_split(data: dict, score_bin_count: int = SCORE_BINS) -> dict:
    split_id = data["split_id"]
    raw = data["raw"]
    uncertainty = data["uncertainty"]
    boundary = data["boundary"]
    labels = (~data["known"]).astype(np.int8)

    auc_raw = float(roc_auc_score(labels, raw))
    auc_observed = float(roc_auc_score(labels, data["observed_adjusted"]))
    observed_delta = auc_observed - auc_raw

    boundary_indices = np.flatnonzero(boundary)
    raw_boundary = raw[boundary]
    uncertainty_boundary = uncertainty[boundary]
    standardized_boundary_mean = float(
        np.mean((uncertainty_boundary - data["uncertainty_mean"]) / data["uncertainty_std"])
    )
    constant_boundary_shift = ALPHA * data["score_std"] * standardized_boundary_mean
    constant_adjusted = raw.copy()
    constant_adjusted[boundary] += constant_boundary_shift
    constant_shift_delta = float(roc_auc_score(labels, constant_adjusted)) - auc_raw
    edges, groups = score_local_bins(raw_boundary, score_bin_count)
    rng = np.random.default_rng(RANDOM_SEED + split_id)
    placebo_deltas = np.empty(PERMUTATIONS, dtype=np.float64)

    for permutation_idx in range(PERMUTATIONS):
        placebo_uncertainty = uncertainty_boundary.copy()
        for group in groups:
            if group.size > 1:
                placebo_uncertainty[group] = uncertainty_boundary[rng.permutation(group)]
        placebo_score = raw.copy()
        placebo_score[boundary_indices] += ALPHA * data["score_std"] * (
            placebo_uncertainty - data["uncertainty_mean"]
        ) / data["uncertainty_std"]
        placebo_deltas[permutation_idx] = float(roc_auc_score(labels, placebo_score)) - auc_raw

    return {
        "split_id": split_id,
        "auc_raw": auc_raw,
        "auc_observed_adjusted": auc_observed,
        "observed_auc_delta": observed_delta,
        "standardized_boundary_uncertainty_mean": standardized_boundary_mean,
        "constant_boundary_score_shift": float(constant_boundary_shift),
        "constant_shift_auc_delta": constant_shift_delta,
        "boundary_sample_count": int(boundary.sum()),
        "boundary_unknown_rate": float(np.mean(boundary[~data["known"]])),
        "boundary_known_rate": float(np.mean(boundary[data["known"]])),
        "requested_score_bin_count": int(score_bin_count),
        "score_bin_count": len(groups),
        "score_bin_sizes": [int(group.size) for group in groups],
        "score_bin_edge_min": float(edges[0]),
        "score_bin_edge_max": float(edges[-1]),
        "permutation_count": PERMUTATIONS,
        "placebo_delta_mean": float(np.mean(placebo_deltas)),
        "placebo_delta_std": float(np.std(placebo_deltas, ddof=1)),
        "placebo_delta_quantiles": {
            "2.5%": float(np.quantile(placebo_deltas, 0.025)),
            "50%": float(np.quantile(placebo_deltas, 0.5)),
            "97.5%": float(np.quantile(placebo_deltas, 0.975)),
        },
        "placebo_count_at_least_observed": int(np.sum(placebo_deltas >= observed_delta)),
        "placebo_max_delta": float(np.max(placebo_deltas)),
        "observed_minus_placebo_max": float(observed_delta - np.max(placebo_deltas)),
        "interpretation": "exploratory placebo comparison; not a confirmatory significance test",
        "test_labels_used_for_permutation_or_bins": False,
        "test_labels_used_only_for_auc_evaluation": True,
    }


def make_markdown(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# 边界不确定性与分数局部置换对照（六组冻结结果）",
        "",
        "## 问题与做法",
        "",
        "前一项样本对分解显示 AUROC 变化主要落在修正边界带内，但这本身不能说明不确定性数值与具体样本的对应关系确实提供了信息。本分析在不重新训练的情况下构造 placebo 对照：每组在已知校准数据确定的边界带内，按测试样本原始拒识分数划分为十个等频区间；仅在同一区间内随机打乱不确定性分数与样本的对应关系，再按原有固定系数重算 AUROC。区间仅由原始分数构造，不使用测试标签。",
        "",
        f"每组进行 `{PERMUTATIONS}` 次置换。保持模型原始分数、已知/未知标签、边界带、修正系数及验证集不确定性标准化统计量不变。测试标签只用于计算观测 AUROC 和置换 placebo AUROC。下表报告置换中达到或超过实际变化的次数，以及置换最大变化与实际变化的距离；这些是本次有限置换的描述性结果，不是显著性检验。该分析为事后机制诊断，不用于选择设置。",
        "",
        "## 汇总",
        "",
        f"- 六组真实不确定性对应关系下的 AUROC 平均变化：`{summary['mean_observed_delta']:+.6f}`；六组中正向 `{summary['positive_observed_splits']}/{summary['split_count']}` 组。",
        f"- 仅给边界带施加与真实不确定性修正相同的平均分数偏移时，AUROC 平均变化为 `{summary['mean_constant_shift_delta']:+.6f}`；六组中非负 `{summary['nonnegative_constant_shift_splits']}/{summary['split_count']}` 组。",
        f"- 六组 placebo 平均 AUROC 变化：`{summary['mean_placebo_delta']:+.6f}`；各组 placebo 均值的范围为 `{summary['min_placebo_mean']:+.6f}` 至 `{summary['max_placebo_mean']:+.6f}`。",
        f"- 真实变化高于各自 placebo 最大值的组数：`{summary['observed_above_placebo_max_splits']}/{summary['split_count']}`。",
        f"- 边界带平均覆盖未知样本 `{summary['mean_boundary_unknown_rate']:.1%}`、已知样本 `{summary['mean_boundary_known_rate']:.1%}`；该带较宽，不宜描述成只涉及少量临界样本。",
        "",
        "## 逐组结果",
        "",
        "| 划分 | 原始 AUROC | 实际修正 AUROC | 实际变化 | 平均偏移单独变化 | placebo 均值 | placebo 95% 区间 | placebo 最大变化 | 实际值高出最大值 | 达到/超过实际值次数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        quantiles = row["placebo_delta_quantiles"]
        lines.append(
            f"| {row['split_id']} | {row['auc_raw']:.6f} | {row['auc_observed_adjusted']:.6f} | "
            f"{row['observed_auc_delta']:+.6f} | {row['constant_shift_auc_delta']:+.6f} | "
            f"{row['placebo_delta_mean']:+.6f} | "
            f"[{quantiles['2.5%']:+.6f}, {quantiles['97.5%']:+.6f}] | "
            f"{row['placebo_max_delta']:+.6f} | {row['observed_minus_placebo_max']:+.6f} | "
            f"{row['placebo_count_at_least_observed']}/{row['permutation_count']} |"
        )
    lines += [
        "",
        "这里报告置换次数而非换算成 p 值：六组实际变化如果都高于本次 2,000 个 placebo 的最大值，只能说明在这批有限、事后构造的置换中没有出现达到实际变化的 placebo；这不构成统计显著性结论或独立验证。",
        "",
        "## 解释边界",
        "",
        "平均偏移对照把所有边界样本都加上同一个分数增量，只保留整体平移效应，不保留样本间不确定性差异。若它本身不能重现实际变化，而真实对应关系高于局部置换 placebo，可作为‘样本级不确定性对应关系可能提供额外排序信息’的机制线索。由于该分析事后开展且只使用六组既有结果，不能证明因果、独立复现、跨数据集泛化或完整新类发现收益；十等频区间的选择也限制了结论范围。",
        "",
        "## 分箱数量敏感性",
        "",
        "为检查结果是否只依赖十等频分箱，另以 5、10、20 箱重复同一置换诊断。三个设定使用相同冻结数据和 2,000 次置换；此处只比较方向一致性，不解释为显著性检验。",
        "",
        "| 请求分箱数 | 实际箱数（按六组顺序） | 单箱样本数范围 | 实际变化高于 placebo 最大值 | 平均 placebo AUROC 变化 | 各划分 placebo 均值范围 | 达到/超过实际变化次数总和 |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for item in result["bin_count_sensitivity"]:
        effective_bins = "/".join(
            str(row["score_bin_count"]) for row in item["split_details"]
        )
        bin_size_min = min(row["min_score_bin_size"] for row in item["split_details"])
        bin_size_max = max(row["max_score_bin_size"] for row in item["split_details"])
        lines.append(
            f"| {item['requested_score_bins']} | "
            f"{effective_bins} | [{bin_size_min}, {bin_size_max}] | "
            f"{item['observed_above_placebo_max_splits']}/{summary['split_count']} | "
            f"{item['mean_placebo_delta']:+.6f} | "
            f"[{item['min_split_placebo_mean']:+.6f}, {item['max_split_placebo_mean']:+.6f}] | "
            f"{item['placebo_count_at_least_observed_total']}/{item['permutation_count'] * summary['split_count']} |"
        )
    lines += [
        "",
        "分箱数敏感性仍属于对既有结果的事后诊断；它只能说明本次有限设定下的描述性稳定程度，不能替代预先规定的独立验证。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts/audit_uncertainty_band_permutation_placebo_6split.py",
        "```",
        "",
        "随机种子固定为 `20261115 + 划分编号`；输出结果保存至同目录下的 JSON 文件。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    split_data = {split_id: load_split(split_id) for split_id in SPLIT_IDS}
    rows = [evaluate_split(split_data[split_id]) for split_id in SPLIT_IDS]
    deltas = [row["observed_auc_delta"] for row in rows]
    placebo_means = [row["placebo_delta_mean"] for row in rows]
    constant_shift_deltas = [row["constant_shift_auc_delta"] for row in rows]
    bin_count_sensitivity = []
    for bin_count in (5, SCORE_BINS, 20):
        sensitivity_rows = (
            rows
            if bin_count == SCORE_BINS
            else [
                evaluate_split(split_data[split_id], score_bin_count=bin_count)
                for split_id in SPLIT_IDS
            ]
        )
        split_placebo_means = [row["placebo_delta_mean"] for row in sensitivity_rows]
        bin_count_sensitivity.append(
            {
                "requested_score_bins": bin_count,
                "permutation_count": PERMUTATIONS,
                "mean_placebo_delta": float(np.mean(split_placebo_means)),
                "min_split_placebo_mean": float(min(split_placebo_means)),
                "max_split_placebo_mean": float(max(split_placebo_means)),
                "observed_above_placebo_max_splits": int(
                    sum(row["observed_auc_delta"] > row["placebo_max_delta"] for row in sensitivity_rows)
                ),
                "placebo_count_at_least_observed_total": int(
                    sum(row["placebo_count_at_least_observed"] for row in sensitivity_rows)
                ),
                "split_details": [
                    {
                        "split_id": row["split_id"],
                        "score_bin_count": row["score_bin_count"],
                        "min_score_bin_size": min(row["score_bin_sizes"]),
                        "max_score_bin_size": max(row["score_bin_sizes"]),
                    }
                    for row in sensitivity_rows
                ],
            }
        )
    result = {
        "question": "Does sample-aligned uncertainty outperform score-local permutation placebos?",
        "protocol": "post-hoc diagnostic on frozen scores; no training, tuning, or threshold changes",
        "split_ids": list(SPLIT_IDS),
        "settings": {
            "target_known_coverage": TARGET_COVERAGE,
            "alpha": ALPHA,
            "boundary_multiplier": BAND_MULTIPLIER,
            "score_quantile_bins": SCORE_BINS,
            "permutations_per_split": PERMUTATIONS,
            "random_seed_rule": "20261115 + split identifier",
            "test_labels_used_for_permutation_or_bins": False,
            "test_labels_used_only_for_auc_evaluation": True,
        },
        "rows": rows,
        "bin_count_sensitivity": bin_count_sensitivity,
        "summary": {
            "split_count": len(rows),
            "mean_observed_delta": float(np.mean(deltas)),
            "positive_observed_splits": int(sum(delta > 0 for delta in deltas)),
            "mean_constant_shift_delta": float(np.mean(constant_shift_deltas)),
            "nonnegative_constant_shift_splits": int(sum(delta >= 0 for delta in constant_shift_deltas)),
            "mean_boundary_unknown_rate": float(np.mean([row["boundary_unknown_rate"] for row in rows])),
            "mean_boundary_known_rate": float(np.mean([row["boundary_known_rate"] for row in rows])),
            "mean_placebo_delta": float(np.mean(placebo_means)),
            "min_placebo_mean": float(min(placebo_means)),
            "max_placebo_mean": float(max(placebo_means)),
            "observed_above_placebo_max_splits": int(
                sum(row["observed_auc_delta"] > row["placebo_max_delta"] for row in rows)
            ),
            "interpretation": "descriptive and exploratory; no independent confirmation or significance claim",
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
