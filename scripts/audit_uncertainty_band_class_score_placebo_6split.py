"""Class- and raw-score-conditional permutation placebo on frozen runs.

This is a post-hoc mechanism diagnostic. It reuses the six frozen paired runs
from ``audit_uncertainty_band_permutation_placebo_6split.py`` and permutes
uncertainty only within strata formed by predicted known class and raw-score
quantile bin. Test labels are used only to calculate AUROC, never to form bins
or permutation strata. No model, threshold, or hyperparameter is changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import audit_uncertainty_band_permutation_placebo_6split as previous


ROOT = Path(__file__).resolve().parents[1]
SPLIT_IDS = previous.SPLIT_IDS
BIN_COUNTS = (5, 10)
PERMUTATIONS = 2000
RANDOM_SEED = 20261116
OUT_JSON = ROOT / "analysis" / "uncertainty_band_class_score_placebo_6split.json"
OUT_MD = ROOT / "analysis" / "uncertainty_band_class_score_placebo_6split.md"


def load_predicted_classes(split_id: int, expected_size: int) -> np.ndarray:
    baseline_path = (
        ROOT
        / "runs"
        / f"rejection_ce_decay_full_s{split_id}_static_control_discover"
        / "discovery_detail.json"
    )
    treatment_path = (
        ROOT
        / "runs"
        / f"rejection_uncertainty_band_full_s{split_id}"
        / "discovery_detail.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
    pred_baseline = np.asarray(baseline["pred_class"], dtype=np.int64).reshape(-1)
    pred_treatment = np.asarray(treatment["pred_class"], dtype=np.int64).reshape(-1)
    true_label_baseline = np.asarray(baseline["true_label"], dtype=np.int64).reshape(-1)
    true_label_treatment = np.asarray(treatment["true_label"], dtype=np.int64).reshape(-1)
    if not (
        pred_baseline.size
        == pred_treatment.size
        == true_label_baseline.size
        == true_label_treatment.size
        == expected_size
    ):
        raise RuntimeError(f"class/label arrays are not aligned for split={split_id}")
    if not np.array_equal(pred_baseline, pred_treatment):
        raise RuntimeError(f"predicted classes differ between arms for split={split_id}")
    if not np.array_equal(true_label_baseline, true_label_treatment):
        raise RuntimeError(f"true labels or sample ordering differ for split={split_id}")
    return pred_baseline


def evaluate_split(data: dict, bin_count: int) -> dict:
    split_id = int(data["split_id"])
    raw = np.asarray(data["raw"], dtype=np.float64)
    uncertainty = np.asarray(data["uncertainty"], dtype=np.float64)
    known = np.asarray(data["known"], dtype=bool)
    boundary = np.asarray(data["boundary"], dtype=bool)
    predicted_class = load_predicted_classes(split_id, raw.size)
    if not all(
        values.shape == raw.shape
        for values in (uncertainty, known, boundary, predicted_class)
    ):
        raise RuntimeError(f"per-sample arrays are not aligned for split={split_id}")

    labels = (~known).astype(np.int8)
    raw_auc = float(roc_auc_score(labels, raw))
    observed_auc = float(roc_auc_score(labels, data["observed_adjusted"]))
    observed_delta = observed_auc - raw_auc

    boundary_indices = np.flatnonzero(boundary)
    raw_boundary = raw[boundary]
    class_boundary = predicted_class[boundary]
    uncertainty_boundary = uncertainty[boundary]
    edges = np.quantile(raw_boundary, np.linspace(0.0, 1.0, bin_count + 1))
    score_bin = np.searchsorted(edges[1:-1], raw_boundary, side="right")
    keys = np.stack((class_boundary, score_bin), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    strata = [np.flatnonzero(inverse == stratum_id) for stratum_id in range(counts.size)]

    rng = np.random.default_rng(RANDOM_SEED + split_id * 100 + bin_count)
    placebo_deltas = np.empty(PERMUTATIONS, dtype=np.float64)
    for permutation_idx in range(PERMUTATIONS):
        placebo_uncertainty = uncertainty_boundary.copy()
        for group in strata:
            if group.size > 1:
                placebo_uncertainty[group] = uncertainty_boundary[rng.permutation(group)]
        placebo_score = raw.copy()
        placebo_score[boundary_indices] += previous.ALPHA * data["score_std"] * (
            placebo_uncertainty - data["uncertainty_mean"]
        ) / data["uncertainty_std"]
        placebo_deltas[permutation_idx] = (
            float(roc_auc_score(labels, placebo_score)) - raw_auc
        )

    placebo_path = (
        ROOT / "analysis" / "uncertainty_band_permutation_placebo_6split.json"
    )
    previous_result = json.loads(placebo_path.read_text(encoding="utf-8"))
    previous_row = next(
        row for row in previous_result["rows"] if int(row["split_id"]) == split_id
    )
    return {
        "split_id": split_id,
        "requested_score_bins": bin_count,
        "auc_raw": raw_auc,
        "auc_observed_adjusted": observed_auc,
        "observed_auc_delta": observed_delta,
        "predicted_class_count": int(np.unique(predicted_class).size),
        "boundary_sample_count": int(boundary.sum()),
        "stratum_count": int(counts.size),
        "singleton_stratum_count": int(np.sum(counts == 1)),
        "singleton_sample_fraction": float(np.sum(counts[counts == 1]) / boundary.sum()),
        "strata_smaller_than_three": int(np.sum(counts < 3)),
        "stratum_size_min": int(counts.min()),
        "stratum_size_median": float(np.median(counts)),
        "stratum_size_max": int(counts.max()),
        "permutation_count": PERMUTATIONS,
        "placebo_delta_mean": float(np.mean(placebo_deltas)),
        "placebo_delta_quantiles": {
            "2.5%": float(np.quantile(placebo_deltas, 0.025)),
            "50%": float(np.quantile(placebo_deltas, 0.5)),
            "97.5%": float(np.quantile(placebo_deltas, 0.975)),
        },
        "placebo_max_delta": float(np.max(placebo_deltas)),
        "placebo_count_at_least_observed": int(
            np.sum(placebo_deltas >= observed_delta)
        ),
        "observed_above_placebo_max": bool(observed_delta > np.max(placebo_deltas)),
        "previous_score_only_placebo_mean": float(previous_row["placebo_delta_mean"]),
        "previous_score_only_placebo_max": float(previous_row["placebo_max_delta"]),
        "previous_score_only_exceedances": int(
            previous_row["placebo_count_at_least_observed"]
        ),
        "test_labels_used_for_strata": False,
        "test_labels_used_only_for_auc": True,
    }


def make_markdown(result: dict) -> str:
    lines = [
        "# 边界不确定性：预测类别与分数条件化置换诊断（六组冻结结果）",
        "",
        "## 目的与方案",
        "",
        "此前的局部置换只在原始拒识分数区间内打乱不确定性，没有控制样本的模型预测类别。本分析进一步检查：AUROC 的观测变化是否仍高于同时匹配预测类别和相近原始分数的置换对照。分层使用模型预测类别与边界内原始拒识分数的等频区间；测试标签不参与分箱或置换，只用于计算 AUROC。",
        "",
        f"沿用六组已有配对结果、边界定义、修正系数和验证集标准化统计量；每组每种分箱设置进行 `{PERMUTATIONS}` 次置换，不训练模型、不调阈值。该事后分析仅是机制诊断，不是显著性检验或独立复现。",
        "",
        "## 结果",
        "",
        "| 划分 | 分箱数 | 实际 AUROC 变化 | 条件置换均值 | 条件置换 95% 区间 | 最大置换变化 | 超过观测值次数 | 实际值高于最大值 | 单例样本比例 |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in result["rows"]:
        q = row["placebo_delta_quantiles"]
        lines.append(
            f"| {row['split_id']} | {row['requested_score_bins']} | "
            f"{row['observed_auc_delta']:+.6f} | {row['placebo_delta_mean']:+.6f} | "
            f"[{q['2.5%']:+.6f}, {q['97.5%']:+.6f}] | "
            f"{row['placebo_max_delta']:+.6f} | "
            f"{row['placebo_count_at_least_observed']}/{row['permutation_count']} | "
            f"{'是' if row['observed_above_placebo_max'] else '否'} | "
            f"{row['singleton_sample_fraction']:.3%} |"
        )
    lines += [
        "",
        "## 解读边界",
        "",
        "与仅按分数分层的置换相比，本分析额外固定了模型预测类别，可排查粗粒度类别组成是否足以解释观测到的排序变化。若观测值仍高于有限置换最大值，只能说明在这些事后构造的条件置换中未出现同等变化；不能推出统计显著、因果关系或跨类别划分稳定泛化。类别预测本身也由模型产生，分箱粒度会影响 placebo 分布，因此结果只用于决定是否值得做预先固定方案的独立验证，不据此调参或修改默认配置。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts/audit_uncertainty_band_class_score_placebo_6split.py",
        "```",
        "",
        f"随机种子规则：`{RANDOM_SEED} + 划分编号 × 100 + 分箱数`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    frozen = {split_id: previous.load_split(split_id) for split_id in SPLIT_IDS}
    rows = [
        evaluate_split(frozen[split_id], bins)
        for bins in BIN_COUNTS
        for split_id in SPLIT_IDS
    ]
    summaries = []
    for bins in BIN_COUNTS:
        selected = [row for row in rows if row["requested_score_bins"] == bins]
        summaries.append(
            {
                "score_bins": bins,
                "mean_observed_delta": float(
                    np.mean([row["observed_auc_delta"] for row in selected])
                ),
                "mean_conditional_placebo_delta": float(
                    np.mean([row["placebo_delta_mean"] for row in selected])
                ),
                "observed_above_placebo_max_splits": int(
                    sum(row["observed_above_placebo_max"] for row in selected)
                ),
                "total_exceedances": int(
                    sum(row["placebo_count_at_least_observed"] for row in selected)
                ),
                "max_singleton_sample_fraction": float(
                    max(row["singleton_sample_fraction"] for row in selected)
                ),
            }
        )
    result = {
        "question": "Does the observed AUROC change exceed placebo permutations conditional on predicted class and raw-score bin?",
        "protocol": "post-hoc diagnostic on six frozen paired runs; no training or tuning",
        "split_ids": list(SPLIT_IDS),
        "bin_counts": list(BIN_COUNTS),
        "permutations_per_split_per_setting": PERMUTATIONS,
        "random_seed_rule": f"{RANDOM_SEED} + split_id * 100 + bin_count",
        "test_labels_used_for_strata": False,
        "test_labels_used_only_for_auc": True,
        "rows": rows,
        "summaries": summaries,
        "interpretation": "exploratory conditional placebo diagnostic; not a significance test or independent replication",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUT_MD.write_text(make_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {"json": str(OUT_JSON), "markdown": str(OUT_MD), "summaries": summaries},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
