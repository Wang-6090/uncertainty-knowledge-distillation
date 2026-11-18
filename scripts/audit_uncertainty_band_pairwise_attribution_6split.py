"""Attribute frozen uncertainty-band AUROC changes to boundary strata.

This is a post-hoc analysis of six already completed, paired runs. It does not
train models, tune settings, or use test labels for calibration or selection.
For each split it decomposes the AUROC change into unknown-known pair strata
based on whether each sample's raw score falls in the validation-defined band.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2121, 2122, 2123, 2124, 2125, 2127)
TARGET_COVERAGE = 0.95
ALPHA = 0.5
BAND_MULTIPLIER = 1.0
OUT_JSON = ROOT / "analysis" / "uncertainty_band_pairwise_attribution_6split.json"
OUT_MD = ROOT / "analysis" / "uncertainty_band_pairwise_attribution_6split.md"

STRATA = (
    ("both_in_boundary", "未知与已知均在边界带内"),
    ("unknown_in_known_out", "未知在带内、已知在带外"),
    ("unknown_out_known_in", "未知在带外、已知在带内"),
    ("both_outside_boundary", "未知与已知均在边界带外"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(name: str, actual: float, expected: float, atol: float = 1e-8) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} mismatch: actual={actual}, expected={expected}")


def auc_component(
    unknown_control: np.ndarray,
    unknown_treatment: np.ndarray,
    known_control: np.ndarray,
    known_treatment: np.ndarray,
) -> dict:
    if unknown_control.shape != unknown_treatment.shape:
        raise RuntimeError("unknown control/treatment samples are not aligned")
    if known_control.shape != known_treatment.shape:
        raise RuntimeError("known control/treatment samples are not aligned")
    n_unknown = int(unknown_control.size)
    n_known = int(known_control.size)
    pair_count = n_unknown * n_known
    if not pair_count:
        raise RuntimeError("an AUROC stratum contains no unknown-known pairs")
    labels = np.concatenate((np.ones(n_unknown, dtype=np.int8), np.zeros(n_known, dtype=np.int8)))
    control = np.concatenate((unknown_control, known_control))
    treatment = np.concatenate((unknown_treatment, known_treatment))
    auc_control = float(roc_auc_score(labels, control))
    auc_treatment = float(roc_auc_score(labels, treatment))
    return {
        "unknown_count": n_unknown,
        "known_count": n_known,
        "pair_count": pair_count,
        "auc_control": auc_control,
        "auc_treatment": auc_treatment,
        "auc_delta": auc_treatment - auc_control,
    }


def evaluate_seed(seed: int) -> dict:
    baseline_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control_discover"
    treatment_dir = ROOT / "runs" / f"rejection_uncertainty_band_full_s{seed}"
    baseline_detail_path = baseline_dir / "discovery_detail.json"
    baseline_report_path = baseline_dir / "calibration_report.json"
    treatment_detail_path = treatment_dir / "discovery_detail.json"
    treatment_report_path = treatment_dir / "calibration_report.json"
    treatment_calibration_path = treatment_dir / "calibration_scores.npz"
    required = (
        baseline_detail_path,
        baseline_report_path,
        treatment_detail_path,
        treatment_report_path,
        treatment_calibration_path,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs:\n" + "\n".join(missing))

    baseline_detail = read_json(baseline_detail_path)
    treatment_detail = read_json(treatment_detail_path)
    baseline_report = read_json(baseline_report_path)
    treatment_report = read_json(treatment_report_path)
    boundary_report = treatment_report["boundary_uncertainty"]
    if not boundary_report.get("enabled"):
        raise RuntimeError(f"boundary correction is not enabled for seed={seed}")
    require_close("alpha", float(boundary_report["alpha"]), ALPHA)
    require_close("band_multiplier", float(boundary_report["band_multiplier"]), BAND_MULTIPLIER)

    with np.load(treatment_calibration_path, allow_pickle=False) as cal:
        raw_cal = np.asarray(cal["raw_score"], dtype=np.float64).reshape(-1)
        uncertainty_cal = np.asarray(cal["validation_head_uncertainty"], dtype=np.float64).reshape(-1)
        saved_adjusted_cal = np.asarray(cal["modified_score"], dtype=np.float64).reshape(-1)
        saved_boundary_cal = np.asarray(cal["boundary_mask"], dtype=bool).reshape(-1)
        saved_threshold = float(np.asarray(cal["calibrated_threshold"]).reshape(-1)[0])
        calibration_source_index = np.asarray(cal["source_index"]).reshape(-1)

    if not (raw_cal.shape == uncertainty_cal.shape == saved_adjusted_cal.shape == saved_boundary_cal.shape):
        raise RuntimeError(f"calibration arrays are not aligned for seed={seed}")
    if calibration_source_index.shape != raw_cal.shape:
        raise RuntimeError(f"calibration source indices are not aligned for seed={seed}")
    if not all(np.all(np.isfinite(array)) for array in (raw_cal, uncertainty_cal, saved_adjusted_cal)):
        raise RuntimeError(f"non-finite calibration values for seed={seed}")

    control_threshold = float(baseline_report["calibrated_threshold"])
    threshold_from_calibration = float(np.quantile(raw_cal, TARGET_COVERAGE, method="linear"))
    require_close("baseline threshold vs known calibration quantile", control_threshold, threshold_from_calibration)
    require_close("reported boundary baseline threshold", float(boundary_report["baseline_threshold"]), control_threshold)

    score_std = max(float(np.std(raw_cal, ddof=1)), 1e-6)
    uncertainty_mean = float(np.mean(uncertainty_cal))
    uncertainty_std = max(float(np.std(uncertainty_cal, ddof=1)), 1e-6)
    cal_boundary = np.abs(raw_cal - control_threshold) <= BAND_MULTIPLIER * score_std
    adjusted_cal = raw_cal.copy()
    adjusted_cal[cal_boundary] += (
        ALPHA * score_std * (uncertainty_cal[cal_boundary] - uncertainty_mean) / uncertainty_std
    )
    if not np.array_equal(cal_boundary, saved_boundary_cal):
        raise RuntimeError(f"saved calibration boundary mask differs for seed={seed}")
    require_close("reconstructed adjusted calibration scores", float(np.max(np.abs(adjusted_cal - saved_adjusted_cal))), 0.0, atol=1e-8)
    treatment_threshold = float(np.quantile(adjusted_cal, TARGET_COVERAGE, method="linear"))
    require_close("treatment calibration threshold", saved_threshold, treatment_threshold)

    baseline_scores = np.asarray(baseline_detail["score"], dtype=np.float64).reshape(-1)
    raw_test = np.asarray(treatment_detail["raw_score"], dtype=np.float64).reshape(-1)
    adjusted_test_saved = np.asarray(treatment_detail["score"], dtype=np.float64).reshape(-1)
    known = np.asarray(baseline_detail["true_known"], dtype=bool).reshape(-1)
    treatment_known = np.asarray(treatment_detail["true_known"], dtype=bool).reshape(-1)
    uncertainty_test = np.asarray(treatment_detail["head_uncertainty"], dtype=np.float64).reshape(-1)
    if not (baseline_scores.shape == raw_test.shape == adjusted_test_saved.shape == known.shape == treatment_known.shape == uncertainty_test.shape):
        raise RuntimeError(f"test arrays are not aligned for seed={seed}")
    if not np.array_equal(known, treatment_known):
        raise RuntimeError(f"test known/unknown labels are not aligned for seed={seed}")
    if not np.array_equal(baseline_scores, raw_test):
        raise RuntimeError(f"baseline score and treatment raw score differ for seed={seed}")
    if not all(np.all(np.isfinite(array)) for array in (baseline_scores, raw_test, adjusted_test_saved, uncertainty_test)):
        raise RuntimeError(f"non-finite test values for seed={seed}")

    test_boundary = np.abs(raw_test - control_threshold) <= BAND_MULTIPLIER * score_std
    adjusted_test = raw_test.copy()
    adjusted_test[test_boundary] += (
        ALPHA * score_std * (uncertainty_test[test_boundary] - uncertainty_mean) / uncertainty_std
    )
    require_close("reconstructed adjusted test scores", float(np.max(np.abs(adjusted_test - adjusted_test_saved))), 0.0, atol=1e-8)
    if int(test_boundary.sum()) != int(boundary_report["test_boundary_count"]):
        raise RuntimeError(f"reported test boundary count differs for seed={seed}")

    unknown = ~known
    labels = unknown.astype(np.int8)
    auc_control = float(roc_auc_score(labels, baseline_scores))
    auc_treatment = float(roc_auc_score(labels, adjusted_test))
    auc_delta = auc_treatment - auc_control
    strata_masks = {
        "both_in_boundary": (unknown & test_boundary, known & test_boundary),
        "unknown_in_known_out": (unknown & test_boundary, known & ~test_boundary),
        "unknown_out_known_in": (unknown & ~test_boundary, known & test_boundary),
        "both_outside_boundary": (unknown & ~test_boundary, known & ~test_boundary),
    }
    total_pairs = int(unknown.sum() * known.sum())
    strata = {}
    for name, (unknown_group, known_group) in strata_masks.items():
        result = auc_component(
            baseline_scores[unknown_group], adjusted_test[unknown_group],
            baseline_scores[known_group], adjusted_test[known_group],
        )
        result["pair_weight"] = result["pair_count"] / total_pairs
        result["weighted_auc_delta"] = result["pair_weight"] * result["auc_delta"]
        strata[name] = result

    pair_weight_sum = float(sum(row["pair_weight"] for row in strata.values()))
    reconstructed_delta = float(sum(row["weighted_auc_delta"] for row in strata.values()))
    reconstruction_error = reconstructed_delta - auc_delta
    require_close("stratum pair weights", pair_weight_sum, 1.0, atol=1e-12)
    require_close("weighted AUROC delta reconstruction", reconstruction_error, 0.0, atol=1e-12)

    return {
        "split": seed,
        "auc_control": auc_control,
        "auc_treatment": auc_treatment,
        "auc_delta": auc_delta,
        "test_sample_count": int(len(known)),
        "known_count": int(known.sum()),
        "unknown_count": int(unknown.sum()),
        "boundary": {
            "control_threshold": control_threshold,
            "treatment_threshold": treatment_threshold,
            "score_std": score_std,
            "unknown_in_boundary_count": int(np.sum(unknown & test_boundary)),
            "unknown_boundary_rate": float(np.mean(test_boundary[unknown])),
            "known_in_boundary_count": int(np.sum(known & test_boundary)),
            "known_boundary_rate": float(np.mean(test_boundary[known])),
            "test_boundary_count": int(test_boundary.sum()),
            "test_boundary_rate": float(test_boundary.mean()),
        },
        "strata": strata,
        "pair_weight_sum": pair_weight_sum,
        "reconstructed_auc_delta": reconstructed_delta,
        "reconstruction_error": reconstruction_error,
        "input_audit": {
            "test_known_masks_match": True,
            "baseline_score_equals_treatment_raw_score": True,
            "calibration_mask_reconstructed": True,
            "treatment_calibration_threshold_reconstructed": True,
            "treatment_test_scores_reconstructed": True,
            "test_labels_used_for_selection": False,
            "test_labels_used_only_for_final_attribution": True,
            "baseline_detail": str(baseline_detail_path.relative_to(ROOT)),
            "treatment_detail": str(treatment_detail_path.relative_to(ROOT)),
            "treatment_calibration": str(treatment_calibration_path.relative_to(ROOT)),
        },
    }


def summarize(rows: list[dict]) -> dict:
    summary = {
        "split_count": len(rows),
        "mean_auc_control": float(np.mean([row["auc_control"] for row in rows])),
        "mean_auc_treatment": float(np.mean([row["auc_treatment"] for row in rows])),
        "mean_auc_delta": float(np.mean([row["auc_delta"] for row in rows])),
        "splits_with_positive_auc_delta": int(sum(row["auc_delta"] > 0 for row in rows)),
        "mean_unknown_boundary_rate": float(np.mean([row["boundary"]["unknown_boundary_rate"] for row in rows])),
        "mean_known_boundary_rate": float(np.mean([row["boundary"]["known_boundary_rate"] for row in rows])),
        "strata": {},
        "max_abs_reconstruction_error": float(max(abs(row["reconstruction_error"]) for row in rows)),
    }
    for name, _label in STRATA:
        values = [row["strata"][name] for row in rows]
        summary["strata"][name] = {
            "mean_pair_weight": float(np.mean([value["pair_weight"] for value in values])),
            "mean_auc_delta_within_stratum": float(np.mean([value["auc_delta"] for value in values])),
            "mean_weighted_auc_delta_contribution": float(np.mean([value["weighted_auc_delta"] for value in values])),
            "positive_within_stratum_delta_splits": int(sum(value["auc_delta"] > 0 for value in values)),
            "positive_weighted_contribution_splits": int(sum(value["weighted_auc_delta"] > 0 for value in values)),
        }
    return summary


def make_markdown(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# 六组冻结结果的边界样本对归因",
        "",
        "## 分析问题与口径",
        "",
        "本分析只读取已完成的六组配对结果，不训练模型、不选择参数，也不改动阈值。每组先用已知校准样本计算原始拒识分数的 95% 分位阈值和标准差，再按固定边界倍数 1.0、修正系数 0.5 重建不确定性修正分数。边界带由校准数据确定；测试标签仅用于最终计算各分层 AUROC。",
        "",
        "将所有未知—已知测试样本对按双方的原始分数是否位于边界带分成四层。每层 AUROC 差值乘以该层样本对占比，四层贡献之和应精确还原整组 AUROC 差值。六组汇总为描述性平均，不将六个类别划分当作 bootstrap 样本或统计显著性检验。",
        "",
        "## 配对核验",
        "",
        "六组均通过以下核验：已知/未知掩码一致；基线分数与处理组原始分数逐项相同；已知校准阈值、校准边界掩码、处理后校准阈值和测试修正分数均可从冻结输入重建；加权样本对贡献之和与总体 AUROC 差值一致。任一核验失败会中止脚本。",
        "",
        "## 总体结果",
        "",
        f"- 六组平均 AUROC：`{summary['mean_auc_control']:.6f} → {summary['mean_auc_treatment']:.6f}`，平均变化 `+{summary['mean_auc_delta']:.6f}`；正向变化 `{summary['splits_with_positive_auc_delta']}/{summary['split_count']}` 组。",
        f"- 边界带平均覆盖：未知样本 `{summary['mean_unknown_boundary_rate']:.1%}`，已知样本 `{summary['mean_known_boundary_rate']:.1%}`。边界带不是仅覆盖少量临界样本的窄区间。",
        f"- 样本对贡献重建的最大绝对误差：`{summary['max_abs_reconstruction_error']:.3e}`。",
        "",
        "## 四层归因",
        "",
        "| 未知/已知样本对分层 | 平均样本对占比 | 层内 AUROC 平均变化 | 对总体 AUROC 的平均加权贡献 | 层内变化为正的组数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in STRATA:
        row = summary["strata"][name]
        lines.append(
            f"| {label} | {row['mean_pair_weight']:.2%} | {row['mean_auc_delta_within_stratum']:+.6f} | "
            f"{row['mean_weighted_auc_delta_contribution']:+.6f} | {row['positive_within_stratum_delta_splits']}/{summary['split_count']} |"
        )
    lines += [
        "",
        "## 按类别划分逐组结果",
        "",
        "| 划分 | AUROC 对照 | AUROC 修正 | 差值 | 未知在带内 | 已知在带内 | 双方在带内样本对的加权贡献 | 其余三层贡献之和 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        both = row["strata"]["both_in_boundary"]["weighted_auc_delta"]
        other = row["auc_delta"] - both
        lines.append(
            f"| {row['split']} | {row['auc_control']:.6f} | {row['auc_treatment']:.6f} | {row['auc_delta']:+.6f} | "
            f"{row['boundary']['unknown_boundary_rate']:.1%} | {row['boundary']['known_boundary_rate']:.1%} | {both:+.6f} | {other:+.6f} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "样本对归因说明观测到的排序变化集中在哪些分层，不构成因果识别，也不能替代完整的新类别发现指标。若双方均在边界带内的层贡献占主导，可据此形成‘边界区样本对是排序变化主要来源’的机制假设；若六组方向不一致，则不支持把单组结果推广为稳定规律。无论结果如何，本分析不作为修改默认系数或扩大参数搜索的依据。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts/audit_uncertainty_band_pairwise_attribution_6split.py",
        "```",
        "",
        "机器可读结果：`analysis/uncertainty_band_pairwise_attribution_6split.json`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = [evaluate_seed(seed) for seed in SEEDS]
    result = {
        "question": "Across six frozen splits, which unknown-known pair strata account for the AUROC change?",
        "protocol": "post-hoc only; fixed validation-defined boundary; test labels used only for final attribution; no training or tuning",
        "seeds": list(SEEDS),
        "settings": {
            "target_known_coverage": TARGET_COVERAGE,
            "alpha": ALPHA,
            "boundary_multiplier": BAND_MULTIPLIER,
            "boundary_definition": "abs(raw_score - known_calibration_95th_percentile) <= boundary_multiplier * sample_std(known_calibration_raw_score)",
        },
        "rows": rows,
        "summary": summarize(rows),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
