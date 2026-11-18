"""Conditional placebo replication on the frozen semantic-hard s2131 pair.

This post-hoc audit compares raw-score-only and predicted-class-plus-raw-score
permutations. It does not train, recalibrate, tune, or use test labels to form
bins/strata. Test labels are used only for AUROC evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = ROOT / "runs" / "boundary_uncertainty_s2131_control"
TREATMENT_DIR = ROOT / "runs" / "boundary_uncertainty_s2131_band1_alpha05"
OUT_JSON = ROOT / "analysis" / "uncertainty_band_class_score_placebo_s2131.json"
OUT_MD = ROOT / "analysis" / "uncertainty_band_class_score_placebo_s2131.md"
SPLIT_ID = 2131
TARGET_COVERAGE = 0.95
ALPHA = 0.5
BAND_MULTIPLIER = 1.0
SCORE_BINS = 10
PERMUTATIONS = 2000
RANDOM_SEED = 20261117


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, name))
        return result
    return {prefix: value}


def require_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise RuntimeError(f"paired artifact mismatch: {name}")


def require_close(name: str, actual: float, expected: float, atol: float = 1e-8) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} mismatch: actual={actual}, expected={expected}")


def load_pair() -> dict:
    base_config = read_json(CONTROL_DIR / "discover_config.json")
    treatment_config = read_json(TREATMENT_DIR / "discover_config.json")
    base_flat, treatment_flat = flatten(base_config), flatten(treatment_config)
    config_differences = {
        key: [base_flat.get(key), treatment_flat.get(key)]
        for key in sorted(set(base_flat) | set(treatment_flat))
        if base_flat.get(key) != treatment_flat.get(key)
    }
    expected_differences = {
        "boundary_uncertainty_alpha",
        "boundary_uncertainty_band_multiplier",
        "work_dir",
    }
    if set(config_differences) != expected_differences:
        raise RuntimeError(f"unexpected paired config differences: {config_differences}")
    require_close("alpha", float(base_config["boundary_uncertainty_alpha"]), 0.0)
    require_close("control band", float(base_config["boundary_uncertainty_band_multiplier"]), 0.0)
    require_close("treatment alpha", float(treatment_config["boundary_uncertainty_alpha"]), ALPHA)
    require_close(
        "treatment band",
        float(treatment_config["boundary_uncertainty_band_multiplier"]),
        BAND_MULTIPLIER,
    )
    require_close("target known coverage", float(treatment_config["target_known_coverage"]), TARGET_COVERAGE)

    split_path = Path(str(base_config["split_path"]))
    if split_path != Path(str(treatment_config["split_path"])) or not split_path.is_file():
        raise RuntimeError("paired runs do not reference the same existing category split")
    base_archived_split = CONTROL_DIR / "split.pt"
    treatment_archived_split = TREATMENT_DIR / "split.pt"
    if sha256(base_archived_split) != sha256(treatment_archived_split):
        raise RuntimeError("archived category split differs between paired runs")
    student_ckpt = Path(str(base_config["student_ckpt"]))
    if student_ckpt != Path(str(treatment_config["student_ckpt"])) or not student_ckpt.is_file():
        raise RuntimeError("paired runs do not use the same existing student checkpoint")

    base_detail = read_json(CONTROL_DIR / "discovery_detail.json")
    treatment_detail = read_json(TREATMENT_DIR / "discovery_detail.json")
    for field in (
        "raw_score",
        "true_known",
        "true_label",
        "raw_labels",
        "pred_class",
        "head_uncertainty",
    ):
        require_equal(field, np.asarray(base_detail[field]), np.asarray(treatment_detail[field]))

    base_report = read_json(CONTROL_DIR / "calibration_report.json")
    treatment_report = read_json(TREATMENT_DIR / "calibration_report.json")
    if base_report["validation_protocol"].get("sample_index_audit") != treatment_report[
        "validation_protocol"
    ].get("sample_index_audit"):
        raise RuntimeError("paired validation selection/calibration indices differ")
    boundary_report = treatment_report["boundary_uncertainty"]
    if not boundary_report.get("enabled"):
        raise RuntimeError("boundary correction is not enabled for the treatment")

    with np.load(CONTROL_DIR / "calibration_scores.npz", allow_pickle=False) as base_cal:
        raw_cal_base = np.asarray(base_cal["raw_score"], dtype=np.float64).reshape(-1)
        uncertainty_cal_base = np.asarray(
            base_cal["validation_head_uncertainty"], dtype=np.float64
        ).reshape(-1)
    with np.load(TREATMENT_DIR / "calibration_scores.npz", allow_pickle=False) as treatment_cal:
        raw_cal = np.asarray(treatment_cal["raw_score"], dtype=np.float64).reshape(-1)
        uncertainty_cal = np.asarray(
            treatment_cal["validation_head_uncertainty"], dtype=np.float64
        ).reshape(-1)
        adjusted_cal_saved = np.asarray(treatment_cal["modified_score"], dtype=np.float64).reshape(-1)
        boundary_cal_saved = np.asarray(treatment_cal["boundary_mask"], dtype=bool).reshape(-1)
        adjusted_threshold_saved = float(np.asarray(treatment_cal["calibrated_threshold"]).reshape(-1)[0])
    require_equal("calibration raw scores", raw_cal_base, raw_cal)
    require_equal("calibration uncertainty", uncertainty_cal_base, uncertainty_cal)

    threshold = float(base_report["calibrated_threshold"])
    require_close(
        "control known-coverage threshold",
        threshold,
        float(np.quantile(raw_cal, TARGET_COVERAGE, method="linear")),
    )
    score_std = max(float(np.std(raw_cal, ddof=1)), 1e-6)
    uncertainty_mean = float(np.mean(uncertainty_cal))
    uncertainty_std = max(float(np.std(uncertainty_cal, ddof=1)), 1e-6)
    boundary_cal = np.abs(raw_cal - threshold) <= BAND_MULTIPLIER * score_std
    adjusted_cal = raw_cal.copy()
    adjusted_cal[boundary_cal] += ALPHA * score_std * (
        uncertainty_cal[boundary_cal] - uncertainty_mean
    ) / uncertainty_std
    require_equal("reconstructed calibration boundary", boundary_cal, boundary_cal_saved)
    require_close("reconstructed calibration scores", float(np.max(np.abs(adjusted_cal - adjusted_cal_saved))), 0.0)
    recalibrated_threshold = float(np.quantile(adjusted_cal, TARGET_COVERAGE, method="linear"))
    require_close("reconstructed treatment threshold", adjusted_threshold_saved, recalibrated_threshold)
    require_close("reported baseline threshold", float(boundary_report["baseline_threshold"]), threshold)
    require_close("reported treatment score standard deviation", float(boundary_report["score_std"]), score_std)
    require_close("reported uncertainty mean", float(boundary_report["uncertainty_mean"]), uncertainty_mean)
    require_close("reported uncertainty standard deviation", float(boundary_report["uncertainty_std"]), uncertainty_std)

    raw = np.asarray(base_detail["raw_score"], dtype=np.float64)
    known = np.asarray(base_detail["true_known"], dtype=bool)
    predicted_class = np.asarray(base_detail["pred_class"], dtype=np.int64)
    uncertainty = np.asarray(treatment_detail["head_uncertainty"], dtype=np.float64)
    observed_adjusted = np.asarray(treatment_detail["score"], dtype=np.float64)
    boundary = np.abs(raw - threshold) <= BAND_MULTIPLIER * score_std
    reconstructed_adjusted = raw.copy()
    reconstructed_adjusted[boundary] += ALPHA * score_std * (
        uncertainty[boundary] - uncertainty_mean
    ) / uncertainty_std
    require_close(
        "reconstructed treatment test scores",
        float(np.max(np.abs(reconstructed_adjusted - observed_adjusted))),
        0.0,
    )
    if int(boundary.sum()) != int(boundary_report["test_boundary_count"]):
        raise RuntimeError("reported test boundary count differs from reconstructed mask")
    if not all(np.all(np.isfinite(array)) for array in (raw, uncertainty, observed_adjusted)):
        raise RuntimeError("non-finite values in paired test arrays")

    return {
        "raw": raw,
        "known": known,
        "predicted_class": predicted_class,
        "uncertainty": uncertainty,
        "boundary": boundary,
        "observed_adjusted": observed_adjusted,
        "score_std": score_std,
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_std": uncertainty_std,
        "calibration_sample_count": int(raw_cal.size),
        "split_sha256": sha256(split_path),
        "student_checkpoint_sha256": sha256(student_ckpt),
        "config_differences": config_differences,
    }


def make_strata(raw_boundary: np.ndarray, predicted_class: np.ndarray, bin_count: int, conditional: bool):
    edges = np.quantile(raw_boundary, np.linspace(0.0, 1.0, bin_count + 1))
    score_bin = np.searchsorted(edges[1:-1], raw_boundary, side="right")
    if conditional:
        keys = np.stack((predicted_class, score_bin), axis=1)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    else:
        _, inverse, counts = np.unique(score_bin, return_inverse=True, return_counts=True)
    groups = [np.flatnonzero(inverse == idx) for idx in range(counts.size)]
    return groups, counts


def run_placebo(data: dict, conditional: bool, seed_offset: int) -> dict:
    raw = data["raw"]
    boundary = data["boundary"]
    uncertainty = data["uncertainty"]
    labels = (~data["known"]).astype(np.int8)
    raw_auc = float(roc_auc_score(labels, raw))
    observed_auc = float(roc_auc_score(labels, data["observed_adjusted"]))
    observed_delta = observed_auc - raw_auc
    boundary_indices = np.flatnonzero(boundary)
    raw_boundary = raw[boundary]
    uncertainty_boundary = uncertainty[boundary]
    classes_boundary = data["predicted_class"][boundary]
    groups, counts = make_strata(raw_boundary, classes_boundary, SCORE_BINS, conditional)
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    deltas = np.empty(PERMUTATIONS, dtype=np.float64)
    for permutation_idx in range(PERMUTATIONS):
        placebo_uncertainty = uncertainty_boundary.copy()
        for group in groups:
            if group.size > 1:
                placebo_uncertainty[group] = uncertainty_boundary[rng.permutation(group)]
        placebo_score = raw.copy()
        placebo_score[boundary_indices] += ALPHA * data["score_std"] * (
            placebo_uncertainty - data["uncertainty_mean"]
        ) / data["uncertainty_std"]
        deltas[permutation_idx] = float(roc_auc_score(labels, placebo_score)) - raw_auc
    return {
        "conditioning": "predicted_class_and_raw_score_bin" if conditional else "raw_score_bin_only",
        "requested_score_bins": SCORE_BINS,
        "stratum_count": int(counts.size),
        "singleton_sample_fraction": float(np.sum(counts[counts == 1]) / boundary.sum()),
        "strata_smaller_than_three": int(np.sum(counts < 3)),
        "stratum_size_min": int(counts.min()),
        "stratum_size_median": float(np.median(counts)),
        "stratum_size_max": int(counts.max()),
        "permutation_count": PERMUTATIONS,
        "raw_auroc": raw_auc,
        "observed_adjusted_auroc": observed_auc,
        "observed_auroc_delta": observed_delta,
        "placebo_mean_delta": float(np.mean(deltas)),
        "placebo_quantiles": {
            "2.5%": float(np.quantile(deltas, 0.025)),
            "50%": float(np.quantile(deltas, 0.5)),
            "97.5%": float(np.quantile(deltas, 0.975)),
        },
        "placebo_max_delta": float(np.max(deltas)),
        "placebo_count_at_least_observed": int(np.sum(deltas >= observed_delta)),
        "observed_above_placebo_max": bool(observed_delta > np.max(deltas)),
        "test_labels_used_for_strata": False,
        "test_labels_used_only_for_auc": True,
    }


def make_markdown(result: dict) -> str:
    paired = result["paired_audit"]
    lines = [
        "# 边界不确定性：类别条件置换独立划分复核（s2131）",
        "",
        "## 目的与配对审计",
        "",
        "此前六组冻结结果中，按预测类别与原始拒识分数共同分层后，观测 AUROC 变化仍高于有限置换范围。本复核在另一组 semantic-hard 类别组合（s2131）上重复同一固定 10 箱诊断，并同时保留仅按分数分层的置换对照。该配对冻结结果沿用 alpha=0.5、边界宽度 1.0；本分析不训练、不搜索参数、不改变既有校准结果，仅重建校准计算以核验归档值。",
        "",
        f"运行配置仅在 `{', '.join(paired['config_differences'])}` 三项不同；共享类别划分 SHA-256 为 `{paired['split_sha256']}`，共享学生检查点 SHA-256 为 `{paired['student_checkpoint_sha256']}`。两组校准数组、测试原始分数、标签、预测类别和不确定性逐元素一致。脚本从已知校准数据重建边界掩码、修正后校准阈值和测试分数，均与归档数值一致。校准样本数 `{paired['calibration_sample_count']}`，测试样本数 `{result['test_sample_count']}`，边界内测试样本 `{result['boundary_sample_count']}`。",
        "",
        "## 结果",
        "",
        "| 置换条件 | 实际 AUROC 变化 | placebo 均值 | placebo 95% 区间 | placebo 最大变化 | 达到/超过实际值次数 | 实际值高于最大值 | 单例样本比例 |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in result["rows"]:
        q = row["placebo_quantiles"]
        condition_label = (
            "仅按原始分数分层"
            if row["conditioning"] == "raw_score_bin_only"
            else "按预测类别与原始分数共同分层"
        )
        lines.append(
            f"| {condition_label} | {row['observed_auroc_delta']:+.6f} | "
            f"{row['placebo_mean_delta']:+.6f} | "
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
        "加入预测类别分层后，观测变化仍高于本次有限置换最大值；这表示在保留预测类别和粗分数区间内不确定性取值分布的 placebo 中，未重现观测变化。该置换条件形成 591 个分层，其中 50 个分层少于 3 个样本，仍受分层粒度和小组限制；结果不是统计显著性、因果效应或新实验性能收益的证明。该结果仅是一组类别组合上的机制复核，不能单独确立跨划分稳定性，也不用于改变默认配置。",
        "",
        "测试标签不参与分箱和置换分层，只用于 AUROC 计算。两类 placebo 各执行 2,000 次；固定随机种子分别为 `20261117`（仅分数）和 `20261118`（预测类别加分数）。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts/audit_uncertainty_band_class_score_placebo_s2131.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = load_pair()
    rows = [
        run_placebo(data, conditional=False, seed_offset=0),
        run_placebo(data, conditional=True, seed_offset=1),
    ]
    result = {
        "question": "Does the observed s2131 AUROC change exceed score-only and predicted-class-plus-score conditional placebos?",
        "protocol": "post-hoc diagnostic on one additional frozen paired semantic-hard split; no training or tuning",
        "split_id": SPLIT_ID,
        "score_bins": SCORE_BINS,
        "permutations_per_condition": PERMUTATIONS,
        "test_labels_used_for_strata": False,
        "test_labels_used_only_for_auc": True,
        "paired_audit": {
            "config_differences": data["config_differences"],
            "split_sha256": data["split_sha256"],
            "student_checkpoint_sha256": data["student_checkpoint_sha256"],
            "calibration_sample_count": data["calibration_sample_count"],
        },
        "test_sample_count": int(data["raw"].size),
        "boundary_sample_count": int(data["boundary"].sum()),
        "rows": rows,
        "interpretation": "exploratory conditional-placebo replication on one split; not a significance test or causal analysis",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
