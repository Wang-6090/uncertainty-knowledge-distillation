"""Post-hoc score-local placebo audit for the frozen s2129 paired runs.

This does not train, tune, or recalibrate a model. Test labels are used only
to evaluate AUROC; score bins and permutation strata depend only on raw scores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "runs/structure_representation_pilot_s2129_feature"
TREATMENT_DIR = ROOT / "runs/rejection_uncertainty_band_s2129"
ALPHA = 0.5
BAND_MULTIPLIER = 1.0
TARGET_COVERAGE = 0.95
PERMUTATIONS = 2000
BIN_COUNTS = (5, 10, 20)
RANDOM_SEED = 20261115
OUT_JSON = ROOT / "analysis/uncertainty_band_placebo_s2129.json"
OUT_MD = ROOT / "analysis/uncertainty_band_placebo_s2129.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise RuntimeError(f"paired artifact mismatch: {name}")


def require_close(name: str, actual: float, expected: float, atol: float = 1e-8) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} mismatch: actual={actual}, expected={expected}")


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, name))
        return result
    return {prefix: value}


def load_pair() -> dict:
    paths = {
        "base_config": BASELINE_DIR / "discover_config.json",
        "treatment_config": TREATMENT_DIR / "discover_config.json",
        "base_detail": BASELINE_DIR / "discovery_detail.json",
        "treatment_detail": TREATMENT_DIR / "discovery_detail.json",
        "base_calibration": BASELINE_DIR / "calibration_report.json",
        "treatment_calibration": TREATMENT_DIR / "calibration_report.json",
        "base_report": BASELINE_DIR / "discovery_report.json",
        "treatment_report": TREATMENT_DIR / "discovery_report.json",
        "calibration_scores": TREATMENT_DIR / "calibration_scores.npz",
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs:\n" + "\n".join(missing))

    base_config = read_json(paths["base_config"])
    treatment_config = read_json(paths["treatment_config"])
    base_flat, treatment_flat = flatten(base_config), flatten(treatment_config)
    config_differences = {
        key: [base_flat.get(key), treatment_flat.get(key)]
        for key in sorted(set(base_flat) | set(treatment_flat))
        if base_flat.get(key) != treatment_flat.get(key)
    }
    allowed_differences = {
        "boundary_uncertainty_alpha",
        "boundary_uncertainty_band_multiplier",
        "save_calibration_scores",
        "work_dir",
    }
    unexpected_differences = set(config_differences) - allowed_differences
    if unexpected_differences:
        raise RuntimeError(f"unexpected discovery config differences: {unexpected_differences}")
    if set(config_differences) != allowed_differences:
        raise RuntimeError(f"expected config differences not found: {config_differences}")
    require_close("alpha", float(treatment_config["boundary_uncertainty_alpha"]), ALPHA)
    require_close(
        "band multiplier",
        float(treatment_config["boundary_uncertainty_band_multiplier"]),
        BAND_MULTIPLIER,
    )
    require_close("target known coverage", float(treatment_config["target_known_coverage"]), TARGET_COVERAGE)

    base_detail = read_json(paths["base_detail"])
    treatment_detail = read_json(paths["treatment_detail"])
    for field in (
        "raw_score",
        "true_known",
        "true_label",
        "raw_labels",
        "pred_class",
        "head_uncertainty",
        "feature_norm",
        "entropy",
        "epistemic",
        "aleatoric",
        "expected_entropy",
    ):
        require_equal(
            field,
            np.asarray(base_detail[field]),
            np.asarray(treatment_detail[field]),
        )

    split_base = Path(str(base_config["split_path"]))
    split_treatment = Path(str(treatment_config["split_path"]))
    if split_base != split_treatment or not split_base.is_file():
        raise RuntimeError("the paired runs do not reference the same existing category split")
    if sha256(split_base) != sha256(split_treatment):
        raise RuntimeError("the paired runs reference category split files with different contents")
    run_split_base = BASELINE_DIR / "split.pt"
    run_split_treatment = TREATMENT_DIR / "split.pt"
    if not run_split_base.is_file() or not run_split_treatment.is_file():
        raise FileNotFoundError("one or both runs are missing their archived split.pt artifact")
    if sha256(run_split_base) != sha256(run_split_treatment):
        raise RuntimeError("the paired runs have different archived split.pt artifacts")
    checkpoint_base = Path(str(base_config["student_ckpt"]))
    checkpoint_treatment = Path(str(treatment_config["student_ckpt"]))
    if checkpoint_base != checkpoint_treatment or not checkpoint_base.is_file():
        raise RuntimeError("the paired runs do not reference the same existing student checkpoint")

    base_calibration = read_json(paths["base_calibration"])
    treatment_calibration = read_json(paths["treatment_calibration"])
    for report_name, report in (("base", base_calibration), ("treatment", treatment_calibration)):
        protocol = report["rejector_protocol"]
        pool = report["discovery_pool_protocol"]
        validation = report["validation_protocol"]
        if protocol.get("score_mode") != "feature_rejector" or protocol.get("training") != "nnpu":
            raise RuntimeError(f"unexpected rejector protocol in {report_name}")
        if pool.get("size") != 5400 or not np.isclose(pool.get("known_fraction"), 0.2):
            raise RuntimeError(f"unexpected discovery-pool composition in {report_name}")
        index_audit = validation.get("sample_index_audit", {})
        if not index_audit.get("disjoint") or index_audit.get("overlap_count") != 0:
            raise RuntimeError(f"validation selection/calibration overlap in {report_name}")
    if base_calibration["validation_protocol"]["sample_index_audit"] != treatment_calibration[
        "validation_protocol"
    ]["sample_index_audit"]:
        raise RuntimeError("checkpoint-selection or calibration indices differ")

    with np.load(paths["calibration_scores"], allow_pickle=False) as calibration:
        raw_calibration = np.asarray(calibration["raw_score"], dtype=np.float64).reshape(-1)
        uncertainty_calibration = np.asarray(
            calibration["validation_head_uncertainty"], dtype=np.float64
        ).reshape(-1)
        adjusted_calibration_saved = np.asarray(
            calibration["modified_score"], dtype=np.float64
        ).reshape(-1)
        boundary_calibration_saved = np.asarray(calibration["boundary_mask"], dtype=bool).reshape(-1)
        adjusted_threshold_saved = float(np.asarray(calibration["calibrated_threshold"]).reshape(-1)[0])

    baseline_threshold = float(base_calibration["calibrated_threshold"])
    treatment_boundary = treatment_calibration["boundary_uncertainty"]
    require_close("baseline thresholds", baseline_threshold, float(treatment_boundary["baseline_threshold"]))
    require_close(
        "baseline threshold from known calibration scores",
        baseline_threshold,
        float(np.quantile(raw_calibration, TARGET_COVERAGE, method="linear")),
    )
    score_std = max(float(np.std(raw_calibration, ddof=1)), 1e-6)
    uncertainty_mean = float(np.mean(uncertainty_calibration))
    uncertainty_std = max(float(np.std(uncertainty_calibration, ddof=1)), 1e-6)
    require_close("reported validation uncertainty mean", uncertainty_mean, float(treatment_boundary["uncertainty_mean"]))
    require_close("reported validation uncertainty std", uncertainty_std, float(treatment_boundary["uncertainty_std"]))
    require_close("reported calibration raw-score std", score_std, float(treatment_boundary["score_std"]))

    calibration_boundary = np.abs(raw_calibration - baseline_threshold) <= BAND_MULTIPLIER * score_std
    reconstructed_calibration = raw_calibration.copy()
    reconstructed_calibration[calibration_boundary] += ALPHA * score_std * (
        uncertainty_calibration[calibration_boundary] - uncertainty_mean
    ) / uncertainty_std
    require_equal("validation boundary mask", calibration_boundary, boundary_calibration_saved)
    require_close(
        "reconstructed adjusted validation scores",
        float(np.max(np.abs(reconstructed_calibration - adjusted_calibration_saved))),
        0.0,
    )
    require_close(
        "reconstructed calibrated threshold",
        adjusted_threshold_saved,
        float(np.quantile(reconstructed_calibration, TARGET_COVERAGE, method="linear")),
    )

    raw = np.asarray(treatment_detail["raw_score"], dtype=np.float64)
    uncertainty = np.asarray(treatment_detail["head_uncertainty"], dtype=np.float64)
    known = np.asarray(treatment_detail["true_known"], dtype=bool)
    observed = np.asarray(treatment_detail["score"], dtype=np.float64)
    boundary = np.abs(raw - baseline_threshold) <= BAND_MULTIPLIER * score_std
    reconstructed = raw.copy()
    reconstructed[boundary] += ALPHA * score_std * (
        uncertainty[boundary] - uncertainty_mean
    ) / uncertainty_std
    require_close("reconstructed treatment scores", float(np.max(np.abs(reconstructed - observed))), 0.0)
    if int(boundary.sum()) != int(treatment_boundary["test_boundary_count"]):
        raise RuntimeError("test boundary count does not match calibration report")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(uncertainty)):
        raise RuntimeError("non-finite score or uncertainty values")

    base_report = read_json(paths["base_report"])
    treatment_report = read_json(paths["treatment_report"])
    labels = (~known).astype(np.int8)
    auc_raw = float(roc_auc_score(labels, raw))
    auc_observed = float(roc_auc_score(labels, observed))
    require_close("baseline reported AUROC", auc_raw, float(base_report["auroc"]))
    require_close("treatment reported AUROC", auc_observed, float(treatment_report["auroc"]))

    boundary_indices = np.flatnonzero(boundary)
    standardized_uncertainty = (uncertainty[boundary] - uncertainty_mean) / uncertainty_std
    constant_shift = ALPHA * score_std * float(np.mean(standardized_uncertainty))
    constant_scores = raw.copy()
    constant_scores[boundary] += constant_shift
    constant_shift_delta = float(roc_auc_score(labels, constant_scores)) - auc_raw

    bin_results = []
    for bin_count in BIN_COUNTS:
        boundary_raw = raw[boundary]
        edges = np.quantile(boundary_raw, np.linspace(0.0, 1.0, bin_count + 1))
        bin_ids = np.searchsorted(edges[1:-1], boundary_raw, side="right")
        groups = [np.flatnonzero(bin_ids == idx) for idx in range(bin_count)]
        if len(groups) != bin_count or any(group.size == 0 for group in groups):
            raise RuntimeError(f"requested {bin_count} score bins did not materialize")
        rng = np.random.default_rng(RANDOM_SEED + 2129)
        placebo_deltas = np.empty(PERMUTATIONS, dtype=np.float64)
        for permutation_idx in range(PERMUTATIONS):
            placebo_uncertainty = uncertainty[boundary].copy()
            for group in groups:
                if group.size > 1:
                    placebo_uncertainty[group] = placebo_uncertainty[rng.permutation(group)]
            placebo_scores = raw.copy()
            placebo_scores[boundary_indices] += ALPHA * score_std * (
                placebo_uncertainty - uncertainty_mean
            ) / uncertainty_std
            placebo_deltas[permutation_idx] = float(roc_auc_score(labels, placebo_scores)) - auc_raw
        bin_results.append(
            {
                "requested_score_bins": bin_count,
                "actual_score_bins": len(groups),
                "score_bin_size_min": int(min(group.size for group in groups)),
                "score_bin_size_max": int(max(group.size for group in groups)),
                "permutations": PERMUTATIONS,
                "placebo_delta_mean": float(np.mean(placebo_deltas)),
                "placebo_delta_std": float(np.std(placebo_deltas, ddof=1)),
                "placebo_delta_quantiles": {
                    "2.5%": float(np.quantile(placebo_deltas, 0.025)),
                    "50%": float(np.quantile(placebo_deltas, 0.5)),
                    "97.5%": float(np.quantile(placebo_deltas, 0.975)),
                },
                "placebo_delta_max": float(np.max(placebo_deltas)),
                "placebo_count_at_least_observed": int(np.sum(placebo_deltas >= auc_observed - auc_raw)),
                "observed_above_placebo_max": bool(auc_observed - auc_raw > float(np.max(placebo_deltas))),
            }
        )

    base_discovery = read_json(BASELINE_DIR / "discovery_report.json")
    treatment_discovery = read_json(TREATMENT_DIR / "discovery_report.json")
    metrics = (
        "auroc",
        "aupr",
        "fpr95",
        "known_accept_rate",
        "unknown_reject_rate",
        "cluster_candidate_count",
        "cluster_candidate_true_unknown_count_before_purification",
        "cluster_candidate_false_reject_count_before_purification",
        "cluster_candidate_purity_before_purification",
        "cluster_acc",
        "cluster_nmi",
        "cluster_ari",
    )
    metric_comparison = {
        key: {
            "baseline": base_discovery.get(key),
            "treatment": treatment_discovery.get(key),
            "delta": (
                float(treatment_discovery[key]) - float(base_discovery[key])
                if base_discovery.get(key) is not None and treatment_discovery.get(key) is not None
                else None
            ),
        }
        for key in metrics
    }
    return {
        "protocol": "post-hoc score-local permutation placebo on frozen s2129 paired runs",
        "training_or_tuning_performed": False,
        "test_labels_used_for_permutation_or_score_bins": False,
        "test_labels_used_only_for_auroc_evaluation": True,
        "paired_conditions": {
            "baseline_run": str(BASELINE_DIR.relative_to(ROOT)),
            "treatment_run": str(TREATMENT_DIR.relative_to(ROOT)),
            "allowed_config_differences": config_differences,
            "split_path": str(split_base),
            "category_split_file_sha256": sha256(split_base),
            "baseline_split_artifact_sha256": sha256(run_split_base),
            "treatment_split_artifact_sha256": sha256(run_split_treatment),
            "student_checkpoint": str(checkpoint_base),
            "student_checkpoint_sha256": sha256(checkpoint_base),
            "split_artifacts_identical": True,
            "raw_scores_identical": True,
            "test_labels_and_known_mask_identical": True,
            "uncertainty_values_identical": True,
            "known_validation_selection_and_calibration_indices_identical": True,
            "discovery_pool_size": 5400,
            "discovery_pool_known_fraction": 0.2,
            "target_known_coverage": TARGET_COVERAGE,
            "alpha": ALPHA,
            "boundary_multiplier": BAND_MULTIPLIER,
            "baseline_threshold": baseline_threshold,
            "validation_score_std": score_std,
            "validation_uncertainty_mean": uncertainty_mean,
            "validation_uncertainty_std": uncertainty_std,
            "boundary_test_sample_count": int(boundary.sum()),
            "boundary_test_unknown_rate": float(np.mean(boundary[~known])),
            "boundary_test_known_rate": float(np.mean(boundary[known])),
        },
        "auroc": {
            "baseline": auc_raw,
            "observed_treatment": auc_observed,
            "observed_delta": auc_observed - auc_raw,
            "constant_mean_boundary_shift_delta": constant_shift_delta,
        },
        "reported_metric_comparison": metric_comparison,
        "placebo_settings": {
            "permutations_per_setting": PERMUTATIONS,
            "score_bins": list(BIN_COUNTS),
            "permutation_random_seed_base": RANDOM_SEED,
            "strata": "quantile bins of test raw scores restricted to validation-defined boundary; no test labels",
        },
        "placebo_results": bin_results,
        "interpretation": (
            "Exploratory, post-hoc mechanism diagnostic on one additional paired category split; "
            "not a significance test, causal estimate, or independent preregistered replication."
        ),
    }


def make_markdown(result: dict) -> str:
    pair = result["paired_conditions"]
    auc = result["auroc"]
    lines = [
        "# 边界不确定性 s2129 局部置换 placebo 复核",
        "",
        "日期：2026-11-15  ",
        "类型：冻结配对结果的事后机制诊断；没有重新训练、调参或重新选择模型。",
        "",
        "## 为什么做这项检查",
        "",
        "此前六组冻结结果显示，边界不确定性修正在六组中的 AUROC 均为正变化，且高于各自的局部分数置换 placebo。为检查这一描述性现象能否在另一个已有类别划分上复现，本次复用 s2129 的基线和处理组结果；在运行分析前先审计了两组的配置和逐样本数据。",
        "",
        "## 配对条件核验",
        "",
        f"- 基线：`{pair['baseline_run']}`；处理组：`{pair['treatment_run']}`。",
        f"- 配置引用的类别划分文件哈希：`{pair['category_split_file_sha256']}`；基线与处理组归档的 `split.pt` 哈希分别为 `{pair['baseline_split_artifact_sha256']}`、`{pair['treatment_split_artifact_sha256']}`，两者一致。学生检查点哈希：`{pair['student_checkpoint_sha256']}`。",
        f"- 两组除边界修正参数、校准分数导出选项和输出目录外，发现阶段配置完全一致。修正参数为 `alpha={pair['alpha']}`、边界倍数 `{pair['boundary_multiplier']}`；两组均以 `{pair['target_known_coverage']:.0%}` 已知覆盖目标确定工作阈值。",
        f"- 发现池均为 {pair['discovery_pool_size']:,} 个样本，已知比例为 {pair['discovery_pool_known_fraction']:.0%}。逐样本原始拒识分数、已知/未知掩码、类别标签及不确定性值完全一致；基线和处理组各有 10,000 个测试样本。",
        f"- 验证集选模与阈值校准索引一致且互不重叠。按实现公式重算验证集和测试集修正分数，均与保存值一致；测试集中有 {pair['boundary_test_sample_count']:,} 个样本落入修正带。该边界带覆盖 {pair['boundary_test_unknown_rate']:.1%} 的未知样本和 {pair['boundary_test_known_rate']:.1%} 的已知样本，不能将作用范围描述为少量临界样本。",
        "",
        "唯一研究因素是是否启用边界不确定性修正；由于 `alpha` 与边界带宽度同时从 0 改为非零，本配对不能区分两者各自的贡献。",
        "",
        "## AUROC 对照",
        "",
        f"- 基线 AUROC：`{auc['baseline']:.6f}`。",
        f"- 启用修正后：`{auc['observed_treatment']:.6f}`，变化 `+{auc['observed_delta']:.6f}`。",
        f"- 若只给边界内样本施加同一个平均分数偏移，AUROC 变化为 `{auc['constant_mean_boundary_shift_delta']:+.6f}`。",
        "",
        "## 局部分数置换 placebo",
        "",
        "在验证集确定的测试边界带内，按测试原始拒识分数建立 5、10、20 个等频区间；每次仅在同一区间内打乱不确定性值与样本的对应关系，再用原有验证集均值、标准差、修正系数和边界规则重算 AUROC。分箱只由原始分数决定，测试标签没有参与分箱、置换或参数选择，只用于最终 AUROC 计算。每种分箱设置运行 2,000 次。",
        "",
        "| 分箱数 | 实际分箱数 | 单箱样本量范围 | placebo AUROC 变化均值 | placebo 95% 分位区间 | placebo 最大变化 | 达到/超过实际变化次数 | 实际变化高于 placebo 最大值 |",
        "|---:|---:|---:|---:|:---|---:|---:|:---:|",
    ]
    for row in result["placebo_results"]:
        q = row["placebo_delta_quantiles"]
        lines.append(
            f"| {row['requested_score_bins']} | {row['actual_score_bins']} | "
            f"[{row['score_bin_size_min']}, {row['score_bin_size_max']}] | "
            f"{row['placebo_delta_mean']:+.6f} | "
            f"[{q['2.5%']:+.6f}, {q['97.5%']:+.6f}] | "
            f"{row['placebo_delta_max']:+.6f} | "
            f"{row['placebo_count_at_least_observed']}/{row['permutations']} | "
            f"{'是' if row['observed_above_placebo_max'] else '否'} |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "如果实际变化高于各分箱设置下所有有限次 placebo 的最大值，这仅表明在本次事后构造的局部置换中，没有一次达到实际变化；不能将其换算成统计显著性，也不能当作独立、预先设计的复现。与此前六组分析一致的方向，可作为‘边界内不确定性与具体样本的对应关系可能包含额外排序信息’的机制线索。它不证明因果，也不能单独支持跨划分稳定收益。",
        "",
        "候选池和聚类指标的 s2129 基线—处理组差值另列于配套 JSON 的 `reported_metric_comparison`；这项 placebo 只针对 AUROC 排序变化，不把聚类变化归因于置换机制。",
        "",
        "复现命令：",
        "",
        "```powershell",
        "python scripts/audit_uncertainty_band_placebo_s2129.py",
        "```",
        "",
        "JSON 结果：`analysis/uncertainty_band_placebo_s2129.json`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    result = load_pair()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "auroc": result["auroc"], "placebo": result["placebo_results"]}, indent=2))


if __name__ == "__main__":
    main()
