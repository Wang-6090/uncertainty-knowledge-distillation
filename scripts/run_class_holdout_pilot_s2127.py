"""Run a frozen-feature class-holdout rejector pilot against the NNPU baseline.

The student checkpoint, semantic-hard split, mixed discovery pool, threshold
policy, candidate selection, and clustering settings are copied from the
existing seed-2127 baseline. The only changed factor is rejector training:
NNPU uses the mixed discovery pool, while class_holdout uses known training
classes as leave-one-class-out surrogate unknowns and never fits on the
discovery or test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


BASELINE_DIR = ROOT / "runs" / "rejection_ce_decay_full_s2127_static_control_discover"
CHECKPOINT = ROOT / "runs" / "rejection_ce_decay_full_s2127_static_control" / "student.pt"
SPLIT = ROOT / "splits" / "semantic_hard" / "seed2127" / "cifar100_60_40_semantic_hard.json"
TREATMENT_DIR = ROOT / "runs" / "class_holdout_pilot_s2127"
ANALYSIS_JSON = ROOT / "analysis" / "class_holdout_rejector_pilot_s2127_20261112.json"
ANALYSIS_MD = ROOT / "analysis" / "class_holdout_rejector_pilot_s2127_20261112.md"


METRIC_KEYS = (
    "auroc",
    "aupr",
    "known_accept_rate",
    "unknown_reject_rate",
    "cluster_candidate_count",
    "cluster_candidate_purity",
    "cluster_candidate_unknown_recall",
    "cluster_acc",
    "cluster_nmi",
    "cluster_ari",
    "cluster_k",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: dict[str, Any], key: str) -> float | int | None:
    value = report.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if key == "cluster_k":
        return int(value)
    return float(value)


def run_treatment() -> dict[str, Any]:
    report_path = TREATMENT_DIR / "discovery_report.json"
    if not report_path.is_file():
        if TREATMENT_DIR.exists():
            raise FileExistsError(
                f"refusing to reuse incomplete output directory: {TREATMENT_DIR}"
            )
        config = read_json(BASELINE_DIR / "discover_config.json")
        args = parse_args(["discover"])
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.seed = 2127
        args.model_seed = None
        args.work_dir = str(TREATMENT_DIR)
        args.student_ckpt = str(CHECKPOINT)
        args.split_path = str(SPLIT)
        args.rejector_training = "class_holdout"
        args.rejector_feature_mode = "embedding"
        args.rejector_model = "logistic"
        args.rejector_class_holdout_max_classes = 0
        print(
            "[discover] class_holdout with fixed checkpoint, split, pool, "
            "threshold, candidate selection, and clustering",
            flush=True,
        )
        discover(args)
    return {
        "output": str(TREATMENT_DIR),
        "config": read_json(TREATMENT_DIR / "discover_config.json"),
        "metrics": read_json(report_path),
        "calibration": read_json(TREATMENT_DIR / "calibration_report.json"),
    }


def build_comparison(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, dict[str, float | int | None]] = {}
    for key in METRIC_KEYS:
        base_value = metric(baseline["metrics"], key)
        treatment_value = metric(treatment["metrics"], key)
        delta = None
        if base_value is not None and treatment_value is not None:
            delta = treatment_value - base_value
        values[key] = {
            "baseline_nnpu": base_value,
            "treatment_class_holdout": treatment_value,
            "delta_treatment_minus_baseline": delta,
        }
    purity = values["cluster_candidate_purity"]
    if purity["treatment_class_holdout"] is not None:
        purity["treatment_candidate_contamination"] = 1.0 - float(
            purity["treatment_class_holdout"]
        )
    if purity["baseline_nnpu"] is not None:
        purity["baseline_candidate_contamination"] = 1.0 - float(purity["baseline_nnpu"])
    return values


def write_markdown(payload: dict[str, Any]) -> None:
    comparison = payload["comparison"]
    lines = [
        "# Class-holdout 拒识器 pilot：seed 2127",
        "",
        "## 研究假设",
        "当前 NNPU 拒识器需要从 mixed discovery pool 学习未知风险，而该池包含已知和未知样本。新实验改为只使用已知训练集：每次留出一个已知类别作为代理未知类，其余已知类别作为已知类，训练一个拒识器；最终对所有留出折的风险分数取固定均值。该方法检验已知类别内部的留出边界能否比 mixed-pool NNPU 更稳定地近似未知边界。",
        "",
        "## 单因素协议",
        "- 固定：同一 ResNet-18 学生 checkpoint、CIFAR-100 semantic-hard 60/40 seed 2127 划分、同一 mixed discovery pool、同一 known-coverage=0.95 阈值策略、同一候选筛选、PCA+KMeans 聚类和评价流程。",
        "- 唯一改变：`rejector_training` 从 `nnpu` 改为 `class_holdout`；class-holdout 的拒识特征改为 `embedding`，模型仍为 logistic。",
        "- 标签隔离：class-holdout 拒识器拟合只读取已知训练样本及其类别标签；discovery/test 标签不用于拒识器训练或阈值校准。",
        "",
        "## 原始结果与变化",
        "| 指标 | NNPU baseline | class-holdout | 变化（后者-前者） |",
        "|---|---:|---:|---:|",
    ]
    names = {
        "auroc": "AUROC",
        "aupr": "AUPR",
        "known_accept_rate": "已知接受率",
        "unknown_reject_rate": "未知拒识率",
        "cluster_candidate_count": "候选数",
        "cluster_candidate_purity": "候选池纯度",
        "cluster_candidate_unknown_recall": "候选未知召回率",
        "cluster_acc": "聚类 ACC",
        "cluster_nmi": "聚类 NMI",
        "cluster_ari": "聚类 ARI",
        "cluster_k": "选择的 K",
    }
    for key in METRIC_KEYS:
        row = comparison[key]
        def fmt(value: Any) -> str:
            if value is None:
                return "NA"
            if key == "cluster_k" or key == "cluster_candidate_count":
                return f"{int(value)}"
            return f"{float(value):.6f}"
        lines.append(
            f"| {names[key]} | {fmt(row['baseline_nnpu'])} | "
            f"{fmt(row['treatment_class_holdout'])} | "
            f"{fmt(row['delta_treatment_minus_baseline'])} |"
        )
    lines.extend(
        [
            "",
            "## 判定规则",
            "主要观察顺序为：未知拒识率和 AUROC 是否同时改善；候选池纯度是否改善且候选未知召回率没有明显下降；最后检查 ACC、NMI、ARI 是否同步改善。若只改善单一局部指标，记录为局部证据，不扩展参数扫描；若没有改善或恶化，则停止该方向。",
            "",
            f"- treatment 输出：`{payload['treatment']['output']}`",
            f"- 对照报告：`{BASELINE_DIR / 'discovery_report.json'}`",
            f"- 机器可读分析：`{ANALYSIS_JSON}`",
        ]
    )
    ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-analysis", action="store_true")
    options = parser.parse_args()
    for required in (BASELINE_DIR / "discovery_report.json", CHECKPOINT, SPLIT):
        if not required.is_file():
            raise FileNotFoundError(required)
    if ANALYSIS_JSON.exists() and not options.force_analysis:
        raise FileExistsError(f"refusing to overwrite {ANALYSIS_JSON}")
    baseline = {
        "output": str(BASELINE_DIR),
        "config": read_json(BASELINE_DIR / "discover_config.json"),
        "metrics": read_json(BASELINE_DIR / "discovery_report.json"),
        "calibration": read_json(BASELINE_DIR / "calibration_report.json"),
    }
    treatment = run_treatment()
    comparison = build_comparison(baseline, treatment)
    payload = {
        "protocol": "frozen seed-2127 class-holdout rejector pilot",
        "seed": 2127,
        "hypothesis": "known-only leave-one-class-out surrogate unknowns can stabilize the unknown boundary relative to mixed-pool NNPU",
        "single_changed_factor": "rejector_training (nnpu -> class_holdout), with embedding features required by the new protocol",
        "label_audit": {
            "discovery_or_test_labels_used_for_rejector_training": False,
            "discovery_or_test_labels_used_for_threshold_calibration": False,
        },
        "baseline": baseline,
        "treatment": treatment,
        "comparison": comparison,
    }
    ANALYSIS_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload)
    print(json.dumps({"json": str(ANALYSIS_JSON), "markdown": str(ANALYSIS_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
