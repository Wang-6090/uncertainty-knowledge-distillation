"""Paired frozen-checkpoint test of boundary uncertainty on split s2131.

Only the boundary correction switches between disabled and the fixed
follow-up setting (band=1.0, alpha=0.5). Model weights, data protocol, rejector fit,
calibration subset, and downstream clustering settings are held fixed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


SOURCE = ROOT / "runs/relational_kd_independent_replication_s2131_20261115_control_discover"
TRAIN = ROOT / "runs/relational_kd_independent_replication_s2131_20261115_control"
OLD_REPORT = SOURCE / "discovery_report.json"
OLD_DETAIL = SOURCE / "discovery_detail.json"
OLD_REJECTOR_SCORES = SOURCE / "feature_rejector_scores.npz"
SPLIT = ROOT / "runs/relational_kd_independent_replication_s2131_20261115/protocol/cifar100_60_40_semantic_hard.json"
OUT_CONTROL = ROOT / "runs/boundary_uncertainty_s2131_control"
OUT_TREATMENT = ROOT / "runs/boundary_uncertainty_s2131_band1_alpha05"
RECORD_JSON = ROOT / "analysis/boundary_uncertainty_s2131_paired_recheck.json"
RECORD_MD = ROOT / "analysis/boundary_uncertainty_s2131_paired_recheck.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_args(source_config: dict, output: Path, band: float, alpha: float):
    args = parse_args(["discover"])
    for key, value in source_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.device = "auto"
    args.seed = 2131
    args.split_path = str(SPLIT)
    args.student_ckpt = str(TRAIN / "student.pt")
    args.work_dir = str(output)
    args.boundary_uncertainty_band_multiplier = band
    args.boundary_uncertainty_alpha = alpha
    args.save_calibration_scores = True
    args.skip_clustering = False
    return args


def verify_pair_config(control, treatment) -> None:
    left, right = vars(control), vars(treatment)
    allowed = {
        "work_dir",
        "boundary_uncertainty_band_multiplier",
        "boundary_uncertainty_alpha",
    }
    differences = {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key) and key not in allowed
    }
    if differences:
        raise RuntimeError(f"paired configurations differ beyond treatment: {differences}")
    if control.student_ckpt != treatment.student_ckpt:
        raise RuntimeError("student checkpoint paths differ")
    if control.split_path != treatment.split_path:
        raise RuntimeError("split paths differ")


def metrics(report: dict) -> dict:
    keys = (
        "auroc",
        "aupr",
        "fpr95",
        "known_accept_rate",
        "unknown_reject_rate",
        "cluster_candidate_count",
        "cluster_candidate_purity",
        "cluster_acc",
        "cluster_nmi",
        "cluster_ari",
    )
    return {key: report.get(key) for key in keys}


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_record(manifest: dict, control_report: dict, treatment_report: dict,
                 checks: dict) -> None:
    before, after = metrics(control_report), metrics(treatment_report)
    delta = {}
    for key in before:
        a, b = before[key], after[key]
        delta[key] = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    payload = {
        "manifest": manifest,
        "checks": checks,
        "control": before,
        "treatment": after,
        "delta_treatment_minus_control": delta,
        "thresholds": {
            "control": load_json(OUT_CONTROL / "calibration_report.json").get("calibrated_threshold"),
            "treatment": load_json(OUT_TREATMENT / "calibration_report.json").get("calibrated_threshold"),
        },
        "boundary": {
            "control": load_json(OUT_CONTROL / "calibration_report.json").get("boundary_uncertainty"),
            "treatment": load_json(OUT_TREATMENT / "calibration_report.json").get("boundary_uncertainty"),
        },
    }
    RECORD_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = {
        "auroc": "AUROC",
        "aupr": "AUPR",
        "fpr95": "FPR95",
        "known_accept_rate": "已知类别接受率",
        "unknown_reject_rate": "未知类别拒识率",
        "cluster_candidate_count": "候选池样本数",
        "cluster_candidate_purity": "候选池纯度",
        "cluster_acc": "聚类 ACC",
        "cluster_nmi": "聚类 NMI",
        "cluster_ari": "聚类 ARI",
    }
    rows = ["| 指标 | 修正关闭 | 修正开启 | 变化（开启−关闭） |", "|---|---:|---:|---:|"]
    for key, label in labels.items():
        rows.append(
            f"| {label} | {fmt(before[key])} | {fmt(after[key])} | "
            f"{fmt(delta[key]) if delta[key] is not None else '—'} |"
        )
    md = [
        "# 边界不确定性 s2131 冻结配对验证",
        "",
        "日期：2026-11-15  ",
        "实验类型：同一冻结学生检查点上的发现阶段配对；不重新训练、不搜索参数。",
        "",
        "## 目的与对照",
        "",
        "验证边界不确定性修正能否改善阈值附近样本的未知拒识。两组使用相同的 s2131 CIFAR-100 semantic-hard 60/40 类别划分、学生检查点、发现池构建、拒识器配置、验证校准样本、阈值策略和聚类设置；唯一机制差别为修正关闭，或固定开启（边界带倍数 1.0、系数 0.5）。阈值均只由独立已知验证校准子集按 95% 已知覆盖目标确定。未知标签只用于最终测试指标。",
        "该配置沿用此前 s2129 单划分配对复核的固定设置；因此 s2131 用于检查新类别划分上的可重复性，不将其表述为事先注册的确认性实验，也不扫描参数。",
        "",
        "## 实施与可比性核验",
        "",
        f"- 学生检查点 SHA-256：`{manifest['checkpoint_sha256']}`。",
        f"- 类别划分 SHA-256：`{manifest['split_sha256']}`。",
        f"- 测试集逐样本原始拒识分数最大绝对差：`{checks['max_abs_test_raw_score_diff']:.12g}`。",
        f"- 校准集逐样本原始拒识分数最大绝对差：`{checks['max_abs_calibration_raw_score_diff']:.12g}`。",
        f"- 两组测试样本索引一致：`{checks['test_indices_equal']}`；两组校准样本索引一致：`{checks['calibration_indices_equal']}`。",
        f"- 旧基线与控制重跑测试索引一致：`{checks['old_vs_rerun_test_indices_equal']}`；拒识器验证样本索引一致：`{checks['old_vs_rerun_validation_indices_equal']}`。",
        f"- 两组测试标签及顺序一致：`{checks['test_labels_equal']}`；旧控制与重跑测试原始拒识分数最大差：`{checks['max_abs_old_vs_rerun_raw_score_diff']:.12g}`；验证集拒识分数最大差：`{checks['max_abs_old_vs_rerun_validation_score_diff']:.12g}`。",
        "",
        "## 结果",
        "",
        *rows,
        "",
        "AUROC/AUPR 越高通常表示排序区分能力越强，FPR95 越低越好；已知接受率、未知拒识率和候选池纯度需要结合目标覆盖率一起解释。聚类 ACC/NMI/ARI 越高表示聚类与真实类别对应关系越好。候选池纯度和聚类指标均为测试标签上的事后评价，不参与阈值或方法选择。",
        "",
        "## 结论边界",
        "",
        "该实验只检验一个类别划分，结果属于单划分补充证据，不代表跨划分稳定性。若控制重跑分数未能复现既有结果，仍以本次两组同代码、同检查点的配对差值作为机制对照，同时将复现偏差单独说明。",
        "",
        "## 产物",
        "",
        f"- 修正关闭：`{OUT_CONTROL.relative_to(ROOT).as_posix()}/`",
        f"- 修正开启：`{OUT_TREATMENT.relative_to(ROOT).as_posix()}/`",
        f"- 机器可读摘要：`{RECORD_JSON.relative_to(ROOT).as_posix()}`",
    ]
    RECORD_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    for required in (SOURCE / "discover_config.json", OLD_REPORT, OLD_DETAIL,
                     OLD_REJECTOR_SCORES, TRAIN / "student.pt", SPLIT):
        if not required.is_file():
            raise FileNotFoundError(required)
    for output in (RECORD_JSON, RECORD_MD):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    existing_pair = (OUT_CONTROL.exists(), OUT_TREATMENT.exists())
    if any(existing_pair) and not all(existing_pair):
        raise RuntimeError(
            "only one paired output directory exists; inspect it before any rerun"
        )
    resume_from_outputs = all(existing_pair)
    if resume_from_outputs:
        required_outputs = (
            OUT_CONTROL / "discover_config.json",
            OUT_CONTROL / "discovery_detail.json",
            OUT_CONTROL / "discovery_report.json",
            OUT_CONTROL / "feature_rejector_scores.npz",
            OUT_CONTROL / "calibration_scores.npz",
            OUT_TREATMENT / "discover_config.json",
            OUT_TREATMENT / "discovery_detail.json",
            OUT_TREATMENT / "discovery_report.json",
            OUT_TREATMENT / "feature_rejector_scores.npz",
            OUT_TREATMENT / "calibration_scores.npz",
        )
        missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
        if missing_outputs:
            raise RuntimeError(
                "paired outputs are incomplete; refusing to recompute or overwrite: "
                + ", ".join(missing_outputs)
            )

    source_config = load_json(SOURCE / "discover_config.json")
    expected_source_config = {
        "command": "discover",
        "dataset": "cifar100",
        "seed": 2131,
        "split_path": str(SPLIT),
        "student_ckpt": str(TRAIN / "student.pt"),
        "threshold_policy": "known_coverage",
        "target_known_coverage": 0.95,
        "score_mode": "feature_rejector",
        "calibration_ratio": 0.25,
        "open_val_ratio": 0.0,
    }
    mismatches = {
        key: (source_config.get(key), expected)
        for key, expected in expected_source_config.items()
        if source_config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"source discovery protocol differs from the planned pair: {mismatches}")
    control_args = make_args(source_config, OUT_CONTROL, 0.0, 0.0)
    treatment_args = make_args(source_config, OUT_TREATMENT, 1.0, 0.5)
    verify_pair_config(control_args, treatment_args)
    if (
        control_args.boundary_uncertainty_band_multiplier != 0.0
        or control_args.boundary_uncertainty_alpha != 0.0
        or treatment_args.boundary_uncertainty_band_multiplier != 1.0
        or treatment_args.boundary_uncertainty_alpha != 0.5
    ):
        raise RuntimeError("boundary-uncertainty treatment settings are not active as specified")
    training_config = load_json(TRAIN / "train_student_config.json")
    if training_config.get("split_path") != str(SPLIT) or training_config.get("seed") != 2131:
        raise RuntimeError("frozen checkpoint training provenance does not match split s2131")

    manifest = {
        "date": "2026-11-15",
        "split_id": "semantic-hard-60-40-s2131",
        "seed": 2131,
        "control": {"boundary_uncertainty_band_multiplier": 0.0, "boundary_uncertainty_alpha": 0.0},
        "treatment": {"boundary_uncertainty_band_multiplier": 1.0, "boundary_uncertainty_alpha": 0.5},
        "checkpoint": str(TRAIN / "student.pt"),
        "checkpoint_sha256": sha256(TRAIN / "student.pt"),
        "split": str(SPLIT),
        "split_sha256": sha256(SPLIT),
        "source_discovery_config_sha256": sha256(SOURCE / "discover_config.json"),
        "training_config_sha256": sha256(TRAIN / "train_student_config.json"),
        "calibration_ratio": source_config.get("calibration_ratio"),
        "threshold_policy": source_config.get("threshold_policy"),
        "target_known_coverage": source_config.get("target_known_coverage"),
        "device": control_args.device,
    }
    if resume_from_outputs:
        saved_control = load_json(OUT_CONTROL / "discover_config.json")
        saved_treatment = load_json(OUT_TREATMENT / "discover_config.json")
        allowed_saved_differences = {
            "work_dir",
            "boundary_uncertainty_band_multiplier",
            "boundary_uncertainty_alpha",
        }
        saved_differences = {
            key: (saved_control.get(key), saved_treatment.get(key))
            for key in sorted(set(saved_control) | set(saved_treatment))
            if saved_control.get(key) != saved_treatment.get(key)
            and key not in allowed_saved_differences
        }
        if saved_differences:
            raise RuntimeError(
                f"saved paired configurations differ beyond treatment: {saved_differences}"
            )
        expected_saved_settings = (
            (saved_control, 0.0, 0.0),
            (saved_treatment, 1.0, 0.5),
        )
        for saved, expected_band, expected_alpha in expected_saved_settings:
            if (
                saved.get("seed") != 2131
                or saved.get("split_path") != str(SPLIT)
                or saved.get("student_ckpt") != str(TRAIN / "student.pt")
                or saved.get("calibration_ratio") != 0.25
                or saved.get("threshold_policy") != "known_coverage"
                or saved.get("target_known_coverage") != 0.95
                or saved.get("boundary_uncertainty_band_multiplier") != expected_band
                or saved.get("boundary_uncertainty_alpha") != expected_alpha
            ):
                raise RuntimeError(
                    "saved paired output configuration does not match the intended protocol"
                )
        print("[resume] Validating completed paired outputs; no experiment rerun.", flush=True)
    else:
        print("[1/2] Running paired control: boundary correction disabled", flush=True)
        discover(control_args)
        print("[2/2] Running paired treatment: band=1.0, alpha=0.5", flush=True)
        discover(treatment_args)

    old_detail = load_json(OLD_DETAIL)
    new_control_detail = load_json(OUT_CONTROL / "discovery_detail.json")
    new_treatment_detail = load_json(OUT_TREATMENT / "discovery_detail.json")
    old_rejector = np.load(OLD_REJECTOR_SCORES)
    new_control_rejector = np.load(OUT_CONTROL / "feature_rejector_scores.npz")
    new_treatment_rejector = np.load(OUT_TREATMENT / "feature_rejector_scores.npz")
    old_cal = np.load(SOURCE / "calibration_scores.npz") if (SOURCE / "calibration_scores.npz").is_file() else None
    new_control_cal = np.load(OUT_CONTROL / "calibration_scores.npz")
    new_treatment_cal = np.load(OUT_TREATMENT / "calibration_scores.npz")

    old_test_scores = np.asarray(old_detail["raw_score"], dtype=np.float64)
    control_test_scores = np.asarray(new_control_detail["raw_score"], dtype=np.float64)
    treatment_test_scores = np.asarray(new_treatment_detail["raw_score"], dtype=np.float64)
    if not (old_test_scores.shape == control_test_scores.shape == treatment_test_scores.shape):
        raise RuntimeError(
            "test-score arrays are not aligned: "
            f"old={old_test_scores.shape}, control={control_test_scores.shape}, "
            f"treatment={treatment_test_scores.shape}"
        )
    old_test_indices = np.asarray(old_rejector["test_source_index"], dtype=np.int64)
    control_test_indices = np.asarray(
        new_control_rejector["test_source_index"], dtype=np.int64
    )
    treatment_test_indices = np.asarray(
        new_treatment_rejector["test_source_index"], dtype=np.int64
    )
    old_validation_indices = np.asarray(
        old_rejector["validation_source_index"], dtype=np.int64
    )
    old_validation_rejector_scores = np.asarray(
        old_rejector["validation_feature_rejector_score"], dtype=np.float64
    )
    control_calibration_indices = np.asarray(
        new_control_cal["source_index"], dtype=np.int64
    )
    treatment_calibration_indices = np.asarray(
        new_treatment_cal["source_index"], dtype=np.int64
    )
    if not (
        len(old_test_indices) == len(old_test_scores)
        and len(control_test_indices) == len(control_test_scores)
        and len(treatment_test_indices) == len(treatment_test_scores)
        and len(old_validation_indices)
        == len(new_control_rejector["validation_source_index"])
        and len(old_validation_rejector_scores)
        == len(new_control_rejector["validation_feature_rejector_score"])
    ):
        raise RuntimeError("saved sample indices do not align with score arrays")

    checks = {
        "max_abs_test_raw_score_diff": float(np.max(np.abs(
            np.asarray(new_control_detail["raw_score"]) - np.asarray(new_treatment_detail["raw_score"])
        ))),
        "max_abs_calibration_raw_score_diff": float(np.max(np.abs(
            new_control_cal["raw_score"] - new_treatment_cal["raw_score"]
        ))),
        "test_indices_equal": bool(np.array_equal(
            control_test_indices, treatment_test_indices
        )),
        "calibration_indices_equal": bool(np.array_equal(
            control_calibration_indices, treatment_calibration_indices
        )),
        "old_vs_rerun_test_indices_equal": bool(
            np.array_equal(old_test_indices, control_test_indices)
        ),
        "old_vs_rerun_validation_indices_equal": bool(
            np.array_equal(
                old_validation_indices,
                new_control_rejector["validation_source_index"],
            )
        ),
        "test_labels_equal": bool(
            new_control_detail["true_label"] == new_treatment_detail["true_label"]
            and new_control_detail["true_known"] == new_treatment_detail["true_known"]
        ),
        "max_abs_old_vs_rerun_raw_score_diff": float(
            np.max(np.abs(old_test_scores - control_test_scores))
        ),
        "max_abs_old_vs_rerun_rejector_score_diff": float(np.max(np.abs(
            old_rejector["test_feature_rejector_score"]
            - new_control_rejector["test_feature_rejector_score"]
        ))),
        "max_abs_old_vs_rerun_validation_score_diff": float(np.max(np.abs(
            old_validation_rejector_scores
            - new_control_rejector["validation_feature_rejector_score"]
        ))),
    }
    if checks["max_abs_test_raw_score_diff"] > 1e-8:
        raise RuntimeError(f"paired raw test scores differ: {checks}")
    if checks["max_abs_calibration_raw_score_diff"] > 1e-8:
        raise RuntimeError(f"paired raw calibration scores differ: {checks}")
    if not all(
        checks[key]
        for key in (
            "test_indices_equal",
            "calibration_indices_equal",
            "old_vs_rerun_test_indices_equal",
            "old_vs_rerun_validation_indices_equal",
            "test_labels_equal",
        )
    ):
        raise RuntimeError(f"paired sample alignment failed: {checks}")
    if sha256(TRAIN / "student.pt") != manifest["checkpoint_sha256"]:
        raise RuntimeError("student checkpoint changed during the paired discovery run")
    if sha256(SPLIT) != manifest["split_sha256"]:
        raise RuntimeError("class split changed during the paired discovery run")

    write_record(
        manifest,
        load_json(OUT_CONTROL / "discovery_report.json"),
        load_json(OUT_TREATMENT / "discovery_report.json"),
        checks,
    )
    print(f"[saved] {RECORD_MD}", flush=True)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
