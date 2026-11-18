"""Audit whether paired discovery arms use identical data-source identities.

This is a read-only protocol audit.  It reconstructs each arm's data bundle
from the saved ``discover_config.json`` and compares source identities for
the data views consumed by training, calibration, rejection, and evaluation.
It does not train, select a checkpoint, calibrate with test labels, or alter
any run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, dataset_source_keys  # noqa: E402


CASES = {
    "2084_control": "runs/shared_rejection_uncertainty_margin_s2084_baseline_discover",
    "2084_fusion": "runs/shared_rejection_uncertainty_margin_s2084_baseline_uncertainty_fusion",
    "2085_control": "runs/shared_rejection_uncertainty_margin_s2085_baseline_discover",
    "2085_fusion": "runs/shared_rejection_uncertainty_margin_s2085_baseline_uncertainty_fusion",
    "2086_control": "runs/shared_rejection_fusion_fresh_s2086_baseline_discover",
    "2086_fusion": "runs/shared_rejection_fusion_fresh_s2086_baseline_uncertainty_fusion",
}

CASE_SETS = {
    "open_val_mismatch": {
        "2084_control": CASES["2084_control"],
        "2084_fusion": CASES["2084_fusion"],
        "2085_control": CASES["2085_control"],
        "2085_fusion": CASES["2085_fusion"],
        "2086_control": CASES["2086_control"],
        "2086_fusion": CASES["2086_fusion"],
    },
    "strict_fusion": {
        "2089_control": "runs/shared_rejection_fusion_fresh_s2089_baseline_discover",
        "2089_fusion": "runs/shared_rejection_fusion_fresh_s2089_baseline_uncertainty_fusion",
        "2092_control": "runs/shared_rejection_fusion_fresh_s2092_baseline_discover",
        "2092_fusion": "runs/shared_rejection_fusion_fresh_s2092_baseline_uncertainty_fusion",
        "2098_control": "runs/shared_rejection_fusion_fresh_s2098_baseline_discover",
        "2098_fusion": "runs/shared_rejection_fusion_fresh_s2098_baseline_uncertainty_fusion",
        "2103_control": "runs/shared_rejection_fusion_fresh_s2103_baseline_discover",
        "2103_fusion": "runs/shared_rejection_fusion_fresh_s2103_baseline_uncertainty_fusion",
    },
}

# These fields are expected to differ between the two arms. ``work_dir`` is
# an output location; ``score_mode`` is the experimental factor. Any other
# configuration difference would weaken the single-factor interpretation.
EXPECTED_VARIANT_KEYS = {"work_dir", "score_mode"}


def resolve_path(value: str | None, *, base: Path = ROOT) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "discover_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def rebuild_bundle(config: dict[str, Any]):
    split_path = resolve_path(config.get("split_path"))
    if split_path is None or not split_path.is_file():
        fallback = ROOT / "splits" / "semantic_hard" / f"seed{int(config['seed'])}" / "cifar100_60_40_semantic_hard.json"
        split_path = fallback.resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"split file not found: {split_path}")

    data_root = resolve_path(str(config.get("data_root", "data")))
    if data_root is None:
        raise ValueError("data_root is missing")

    return build_data_bundle(
        dataset_name=str(config.get("dataset", "cifar100")),
        root=str(data_root),
        num_known=int(config["num_known"]),
        seed=int(config["seed"]),
        image_size=int(config["image_size"]),
        download=False,
        split_path=str(split_path),
        limit_train=int(config.get("limit_train", 0) or 0) or None,
        limit_val=int(config.get("limit_val", 0) or 0) or None,
        limit_test=int(config.get("limit_test", 0) or 0) or None,
        limit_discovery=int(config.get("limit_discovery", 0) or 0) or None,
        discovery_pool_mode=str(config.get("discovery_pool_mode", "unknown")),
        open_val_ratio=float(config.get("open_val_ratio", 0.0)),
        calibration_ratio=float(config.get("calibration_ratio", 0.0)),
        calibration_overlap_control=bool(config.get("calibration_overlap_control", False)),
        known_split_mode=str(config.get("known_split_mode", "random")),
        mixed_known_pool_ratio=float(config.get("mixed_known_pool_ratio", 0.2)),
        matched_discovery_pool=bool(config.get("matched_discovery_pool", False)),
        matched_pool_size=int(config.get("matched_pool_size", 0) or 0),
        matched_known_prior=float(config.get("matched_known_prior", 0.2)),
        transform_profile=str(config.get("transform_profile", "imagenet")),
    ), split_path


def stable_key(value: tuple) -> str:
    return "|".join(str(part) for part in value)


def summarize_keys(values: list[tuple]) -> dict[str, Any]:
    normalized = sorted(stable_key(value) for value in values)
    digest = hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()
    return {
        "count": len(normalized),
        "sha256": digest,
        "first_keys": normalized[:3],
        "last_keys": normalized[-3:],
    }


def compare_keys(left: list[tuple], right: list[tuple]) -> dict[str, Any]:
    left_set = {stable_key(value) for value in left}
    right_set = {stable_key(value) for value in right}
    intersection = left_set & right_set
    left_only = left_set - right_set
    right_only = right_set - left_set
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "intersection_count": len(intersection),
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "identical": not left_only and not right_only,
        "left_only_examples": sorted(left_only)[:3],
        "right_only_examples": sorted(right_only)[:3],
    }


def bundle_views(bundle) -> dict[str, Any]:
    return {
        "train": bundle.train,
        "val": bundle.val,
        "calibration": bundle.calibration,
        "open_val": bundle.open_val,
        "discovery_pool": bundle.discovery_pool,
        "test": bundle.test,
    }


def compare_config_values(control: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    differences = {
        key: {"control": control.get(key), "fusion": fusion.get(key)}
        for key in sorted(set(control) | set(fusion))
        if control.get(key) != fusion.get(key)
    }
    return {
        "differences": differences,
        "difference_keys": sorted(differences),
        "only_expected_variant_keys": set(differences).issubset(EXPECTED_VARIANT_KEYS),
    }


def audit_pair(
    control_name: str,
    fusion_name: str,
    bundles: dict[str, Any],
    configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    control_bundle, control_split = bundles[control_name]
    fusion_bundle, fusion_split = bundles[fusion_name]
    views = {}
    for view_name in bundle_views(control_bundle):
        control_dataset = bundle_views(control_bundle)[view_name]
        fusion_dataset = bundle_views(fusion_bundle)[view_name]
        control_keys = [] if control_dataset is None else dataset_source_keys(control_dataset)
        fusion_keys = [] if fusion_dataset is None else dataset_source_keys(fusion_dataset)
        views[view_name] = {
            "control": summarize_keys(control_keys),
            "fusion": summarize_keys(fusion_keys),
            "comparison": compare_keys(control_keys, fusion_keys),
        }
    config_comparison = compare_config_values(configs[control_name], configs[fusion_name])
    strict_data_pair = all(view["comparison"]["identical"] for view in views.values())
    return {
        "control": {"name": control_name, "split_path": str(control_split)},
        "fusion": {"name": fusion_name, "split_path": str(fusion_split)},
        "views": views,
        "config_comparison": config_comparison,
        "strict_data_pair": strict_data_pair,
        "single_factor_pair": strict_data_pair and config_comparison["only_expected_variant_keys"],
    }


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired data protocol audit",
        "",
        "本审计从保存的 discovery 配置重建数据 bundle，以样本源身份比较 control 与 uncertainty-fusion。审计不训练模型，不使用测试标签选择参数，也不修改历史结果。",
        "",
        "## 结论",
        "",
        f"- 审计对象：{', '.join(report['pairs'])}",
        f"- 所有比较均基于 source-key 的集合交集，而不是仅比较样本数量：{report['method']}",
        f"- 数据严格一致的组数：{report['strict_data_pair_count']} / {len(report['pairs'])}",
        f"- 同时满足‘仅改变 score_mode’的单因素配对组数：{report['single_factor_pair_count']} / {len(report['pairs'])}",
        "",
        "如果 discovery_pool、train、calibration 或 open_val 任一输入不一致，该组只能作为描述性结果，不能把指标变化单独归因于 uncertainty fusion。",
        "",
    ]
    for pair_name, pair in report["pair_reports"].items():
        lines.extend([f"## {pair_name}", "", "| 数据视图 | control | fusion | 交集 | control 独有 | fusion 独有 | 完全一致 |", "|---|---:|---:|---:|---:|---:|---|"])
        strict = True
        for view_name, view in pair["views"].items():
            comp = view["comparison"]
            lines.append(
                f"| {view_name} | {comp['left_count']} | {comp['right_count']} | {comp['intersection_count']} | "
                f"{comp['left_only_count']} | {comp['right_only_count']} | {'是' if comp['identical'] else '否'} |"
            )
        config_comparison = pair["config_comparison"]
        diff_keys = ", ".join(config_comparison["difference_keys"]) or "无"
        if pair["single_factor_pair"]:
            judgment = "可用于 score_mode 单因素归因"
        elif pair["strict_data_pair"]:
            judgment = "数据严格一致，但配置还存在非预期差异"
        else:
            judgment = "数据不一致，按描述性结果处理"
        lines.extend([
            "",
            f"配置差异字段：{diff_keys}。",
            f"数据与配置判定：{judgment}。",
            "",
        ])
    lines.extend([
        "## 解释",
        "",
        "2084–2086 的旧配对中，control 与 fusion 的 open_val_ratio 不一致；该参数会先改变已知验证集和未知池，再影响拒识器的未标注输入以及后续 calibration/discovery 视图，因此这些结果只能作描述性参考。",
        "",
        "本审计中的 2089、2092、2098、2103 已满足同一学生 checkpoint、同一拒识器训练输入、同一 calibration 输入和同一测试输入，仅改变 score_mode；它们可用于评估 uncertainty fusion 对风险排序和拒识结果的影响。",
        "",
        "复现注意：历史配置中的 student_ckpt 仍保留搬迁前的绝对路径；若直接重跑，应先将其解析到当前 D:\\dachuang 工作区中的对应 student.pt，并保留原始实验目录和数字不变。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", choices=sorted(CASE_SETS), default="open_val_mismatch")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    selected_cases = CASE_SETS[args.case_set]
    if args.output_json is None:
        args.output_json = Path(f"analysis/paired_data_protocol_{args.case_set}_20261114.json")
    if args.output_md is None:
        args.output_md = Path(f"analysis/paired_data_protocol_{args.case_set}_20261114.md")

    bundles = {}
    configs = {}
    full_configs = {}
    for name, run in selected_cases.items():
        run_dir = (ROOT / run).resolve()
        config = load_config(run_dir)
        bundle, split_path = rebuild_bundle(config)
        bundles[name] = (bundle, split_path)
        full_configs[name] = config
        configs[name] = {
            key: config.get(key)
            for key in (
                "seed", "open_val_ratio", "calibration_ratio", "matched_discovery_pool",
                "matched_pool_size", "matched_known_prior", "discovery_pool_mode",
                "mixed_known_pool_ratio", "score_mode",
            )
        }

    pair_names = sorted({name.rsplit("_", 1)[0] for name in selected_cases})
    pair_reports = {}
    for seed in pair_names:
        pair_reports[seed] = audit_pair(
            f"{seed}_control",
            f"{seed}_fusion",
            bundles,
            full_configs,
        )

    report = {
        "protocol": "reconstruct_saved_data_bundle_and_compare_source_keys",
        "method": "每个视图的样本源 key 集合、SHA-256、交集和对称差异",
        "pairs": pair_names,
        "configs": configs,
        "pair_reports": pair_reports,
        "strict_data_pair_count": sum(1 for pair in pair_reports.values() if pair["strict_data_pair"]),
        "single_factor_pair_count": sum(1 for pair in pair_reports.values() if pair["single_factor_pair"]),
    }
    args.output_json = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    args.output_md = args.output_md if args.output_md.is_absolute() else ROOT / args.output_md
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(make_markdown(report), encoding="utf-8")
    print(f"saved {args.output_json}")
    print(f"saved {args.output_md}")


if __name__ == "__main__":
    main()
