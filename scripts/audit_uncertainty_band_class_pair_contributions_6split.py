"""Post-hoc class-pair attribution for six frozen uncertainty-band runs.

This is a descriptive analysis only. It reads finalized test scores and labels;
it does not train, calibrate, select parameters, or alter any prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2121, 2122, 2123, 2124, 2125, 2127)
OUT_JSON = ROOT / "analysis" / "uncertainty_band_class_pair_contributions_6split.json"
OUT_MD = ROOT / "analysis" / "uncertainty_band_class_pair_contributions_6split.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def pair_auc(unknown_scores: np.ndarray, known_scores: np.ndarray) -> float:
    """P(unknown score > known score), awarding half credit for ties."""
    greater = unknown_scores[:, None] > known_scores[None, :]
    equal = unknown_scores[:, None] == known_scores[None, :]
    return float(np.mean(greater) + 0.5 * np.mean(equal))


def top_share(values: list[dict[str, Any]], count: int, value_key: str) -> float:
    positive_total = sum(max(0.0, float(row[value_key])) for row in values)
    if positive_total == 0.0:
        return 0.0
    top = sorted(values, key=lambda row: float(row[value_key]), reverse=True)[:count]
    return sum(max(0.0, float(row[value_key])) for row in top) / positive_total


def summarize_pair_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["weighted_contribution"] > 1e-15]
    negative = [row for row in rows if row["weighted_contribution"] < -1e-15]
    return {
        "pair_count": len(rows),
        "positive_pairs": len(positive),
        "negative_pairs": len(negative),
        "zero_or_negligible_pairs": len(rows) - len(positive) - len(negative),
        "mean_pair_auc_control": float(np.mean([row["auc_control"] for row in rows])),
        "mean_pair_auc_treatment": float(np.mean([row["auc_treatment"] for row in rows])),
        "mean_pair_auc_delta": float(np.mean([row["auc_delta"] for row in rows])),
        "weighted_contribution": float(sum(row["weighted_contribution"] for row in rows)),
        "gross_positive_contribution": float(sum(row["weighted_contribution"] for row in positive)),
        "gross_negative_contribution": float(sum(row["weighted_contribution"] for row in negative)),
    }


def analyze_split(seed: int) -> dict[str, Any]:
    split_path = ROOT / "splits" / "semantic_hard" / f"seed{seed}" / "cifar100_60_40_semantic_hard.json"
    baseline_path = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control_discover" / "discovery_detail.json"
    treatment_path = ROOT / "runs" / f"rejection_uncertainty_band_full_s{seed}" / "discovery_detail.json"
    split = read_json(split_path)
    baseline = read_json(baseline_path)
    treatment = read_json(treatment_path)

    known_classes = set(map(int, split["known_classes"]))
    fine_names = split["fine_class_names"]
    fine_to_coarse = {int(fine_id): int(coarse_id) for fine_id, coarse_id in split["fine_to_coarse"].items()}
    coarse_names = split["coarse_class_names"]
    require(len(known_classes) == 60 and len(fine_names) == 100, f"seed {seed}: malformed split metadata")
    require(set(fine_to_coarse) == set(range(100)), f"seed {seed}: incomplete fine-to-coarse mapping")
    require(len(coarse_names) == 20, f"seed {seed}: expected 20 CIFAR-100 coarse names")
    require(set(fine_to_coarse.values()) == set(range(20)), f"seed {seed}: expected CIFAR-100's 20 coarse labels")
    unknown_classes = set(range(100)) - known_classes
    fine_classes_by_coarse = {
        coarse_id: {fine_id for fine_id, mapped_coarse in fine_to_coarse.items() if mapped_coarse == coarse_id}
        for coarse_id in range(20)
    }
    require(all(len(classes) == 5 for classes in fine_classes_by_coarse.values()),
            f"seed {seed}: each CIFAR-100 coarse label must map to five fine labels")
    for coarse_id, fine_classes in fine_classes_by_coarse.items():
        known_in_coarse = len(fine_classes & known_classes)
        unknown_in_coarse = len(fine_classes & unknown_classes)
        require((known_in_coarse, unknown_in_coarse) == (3, 2),
                f"seed {seed}: coarse label {coarse_id} has {known_in_coarse} known and {unknown_in_coarse} unknown labels")

    baseline_raw = np.asarray(baseline["raw_labels"], dtype=np.int64)
    treatment_raw = np.asarray(treatment["raw_labels"], dtype=np.int64)
    baseline_known = np.asarray(baseline["true_known"], dtype=bool)
    treatment_known = np.asarray(treatment["true_known"], dtype=bool)
    control_score = np.asarray(baseline["score"], dtype=np.float64)
    treatment_score = np.asarray(treatment["score"], dtype=np.float64)
    treatment_raw_score = np.asarray(treatment["raw_score"], dtype=np.float64)

    lengths = {len(x) for x in (baseline_raw, treatment_raw, baseline_known, treatment_known,
                                control_score, treatment_score, treatment_raw_score)}
    require(len(lengths) == 1, f"seed {seed}: per-sample fields have different lengths")
    require(np.array_equal(baseline_raw, treatment_raw), f"seed {seed}: raw labels are not aligned")
    require(np.array_equal(baseline_known, treatment_known), f"seed {seed}: known masks are not aligned")
    require(np.array_equal(control_score, treatment_raw_score), f"seed {seed}: baseline score differs from treatment raw score")
    require(np.all(np.isfinite(control_score)) and np.all(np.isfinite(treatment_score)),
            f"seed {seed}: non-finite scores")

    expected_known = np.isin(baseline_raw, list(known_classes))
    require(np.array_equal(expected_known, baseline_known), f"seed {seed}: raw CIFAR labels disagree with protocol mask")
    observed = set(map(int, np.unique(baseline_raw)))
    require(observed == set(range(100)), f"seed {seed}: test data does not contain all 100 fine classes")

    class_counts = {int(label): int(np.sum(baseline_raw == label)) for label in observed}
    require(set(class_counts.values()) == {100}, f"seed {seed}: expected 100 test images per fine class")

    pair_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    total_pairs = 0
    weighted_delta_sum = 0.0
    for unknown_id in sorted(unknown_classes):
        u_mask = baseline_raw == unknown_id
        u_control = control_score[u_mask]
        u_treatment = treatment_score[u_mask]
        known_control_all = control_score[baseline_known]
        known_treatment_all = treatment_score[baseline_known]
        class_auc_control = pair_auc(u_control, known_control_all)
        class_auc_treatment = pair_auc(u_treatment, known_treatment_all)
        class_delta = class_auc_treatment - class_auc_control
        class_contribution = class_delta / len(unknown_classes)
        class_rows.append({
            "unknown_class_id": unknown_id,
            "unknown_class_name": fine_names[unknown_id],
            "auc_control": class_auc_control,
            "auc_treatment": class_auc_treatment,
            "auc_delta": class_delta,
            "weighted_contribution": class_contribution,
        })

        for known_id in sorted(known_classes):
            k_mask = baseline_raw == known_id
            control_cell = pair_auc(u_control, control_score[k_mask])
            treatment_cell = pair_auc(u_treatment, treatment_score[k_mask])
            delta = treatment_cell - control_cell
            contribution = delta / (len(unknown_classes) * len(known_classes))
            pair_rows.append({
                "unknown_class_id": unknown_id,
                "unknown_class_name": fine_names[unknown_id],
                "unknown_coarse_id": fine_to_coarse[unknown_id],
                "unknown_coarse_name": coarse_names[fine_to_coarse[unknown_id]],
                "known_class_id": known_id,
                "known_class_name": fine_names[known_id],
                "known_coarse_id": fine_to_coarse[known_id],
                "known_coarse_name": coarse_names[fine_to_coarse[known_id]],
                "coarse_pair_group": "same_coarse" if fine_to_coarse[unknown_id] == fine_to_coarse[known_id] else "cross_coarse",
                "auc_control": control_cell,
                "auc_treatment": treatment_cell,
                "auc_delta": delta,
                "weighted_contribution": contribution,
            })
            total_pairs += 100 * 100
            weighted_delta_sum += contribution

    labels = (~baseline_known).astype(np.int8)
    auc_control = float(roc_auc_score(labels, control_score))
    auc_treatment = float(roc_auc_score(labels, treatment_score))
    auc_delta = auc_treatment - auc_control
    require(abs(float(np.mean([row["auc_control"] for row in pair_rows])) - auc_control) <= 1e-12,
            f"seed {seed}: mean control class-pair AUC does not reconstruct overall AUROC")
    require(abs(float(np.mean([row["auc_treatment"] for row in pair_rows])) - auc_treatment) <= 1e-12,
            f"seed {seed}: mean treatment class-pair AUC does not reconstruct overall AUROC")
    require(abs(weighted_delta_sum - auc_delta) <= 1e-12,
            f"seed {seed}: pair contribution reconstruction mismatch ({weighted_delta_sum} vs {auc_delta})")

    coarse_group_rows = {
        group: [row for row in pair_rows if row["coarse_pair_group"] == group]
        for group in ("same_coarse", "cross_coarse")
    }
    coarse_pair_effects = {group: summarize_pair_group(rows) for group, rows in coarse_group_rows.items()}
    require(coarse_pair_effects["same_coarse"]["pair_count"] == 120,
            f"seed {seed}: expected 120 same-coarse known/unknown class pairs")
    require(coarse_pair_effects["cross_coarse"]["pair_count"] == 2280,
            f"seed {seed}: expected 2280 cross-coarse known/unknown class pairs")
    require(abs(sum(item["weighted_contribution"] for item in coarse_pair_effects.values()) - auc_delta) <= 1e-12,
            f"seed {seed}: coarse-group contributions do not reconstruct overall AUROC delta")

    positive_pair_rows = [row for row in pair_rows if row["weighted_contribution"] > 1e-15]
    negative_pair_rows = [row for row in pair_rows if row["weighted_contribution"] < -1e-15]
    positive_class_rows = [row for row in class_rows if row["weighted_contribution"] > 1e-15]
    negative_class_rows = [row for row in class_rows if row["weighted_contribution"] < -1e-15]
    return {
        "seed": seed,
        "audit": {
            "sample_count": int(len(baseline_raw)),
            "known_count": int(baseline_known.sum()),
            "unknown_count": int((~baseline_known).sum()),
            "known_classes": len(known_classes),
            "unknown_classes": len(unknown_classes),
            "samples_per_fine_class": 100,
            "labels_aligned": True,
            "split_membership_matches_known_mask": True,
            "baseline_score_matches_treatment_raw_score": True,
            "pair_contribution_reconstructs_auc_delta": True,
            "coarse_group_contributions_reconstruct_auc_delta": True,
            "unknown_known_class_pairs": len(pair_rows),
            "image_pairs": total_pairs,
            "coarse_group_counts": {group: item["pair_count"] for group, item in coarse_pair_effects.items()},
        },
        "auc_control": auc_control,
        "auc_treatment": auc_treatment,
        "auc_delta": auc_delta,
        "pair_contribution_sum": weighted_delta_sum,
        "pair_contribution_reconstruction_error": weighted_delta_sum - auc_delta,
        "coarse_pair_effect_distribution": coarse_pair_effects,
        "pair_effect_distribution": {
            "positive_pairs": len(positive_pair_rows),
            "negative_pairs": len(negative_pair_rows),
            "zero_or_negligible_pairs": len(pair_rows) - len(positive_pair_rows) - len(negative_pair_rows),
            "gross_positive_contribution": sum(row["weighted_contribution"] for row in positive_pair_rows),
            "gross_negative_contribution": sum(row["weighted_contribution"] for row in negative_pair_rows),
            "top1_positive_share": top_share(pair_rows, 1, "weighted_contribution"),
            "top5_positive_share": top_share(pair_rows, 5, "weighted_contribution"),
            "top10_positive_share": top_share(pair_rows, 10, "weighted_contribution"),
            "top_positive_pairs": sorted(pair_rows, key=lambda row: row["weighted_contribution"], reverse=True)[:10],
            "top_negative_pairs": sorted(pair_rows, key=lambda row: row["weighted_contribution"])[:10],
        },
        "unknown_class_effect_distribution": {
            "positive_classes": len(positive_class_rows),
            "negative_classes": len(negative_class_rows),
            "zero_or_negligible_classes": len(class_rows) - len(positive_class_rows) - len(negative_class_rows),
            "gross_positive_contribution": sum(row["weighted_contribution"] for row in positive_class_rows),
            "gross_negative_contribution": sum(row["weighted_contribution"] for row in negative_class_rows),
            "top1_positive_share": top_share(class_rows, 1, "weighted_contribution"),
            "top5_positive_share": top_share(class_rows, 5, "weighted_contribution"),
            "top10_positive_share": top_share(class_rows, 10, "weighted_contribution"),
            "top_positive_classes": sorted(class_rows, key=lambda row: row["weighted_contribution"], reverse=True)[:10],
            "top_negative_classes": sorted(class_rows, key=lambda row: row["weighted_contribution"])[:10],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    rows = result["splits"]
    out = [
        "# 六组冻结结果的类别对 AUROC 贡献分析",
        "",
        "## 分析范围与审计口径",
        "",
        "本分析读取六组已完成的 baseline / uncertainty-band 测试结果，不重新训练、校准、选择阈值或调整方法。未知/已知身份只用于最终测试指标和本次事后归因；归因结果不用于回头调参。官方细类别身份取自 `raw_labels`，并逐组核验其与协议 JSON 的已知类集合及 `true_known` 掩码一致。baseline 分数与 treatment 保存的原始分数逐项相同。",
        "",
        "每个划分含 40 个未知细类别和 60 个已知细类别，每类各 100 张测试图像。将 AUROC 展开为未知图像与已知图像的成对排序概率后，再按 40×60 个类别对分解；每个类别对权重相同。类别对加权贡献之和必须还原总体 AUROC 差值，否则脚本会报错。结果是描述性诊断，不作因果或统计显著性结论。",
        "",
        "## 汇总",
        "",
        "| 划分 | AUROC 对照 | AUROC 修正 | 差值 | 类别对正 / 负贡献合计 | 正贡献类别对 / 2400 | 前 5 个类别对占正贡献比例 | 正贡献未知类 / 40 | 前 5 个未知类占正贡献比例 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair = row["pair_effect_distribution"]
        cls = row["unknown_class_effect_distribution"]
        out.append(
            f"| {row['seed']} | {row['auc_control']:.6f} | {row['auc_treatment']:.6f} | {row['auc_delta']:+.6f} "
            f"| +{pair['gross_positive_contribution']:.5f} / {pair['gross_negative_contribution']:.5f} "
            f"| {pair['positive_pairs']} | {pair['top5_positive_share']:.1%} "
            f"| {cls['positive_classes']} | {cls['top5_positive_share']:.1%} |"
        )

    out.extend(["", "## 各划分贡献最大的未知类别", ""])
    for row in rows:
        top = row["unknown_class_effect_distribution"]["top_positive_classes"][:5]
        rendered = "；".join(
            f"{item['unknown_class_name']}（加权贡献 {item['weighted_contribution']:+.6f}）"
            for item in top
        )
        out.append(f"- s{row['seed']}：{rendered}")

    out.extend(["", "## 各划分贡献最大的类别对", ""])
    for row in rows:
        top = row["pair_effect_distribution"]["top_positive_pairs"][:5]
        rendered = "；".join(
            f"未知 {item['unknown_class_name']}—已知 {item['known_class_name']}（{item['weighted_contribution']:+.6f}）"
            for item in top
        )
        out.append(f"- s{row['seed']}：{rendered}")

    out.extend([
        "",
        "## 按 CIFAR-100 粗类别关系分层",
        "",
        "粗类别指 CIFAR-100 官方提供的 20 个上位类别标签；‘同粗类别’表示一对已知细类别与未知细类别共享该上位标签，‘跨粗类别’表示两者上位标签不同。这是类别标签层面的描述性分层，不代表所有同粗类别图像都具有相同视觉相似度。每组贡献按其类别对数量占全部 2,400 对的比例加权，因此两组贡献相加应等于该划分的总体 AUROC 差值。",
        "",
        "| 划分 | 同粗类别对数 | 同粗类别单对平均 AUROC 变化 | 同粗类别加权贡献 | 跨粗类别对数 | 跨粗类别单对平均 AUROC 变化 | 跨粗类别加权贡献 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        same = row["coarse_pair_effect_distribution"]["same_coarse"]
        cross = row["coarse_pair_effect_distribution"]["cross_coarse"]
        out.append(
            f"| {row['seed']} | {same['pair_count']} | {same['mean_pair_auc_delta']:+.6f} | {same['weighted_contribution']:+.6f} "
            f"| {cross['pair_count']} | {cross['mean_pair_auc_delta']:+.6f} | {cross['weighted_contribution']:+.6f} |"
        )
    for group, label in (("same_coarse", "同粗类别"), ("cross_coarse", "跨粗类别")):
        mean_delta = float(np.mean([row["coarse_pair_effect_distribution"][group]["mean_pair_auc_delta"] for row in rows]))
        mean_contribution = float(np.mean([row["coarse_pair_effect_distribution"][group]["weighted_contribution"] for row in rows]))
        out.append(f"- 六组平均：{label}类别对单对 AUROC 变化 `{mean_delta:+.6f}`；对总体 AUROC 变化的平均加权贡献 `{mean_contribution:+.6f}`。")

    out.extend([
        "",
        "## 解读边界",
        "",
        "前五贡献比例越高，说明正向 AUROC 贡献越集中；但负贡献类别对会抵消正贡献，故还需结合正、负项数量和净 AUROC 差值理解。类别对贡献回答的是‘哪些已知/未知类别组合与观测到的排序变化同时出现’，不能证明修正机制对特定类别产生因果作用。六组类别划分存在重叠，也不是六个独立数据集；跨划分汇总只作描述。该分析不构成调参依据，若据此提出类别定向方法，必须另用未参与本次归因的类别划分进行预先设定的验证。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts/audit_uncertainty_band_class_pair_contributions_6split.py",
        "```",
        "",
        f"机器可读结果：`{OUT_JSON.relative_to(ROOT).as_posix()}`。",
        "",
    ])
    return "\n".join(out)


def main() -> None:
    splits = [analyze_split(seed) for seed in SEEDS]
    auc_deltas = [row["auc_delta"] for row in splits]
    result = {
        "question": "Are frozen AUROC gains distributed across unknown-known fine-class pairs or concentrated in a few pairs/classes?",
        "protocol": "post-hoc descriptive attribution; no training, calibration, threshold selection, or tuning",
        "seeds": list(SEEDS),
        "summary": {
            "split_count": len(splits),
            "mean_auc_delta": float(np.mean(auc_deltas)),
            "min_auc_delta": float(np.min(auc_deltas)),
            "max_auc_delta": float(np.max(auc_deltas)),
            "mean_pair_gross_positive_contribution": float(np.mean([
                row["pair_effect_distribution"]["gross_positive_contribution"] for row in splits
            ])),
            "mean_pair_gross_negative_contribution": float(np.mean([
                row["pair_effect_distribution"]["gross_negative_contribution"] for row in splits
            ])),
            "mean_same_coarse_pair_auc_delta": float(np.mean([
                row["coarse_pair_effect_distribution"]["same_coarse"]["mean_pair_auc_delta"] for row in splits
            ])),
            "mean_cross_coarse_pair_auc_delta": float(np.mean([
                row["coarse_pair_effect_distribution"]["cross_coarse"]["mean_pair_auc_delta"] for row in splits
            ])),
            "mean_same_coarse_weighted_contribution": float(np.mean([
                row["coarse_pair_effect_distribution"]["same_coarse"]["weighted_contribution"] for row in splits
            ])),
            "mean_cross_coarse_weighted_contribution": float(np.mean([
                row["coarse_pair_effect_distribution"]["cross_coarse"]["weighted_contribution"] for row in splits
            ])),
            "all_pair_contributions_reconstruct_auc_delta": all(
                row["audit"]["pair_contribution_reconstructs_auc_delta"] for row in splits
            ),
            "all_coarse_group_contributions_reconstruct_auc_delta": all(
                row["audit"]["coarse_group_contributions_reconstruct_auc_delta"] for row in splits
            ),
        },
        "splits": splits,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Mean AUROC delta: {result['summary']['mean_auc_delta']:+.6f}")
    for row in splits:
        pair = row["pair_effect_distribution"]
        cls = row["unknown_class_effect_distribution"]
        print(
            f"s{row['seed']}: delta={row['auc_delta']:+.6f}, "
            f"positive pairs={pair['positive_pairs']}/2400, top5-pair share={pair['top5_positive_share']:.1%}, "
            f"positive novel classes={cls['positive_classes']}/40, top5-class share={cls['top5_positive_share']:.1%}"
        )


if __name__ == "__main__":
    main()
