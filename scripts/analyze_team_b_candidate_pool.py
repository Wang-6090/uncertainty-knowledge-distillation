"""Candidate-pool error decomposition for the team-B rejection line.

Inputs are the artifacts already written by ``train.py discover``:

* ``discovery_detail.json``  -- per-sample score, pred_known, raw label, prediction
* ``calibration_report.json`` -- calibrated threshold and policy
* ``discovery_report.json``   -- aggregate detection / candidate / cluster metrics
* ``calibration_scores.npz``  -- known-validation scores (optional, needed for the
  near-threshold band)

The script answers section 6.2 and 6.5 of ``docs/parallel_work_division.md``:

*风险分数分布与已知/未知重叠、阈值、已知接受率、未知拒识率、候选数量、
候选池未知召回/纯度/污染、阈值附近样本在不同种子下的候选标记是否变化、
以及按类别的候选数量与污染统计。

Unknown labels are used only for retrospective evaluation.  They never feed
threshold selection or sample screening, and no artifact written here is read
back by any training or calibration step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINE_NAMES = REPO_ROOT / "splits" / "protocols" / "cifar100_60_40_random.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fine_class_names(split: dict, explicit: Path | None) -> list[str] | None:
    """Return the CIFAR-100 index -> class-name table, or None if unavailable."""
    if explicit is not None:
        names = _read_json(explicit).get("fine_class_names")
        return [str(value) for value in names] if names else None
    if split.get("fine_class_names"):
        return [str(value) for value in split["fine_class_names"]]
    if DEFAULT_FINE_NAMES.is_file():
        names = _read_json(DEFAULT_FINE_NAMES).get("fine_class_names")
        if names:
            return [str(value) for value in names]
    return None


def _class_indices(entries: list, fine_names: list[str] | None) -> list[int]:
    """Map a split class list (names or raw indices) to raw CIFAR-100 indices."""
    resolved: list[int] = []
    for entry in entries:
        if isinstance(entry, str):
            if fine_names is None:
                raise ValueError(
                    "split lists class names but no fine_class_names table is available; "
                    "pass --fine-names"
                )
            resolved.append(fine_names.index(entry))
        else:
            resolved.append(int(entry))
    return resolved


def _per_sample_threshold(threshold, pred_class: np.ndarray) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0:
        return np.full(pred_class.shape, float(values), dtype=np.float64)
    if values.ndim == 1 and values.size:
        return values[np.clip(pred_class, 0, values.size - 1)]
    raise ValueError("calibrated_threshold must be a scalar or a non-empty vector")


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _overlap(known: np.ndarray, unknown: np.ndarray) -> dict:
    """Non-parametric overlap diagnostics between known and unknown scores."""
    if known.size == 0 or unknown.size == 0:
        return {}
    all_scores = np.concatenate([known, unknown])
    bins = np.linspace(float(all_scores.min()), float(all_scores.max()), 51)
    known_hist, _ = np.histogram(known, bins=bins, density=True)
    unknown_hist, _ = np.histogram(unknown, bins=bins, density=True)
    width = bins[1] - bins[0]
    overlap_area = float(np.sum(np.minimum(known_hist, unknown_hist) * width))
    return {
        "overlap_area_50bins": overlap_area,
        "known_median": float(np.median(known)),
        "unknown_median": float(np.median(unknown)),
        "median_gap": float(np.median(unknown) - np.median(known)),
    }


# --------------------------------------------------------------------------- #
# per-run analysis
# --------------------------------------------------------------------------- #
def analyse_run(run_dir: Path, split: dict, fine_names: list[str] | None) -> dict:
    detail = _read_json(run_dir / "discovery_detail.json")
    report = _read_json(run_dir / "discovery_report.json")
    calibration = _read_json(run_dir / "calibration_report.json")

    score = np.asarray(detail["score"], dtype=np.float64)
    raw_labels = np.asarray(detail["raw_labels"], dtype=np.int64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    pred_class = np.asarray(detail["pred_class"], dtype=np.int64)
    pred_known = np.asarray(detail["pred_known"], dtype=bool)
    threshold_values = calibration["calibrated_threshold"]
    per_sample_threshold = _per_sample_threshold(threshold_values, pred_class)
    candidate = ~pred_known
    total = int(score.size)

    known_mask = true_known
    unknown_mask = ~true_known
    candidate_known = candidate & known_mask          # pollution (false rejects)
    candidate_unknown = candidate & unknown_mask      # true positives
    missed_unknown = (~candidate) & unknown_mask      # missed unknown
    accepted_known = (~candidate) & known_mask

    known_scores = score[known_mask]
    unknown_scores = score[unknown_mask]

    # Near-threshold band, defined on known-validation scores (rank based).
    band: dict = {"definition": None}
    calibration_path = run_dir / "calibration_scores.npz"
    band_mask = np.zeros(total, dtype=bool)
    if calibration_path.is_file():
        with np.load(calibration_path, allow_pickle=False) as payload:
            calibration_scores = np.asarray(payload["score"], dtype=np.float64)
        lo = float(np.quantile(calibration_scores, 0.90))
        hi = float(np.quantile(calibration_scores, 0.99))
        band_mask = (score >= lo) & (score <= hi)
        band = {
            "definition": (
                "samples whose open risk score falls between the q90 and q99 of the "
                "known-validation calibration scores (rank-based neighbourhood of the "
                "operating point)"
            ),
            "known_validation_q90": lo,
            "known_validation_q99": hi,
            "sample_count": int(band_mask.sum()),
            "known_count": int((band_mask & known_mask).sum()),
            "unknown_count": int((band_mask & unknown_mask).sum()),
            "candidate_count": int((band_mask & candidate).sum()),
        }

    # Per-class tables.
    known_raw = _class_indices(split["known_classes"], fine_names)
    novel_raw = _class_indices(split["novel_classes"], fine_names)
    used_known = sorted({int(value) for value in raw_labels[known_mask]})

    known_rows = []
    for raw_label in known_raw:
        mask = known_mask & (raw_labels == raw_label)
        if not mask.any():
            continue
        known_rows.append(
            {
                "raw_label": raw_label,
                "class_name": fine_names[raw_label] if fine_names else str(raw_label),
                "count": int(mask.sum()),
                "false_reject_count": int(candidate_known[mask].sum()),
                "false_reject_rate": float(candidate_known[mask].mean()),
                "classification_accuracy": float(
                    (pred_class[mask] == raw_labels[mask]).mean()
                ),
                "score_median": float(np.median(score[mask])),
            }
        )
    known_rows.sort(key=lambda row: (-row["false_reject_rate"], -row["count"]))

    novel_rows = []
    for raw_label in novel_raw:
        mask = unknown_mask & (raw_labels == raw_label)
        if not mask.any():
            continue
        novel_rows.append(
            {
                "raw_label": raw_label,
                "class_name": fine_names[raw_label] if fine_names else str(raw_label),
                "count": int(mask.sum()),
                "candidate_count": int(candidate_unknown[mask].sum()),
                "unknown_recall": float(candidate_unknown[mask].mean()),
                "missed_count": int(missed_unknown[mask].sum()),
                "score_median": float(np.median(score[mask])),
            }
        )
    novel_rows.sort(key=lambda row: (-row["unknown_recall"], -row["count"]))

    run_name = run_dir.name
    threshold_array = np.asarray(threshold_values, dtype=np.float64)
    threshold_record = (
        {"type": "global", "value": float(threshold_array)}
        if threshold_array.ndim == 0
        else {
            "type": "class_conditional",
            "count": int(threshold_array.size),
            "min": float(threshold_array.min()) if threshold_array.size else None,
            "median": float(np.median(threshold_array)) if threshold_array.size else None,
            "max": float(threshold_array.max()) if threshold_array.size else None,
        }
    )
    return {
        "run": run_name,
        "run_dir": str(run_dir),
        "score_mode": detail.get("score_mode"),
        "threshold": threshold_record,
        "threshold_policy": calibration.get("threshold_policy", {}).get("type"),
        "sample_count": total,
        "known_count": int(known_mask.sum()),
        "unknown_count": int(unknown_mask.sum()),
        "candidate_count": int(candidate.sum()),
        "candidate_known_count": int(candidate_known.sum()),
        "candidate_unknown_count": int(candidate_unknown.sum()),
        "known_accept_rate": float(accepted_known.sum() / max(known_mask.sum(), 1)),
        # "rejected" means pushed out of the known set, i.e. admitted to the
        # candidate pool; this matches discovery_report["unknown_reject_rate"].
        "unknown_reject_rate": float(
            candidate_unknown.sum() / max(unknown_mask.sum(), 1)
        ),
        "candidate_purity": float(
            candidate_unknown.sum() / max(candidate.sum(), 1)
        ),
        "known_pollution_rate": float(
            candidate_known.sum() / max(candidate.sum(), 1)
        ),
        "unknown_recall_in_pool": float(
            candidate_unknown.sum() / max(unknown_mask.sum(), 1)
        ),
        "missed_unknown_count": int(missed_unknown.sum()),
        "score_distribution": {
            "known": _distribution(known_scores),
            "unknown": _distribution(unknown_scores),
        },
        "known_unknown_overlap": _overlap(known_scores, unknown_scores),
        "near_threshold_band": band,
        "detection": {
            key: report.get(key)
            for key in ("auroc", "aupr", "fpr95", "oscr", "known_accept_rate",
                        "unknown_reject_rate", "threshold_type")
        },
        "clustering": {
            key: report.get(key)
            for key in ("cluster_candidate_count", "cluster_candidate_purity",
                        "cluster_candidate_unknown_recall", "cluster_true_unknown_count",
                        "cluster_false_reject_count", "cluster_k", "cluster_true_k",
                        "cluster_k_abs_error", "cluster_acc", "cluster_nmi", "cluster_ari",
                        "cluster_unknown_only_acc", "cluster_unknown_only_nmi",
                        "cluster_unknown_only_ari", "cluster_all_unknown_oracle_acc",
                        "cluster_all_unknown_oracle_nmi", "cluster_all_unknown_oracle_ari",
                        "cluster_candidate_unknown_oracle_acc",
                        "cluster_candidate_unknown_oracle_nmi",
                        "cluster_candidate_unknown_oracle_ari")
        },
        "known_classes": known_rows,
        "novel_classes": novel_rows,
        # kept in memory for the cross-run step, not serialised verbatim
        "_candidate": candidate,
        "_band": band_mask,
        "_true_known": known_mask,
        "_raw_labels": raw_labels,
        "_score": score,
        "_threshold": per_sample_threshold,
    }


# --------------------------------------------------------------------------- #
# cross-run analysis
# --------------------------------------------------------------------------- #
def cross_run(runs: list[dict]) -> dict:
    """Sample-level agreement of the candidate flag across seeds / configurations."""
    if len(runs) < 2:
        return {
            "comparison": None,
            "note": "at least two runs are needed for a cross-run stability analysis",
        }
    names = [run["run"] for run in runs]
    lengths = {len(run["_candidate"]) for run in runs}
    if len(lengths) != 1:
        raise ValueError(
            "cross-run analysis needs identical test-set ordering; got lengths "
            f"{sorted(lengths)}"
        )
    flags = np.stack([run["_candidate"] for run in runs], axis=0)
    true_known = runs[0]["_true_known"]
    raw_labels = runs[0]["_raw_labels"]
    band = runs[0]["_band"]
    n_runs = flags.shape[0]
    support = flags.sum(axis=0)
    unstable = (support > 0) & (support < n_runs)

    # distance of each sample's score to its own threshold, averaged over runs
    gap = np.stack(
        [np.abs(run["_score"] - run["_threshold"]) for run in runs], axis=0
    ).mean(axis=0)

    unstable_known = int((unstable & true_known).sum())
    unstable_unknown = int((unstable & ~true_known).sum())

    per_class = []
    for raw_label in np.unique(raw_labels):
        mask = raw_labels == raw_label
        if not mask.any():
            continue
        per_class.append(
            {
                "raw_label": int(raw_label),
                "is_known": bool(true_known[mask][0]),
                "count": int(mask.sum()),
                "unstable_count": int((unstable & mask).sum()),
                "unstable_rate": float((unstable & mask).sum() / max(mask.sum(), 1)),
                "candidate_rate_min": float(flags[:, mask].mean(axis=1).min()),
                "candidate_rate_max": float(flags[:, mask].mean(axis=1).max()),
            }
        )
    per_class.sort(key=lambda row: (-row["unstable_rate"], -row["count"]))

    return {
        "comparison": "sample-level candidate-flag stability across runs",
        "runs": names,
        "run_count": int(n_runs),
        "sample_count": int(flags.shape[1]),
        "always_candidate": int((support == n_runs).sum()),
        "never_candidate": int((support == 0).sum()),
        "unstable_count": int(unstable.sum()),
        "unstable_rate": float(unstable.mean()),
        "unstable_known_count": unstable_known,
        "unstable_unknown_count": unstable_unknown,
        "near_threshold_band": {
            "band_sample_count": int(band.sum()),
            "band_unstable_count": int((unstable & band).sum()),
            "band_unstable_rate": (
                float((unstable & band).sum() / band.sum()) if band.any() else None
            ),
            "outside_band_unstable_rate": (
                float((unstable & ~band).sum() / (~band).sum()) if (~band).any() else None
            ),
        },
        "mean_distance_to_threshold": {
            "stable": float(gap[~unstable].mean()) if (~unstable).any() else None,
            "unstable": float(gap[unstable].mean()) if unstable.any() else None,
        },
        "per_class": per_class,
        "by_run_candidate_counts": {
            name: int(flags[index].sum()) for index, name in enumerate(names)
        },
    }


# --------------------------------------------------------------------------- #
# error decomposition (section 6.5)
# --------------------------------------------------------------------------- #
def error_decomposition(run: dict, stability: dict | None) -> dict:
    unknown_total = run["unknown_count"]
    known_total = run["known_count"]
    missed = run["missed_unknown_count"]
    polluted = run["candidate_known_count"]
    candidate_unknown = run["candidate_unknown_count"]
    unstable = (stability or {}).get("unstable_count", 0)
    cluster = run["clustering"]
    decomposition = {
        "unknown_missed": {
            "count": missed,
            "rate_of_unknown": float(missed / max(unknown_total, 1)),
            "meaning": "unknown samples that never enter the candidate pool",
        },
        "known_pollution": {
            "count": polluted,
            "rate_of_known": float(polluted / max(known_total, 1)),
            "rate_of_pool": float(polluted / max(run["candidate_count"], 1)),
            "meaning": "known samples wrongly admitted into the candidate pool",
        },
        "cross_run_flag_instability": {
            "count": int(unstable),
            "rate": float(unstable / max(run["sample_count"], 1)) if unstable else 0.0,
            "meaning": (
                "samples whose candidate flag changes across seeds/configurations while "
                "the risk score itself keeps the same ordering"
            ),
        },
        "clustering_given_candidate_pool": {
            "candidate_unknown_samples": candidate_unknown,
            "cluster_acc": cluster.get("cluster_unknown_only_acc", cluster.get("cluster_acc")),
            "cluster_nmi": cluster.get("cluster_unknown_only_nmi", cluster.get("cluster_nmi")),
            "cluster_ari": cluster.get("cluster_unknown_only_ari", cluster.get("cluster_ari")),
            "meaning": (
                "cluster quality restricted to true-unknown candidates; low values mean "
                "the bottleneck is the candidate pool, not only the clustering stage"
            ),
        },
    }
    # Dominant-source heuristic: rank the three upstream losses by rate.
    upstream = {
        "unknown_missed": decomposition["unknown_missed"]["rate_of_unknown"],
        "known_pollution": decomposition["known_pollution"]["rate_of_known"],
        "flag_instability": decomposition["cross_run_flag_instability"]["rate"],
    }
    decomposition["dominant_upstream_loss"] = max(upstream, key=upstream.get)
    return decomposition


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _threshold_text(record) -> str:
    if isinstance(record, dict):
        if record.get("type") == "global" and record.get("value") is not None:
            return f"{record['value']:.6f}"
        return (
            f"classwise[{record.get('count')}] "
            f"{_fmt(record.get('min'))}–{_fmt(record.get('max'))}"
        )
    return _fmt(record)


def render_markdown(results: list[dict], stability: dict | None, note: str) -> str:
    lines = [
        "# 候选池与未知风险误差分解（李柯颖线）",
        "",
        note,
        "",
        "## 1. 检出与候选池总览",
        "",
        "| run | 阈值 | AUROC | AUPR | FPR95 | OSCR | 已知接受率 | 未知拒识率 | 候选数 | 候选纯度 | 已知污染率 | 未知召回 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in results:
        detection = run["detection"]
        lines.append(
            "| {run} | {threshold} | {auroc} | {aupr} | {fpr95} | {oscr} | {ka} | {ur} | {cand} | {purity} | {poll} | {recall} |".format(
                run=run["run"],
                threshold=_threshold_text(run["threshold"]),
                auroc=_fmt(detection.get("auroc")),
                aupr=_fmt(detection.get("aupr")),
                fpr95=_fmt(detection.get("fpr95")),
                oscr=_fmt(detection.get("oscr")),
                ka=_fmt(run["known_accept_rate"]),
                ur=_fmt(run["unknown_reject_rate"]),
                cand=run["candidate_count"],
                purity=_fmt(run["candidate_purity"]),
                poll=_fmt(run["known_pollution_rate"]),
                recall=_fmt(run["unknown_recall_in_pool"]),
            )
        )

    lines += ["", "## 2. 风险分数分布与重叠", ""]
    lines.append("| run | 已知 mean/median/p90 | 未知 mean/median/p90 | 重叠面积 | median gap |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for run in results:
        known = run["score_distribution"]["known"]
        unknown = run["score_distribution"]["unknown"]
        overlap = run["known_unknown_overlap"]
        lines.append(
            f"| {run['run']} | {_fmt(known.get('mean'))} / {_fmt(known.get('median'))} / "
            f"{_fmt(known.get('q90'))} | {_fmt(unknown.get('mean'))} / "
            f"{_fmt(unknown.get('median'))} / {_fmt(unknown.get('q90'))} | "
            f"{_fmt(overlap.get('overlap_area_50bins'))} | {_fmt(overlap.get('median_gap'))} |"
        )

    lines += ["", "## 3. 阈值附近样本", ""]
    for run in results:
        band = run["near_threshold_band"]
        if not band.get("definition"):
            lines.append(f"- {run['run']}: 未导出 calibration_scores.npz，跳过阈值邻域统计。")
            continue
        lines.append(
            f"- {run['run']}: 区间 [{_fmt(band['known_validation_q90'])}, "
            f"{_fmt(band['known_validation_q99'])}]，样本 {band['sample_count']} "
            f"（已知 {band['known_count']} / 未知 {band['unknown_count']}），"
            f"其中候选 {band['candidate_count']}。"
        )

    if stability and stability.get("runs"):
        lines += ["", "## 4. 跨种子候选标记稳定性", ""]
        lines.append(
            f"- 参与比较的运行：{', '.join(stability['runs'])}（{stability['run_count']} 个）"
        )
        lines.append(f"- 测试样本数：{stability['sample_count']}")
        lines.append(
            f"- 始终候选 {stability['always_candidate']}，始终非候选 {stability['never_candidate']}，"
            f"跨运行翻转 {stability['unstable_count']}（{_fmt(stability['unstable_rate'])}），"
            f"其中已知 {stability['unstable_known_count']}、未知 {stability['unstable_unknown_count']}。"
        )
        band = stability["near_threshold_band"]
        lines.append(
            f"- 阈值邻域内翻转率 {_fmt(band['band_unstable_rate'])}，"
            f"邻域外翻转率 {_fmt(band['outside_band_unstable_rate'])}。"
        )
        gap = stability["mean_distance_to_threshold"]
        lines.append(
            f"- 距阈值平均距离：稳定样本 {_fmt(gap['stable'])}，翻转样本 {_fmt(gap['unstable'])}。"
        )
        lines += ["", "### 翻转率最高的类别", ""]
        lines.append("| raw | 类别 | 属于 | 样本数 | 翻转数 | 翻转率 | 候选率范围 |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | --- |")
        for row in stability["per_class"][:15]:
            lines.append(
                f"| {row['raw_label']} | {row.get('class_name', '-')} | "
                f"{'已知' if row['is_known'] else '未知'} | {row['count']} | "
                f"{row['unstable_count']} | {_fmt(row['unstable_rate'])} | "
                f"{_fmt(row['candidate_rate_min'])}–{_fmt(row['candidate_rate_max'])} |"
            )

    lines += ["", "## 5. 误差分解", ""]
    for run in results:
        dec = run["error_decomposition"]
        lines.append(f"### {run['run']}")
        lines.append("")
        lines.append(
            f"- 未知漏检 {dec['unknown_missed']['count']} "
            f"（占未知 {_fmt(dec['unknown_missed']['rate_of_unknown'])}）"
        )
        lines.append(
            f"- 已知污染 {dec['known_pollution']['count']} "
            f"（占已知 {_fmt(dec['known_pollution']['rate_of_known'])}，"
            f"占候选池 {_fmt(dec['known_pollution']['rate_of_pool'])}）"
        )
        cluster = dec["clustering_given_candidate_pool"]
        lines.append(
            f"- 候选池内真实未知 {cluster['candidate_unknown_samples']}，"
            f"候选池聚类 ACC/NMI/ARI = {_fmt(cluster['cluster_acc'])} / "
            f"{_fmt(cluster['cluster_nmi'])} / {_fmt(cluster['cluster_ari'])}"
        )
        lines.append(
            f"- 主要上游损失（按比率最大项）：**{dec['dominant_upstream_loss']}**"
        )
        lines.append("")

    lines += ["", "## 6. 按类别的污染情况（每类该策略下的候选数量）", ""]
    for run in results:
        lines.append(f"### {run['run']}")
        lines.append("")
        lines.append("| 已知类别 | 样本数 | 误拒数 | 误拒率 | 分类准确率 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in run["known_classes"][:10]:
            lines.append(
                f"| {row['class_name']} | {row['count']} | {row['false_reject_count']} | "
                f"{_fmt(row['false_reject_rate'])} | {_fmt(row['classification_accuracy'])} |"
            )
        lines.append("")
        lines.append("| 未知类别 | 样本数 | 进入候选数 | 未知召回 | 漏检数 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in run["novel_classes"][:10]:
            lines.append(
                f"| {row['class_name']} | {row['count']} | {row['candidate_count']} | "
                f"{_fmt(row['unknown_recall'])} | {row['missed_count']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="discover run directories")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="display labels, aligned with --runs")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fine-names", type=Path, default=None,
                        help="split json that carries fine_class_names for index->name")
    parser.add_argument("--output", type=Path, required=True, help="json output")
    parser.add_argument("--markdown", type=Path, default=None, help="markdown output")
    parser.add_argument("--note", default="", help="context line printed at the top")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    split = _read_json(args.split)
    fine_names = _fine_class_names(split, args.fine_names)
    runs = [analyse_run(path, split, fine_names) for path in args.runs]

    labels = args.labels or [run["run"] for run in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels must match --runs")
    for run, label in zip(runs, labels):
        run["run"] = label
        run["run_dir_name"] = Path(run["run_dir"]).name

    stability = cross_run(runs)
    if fine_names and stability.get("per_class"):
        for row in stability["per_class"]:
            row["class_name"] = fine_names[row["raw_label"]]

    for run in runs:
        run["error_decomposition"] = error_decomposition(run, stability)

    note = args.note or (
        "未知标签只用于事后评测，未参与阈值校准、模型选择或候选筛选。"
    )
    payload = {
        "protocol_note": note,
        "runs": [{k: v for k, v in run.items() if not k.startswith("_")} for run in runs],
        "cross_run_stability": stability,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_markdown(runs, stability, note), encoding="utf-8"
        )
    print(f"saved: {args.output}")
    if args.markdown:
        print(f"saved: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
