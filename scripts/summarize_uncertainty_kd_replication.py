#!/usr/bin/env python3
"""Audit and summarize the independent uncertainty-KD replication.

The script intentionally reads only saved train/discovery/calibration reports.
It never uses mixed-pool or test labels for selection; those labels are present
only in retrospective metric reports produced by the frozen evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = ("no_kd", "standard", "uncertainty")
METRICS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "unknown_reject_rate",
    "known_class_accuracy_after_accept",
    "cluster_nmi",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def row_for(root: Path, seed: int, arm: str) -> dict[str, Any]:
    train_dir = root / f"uncertainty_calibrated_fusion_replication_s{seed}_{arm}"
    discover_dir = root / f"uncertainty_calibrated_fusion_replication_s{seed}_{arm}_discover"
    fusion_dir = root.parent / "runs" / f"calibrated_uncertainty_fusion_replication_s{seed}_{arm}"

    train = load_json(train_dir / "train_student_config.json")
    base = load_json(discover_dir / "discovery_report.json")
    fusion = load_json(fusion_dir / "discovery_report.json")
    calibration = load_json(fusion_dir / "calibration_report.json")

    calibration_audit = calibration["validation_uncertainty_error_calibration"]
    sample_audit = calibration["validation_protocol"]["sample_index_audit"]

    # These are the pre-registered treatment definitions.  Failing loudly is
    # preferable to silently summarizing a contaminated or mismatched triad.
    expected_alpha = 0.0 if arm == "no_kd" else 1.0
    expected_mode = "uncertainty" if arm == "uncertainty" else "standard"
    if float(train.get("alpha_kd", -1.0)) != expected_alpha:
        raise ValueError(f"{seed}/{arm}: unexpected alpha_kd")
    if train.get("kd_mode") != expected_mode:
        raise ValueError(f"{seed}/{arm}: unexpected kd_mode")
    if float(train.get("alpha_feat_kd", -1.0)) != 0.0:
        raise ValueError(f"{seed}/{arm}: feature KD is not disabled")
    if float(train.get("alpha_rejection_kd", -1.0)) != 0.0:
        raise ValueError(f"{seed}/{arm}: rejection KD is not disabled")
    if calibration_audit.get("calibration_source") != "known_validation_only":
        raise ValueError(f"{seed}/{arm}: non-validation calibration source")
    if calibration_audit.get("labels_used_for_training_or_test_selection") is not False:
        raise ValueError(f"{seed}/{arm}: labels leaked into selection")
    if calibration.get("auto_score_selection_uses_open_labels") is not False:
        raise ValueError(f"{seed}/{arm}: open labels used for score selection")
    if sample_audit.get("overlap_count") != 0 or sample_audit.get("disjoint") is not True:
        raise ValueError(f"{seed}/{arm}: calibration subset is not disjoint")

    return {
        "seed": seed,
        "arm": arm,
        "alpha_kd": train["alpha_kd"],
        "kd_mode": train["kd_mode"],
        "best_epoch": train.get("best_epoch"),
        "best_val_known_acc": train.get("best_val_known_acc"),
        "base": {metric: base[metric] for metric in METRICS},
        "fusion": {metric: fusion[metric] for metric in METRICS},
        "delta_fusion_minus_base": {
            metric: fusion[metric] - base[metric] for metric in METRICS
        },
        "calibration": {
            "source": calibration_audit["calibration_source"],
            "labels_used_for_training_or_test_selection": calibration_audit[
                "labels_used_for_training_or_test_selection"
            ],
            "validation_sample_count": calibration_audit["sample_count"],
            "subset_overlap_count": sample_audit["overlap_count"],
            "target_known_coverage": calibration["threshold_policy"][
                "target_known_coverage"
            ],
            "realized_known_accept_rate": calibration["threshold_policy"][
                "known_accept_rate"
            ],
        },
    }


def paired(rows: list[dict[str, Any]], treatment: str, control: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        values = []
        for row in rows:
            if row["arm"] != treatment:
                continue
            control_row = next(
                candidate
                for candidate in rows
                if candidate["seed"] == row["seed"] and candidate["arm"] == control
            )
            values.append(row["fusion"][metric] - control_row["fusion"][metric])
        result[metric] = mean(values)
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Independent uncertainty-KD replication summary",
        "",
        "This report is generated from frozen saved reports; no labels are used to select a method, checkpoint, threshold, or fusion weight.",
        "",
        "## Per-seed frozen fusion results",
        "",
        "| seed | arm | AUROC | AUPR | FPR95 | OSCR | unknown rejection | accepted-known accuracy | cluster NMI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        values = row["fusion"]
        lines.append(
            "| {seed} | {arm} | {auroc:.4f} | {aupr:.4f} | {fpr95:.4f} | {oscr:.4f} | {unknown_reject_rate:.4f} | {known_class_accuracy_after_accept:.4f} | {cluster_nmi:.4f} |".format(
                seed=row["seed"], arm=row["arm"], **values
            )
        )
    lines += [
        "",
        "## Mean paired contrasts (uncertainty minus control)",
        "",
        "| control | AUROC | AUPR | FPR95 | OSCR | unknown rejection | accepted-known accuracy | cluster NMI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for control, values in report["paired_contrasts"].items():
        lines.append(
            "| {control} | {auroc:+.4f} | {aupr:+.4f} | {fpr95:+.4f} | {oscr:+.4f} | {unknown_reject_rate:+.4f} | {known_class_accuracy_after_accept:+.4f} | {cluster_nmi:+.4f} |".format(
                control=control, **values
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        "The uncertainty-weighted logits-KD treatment is not promoted. Across the two independent seeds it does not show a consistent joint improvement over either no-KD or standard-KD: rejection, known utility, and ranking outcomes trade off by seed. The validation-only fusion remains an opt-in diagnostic, not evidence that uncertainty-KD training is effective.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2077, 2078])
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    rows = [row_for(args.run_root, seed, arm) for seed in args.seeds for arm in ARMS]
    report = {
        "protocol": "uncertainty_validation_fusion_replication_20261023",
        "seeds": args.seeds,
        "arms": list(ARMS),
        "rows": rows,
        "paired_contrasts": {
            "no_kd": paired(rows, "uncertainty", "no_kd"),
            "standard": paired(rows, "uncertainty", "standard"),
        },
        "decision": "do_not_promote_uncertainty_kd_training",
    }
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "uncertainty_kd_replication_s2077_s2078_summary.json"
    markdown_path = args.analysis_dir / "uncertainty_kd_replication_s2077_s2078_20261023.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
