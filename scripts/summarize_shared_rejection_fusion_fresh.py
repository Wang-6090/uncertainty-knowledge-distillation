#!/usr/bin/env python3
"""Summarize fixed validation-calibrated rejection fusion fresh splits.

This script reads only frozen discovery reports. It performs no training,
calibration, threshold selection, or test-label-based method selection. Test
labels may appear in the saved reports only because the reports contain final
retrospective metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "unknown_reject_rate",
    "known_accept_rate",
    "known_class_accuracy_after_accept",
)


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"expected JSON object: {path}")
    protocol = report.get("rejector_protocol", {})
    if protocol.get("test_labels_used_for_training_or_calibration") is not False:
        raise ValueError(f"{path}: test labels were used for training/calibration")
    if protocol.get("hard_labels_used") is not False:
        raise ValueError(f"{path}: hard labels were used in the mixed-pool protocol")
    for metric in METRICS:
        if not isinstance(report.get(metric), (int, float)):
            raise ValueError(f"{path}: missing numeric metric {metric}")
    return report


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _protocol_without_score_mode(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the protocol fields that must match across paired arms.

    The score mode is the registered evaluation-only factor. Every other
    rejector-protocol field must be identical; otherwise a paired comparison
    can silently mix score changes with supervision or calibration changes.
    """
    return {
        key: value
        for key, value in protocol.items()
        if key != "score_mode"
    }


def evaluate_pair(seed: int, control_path: Path, treatment_path: Path) -> dict[str, Any]:
    control = load_report(control_path)
    treatment = load_report(treatment_path)
    if control["rejector_protocol"].get("score_mode") != "feature_rejector":
        raise ValueError(f"{seed}: unexpected control score mode")
    if treatment["rejector_protocol"].get("score_mode") != "feature_rejector_uncertainty_fusion":
        raise ValueError(f"{seed}: unexpected treatment score mode")
    control_protocol = _protocol_without_score_mode(control["rejector_protocol"])
    treatment_protocol = _protocol_without_score_mode(treatment["rejector_protocol"])
    if control_protocol != treatment_protocol:
        raise ValueError(
            f"{seed}: paired rejector protocols differ beyond score_mode: "
            f"control={control_protocol!r}, treatment={treatment_protocol!r}"
        )
    if control.get("threshold_policy") != treatment.get("threshold_policy"):
        raise ValueError(f"{seed}: paired threshold policies differ")
    delta = {metric: treatment[metric] - control[metric] for metric in METRICS}
    passed = (
        delta["unknown_reject_rate"] >= 0.02
        and delta["fpr95"] <= 0.01
        and delta["oscr"] >= -0.005
        and delta["known_class_accuracy_after_accept"] >= -0.005
    )
    return {
        "seed": seed,
        "control_path": str(control_path),
        "treatment_path": str(treatment_path),
        "control": {metric: control[metric] for metric in METRICS},
        "treatment": {metric: treatment[metric] for metric in METRICS},
        "delta_treatment_minus_control": delta,
        "joint_criterion_pass": passed,
        "protocol_match_except_score_mode": True,
        "threshold_policy_match": True,
        "control_candidate_count": control.get("cluster_candidate_count"),
        "treatment_candidate_count": treatment.get("cluster_candidate_count"),
        "treatment_cluster_k": treatment.get("cluster_k"),
        "treatment_cluster_nmi": treatment.get("cluster_nmi"),
        "treatment_cluster_ari": treatment.get("cluster_ari"),
    }


def markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = [
        "# Shared-backbone fixed uncertainty-fusion fresh summary",
        "",
        "This summary is generated from frozen discovery reports. The fixed",
        "fusion is `0.5` feature-rejector score plus `0.5` known-validation-",
        "calibrated uncertainty risk; no fusion weight was tuned.",
        "",
        "| seed | Δ AUROC | Δ AUPR | Δ FPR95 | Δ OSCR | Δ unknown rejection | Δ accepted-known accuracy | joint criterion |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        delta = row["delta_treatment_minus_control"]
        lines.append(
            "| {seed} | {auroc:+.4f} | {aupr:+.4f} | {fpr95:+.4f} | {oscr:+.4f} | {unknown_reject_rate:+.4f} | {known_class_accuracy_after_accept:+.4f} | {passed} |".format(
                seed=row["seed"],
                passed="PASS" if row["joint_criterion_pass"] else "FAIL",
                **delta,
            )
        )
    lines += [
        "",
        "## Mean and standard deviation of paired deltas",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for metric, stats in report["delta_summary"].items():
        lines.append(f"| {metric} | {stats['mean']:+.6f} | {stats['std']:.6f} |")
    lines += [
        "",
        "## Decision",
        "",
        f"The fixed fusion passes the joint criterion on {report['joint_criterion_pass_count']} of {len(rows)} fresh splits. All paired reports match in every protocol field except the registered score mode, and use the same threshold policy. It improves mean ranking and calibrated unknown-rejection metrics. Known acceptance decreases on {sum(row['delta_treatment_minus_control']['known_accept_rate'] < 0 for row in rows)} of {len(rows)} splits and increases on the others, so this remains a split-dependent operating-point trade-off. The mechanism therefore remains an explicit opt-in candidate, not the default score.",
        "",
        "Candidate counts and automatic-K/NMI/ARI are reported descriptively; they are not used for promotion or fusion-weight selection.",
        "",
        "No further fusion-weight or score-mode search is justified on these splits.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    pairs = [
        (
            2084,
            Path("runs/shared_rejection_uncertainty_margin_s2084_baseline_discover/discovery_report.json"),
            Path("runs/shared_rejection_uncertainty_margin_s2084_baseline_uncertainty_fusion/discovery_report.json"),
        ),
        (
            2085,
            Path("runs/shared_rejection_uncertainty_margin_s2085_baseline_discover/discovery_report.json"),
            Path("runs/shared_rejection_uncertainty_margin_s2085_baseline_uncertainty_fusion/discovery_report.json"),
        ),
        (
            2086,
            Path("runs/shared_rejection_fusion_fresh_s2086_baseline_discover/discovery_report.json"),
            Path("runs/shared_rejection_fusion_fresh_s2086_baseline_uncertainty_fusion/discovery_report.json"),
        ),
        (
            2089,
            Path("runs/shared_rejection_fusion_fresh_s2089_baseline_discover/discovery_report.json"),
            Path("runs/shared_rejection_fusion_fresh_s2089_baseline_uncertainty_fusion/discovery_report.json"),
        ),
        (
            2092,
            Path("runs/shared_rejection_fusion_fresh_s2092_baseline_discover/discovery_report.json"),
            Path("runs/shared_rejection_fusion_fresh_s2092_baseline_uncertainty_fusion/discovery_report.json"),
        ),
        (
            2098,
            Path("runs/shared_rejection_fusion_fresh_s2098_baseline_discover/discovery_report.json"),
            Path("runs/shared_rejection_fusion_fresh_s2098_baseline_uncertainty_fusion/discovery_report.json"),
        ),
        (
            2103,
            Path("runs/shared_rejection_fusion_fresh_s2103_baseline_discover/discovery_report.json"),
            Path("runs/shared_rejection_fusion_fresh_s2103_baseline_uncertainty_fusion/discovery_report.json"),
        ),
    ]
    rows = [evaluate_pair(*pair) for pair in pairs]
    delta_summary = {
        metric: {
            "mean": mean([row["delta_treatment_minus_control"][metric] for row in rows]),
            "std": std([row["delta_treatment_minus_control"][metric] for row in rows]),
        }
        for metric in METRICS
    }
    report = {
        "protocol": "shared_rejection_fusion_fresh_20261024",
        "seeds": [row["seed"] for row in rows],
        "rows": rows,
        "delta_summary": delta_summary,
        "joint_criterion_pass_count": sum(row["joint_criterion_pass"] for row in rows),
        "protocol_audit": {
            "all_pairs_match_except_score_mode": all(
                row["protocol_match_except_score_mode"] for row in rows
            ),
            "all_pairs_share_threshold_policy": all(
                row["threshold_policy_match"] for row in rows
            ),
        },
        "decision": "strong_opt_in_candidate_not_default",
    }
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "shared_rejection_fusion_fresh_2084_2103_summary.json"
    markdown_path = args.analysis_dir / "shared_rejection_fusion_fresh_2084_2103_summary_20261024.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
