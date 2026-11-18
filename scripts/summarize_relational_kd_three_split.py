"""Aggregate frozen relational-KD paired results; does not train or rewrite runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    2109: ROOT / "analysis" / "relational_kd_paired_pilot_s2109_20261114.json",
    2130: ROOT / "analysis" / "relational_kd_independent_replication_s2130_20261115.json",
    2131: ROOT / "analysis" / "relational_kd_independent_replication_s2131_20261115.json",
}
OUT = ROOT / "analysis" / "relational_kd_three_split_summary_20261115.json"
METRICS = (
    "auroc",
    "fpr95",
    "unknown_reject_rate",
    "cluster_candidate_count",
    "cluster_candidate_true_unknown_count_before_purification",
    "cluster_candidate_false_reject_count_before_purification",
    "cluster_candidate_purity_before_purification",
    "cluster_acc",
    "cluster_nmi",
    "cluster_ari",
    "cluster_k",
    "cluster_k_abs_error",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_metrics(seed: int, report: dict[str, Any]) -> dict[str, float]:
    if seed == 2109:
        raw = report["metrics"]
        return {name: float(raw[name]["delta"]) for name in METRICS}

    arms = report["arms"]
    control = arms["control"]["metrics"]
    treatment = arms["relational_kd"]["metrics"]
    return {name: float(treatment[name]) - float(control[name]) for name in METRICS}


def main() -> None:
    split_results: dict[str, dict[str, float]] = {}
    for seed, path in SOURCES.items():
        report = read_json(path)
        if report.get("protocol") != "semantic_hard":
            raise ValueError(f"Unexpected split protocol in {path}")
        if "3.34" not in report.get("single_changed_factor", ""):
            raise ValueError(f"Unexpected treatment factor in {path}")
        if seed == 2109:
            if report.get("test_labels_used_for_training_or_calibration") is not False:
                raise ValueError("Unknown test labels were not explicitly excluded")
        elif (
            report.get("unknown_labels_used_for_training_selection_or_calibration")
            is not False
            or report.get("unknown_labels_used_only_for_final_metrics") is not True
        ):
            raise ValueError("Unknown-label isolation check failed")
        split_results[str(seed)] = paired_metrics(seed, report)

    means = {
        name: sum(values[name] for values in split_results.values()) / len(split_results)
        for name in METRICS
    }
    higher_better = (
        "auroc",
        "unknown_reject_rate",
        "cluster_candidate_purity_before_purification",
        "cluster_acc",
        "cluster_nmi",
        "cluster_ari",
    )
    lower_better = ("fpr95", "cluster_k_abs_error")
    direction_counts = {
        name: {
            "improved": sum(split[name] > 0 for split in split_results.values())
            if name in higher_better
            else sum(split[name] < 0 for split in split_results.values()),
            "unchanged": sum(split[name] == 0 for split in split_results.values()),
            "worsened": sum(split[name] < 0 for split in split_results.values())
            if name in higher_better
            else sum(split[name] > 0 for split in split_results.values()),
            "preferred_direction": "higher" if name in higher_better else "lower",
        }
        for name in (*higher_better, *lower_better)
    }

    output = {
        "question": "Descriptive paired summary of frozen relational-KD treatment results across independent semantic-hard class splits.",
        "treatment": "alpha_relational_kd: 0.0 vs 3.34; all other registered training and discovery conditions matched within each split.",
        "unknown_test_labels_used_for_training_selection_or_calibration": False,
        "class_splits": list(SOURCES),
        "split_deltas_treatment_minus_control": split_results,
        "mean_delta_across_splits": means,
        "direction_counts": direction_counts,
        "interpretation": (
            "Descriptive evidence only: AUROC is mixed, unknown rejection rate improves in two of three splits, "
            "and clustering ACC decreases in all three. The results do not support a stable end-to-end benefit, "
            "a default change, or further scanning of the same relational-KD weight."
        ),
        "source_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in SOURCES.values()],
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)
    print(json.dumps({"mean_delta_across_splits": means, "direction_counts": direction_counts}, indent=2))


if __name__ == "__main__":
    main()
