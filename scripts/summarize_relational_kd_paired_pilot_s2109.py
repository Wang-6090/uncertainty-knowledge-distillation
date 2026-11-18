"""Summarize the already completed relational-KD paired pilot; never trains models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAG = "relational_kd_paired_pilot_s2109_20261114"
ARMS = ("control", "relational_kd")
OUT = ROOT / "analysis" / f"{TAG}.json"
RESULT_FIELDS = {
    "best_epoch",
    "best_val_known_acc",
    "best_checkpoint_selection_score",
    "best_open_checkpoint_stats",
}
PATH_FIELDS = {"work_dir", "student_ckpt"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    runs: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_dir = ROOT / "runs" / f"{TAG}_{arm}"
        discover_dir = ROOT / "runs" / f"{TAG}_{arm}_discover"
        runs[arm] = {
            "config": read_json(run_dir / "train_student_config.json"),
            "history": read_json(run_dir / "train_history.json"),
            "metrics": read_json(discover_dir / "discovery_report.json"),
            "train_dir": str(run_dir),
            "discover_dir": str(discover_dir),
        }
        assert len(runs[arm]["history"]) == 4
        assert runs[arm]["metrics"]["rejector_protocol"][
            "test_labels_used_for_training_or_calibration"
        ] is False

    control = runs["control"]["config"]
    treatment = runs["relational_kd"]["config"]
    diffs = {
        key: [control.get(key), treatment.get(key)]
        for key in sorted(set(control) | set(treatment))
        if control.get(key) != treatment.get(key)
    }
    unexpected = set(diffs) - PATH_FIELDS - RESULT_FIELDS - {"alpha_relational_kd"}
    if unexpected:
        raise RuntimeError(f"Unexpected treatment/config differences: {sorted(unexpected)}")
    if diffs.get("alpha_relational_kd") != [0.0, 3.34]:
        raise RuntimeError("The registered relational-KD treatment contrast changed")

    metric_keys = (
        "auroc",
        "aupr",
        "fpr95",
        "known_class_accuracy_all_known",
        "known_class_accuracy_after_accept",
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
    deltas = {}
    for key in metric_keys:
        a = runs["control"]["metrics"][key]
        b = runs["relational_kd"]["metrics"][key]
        deltas[key] = {"control": a, "relational_kd": b, "delta": b - a}

    train_audit = {}
    for arm, values in runs.items():
        history = values["history"]
        train_audit[arm] = [
            {
                "epoch": epoch["epoch"],
                "raw_relational_kd_loss": epoch["train"]["relational_kd"],
                "relational_kd_backbone_grad_norm": epoch["train"].get(
                    "gradient_audit_grad_norm_relational_kd_term"
                ),
                "gradient_cosine_vs_classification_ce": epoch["train"].get(
                    "gradient_audit_grad_cos_ce_relational_kd_term"
                ),
                "gradient_cosine_vs_logit_kd": epoch["train"].get(
                    "gradient_audit_grad_cos_kd_relational_kd_term"
                ),
            }
            for epoch in history
        ]

    report = {
        "question": "Does adding batchwise relational knowledge distillation improve the four-epoch semantic-hard seed-2109 pipeline?",
        "seed": 2109,
        "protocol": "semantic_hard",
        "single_changed_factor": "alpha_relational_kd: 0.0 vs 3.34",
        "config_differences": diffs,
        "config_difference_interpretation": {
            "alpha_relational_kd": "intended treatment factor",
            "work_dir": "separate output locations",
            "student_ckpt": "separate checkpoint paths",
            "best_val_known_acc": "observed validation outcome used for checkpoint selection, not a configured treatment factor",
            "best_checkpoint_selection_score": "observed checkpoint-selection outcome, not a configured treatment factor",
        },
        "test_labels_used_for_training_or_calibration": False,
        "metrics": deltas,
        "training_mechanism_audit": train_audit,
        "runs": {
            arm: {"train_dir": values["train_dir"], "discover_dir": values["discover_dir"]}
            for arm, values in runs.items()
        },
        "interpretation": "Single-split pilot: relational KD was active, but the trade-off is mixed and does not support an overall improvement claim or a default change.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)
    print(json.dumps({"config_differences": diffs, "metrics": deltas}, indent=2))


if __name__ == "__main__":
    main()
