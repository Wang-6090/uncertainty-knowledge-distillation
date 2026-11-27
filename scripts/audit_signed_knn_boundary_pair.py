"""Audit and summarize an already completed signed-kNN boundary pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def summarize_history(path: Path) -> dict[str, Any]:
    history = read_json(path)
    if not isinstance(history, list) or not history:
        raise ValueError(f"invalid train history: {path}")
    rows = [row.get("train", row) if isinstance(row, dict) else {} for row in history]
    boundary = [float(row.get("discovery_knn_boundary", 0.0)) for row in rows]
    gradients = [
        float(row.get("gradient_audit_grad_norm_discovery_knn_boundary", 0.0))
        for row in rows
    ]
    return {
        "epochs": len(rows),
        "boundary_loss_by_epoch": boundary,
        "boundary_loss_mean": sum(boundary) / len(boundary),
        "boundary_gradient_norm_by_epoch": gradients,
        "boundary_gradient_norm_min": min(gradients),
        "boundary_gradient_norm_mean": sum(gradients) / len(gradients),
        "boundary_gradient_nonzero_all_epochs": all(value > 0.0 for value in gradients),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2091)
    parser.add_argument("--tag", default="signed_knn_boundary_pair")
    options = parser.parse_args()

    base = ROOT / "runs"
    arm_dirs = {
        "outward": base / f"{options.tag}_s{options.seed}_outward",
        "signed_support": base / f"{options.tag}_s{options.seed}_signed_support",
    }
    discover_dirs = {
        name: base / f"{options.tag}_s{options.seed}_{name}_discover"
        for name in arm_dirs
    }
    for path in (*arm_dirs.values(), *discover_dirs.values()):
        if not path.is_dir():
            raise FileNotFoundError(path)

    configs = {
        name: read_json(path / "train_student_config.json")
        for name, path in arm_dirs.items()
    }
    allowed_differences = {
        "work_dir",
        "student_ckpt",
        "discovery_knn_boundary_objective",
        "best_epoch",
        "best_val_known_acc",
        "best_checkpoint_selection_score",
        "best_open_checkpoint_stats",
        "prototype_init_stats",
        "prototype_refresh_history",
    }
    differences = {
        key: (configs["outward"].get(key), configs["signed_support"].get(key))
        for key in sorted(set(configs["outward"]) | set(configs["signed_support"]))
        if key not in allowed_differences
        and configs["outward"].get(key) != configs["signed_support"].get(key)
    }
    if differences:
        raise ValueError(f"unexpected paired train-config differences: {differences}")
    if configs["outward"].get("discovery_knn_boundary_objective") != "outward":
        raise ValueError("outward arm has the wrong saved objective")
    if configs["signed_support"].get("discovery_knn_boundary_objective") != "signed_support":
        raise ValueError("signed arm has the wrong saved objective")

    reports = {
        name: read_json(path / "discovery_report.json")
        for name, path in discover_dirs.items()
    }
    for name, report in reports.items():
        protocol = report.get("rejector_protocol", {})
        if protocol.get("hard_labels_used") is not False:
            raise ValueError(f"{name}: hard mixed-pool labels were used")
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise ValueError(f"{name}: test labels entered training/calibration")
        if report.get("threshold_policy") != "known_val_known_coverage":
            raise ValueError(f"{name}: unexpected threshold policy")

    metric_keys = (
        "auroc",
        "aupr",
        "fpr95",
        "oscr",
        "unknown_reject_rate",
        "known_accept_rate",
        "known_class_accuracy_after_accept",
    )
    metrics = {
        name: {key: reports[name].get(key) for key in metric_keys}
        for name in reports
    }
    delta = {
        key: float(metrics["signed_support"][key]) - float(metrics["outward"][key])
        for key in metric_keys
    }
    result = {
        "protocol": f"signed_knn_boundary_pair_s{options.seed}_20261024",
        "question": "Does signed class-conditional support supervision improve semantic-hard open-set separation?",
        "seed": int(options.seed),
        "single_changed_factor": "discovery_knn_boundary_objective",
        "control": "outward",
        "treatment": "signed_support",
        "treatment_weight": configs["signed_support"].get("alpha_discovery_knn_boundary"),
        "split_path": configs["outward"].get("split_path"),
        "teacher_checkpoint": configs["outward"].get("teacher_ckpt"),
        "candidate_protocol": "selection-conditional single-view shared-boundary gate",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "paired_train_config_differences": {
            key: value
            for key, value in {
                "objective": (
                    configs["outward"].get("discovery_knn_boundary_objective"),
                    configs["signed_support"].get("discovery_knn_boundary_objective"),
                ),
                "best_validation_selection": (
                    configs["outward"].get("best_val_known_acc"),
                    configs["signed_support"].get("best_val_known_acc"),
                ),
            }.items()
        },
        "training_signal_audit": {
            name: summarize_history(path / "train_history.json")
            for name, path in arm_dirs.items()
        },
        "metrics": metrics,
        "delta_signed_minus_outward": delta,
        "score_distribution": {
            name: reports[name].get("score_distribution", {}) for name in reports
        },
        "evidence_level": "single fresh semantic-hard split exploratory comparison",
        "decision": "do not change defaults from one split; use the joint operating-point criterion before any replication",
    }
    result_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
