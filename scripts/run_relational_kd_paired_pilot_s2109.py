"""Matched one-factor pilot for adding relational KD on semantic-hard seed 2109."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


SEED = 2109
RELATIONAL_ALPHA = 3.34  # Frozen from the known-train gradient-scale diagnostic median.
TAG = "relational_kd_paired_pilot_s2109_20261114"
SOURCE_STUDENT = ROOT / "runs/rejection_epoch_budget_s2109_four_epoch/train_student_config.json"
SOURCE_DISCOVER = ROOT / "runs/rejection_epoch_budget_s2109_four_epoch_discover/discover_config.json"
TEACHER_CKPT = ROOT / "runs/rejection_epoch_budget_s2109_teacher/teacher.pt"
SPLIT = ROOT / "splits/semantic_hard/seed2109/cifar100_60_40_semantic_hard.json"
RESULT = ROOT / "analysis" / f"{TAG}.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_args(path: Path, command: str, work_dir: Path):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = SEED
    args.model_seed = None
    args.data_root = str(ROOT / "data")
    args.split_path = str(SPLIT)
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    for path in (SOURCE_STUDENT, SOURCE_DISCOVER, TEACHER_CKPT, SPLIT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if RESULT.exists():
        raise FileExistsError(f"Refusing to overwrite {RESULT}")

    split = read_json(SPLIT)
    if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != SEED:
        raise ValueError("Split protocol/seed mismatch")
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("Expected the registered 60-known/40-novel split")

    source_student = read_json(SOURCE_STUDENT)
    source_discover = read_json(SOURCE_DISCOVER)
    if float(source_student.get("alpha_feat_kd", -1.0)) != 0.1:
        raise ValueError("Pilot source must retain the registered feature-KD weight 0.1")
    if int(source_student.get("epochs", 0)) != 4:
        raise ValueError("Pilot source must use the registered four-epoch recipe")
    if source_discover.get("score_mode") != "feature_rejector":
        raise ValueError("Unexpected discovery evaluation pipeline")
    if source_discover.get("threshold_policy") != "known_coverage":
        raise ValueError("Threshold must be calibrated from known validation coverage")
    if float(source_discover.get("target_known_coverage", 0.0)) != 0.95:
        raise ValueError("Unexpected known-only threshold target")
    if float(source_student.get("alpha_relational_kd", 0.0)) != 0.0:
        raise ValueError("Source baseline is expected to have relational KD disabled")

    arms: dict[str, dict[str, Any]] = {}
    for arm, relational_alpha in (("control", 0.0), ("relational_kd", RELATIONAL_ALPHA)):
        run_dir = ROOT / "runs" / f"{TAG}_{arm}"
        discover_dir = ROOT / "runs" / f"{TAG}_{arm}_discover"
        if run_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"Refusing to overwrite outputs for {arm}: {run_dir}, {discover_dir}")

        train_args = load_args(SOURCE_STUDENT, "train_student", run_dir)
        train_args.teacher_ckpt = str(TEACHER_CKPT)
        train_args.student_ckpt = str(run_dir / "student.pt")
        train_args.alpha_relational_kd = relational_alpha
        print(f"[train] arm={arm}, alpha_relational_kd={relational_alpha}", flush=True)
        fit_student(train_args)

        discover_args = load_args(SOURCE_DISCOVER, "discover", discover_dir)
        discover_args.student_ckpt = str(run_dir / "student.pt")
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)

        metrics_path = discover_dir / "discovery_report.json"
        history_path = run_dir / "train_history.json"
        train_config_path = run_dir / "train_student_config.json"
        for path in (metrics_path, history_path, train_config_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        metrics = read_json(metrics_path)
        train_config = read_json(train_config_path)
        if float(train_config.get("alpha_relational_kd", -1.0)) != relational_alpha:
            raise RuntimeError(f"Saved training config lost treatment assignment for {arm}")
        if metrics.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"Test labels entered training/calibration for {arm}")
        calibration_coverage = float(metrics.get("calibration_known_accept_rate", float("nan")))
        if not (0.93 <= calibration_coverage <= 0.97):
            raise RuntimeError(f"Known-only threshold calibration outside tolerance for {arm}: {calibration_coverage}")
        arms[arm] = {
            "train_config": train_config,
            "history": read_json(history_path),
            "metrics": metrics,
            "train_dir": str(run_dir),
            "discover_dir": str(discover_dir),
        }

    control_config = arms["control"]["train_config"]
    treatment_config = arms["relational_kd"]["train_config"]
    # Check treatment assignments separately from outcomes written back into
    # the saved config after training/checkpoint selection.
    allowed_differences = {
        "alpha_relational_kd",
        "work_dir",
        "student_ckpt",
        "best_epoch",
        "best_val_known_acc",
        "best_checkpoint_selection_score",
        "best_open_checkpoint_stats",
    }
    differences = {
        key: (control_config.get(key), treatment_config.get(key))
        for key in sorted(set(control_config) | set(treatment_config))
        if control_config.get(key) != treatment_config.get(key)
    }
    unexpected = {key: pair for key, pair in differences.items() if key not in allowed_differences}
    if unexpected:
        raise RuntimeError(f"Paired training configurations differ unexpectedly: {unexpected}")

    report = {
        "question": "Does adding known-batch relational KD improve the four-epoch semantic-hard seed-2109 pipeline?",
        "seed": SEED,
        "protocol": "semantic_hard",
        "split": str(SPLIT),
        "single_changed_factor": "alpha_relational_kd: 0.0 vs 3.34",
        "weight_basis": "median gradient-norm matching coefficient from the separate known-train diagnostic; fixed before outcome evaluation",
        "feature_kd_retained_in_both_arms": 0.1,
        "unknown_identity_labels_used_for_training_selection_or_calibration": False,
        "unknown_labels_used_only_for_final_metrics": True,
        "training_config_differences": differences,
        "arms": arms,
    }
    write_json(RESULT, report)
    print(json.dumps({"result": str(RESULT), "config_differences": differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
