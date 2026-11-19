"""Run a paired fresh-split pilot for relational KD into rejection features.

The control and treatment share the same split-matched teacher, student recipe,
training budget, discovery protocol, and checkpoint/calibration rules. The
only intended training difference is ``alpha_rejection_relational_kd``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rejection_feature_kd_pair import load_args, make_split, read_json, write_json
from train import discover, fit_student, fit_teacher


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def complete_teacher_artifacts(teacher_dir: Path, split_path: Path, seed: int) -> bool:
    """Return whether an existing teacher can be safely reused for this pair."""
    required = (
        teacher_dir / "teacher.pt",
        teacher_dir / "train_teacher_config.json",
        teacher_dir / "train_history.json",
    )
    if not all(path.is_file() for path in required):
        return False
    config = read_json(teacher_dir / "train_teacher_config.json")
    if int(config.get("seed", -1)) != int(seed):
        return False
    configured_split = config.get("split_path")
    if not configured_split or resolve(Path(str(configured_split))) != split_path.resolve():
        return False
    history = read_json(teacher_dir / "train_history.json")
    return len(history) == int(config.get("epochs", -1))


def apply_overrides(args: Any, options: argparse.Namespace, work_dir: Path) -> None:
    args.seed = int(options.seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    if options.epochs is not None:
        args.epochs = int(options.epochs)
    if options.limit_train is not None:
        args.limit_train = int(options.limit_train)
    if options.limit_val is not None:
        args.limit_val = int(options.limit_val)
    if options.limit_test is not None:
        args.limit_test = int(options.limit_test)
    if options.limit_discovery is not None:
        args.limit_discovery = int(options.limit_discovery)
    if options.smoke:
        args.epochs = 1
        # The support bank and known-only calibration require all 60 known
        # classes in the train/validation subsets, so smoke shortens the
        # schedule but keeps those subsets complete.
        args.limit_train = 0
        # Validation must retain all 60 known classes for support-center
        # construction and known-only calibration.
        args.limit_val = 0
        args.limit_test = 256
        # The inherited source uses a matched discovery pool.  Its loader
        # requires the full configured pool size, so the smoke keeps the pool
        # protocol and limits only train/validation/test data.
        args.limit_discovery = 0


def preflight(options: argparse.Namespace, sources: list[Path], split_path: Path) -> dict[str, Any]:
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    student_source = read_json(sources[1])
    discover_source = read_json(sources[2])
    if student_source.get("independent_rejection_encoder") is not True:
        raise ValueError("source student must use an independent rejection encoder")
    if int(student_source.get("rejection_feature_dim", 0)) <= 0:
        raise ValueError("source student must enable rejection features")
    if student_source.get("checkpoint_selection") != "known_acc":
        raise ValueError("student checkpoint selection must use known validation accuracy")
    if discover_source.get("score_mode") != "feature_rejector":
        raise ValueError("pilot must keep feature_rejector as the discovery score")
    if split_path.exists():
        split = read_json(split_path)
        if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != options.seed:
            raise ValueError(f"existing split does not match the requested protocol: {split_path}")
        split_status = "existing matching split"
    else:
        split_status = "fresh split will be generated"
    return {
        "seed": int(options.seed),
        "tag": options.tag,
        "split": str(split_path),
        "split_status": split_status,
        "alpha_rejection_relational_kd": float(options.alpha),
        "source_configs": [str(path) for path in sources],
        "dry_run": bool(options.dry_run),
        "smoke": bool(options.smoke),
        "unknown_labels_used_for_training_or_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2140)
    parser.add_argument("--tag", default="rejection_relational_kd_s2140")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--limit-discovery", type=int)
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    options = parser.parse_args()

    sources = [
        resolve(options.source_teacher_config),
        resolve(options.source_student_config),
        resolve(options.source_discover_config),
    ]
    student_source = read_json(sources[1])
    data_root = resolve(Path(str(student_source.get("data_root", "data"))))
    split_path = ROOT / f"splits/semantic_hard/seed{options.seed}/cifar100_60_40_semantic_hard.json"
    audit = preflight(options, sources, split_path)
    if options.dry_run:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    make_split(options.seed, split_path, data_root)
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard CIFAR-100 60/40 split")

    runs_root = ROOT / "runs"
    teacher_dir = runs_root / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    arm_names = ("control", "relational")
    arm_dirs = {arm: runs_root / f"{options.tag}_{arm}" for arm in arm_names}
    discover_dirs = {arm: Path(f"{arm_dirs[arm]}_discover") for arm in arm_names}
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    teacher_exists = teacher_dir.exists()
    teacher_complete = complete_teacher_artifacts(teacher_dir, split_path, options.seed)
    if teacher_exists and not teacher_complete:
        raise FileExistsError(
            "refusing to reuse incomplete or mismatched teacher artifacts: "
            f"{teacher_dir}"
        )
    artifacts = [result_path, *arm_dirs.values(), *discover_dirs.values()]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("refusing to overwrite an existing relational-KD pilot artifact")

    if not teacher_complete:
        teacher_args = load_args(sources[0], "train_teacher", teacher_dir, options.seed)
        apply_overrides(teacher_args, options, teacher_dir)
        teacher_args.split_path = str(split_path)
        teacher_args.teacher_ckpt = str(teacher_ckpt)
        teacher_args.open_val_ratio = 0.0
        fit_teacher(teacher_args)

    results: dict[str, Any] = {}
    for arm in arm_names:
        train_args = load_args(sources[1], "train_student", arm_dirs[arm], options.seed)
        apply_overrides(train_args, options, arm_dirs[arm])
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(arm_dirs[arm] / "student.pt")
        train_args.alpha_rejection_feat_kd = 0.0
        train_args.alpha_rejection_kd = 0.0
        train_args.alpha_rejection_relational_kd = (
            0.0 if arm == "control" else float(options.alpha)
        )
        fit_student(train_args)

        discover_args = load_args(sources[2], "discover", discover_dirs[arm], options.seed)
        apply_overrides(discover_args, options, discover_dirs[arm])
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(arm_dirs[arm] / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.open_val_ratio = 0.0
        discover_args.cluster_k = "auto"
        discover(discover_args)
        results[arm] = {
            "train_dir": str(arm_dirs[arm]),
            "discover_dir": str(discover_dirs[arm]),
            "train_config": read_json(arm_dirs[arm] / "train_student_config.json"),
            "discover_config": read_json(discover_dirs[arm] / "discover_config.json"),
            "metrics": read_json(discover_dirs[arm] / "discovery_report.json"),
            "history": read_json(arm_dirs[arm] / "train_history.json"),
        }

    control = results["control"]["train_config"]
    relational = results["relational"]["train_config"]
    allowed = {
        "alpha_rejection_relational_kd",
        "work_dir",
        "student_ckpt",
        "best_epoch",
        "best_val_known_acc",
        "best_checkpoint_selection_score",
        "best_open_checkpoint_stats",
        "transform_profile_requested",
    }
    differences = {
        key: (control.get(key), relational.get(key))
        for key in sorted(set(control) | set(relational))
        if control.get(key) != relational.get(key)
    }
    unexpected = {key: value for key, value in differences.items() if key not in allowed}
    if unexpected:
        raise RuntimeError(f"unexpected training configuration differences: {unexpected}")

    expected_epochs = int(results["control"]["train_config"].get("epochs", 0))
    for arm in arm_names:
        history = results[arm]["history"]
        if len(history) != expected_epochs:
            raise RuntimeError(f"incomplete training history for {arm}: {len(history)} != {expected_epochs}")
        report = results[arm]["metrics"]
        if report.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"test labels entered training/calibration for {arm}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"known-validation calibration failed for {arm}")
    treatment_losses = [float(epoch.get("train", {}).get("rejection_relational_kd", 0.0)) for epoch in results["relational"]["history"]]
    treatment_gradients = [float(epoch.get("train", {}).get("gradient_audit_grad_norm_rejection_relational_kd_term", 0.0)) for epoch in results["relational"]["history"]]
    if any(value <= 0.0 for value in treatment_losses):
        raise RuntimeError(f"relational rejection KD inactive: {treatment_losses}")
    if any(value <= 0.0 for value in treatment_gradients):
        raise RuntimeError(f"relational rejection KD gradient inactive: {treatment_gradients}")

    result = {
        "question": "Does relational KD from teacher classification geometry improve the student rejection representation?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "alpha_rejection_relational_kd: 0 vs fixed pilot coefficient",
        "alpha_rejection_relational_kd": float(options.alpha),
        "pool_or_test_labels_used_for_training_or_selection": False,
        "training_config_differences": differences,
        "treatment_relation_loss_by_epoch": treatment_losses,
        "treatment_relation_gradient_norm_by_epoch": treatment_gradients,
        "arms": results,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "config_differences": differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
