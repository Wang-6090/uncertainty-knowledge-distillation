"""Fresh-split pair: confidence-targeted vs teacher-uncertainty-distilled head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_cifar100_splits import build_protocols, read_cifar100_taxonomy  # noqa: E402
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def make_split(seed: int, path: Path, data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        split = read_json(path)
        if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != seed:
            raise ValueError(f"existing split does not match protocol: {path}")
        return
    names, coarse, mapping = read_cifar100_taxonomy(data_root)
    write_json(path, build_protocols(names, coarse, mapping, seed)["semantic_hard"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--tag", default="teacher_uncertainty_transfer_s2101")
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    options = parser.parse_args()
    sources = [p if p.is_absolute() else (ROOT / p).resolve() for p in (
        options.source_teacher_config, options.source_student_config, options.source_discover_config
    )]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    seed = int(options.seed)
    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{seed}" / "cifar100_60_40_semantic_hard.json"
    base_student = read_json(sources[1])
    data_root = (ROOT / str(base_student.get("data_root", ".\\data"))).resolve()
    make_split(seed, split_path, data_root)
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard CIFAR-100 60/40 split")
    if base_student.get("checkpoint_selection") != "known_acc":
        raise ValueError("student source must select checkpoints using known validation accuracy")
    if base_student.get("kd_uncertainty_source", "head") != "head":
        raise ValueError("teacher uncertainty transfer requires kd_uncertainty_source=head")
    if int(read_json(sources[0]).get("rejection_feature_dim", 0)) != 0:
        raise ValueError("protocol expects the teacher's learned uncertainty head, without an independent rejection branch")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    artifacts = [teacher_dir, student_config_path, result_path]
    arms = ["confidence", "teacher"]
    run_dirs = [ROOT / "runs" / f"{options.tag}_{arm}" for arm in arms]
    discover_dirs = [ROOT / "runs" / f"{options.tag}_{arm}_discover" for arm in arms]
    if any(p.exists() for p in artifacts + run_dirs + discover_dirs):
        raise FileExistsError("refusing to overwrite any existing experiment artifact")

    teacher_args = load_args(sources[0], "train_teacher", teacher_dir, seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    teacher_args.open_val_ratio = 0.0
    print("[teacher] fit one split-matched teacher shared by both arms", flush=True)
    fit_teacher(teacher_args)

    base_student.update({
        "split_path": str(split_path),
        "seed": seed,
        "model_seed": None,
        "open_val_ratio": 0.0,
        "uncertainty_target_mode": "confidence",
    })
    write_json(student_config_path, base_student)

    results = {}
    for arm, target_mode, run_dir, discover_dir in zip(
        arms, arms, run_dirs, discover_dirs
    ):
        train_args = load_args(student_config_path, "train_student", run_dir, seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(run_dir / "student.pt")
        train_args.uncertainty_target_mode = target_mode
        print(f"[student] arm={arm}, uncertainty_target_mode={target_mode}", flush=True)
        fit_student(train_args)

        discover_args = load_args(sources[2], "discover", discover_dir, seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(run_dir / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.open_val_ratio = 0.0
        discover_args.skip_clustering = False
        discover_args.cluster_k = "auto"
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)
        results[arm] = {
            "train_dir": str(run_dir),
            "discover_dir": str(discover_dir),
            "train_config": read_json(run_dir / "train_student_config.json"),
            "discover_config": read_json(discover_dir / "discover_config.json"),
            "metrics": read_json(discover_dir / "discovery_report.json"),
            "history": read_json(run_dir / "train_history.json"),
        }

    a, b = results["confidence"]["train_config"], results["teacher"]["train_config"]
    excluded = {"uncertainty_target_mode", "work_dir", "student_ckpt", "best_epoch",
                "best_val_known_acc", "best_checkpoint_selection_score",
                "best_open_checkpoint_stats", "transform_profile_requested"}
    diffs = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    unexpected = {k: v for k, v in diffs.items() if k not in excluded}
    if unexpected:
        raise RuntimeError(f"unexpected training configuration differences: {unexpected}")
    if any(x["metrics"].get("rejector_protocol", {}).get("hard_labels_used") is not False for x in results.values()):
        raise RuntimeError("hard labels were used by a rejector protocol")
    if any(x["metrics"].get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False for x in results.values()):
        raise RuntimeError("test labels entered training or calibration")
    for arm, report in results.items():
        if report["metrics"].get("cluster_k") is None or report["metrics"].get("clustering_skipped"):
            raise RuntimeError(f"automatic clustering did not complete for arm {arm}")
        coverage = report["metrics"].get("calibration_known_accept_rate")
        if coverage is None or abs(float(coverage) - 0.95) > 0.02:
            raise RuntimeError(f"known-validation coverage calibration out of tolerance: {arm}={coverage}")
        history = report["history"]
        if len(history) != int(base_student.get("epochs", len(history))):
            raise RuntimeError(f"incomplete training history for arm {arm}")
        if any(float(entry.get("train", {}).get("unc", 0.0)) <= 0.0 for entry in history):
            raise RuntimeError(f"uncertainty target loss was inactive for arm {arm}")

    result = {
        "question": "Does direct transfer of the teacher uncertainty head improve semantic-hard novel-class rejection over the current label-derived uncertainty target?",
        "seed": seed,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "uncertainty_target_mode: confidence vs teacher-head MSE target",
        "teacher_uncertainty_source": "frozen teacher uncertainty head on known training images; detached MSE target for the student rejection uncertainty head",
        "pool_or_test_labels_used_for_training_or_selection": False,
        "training_config_differences": diffs,
        "arms": results,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "config_differences": diffs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
