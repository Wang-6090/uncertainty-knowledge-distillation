"""Fresh paired 4-vs-8 epoch test with feature_pca clustering fixed."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_cifar100_splits import build_protocols, read_cifar100_taxonomy  # noqa: E402
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402

SEED = 2113
TAG = f"rejection_epoch_feature_pca_s{SEED}_v2"
SOURCE = {
    "teacher": ROOT / "runs/rejection_epoch_feature_pca_s2113_teacher/train_teacher_config.json",
    "student": ROOT / "analysis/rejection_feature_kd_mc_epistemic_s2105_student_config.json",
    "discover": ROOT / "runs/rejection_epoch_budget_s2109_four_epoch_discover/discover_config.json",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_args(path: Path, command: str, work_dir: Path):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = SEED
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    for path in SOURCE.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{SEED}/cifar100_60_40_semantic_hard.json"
    if split_path.exists():
        split = read_json(split_path)
        if split.get("metadata", {}).get("seed") != SEED or split.get("protocol") != "semantic_hard":
            raise ValueError(f"split mismatch: {split_path}")
    else:
        cfg = read_json(SOURCE["student"])
        data_root = Path(str(cfg.get("data_root", ".\\data")))
        if not data_root.is_absolute():
            data_root = (ROOT / data_root).resolve()
        names, coarse, mapping = read_cifar100_taxonomy(data_root)
        split = build_protocols(names, coarse, mapping, SEED)["semantic_hard"]
        write_json(split_path, split)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard CIFAR-100 60/40 split")

    teacher_dir = ROOT / "runs" / f"rejection_epoch_feature_pca_s{SEED}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_dirs = {name: ROOT / "runs" / f"{TAG}_{name}" for name in ("four_epoch", "eight_epoch")}
    discover_dirs = {name: ROOT / "runs" / f"{TAG}_{name}_discover" for name in student_dirs}
    result_path = ROOT / "analysis" / f"{TAG}.json"
    if result_path.exists():
        raise FileExistsError(f"result already exists: {result_path}")
    student_cfg_source = read_json(SOURCE["student"])
    rejection_dim = int(student_cfg_source["rejection_feature_dim"])
    if int(read_json(teacher_dir / "train_teacher_config.json").get("rejection_feature_dim", 0)) != rejection_dim:
        raise ValueError("split-matched teacher and registered student rejection dimensions differ")

    if not teacher_ckpt.is_file():
        args = load_args(SOURCE["teacher"], "train_teacher", teacher_dir)
        args.split_path, args.teacher_ckpt = str(split_path), str(teacher_ckpt)
        args.rejection_feature_dim = rejection_dim
        fit_teacher(args)

    outputs: dict[str, Any] = {}
    for name, epochs in (("four_epoch", 4), ("eight_epoch", 8)):
        student_dir, discover_dir = student_dirs[name], discover_dirs[name]
        student_ckpt = student_dir / "student.pt"
        if student_dir.exists() and not student_ckpt.is_file():
            raise FileExistsError(f"incomplete student run: {student_dir}")
        if not student_ckpt.is_file():
            args = load_args(SOURCE["student"], "train_student", student_dir)
            args.split_path, args.teacher_ckpt = str(split_path), str(teacher_ckpt)
            args.student_ckpt, args.epochs = str(student_ckpt), epochs
            args.rejection_feature_dim = rejection_dim
            fit_student(args)
        if discover_dir.exists():
            raise FileExistsError(f"refusing to reuse discovery output: {discover_dir}")
        args = load_args(SOURCE["discover"], "discover", discover_dir)
        args.split_path, args.student_ckpt = str(split_path), str(student_ckpt)
        args.score_mode = "feature_rejector"
        args.threshold_policy, args.target_known_coverage = "known_coverage", 0.95
        args.open_val_ratio, args.cluster_k = 0.0, "auto"
        args.cluster_feature, args.cluster_selection = "feature_pca", "silhouette"
        args.rejection_feature_dim = rejection_dim
        args.skip_clustering = False
        print(f"[student/discover] arm={name} epochs={epochs}", flush=True)
        discover(args)
        train_cfg = read_json(student_dir / "train_student_config.json")
        history = read_json(student_dir / "train_history.json")
        report = read_json(discover_dir / "discovery_report.json")
        discover_cfg = read_json(discover_dir / "discover_config.json")
        if report.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"test-label flow audit failed: {name}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"calibration outside tolerance: {name}")
        outputs[name] = {"train_config": train_cfg, "history": history, "discover_config": discover_cfg,
                         "metrics": report, "student_dir": str(student_dir), "discover_dir": str(discover_dir)}

    c, t = outputs["four_epoch"]["train_config"], outputs["eight_epoch"]["train_config"]
    allowed_train_diffs = {"epochs", "work_dir", "student_ckpt", "best_epoch", "best_val_known_acc",
                           "best_checkpoint_selection_score"}
    train_diffs = {k: (c.get(k), t.get(k)) for k in sorted(set(c) | set(t)) if c.get(k) != t.get(k)}
    if set(train_diffs) - allowed_train_diffs:
        raise RuntimeError(f"unexpected training config differences: {train_diffs}")
    dc, dt = outputs["four_epoch"]["discover_config"], outputs["eight_epoch"]["discover_config"]
    allowed_discover_diffs = {"work_dir", "student_ckpt"}
    discover_diffs = {k: (dc.get(k), dt.get(k)) for k in sorted(set(dc) | set(dt)) if dc.get(k) != dt.get(k)}
    if set(discover_diffs) - allowed_discover_diffs:
        raise RuntimeError(f"unexpected discovery config differences: {discover_diffs}")
    result = {
        "protocol": "fresh paired four-vs-eight epochs; feature_pca fixed",
        "seed": SEED,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "test_or_mixed_labels_used_for_training_selection_or_calibration": False,
        "test_labels_used_only_for_final_retrospective_metrics": True,
        "single_changed_training_factor": "epochs: 4 vs 8",
        "fixed_cluster_feature": "feature_pca",
        "training_config_differences": train_diffs,
        "discovery_config_differences": discover_diffs,
        "arms": outputs,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "train_differences": train_diffs,
                      "discover_differences": discover_diffs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
