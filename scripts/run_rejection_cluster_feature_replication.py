"""Fresh-split, frozen-checkpoint replication of clustering input choice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_cifar100_splits import build_protocols, read_cifar100_taxonomy  # noqa: E402
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


SEED = 2110
SOURCE = {
    "teacher": ROOT / "runs/rejection_epoch_budget_s2109_teacher/train_teacher_config.json",
    "student": ROOT / "runs/rejection_epoch_budget_s2109_four_epoch/train_student_config.json",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    seed = parser.parse_args().seed
    if seed < SEED:
        raise ValueError("replication seeds must be 2110 or newer")
    tag = f"rejection_cluster_feature_replication_s{seed}"
    for path in SOURCE.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{seed}/cifar100_60_40_semantic_hard.json"
    if split_path.exists():
        split = read_json(split_path)
        if split.get("metadata", {}).get("seed") != seed or split.get("protocol") != "semantic_hard":
            raise ValueError(f"split does not match locked protocol: {split_path}")
    else:
        source_student = read_json(SOURCE["student"])
        data_root = Path(str(source_student.get("data_root", ".\\data")))
        if not data_root.is_absolute():
            data_root = (ROOT / data_root).resolve()
        names, coarse, mapping = read_cifar100_taxonomy(data_root)
        split = build_protocols(names, coarse, mapping, seed)["semantic_hard"]
        write_json(split_path, split)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard CIFAR-100 60/40 split")

    teacher_dir = ROOT / "runs" / f"{tag}_teacher"
    student_dir = ROOT / "runs" / f"{tag}_student"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_ckpt = student_dir / "student.pt"
    discover_dirs = {name: ROOT / "runs" / f"{tag}_{name}" for name in ("projection", "feature")}
    result_path = ROOT / "analysis" / f"{tag}.json"
    if result_path.exists():
        raise FileExistsError(f"result already exists; refusing silent reuse: {result_path}")
    artifacts = [teacher_dir, student_dir, *discover_dirs.values()]
    if any(path.exists() for path in artifacts) and not (teacher_ckpt.is_file() and student_ckpt.is_file()):
        raise FileExistsError("partial prior run exists; refusing to overwrite")

    if not teacher_ckpt.is_file():
        args = load_args(SOURCE["teacher"], "train_teacher", teacher_dir)
        args.split_path = str(split_path)
        args.teacher_ckpt = str(teacher_ckpt)
        fit_teacher(args)
    if not student_ckpt.is_file():
        args = load_args(SOURCE["student"], "train_student", student_dir)
        args.split_path = str(split_path)
        args.teacher_ckpt = str(teacher_ckpt)
        args.student_ckpt = str(student_ckpt)
        args.epochs = 4
        fit_student(args)

    outputs: dict[str, Any] = {}
    for name, cluster_feature in (("projection", "projection_pca"), ("feature", "feature_pca")):
        out_dir = discover_dirs[name]
        if out_dir.exists():
            raise FileExistsError(f"refusing to reuse existing discovery output: {out_dir}")
        args = load_args(SOURCE["discover"], "discover", out_dir)
        args.split_path = str(split_path)
        args.student_ckpt = str(student_ckpt)
        args.score_mode = "feature_rejector"
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.cluster_feature = cluster_feature
        print(f"[discover] seed={seed} cluster_feature={cluster_feature}", flush=True)
        discover(args)
        report = read_json(out_dir / "discovery_report.json")
        config = read_json(out_dir / "discover_config.json")
        if report.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"label-flow audit failed: {out_dir}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"known validation calibration outside tolerance: {out_dir}")
        outputs[name] = {"config": config, "metrics": report, "output": str(out_dir)}

    left, right = outputs["projection"], outputs["feature"]
    allowed = {"cluster_feature", "work_dir"}
    diffs = {
        key: (left["config"].get(key), right["config"].get(key))
        for key in sorted(set(left["config"]) | set(right["config"]))
        if left["config"].get(key) != right["config"].get(key)
    }
    unexpected = {key: value for key, value in diffs.items() if key not in allowed}
    if unexpected:
        raise RuntimeError(f"unexpected condition differences: {unexpected}")
    identical_metrics = (
        "auroc", "aupr", "fpr95", "oscr", "known_accept_rate", "unknown_reject_rate",
        "known_class_accuracy_after_accept", "accepted_correct_fraction_of_all_known",
        "calibration_known_accept_rate",
    )
    metric_differences = {
        key: (left["metrics"].get(key), right["metrics"].get(key))
        for key in identical_metrics
        if left["metrics"].get(key) != right["metrics"].get(key)
    }
    if metric_differences:
        raise RuntimeError(f"clustering input changed non-clustering metrics: {metric_differences}")
    outputs["config_differences"] = diffs
    outputs["metric_identity_audit"] = {key: left["metrics"].get(key) for key in identical_metrics}
    result = {
        "protocol": "independent fresh-split feature_pca replication",
        "seed": seed,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "training_performed": True,
        "test_and_mixed_labels_used_for_training_selection_or_calibration": False,
        "test_labels_used_only_for_retrospective_metrics": True,
        "single_changed_factor": "cluster_feature",
        "arms": outputs,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "config_differences": diffs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
