"""Compare existing label-free auto-K selectors on one fresh split/checkpoint."""

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

SEED = 2112
TAG = f"auto_k_selector_s{SEED}"
SELECTORS = ("silhouette", "composite", "stability", "gmm_bic")
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
    for path in SOURCE.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{SEED}/cifar100_60_40_semantic_hard.json"
    if split_path.exists():
        split = read_json(split_path)
        if split.get("metadata", {}).get("seed") != SEED or split.get("protocol") != "semantic_hard":
            raise ValueError(f"split mismatch: {split_path}")
    else:
        src = read_json(SOURCE["student"])
        data_root = Path(str(src.get("data_root", ".\\data")))
        if not data_root.is_absolute():
            data_root = (ROOT / data_root).resolve()
        names, coarse, mapping = read_cifar100_taxonomy(data_root)
        split = build_protocols(names, coarse, mapping, SEED)["semantic_hard"]
        write_json(split_path, split)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard 60/40 split")

    teacher_dir = ROOT / "runs" / f"{TAG}_teacher"
    student_dir = ROOT / "runs" / f"{TAG}_student"
    teacher_ckpt, student_ckpt = teacher_dir / "teacher.pt", student_dir / "student.pt"
    outputs = {name: ROOT / "runs" / f"{TAG}_{name}" for name in SELECTORS}
    result_path = ROOT / "analysis" / f"{TAG}.json"
    if result_path.exists():
        raise FileExistsError(f"result already exists: {result_path}")
    if any(path.exists() for path in (*outputs.values(),)) and not (teacher_ckpt.is_file() and student_ckpt.is_file()):
        raise FileExistsError("partial run exists; refusing to overwrite")

    if not teacher_ckpt.is_file():
        args = load_args(SOURCE["teacher"], "train_teacher", teacher_dir)
        args.split_path, args.teacher_ckpt = str(split_path), str(teacher_ckpt)
        fit_teacher(args)
    if not student_ckpt.is_file():
        args = load_args(SOURCE["student"], "train_student", student_dir)
        args.split_path, args.teacher_ckpt = str(split_path), str(teacher_ckpt)
        args.student_ckpt = str(student_ckpt)
        args.epochs = 4
        fit_student(args)

    results: dict[str, Any] = {}
    for selector in SELECTORS:
        out_dir = outputs[selector]
        if out_dir.exists():
            raise FileExistsError(f"refusing to reuse existing selector output: {out_dir}")
        args = load_args(SOURCE["discover"], "discover", out_dir)
        args.split_path, args.student_ckpt = str(split_path), str(student_ckpt)
        args.score_mode = "feature_rejector"
        args.threshold_policy, args.target_known_coverage = "known_coverage", 0.95
        args.open_val_ratio, args.cluster_k = 0.0, "auto"
        args.skip_clustering = False
        args.cluster_feature, args.cluster_selection = "feature_pca", selector
        print(f"[discover] seed={SEED} selector={selector}", flush=True)
        discover(args)
        report = read_json(out_dir / "discovery_report.json")
        config = read_json(out_dir / "discover_config.json")
        if report.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"test-label flow audit failed for {selector}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"known validation calibration outside tolerance for {selector}")
        results[selector] = {"config": config, "metrics": report, "output": str(out_dir)}

    ref_cfg = results[SELECTORS[0]]["config"]
    allowed_config_differences = {"cluster_selection", "work_dir"}
    for selector in SELECTORS[1:]:
        cfg = results[selector]["config"]
        diffs = {
            k: (ref_cfg.get(k), cfg.get(k))
            for k in sorted(set(ref_cfg) | set(cfg))
            if ref_cfg.get(k) != cfg.get(k)
        }
        if set(diffs) - allowed_config_differences:
            raise RuntimeError(f"unexpected config differences for {selector}: {diffs}")
        results[selector]["config_differences_from_silhouette"] = diffs

    invariant = ("auroc", "aupr", "fpr95", "oscr", "known_accept_rate", "unknown_reject_rate",
                 "known_class_accuracy_after_accept", "accepted_correct_fraction_of_all_known",
                 "calibration_known_accept_rate")
    baseline = results["silhouette"]["metrics"]
    for selector in SELECTORS[1:]:
        changed = {k: (baseline.get(k), results[selector]["metrics"].get(k))
                   for k in invariant if baseline.get(k) != results[selector]["metrics"].get(k)}
        if changed:
            raise RuntimeError(f"selector changed rejection/known metrics: {selector}: {changed}")
    result = {
        "protocol": "four existing auto-K selectors; fresh split; one frozen model",
        "seed": SEED,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "test_or_mixed_labels_used_for_training_selection_calibration_or_selector_choice": False,
        "test_labels_used_only_for_final_retrospective_metrics": True,
        "fixed_cluster_feature": "feature_pca",
        "single_changed_factor": "cluster_selection",
        "invariant_metric_audit": {k: baseline.get(k) for k in invariant},
        "arms": results,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
