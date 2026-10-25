"""Fresh paired 4-versus-8 epoch audit for the shared rejection branch."""

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


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
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
    if path.exists():
        split = read_json(path)
        if split.get("metadata", {}).get("seed") != seed or split.get("protocol") != "semantic_hard":
            raise ValueError(f"existing split does not match seed/protocol: {path}")
        return
    names, coarse, mapping = read_cifar100_taxonomy(data_root)
    write_json(path, build_protocols(names, coarse, mapping, seed)["semantic_hard"])


def complete(run_dir: Path, discover_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            run_dir / "train_student_config.json",
            run_dir / "train_history.json",
            discover_dir / "discovery_report.json",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2107)
    parser.add_argument("--tag", default="rejection_epoch_budget_s2107")
    parser.add_argument(
        "--source-teacher-config",
        type=Path,
        default=Path("runs/rejection_feature_kd_mc_epistemic_s2105_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--source-student-config",
        type=Path,
        default=Path("analysis/rejection_feature_kd_mc_epistemic_s2105_student_config.json"),
    )
    parser.add_argument(
        "--source-discover-config",
        type=Path,
        default=Path("runs/rejection_feature_kd_mc_epistemic_s2105_uniform_weight_discover/discover_config.json"),
    )
    options = parser.parse_args()

    teacher_source = resolve(options.source_teacher_config)
    student_source = resolve(options.source_student_config)
    discover_source = resolve(options.source_discover_config)
    for path in (teacher_source, student_source, discover_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_student = read_json(student_source)
    if source_student.get("independent_rejection_encoder") is not True:
        raise ValueError("source student must use the independent rejection branch")
    if source_student.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student must use shared_backbone initialization")
    if float(source_student.get("alpha_rejection_feat_kd", 0.0)) != 0.0:
        raise ValueError("epoch-budget audit requires feature KD disabled")
    if float(source_student.get("alpha_vos", 0.0)) or float(source_student.get("alpha_vos_rejection", 0.0)):
        raise ValueError("epoch-budget audit requires VOS disabled")
    if int(source_student.get("epochs", 0)) != 4:
        raise ValueError("source student must be the registered four-epoch recipe")

    seed = int(options.seed)
    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{seed}" / "cifar100_60_40_semantic_hard.json"
    data_root = resolve(Path(str(source_student.get("data_root", ".\\data"))))
    make_split(seed, split_path, data_root)
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard CIFAR-100 60/40 split")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    arms = {"four_epoch": 4, "eight_epoch": 8}
    run_dirs = {arm: ROOT / "runs" / f"{options.tag}_{arm}" for arm in arms}
    discover_dirs = {arm: Path(f"{run_dirs[arm]}_discover") for arm in arms}
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    student_config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"

    if result_path.is_file():
        print(json.dumps(read_json(result_path), indent=2, ensure_ascii=False))
        return
    if any(path.exists() for path in (teacher_dir, student_config_path, *run_dirs.values(), *discover_dirs.values())):
        existing_complete = teacher_ckpt.is_file() and student_config_path.is_file() and all(
            complete(run_dirs[arm], discover_dirs[arm]) for arm in arms
        )
        if not existing_complete:
            raise FileExistsError("refusing to overwrite incomplete epoch-budget artifacts")

    if not teacher_ckpt.is_file():
        teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, seed)
        teacher_args.split_path = str(split_path)
        teacher_args.teacher_ckpt = str(teacher_ckpt)
        teacher_args.rejection_feature_dim = int(source_student["rejection_feature_dim"])
        fit_teacher(teacher_args)

    if not student_config_path.is_file():
        write_json(
            student_config_path,
            {
                "source_student_config": str(student_source),
                "single_changed_factor": "epochs: 4 vs 8",
                "seed": seed,
                "split_path": str(split_path),
                "teacher_checkpoint": str(teacher_ckpt),
            },
        )

    for arm, epochs in arms.items():
        if complete(run_dirs[arm], discover_dirs[arm]):
            continue
        train_args = load_args(student_source, "train_student", run_dirs[arm], seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(run_dirs[arm] / "student.pt")
        train_args.epochs = epochs
        train_args.rejection_feature_dim = int(source_student["rejection_feature_dim"])
        print(f"[student] arm={arm}, epochs={epochs}", flush=True)
        fit_student(train_args)

        discover_args = load_args(discover_source, "discover", discover_dirs[arm], seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(run_dirs[arm] / "student.pt")
        discover_args.independent_rejection_encoder = True
        discover_args.independent_rejection_encoder_init = "shared_backbone"
        discover_args.rejection_feature_dim = int(source_student["rejection_feature_dim"])
        discover_args.score_mode = "feature_rejector"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.open_val_ratio = 0.0
        discover_args.cluster_k = "auto"
        discover_args.skip_clustering = False
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)

    results = {
        arm: {
            "train_config": read_json(run_dirs[arm] / "train_student_config.json"),
            "history": read_json(run_dirs[arm] / "train_history.json"),
            "metrics": read_json(discover_dirs[arm] / "discovery_report.json"),
            "train_dir": str(run_dirs[arm]),
            "discover_dir": str(discover_dirs[arm]),
        }
        for arm in arms
    }
    control = results["four_epoch"]["train_config"]
    treatment = results["eight_epoch"]["train_config"]
    allowed = {"epochs", "work_dir", "student_ckpt", "best_epoch", "best_val_known_acc", "best_checkpoint_selection_score"}
    differences = {
        key: (control.get(key), treatment.get(key))
        for key in sorted(set(control) | set(treatment))
        if control.get(key) != treatment.get(key)
    }
    unexpected = {key: value for key, value in differences.items() if key not in allowed}
    if unexpected:
        raise RuntimeError(f"unexpected training configuration differences: {unexpected}")
    for arm in arms:
        report = results[arm]["metrics"]
        if report.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"test labels entered training/calibration for {arm}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"known validation coverage out of tolerance for {arm}")
        if report.get("cluster_k") is None:
            raise RuntimeError(f"automatic clustering missing for {arm}")

    result = {
        "question": "Does extending shared-backbone rejection training from 4 to 8 epochs improve semantic-hard rejection?",
        "seed": seed,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "epochs: 4 vs 8",
        "test_or_mixed_labels_used_for_training_selection_or_calibration": False,
        "training_config_differences": differences,
        "arms": results,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "config_differences": differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
