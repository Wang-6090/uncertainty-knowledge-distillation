"""Fresh semantic-hard pair for combined rejection-branch Outlier Exposure.

The sole algorithmic factor is the auxiliary CIFAR-10 uniform-logit loss on
the independent rejection head.  Both arms already use the same CIFAR-10
feature-margin exposure term; the treatment additionally enables the fixed
uniform-logit term.  This runner deliberately does not inspect mixed-pool or
test labels for training, checkpoint selection, calibration, or method choice.
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

from scripts.make_cifar100_splits import build_protocols, read_cifar100_taxonomy  # noqa: E402
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def ensure_split(seed: int, path: Path, data_root: Path) -> None:
    if path.is_file():
        split = read_json(path)
        if split.get("metadata", {}).get("seed") != seed:
            raise ValueError(f"existing split seed mismatch: {path}")
        if split.get("protocol") != "semantic_hard":
            raise ValueError(f"existing split is not semantic_hard: {path}")
        return
    names, coarse, mapping = read_cifar100_taxonomy(data_root)
    write_json(path, build_protocols(names, coarse, mapping, seed)["semantic_hard"])


def complete(train_dir: Path, discover_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            train_dir / "train_student_config.json",
            train_dir / "train_history.json",
            train_dir / "student.pt",
            discover_dir / "discover_config.json",
            discover_dir / "discovery_report.json",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2120)
    parser.add_argument("--tag", default="rejection_oe_semantic_hard_s2120")
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
        raise ValueError("source student must use the independent rejection encoder")
    if source_student.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student must use shared_backbone rejection initialization")
    if int(source_student.get("rejection_feature_dim", 0)) <= 0:
        raise ValueError("source student must have a rejection embedding")
    if source_student.get("outlier_dataset") != "cifar10":
        raise ValueError("this protocol requires CIFAR-10 external outliers")
    if float(source_student.get("alpha_outlier_rejection_feature_margin", 0.0)) != 0.05:
        raise ValueError("this protocol fixes the rejection feature-margin weight at 0.05")
    if float(source_student.get("outlier_rejection_feature_margin", 0.0)) != 0.2:
        raise ValueError("this protocol fixes the rejection feature margin at 0.2")
    if float(source_student.get("alpha_outlier_rejection_uniform", 0.0)) != 0.0:
        raise ValueError("source config must be the margin-only control")

    seed = int(options.seed)
    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{seed}" / "cifar100_60_40_semantic_hard.json"
    ensure_split(seed, split_path, resolve(Path(str(source_student.get("data_root", ".\\data")))))
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected a semantic-hard CIFAR-100 60/40 split")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    arms = {"margin_only": 0.0, "margin_plus_uniform": 0.05}
    train_dirs = {name: ROOT / "runs" / f"{options.tag}_{name}" for name in arms}
    discover_dirs = {name: ROOT / "runs" / f"{options.tag}_{name}_discover" for name in arms}

    if result_path.is_file():
        print(json.dumps(read_json(result_path), indent=2, ensure_ascii=False))
        return
    if any(path.exists() for path in (teacher_dir, student_config_path, *train_dirs.values(), *discover_dirs.values())):
        raise FileExistsError("refusing to overwrite existing or incomplete fresh-pair artifacts")

    teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    base = {
        "source_student_config": str(student_source),
        "single_changed_factor": "alpha_outlier_rejection_uniform",
        "control": 0.0,
        "treatment": 0.05,
        "fixed_outlier_feature_margin_weight": 0.05,
        "fixed_outlier_feature_margin": 0.2,
        "seed": seed,
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
    }
    write_json(student_config_path, base)

    reports: dict[str, Any] = {}
    for arm, uniform_weight in arms.items():
        train_dir = train_dirs[arm]
        discover_dir = discover_dirs[arm]
        train_args = load_args(student_source, "train_student", train_dir, seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.outlier_data_root = str(resolve(Path(str(source_student.get("data_root", ".\\data")))))
        train_args.outlier_download = False
        train_args.alpha_outlier_rejection_uniform = float(uniform_weight)
        print(f"[student] arm={arm}, alpha_outlier_rejection_uniform={uniform_weight:g}", flush=True)
        fit_student(train_args)

        history = read_json(train_dir / "train_history.json")
        values = [float(epoch["train"].get("outlier_rejection_uniform", 0.0)) for epoch in history]
        if arm == "margin_plus_uniform" and not any(value > 0.0 for value in values):
            raise RuntimeError(f"uniform rejection OE was inactive: {values}")
        if arm == "margin_only" and any(abs(value) > 1e-12 for value in values):
            raise RuntimeError(f"control unexpectedly enabled uniform rejection OE: {values}")

        discover_args = load_args(discover_source, "discover", discover_dir, seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.outlier_data_root = str(resolve(Path(str(source_student.get("data_root", ".\\data")))))
        discover_args.score_mode = "feature_rejector"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.cluster_k = "auto"
        discover_args.skip_clustering = False
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)
        reports[arm] = {
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "uniform_term_by_epoch": values,
            "train_config": read_json(train_dir / "train_student_config.json"),
            "metrics": read_json(discover_dir / "discovery_report.json"),
        }

    result = {
        "protocol": options.tag,
        "question": "Does adding fixed uniform-logit rejection Outlier Exposure to the current external feature-margin exposure improve semantic-hard discovery?",
        "seed": seed,
        "split_path": str(split_path),
        "single_changed_factor": "alpha_outlier_rejection_uniform",
        "control": "margin-only external CIFAR-10 rejection exposure",
        "treatment": "margin-plus-uniform external CIFAR-10 rejection exposure",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "mixed_pool_labels_used_for_training_or_calibration": False,
        "arms": reports,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2))


if __name__ == "__main__":
    main()
