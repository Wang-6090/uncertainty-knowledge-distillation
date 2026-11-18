"""Fresh full-metric replication of the fixed late rejection-CE decay.

The pair changes only the rejection known-class CE schedule.  Both arms share
the same split-matched teacher, student configuration, mixed-pool NNPU
training, independent shared-backbone rejection encoder, checkpoint rule,
feature-rejector detector, and known-validation-only 95% calibration.  The
full clustering report is retained so the training intervention is judged on
rejection, known utility, overlap, candidate quality, automatic K, and
unknown-class clustering together.

No mixed-pool or test labels are read by the runner for training,
checkpointing, calibration, or method selection.
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
    fine_names, coarse_names, mapping = read_cifar100_taxonomy(data_root)
    write_json(path, build_protocols(fine_names, coarse_names, mapping, seed)["semantic_hard"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2121)
    parser.add_argument("--tag", default="rejection_ce_decay_full_s2121")
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
    parser.add_argument("--decay-start-epoch", type=int, default=3)
    parser.add_argument("--decay-epochs", type=int, default=2)
    parser.add_argument("--final-scale", type=float, default=0.25)
    options = parser.parse_args()

    if options.decay_start_epoch <= 0 or options.decay_epochs <= 0:
        raise ValueError("decay start and duration must be positive")
    if not 0.0 <= options.final_scale <= 1.0:
        raise ValueError("final scale must be in [0, 1]")

    teacher_source = resolve(options.source_teacher_config)
    student_source = resolve(options.source_student_config)
    discover_source = resolve(options.source_discover_config)
    for path in (teacher_source, student_source, discover_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_student = read_json(student_source)
    if source_student.get("discovery_pool_mode") != "mixed":
        raise ValueError("source student must use a mixed discovery pool")
    if source_student.get("independent_rejection_encoder") is not True:
        raise ValueError("source student must use an independent rejection encoder")
    if source_student.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student must use shared_backbone rejection initialization")
    if source_student.get("alpha_vos", 0.0) or source_student.get("alpha_vos_rejection", 0.0):
        raise ValueError("VOS must be disabled for this single-factor comparison")
    if float(source_student.get("alpha_rejection_known_ce", 0.0)) <= 0.0:
        raise ValueError("the rejection known-class CE term must be active")

    seed = int(options.seed)
    split_path = ROOT / f"splits/semantic_hard/seed{seed}" / "cifar100_60_40_semantic_hard.json"
    ensure_split(seed, split_path, resolve(Path(str(source_student.get("data_root", ".\\data")))))
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected a semantic-hard CIFAR-100 60/40 split")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    arms = {
        "static_control": {"start": 0, "duration": 0, "scale": 1.0},
        "late_decay": {
            "start": options.decay_start_epoch,
            "duration": options.decay_epochs,
            "scale": options.final_scale,
        },
    }
    train_dirs = {name: ROOT / "runs" / f"{options.tag}_{name}" for name in arms}
    discover_dirs = {name: ROOT / "runs" / f"{options.tag}_{name}_discover" for name in arms}

    if result_path.is_file():
        print(json.dumps(read_json(result_path), indent=2, ensure_ascii=False))
        return
    if any(path.exists() for path in (teacher_dir, config_path, *train_dirs.values(), *discover_dirs.values())):
        raise FileExistsError("refusing to overwrite existing or incomplete fresh-pair artifacts")

    teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    base_config = dict(source_student)
    base_config.update(
        {
            "split_path": str(split_path),
            "seed": seed,
            "rejection_known_ce_decay_start_epoch": 0,
            "rejection_known_ce_decay_epochs": 0,
            "rejection_known_ce_final_scale": 1.0,
            "single_changed_factor": "rejection_known_ce_decay_schedule",
        }
    )
    write_json(config_path, base_config)

    reports: dict[str, Any] = {}
    for arm, schedule in arms.items():
        train_dir = train_dirs[arm]
        discover_dir = discover_dirs[arm]
        train_args = load_args(config_path, "train_student", train_dir, seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.rejection_known_ce_decay_start_epoch = schedule["start"]
        train_args.rejection_known_ce_decay_epochs = schedule["duration"]
        train_args.rejection_known_ce_final_scale = schedule["scale"]
        print(f"[student] arm={arm}, schedule={schedule}", flush=True)
        fit_student(train_args)

        history = read_json(train_dir / "train_history.json")
        ce_values = [
            float(epoch["train"].get("rejection_ce", 0.0))
            for epoch in history
        ]

        discover_args = load_args(discover_source, "discover", discover_dir, seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.rejector_training = "nnpu"
        discover_args.rejector_nnpu_risk = "nu_corrected"
        discover_args.rejector_known_prior = 0.2
        discover_args.rejector_pool_selection = "all"
        discover_args.cluster_feature = "feature_pca"
        discover_args.cluster_selection = "silhouette"
        discover_args.cluster_k = "auto"
        discover_args.cluster_candidate_mode = "detector"
        discover_args.candidate_purify = "none"
        discover_args.candidate_keep_ratio = 1.0
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.skip_clustering = False
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)

        reports[arm] = {
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "schedule": schedule,
            "rejection_known_ce_loss_by_epoch": ce_values,
            "train_config": read_json(train_dir / "train_student_config.json"),
            "metrics": read_json(discover_dir / "discovery_report.json"),
            "calibration": read_json(discover_dir / "calibration_report.json"),
        }

    result = {
        "protocol": options.tag,
        "question": "Does a fixed late decay of rejection known-class CE improve semantic-hard discovery under the full metric protocol?",
        "seed": seed,
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "rejection_known_ce_decay_schedule",
        "control": "static rejection known-class CE",
        "treatment": "fixed final-two-epoch linear decay to 0.25",
        "decay_schedule": arms["late_decay"],
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "mixed_pool_labels_used_for_training_or_calibration": False,
        "cluster_selection_used_test_or_mixed_labels": False,
        "evidence_level": "single fresh semantic-hard full-metric paired replication",
        "arms": reports,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2))


if __name__ == "__main__":
    main()
