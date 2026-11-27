"""Test late decay of rejection known-class CE on a fresh semantic-hard split.

The control keeps the known rejection CE coefficient fixed. The treatment
keeps it unchanged for the first two epochs, then decays it over the last two
epochs, leaving the mixed-pool uncertainty-PU terms unchanged. This tests
whether preserving early known representation learning while reducing late
known-only pressure helps the open-set boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2054)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--decay-start-epoch", type=int, default=3)
    parser.add_argument("--decay-epochs", type=int, default=2)
    parser.add_argument("--final-scale", type=float, default=0.25)
    parser.add_argument("--tag", default="rejection_ce_decay_s2054")
    options = parser.parse_args()

    resolve = lambda path: path if path.is_absolute() else (ROOT / path).resolve()
    split_path = resolve(options.split_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed:
        raise ValueError("split metadata seed does not match requested seed")
    if split.get("protocol") != "semantic_hard":
        raise ValueError("this pair requires a semantic_hard split")
    if options.decay_start_epoch <= 0 or options.decay_epochs <= 0:
        raise ValueError("decay start and duration must be positive")
    if not 0.0 <= options.final_scale <= 1.0:
        raise ValueError("final scale must be in [0, 1]")

    source_teacher = resolve(options.source_teacher_config)
    source_student = resolve(options.source_student_config)
    source_discover = resolve(options.source_discover_config)
    base_config = json.loads(source_student.read_text(encoding="utf-8"))
    if base_config.get("discovery_pool_mode") != "mixed":
        raise ValueError("this pair requires a mixed discovery pool")
    if not base_config.get("independent_rejection_encoder", False):
        raise ValueError("source student must use the independent rejection encoder")
    if base_config.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student must use shared_backbone initialization")
    if base_config.get("alpha_vos", 0.0) or base_config.get("alpha_vos_rejection", 0.0):
        raise ValueError("VOS must be disabled for this single-factor comparison")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    if teacher_dir.exists() or config_path.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite an existing experiment artifact")

    teacher_args = load_args(source_teacher, "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    base_config["rejection_known_ce_decay_start_epoch"] = 0
    base_config["rejection_known_ce_decay_epochs"] = 0
    base_config["rejection_known_ce_final_scale"] = 1.0
    config_path.write_text(
        json.dumps(base_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    arms = {
        "static_control": {"start": 0, "duration": 0, "scale": 1.0},
        "late_decay": {
            "start": options.decay_start_epoch,
            "duration": options.decay_epochs,
            "scale": options.final_scale,
        },
    }
    reports = {}
    for arm, schedule in arms.items():
        run_dir = ROOT / "runs" / f"{options.tag}_{arm}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{arm}_discover"
        if run_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for {arm}")

        train_args = load_args(config_path, "train_student", run_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(run_dir / "student.pt")
        train_args.rejection_known_ce_decay_start_epoch = schedule["start"]
        train_args.rejection_known_ce_decay_epochs = schedule["duration"]
        train_args.rejection_known_ce_final_scale = schedule["scale"]
        print(f"[train] arm={arm} schedule={schedule}", flush=True)
        fit_student(train_args)

        discover_args = load_args(source_discover, "discover", discover_dir, options.seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = train_args.student_ckpt
        discover_args.independent_rejection_encoder = True
        discover_args.independent_rejection_encoder_init = "shared_backbone"
        discover_args.score_mode = "feature_rejector"
        discover_args.skip_clustering = True
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)

        reports[arm] = {
            "train_dir": str(run_dir),
            "discover_dir": str(discover_dir),
            "schedule": schedule,
            "train_config": json.loads(
                (run_dir / "train_student_config.json").read_text(encoding="utf-8")
            ),
            "metrics": json.loads(
                (discover_dir / "discovery_report.json").read_text(encoding="utf-8")
            ),
        }

    result = {
        "question": "Does late decay of known rejection CE improve semantic-hard rejection while preserving early known learning?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "rejection_known_ce_decay_schedule",
        "base_alpha_rejection_known_ce": base_config.get("alpha_rejection_known_ce"),
        "discovery_pool": "mixed, matched known prior from source config",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "arms": reports,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2))


if __name__ == "__main__":
    main()
