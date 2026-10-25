"""Compare known-accuracy and validation-open KS checkpoint selection.

The open-validation subset is carved from the training partition and is never
the final test set.  Both arms use the same reserved subset, student seed and
training configuration; only the checkpoint selection criterion changes:
historical known-only accuracy versus threshold-free KS on the reserved
known/novel open-validation subset.
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
    parser.add_argument("--seed", type=int, default=2058)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--open-val-ratio", type=float, default=0.1)
    parser.add_argument("--selection-score", choices=["max_softmax", "entropy_proto", "energy", "head_uncertainty"], default="energy")
    parser.add_argument("--tag", default="open_checkpoint_selection_s2058")
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
    if not 0.0 < options.open_val_ratio < 1.0:
        raise ValueError("open-validation ratio must be in (0, 1)")

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

    base_config["open_val_ratio"] = float(options.open_val_ratio)
    base_config["checkpoint_selection_score"] = options.selection_score
    config_path.write_text(
        json.dumps(base_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    arms = {
        "known_acc": {"criterion": "known_acc"},
        "open_ks": {"criterion": "open_ks"},
    }
    reports = {}
    for arm, selection in arms.items():
        run_dir = ROOT / "runs" / f"{options.tag}_{arm}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{arm}_discover"
        if run_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for {arm}")

        train_args = load_args(config_path, "train_student", run_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(run_dir / "student.pt")
        train_args.checkpoint_selection = selection["criterion"]
        print(
            f"[train] arm={arm} criterion={train_args.checkpoint_selection} "
            f"score={train_args.checkpoint_selection_score} "
            f"open_val_ratio={train_args.open_val_ratio}",
            flush=True,
        )
        fit_student(train_args)

        discover_args = load_args(source_discover, "discover", discover_dir, options.seed)
        discover_args.split_path = str(split_path)
        discover_args.open_val_ratio = float(options.open_val_ratio)
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
            "selection": selection,
            "train_config": json.loads(
                (run_dir / "train_student_config.json").read_text(encoding="utf-8")
            ),
            "metrics": json.loads(
                (discover_dir / "discovery_report.json").read_text(encoding="utf-8")
            ),
        }

    result = {
        "question": "Does validation-only open KS checkpoint selection improve the fixed-coverage rejection tradeoff?",
        "seed": int(options.seed),
        "split": str(split_path),
        "single_changed_factor": "checkpoint_selection_criterion",
        "open_validation_ratio": float(options.open_val_ratio),
        "checkpoint_selection_score": options.selection_score,
        "open_validation_is_from_training_partition": True,
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
