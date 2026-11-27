"""Paired control/treatment test of superclass-aware rejection SupCon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


TREATMENT_ALPHA = 0.1


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2047)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path(
            "splits_cifar100_semantic_hard_seed2047/"
            "cifar100_60_40_semantic_hard.json"
        ),
    )
    parser.add_argument(
        "--source-teacher-config",
        type=Path,
        default=Path("runs/semantic_hard_rejector_seed2028_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--source-student-config",
        type=Path,
        default=Path("runs/semantic_hard_vos_boundary_s2040_e4_control/train_student_config.json"),
    )
    parser.add_argument(
        "--source-discover-config",
        type=Path,
        default=Path("runs/semantic_hard_vos_boundary_s2040_e4_control_discover/discover_config.json"),
    )
    parser.add_argument("--tag", default="superclass_rejection_supcon")
    options = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (ROOT / path).resolve()

    split_path = resolve(options.split_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed:
        raise ValueError("split metadata seed does not match requested seed")
    if split.get("protocol") != "semantic_hard":
        raise ValueError("paired experiment requires a semantic_hard split")
    if "fine_to_coarse" not in split:
        raise ValueError("semantic-hard split must include fine_to_coarse taxonomy")

    teacher_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    result_dirs = [
        ROOT / "runs" / f"{options.tag}_s{options.seed}_{arm}"
        for arm in ("control", "treatment")
    ]
    discover_dirs = [Path(f"{run_dir}_discover") for run_dir in result_dirs]
    for path in [teacher_dir, *result_dirs, *discover_dirs]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing experiment output: {path}")

    teacher_args = load_args(
        resolve(options.source_teacher_config),
        "train_teacher",
        teacher_dir,
        options.seed,
    )
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    source_student = json.loads(
        resolve(options.source_student_config).read_text(encoding="utf-8")
    )
    if source_student.get("alpha_vos", 0.0) or source_student.get("alpha_vos_rejection", 0.0):
        raise ValueError("source student config must have VOS disabled")

    discover_source = resolve(options.source_discover_config)
    for arm, alpha in (("control", 0.0), ("treatment", TREATMENT_ALPHA)):
        run_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_{arm}"
        discover_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_{arm}_discover"
        student_args = load_args(
            resolve(options.source_student_config),
            "train_student",
            run_dir,
            options.seed,
        )
        student_args.split_path = str(split_path)
        student_args.teacher_ckpt = str(teacher_ckpt)
        student_args.independent_rejection_encoder = True
        student_args.independent_rejection_encoder_init = "shared_backbone"
        student_args.alpha_rejection_superclass_supcon = alpha
        student_args.student_ckpt = str(run_dir / "student.pt")
        fit_student(student_args)

        discover_args = load_args(
            discover_source,
            "discover",
            discover_dir,
            options.seed,
        )
        discover_args.split_path = str(split_path)
        discover_args.independent_rejection_encoder = True
        discover_args.independent_rejection_encoder_init = "shared_backbone"
        discover_args.student_ckpt = student_args.student_ckpt
        discover(discover_args)

    summary = {
        "seed": options.seed,
        "split_path": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "control": {"alpha_rejection_superclass_supcon": 0.0},
        "treatment": {
            "alpha_rejection_superclass_supcon": TREATMENT_ALPHA,
            "same_superclass_negative_weight": 2.0,
        },
        "all_other_student_and_discovery_config": "matched to source configs; independent rejection init fixed to shared_backbone",
        "training_runs": [str(path) for path in result_dirs],
        "discovery_runs": [str(path) for path in discover_dirs],
    }
    report_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}_pair.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
