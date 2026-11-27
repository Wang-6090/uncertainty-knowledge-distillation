"""Run a 2x2 semantic-hard factorial: rejection init x decorrelation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def load_args(path: Path, command: str, output: Path, seed: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.work_dir = str(output)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--tag", default="rejection_init_decor_factorial")
    parser.add_argument("--split", type=Path)
    parser.add_argument(
        "--teacher-config",
        type=Path,
        default=Path("runs/semantic_hard_rejector_seed2028_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--student-config",
        type=Path,
        default=Path("runs/semantic_hard_vos_boundary_s2040_e4_control/train_student_config.json"),
    )
    parser.add_argument(
        "--discover-config",
        type=Path,
        default=Path("runs/semantic_hard_vos_boundary_s2040_e4_control_discover/discover_config.json"),
    )
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else (ROOT / p).resolve()
    split_path = resolve(args.split or Path(
        f"splits_cifar100_semantic_hard_seed{args.seed}/cifar100_60_40_semantic_hard.json"
    ))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != args.seed:
        raise ValueError("split must be a semantic-hard split matching --seed")

    student_config = resolve(args.student_config)
    discover_config = resolve(args.discover_config)
    source = json.loads(student_config.read_text(encoding="utf-8"))
    if source.get("alpha_vos", 0.0) or source.get("alpha_vos_rejection", 0.0):
        raise ValueError("factorial requires both VOS losses disabled")

    teacher_dir = ROOT / "runs" / f"{args.tag}_s{args.seed}_teacher"
    if teacher_dir.exists():
        raise FileExistsError(f"refusing to overwrite {teacher_dir}")
    teacher_ckpt = teacher_dir / "teacher.pt"
    teacher_args = load_args(resolve(args.teacher_config), "train_teacher", teacher_dir, args.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    arms = []
    for init in ("shared_backbone", "random_backbone"):
        for decor_alpha in (0.0, 0.01):
            label = f"{init}_decor{decor_alpha:g}"
            run_dir = ROOT / "runs" / f"{args.tag}_s{args.seed}_{label}"
            discover_dir = ROOT / "runs" / f"{args.tag}_s{args.seed}_{label}_discover"
            if run_dir.exists() or discover_dir.exists():
                raise FileExistsError(f"refusing to overwrite factorial arm {label}")
            train_args = load_args(student_config, "train_student", run_dir, args.seed)
            train_args.split_path = str(split_path)
            train_args.teacher_ckpt = str(teacher_ckpt)
            train_args.independent_rejection_encoder = True
            train_args.independent_rejection_encoder_init = init
            train_args.alpha_rejection_decorrelation = decor_alpha
            train_args.student_ckpt = str(run_dir / "student.pt")
            fit_student(train_args)

            eval_args = load_args(discover_config, "discover", discover_dir, args.seed)
            eval_args.split_path = str(split_path)
            eval_args.independent_rejection_encoder = True
            eval_args.independent_rejection_encoder_init = init
            eval_args.student_ckpt = train_args.student_ckpt
            discover(eval_args)
            report = json.loads((discover_dir / "discovery_report.json").read_text(encoding="utf-8"))
            arms.append({"arm": label, "init": init, "decorrelation_alpha": decor_alpha,
                         "run_dir": str(run_dir), "discover_dir": str(discover_dir),
                         "report": report})

    output = ROOT / "analysis" / f"{args.tag}_s{args.seed}.json"
    output.write_text(json.dumps({"seed": args.seed, "split": str(split_path), "arms": arms},
                                 indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
