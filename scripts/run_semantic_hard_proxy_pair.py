"""Compare a proxy representation loss on a semantic-hard fresh split.

The pair changes only ``alpha_proxy`` in an otherwise fixed hard-nnPU
student configuration. The proxy loss is evaluated as an upstream
representation intervention; candidate selection, rejection PU risk,
discovery pool, detector, and known-validation calibration remain fixed.
Pool and test labels are not used for training or calibration.
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


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


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
    parser.add_argument("--seed", type=int, default=2069)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--alpha-proxy", type=float, default=0.05)
    parser.add_argument("--tag", default="semantic_hard_proxy_s2069")
    options = parser.parse_args()

    split_path = resolve(options.split_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed:
        raise ValueError("split metadata seed does not match requested seed")
    if split.get("protocol") != "semantic_hard":
        raise ValueError("this experiment requires a semantic_hard split")
    if options.alpha_proxy <= 0.0:
        raise ValueError("--alpha-proxy must be positive")

    source_teacher = resolve(options.source_teacher_config)
    source_student = resolve(options.source_student_config)
    source_discover = resolve(options.source_discover_config)
    for path in (source_teacher, source_student, source_discover):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = json.loads(source_student.read_text(encoding="utf-8"))
    if base.get("discovery_pool_mode") != "mixed":
        raise ValueError("source student config must use a mixed discovery pool")
    if base.get("discovery_rejection_pu_risk") != "hard_nnpu":
        raise ValueError("source student config must use hard nnPU")
    if base.get("alpha_vos", 0.0) or base.get("alpha_vos_rejection", 0.0):
        raise ValueError("source student config must have VOS disabled")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    if teacher_dir.exists() or config_path.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite existing experiment artifacts")

    teacher_args = load_args(source_teacher, "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    base.update({
        "split_path": str(split_path),
        "seed": int(options.seed),
        "alpha_proxy": 0.0,
    })
    config_path.write_text(
        json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    reports = {}
    for arm, weight in (("baseline", 0.0), ("proxy005", options.alpha_proxy)):
        train_dir = ROOT / "runs" / f"{options.tag}_{arm}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{arm}_discover"
        if train_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for arm={arm}")

        train_args = load_args(config_path, "train_student", train_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.alpha_proxy = float(weight)
        print(f"[train] arm={arm} alpha_proxy={weight}", flush=True)
        fit_student(train_args)

        discover_args = load_args(source_discover, "discover", discover_dir, options.seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.skip_clustering = True
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)
        reports[arm] = {
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "alpha_proxy": float(weight),
            "train_config": json.loads(
                (train_dir / "train_student_config.json").read_text(encoding="utf-8")
            ),
            "discover_config": json.loads(
                (discover_dir / "discover_config.json").read_text(encoding="utf-8")
            ),
            "metrics": json.loads(
                (discover_dir / "discovery_report.json").read_text(encoding="utf-8")
            ),
        }

    result = {
        "question": "Does a proxy representation loss improve semantic-hard rejection while preserving known utility?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "alpha_proxy",
        "baseline_alpha_proxy": 0.0,
        "treatment_alpha_proxy": float(options.alpha_proxy),
        "discovery_pool": "matched mixed pool; pool labels excluded from training decisions",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single fresh semantic-hard paired representation experiment",
        "arms": reports,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
