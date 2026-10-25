"""Compare hard vs soft nnPU on the rejection feature-margin loss.

This fresh semantic-hard pair changes only the non-negative PU-risk
formulation. The margin coefficient and all model/data/evaluation settings are
held fixed. Pool/test labels are never used for training or calibration.
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
    parser.add_argument("--seed", type=int, default=2066)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tag", default="rejection_pu_risk_semantic_hard_s2066")
    options = parser.parse_args()

    split_path = resolve(options.split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed:
        raise ValueError("split metadata seed does not match requested seed")
    if split.get("protocol") != "semantic_hard":
        raise ValueError("this experiment requires a semantic_hard split")
    if options.alpha <= 0.0:
        raise ValueError("--alpha must be positive")

    source_teacher = resolve(options.source_teacher_config)
    source_student = resolve(options.source_student_config)
    source_discover = resolve(options.source_discover_config)
    for path in (source_teacher, source_student, source_discover):
        if not path.is_file():
            raise FileNotFoundError(path)

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    if teacher_dir.exists():
        raise FileExistsError(f"refusing to overwrite {teacher_dir}")
    teacher_args = load_args(source_teacher, "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    base = json.loads(source_student.read_text(encoding="utf-8"))
    if base.get("discovery_pool_mode") != "mixed":
        raise ValueError("source student config must use a mixed discovery pool")
    if base.get("alpha_vos", 0.0) or base.get("alpha_vos_rejection", 0.0):
        raise ValueError("source student config must have VOS disabled")
    base.update(
        {
            "split_path": str(split_path),
            "seed": int(options.seed),
            "alpha_discovery_rejection_pu_feature_margin": float(options.alpha),
            "discovery_rejection_pu_softness": 0.05,
            "alpha_discovery_rejection_boundary": 0.0,
            "alpha_discovery_rejection_knn_boundary": 0.0,
            "alpha_discovery_knn_boundary": 0.0,
            "alpha_rejection_decorrelation": 0.0,
        }
    )
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    if config_path.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite existing experiment artifacts")
    config_path.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reports = {}
    for arm, risk_mode in (("hard", "hard_nnpu"), ("soft", "soft_nnpu")):
        train_dir = ROOT / "runs" / f"{options.tag}_{arm}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{arm}_discover"
        if train_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for arm={arm}")

        train_args = load_args(config_path, "train_student", train_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.discovery_rejection_pu_risk = risk_mode
        print(f"[train] arm={arm} rejection_pu_risk={risk_mode}", flush=True)
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
            "train_config": json.loads((train_dir / "train_student_config.json").read_text(encoding="utf-8")),
            "discover_config": json.loads((discover_dir / "discover_config.json").read_text(encoding="utf-8")),
            "metrics": json.loads((discover_dir / "discovery_report.json").read_text(encoding="utf-8")),
        }

    result = {
        "question": "Does soft non-negative PU risk prevent rejection feature-margin collapse and improve semantic-hard open-set separation?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "discovery_rejection_pu_risk",
        "alpha_discovery_rejection_pu_feature_margin": float(options.alpha),
        "softness": 0.05,
        "discovery_pool": "matched mixed pool; pool labels excluded from training decisions",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single fresh semantic-hard paired transfer experiment",
        "arms": reports,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
