"""Compare a PU class-conditional rejection-boundary training loss.

The pair fixes the shared-backbone rejection encoder and changes only
``alpha_discovery_rejection_boundary``.  The loss treats the discovery pool as
a known/unknown mixture and uses a PU-corrected class-conditional support
boundary; no discovery-pool labels or test labels are used by training or
calibration.
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
    parser.add_argument("--seed", type=int, default=2052)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--alpha-boundary", type=float, default=0.1)
    parser.add_argument("--tag", default="rejection_boundary_train_s2052")
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
    if options.alpha_boundary <= 0.0:
        raise ValueError("--alpha-boundary must be positive")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    if teacher_dir.exists():
        raise FileExistsError(f"refusing to overwrite {teacher_dir}")
    teacher_args = load_args(resolve(options.source_teacher_config), "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    source_student = resolve(options.source_student_config)
    source_discover = resolve(options.source_discover_config)
    base_config = json.loads(source_student.read_text(encoding="utf-8"))
    if base_config.get("alpha_vos", 0.0) != 0.0 or base_config.get("alpha_vos_rejection", 0.0) != 0.0:
        raise ValueError("source student config must have VOS disabled")
    if base_config.get("discovery_pool_mode") != "mixed":
        raise ValueError("the pair requires discovery_pool_mode=mixed")
    if base_config.get("discovery_boundary_mode") != "pu":
        raise ValueError("the pair requires discovery_boundary_mode=pu")
    base_config["independent_rejection_encoder"] = True
    base_config["independent_rejection_encoder_init"] = "shared_backbone"
    base_config["alpha_rejection_decorrelation"] = 0.0
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    if config_path.exists():
        raise FileExistsError(f"refusing to overwrite {config_path}")
    config_path.write_text(json.dumps(base_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    arms = {"baseline": 0.0, "pu_boundary": float(options.alpha_boundary)}
    reports = {}
    for name, alpha in arms.items():
        run_dir = ROOT / "runs" / f"{options.tag}_{name}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{name}_discover"
        if run_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for {name}")

        train_args = load_args(config_path, "train_student", run_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(run_dir / "student.pt")
        train_args.alpha_discovery_rejection_boundary = alpha
        train_args.alpha_discovery_rejection_knn_boundary = 0.0
        train_args.alpha_discovery_rejection_pu_feature_margin = 0.0
        train_args.alpha_discovery_boundary = 0.0
        train_args.discovery_boundary_mode = "pu"
        print(f"[train] arm={name} alpha_boundary={alpha}", flush=True)
        fit_student(train_args)

        discover_args = load_args(source_discover, "discover", discover_dir, options.seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(run_dir / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.skip_clustering = True
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        print(f"[discover] arm={name}", flush=True)
        discover(discover_args)
        report_path = discover_dir / "discovery_report.json"
        reports[name] = {
            "train_dir": str(run_dir),
            "discover_dir": str(discover_dir),
            "train_config": json.loads((run_dir / "train_student_config.json").read_text(encoding="utf-8")),
            "discover_config": json.loads((discover_dir / "discover_config.json").read_text(encoding="utf-8")),
            "metrics": json.loads(report_path.read_text(encoding="utf-8")),
        }

    result = {
        "question": "Does PU class-conditional rejection-boundary training reduce semantic-hard unknown/known overlap?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "alpha_discovery_rejection_boundary",
        "baseline_value": 0.0,
        "treatment_value": float(options.alpha_boundary),
        "shared_backbone": True,
        "discovery_pool": "mixed",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single-seed exploratory training-objective comparison",
        "arms": reports,
    }
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
