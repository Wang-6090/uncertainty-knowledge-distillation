"""Compare student- vs teacher-selected mixed-pool boundary candidates.

Only ``discovery_selection_model`` changes. Both arms share a fresh
semantic-hard split, split-matched teacher, fixed first-view shared-boundary
candidate event and budget, selection-conditional PU correction, detector, and
known-validation-only calibration.
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
    parser.add_argument("--seed", type=int, default=2070)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("splits_cifar100_semantic_hard_seed2070/cifar100_60_40_semantic_hard.json"),
    )
    parser.add_argument(
        "--source-teacher-config",
        type=Path,
        default=Path("runs/rejection_candidate_budget_s2060_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--source-student-config",
        type=Path,
        default=Path("analysis/rejection_candidate_budget_wired3_s2060_student_config.json"),
    )
    parser.add_argument(
        "--source-discover-config",
        type=Path,
        default=Path("runs/rejection_candidate_budget_wired3_s2060_ratio_025_discover/discover_config.json"),
    )
    parser.add_argument("--tag", default="boundary_selector_source_s2070")
    options = parser.parse_args()

    split_path = resolve(options.split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed:
        raise ValueError("split metadata seed does not match requested seed")
    if split.get("protocol") != "semantic_hard":
        raise ValueError("this paired experiment requires a semantic_hard split")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    if teacher_dir.exists():
        raise FileExistsError(f"refusing to overwrite {teacher_dir}")
    teacher_args = load_args(
        resolve(options.source_teacher_config), "train_teacher", teacher_dir, options.seed
    )
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    source_student = resolve(options.source_student_config)
    source_discover = resolve(options.source_discover_config)
    base = json.loads(source_student.read_text(encoding="utf-8"))
    fixed = {
        "split_path": str(split_path),
        "seed": int(options.seed),
        "alpha_discovery_rejection_boundary": 0.1,
        "discovery_select_ratio": 0.25,
        "discovery_feature_candidate_gating": True,
        "discovery_feature_candidate_space": "shared_boundary",
        "discovery_cross_view_gating": False,
        "discovery_single_view_candidate_gating": True,
        "discovery_pool_mode": "mixed",
        "discovery_boundary_mode": "pu",
        "discovery_boundary_pu_risk": "soft_nnpu",
        "discovery_knn_selection_conditional": True,
        "alpha_discovery_knn_boundary": 0.0,
        "alpha_discovery_rejection_knn_boundary": 0.0,
        "alpha_discovery_boundary": 0.0,
    }
    base.update(fixed)
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    if config_path.exists():
        raise FileExistsError(f"refusing to overwrite {config_path}")
    config_path.write_text(
        json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    reports = {}
    for arm, selector in (("student_gate", "student"), ("teacher_gate", "teacher")):
        train_dir = ROOT / "runs" / f"{options.tag}_{arm}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{arm}_discover"
        if train_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for {arm}")
        train_args = load_args(config_path, "train_student", train_dir, options.seed)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.discovery_selection_model = selector
        print(f"[train] arm={arm} discovery_selection_model={selector}", flush=True)
        fit_student(train_args)

        history = json.loads((train_dir / "train_history.json").read_text(encoding="utf-8"))
        candidate_rate = [float(e["train"].get("feature_candidate_ratio", 0.0)) for e in history]
        if not candidate_rate or any(rate <= 0.0 for rate in candidate_rate):
            raise RuntimeError(f"candidate gate inactive for {arm}: {candidate_rate}")
        effective_prior = [
            float(e["train"].get("discovery_boundary_effective_prior", 0.0))
            for e in history
        ]
        if not effective_prior or any(not 0.0 < value < 1.0 for value in effective_prior):
            raise RuntimeError(f"invalid selection-conditional prior for {arm}: {effective_prior}")

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
            "candidate_training_ratio_by_epoch": candidate_rate,
            "boundary_effective_known_prior_by_epoch": effective_prior,
            "discover_config": json.loads((discover_dir / "discover_config.json").read_text(encoding="utf-8")),
            "metrics": json.loads((discover_dir / "discovery_report.json").read_text(encoding="utf-8")),
        }

    result = {
        "question": "Does a split-matched teacher provide a better label-free selector for mixed-pool semantic-boundary PU training than the student itself?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "discovery_selection_model",
        "candidate_budget": 0.25,
        "candidate_event": "same first-view shared-boundary gate in both arms",
        "boundary_alpha": 0.1,
        "pool": "matched mixed pool; pool labels excluded from selection/training decisions",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single fresh semantic-hard paired mechanism experiment",
        "arms": reports,
    }
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2))


if __name__ == "__main__":
    main()
