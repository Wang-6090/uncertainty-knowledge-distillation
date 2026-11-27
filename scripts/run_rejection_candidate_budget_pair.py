"""Compare discovery candidate budgets with a fixed PU rejection boundary loss.

The only algorithmic factor is ``discovery_select_ratio``. Candidate gating is
applied to the PU rejection-boundary loss, with a selection-conditional known
prior so the selected mixed pool retains a valid PU correction.
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
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="reuse an existing split-matched teacher checkpoint",
    )
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--boundary-alpha", type=float, default=0.1)
    parser.add_argument("--low-ratio", type=float, default=0.25)
    parser.add_argument("--high-ratio", type=float, default=0.50)
    parser.add_argument("--tag", default="rejection_candidate_budget_s2054")
    options = parser.parse_args()

    resolve = lambda path: path if path.is_absolute() else (ROOT / path).resolve()
    split_path = resolve(options.split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed:
        raise ValueError("split metadata seed does not match requested seed")
    if split.get("protocol") != "semantic_hard":
        raise ValueError("this pair requires a semantic_hard split")
    if not 0.0 < options.low_ratio < options.high_ratio <= 1.0:
        raise ValueError("candidate ratios must satisfy 0 < low < high <= 1")
    if options.boundary_alpha <= 0.0:
        raise ValueError("--boundary-alpha must be positive")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    if options.teacher_checkpoint is not None:
        teacher_ckpt = resolve(options.teacher_checkpoint)
        if not teacher_ckpt.is_file():
            raise FileNotFoundError(teacher_ckpt)
    else:
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
    base_config = json.loads(source_student.read_text(encoding="utf-8"))
    if not base_config.get("discovery_feature_candidate_gating", False):
        raise ValueError("candidate-gating must be enabled for this pair")
    if base_config.get("discovery_feature_candidate_space") != "shared_boundary":
        raise ValueError("this pair requires shared_boundary candidate space")
    if base_config.get("discovery_pool_mode") != "mixed":
        raise ValueError("this pair requires discovery_pool_mode=mixed")
    base_config["independent_rejection_encoder"] = True
    base_config["independent_rejection_encoder_init"] = "shared_backbone"
    base_config["alpha_discovery_rejection_boundary"] = float(options.boundary_alpha)
    base_config["alpha_discovery_rejection_knn_boundary"] = 0.0
    base_config["alpha_discovery_rejection_pu_feature_margin"] = 0.0
    base_config["discovery_knn_selection_conditional"] = True
    base_config["discovery_cross_view_gating"] = False
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    if config_path.exists():
        raise FileExistsError(f"refusing to overwrite {config_path}")
    config_path.write_text(
        json.dumps(base_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    arms = {"ratio_025": float(options.low_ratio), "ratio_050": float(options.high_ratio)}
    reports = {}
    for name, ratio in arms.items():
        run_dir = ROOT / "runs" / f"{options.tag}_{name}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{name}_discover"
        if run_dir.exists() or discover_dir.exists():
            raise FileExistsError(f"refusing to overwrite outputs for {name}")
        train_args = load_args(config_path, "train_student", run_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(run_dir / "student.pt")
        train_args.alpha_discovery_rejection_boundary = float(options.boundary_alpha)
        train_args.discovery_select_ratio = ratio
        print(f"[train] arm={name} candidate_ratio={ratio:g}", flush=True)
        fit_student(train_args)
        history_path = run_dir / "train_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        candidate_rates = [
            float(epoch["train"].get("feature_candidate_ratio", 0.0))
            for epoch in history
        ]
        if not candidate_rates or any(rate <= 0.0 for rate in candidate_rates):
            raise RuntimeError(
                f"candidate gating was inactive in at least one epoch for {name}: "
                f"{candidate_rates}"
            )
        boundary_rates = [
            float(epoch["train"].get("discovery_boundary_effective_prior", 0.0))
            for epoch in history
        ]
        if not boundary_rates or any(not 0.0 < prior < 1.0 for prior in boundary_rates):
            raise RuntimeError(
                f"invalid selection-conditional known prior for {name}: "
                f"{boundary_rates}"
            )

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
            "train_config": json.loads(
                (run_dir / "train_student_config.json").read_text(encoding="utf-8")
            ),
            "candidate_training_ratio_by_epoch": candidate_rates,
            "boundary_effective_known_prior_by_epoch": boundary_rates,
            "discover_config": json.loads(
                (discover_dir / "discover_config.json").read_text(encoding="utf-8")
            ),
            "metrics": json.loads(report_path.read_text(encoding="utf-8")),
        }

    result = {
        "question": "Does increasing the label-free candidate budget improve semantic-hard rejection when PU boundary training is fixed?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "discovery_select_ratio",
        "low_ratio": float(options.low_ratio),
        "high_ratio": float(options.high_ratio),
        "fixed_boundary_alpha": float(options.boundary_alpha),
        "discovery_pool": "mixed",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single-seed exploratory candidate-budget comparison",
        "arms": reports,
    }
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2))


if __name__ == "__main__":
    main()
