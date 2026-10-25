"""Run a strict teacher-vs-student hard-unknown selector pair.

The pair tests one label-free data/supervision factor on a fresh
``semantic_hard`` split: the model used to score shared-feature support
boundary candidates in the mixed discovery pool.  Both arms use the same
selection event, budget, selection-conditional PU correction, independent
shared-backbone rejection branch, and downstream known-validation-only
calibration.  Only ``discovery_selection_model`` changes.

Pool/test labels are never read by the optimization, checkpoint, or threshold
code.  The training history may contain retrospective contamination and
per-novel-class coverage diagnostics; those fields are written after the
selection mask has already been formed and are not used for any decision.
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
    values = read_json(path)
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def ensure_split(seed: int, split_path: Path, data_root: Path) -> Path:
    if split_path.exists():
        split = read_json(split_path)
        if split.get("metadata", {}).get("seed") != seed:
            raise ValueError(f"existing split seed does not match: {split_path}")
        if split.get("protocol") != "semantic_hard":
            raise ValueError(f"existing split is not semantic_hard: {split_path}")
        return split_path
    fine_names, coarse_names, mapping = read_cifar100_taxonomy(data_root)
    split = build_protocols(fine_names, coarse_names, mapping, seed)["semantic_hard"]
    write_json(split_path, split)
    return split_path


def apply_strict_baseline(args, split_path: Path, seed: int, device: str):
    """Materialize the current mainline without uncertainty-KD or extra PU."""
    args.split_path = str(split_path)
    args.seed = int(seed)
    args.model_seed = None
    args.device = device
    args.independent_rejection_encoder = True
    args.independent_rejection_encoder_init = "shared_backbone"
    args.independent_rejection_input_norm = "none"
    args.rejection_feature_dim = 128
    args.epochs = 4
    args.discovery_pool_mode = "mixed"
    args.matched_discovery_pool = True
    args.matched_pool_size = 5400
    args.matched_known_prior = 0.2
    args.checkpoint_selection = "known_acc"

    # Keep the student-teacher baseline explicit and auditable.
    args.kd_mode = "standard"
    args.alpha_discovery_uncertainty_pu = 0.0
    args.alpha_discovery_uncertainty_feature_margin = 0.0
    args.alpha_discovery_uncertainty_separation = 0.0

    # The sole training mechanism: a fixed-budget, label-free shared-support
    # boundary candidate event feeding selection-conditional PU boundary loss.
    args.discovery_feature_candidate_gating = True
    args.discovery_feature_candidate_space = "shared_boundary"
    args.discovery_select_ratio = 0.25
    args.discovery_select_mode = "consensus"
    args.discovery_single_view_candidate_gating = True
    args.discovery_cross_view_gating = False
    args.discovery_knn_selection_conditional = True
    args.discovery_knn_k = 5
    args.discovery_knn_quantile = 0.95
    args.alpha_discovery_rejection_boundary = 0.1
    args.discovery_boundary_mode = "pu"
    args.discovery_boundary_pu_risk = "soft_nnpu"
    args.discovery_boundary_pu_softness = 0.05

    # Do not mix in other hard-unknown or post-hoc mechanisms.
    for name in (
        "alpha_vos",
        "alpha_vos_rejection",
        "alpha_discovery_rejection_knn_boundary",
        "alpha_discovery_knn_boundary",
        "alpha_discovery_rejection_pu_feature_margin",
        "alpha_discovery_feature_margin",
        "alpha_discovery_rejection_feature_margin",
        "alpha_discovery_rejection_feature_separation",
        "alpha_rejection_decorrelation",
        "alpha_rejection_superclass_supcon",
        "alpha_outlier_rejection_feature_margin",
        "alpha_outlier_rejection_uniform",
    ):
        if hasattr(args, name):
            setattr(args, name, 0.0)
    return args


def materialized_differences(control, treatment, ignored: set[str]) -> dict[str, Any]:
    return {
        key: [getattr(control, key), getattr(treatment, key)]
        for key in vars(control)
        if key not in ignored and getattr(control, key) != getattr(treatment, key)
    }


def audit_training_history(path: Path) -> dict[str, Any]:
    history = read_json(path)
    if not history:
        raise RuntimeError(f"empty training history: {path}")
    train_rows = [entry["train"] for entry in history]
    required = (
        "feature_candidate_ratio",
        "feature_candidate_known_fraction",
        "feature_candidate_unknown_recall",
        "discovery_rejection_boundary",
        "discovery_boundary_effective_prior",
        "gradient_audit_grad_norm_rejection_discovery_boundary_term",
    )
    missing = [key for key in required if key not in train_rows[-1]]
    if missing:
        raise RuntimeError(
            f"training history lacks hard-unknown audit fields {missing}; "
            "the candidate path was not auditable"
        )
    if any(float(row["feature_candidate_ratio"]) <= 0.0 for row in train_rows):
        raise RuntimeError("candidate gate was inactive in at least one epoch")
    # This experiment applies the boundary risk to the independent rejection
    # branch.  The classifier-branch field ``discovery_boundary`` is expected
    # to remain zero; checking it here would falsely reject a valid run.
    if any(float(row["discovery_rejection_boundary"]) <= 0.0 for row in train_rows):
        raise RuntimeError("rejection-branch PU boundary loss was inactive in at least one epoch")
    if any(float(row["gradient_audit_grad_norm_rejection_discovery_boundary_term"]) <= 0.0 for row in train_rows):
        raise RuntimeError("rejection-branch PU boundary gradient audit was zero in at least one epoch")
    if any(abs(float(row["discovery_boundary_effective_prior"]) - 0.2) > 1e-5 for row in train_rows):
        raise RuntimeError("selection-conditional effective prior drifted from 0.2")
    if any(bool(row.get("kd_uncertainty_weight_applied", False)) for row in train_rows):
        raise RuntimeError("uncertainty-KD unexpectedly remained active")
    if any(float(row.get("discovery_uncertainty_pu", 0.0)) != 0.0 for row in train_rows):
        raise RuntimeError("uncertainty-PU unexpectedly remained active")
    return {
        "epochs": len(train_rows),
        "candidate_ratio_by_epoch": [float(row["feature_candidate_ratio"]) for row in train_rows],
        "candidate_known_contamination_by_epoch": [
            float(row["feature_candidate_known_fraction"]) for row in train_rows
        ],
        "candidate_unknown_recall_by_epoch": [
            float(row["feature_candidate_unknown_recall"]) for row in train_rows
        ],
        "candidate_known_selection_rate_by_epoch": [
            float(row["feature_candidate_known_selection_rate"]) for row in train_rows
        ],
        "boundary_loss_by_epoch": [float(row["discovery_rejection_boundary"]) for row in train_rows],
        "boundary_grad_norm_by_epoch": [
            float(row["gradient_audit_grad_norm_rejection_discovery_boundary_term"]) for row in train_rows
        ],
        "effective_prior_by_epoch": [
            float(row["discovery_boundary_effective_prior"]) for row in train_rows
        ],
        "novel_recall_summary_by_epoch": [
            row.get("feature_candidate_novel_recall_summary", {}) for row in train_rows
        ],
        "novel_recall_by_raw_label_by_epoch": [
            row.get("feature_candidate_novel_recall_by_raw_label", {}) for row in train_rows
        ],
        "labels_used_only_for_retrospective_diagnostics": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2088)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("splits_cifar100_semantic_hard_seed2088/cifar100_60_40_semantic_hard.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--source-teacher-config",
        type=Path,
        default=Path("runs/shared_rejection_input_norm_s2087_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--source-student-config",
        type=Path,
        default=Path("runs/shared_rejection_input_norm_s2087_control_none/train_student_config.json"),
    )
    parser.add_argument(
        "--source-discover-config",
        type=Path,
        default=Path("runs/shared_rejection_input_norm_s2087_control_none_discover/discover_config.json"),
    )
    parser.add_argument("--tag", default="semantic_hard_boundary_selector_s2088")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help=(
            "reuse only complete teacher/student checkpoints from an interrupted run; "
            "missing arms and discovery stages are still executed"
        ),
    )
    options = parser.parse_args()

    split_path = ensure_split(options.seed, resolve(options.split_path), resolve(options.data_root))
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("semantic-hard split must contain 60 known and 40 novel classes")

    teacher_source = resolve(options.source_teacher_config)
    student_source = resolve(options.source_student_config)
    discover_source = resolve(options.source_discover_config)
    for path in (teacher_source, student_source, discover_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    arm_names = ("student_selector", "teacher_selector")
    arm_dirs = {
        name: ROOT / "runs" / f"{options.tag}_{name}" for name in arm_names
    }
    discover_dirs = {
        name: ROOT / "runs" / f"{options.tag}_{name}_discover" for name in arm_names
    }
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    for path in (teacher_dir, config_path, result_path, *arm_dirs.values(), *discover_dirs.values()):
        if path.exists() and not options.resume_partial:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.device = options.device
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    teacher_complete = all(
        path.is_file()
        for path in (
            teacher_dir / "teacher.pt",
            teacher_dir / "train_history.json",
            teacher_dir / "train_teacher_config.json",
        )
    )
    if options.resume_partial and teacher_complete:
        print(f"[resume] teacher checkpoint={teacher_ckpt}", flush=True)
    else:
        fit_teacher(teacher_args)

    source_student = read_json(student_source)
    source_student.update(
        {
            "split_path": str(split_path),
            "seed": int(options.seed),
            "device": options.device,
        }
    )
    if not (options.resume_partial and config_path.is_file()):
        config_path.write_text(
            json.dumps(source_student, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    reports = {}
    control_args = apply_strict_baseline(
        load_args(config_path, "train_student", arm_dirs["student_selector"], options.seed),
        split_path,
        options.seed,
        options.device,
    )
    control_args.discovery_selection_model = "student"
    # Both paired arms must point to the exact same frozen teacher checkpoint.
    # The source config may contain an older path; leaving it here would create
    # a spurious materialized difference even though the intended teacher is
    # shared.
    control_args.teacher_ckpt = str(teacher_ckpt)
    for arm, selector in (("student_selector", "student"), ("teacher_selector", "teacher")):
        train_dir = arm_dirs[arm]
        discover_dir = discover_dirs[arm]
        train_args = apply_strict_baseline(
            load_args(config_path, "train_student", train_dir, options.seed),
            split_path,
            options.seed,
            options.device,
        )
        train_args.discovery_selection_model = selector
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        if arm == "teacher_selector":
            differences = materialized_differences(
                control_args,
                train_args,
                {"work_dir", "student_ckpt", "discovery_selection_model"},
            )
            if differences:
                raise ValueError(f"unexpected paired training differences: {differences}")
        train_complete = all(
            path.is_file()
            for path in (
                train_dir / "student.pt",
                train_dir / "train_history.json",
                train_dir / "train_student_config.json",
            )
        )
        if options.resume_partial and train_complete:
            print(f"[resume] arm={arm} checkpoint={train_dir / 'student.pt'}", flush=True)
        else:
            print(f"[train] arm={arm} selector={selector}", flush=True)
            fit_student(train_args)
        history_audit = audit_training_history(train_dir / "train_history.json")

        discover_args = load_args(discover_source, "discover", discover_dir, options.seed)
        discover_args.split_path = str(split_path)
        discover_args.device = options.device
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.independent_rejection_encoder = True
        discover_args.independent_rejection_encoder_init = "shared_backbone"
        discover_args.independent_rejection_input_norm = "none"
        discover_args.score_mode = "feature_rejector"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.skip_clustering = False
        discover_args.cluster_k = "auto"
        discovery_complete = all(
            path.is_file()
            for path in (
                discover_dir / "discovery_report.json",
                discover_dir / "discover_config.json",
            )
        )
        if options.resume_partial and discovery_complete:
            print(f"[resume] discovery report={discover_dir / 'discovery_report.json'}", flush=True)
        else:
            print(f"[discover] arm={arm}", flush=True)
            discover(discover_args)
        reports[arm] = {
            "selector": selector,
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "train_config": read_json(train_dir / "train_student_config.json"),
            "training_audit": history_audit,
            "discover_config": read_json(discover_dir / "discover_config.json"),
            "metrics": read_json(discover_dir / "discovery_report.json"),
        }

    result = {
        "question": "Does a split-matched teacher provide a better label-free shared-support boundary selector than the student for semantic-hard PU training?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "discovery_selection_model",
        "control": "student selector",
        "treatment": "split-matched frozen teacher selector",
        "candidate_event": "single-view shared-feature kNN support-boundary top 25%",
        "selection_conditional_pu": True,
        "discovery_pool": "matched mixed pool, size 5400, known prior 0.2",
        "calibration": "known validation only, target 95% coverage",
        "clustering": "enabled; automatic-K is reported descriptively and not used for method selection",
        "uncertainty_kd": "disabled; kd_mode=standard",
        "pool_labels_used_for_training_selection_or_calibration": False,
        "test_labels_used_for_training_selection_or_calibration": False,
        "training_history_label_boundary": "raw pool labels are written only as retrospective contamination/tail-coverage diagnostics after masks are formed",
        "evidence_level": "single fresh semantic-hard paired mechanism experiment",
        "arms": reports,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
