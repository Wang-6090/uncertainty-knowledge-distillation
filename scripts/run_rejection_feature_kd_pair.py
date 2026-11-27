"""Fresh-split paired tests for rejection-feature KD weighting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_cifar100_splits import build_protocols, read_cifar100_taxonomy  # noqa: E402
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def make_split(seed: int, path: Path, data_root: Path) -> None:
    if path.exists():
        split = read_json(path)
        if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != seed:
            raise ValueError(f"existing split does not match the protocol: {path}")
        return
    names, coarse, mapping = read_cifar100_taxonomy(data_root)
    write_json(path, build_protocols(names, coarse, mapping, seed)["semantic_hard"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2102)
    parser.add_argument("--tag", default="rejection_feature_kd_s2102")
    parser.add_argument(
        "--kd-uncertainty-source",
        choices=("head", "mc_epistemic", "mc_predictive_entropy"),
        default=None,
        help="Override the shared teacher uncertainty source for both arms.",
    )
    parser.add_argument(
        "--kd-mode",
        choices=("standard", "uncertainty"),
        default=None,
        help="Override the shared logits-KD mode for both arms.",
    )
    parser.add_argument(
        "--compare-weighting-only",
        action="store_true",
        help=(
            "Compare uniform versus teacher-uncertainty weighting at the same "
            "nonzero rejection-feature KD coefficient."
        ),
    )
    parser.add_argument(
        "--feature-kd-weighting",
        choices=("uniform", "teacher_uncertainty"),
        default="teacher_uncertainty",
        help="Weighting for the nonzero feature-KD arm when comparing against no-KD.",
    )
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    options = parser.parse_args()
    sources = [
        p if p.is_absolute() else (ROOT / p).resolve()
        for p in (
            options.source_teacher_config,
            options.source_student_config,
            options.source_discover_config,
        )
    ]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    seed = int(options.seed)
    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{seed}" / "cifar100_60_40_semantic_hard.json"
    base_student = read_json(sources[1])
    if base_student.get("independent_rejection_encoder") is not True:
        raise ValueError("source student must use the validated independent rejection branch")
    if base_student.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("source student must use shared_backbone initialization")
    if int(base_student.get("rejection_feature_dim", 0)) <= 0:
        raise ValueError("source student must enable rejection features")
    if base_student.get("checkpoint_selection") != "known_acc":
        raise ValueError("source student must select checkpoints using known validation accuracy")
    data_root = (ROOT / str(base_student.get("data_root", ".\\data"))).resolve()
    make_split(seed, split_path, data_root)
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard CIFAR-100 60/40 split")

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    arms = (
        ("uniform_weight", "uncertainty_weight")
        if options.compare_weighting_only
        else ("control", "feature_kd")
    )
    run_dirs = {arm: ROOT / "runs" / f"{options.tag}_{arm}" for arm in arms}
    discover_dirs = {arm: Path(f"{run_dirs[arm]}_discover") for arm in arms}
    artifacts = [teacher_dir, student_config_path, result_path, *run_dirs.values(), *discover_dirs.values()]
    completed_outputs = [
        teacher_ckpt,
        student_config_path,
        *[run_dirs[arm] / "train_student_config.json" for arm in arms],
        *[run_dirs[arm] / "train_history.json" for arm in arms],
        *[discover_dirs[arm] / "discovery_report.json" for arm in arms],
    ]
    resume_completed = all(path.is_file() for path in completed_outputs)
    arm_completed = {
        arm: all(
            path.is_file()
            for path in (
                run_dirs[arm] / "train_student_config.json",
                run_dirs[arm] / "train_history.json",
                discover_dirs[arm] / "discovery_report.json",
            )
        )
        for arm in arms
    }
    partial_resume = (
        teacher_ckpt.is_file()
        and student_config_path.is_file()
        and any(arm_completed.values())
        and not resume_completed
    )
    if any(path.exists() for path in artifacts) and not (resume_completed or partial_resume):
        raise FileExistsError("refusing to overwrite existing experiment artifacts")

    if not teacher_ckpt.is_file():
        teacher_args = load_args(sources[0], "train_teacher", teacher_dir, seed)
        teacher_args.split_path = str(split_path)
        teacher_args.teacher_ckpt = str(teacher_ckpt)
        teacher_args.open_val_ratio = 0.0
        teacher_args.rejection_feature_dim = int(base_student["rejection_feature_dim"])
        print("[teacher] fit one split-matched teacher shared by both arms", flush=True)
        fit_teacher(teacher_args)

    if student_config_path.is_file():
        base_student = read_json(student_config_path)
    else:
        base_student.update({
            "split_path": str(split_path),
            "seed": seed,
            "model_seed": None,
            "open_val_ratio": 0.0,
            "alpha_rejection_feat_kd": 0.0,
            "rejection_feature_kd_weighting": "uniform",
            "teacher_rejection_feature_dim": int(base_student["rejection_feature_dim"]),
        })
        if options.kd_uncertainty_source is not None:
            base_student["kd_uncertainty_source"] = options.kd_uncertainty_source
        if options.kd_mode is not None:
            base_student["kd_mode"] = options.kd_mode
        write_json(student_config_path, base_student)

    results = {}
    for arm in arms:
        if not arm_completed[arm]:
            train_args = load_args(student_config_path, "train_student", run_dirs[arm], seed)
            train_args.split_path = str(split_path)
            train_args.teacher_ckpt = str(teacher_ckpt)
            train_args.student_ckpt = str(run_dirs[arm] / "student.pt")
            train_args.alpha_rejection_feat_kd = (
                0.1 if options.compare_weighting_only or arm == "feature_kd" else 0.0
            )
            train_args.rejection_feature_kd_weighting = (
                "uniform" if arm == "uniform_weight" else
                "teacher_uncertainty" if arm == "uncertainty_weight" else
                options.feature_kd_weighting if arm == "feature_kd" else "uniform"
            )
            print(f"[student] arm={arm}, alpha_rejection_feat_kd={train_args.alpha_rejection_feat_kd}", flush=True)
            fit_student(train_args)

            discover_args = load_args(sources[2], "discover", discover_dirs[arm], seed)
            discover_args.split_path = str(split_path)
            discover_args.student_ckpt = str(run_dirs[arm] / "student.pt")
            discover_args.independent_rejection_encoder = True
            discover_args.independent_rejection_encoder_init = "shared_backbone"
            discover_args.rejection_feature_dim = int(base_student["rejection_feature_dim"])
            discover_args.score_mode = "feature_rejector"
            discover_args.threshold_policy = "known_coverage"
            discover_args.target_known_coverage = 0.95
            discover_args.open_val_ratio = 0.0
            discover_args.skip_clustering = False
            discover_args.cluster_k = "auto"
            print(f"[discover] arm={arm}", flush=True)
            discover(discover_args)
        results[arm] = {
            "train_dir": str(run_dirs[arm]),
            "discover_dir": str(discover_dirs[arm]),
            "train_config": read_json(run_dirs[arm] / "train_student_config.json"),
            "discover_config": read_json(discover_dirs[arm] / "discover_config.json"),
            "metrics": read_json(discover_dirs[arm] / "discovery_report.json"),
            "history": read_json(run_dirs[arm] / "train_history.json"),
        }

    control_arm = "uniform_weight" if options.compare_weighting_only else "control"
    treatment_arm = "uncertainty_weight" if options.compare_weighting_only else "feature_kd"
    control = results[control_arm]["train_config"]
    treatment = results[treatment_arm]["train_config"]
    allowed = {
        "alpha_rejection_feat_kd", "rejection_feature_kd_weighting", "work_dir", "student_ckpt", "best_epoch",
        "best_val_known_acc", "best_checkpoint_selection_score",
        "best_open_checkpoint_stats", "transform_profile_requested",
    }
    differences = {
        key: (control.get(key), treatment.get(key))
        for key in sorted(set(control) | set(treatment))
        if control.get(key) != treatment.get(key)
    }
    unexpected = {key: value for key, value in differences.items() if key not in allowed}
    if unexpected:
        raise RuntimeError(f"unexpected training configuration differences: {unexpected}")
    for arm, report in results.items():
        metrics = report["metrics"]
        protocol = metrics.get("rejector_protocol", {})
        if protocol.get("hard_labels_used") is not False:
            raise RuntimeError(f"hard labels used in rejector protocol for {arm}")
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"test labels entered training/calibration for {arm}")
        coverage = metrics.get("calibration_known_accept_rate")
        if coverage is None or abs(float(coverage) - 0.95) > 0.02:
            raise RuntimeError(f"known-validation coverage out of tolerance for {arm}: {coverage}")
        if metrics.get("cluster_k") is None or metrics.get("clustering_skipped"):
            raise RuntimeError(f"automatic clustering did not complete for {arm}")
        if len(report["history"]) != int(base_student.get("epochs", 4)):
            raise RuntimeError(f"incomplete training history for {arm}")
    treatment_losses = [
        float(epoch.get("train", {}).get("rejection_feat_kd", 0.0))
        for epoch in results[treatment_arm]["history"]
    ]
    if any(loss <= 0.0 for loss in treatment_losses):
        raise RuntimeError(f"rejection feature KD inactive in an epoch: {treatment_losses}")
    if options.compare_weighting_only:
        uniform_losses = [
            float(epoch.get("train", {}).get("rejection_feat_kd", 0.0))
            for epoch in results["uniform_weight"]["history"]
        ]
        if any(loss <= 0.0 for loss in uniform_losses):
            raise RuntimeError(f"uniform rejection feature KD inactive: {uniform_losses}")

        if base_student.get("kd_uncertainty_source") != "head":
            weight_spreads = {
                arm: [
                    float(epoch.get("train", {}).get("kd_weight_std", 0.0))
                    for epoch in results[arm]["history"]
                ]
                for arm in arms
            }
            if any(spread <= 1e-4 for values in weight_spreads.values() for spread in values):
                raise RuntimeError(
                    "configured teacher uncertainty source has degenerate KD weights: "
                    f"{weight_spreads}"
                )
        else:
            weight_spreads = {
                arm: [
                    float(epoch.get("train", {}).get("kd_weight_std", 0.0))
                    for epoch in results[arm]["history"]
                ]
                for arm in arms
            }

    result = {
        "question": (
            "Does teacher-uncertainty weighting add value over uniform "
            "rejection-feature KD at fixed coefficient and known coverage?"
            if options.compare_weighting_only
            else "Does uncertainty-weighted rejection-feature KD improve semantic-hard novel rejection at fixed known coverage?"
        ),
        "seed": seed,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": (
            "rejection_feature_kd_weighting: uniform vs teacher_uncertainty at alpha=0.1"
            if options.compare_weighting_only
            else f"alpha_rejection_feat_kd: 0 vs 0.1 (weighting={options.feature_kd_weighting})"
        ),
        "teacher_signal": (
            "frozen teacher rejection embedding; cosine feature KD weighted by detached teacher uncertainty"
            if options.feature_kd_weighting == "teacher_uncertainty"
            else "frozen teacher rejection embedding; cosine feature KD with uniform sample weights"
        ),
        "pool_or_test_labels_used_for_training_or_selection": False,
        "training_config_differences": differences,
        "treatment_rejection_feature_kd_loss_by_epoch": treatment_losses,
        "uniform_rejection_feature_kd_loss_by_epoch": (
            uniform_losses if options.compare_weighting_only else None
        ),
        "teacher_uncertainty_source": base_student.get("kd_uncertainty_source"),
        "teacher_uncertainty_weight_std_by_epoch": (
            weight_spreads if options.compare_weighting_only else None
        ),
        "arms": results,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "config_differences": differences, "treatment_losses": treatment_losses}, indent=2), flush=True)


if __name__ == "__main__":
    main()
