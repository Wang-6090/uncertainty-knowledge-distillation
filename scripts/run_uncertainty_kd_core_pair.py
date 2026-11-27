"""Run a fresh-split no-KD / standard-KD / uncertainty-KD comparison.

The original two-arm comparison changes only ``kd_mode``. An optional third
arm disables logits KD (``alpha_kd=0``) while retaining the same auxiliary
training objective; it is deliberately not described as pure CE. Feature KD
and rejection-branch KD are disabled in all arms.

``--objective-profile ce_only`` adds a separate isolated objective ablation.
It creates CE-only, CE+standard-KD, and CE+uncertainty-KD arms by zeroing all
numeric ``alpha_*`` auxiliary coefficients, disabling discovery-pool training,
and evaluating every arm with ``max_softmax``. This profile is kept separate
from the current auxiliary-loss profile so the existing results remain
interpretable.
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

from scripts.make_cifar100_splits import read_cifar100_taxonomy, build_protocols  # noqa: E402
from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


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


def make_split(seed: int, split_dir: Path, data_root: Path) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "cifar100_60_40_semantic_hard.json"
    if split_path.exists():
        split = read_json(split_path)
        if split.get("metadata", {}).get("seed") != seed or split.get("protocol") != "semantic_hard":
            raise ValueError(f"existing split does not match requested protocol: {split_path}")
        return split_path
    fine_names, coarse_names, mapping = read_cifar100_taxonomy(data_root)
    split = build_protocols(fine_names, coarse_names, mapping, seed)["semantic_hard"]
    write_json(split_path, split)
    return split_path


def config_differences(left: dict[str, Any], right: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    keys = sorted(set(left) | set(right))
    return {
        key: (left.get(key), right.get(key))
        for key in keys
        if left.get(key) != right.get(key)
    }


def make_ce_only_student_config(source: dict[str, Any]) -> dict[str, Any]:
    """Create an explicit CE/KD-only student objective from a saved config."""

    config = dict(source)
    for key, value in list(config.items()):
        if key.startswith("alpha_") and isinstance(value, (int, float)) and not isinstance(value, bool):
            config[key] = 0.0
    config.update(
        {
            "alpha_kd": 1.0,
            "alpha_feat_kd": 0.0,
            "alpha_rejection_kd": 0.0,
            "alpha_rejection_feat_kd": 0.0,
            "discovery_pool": False,
            "discovery_pool_mode": "unknown",
            # The saved auxiliary-pipeline config may carry a mixed-pool
            # candidate-gating choice even when every discovery coefficient
            # is zero.  Clear those declarative switches as well so the
            # CE/KD-only profile is internally valid and cannot accidentally
            # re-enable a discovery-path precondition.
            "discovery_feature_candidate_gating": False,
            "discovery_cross_view_gating": False,
            "discovery_single_view_candidate_gating": False,
            "discovery_feature_candidate_space": "shared",
            "joint_discovery": False,
            "joint_candidate_gating": False,
            "joint_mixed_known_consistency": False,
            "outlier_batch_size": 0,
            "rejection_feature_dim": 0,
            "independent_rejection_encoder": False,
            "checkpoint_selection": "known_acc",
            "open_val_ratio": 0.0,
        }
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2071, 2072, 2073])
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--tag", default="uncertainty_kd_core")
    parser.add_argument(
        "--objective-profile",
        choices=["current_auxiliary", "ce_only"],
        default="current_auxiliary",
        help=(
            "current_auxiliary compares the existing auxiliary-loss pipeline; "
            "ce_only compares CE-only, CE+standard-KD, and CE+uncertainty-KD"
        ),
    )
    parser.add_argument(
        "--include-no-kd",
        action="store_true",
        help="also run a matched arm with alpha_kd=0; other auxiliary losses remain active",
    )
    parser.add_argument(
        "--kd-uncertainty-source",
        choices=["head", "mc_epistemic", "mc_predictive_entropy"],
        default="head",
        help="uncertainty source shared by both KD arms; only uncertainty mode applies its weights",
    )
    parser.add_argument(
        "--kd-mc-samples",
        type=int,
        default=4,
        help="teacher stochastic passes when an MC uncertainty source is selected",
    )
    options = parser.parse_args()

    if options.kd_uncertainty_source != "head" and options.kd_mc_samples < 2:
        raise ValueError("--kd-mc-samples must be at least 2 for MC uncertainty")

    teacher_source = resolve(options.source_teacher_config)
    student_source = resolve(options.source_student_config)
    discover_source = resolve(options.source_discover_config)
    for path in (teacher_source, student_source, discover_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_student = read_json(student_source)
    base_teacher = read_json(teacher_source)
    if base_student.get("checkpoint_selection") != "known_acc":
        raise ValueError("use known-only checkpoint selection for this experiment")
    if float(base_student.get("alpha_kd", 0.0)) <= 0.0:
        raise ValueError("logit KD must be active")
    if options.objective_profile == "current_auxiliary":
        if base_student.get("discovery_pool_mode") != "mixed":
            raise ValueError("student reference config must use a mixed discovery pool")
        if float(base_student.get("alpha_rejection_kd", 0.0)) != 0.0:
            raise ValueError("rejection-logit KD must remain disabled to isolate the main KD term")
        if float(base_student.get("alpha_rejection_feat_kd", 0.0)) != 0.0:
            raise ValueError("rejection-feature KD must remain disabled")
        if float(base_student.get("alpha_discovery_rejection_pu_feature_margin", 0.0)) <= 0.0:
            raise ValueError("the fixed new-class rejection objective must be active")
    else:
        base_student = make_ce_only_student_config(base_student)

    base_student["alpha_feat_kd"] = 0.0
    base_student["kd_mode"] = "standard"
    base_student["kd_uncertainty_source"] = options.kd_uncertainty_source
    base_student["kd_mc_samples"] = int(options.kd_mc_samples)
    base_student["open_val_ratio"] = 0.0
    base_discover = read_json(discover_source)
    if base_discover.get("threshold_policy") not in {"known_coverage", None}:
        raise ValueError("discovery reference config must use known-only calibration")

    all_results: dict[str, Any] = {}
    data_root = (ROOT / str(base_student.get("data_root", ".\\data"))).resolve()
    for seed in options.seeds:
        split_dir = ROOT / f"splits_cifar100_semantic_hard_seed{seed}"
        split_path = make_split(seed, split_dir, data_root)
        split_metadata = read_json(split_path)
        if len(split_metadata.get("known_classes", [])) != 60 or len(split_metadata.get("novel_classes", [])) != 40:
            raise ValueError("semantic-hard split must have 60 known and 40 novel classes")

        teacher_dir = ROOT / "runs" / f"{options.tag}_s{seed}_teacher"
        teacher_ckpt = teacher_dir / "teacher.pt"
        student_config_path = ROOT / "analysis" / f"{options.tag}_s{seed}_student_config.json"
        result_path = ROOT / "analysis" / f"{options.tag}_s{seed}.json"
        if options.objective_profile == "ce_only":
            if options.include_no_kd:
                raise ValueError("--include-no-kd is only valid for current_auxiliary")
            arm_specs = [
                ("ce_only", "standard", "mean_normalized", 0.0),
                ("standard", "standard", "mean_normalized", float(base_student["alpha_kd"])),
                ("uncertainty", "uncertainty", "mean_normalized", float(base_student["alpha_kd"])),
            ]
        else:
            arm_specs = [("standard", "standard", "mean_normalized", float(base_student["alpha_kd"]))]
            if options.include_no_kd:
                arm_specs.insert(0, ("no_kd", "standard", "mean_normalized", 0.0))
            arm_specs.append(("uncertainty", "uncertainty", "mean_normalized", float(base_student["alpha_kd"])))
        arm_names = [spec[0] for spec in arm_specs]
        run_dirs = [ROOT / "runs" / f"{options.tag}_s{seed}_{arm}" for arm in arm_names]
        discover_dirs = [ROOT / "runs" / f"{options.tag}_s{seed}_{arm}_discover" for arm in arm_names]
        # Resume only from verifiable artifacts. Incomplete/unrecognized paths
        # are never overwritten.
        if result_path.exists():
            completed = read_json(result_path)
            if int(completed.get("seed", -1)) != int(seed):
                raise ValueError(f"existing result has a mismatched seed: {result_path}")
            completed_arms = completed.get("arms", {})
            if set(completed_arms) != set(arm_names):
                raise ValueError(f"existing result does not contain the requested arms: {result_path}")
            for arm, run_dir, discover_dir in zip(arm_names, run_dirs, discover_dirs):
                if not (run_dir / "student.pt").is_file():
                    raise FileNotFoundError(f"completed result is missing its student checkpoint: {run_dir}")
                if not (run_dir / "train_student_config.json").is_file():
                    raise FileNotFoundError(f"completed result is missing its saved training config: {run_dir}")
                if not (discover_dir / "discovery_report.json").is_file():
                    raise FileNotFoundError(f"completed result is missing its discovery report: {discover_dir}")
                if completed_arms[arm].get("train_dir") != str(run_dir):
                    raise ValueError(f"completed result points to a different {arm} run: {result_path}")
                if completed_arms[arm].get("discover_dir") != str(discover_dir):
                    raise ValueError(f"completed result points to a different {arm} discovery run: {result_path}")
                saved_config = read_json(run_dir / "train_student_config.json")
                recorded_config = completed_arms[arm].get("train_config", {})
                if saved_config != recorded_config:
                    raise ValueError(f"completed result training config disagrees with its run artifact: {arm}, {result_path}")
                expected_mode = "uncertainty" if arm == "uncertainty" else "standard"
                expected_alpha = 0.0 if arm in {"no_kd", "ce_only"} else float(base_student["alpha_kd"])
                if (
                    saved_config.get("kd_mode") != expected_mode
                    or float(saved_config.get("alpha_kd", -1)) != expected_alpha
                    or float(saved_config.get("alpha_feat_kd", -1)) != 0.0
                    or saved_config.get("kd_uncertainty_source") != options.kd_uncertainty_source
                    or int(saved_config.get("kd_mc_samples", -1)) != int(options.kd_mc_samples)
                ):
                    raise ValueError(f"completed result arm has an unexpected KD treatment: {arm}, {result_path}")
            print(f"[seed={seed}] reuse verified completed result", flush=True)
            all_results[str(seed)] = str(result_path)
            continue
        for run_dir in run_dirs:
            if run_dir.exists() and not (run_dir / "student.pt").is_file():
                raise FileExistsError(f"refusing to overwrite incomplete student run: {run_dir}")
        for discover_dir in discover_dirs:
            if discover_dir.exists() and not (discover_dir / "discovery_report.json").is_file():
                raise FileExistsError(f"refusing to overwrite incomplete discover run: {discover_dir}")
        if teacher_dir.exists() and not teacher_ckpt.is_file():
            raise FileExistsError(f"refusing to overwrite incomplete teacher run: {teacher_dir}")

        teacher_args = load_args(teacher_source, "train_teacher", teacher_dir, seed)
        teacher_args.split_path = str(split_path)
        teacher_args.teacher_ckpt = str(teacher_ckpt)
        teacher_args.open_val_ratio = 0.0
        if not teacher_ckpt.is_file():
            print(f"[seed={seed}] train split-matched teacher", flush=True)
            fit_teacher(teacher_args)
        else:
            print(f"[seed={seed}] reuse existing split-matched teacher", flush=True)

        student_config = dict(base_student)
        student_config.update({
            "split_path": str(split_path),
            "seed": int(seed),
            "alpha_feat_kd": 0.0,
            "kd_mode": "standard",
            "open_val_ratio": 0.0,
        })
        if not student_config_path.exists():
            write_json(student_config_path, student_config)
        else:
            saved_student_config = read_json(student_config_path)
            config_diffs = config_differences(saved_student_config, student_config)
            if config_diffs:
                raise ValueError(
                    "existing paired student reference config does not match the "
                    f"requested objective profile: {student_config_path}; differences={config_diffs}"
                )

        arms: dict[str, Any] = {}
        for arm, kd_mode, weight_mode, alpha_kd in arm_specs:
            run_dir = ROOT / "runs" / f"{options.tag}_s{seed}_{arm}"
            discover_dir = ROOT / "runs" / f"{options.tag}_s{seed}_{arm}_discover"
            train_args = load_args(student_config_path, "train_student", run_dir, seed)
            train_args.split_path = str(split_path)
            train_args.teacher_ckpt = str(teacher_ckpt)
            train_args.student_ckpt = str(run_dir / "student.pt")
            train_args.kd_mode = kd_mode
            train_args.alpha_kd = alpha_kd
            # Match the mean KD scale to standard KL; only redistribute
            # per-example mass according to teacher uncertainty.
            train_args.uncertainty_weight_mode = weight_mode
            existing_run = run_dir / "student.pt"
            if existing_run.is_file():
                saved_train_config = read_json(run_dir / "train_student_config.json")
                if (
                    saved_train_config.get("kd_mode") != kd_mode
                    or float(saved_train_config.get("alpha_feat_kd", -1)) != 0.0
                    or saved_train_config.get("uncertainty_weight_mode") != weight_mode
                    or float(saved_train_config.get("alpha_kd", -1)) != alpha_kd
                ):
                    raise ValueError(f"existing arm configuration is not the requested treatment: {run_dir}")
                print(f"[seed={seed}] reuse verified arm={arm}", flush=True)
            else:
                print(f"[seed={seed}] train arm={arm} kd_mode={kd_mode}", flush=True)
                fit_student(train_args)

            discover_args = load_args(discover_source, "discover", discover_dir, seed)
            discover_args.split_path = str(split_path)
            discover_args.student_ckpt = str(run_dir / "student.pt")
            discover_args.score_mode = (
                "max_softmax" if options.objective_profile == "ce_only" else "feature_rejector"
            )
            if options.objective_profile == "ce_only":
                # The CE-only evaluation deliberately does not fit or use a
                # rejector.  Clear the inherited rejector feature ablation so
                # validation cannot reject an otherwise valid max-softmax run.
                discover_args.rejector_feature_mode = "embedding"
                # Match the CE-only checkpoint architecture as well.  The
                # discovery source config belongs to the historical rejector
                # pipeline and may request an independent rejection encoder;
                # leaving that request active makes checkpoint loading fail
                # before max-softmax evaluation starts.
                discover_args.rejection_feature_dim = 0
                discover_args.independent_rejection_encoder = False
            discover_args.threshold_policy = "known_coverage"
            discover_args.target_known_coverage = 0.95
            discover_args.skip_clustering = False
            discover_args.cluster_k = "auto"
            discover_args.open_val_ratio = 0.0
            if not (discover_dir / "discovery_report.json").is_file():
                print(f"[seed={seed}] discover arm={arm}, automatic K", flush=True)
                discover(discover_args)
            else:
                print(f"[seed={seed}] reuse existing discovery report arm={arm}", flush=True)

            training = read_json(run_dir / "train_student_config.json")
            metrics = read_json(discover_dir / "discovery_report.json")
            history = read_json(run_dir / "train_history.json")
            train_stats = [entry.get("train", {}) for entry in history]
            arms[arm] = {
                "train_dir": str(run_dir),
                "discover_dir": str(discover_dir),
                "train_config": training,
                "discover_config": read_json(discover_dir / "discover_config.json"),
                "metrics": metrics,
                "best_epoch": training.get("best_epoch"),
                "training_epochs": len(history),
                "kd_loss_by_epoch": [stats.get("kd") for stats in train_stats],
                "feature_kd_loss_by_epoch": [stats.get("feat_kd") for stats in train_stats],
                "kd_weight_applied_by_epoch": [stats.get("kd_uncertainty_weight_applied") for stats in train_stats],
                "kd_weight_std_by_epoch": [stats.get("kd_weight_std") for stats in train_stats],
                "kd_weight_ess_fraction_by_epoch": [stats.get("kd_weight_ess_fraction") for stats in train_stats],
            }

        diff = config_differences(arms["standard"]["train_config"], arms["uncertainty"]["train_config"])
        algorithm_diffs = {
            key: values for key, values in diff.items()
            if key not in {"kd_mode", "work_dir", "student_ckpt", "best_epoch", "best_val_known_acc", "best_checkpoint_selection_score", "best_open_checkpoint_stats", "transform_profile_requested"}
        }
        if algorithm_diffs:
            raise RuntimeError(f"unexpected paired training config differences for seed {seed}: {algorithm_diffs}")
        control_arm = "ce_only" if options.objective_profile == "ce_only" else "no_kd"
        if options.include_no_kd or options.objective_profile == "ce_only":
            no_kd_diffs = config_differences(arms["standard"]["train_config"], arms[control_arm]["train_config"])
            allowed_no_kd_diffs = {
                key: values for key, values in no_kd_diffs.items()
                if key not in {"alpha_kd", "work_dir", "student_ckpt", "best_epoch", "best_val_known_acc", "best_checkpoint_selection_score", "best_open_checkpoint_stats", "transform_profile_requested"}
            }
            if allowed_no_kd_diffs:
                raise RuntimeError(f"unexpected control config differences for seed {seed}: {allowed_no_kd_diffs}")
        completed_epoch_counts = {arms[arm]["training_epochs"] for arm in arm_names}
        if len(completed_epoch_counts) != 1:
            raise RuntimeError(f"comparison arms did not complete the same number of epochs for seed {seed}: {completed_epoch_counts}")
        if any(float(arms[arm]["train_config"].get("alpha_feat_kd", -1)) != 0.0 for arm in arm_names):
            raise RuntimeError(f"feature KD coefficient was not zero for seed {seed}")
        if options.include_no_kd or options.objective_profile == "ce_only":
            if float(arms[control_arm]["train_config"].get("alpha_kd", -1)) != 0.0:
                raise RuntimeError(f"control arm has a nonzero logits KD coefficient for seed {seed}")
        if options.objective_profile == "ce_only":
            for key, value in arms[control_arm]["train_config"].items():
                if (
                    key.startswith("alpha_")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and float(value) != 0.0
                ):
                    raise RuntimeError(f"CE-only control has active auxiliary coefficient {key}={value}")
        standard_arm = "standard" if "standard" in arms else "no_kd"
        if any(arms[standard_arm]["kd_weight_applied_by_epoch"]):
            raise RuntimeError(f"standard-KD arm unexpectedly applied uncertainty weights for seed {seed}")
        if not all(arms["uncertainty"]["kd_weight_applied_by_epoch"]):
            raise RuntimeError(f"uncertainty-KD arm did not apply weights in every epoch for seed {seed}")
        if any(value is None or value <= 1e-8 for value in arms["uncertainty"]["kd_weight_std_by_epoch"]):
            raise RuntimeError(f"uncertainty-KD weights have no measurable spread for seed {seed}")
        if any(value is None or value <= 0.0 for value in arms["uncertainty"]["kd_loss_by_epoch"]):
            raise RuntimeError(f"uncertainty-KD loss path was not active for seed {seed}")
        if "standard" in arms and any(value is None or value <= 0.0 for value in arms["standard"]["kd_loss_by_epoch"]):
            raise RuntimeError(f"standard-KD loss path was not active for seed {seed}")
        for arm in arm_names:
            report = arms[arm]["metrics"]
            if report.get("cluster_k") is None or report.get("clustering_skipped"):
                raise RuntimeError(f"automatic-K clustering did not complete for {arm}, seed {seed}")
            coverage = report.get("calibration_known_accept_rate")
            if coverage is None or abs(float(coverage) - 0.95) > 0.02:
                raise RuntimeError(f"known-validation calibration missed target coverage for {arm}, seed {seed}: {coverage}")

        result = {
            "question": (
                "Does MC-epistemic weighting of logits KD improve semantic-hard novel-class discovery over unweighted KL on the same MC-mean teacher logits?"
                if options.kd_uncertainty_source == "mc_epistemic"
                else "Does teacher-uncertainty weighting of logits KD improve semantic-hard novel-class discovery over standard KL, conditional on the fixed current training pipeline?"
            ),
            "seed": int(seed),
            "split": str(split_path),
            "teacher_checkpoint": str(teacher_ckpt),
            "objective_profile": options.objective_profile,
            "kd_uncertainty_source": options.kd_uncertainty_source,
            "kd_mc_samples": int(options.kd_mc_samples),
            "single_changed_factor": (
                "CE-only vs CE+standard logits KD vs CE+uncertainty-weighted logits KD"
                if options.objective_profile == "ce_only"
                else ("alpha_kd=0 vs standard logits KD vs uncertainty-weighted logits KD" if options.include_no_kd else "kd_mode: standard vs uncertainty")
            ),
            "feature_kd": "disabled in all arms to isolate logits KD",
            "no_kd_control_definition": ("alpha_kd=0; all other supervised and auxiliary objectives remain active" if options.include_no_kd else None),
            "ce_only_control_definition": ("all numeric alpha_* coefficients are zero, discovery/outlier training is disabled, and evaluation uses max_softmax" if options.objective_profile == "ce_only" else None),
            "other_training_conditions": ("CE/KD-only known-class training; identical config except declared KD treatment and run/result-derived fields" if options.objective_profile == "ce_only" else "matched mixed-pool uncertainty-PU and rejection feature-margin pipeline; identical config except declared KD treatment and run/result-derived fields"),
            "checkpoint_selection": "known validation accuracy only",
            "threshold_calibration": "known validation only, target 95% coverage",
            "clustering": "automatic K; K selected from unlabeled candidate geometry; oracle-K metrics are separate upper-bound diagnostics",
            "pool_or_test_labels_used_for_training_or_selection": False,
            "training_config_differences": diff,
            "arms": arms,
        }
        write_json(result_path, result)
        all_results[str(seed)] = str(result_path)

    summary_path = ROOT / "analysis" / f"{options.tag}_summary.json"
    previous_results = read_json(summary_path).get("seeds", {}) if summary_path.exists() else {}
    previous_results.update(all_results)
    expected_seeds = set(int(seed) for seed in options.seeds)
    completed_seeds = {int(seed) for seed in previous_results}
    status = "completed" if completed_seeds == expected_seeds else "partial"
    write_json(summary_path, {
        "seeds": previous_results,
        "planned_seeds": sorted(expected_seeds),
        "objective_profile": options.objective_profile,
        "kd_uncertainty_source": options.kd_uncertainty_source,
        "kd_mc_samples": int(options.kd_mc_samples),
        "single_changed_factor": (
            "CE-only vs CE+standard logits KD vs CE+uncertainty-weighted logits KD"
            if options.objective_profile == "ce_only"
            else ("alpha_kd and kd_mode" if options.include_no_kd else "kd_mode")
        ),
        "status": status,
    })
    print(json.dumps({"summary": str(summary_path), "results": all_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
