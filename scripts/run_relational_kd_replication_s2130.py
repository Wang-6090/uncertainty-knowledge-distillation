"""Independent semantic-hard replication of the relational-KD paired pilot.

The relational-KD coefficient is frozen at the value used in the seed-2109
pilot. The only intended training contrast is alpha_relational_kd=0 versus
3.34. A split-matched teacher is trained once and shared by both student arms.
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


DEFAULT_SEED = 2130
RELATIONAL_ALPHA = 3.34
SOURCE_TEACHER = ROOT / "runs/rejection_epoch_budget_s2109_teacher/train_teacher_config.json"
SOURCE_STUDENT = ROOT / "runs/rejection_epoch_budget_s2109_four_epoch/train_student_config.json"
SOURCE_DISCOVER = ROOT / "runs/rejection_epoch_budget_s2109_four_epoch_discover/discover_config.json"
ARMS = ("control", "relational_kd")


def experiment_paths(seed: int) -> dict[str, Path | str]:
    tag = f"relational_kd_independent_replication_s{seed}_20261115"
    base = ROOT / "runs" / tag
    teacher_dir = ROOT / "runs" / f"{tag}_teacher"
    return {
        "tag": tag,
        "base": base,
        "split_path": base / "protocol" / "cifar100_60_40_semantic_hard.json",
        "teacher_dir": teacher_dir,
        "teacher_ckpt": teacher_dir / "teacher.pt",
        "result_path": ROOT / "analysis" / f"{tag}.json",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_args(config_path: Path, command: str, work_dir: Path, seed: int, split_path: Path):
    args = parse_args([command])
    for key, value in read_json(config_path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.data_root = str(ROOT / "data")
    args.split_path = str(split_path)
    args.work_dir = str(work_dir)
    args.device = "auto"
    return args


def build_and_validate_split(seed: int, split_path: Path) -> dict[str, Any]:
    fine_names, coarse_names, mapping = read_cifar100_taxonomy(ROOT / "data")
    generated = build_protocols(fine_names, coarse_names, mapping, seed)["semantic_hard"]
    if split_path.exists():
        split = read_json(split_path)
        if split != generated:
            raise RuntimeError(f"Existing split differs from deterministic protocol: {split_path}")
    else:
        split = generated

    known = set(split["known_classes"])
    novel = set(split["novel_classes"])
    if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != seed:
        raise ValueError("Generated protocol metadata does not match this replication")
    if len(known) != 60 or len(novel) != 40 or known & novel or known | novel != set(range(100)):
        raise ValueError("Expected a disjoint, exhaustive 60-known/40-novel split")
    known_per_coarse = {coarse: 0 for coarse in range(20)}
    novel_per_coarse = {coarse: 0 for coarse in range(20)}
    for fine_id in known:
        known_per_coarse[int(mapping[fine_id])] += 1
    for fine_id in novel:
        novel_per_coarse[int(mapping[fine_id])] += 1
    if set(known_per_coarse.values()) != {3} or set(novel_per_coarse.values()) != {2}:
        raise ValueError("Semantic-hard protocol must contain 3 known and 2 novel per superclass")
    previous_splits = (
        ROOT / "splits/semantic_hard/seed2109/cifar100_60_40_semantic_hard.json",
        ROOT / "runs/relational_kd_independent_replication_s2130_20261115/protocol/cifar100_60_40_semantic_hard.json",
    )
    for previous_path in previous_splits:
        if previous_path.is_file():
            previous_known = set(read_json(previous_path)["known_classes"])
            if known == previous_known:
                raise ValueError(f"Replication split duplicates an existing class assignment: {previous_path}")
    return split


def source_config_audit() -> dict[str, Any]:
    configs = {
        "teacher": read_json(SOURCE_TEACHER),
        "student": read_json(SOURCE_STUDENT),
        "discover": read_json(SOURCE_DISCOVER),
    }
    for name, config in configs.items():
        if config.get("dataset") != "cifar100":
            raise ValueError(f"Unexpected dataset in {name} source config")
    student = configs["student"]
    discover_config = configs["discover"]
    checks = {
        "student_epochs": student.get("epochs") == 4,
        "student_feature_kd": student.get("alpha_feat_kd") == 0.1,
        "student_relational_kd_baseline_disabled": float(student.get("alpha_relational_kd", 0.0)) == 0.0,
        "known_only_checkpoint_selection": student.get("checkpoint_selection") == "known_acc",
        "discover_feature_rejector": discover_config.get("score_mode") == "feature_rejector",
        "known_validation_threshold": discover_config.get("threshold_policy") == "known_coverage",
        "target_known_coverage_95": discover_config.get("target_known_coverage") == 0.95,
        "enabled_clustering": discover_config.get("skip_clustering") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Source recipe audit failed: {failed}")
    return {"checks": checks, "configs": configs}


def materialized_differences(left, right, ignored: set[str]) -> dict[str, Any]:
    return {
        key: [getattr(left, key), getattr(right, key)]
        for key in sorted(vars(left))
        if key not in ignored and getattr(left, key) != getattr(right, key)
    }


def run(seed: int = DEFAULT_SEED, resume_partial: bool = False, dry_run: bool = False) -> None:
    paths = experiment_paths(seed)
    tag = str(paths["tag"])
    split_path = Path(paths["split_path"])
    teacher_dir = Path(paths["teacher_dir"])
    teacher_ckpt = Path(paths["teacher_ckpt"])
    result_path = Path(paths["result_path"])
    for path in (SOURCE_TEACHER, SOURCE_STUDENT, SOURCE_DISCOVER):
        if not path.is_file():
            raise FileNotFoundError(path)
    if result_path.exists() and not resume_partial:
        raise FileExistsError(f"Refusing to overwrite {result_path}")

    split = build_and_validate_split(seed, split_path)
    source_audit = source_config_audit()
    source_teacher = source_audit["configs"]["teacher"]
    source_student = source_audit["configs"]["student"]
    source_discover = source_audit["configs"]["discover"]

    train_args = {}
    discover_args = {}
    for arm, alpha in (("control", 0.0), ("relational_kd", RELATIONAL_ALPHA)):
        train_dir = ROOT / "runs" / f"{tag}_{arm}"
        discovery_dir = ROOT / "runs" / f"{tag}_{arm}_discover"
        train_args[arm] = load_args(SOURCE_STUDENT, "train_student", train_dir, seed, split_path)
        train_args[arm].teacher_ckpt = str(teacher_ckpt)
        train_args[arm].student_ckpt = str(train_dir / "student.pt")
        train_args[arm].alpha_relational_kd = alpha
        discover_args[arm] = load_args(SOURCE_DISCOVER, "discover", discovery_dir, seed, split_path)
        discover_args[arm].student_ckpt = str(train_dir / "student.pt")

    pretrain_diffs = materialized_differences(
        train_args["control"], train_args["relational_kd"],
        {"alpha_relational_kd", "work_dir", "student_ckpt"},
    )
    if pretrain_diffs:
        raise RuntimeError(f"Paired training arguments differ before training: {pretrain_diffs}")
    pred_discover_diffs = materialized_differences(
        discover_args["control"], discover_args["relational_kd"],
        {"work_dir", "student_ckpt"},
    )
    if pred_discover_diffs:
        raise RuntimeError(f"Paired discovery arguments differ before evaluation: {pred_discover_diffs}")

    output_dirs = [teacher_dir, result_path]
    for arm in ARMS:
        output_dirs.extend([
            ROOT / "runs" / f"{tag}_{arm}",
            ROOT / "runs" / f"{tag}_{arm}_discover",
        ])
    if not resume_partial:
        collisions = [str(path) for path in output_dirs if path.exists()]
        if collisions:
            raise FileExistsError(f"Refusing to overwrite existing experiment outputs: {collisions}")

    if dry_run:
        print(json.dumps({
            "seed": seed,
            "split_path": str(split_path),
            "known_classes": len(split["known_classes"]),
            "novel_classes": len(split["novel_classes"]),
            "source_checks": source_audit["checks"],
            "training_pair_differences": pretrain_diffs,
            "discovery_pair_differences": pred_discover_diffs,
            "relational_alpha": RELATIONAL_ALPHA,
            "outputs": [str(path) for path in output_dirs],
            "dry_run": True,
        }, indent=2, ensure_ascii=False))
        return

    if not split_path.exists():
        write_json(split_path, split)

    teacher_complete = all((teacher_ckpt.is_file(), (teacher_dir / "train_history.json").is_file(),
                            (teacher_dir / "train_teacher_config.json").is_file()))
    if not (resume_partial and teacher_complete):
        teacher_args = load_args(SOURCE_TEACHER, "train_teacher", teacher_dir, seed, split_path)
        teacher_args.teacher_ckpt = str(teacher_ckpt)
        print(f"[teacher] split-matched training seed={seed}", flush=True)
        fit_teacher(teacher_args)
    teacher_payload = __import__("torch").load(teacher_ckpt, map_location="cpu", weights_only=False)
    if teacher_payload.get("known_classes") != split["known_classes"] or teacher_payload.get("novel_classes") != split["novel_classes"]:
        raise RuntimeError("Teacher checkpoint class lists do not match the replication split")

    arms_report: dict[str, Any] = {}
    for arm in ARMS:
        train_dir = Path(train_args[arm].work_dir)
        discovery_dir = Path(discover_args[arm].work_dir)
        training_complete = all((train_dir / name).is_file() for name in (
            "student.pt", "train_history.json", "train_student_config.json"))
        if not (resume_partial and training_complete):
            print(f"[train] arm={arm} alpha_relational_kd={train_args[arm].alpha_relational_kd}", flush=True)
            fit_student(train_args[arm])
        discovery_complete = all((discovery_dir / name).is_file() for name in (
            "discovery_report.json", "discover_config.json"))
        if not (resume_partial and discovery_complete):
            print(f"[discover] arm={arm}", flush=True)
            discover(discover_args[arm])

        train_config = read_json(train_dir / "train_student_config.json")
        history = read_json(train_dir / "train_history.json")
        metrics = read_json(discovery_dir / "discovery_report.json")
        if len(history) != 4:
            raise RuntimeError(f"Expected four training epochs for {arm}, got {len(history)}")
        if float(train_config.get("alpha_relational_kd", -1)) != train_args[arm].alpha_relational_kd:
            raise RuntimeError(f"Saved treatment coefficient mismatch for {arm}")
        protocol = metrics.get("rejector_protocol", {})
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"Test labels entered training/calibration in {arm}")
        known_coverage = float(metrics.get("calibration_known_accept_rate", float("nan")))
        if not 0.93 <= known_coverage <= 0.97:
            raise RuntimeError(f"Known-only calibration coverage out of tolerance for {arm}: {known_coverage}")
        arms_report[arm] = {
            "train_config": train_config,
            "history": history,
            "metrics": metrics,
            "train_dir": str(train_dir),
            "discover_dir": str(discovery_dir),
        }

    control = arms_report["control"]["train_config"]
    treatment = arms_report["relational_kd"]["train_config"]
    result_only = {"best_epoch", "best_val_known_acc", "best_checkpoint_selection_score", "best_open_checkpoint_stats"}
    path_only = {"work_dir", "student_ckpt"}
    differences = {
        key: [control.get(key), treatment.get(key)]
        for key in sorted(set(control) | set(treatment))
        if control.get(key) != treatment.get(key)
    }
    unexpected = set(differences) - result_only - path_only - {"alpha_relational_kd"}
    if unexpected or differences.get("alpha_relational_kd") != [0.0, RELATIONAL_ALPHA]:
        raise RuntimeError(f"Post-run paired config audit failed: {differences}")

    report = {
        "question": "Does frozen batchwise relational KD improve the four-epoch semantic-hard pipeline on an independent class split?",
        "seed": seed,
        "protocol": "semantic_hard",
        "split_path": str(split_path),
        "split_class_counts": {"known": 60, "novel": 40, "known_per_coarse": 3, "novel_per_coarse": 2},
        "source_recipe_audit": source_audit["checks"],
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "alpha_relational_kd: 0.0 vs 3.34",
        "coefficient_basis": "frozen from the known-training gradient-scale diagnostic before outcome evaluation",
        "pretraining_argument_differences": pretrain_diffs,
        "prediscovery_argument_differences": pred_discover_diffs,
        "posttraining_config_differences": differences,
        "unknown_labels_used_for_training_selection_or_calibration": False,
        "unknown_labels_used_only_for_final_metrics": True,
        "arms": arms_report,
    }
    write_json(result_path, report)
    print(json.dumps({"result": str(result_path), "config_differences": differences}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    run(seed=options.seed, resume_partial=options.resume_partial, dry_run=options.dry_run)
