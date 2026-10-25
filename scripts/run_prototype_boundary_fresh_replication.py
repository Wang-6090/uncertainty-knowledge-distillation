"""Fresh-split replication of the fixed prototype-boundary rejector treatment."""

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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def make_split(seed: int, path: Path, data_root: Path) -> Path:
    if path.exists():
        split = read_json(path)
        if split.get("metadata", {}).get("seed") != seed or split.get("protocol") != "semantic_hard":
            raise ValueError(f"split does not match semantic-hard seed {seed}: {path}")
        return path
    names, coarse, mapping = read_cifar100_taxonomy(data_root)
    write_json(path, build_protocols(names, coarse, mapping, seed)["semantic_hard"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2119)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tag", default="prototype_boundary_fresh_replication")
    options = parser.parse_args()
    seed = int(options.seed)

    source_teacher = ROOT / "runs/rejection_encoder_init_fresh_s2083_teacher/train_teacher_config.json"
    source_student = ROOT / "runs/rejection_encoder_init_fresh_s2083_shared_backbone/train_student_config.json"
    source_discover = ROOT / "runs/rejection_encoder_init_fresh_s2083_shared_backbone_discover/discover_config.json"
    for source in (source_teacher, source_student, source_discover):
        if not source.is_file():
            raise FileNotFoundError(source)

    split_path = make_split(
        seed,
        ROOT / f"splits_cifar100_semantic_hard_seed{seed}/cifar100_60_40_semantic_hard.json",
        resolve(options.data_root),
    )
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard 60/40 split")

    teacher_dir = ROOT / "runs" / f"{options.tag}_s{seed}_teacher"
    student_dir = ROOT / "runs" / f"{options.tag}_s{seed}_student"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_ckpt = student_dir / "student.pt"
    arm_dirs = {
        "all_pool": ROOT / "runs" / f"{options.tag}_s{seed}_all_pool",
        "prototype_boundary_weighted": ROOT / "runs" / f"{options.tag}_s{seed}_prototype_boundary_weighted",
    }
    result_path = ROOT / "analysis" / f"{options.tag}_s{seed}.json"
    if result_path.exists() or any(p.exists() for p in (teacher_dir, student_dir, *arm_dirs.values())):
        raise FileExistsError("refusing to overwrite existing fresh replication artifacts")

    student_cfg = read_json(source_student)
    required = {
        "independent_rejection_encoder_init": "shared_backbone",
        "independent_rejection_input_norm": "none",
        "alpha_vos": 0.0,
        "alpha_vos_rejection": 0.0,
        "alpha_discovery_uncertainty_feature_margin": 0.0,
    }
    for key, expected in required.items():
        if student_cfg.get(key, 0.0 if isinstance(expected, float) else None) != expected:
            raise ValueError(f"source student config violates fixed protocol: {key}")

    teacher_args = load_args(source_teacher, "train_teacher", teacher_dir, seed)
    teacher_args.split_path = str(split_path)
    teacher_args.device = options.device
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    fit_teacher(teacher_args)

    student_args = load_args(source_student, "train_student", student_dir, seed)
    student_args.split_path = str(split_path)
    student_args.device = options.device
    student_args.teacher_ckpt = str(teacher_ckpt)
    student_args.independent_rejection_encoder = True
    student_args.independent_rejection_encoder_init = "shared_backbone"
    student_args.independent_rejection_input_norm = "none"
    student_args.alpha_vos = 0.0
    student_args.alpha_vos_rejection = 0.0
    student_args.alpha_discovery_uncertainty_feature_margin = 0.0
    student_args.student_ckpt = str(student_ckpt)
    fit_student(student_args)

    arms = {
        "all_pool": {"rejector_pool_selection": "all"},
        "prototype_boundary_weighted": {
            "rejector_pool_selection": "weighted_boundary",
            "rejector_candidate_mode": "prototype_boundary",
            "rejector_candidate_ratio": 0.25,
            "rejector_candidate_weight": 2.0,
        },
    }
    outputs: dict[str, Any] = {}
    for name, overrides in arms.items():
        discover_args = load_args(source_discover, "discover", arm_dirs[name], seed)
        discover_args.split_path = str(split_path)
        discover_args.device = options.device
        discover_args.student_ckpt = str(student_ckpt)
        discover_args.score_mode = "feature_rejector"
        discover_args.rejector_pool_selection = overrides["rejector_pool_selection"]
        if name != "all_pool":
            discover_args.rejector_candidate_mode = "prototype_boundary"
            discover_args.rejector_candidate_ratio = 0.25
            discover_args.rejector_candidate_weight = 2.0
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.open_val_ratio = 0.0
        discover_args.skip_clustering = True
        print(f"[discover] seed={seed} arm={name}", flush=True)
        discover(discover_args)
        report = read_json(arm_dirs[name] / "discovery_report.json")
        config = read_json(arm_dirs[name] / "discover_config.json")
        protocol = report.get("rejector_protocol", {})
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"label-flow audit failed: {name}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"known validation calibration outside tolerance: {name}")
        outputs[name] = {"config": config, "metrics": report, "work_dir": str(arm_dirs[name])}

    allowed = {"rejector_pool_selection", "rejector_candidate_mode", "rejector_candidate_ratio", "rejector_candidate_weight", "work_dir"}
    reference = outputs["all_pool"]["config"]
    differences = {}
    for name, output in outputs.items():
        differences[name] = {
            key: (reference.get(key), output["config"].get(key))
            for key in sorted(set(reference) | set(output["config"]))
            if reference.get(key) != output["config"].get(key) and key in allowed
        }
    unexpected = {}
    for name, output in outputs.items():
        current = {
            key: (reference.get(key), output["config"].get(key))
            for key in sorted(set(reference) | set(output["config"]))
            if reference.get(key) != output["config"].get(key) and key not in allowed
        }
        if current:
            unexpected[name] = current
    if unexpected:
        raise RuntimeError(f"unexpected configuration differences: {unexpected}")

    result = {
        "protocol": "fresh semantic-hard prototype-boundary weighted rejector replication",
        "seed": seed,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "training_performed": True,
        "single_changed_detector_factor": "rejector_pool_selection: all vs weighted_boundary with fixed prototype_boundary ratio=0.25 weight=2.0",
        "test_and_mixed_labels_used_for_training_selection_or_calibration": False,
        "test_labels_used_only_for_retrospective_metrics": True,
        "arms": outputs,
        "allowed_config_differences": differences,
        "success_criterion": {
            "unknown_rejection_gain_at_least": 0.02,
            "known_acceptance_drop_at_most": 0.01,
            "fpr95_increase_at_most": 0.01,
            "oscr_drop_at_most": 0.005,
            "accepted_known_accuracy_drop_at_most": 0.005,
        },
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "arms": list(outputs)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
