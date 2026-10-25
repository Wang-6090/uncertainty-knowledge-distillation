"""Run a preregistered 2x2 factorial audit of the current full pipeline.

The same split-matched teacher and student checkpoint are evaluated in four
discovery arms.  The crossed factors are clustering representation
(`projection_pca`/`feature_pca`) and score (`feature_rejector`/fixed
validation-calibrated uncertainty fusion).  Test and mixed labels are never
used for training, checkpoint selection, calibration or arm selection.
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def make_split(seed: int, split_path: Path, data_root: Path) -> Path:
    if split_path.exists():
        split = read_json(split_path)
        if split.get("metadata", {}).get("seed") != seed or split.get("protocol") != "semantic_hard":
            raise ValueError(f"existing split does not match locked protocol: {split_path}")
        return split_path
    names, coarse, mapping = read_cifar100_taxonomy(data_root)
    split = build_protocols(names, coarse, mapping, seed)["semantic_hard"]
    write_json(split_path, split)
    return split_path


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2118)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tag", default="rejection_full_pipeline_factorial")
    options = parser.parse_args()
    seed = int(options.seed)

    source_teacher = ROOT / "runs/rejection_encoder_init_fresh_s2083_teacher/train_teacher_config.json"
    source_student = ROOT / "runs/rejection_encoder_init_fresh_s2083_shared_backbone/train_student_config.json"
    source_discover = ROOT / "runs/rejection_encoder_init_fresh_s2083_shared_backbone_discover/discover_config.json"
    for path in (source_teacher, source_student, source_discover):
        if not path.is_file():
            raise FileNotFoundError(path)

    data_root = resolve(options.data_root)
    split_path = ROOT / f"splits_cifar100_semantic_hard_seed{seed}/cifar100_60_40_semantic_hard.json"
    split_path = make_split(seed, split_path, data_root)
    split = read_json(split_path)
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise ValueError("expected semantic-hard CIFAR-100 60/40 split")

    teacher_dir = ROOT / "runs" / f"{options.tag}_s{seed}_teacher"
    student_dir = ROOT / "runs" / f"{options.tag}_s{seed}_student"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_ckpt = student_dir / "student.pt"
    arms = {
        f"{score}_{feature}": (score, feature)
        for score in ("rejector", "fusion")
        for feature in ("projection_pca", "feature_pca")
    }
    arm_dirs = {name: ROOT / "runs" / f"{options.tag}_s{seed}_{name}" for name in arms}
    result_path = ROOT / "analysis" / f"{options.tag}_s{seed}.json"
    if result_path.exists() or any(path.exists() for path in (teacher_dir, student_dir, *arm_dirs.values())):
        raise FileExistsError("refusing to overwrite an existing factorial experiment")

    student_cfg = read_json(source_student)
    if student_cfg.get("independent_rejection_encoder_init") != "shared_backbone":
        raise ValueError("student source must use shared_backbone rejection initialization")
    if student_cfg.get("independent_rejection_input_norm", "none") != "none":
        raise ValueError("student source must use the registered none input norm")
    if student_cfg.get("alpha_vos", 0.0) != 0.0 or student_cfg.get("alpha_vos_rejection", 0.0) != 0.0:
        raise ValueError("VOS must be disabled")
    if student_cfg.get("alpha_discovery_uncertainty_feature_margin", 0.0) != 0.0:
        raise ValueError("training-time uncertainty feature margin must be disabled")

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
    student_args.alpha_discovery_uncertainty_feature_margin = 0.0
    student_args.student_ckpt = str(student_ckpt)
    fit_student(student_args)

    outputs: dict[str, Any] = {}
    for name, (score, cluster_feature) in arms.items():
        out_dir = arm_dirs[name]
        discover_args = load_args(source_discover, "discover", out_dir, seed)
        discover_args.split_path = str(split_path)
        discover_args.device = options.device
        discover_args.independent_rejection_encoder = True
        discover_args.independent_rejection_encoder_init = "shared_backbone"
        discover_args.student_ckpt = str(student_ckpt)
        discover_args.score_mode = "feature_rejector" if score == "rejector" else "feature_rejector_uncertainty_fusion"
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.open_val_ratio = 0.0
        discover_args.cluster_k = "auto"
        discover_args.skip_clustering = False
        discover_args.cluster_feature = cluster_feature
        discover(discover_args)
        report = read_json(out_dir / "discovery_report.json")
        config = read_json(out_dir / "discover_config.json")
        protocol = report.get("rejector_protocol", {})
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"label-flow audit failed: {out_dir}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"known validation calibration outside tolerance: {out_dir}")
        outputs[name] = {"config": config, "metrics": report, "output": str(out_dir)}

    config_differences: dict[str, list[Any]] = {}
    reference = outputs["rejector_projection_pca"]["config"]
    for name, output in outputs.items():
        config_differences[name] = {
            key: (reference.get(key), output["config"].get(key))
            for key in sorted(set(reference) | set(output["config"]))
            if reference.get(key) != output["config"].get(key)
        }
    result = {
        "protocol": "fresh-split 2x2 full-pipeline factorial",
        "seed": seed,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "training_performed": True,
        "test_and_mixed_labels_used_for_training_selection_or_calibration": False,
        "test_labels_used_only_for_retrospective_metrics": True,
        "factors": {
            "score": ["feature_rejector", "feature_rejector_uncertainty_fusion"],
            "cluster_feature": ["projection_pca", "feature_pca"],
        },
        "arms": outputs,
        "config_differences_from_rejector_projection": config_differences,
        "success_criterion": {
            "full_treatment": "fusion_feature_pca",
            "unknown_rejection_gain_at_least": 0.02,
            "known_acceptance_drop_at_most": 0.01,
            "fpr95_increase_at_most": 0.01,
            "oscr_drop_at_most": 0.005,
            "accepted_known_accuracy_drop_at_most": 0.005,
            "auto_k_absolute_error_must_not_increase": True,
            "no_test_or_mixed_label_selection": True,
        },
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "arms": list(outputs)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
