"""Fresh semantic-hard paired audit for continuous mixed-pool risk weighting."""

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


def ensure_split(seed: int, path: Path, data_root: Path) -> Path:
    if path.is_file():
        split = read_json(path)
        if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != seed:
            raise ValueError(f"split does not match semantic-hard seed {seed}: {path}")
        return path
    names, coarse, mapping = read_cifar100_taxonomy(data_root)
    write_json(path, build_protocols(names, coarse, mapping, seed)["semantic_hard"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2161)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tag", default="rejector_continuous_risk_s2161")
    parser.add_argument(
        "--treatment-pool-selection",
        choices=("weighted_risk", "weighted_neighbor_risk"),
        default="weighted_risk",
    )
    parser.add_argument(
        "--source-teacher",
        type=Path,
        default=Path("runs/rejection_ce_decay_s2160_20261121_teacher/train_teacher_config.json"),
    )
    parser.add_argument(
        "--source-student",
        type=Path,
        default=Path("analysis/rejection_ce_decay_s2160_20261121_student_config.json"),
    )
    parser.add_argument(
        "--source-discover",
        type=Path,
        default=Path("runs/rejection_ce_decay_s2160_20261121_static_control_discover/discover_config.json"),
    )
    options = parser.parse_args()
    seed = int(options.seed)
    source_teacher = resolve(options.source_teacher)
    source_student = resolve(options.source_student)
    source_discover = resolve(options.source_discover)
    for path in (source_teacher, source_student, source_discover):
        if not path.is_file():
            raise FileNotFoundError(path)

    split_path = ensure_split(
        seed,
        ROOT / f"splits/semantic_hard/seed{seed}/cifar100_60_40_semantic_hard.json",
        ROOT / "data",
    )
    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    student_dir = ROOT / "runs" / f"{options.tag}_student"
    teacher_ckpt = teacher_dir / "teacher.pt"
    student_ckpt = student_dir / "student.pt"
    arm_dirs = {
        "all_pool": ROOT / "runs" / f"{options.tag}_all_pool",
        options.treatment_pool_selection: ROOT
        / "runs"
        / f"{options.tag}_{options.treatment_pool_selection}",
    }
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    if result_path.exists() or any(p.exists() for p in (teacher_dir, student_dir, *arm_dirs.values())):
        raise FileExistsError("refusing to overwrite existing fresh experiment artifacts")

    student_source = read_json(source_student)
    required = {
        "discovery_pool_mode": "mixed",
        "independent_rejection_encoder": True,
        "independent_rejection_encoder_init": "shared_backbone",
        "alpha_vos": 0.0,
        "alpha_vos_rejection": 0.0,
    }
    for key, expected in required.items():
        if student_source.get(key) != expected:
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
    student_args.student_ckpt = str(student_ckpt)
    student_args.rejection_known_ce_decay_start_epoch = 0
    student_args.rejection_known_ce_decay_epochs = 0
    student_args.rejection_known_ce_final_scale = 1.0
    fit_student(student_args)

    outputs = {}
    treatment_arm = options.treatment_pool_selection
    for arm, pool_selection in (
        ("all_pool", "all"),
        (treatment_arm, treatment_arm),
    ):
        discover_args = load_args(source_discover, "discover", arm_dirs[arm], seed)
        discover_args.split_path = str(split_path)
        discover_args.device = options.device
        discover_args.student_ckpt = str(student_ckpt)
        discover_args.score_mode = "feature_rejector"
        discover_args.rejector_training = "nnpu"
        discover_args.rejector_nnpu_risk = "nu_corrected"
        discover_args.rejector_pool_selection = pool_selection
        discover_args.rejector_candidate_weight = 2.0
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.open_val_ratio = 0.0
        discover_args.skip_clustering = False
        print(f"[discover] seed={seed} arm={arm} pool_selection={pool_selection}", flush=True)
        discover(discover_args)
        report_path = arm_dirs[arm] / "discovery_report.json"
        report = read_json(report_path)
        protocol = report.get("rejector_protocol", {})
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"label-flow audit failed: {arm}")
        outputs[arm] = {
            "work_dir": str(arm_dirs[arm]),
            "config": read_json(arm_dirs[arm] / "discover_config.json"),
            "metrics": report,
        }

    reference = outputs["all_pool"]["config"]
    allowed = {"rejector_pool_selection", "rejector_candidate_weight", "work_dir"}
    unexpected = {}
    for arm, output in outputs.items():
        differences = {
            key: (reference.get(key), output["config"].get(key))
            for key in sorted(set(reference) | set(output["config"]))
            if reference.get(key) != output["config"].get(key) and key not in allowed
        }
        if differences:
            unexpected[arm] = differences
    if unexpected:
        raise RuntimeError(f"unexpected configuration differences: {unexpected}")

    result = {
        "protocol": "fresh semantic-hard continuous label-free risk-weighted mixed-pool nnPU audit",
        "seed": seed,
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "single_changed_detector_factor": (
            f"rejector_pool_selection all vs {treatment_arm}"
        ),
        "test_and_mixed_labels_used_for_training_selection_or_calibration": False,
        "test_labels_used_only_for_retrospective_metrics": True,
        "arms": outputs,
    }
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "arms": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
