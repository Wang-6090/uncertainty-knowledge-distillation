"""Paired fresh-split test of label-free MC epistemic/head rejector candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


def resolved(path: Path) -> Path:
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
    parser.add_argument("--seed", type=int, default=2068)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--source-teacher-config", type=Path, required=True)
    parser.add_argument("--source-student-config", type=Path, required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--candidate-ratio", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument(
        "--treatment-mode",
        choices=("mc_epistemic_head_fusion", "mc_epistemic_plus_head_uncertainty"),
        default="mc_epistemic_head_fusion",
        help="The one treatment selector; control remains entropy_uncertainty.",
    )
    parser.add_argument("--tag", default="rejection_mc_candidate_selector_s2068")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume only when existing artifacts match this exact seed/split/protocol.",
    )
    options = parser.parse_args()
    if not 0.0 < options.candidate_ratio <= 1.0 or options.mc_samples < 2:
        raise ValueError("candidate ratio must be in (0, 1] and MC samples >= 2")

    split_path = resolved(options.split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("metadata", {}).get("seed") != options.seed or split.get("protocol") != "semantic_hard":
        raise ValueError("split must be semantic-hard and match --seed")

    source_teacher = resolved(options.source_teacher_config)
    source_student = resolved(options.source_student_config)
    source_discover = resolved(options.source_discover_config)
    for path in (source_teacher, source_student, source_discover):
        if not path.is_file():
            raise FileNotFoundError(path)

    teacher_dir = ROOT / "runs" / f"{options.tag}_teacher"
    teacher_ckpt = teacher_dir / "teacher.pt"
    teacher_args = load_args(source_teacher, "train_teacher", teacher_dir, options.seed)
    teacher_args.split_path = str(split_path)
    teacher_args.teacher_ckpt = str(teacher_ckpt)
    if teacher_dir.exists():
        if not options.resume_existing:
            raise FileExistsError(f"refusing to overwrite {teacher_dir}")
        teacher_config = teacher_dir / "train_teacher_config.json"
        if not teacher_ckpt.is_file() or not teacher_config.is_file():
            raise RuntimeError(f"cannot resume: incomplete teacher artifacts in {teacher_dir}")
        saved_teacher = json.loads(teacher_config.read_text(encoding="utf-8"))
        if (
            saved_teacher.get("seed") != options.seed
            or Path(saved_teacher.get("split_path", "")).resolve() != split_path
        ):
            raise RuntimeError("existing teacher config does not match requested seed/split")
    else:
        fit_teacher(teacher_args)

    base = json.loads(source_student.read_text(encoding="utf-8"))
    if base.get("discovery_pool_mode") != "mixed":
        raise ValueError("student config must use a mixed discovery pool")
    base.update({
        "split_path": str(split_path),
        "seed": int(options.seed),
        "rejector_pool_selection": "high_risk",
        "rejector_candidate_ratio": float(options.candidate_ratio),
        "rejector_mc_samples": int(options.mc_samples),
    })
    config_path = ROOT / "analysis" / f"{options.tag}_student_config.json"
    result_path = ROOT / "analysis" / f"{options.tag}.json"
    if config_path.exists() or result_path.exists():
        if not options.resume_existing or result_path.exists() or not config_path.is_file():
            raise FileExistsError("refusing to overwrite existing pair artifacts")
        saved_base = json.loads(config_path.read_text(encoding="utf-8"))
        expected = dict(base)
        expected["work_dir"] = saved_base.get("work_dir")
        expected["student_ckpt"] = saved_base.get("student_ckpt")
        if saved_base != expected:
            raise RuntimeError("existing student config differs from requested protocol")
    else:
        config_path.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    treatment_arm = (
        "mc_rank_fusion"
        if options.treatment_mode == "mc_epistemic_head_fusion"
        else "mc_raw_sum"
    )
    reports = {}
    for arm, mode in (
        ("baseline", "entropy_uncertainty"),
        (treatment_arm, options.treatment_mode),
    ):
        train_dir = ROOT / "runs" / f"{options.tag}_{arm}"
        discover_dir = ROOT / "runs" / f"{options.tag}_{arm}_discover"
        train_args = load_args(config_path, "train_student", train_dir, options.seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.rejector_candidate_mode = mode
        if train_dir.exists():
            if not options.resume_existing:
                raise FileExistsError(f"refusing to overwrite outputs for {arm}")
            saved_train = train_dir / "train_student_config.json"
            if not train_args.student_ckpt or not Path(train_args.student_ckpt).is_file() or not saved_train.is_file():
                raise RuntimeError(f"cannot resume: incomplete training artifacts for {arm}")
            saved_train_config = json.loads(saved_train.read_text(encoding="utf-8"))
            if (
                saved_train_config.get("seed") != options.seed
                or Path(saved_train_config.get("split_path", "")).resolve() != split_path
                or saved_train_config.get("rejector_candidate_mode") != mode
                or float(saved_train_config.get("rejector_candidate_ratio", -1)) != options.candidate_ratio
                or int(saved_train_config.get("rejector_mc_samples", -1)) != options.mc_samples
            ):
                raise RuntimeError(f"existing training config does not match requested protocol for {arm}")
            print(f"[resume] using completed student arm={arm}", flush=True)
        else:
            print(f"[train] arm={arm} candidate_mode={mode}", flush=True)
            fit_student(train_args)

        discover_args = load_args(source_discover, "discover", discover_dir, options.seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.skip_clustering = True
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_report = discover_dir / "discovery_report.json"
        if discover_report.is_file():
            if not options.resume_existing:
                raise FileExistsError(f"refusing to overwrite discovery output {discover_dir}")
            print(f"[resume] using completed discovery arm={arm}", flush=True)
        else:
            if discover_dir.exists() and not options.resume_existing:
                raise FileExistsError(f"refusing to overwrite discovery output {discover_dir}")
            print(f"[discover] arm={arm}", flush=True)
            discover(discover_args)
        reports[arm] = {
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "train_config": json.loads((train_dir / "train_student_config.json").read_text(encoding="utf-8")),
            "discover_config": json.loads((discover_dir / "discover_config.json").read_text(encoding="utf-8")),
            "metrics": json.loads((discover_dir / "discovery_report.json").read_text(encoding="utf-8")),
        }

    result = {
        "question": f"Does {options.treatment_mode} high-risk selection improve rejection without reducing known utility?",
        "seed": int(options.seed),
        "split": str(split_path),
        "teacher_checkpoint": str(teacher_ckpt),
        "single_changed_factor": "rejector_candidate_mode",
        "candidate_ratio": float(options.candidate_ratio),
        "mc_samples": int(options.mc_samples),
        "treatment_mode": options.treatment_mode,
        "calibration": "known validation only, target 95% coverage",
        "pool_labels_used_for_selection_or_training": False,
        "test_labels_used_for_training_or_calibration": False,
        "evidence_level": "single fresh semantic-hard paired transfer experiment",
        "arms": reports,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "arms": list(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
