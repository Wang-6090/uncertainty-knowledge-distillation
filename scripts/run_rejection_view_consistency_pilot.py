"""Single-factor pilot for candidate-boundary rejection-score consistency.

The pilot starts from the existing late-decay protocol on semantic-hard
seed=2125.  Control and treatment share the teacher, seed, optimizer/model
configuration, late rejection-CE schedule, discovery gate, detector, known
validation threshold policy, and clustering protocol. Treatment changes only
the scalar rejection-risk consistency weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    args = parse_args([command])
    for key, value in read_json(path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2125)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--tag", default="rejection_score_consistency_pilot_s2125")
    args = parser.parse_args()
    if args.weight <= 0.0:
        raise ValueError("--weight must be positive for the treatment arm")

    seed = int(args.seed)
    source_run = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_late_decay"
    source_config = source_run / "config.json"
    source_discover = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_late_decay_discover" / "config.json"
    teacher_ckpt = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_teacher" / "teacher.pt"
    split_path = ROOT / "splits" / "semantic_hard" / f"seed{seed}" / "cifar100_60_40_semantic_hard.json"
    for path in (source_config, source_discover, teacher_ckpt, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result_path = ROOT / "analysis" / f"{args.tag}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")

    arms = {
        "control": 0.0,
        "treatment": float(args.weight),
    }
    train_dirs = {
        arm: ROOT / "runs" / f"{args.tag}_{arm}"
        for arm in arms
    }
    discover_dirs = {
        arm: ROOT / "runs" / f"{args.tag}_{arm}_discover"
        for arm in arms
    }
    if any(path.exists() for path in (*train_dirs.values(), *discover_dirs.values())):
        raise FileExistsError("refusing to overwrite existing pilot artifacts")

    reports = {}
    for arm, consistency_weight in arms.items():
        train_dir = train_dirs[arm]
        discover_dir = discover_dirs[arm]
        train_args = load_args(source_config, "train_student", train_dir, seed)
        train_args.split_path = str(split_path)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.rejection_known_ce_decay_start_epoch = 3
        train_args.rejection_known_ce_decay_epochs = 2
        train_args.rejection_known_ce_final_scale = 0.25
        train_args.alpha_discovery_rejection_score_consistency = consistency_weight
        train_args.discovery_feature_candidate_gating = True
        train_args.discovery_cross_view_gating = True
        train_args.discovery_cross_view_aggregation = "union_budgeted"
        train_args.single_changed_factor = "alpha_discovery_rejection_score_consistency"
        print(f"[student] arm={arm}, consistency_weight={consistency_weight}", flush=True)
        fit_student(train_args)

        discover_args = load_args(source_discover, "discover", discover_dir, seed)
        discover_args.split_path = str(split_path)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover_args.score_mode = "feature_rejector"
        discover_args.rejector_training = "nnpu"
        discover_args.rejector_nnpu_risk = "nu_corrected"
        discover_args.rejector_pool_selection = "all"
        discover_args.cluster_feature = "feature_pca"
        discover_args.cluster_selection = "silhouette"
        discover_args.cluster_k = "auto"
        discover_args.cluster_candidate_mode = "detector"
        discover_args.candidate_purify = "none"
        discover_args.candidate_keep_ratio = 1.0
        discover_args.threshold_policy = "known_coverage"
        discover_args.target_known_coverage = 0.95
        discover_args.skip_clustering = False
        print(f"[discover] arm={arm}", flush=True)
        discover(discover_args)

        history = read_json(train_dir / "train_history.json")
        reports[arm] = {
            "consistency_weight": consistency_weight,
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "train_config": read_json(train_dir / "train_student_config.json"),
            "train_history": history,
            "discovery_report": read_json(discover_dir / "discovery_report.json"),
            "calibration_report": read_json(discover_dir / "calibration_report.json"),
        }

    write_json(result_path, {
        "protocol": args.tag,
        "seed": seed,
        "split_path": str(split_path),
        "question": "Does candidate-gated scalar rejection-risk consistency improve the unstable unknown boundary on semantic-hard seed=2125?",
        "control": "late-decay protocol with scalar rejection-risk consistency disabled",
        "treatment": f"same protocol with scalar rejection-risk consistency weight {args.weight:g}",
        "single_changed_factor": "alpha_discovery_rejection_score_consistency",
        "candidate_gate": "label-free paired shared-boundary union_budgeted gate",
        "calibration": "known validation only, target 95% coverage",
        "test_labels_used_for_training_or_calibration": False,
        "mixed_pool_labels_used_for_training_or_calibration": False,
        "cluster_selection_used_test_or_mixed_labels": False,
        "arms": reports,
    })
    print(f"saved {result_path}")


if __name__ == "__main__":
    main()
