"""Run a matched shared-boundary versus soft-consensus control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


def load_args(path: Path, command: str, work_dir: Path, seed: int, epochs: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.epochs = int(epochs)
    args.work_dir = str(work_dir)
    return args


def run_arm(
    source_train: Path,
    source_discover: Path,
    seed: int,
    epochs: int,
    space: str,
    tag: str,
    soft_floor: float,
):
    work_dir = ROOT / "runs" / f"{tag}_{space}"
    discover_dir = ROOT / "runs" / f"{tag}_{space}_discover"
    student_ckpt = work_dir / "student.pt"
    if not student_ckpt.is_file():
        train_args = load_args(source_train, "train_student", work_dir, seed, epochs)
        train_args.student_ckpt = str(student_ckpt)
        train_args.alpha_discovery_knn_boundary = 0.1
        train_args.alpha_discovery_rejection_knn_boundary = 0.0
        train_args.alpha_discovery_rejection_boundary = 0.0
        train_args.alpha_discovery_rejection_pu_feature_margin = 0.0
        train_args.discovery_boundary_mode = "pu"
        train_args.discovery_boundary_pu_risk = "hard_nnpu"
        train_args.discovery_feature_candidate_gating = True
        train_args.discovery_cross_view_gating = True
        train_args.discovery_feature_candidate_space = space
        train_args.discovery_feature_candidate_soft_floor = (
            float(soft_floor) if space == "shared_boundary_soft_consensus" else 0.25
        )
        fit_student(train_args)
    if not (discover_dir / "discovery_report.json").is_file():
        discover_args = load_args(source_discover, "discover", discover_dir, seed, epochs)
        discover_args.student_ckpt = str(student_ckpt)
        discover_args.discovery_feature_candidate_gating = True
        discover_args.discovery_cross_view_gating = True
        discover_args.discovery_feature_candidate_space = space
        discover(discover_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--config-seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--soft-floor", type=float, default=0.25)
    parser.add_argument("--tag", default="shared_boundary_soft_consensus_control_s44_e4")
    args = parser.parse_args()
    source_train = ROOT / (
        f"runs/rejection_boundary_epoch_control_s{args.config_seed}_e4_"
        "soft_nnpu_audit_w0p1/train_student_config.json"
    )
    source_discover = ROOT / (
        f"runs/rejection_boundary_epoch_control_s{args.config_seed}_e4_"
        "soft_nnpu_audit_w0p1_discover/discover_config.json"
    )
    if not source_train.is_file() or not source_discover.is_file():
        raise FileNotFoundError("audited equal-epoch configs are required")
    for space in ("shared_boundary", "shared_boundary_soft_consensus"):
        run_arm(
            source_train,
            source_discover,
            int(args.seed),
            int(args.epochs),
            space,
            str(args.tag),
            float(args.soft_floor),
        )


if __name__ == "__main__":
    main()
