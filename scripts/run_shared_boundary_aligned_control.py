"""Aligned paired control for shared and boundary candidate gating.

This control fixes a protocol mismatch in the earlier candidate experiment:
the candidate gate is computed in the shared classification representation, so
the loss must also be the shared-feature kNN boundary loss.  Both arms below
therefore enable ``alpha_discovery_knn_boundary`` and disable the independent
rejection-branch boundary loss.  All other training, calibration, and
discovery settings are copied from the audited equal-epoch configuration.
"""

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
    args.seed = seed
    args.model_seed = None
    args.epochs = epochs
    args.work_dir = str(work_dir)
    return args


def run_arm(
    source_train: Path,
    source_discover: Path,
    seed: int,
    epochs: int,
    space: str,
    tag: str,
) -> None:
    work_dir = ROOT / "runs" / f"{tag}_{space}"
    discover_dir = ROOT / "runs" / f"{tag}_{space}_discover"
    student_ckpt = work_dir / "student.pt"

    if not student_ckpt.is_file():
        train_args = load_args(source_train, "train_student", work_dir, seed, epochs)
        train_args.student_ckpt = str(student_ckpt)
        # The candidate gate and the boundary loss intentionally use the same
        # shared representation in this protocol.
        train_args.alpha_discovery_knn_boundary = 0.1
        train_args.alpha_discovery_rejection_knn_boundary = 0.0
        train_args.alpha_discovery_rejection_boundary = 0.0
        train_args.alpha_discovery_rejection_pu_feature_margin = 0.0
        train_args.discovery_boundary_mode = "pu"
        train_args.discovery_boundary_pu_risk = "hard_nnpu"
        train_args.discovery_feature_candidate_gating = True
        train_args.discovery_cross_view_gating = True
        train_args.discovery_feature_candidate_space = space
        fit_student(train_args)

    report = discover_dir / "discovery_report.json"
    if not report.is_file():
        discover_args = load_args(
            source_discover, "discover", discover_dir, seed, epochs
        )
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
    parser.add_argument(
        "--tag",
        default="shared_boundary_aligned_control_s44_e4",
        help="Output prefix under runs/.",
    )
    args = parser.parse_args()

    source_train = (
        ROOT
        / f"runs/rejection_boundary_epoch_control_s{args.config_seed}_e4_soft_nnpu_audit_w0p1/train_student_config.json"
    )
    source_discover = (
        ROOT
        / f"runs/rejection_boundary_epoch_control_s{args.config_seed}_e4_soft_nnpu_audit_w0p1_discover/discover_config.json"
    )
    if not source_train.is_file() or not source_discover.is_file():
        raise FileNotFoundError(
            "The audited equal-epoch rejection-boundary configs are required"
        )

    for space in ("shared", "shared_boundary", "shared_boundary_consensus"):
        run_arm(
            source_train,
            source_discover,
            int(args.seed),
            int(args.epochs),
            space,
            str(args.tag),
        )


if __name__ == "__main__":
    main()
