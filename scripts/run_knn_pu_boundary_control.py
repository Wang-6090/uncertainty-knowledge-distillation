"""Run a matched pure-kNN versus PU-kNN boundary-loss control."""

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
    risk: str,
    known_prior: float,
    tag: str,
):
    work_dir = ROOT / "runs" / f"{tag}_{risk}"
    discover_dir = ROOT / "runs" / f"{tag}_{risk}_discover"
    student_ckpt = work_dir / "student.pt"
    if not student_ckpt.is_file():
        train_args = load_args(source_train, "train_student", work_dir, seed, epochs)
        train_args.student_ckpt = str(student_ckpt)
        train_args.alpha_discovery_knn_boundary = 0.1
        train_args.alpha_discovery_rejection_knn_boundary = 0.0
        # Keep this control focused on the kNN boundary risk.  The source
        # recipe may come from an older rejection-boundary experiment where
        # the independent mixed-pool rejection term was enabled; leaving it
        # untouched would confound pure-vs-NNPU kNN risk with that term.
        train_args.alpha_discovery_rejection_boundary = 0.0
        train_args.alpha_discovery_boundary = 0.0
        train_args.discovery_boundary_mode = "pu"
        train_args.discovery_knn_risk = risk
        train_args.discovery_knn_known_prior = float(known_prior)
        train_args.discovery_knn_softness = 0.05
        train_args.discovery_feature_candidate_gating = True
        train_args.discovery_cross_view_gating = True
        train_args.discovery_feature_candidate_space = "shared_boundary"
        fit_student(train_args)
    if not (discover_dir / "discovery_report.json").is_file():
        discover_args = load_args(source_discover, "discover", discover_dir, seed, epochs)
        discover_args.student_ckpt = str(student_ckpt)
        discover_args.discovery_feature_candidate_gating = True
        discover_args.discovery_cross_view_gating = True
        discover_args.discovery_feature_candidate_space = "shared_boundary"
        # Discover does not train this loss, but keeping the recipe explicit
        # makes the saved configuration auditable and prevents accidental
        # reuse of the historical rejection-boundary setting.
        discover_args.alpha_discovery_rejection_boundary = 0.0
        discover_args.alpha_discovery_rejection_knn_boundary = 0.0
        discover_args.alpha_discovery_knn_boundary = 0.1
        discover_args.alpha_discovery_boundary = 0.0
        discover(discover_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--config-seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--known-prior", type=float, default=0.2)
    parser.add_argument("--tag", default="knn_pu_boundary_control_s44_e4")
    parser.add_argument(
        "--source-train",
        type=Path,
        default=None,
        help="Optional matched train_student_config.json used as the immutable source recipe.",
    )
    parser.add_argument(
        "--source-discover",
        type=Path,
        default=None,
        help="Optional matched discover_config.json used as the immutable source recipe.",
    )
    args = parser.parse_args()
    source_train = args.source_train or (
        ROOT / (
            f"runs/rejection_boundary_epoch_control_s{args.config_seed}_e4_"
            "soft_nnpu_audit_w0p1/train_student_config.json"
        )
    )
    source_discover = args.source_discover or (
        ROOT / (
            f"runs/rejection_boundary_epoch_control_s{args.config_seed}_e4_"
            "soft_nnpu_audit_w0p1_discover/discover_config.json"
        )
    )
    source_train = source_train if source_train.is_absolute() else ROOT / source_train
    source_discover = source_discover if source_discover.is_absolute() else ROOT / source_discover
    if not source_train.is_file() or not source_discover.is_file():
        raise FileNotFoundError("audited equal-epoch configs are required")
    # Keep the historical pure arm and hard-NNPU arm, but also expose the
    # smooth non-negative correction.  The latter is a diagnostic control for
    # the known-risk subtraction: hard clamping can make the mixed-pool
    # boundary gradient exactly zero even when unknown samples remain inside
    # the support.  All three arms share the same recipe and differ only in
    # ``discovery_knn_risk``.
    for risk in ("pure", "hard_nnpu", "soft_nnpu"):
        run_arm(
            source_train,
            source_discover,
            int(args.seed),
            int(args.epochs),
            risk,
            float(args.known_prior),
            str(args.tag),
        )


if __name__ == "__main__":
    main()
