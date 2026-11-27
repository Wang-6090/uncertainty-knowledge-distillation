"""Run a short paired audit of the shared-encoder kNN-boundary gradients.

The experiment is intentionally training-only: it does not claim an open-set
improvement.  It records whether the pure and soft-NNPU kNN boundary terms
reach the shared encoder, and whether their directions conflict with CE/KD.
All recipe and data settings are inherited from one completed control arm;
the only algorithmic variable is ``discovery_knn_risk``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import fit_student  # noqa: E402
from scripts.run_knn_pu_boundary_control import load_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--source-train",
        type=Path,
        default=ROOT
        / "runs/knn_pu_boundary_control_s44_e4_soft_added_pure/train_student_config.json",
    )
    parser.add_argument("--tag", default="knn_gradient_audit_s44_e1")
    args = parser.parse_args()
    source = args.source_train if args.source_train.is_absolute() else ROOT / args.source_train
    if not source.is_file():
        raise FileNotFoundError(source)

    for risk in ("pure", "soft_nnpu"):
        work_dir = ROOT / "runs" / f"{args.tag}_{risk}"
        checkpoint = work_dir / "student.pt"
        history_path = work_dir / "train_history.json"
        if not history_path.is_file():
            train_args = load_args(
                source,
                "train_student",
                work_dir,
                int(args.seed),
                int(args.epochs),
            )
            train_args.student_ckpt = str(checkpoint)
            train_args.alpha_discovery_knn_boundary = 0.1
            train_args.alpha_discovery_rejection_knn_boundary = 0.0
            train_args.alpha_discovery_rejection_boundary = 0.0
            train_args.alpha_discovery_boundary = 0.0
            train_args.discovery_boundary_mode = "pu"
            train_args.discovery_knn_risk = risk
            train_args.discovery_knn_known_prior = 0.2
            train_args.discovery_knn_softness = 0.05
            train_args.discovery_feature_candidate_gating = True
            train_args.discovery_cross_view_gating = True
            train_args.discovery_feature_candidate_space = "shared_boundary"
            train_args.audit_loss_gradients = True
            fit_student(train_args)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        last = history[-1]
        train_stats = last.get("train", last)
        selected = {
            key: value
            for key, value in train_stats.items()
            if key.startswith("gradient_audit_")
            or key in {"discovery_knn_boundary", "feature_candidate_ratio"}
        }
        print(json.dumps({"risk": risk, "epoch": last.get("epoch"), **selected}, indent=2))


if __name__ == "__main__":
    main()
