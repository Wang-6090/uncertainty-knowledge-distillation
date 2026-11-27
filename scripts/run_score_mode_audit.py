"""Audit detector configurations on one fixed student checkpoint.

Runs reuse the source discovery configuration, checkpoint, seed, pool, and
known-coverage calibration unless an explicit detector override is supplied.
This supports score-family and rejector candidate-selection diagnostics;
student weights remain frozen. Test labels are not used by the runner for
calibration or model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


# Keep this list aligned with novel_discovery.pipeline.build_rejector_features.
# The audit is intended to compare valid representations under one protocol;
# accepting an arbitrary string only postpones the error until after data
# extraction has already started.
VALID_REJECTOR_FEATURE_MODES = (
    "embedding",
    "rejection_embedding",
    "augmented",
    "support_augmented",
    "uncertainty_augmented",
    "support_uncertainty_augmented",
    "rejection_uncertainty_augmented",
    "rejection_support_uncertainty_augmented",
    "rejection_support_augmented",
    "rejection_balanced_support_augmented",
    "rejection_logit_augmented",
    "rejection_support_logit_augmented",
    "prototype_augmented",
    "prototype_support_augmented",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-discover", type=Path, required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument(
        "--score-modes",
        nargs="+",
        default=[
            "feature_rejector",
            "known_support",
            "known_conformal",
            "normalized_entropy_proto",
            "feature_rejector_fusion",
        ],
    )
    parser.add_argument(
        "--fusion-base-score",
        default="normalized_entropy_proto",
    )
    parser.add_argument("--fusion-rejector-weight", type=float, default=0.5)
    parser.add_argument(
        "--rejector-candidate-mode",
        default=None,
        choices=[
            "entropy",
            "max_softmax",
            "energy",
            "entropy_uncertainty",
            "consensus",
            "prototype_boundary",
            "confident_boundary",
        ],
        help=(
            "Optional override for rejector candidate selection. Use the same "
            "score mode and checkpoint to isolate this selector change."
        ),
    )
    parser.add_argument(
        "--rejector-pool-selection",
        default=None,
        choices=["all", "high_risk", "weighted_boundary"],
        help=(
            "Optional override for how candidate mining affects rejector "
            "training. Candidate-mode overrides are inert when this is 'all'."
        ),
    )
    parser.add_argument(
        "--rejector-feature-mode",
        default=None,
        choices=VALID_REJECTOR_FEATURE_MODES,
        help=(
            "Optional override for the frozen rejector input representation. "
            "Use this to audit one representation change while keeping the "
            "checkpoint and discovery protocol fixed."
        ),
    )
    parser.add_argument(
        "--rejector-mc-samples",
        type=int,
        default=None,
        help=(
            "Optional override for MC-dropout samples used when the selected "
            "rejector representation consumes epistemic uncertainty."
        ),
    )
    parser.add_argument(
        "--rejector-max-samples",
        type=int,
        default=None,
        help=(
            "Optional override for the maximum known/mixed support samples "
            "used by the rejector. Use 0 for all available samples."
        ),
    )
    parser.add_argument(
        "--threshold-policy",
        choices=["known_coverage", "class_conditional"],
        default="known_coverage",
        help="Threshold policy used for the post-hoc score audit.",
    )
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()

    source = json.loads(args.source_discover.read_text(encoding="utf-8"))
    for mode in args.score_modes:
        work_dir = ROOT / "runs" / f"{args.output_prefix}_{mode}"
        report = work_dir / "discovery_report.json"
        if report.is_file():
            print(f"reuse {report}")
            continue
        run_args = parse_args(["discover"])
        for key, value in source.items():
            if hasattr(run_args, key):
                setattr(run_args, key, value)
        run_args.student_ckpt = str(args.student_ckpt)
        run_args.work_dir = str(work_dir)
        run_args.score_mode = mode
        run_args.auto_calibrate_score = False
        run_args.skip_clustering = True
        run_args.fusion_base_score = args.fusion_base_score
        run_args.fusion_rejector_weight = args.fusion_rejector_weight
        if args.rejector_candidate_mode is not None:
            run_args.rejector_candidate_mode = args.rejector_candidate_mode
        if args.rejector_pool_selection is not None:
            run_args.rejector_pool_selection = args.rejector_pool_selection
        run_args.threshold_policy = args.threshold_policy
        run_args.threshold_percentile = args.threshold_percentile
        if args.rejector_feature_mode is not None:
            run_args.rejector_feature_mode = args.rejector_feature_mode
        if args.rejector_mc_samples is not None:
            if args.rejector_mc_samples < 1:
                raise ValueError("--rejector-mc-samples must be at least 1")
            run_args.rejector_mc_samples = args.rejector_mc_samples
        if args.rejector_max_samples is not None:
            if args.rejector_max_samples < 0:
                raise ValueError("--rejector-max-samples must be non-negative")
            run_args.rejector_max_samples = args.rejector_max_samples
        # The parser validates rejector-specific options even for support or
        # prototype-only scores.  Make that irrelevant inherited option
        # explicit so the score-mode comparison remains single-factor.
        if mode not in {"feature_rejector", "feature_rejector_fusion", "virtual_rejector"}:
            run_args.rejector_feature_mode = "embedding"
        # Some explicit score modes are only defined when their auxiliary
        # statistics are actually collected.  Keep this audit single-factor:
        # derive the minimum required wiring from the score mode instead of
        # silently dropping the mode or asking callers to duplicate parser
        # internals.  The source configuration remains unchanged for all
        # unrelated options.
        if "_knn" in mode:
            run_args.knn_ood = True
        discover(run_args)


if __name__ == "__main__":
    main()
