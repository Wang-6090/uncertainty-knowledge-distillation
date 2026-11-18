"""Frozen-checkpoint audit of uncertainty-augmented rejector features.

The student checkpoint and all downstream discovery settings are fixed. The
only post-hoc rejector factor is the feature bundle used by the mixed-pool
nnPU rejector: support-augmented baseline versus uncertainty-augmented
features with four deterministic/stochastic extraction passes as configured.
No training labels from the mixed pool or test set are used for fitting,
thresholding, or method selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "runs" / "rejection_oe_semantic_hard_s2120_margin_plus_uniform_discover"
CHECKPOINT = ROOT / "runs" / "rejection_oe_semantic_hard_s2120_margin_plus_uniform" / "student.pt"
SPLIT = ROOT / "splits/semantic_hard/seed2120" / "cifar100_60_40_semantic_hard.json"
MODES = ("support_augmented", "uncertainty_augmented")

from train import discover, parse_args  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(feature_mode: str) -> dict[str, Any]:
    output = ROOT / "runs" / f"rejection_feature_mode_s2120_{feature_mode}"
    report_path = output / "discovery_report.json"
    if not report_path.is_file():
        if output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output}")
        args = parse_args(["discover"])
        for key, value in read_json(SOURCE / "discover_config.json").items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.seed = 2120
        args.model_seed = None
        args.work_dir = str(output)
        args.student_ckpt = str(CHECKPOINT)
        args.split_path = str(SPLIT)
        args.score_mode = "feature_rejector"
        args.rejector_feature_mode = feature_mode
        args.rejector_mc_samples = 4
        args.rejector_training = "nnpu"
        args.rejector_nnpu_risk = "nu_corrected"
        args.rejector_nnpu_model = "linear"
        args.rejector_known_prior = 0.2
        args.rejector_pool_selection = "all"
        args.rejector_candidate_mode = "entropy_uncertainty"
        args.cluster_feature = "feature_pca"
        args.cluster_selection = "silhouette"
        args.cluster_k = "auto"
        args.cluster_candidate_mode = "detector"
        args.candidate_purify = "none"
        args.candidate_keep_ratio = 1.0
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.skip_clustering = False
        print(f"[discover] rejector_feature_mode={feature_mode}", flush=True)
        discover(args)
    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("rejector_feature_mode") != feature_mode:
        raise RuntimeError(f"feature mode did not materialize: {output}")
    if config.get("score_mode") != "feature_rejector":
        raise RuntimeError(f"score mode changed unexpectedly: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    results = {mode: run(mode) for mode in MODES}
    output = ROOT / "analysis" / "rejection_feature_mode_s2120.json"
    payload = {
        "protocol": "seed-2120 frozen rejector feature-mode audit",
        "seed": 2120,
        "training_performed": False,
        "student_checkpoint": str(CHECKPOINT),
        "score_mode": "feature_rejector",
        "rejector_training": "nnpu",
        "rejector_nnpu_risk": "nu_corrected",
        "rejector_known_prior": 0.2,
        "rejector_mc_samples": 4,
        "cluster_feature": "feature_pca",
        "cluster_selection": "silhouette",
        "criterion_selected_using_test_or_mixed_labels": False,
        "single_changed_factor": "rejector_feature_mode",
        "modes": list(MODES),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
