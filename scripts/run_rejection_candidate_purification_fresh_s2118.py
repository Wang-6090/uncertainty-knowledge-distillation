"""Fresh fixed-rule replication of candidate-pool purification on seed 2118.

This is an evaluation-only replication on an already trained, split-matched
fusion checkpoint.  The only changed factor is the clustering-only
``head_uncertainty`` purifier with the ratio fixed to 0.8 before final metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 2118
SOURCE = ROOT / "runs" / "rejection_full_pipeline_factorial_s2118_fusion_feature_pca"
CHECKPOINT = ROOT / "runs" / "rejection_full_pipeline_factorial_s2118_student" / "student.pt"
SPLIT = ROOT / "splits/semantic_hard/seed2118" / "cifar100_60_40_semantic_hard.json"
KEEP_RATIO = 0.8
MODES = ("none", "head_uncertainty", "head_uncertainty_classwise")

from train import discover, parse_args  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(mode: str) -> dict[str, Any]:
    suffix = {
        "none": "none",
        "head_uncertainty": "head_uncertainty_r08",
        "head_uncertainty_classwise": "head_uncertainty_classwise_r08_v2",
    }[mode]
    output = ROOT / "runs" / f"rejection_candidate_purification_s{SEED}_{suffix}"
    report_path = output / "discovery_report.json"
    if not report_path.is_file():
        if output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output}")
        args = parse_args(["discover"])
        for key, value in read_json(SOURCE / "discover_config.json").items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.seed = SEED
        args.model_seed = None
        args.work_dir = str(output)
        args.student_ckpt = str(CHECKPOINT)
        args.split_path = str(SPLIT)
        args.score_mode = "feature_rejector_uncertainty_fusion"
        args.cluster_feature = "feature_pca"
        args.cluster_selection = "silhouette"
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.candidate_purify = mode
        args.candidate_keep_ratio = (
            KEEP_RATIO if mode in {"head_uncertainty", "head_uncertainty_classwise"} else 1.0
        )
        print(f"[discover] seed={SEED} candidate_purify={mode}", flush=True)
        discover(args)
    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("candidate_purify") != mode:
        raise RuntimeError(f"candidate purifier did not materialize: {output}")
    if config.get("score_mode") != "feature_rejector_uncertainty_fusion":
        raise RuntimeError(f"score mode changed unexpectedly: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    results = {mode: run(mode) for mode in MODES}
    output = ROOT / "analysis" / f"rejection_candidate_purification_fresh_s{SEED}.json"
    payload = {
        "protocol": f"seed-{SEED} fresh fixed-rule candidate-pool purification replication",
        "seed": SEED,
        "training_performed": False,
        "score_mode": "feature_rejector_uncertainty_fusion",
        "cluster_feature": "feature_pca",
        "cluster_selection": "silhouette",
        "keep_ratio": KEEP_RATIO,
        "keep_ratio_basis": "fixed from prior seed before fresh final metrics",
        "modes": list(MODES),
        "criterion_selected_using_test_or_mixed_labels": False,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
