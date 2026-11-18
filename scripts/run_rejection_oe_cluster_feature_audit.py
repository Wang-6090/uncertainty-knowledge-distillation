"""Evaluation-only projection-PCA vs feature-PCA audit for the seed-2120 OE pair.

This script reuses the two already-trained seed-2120 checkpoints.  The only
discovery-time factor is ``cluster_feature``; score mode, rejector fitting,
known-validation calibration, candidate pool, and all model parameters are
held fixed.  Labels are not used for calibration or selection and are read
only by the final retrospective clustering metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


ARMS = {
    "margin_only": ROOT / "runs" / "rejection_oe_semantic_hard_s2120_margin_only",
    "margin_plus_uniform": ROOT / "runs" / "rejection_oe_semantic_hard_s2120_margin_plus_uniform",
}
CLUSTER_FEATURES = {"projection": "projection_pca", "feature": "feature_pca"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_args(config: dict[str, Any], seed: int, work_dir: Path, checkpoint: Path):
    args = parse_args(["discover"])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    args.student_ckpt = str(checkpoint)
    args.score_mode = "feature_rejector"
    args.threshold_policy = "known_coverage"
    args.target_known_coverage = 0.95
    args.open_val_ratio = 0.0
    args.cluster_k = "auto"
    args.skip_clustering = False
    return args


def run_arm(arm: str, source_dir: Path, feature_name: str) -> dict[str, Any]:
    config_path = source_dir / "discover_config.json"
    checkpoint = source_dir.parent / source_dir.name.removesuffix("_discover") / "student.pt"
    split_path = ROOT / "splits/semantic_hard/seed2120" / "cifar100_60_40_semantic_hard.json"
    for path in (config_path, checkpoint, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = ROOT / "runs" / f"rejection_oe_cluster_feature_s2120_{arm}_{feature_name}"
    report_path = output / "discovery_report.json"
    if not report_path.is_file():
        if output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output}")
        args = load_args(read_json(config_path), 2120, output, checkpoint)
        args.split_path = str(split_path)
        args.cluster_feature = CLUSTER_FEATURES[feature_name]
        print(f"[discover] arm={arm} cluster_feature={feature_name}", flush=True)
        discover(args)

    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("cluster_feature") != CLUSTER_FEATURES[feature_name]:
        raise RuntimeError(f"cluster feature did not materialize: {output}")
    if config.get("score_mode") != "feature_rejector":
        raise RuntimeError(f"score mode changed unexpectedly: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed for {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    results: dict[str, Any] = {}
    for arm, source_dir in ARMS.items():
        results[arm] = {
            feature_name: run_arm(
                arm,
                source_dir.parent / f"{source_dir.name}_discover",
                feature_name,
            )
            for feature_name in CLUSTER_FEATURES
        }
        projection = results[arm]["projection"]["config"]
        feature = results[arm]["feature"]["config"]
        differences = {
            key: (projection.get(key), feature.get(key))
            for key in sorted(set(projection) | set(feature))
            if projection.get(key) != feature.get(key)
        }
        unexpected = {key: value for key, value in differences.items() if key not in {"cluster_feature", "work_dir"}}
        if unexpected:
            raise RuntimeError(f"unexpected discovery differences for {arm}: {unexpected}")
        results[arm]["config_differences"] = differences

    output = ROOT / "analysis" / "rejection_oe_cluster_feature_audit_s2120.json"
    payload = {
        "protocol": "seed-2120 frozen-checkpoint projection_pca vs feature_pca audit",
        "seed": 2120,
        "training_performed": False,
        "single_changed_factor": "cluster_feature",
        "score_mode": "feature_rejector",
        "threshold_policy": "known-validation-only 95% coverage",
        "test_labels_used_only_for_retrospective_metrics": True,
        "mixed_pool_labels_used_only_for_retrospective_metrics": True,
        "arms": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

