"""Evaluation-only uncertainty-fusion audit for the seed-2120 OE pair.

The student checkpoints are fixed.  The audit compares the existing
``feature_rejector`` score with the pre-registered fixed 0.5/0.5
validation-calibrated uncertainty fusion.  No training or weight search is
performed; mixed/test labels are used only for retrospective metrics.
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


ARMS = ("margin_only", "margin_plus_uniform")
SCORES = {"rejector": "feature_rejector", "fusion": "feature_rejector_uncertainty_fusion"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(arm: str, score_name: str) -> dict[str, Any]:
    source_dir = ROOT / "runs" / f"rejection_oe_semantic_hard_s2120_{arm}_discover"
    config_path = source_dir / "discover_config.json"
    checkpoint = source_dir.parent / f"rejection_oe_semantic_hard_s2120_{arm}" / "student.pt"
    split_path = ROOT / "splits/semantic_hard/seed2120" / "cifar100_60_40_semantic_hard.json"
    for path in (config_path, checkpoint, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = ROOT / "runs" / f"rejection_oe_uncertainty_fusion_s2120_{arm}_{score_name}"
    report_path = output / "discovery_report.json"
    if not report_path.is_file():
        if output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output}")
        args = parse_args(["discover"])
        for key, value in read_json(config_path).items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.seed = 2120
        args.model_seed = None
        args.work_dir = str(output)
        args.student_ckpt = str(checkpoint)
        args.split_path = str(split_path)
        args.score_mode = SCORES[score_name]
        args.cluster_feature = "feature_pca"
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        print(f"[discover] arm={arm} score={score_name}", flush=True)
        discover(args)
    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("score_mode") != SCORES[score_name]:
        raise RuntimeError(f"score mode did not materialize: {output}")
    if config.get("cluster_feature") != "feature_pca":
        raise RuntimeError(f"cluster feature did not materialize: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {score_name: run(arm, score_name) for score_name in SCORES}
    output = ROOT / "analysis" / "rejection_oe_uncertainty_fusion_audit_s2120.json"
    payload = {
        "protocol": "seed-2120 frozen-checkpoint fixed uncertainty fusion audit",
        "seed": 2120,
        "training_performed": False,
        "fusion_rule": "0.5 * z(feature_rejector) + 0.5 * z(validation-calibrated uncertainty-error)",
        "fusion_weight_tuned": False,
        "cluster_feature": "feature_pca",
        "threshold_policy": "known-validation-only 95% coverage",
        "test_or_mixed_pool_labels_used_for_decision": False,
        "arms": result,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

