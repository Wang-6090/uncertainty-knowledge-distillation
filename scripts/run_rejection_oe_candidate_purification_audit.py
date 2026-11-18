"""Evaluation-only candidate-pool purification audit for seed 2120.

The student checkpoint, rejection score, threshold, and clustering feature are
fixed.  Only the label-free ranking used to remove the lowest-confidence
rejected candidates before clustering changes.  ``keep_ratio=0.8`` is fixed
before reading final test metrics, matching the nominal 0.2 known prior of the
mixed discovery pool.  Test and mixed-pool labels are used only for
retrospective metrics emitted by ``discover``.
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
KEEP_RATIO = 0.8
# The frozen fusion discovery configuration does not build a labeled kNN bank,
# so the support-based purifier is intentionally excluded from this audit.
# It is a separate opt-in protocol and must not be silently treated as a
# comparable result when its required support bank is absent.
MODES = (
    "none",
    "open_score",
    "entropy",
    "head_uncertainty",
    "head_uncertainty_classwise",
    "uncertainty_consensus",
)

from train import discover, parse_args  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(mode: str) -> dict[str, Any]:
    suffix = "none" if mode == "none" else f"{mode}_r{str(KEEP_RATIO).replace('.', '')}"
    if mode == "head_uncertainty_classwise":
        suffix = "head_uncertainty_classwise_r08_v2"
    output = ROOT / "runs" / f"rejection_oe_candidate_purify_s2120_{suffix}"
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
        args.score_mode = "feature_rejector_uncertainty_fusion"
        args.cluster_feature = "feature_pca"
        args.cluster_selection = "silhouette"
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.candidate_purify = mode
        args.candidate_keep_ratio = KEEP_RATIO if mode != "none" else 1.0
        print(f"[discover] candidate_purify={mode} keep_ratio={args.candidate_keep_ratio}", flush=True)
        discover(args)
    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("candidate_purify") != mode:
        raise RuntimeError(f"candidate purifier did not materialize: {output}")
    expected_ratio = KEEP_RATIO if mode != "none" else 1.0
    if abs(float(config.get("candidate_keep_ratio", 1.0)) - expected_ratio) > 1e-12:
        raise RuntimeError(f"candidate keep ratio changed unexpectedly: {output}")
    if config.get("score_mode") != "feature_rejector_uncertainty_fusion":
        raise RuntimeError(f"score mode changed unexpectedly: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    results = {mode: run(mode) for mode in MODES}
    output = ROOT / "analysis" / "rejection_oe_candidate_purification_s2120.json"
    payload = {
        "protocol": "seed-2120 frozen treatment label-free candidate-pool purification audit",
        "seed": 2120,
        "training_performed": False,
        "score_mode": "feature_rejector_uncertainty_fusion",
        "cluster_feature": "feature_pca",
        "cluster_selection": "silhouette",
        "keep_ratio": KEEP_RATIO,
        "keep_ratio_basis": "nominal mixed-pool known prior 0.2, fixed before final metrics",
        "modes": list(MODES),
        "criterion_selected_using_test_or_mixed_labels": False,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
