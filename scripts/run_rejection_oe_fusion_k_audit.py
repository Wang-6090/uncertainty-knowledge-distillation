"""Evaluation-only Auto-K criterion audit for the seed-2120 fusion pair.

The frozen OE checkpoints and fixed validation-calibrated uncertainty fusion
are unchanged.  Only the existing unsupervised ``cluster_selection``
criterion changes.  Test and mixed-pool labels are used only for retrospective
metrics, never for criterion selection.
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
CRITERIA = (
    "silhouette",
    "composite",
    "stability",
    "bootstrap_stability",
    "gmm_bic",
)

from train import discover, parse_args  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(criterion: str) -> dict[str, Any]:
    output = ROOT / "runs" / f"rejection_oe_fusion_k_s2120_{criterion}"
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
        args.cluster_selection = criterion
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        print(f"[discover] criterion={criterion}", flush=True)
        discover(args)
    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("cluster_selection") != criterion:
        raise RuntimeError(f"criterion did not materialize: {output}")
    if config.get("score_mode") != "feature_rejector_uncertainty_fusion":
        raise RuntimeError(f"score mode changed unexpectedly: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    results = {criterion: run(criterion) for criterion in CRITERIA}
    output = ROOT / "analysis" / "rejection_oe_fusion_k_audit_s2120.json"
    payload = {
        "protocol": "seed-2120 frozen treatment fixed-fusion Auto-K criterion audit",
        "seed": 2120,
        "training_performed": False,
        "score_mode": "feature_rejector_uncertainty_fusion",
        "cluster_feature": "feature_pca",
        "criteria": list(CRITERIA),
        "criterion_selected_using_test_or_mixed_labels": False,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

