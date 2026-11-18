"""Fixed-count candidate-rank control for the uncertainty-aware rejection arm."""

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
MODE = "rejection_support_uncertainty_augmented"

from train import discover, parse_args  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    output = ROOT / "runs" / "rejection_feature_rank_s2120_rejection_support_uncertainty"
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
        args.rejector_feature_mode = MODE
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
        args.cluster_candidate_mode = "feature_rejector_rank"
        args.candidate_purify = "none"
        args.candidate_keep_ratio = 1.0
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.skip_clustering = False
        print(f"[discover] mode={MODE}, cluster_candidate_mode=feature_rejector_rank", flush=True)
        discover(args)
    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("rejector_feature_mode") != MODE:
        raise RuntimeError("feature mode did not materialize")
    if config.get("cluster_candidate_mode") != "feature_rejector_rank":
        raise RuntimeError("candidate rank control did not materialize")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError("label-flow audit failed")
    payload = {
        "protocol": "seed-2120 fixed-count candidate-rank control",
        "seed": 2120,
        "training_performed": False,
        "student_checkpoint": str(CHECKPOINT),
        "feature_mode": MODE,
        "cluster_candidate_mode": "feature_rejector_rank",
        "criterion_selected_using_test_or_mixed_labels": False,
        "result": {"output": str(output), "config": config, "metrics": report, "calibration": calibration},
    }
    target = ROOT / "analysis" / "rejection_feature_rank_s2120.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(target)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
