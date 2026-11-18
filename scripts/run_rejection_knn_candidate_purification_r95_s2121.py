"""Evaluation-only known-neighbor support purification on seed 2121."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 2121
KEEP_RATIO = 0.95
SPLIT = ROOT / "splits/semantic_hard/seed2121/cifar100_60_40_semantic_hard.json"
ARMS = {
    "static_control": ROOT / "runs/rejection_ce_decay_full_s2121_static_control",
    "late_decay": ROOT / "runs/rejection_ce_decay_full_s2121_late_decay",
}

from train import discover, parse_args  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(arm: str, purifier: str) -> dict[str, Any]:
    source = ARMS[arm]
    source_config = read_json(source.with_name(source.name + "_discover") / "discover_config.json")
    suffix = "" if purifier == "knn_support" else "_none"
    output = ROOT / "runs" / f"rejection_knn_candidate_purification_r95_s{SEED}_{arm}{suffix}"
    report_path = output / "discovery_report.json"
    if not report_path.is_file():
        if output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output}")
        args = parse_args(["discover"])
        for key, value in source_config.items():
            if key != "command" and hasattr(args, key):
                setattr(args, key, value)
        args.seed = SEED
        args.model_seed = None
        args.work_dir = str(output)
        args.student_ckpt = str(source / "student.pt")
        args.split_path = str(SPLIT)
        args.score_mode = "feature_rejector"
        args.cluster_feature = "feature_pca"
        args.cluster_selection = "silhouette"
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.knn_ood = True
        args.knn_feature = "features"
        args.knn_k = 10
        args.knn_support_quantile = 0.95
        args.candidate_purify = purifier
        args.candidate_keep_ratio = KEEP_RATIO
        print(f"[discover] arm={arm} purifier={purifier} keep_ratio={KEEP_RATIO:.2f}", flush=True)
        discover(args)
    config_path = output / "discover_config.json"
    if not config_path.is_file():
        config_path = output / "config.json"
    config = read_json(config_path)
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("candidate_purify") != purifier:
        raise RuntimeError(f"candidate purifier did not materialize: {output}")
    if abs(float(config.get("candidate_keep_ratio", -1.0)) - KEEP_RATIO) > 1e-6:
        raise RuntimeError(f"keep ratio did not materialize: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    results = {
        arm: {purifier: run(arm, purifier) for purifier in ("none", "knn_support")}
        for arm in ARMS
    }
    output = ROOT / "analysis" / "rejection_knn_candidate_purification_r95_s2121.json"
    payload = {
        "protocol": "seed-2121 evaluation-only known-neighbor support candidate purification at fixed 95% retention",
        "seed": SEED,
        "training_performed": False,
        "score_mode": "feature_rejector",
        "cluster_feature": "feature_pca",
        "cluster_selection": "silhouette",
        "knn_ood": True,
        "knn_feature": "features",
        "knn_k": 10,
        "knn_support_quantile": 0.95,
        "candidate_purify": "knn_support",
        "candidate_keep_ratio": KEEP_RATIO,
        "criterion_selected_using_test_or_mixed_labels": False,
        "arms": list(ARMS),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
