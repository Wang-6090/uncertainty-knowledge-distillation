"""Frozen-checkpoint paired clustering-feature audit for epoch-budget runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


SEEDS = (2107, 2108, 2109)
EPOCH_ARMS = ("four_epoch", "eight_epoch")
CLUSTER_FEATURES = {"projection": "projection_pca", "backbone": "feature_pca"}


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


def run(seed: int, epoch_arm: str, cluster_name: str) -> dict[str, Any]:
    stem = ROOT / "runs" / f"rejection_epoch_budget_s{seed}_{epoch_arm}"
    checkpoint = stem / "student.pt"
    source_path = ROOT / "runs" / f"rejection_epoch_budget_s{seed}_{epoch_arm}_discover" / "discover_config.json"
    split = ROOT / f"splits_cifar100_semantic_hard_seed{seed}" / "cifar100_60_40_semantic_hard.json"
    for path in (checkpoint, source_path, split):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = read_json(source_path)
    output = ROOT / "runs" / f"rejection_epoch_cluster_feature_s{seed}_{epoch_arm}_{cluster_name}"
    if not (output / "discovery_report.json").is_file():
        if output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output}")
        args = load_args(source, seed, output, checkpoint)
        args.split_path = str(split)
        args.cluster_feature = CLUSTER_FEATURES[cluster_name]
        print(f"[discover] seed={seed} epoch={epoch_arm} cluster={cluster_name}", flush=True)
        discover(args)
    config = read_json(output / "discover_config.json")
    report = read_json(output / "discovery_report.json")
    calibration = read_json(output / "calibration_report.json")
    if config.get("cluster_feature") != CLUSTER_FEATURES[cluster_name]:
        raise RuntimeError(f"cluster feature did not materialize: {output}")
    if calibration.get("rejector_protocol", {}).get("test_labels_used_for_training_or_calibration") is not False:
        raise RuntimeError(f"label-flow audit failed for {output}")
    if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
        raise RuntimeError(f"known-validation calibration out of tolerance: {output}")
    return {
        "output": str(output),
        "config": config,
        "metrics": report,
        "calibration": calibration,
    }


def main() -> None:
    results: dict[str, Any] = {}
    for seed in SEEDS:
        results[str(seed)] = {}
        for arm in EPOCH_ARMS:
            results[str(seed)][arm] = {
                name: run(seed, arm, name) for name in CLUSTER_FEATURES
            }
            control = results[str(seed)][arm]["projection"]["config"]
            treatment = results[str(seed)][arm]["backbone"]["config"]
            allowed = {"cluster_feature", "work_dir"}
            differences = {
                key: (control.get(key), treatment.get(key))
                for key in sorted(set(control) | set(treatment))
                if control.get(key) != treatment.get(key)
            }
            unexpected = {key: val for key, val in differences.items() if key not in allowed}
            if unexpected:
                raise RuntimeError(f"unexpected discovery differences seed={seed} arm={arm}: {unexpected}")
            results[str(seed)][arm]["config_differences"] = differences

    summary = {
        "protocol": "frozen-checkpoint projection_pca vs feature_pca clustering audit",
        "seeds": list(SEEDS),
        "training_performed": False,
        "test_labels_used_only_for_retrospective_metrics": True,
        "single_changed_factor": "cluster_feature",
        "arms": results,
    }
    output = ROOT / "analysis" / "rejection_epoch_cluster_feature_audit_s2107_s2109.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
