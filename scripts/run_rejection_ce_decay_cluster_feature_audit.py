"""Frozen-checkpoint audit of the clustering representation after CE decay.

The audit reuses the four completed static/late-decay student checkpoints and
changes only the representation consumed by the clustering stage:
``feature_pca`` versus ``projection_pca``. Training, rejection scoring,
candidate selection, calibration, K selection, and KMeans settings are kept
from the completed discovery configuration. No new split or checkpoint is
created.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


SEEDS = (2121, 2124, 2125, 2127)
ARMS = ("static_control", "late_decay")
FEATURES = {"feature": "feature_pca", "projection": "projection_pca"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_args(config: dict[str, Any], work_dir: Path, student_ckpt: Path, split_path: Path):
    args = parse_args(["discover"])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(config.get("seed", 0))
    args.model_seed = None
    args.work_dir = str(work_dir)
    args.student_ckpt = str(student_ckpt)
    args.split_path = str(split_path)
    args.cluster_k = "auto"
    args.skip_clustering = False
    return args


def run_one(seed: int, arm: str, name: str) -> dict[str, Any]:
    train_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_{arm}"
    source_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_{arm}_discover"
    source_config_path = source_dir / "discover_config.json"
    student_ckpt = train_dir / "student.pt"
    split_path = ROOT / f"splits/semantic_hard/seed{seed}/cifar100_60_40_semantic_hard.json"
    for path in (source_config_path, student_ckpt, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = ROOT / "runs" / f"rejection_ce_decay_cluster_feature_audit_s{seed}_{arm}_{name}"
    report_path = output / "discovery_report.json"
    if report_path.is_file():
        config = read_json(output / "discover_config.json")
        report = read_json(report_path)
        calibration = read_json(output / "calibration_report.json")
        return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}
    if output.exists():
        raise FileExistsError(f"refusing to reuse incomplete output: {output}")

    source_config = read_json(source_config_path)
    args = load_args(source_config, output, student_ckpt, split_path)
    args.cluster_feature = FEATURES[name]
    print(f"[discover] seed={seed} arm={arm} cluster_feature={FEATURES[name]}", flush=True)
    discover(args)

    config = read_json(output / "discover_config.json")
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("cluster_feature") != FEATURES[name]:
        raise RuntimeError(f"cluster feature did not materialize: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
        raise RuntimeError(f"known-validation calibration outside tolerance: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    options = parser.parse_args()

    results: dict[str, Any] = {}
    for seed in options.seeds:
        results[str(seed)] = {}
        for arm in ARMS:
            results[str(seed)][arm] = {
                name: run_one(seed, arm, name) for name in FEATURES
            }
            left = results[str(seed)][arm]["feature"]
            right = results[str(seed)][arm]["projection"]
            allowed = {"cluster_feature", "work_dir"}
            differences = {
                key: (left["config"].get(key), right["config"].get(key))
                for key in sorted(set(left["config"]) | set(right["config"]))
                if left["config"].get(key) != right["config"].get(key)
            }
            unexpected = {key: value for key, value in differences.items() if key not in allowed}
            if unexpected:
                raise RuntimeError(
                    f"unexpected condition differences seed={seed} arm={arm}: {unexpected}"
                )
            results[str(seed)][arm]["config_differences"] = differences

    output = ROOT / "analysis" / "rejection_ce_decay_cluster_feature_audit_4split.json"
    summary = {
        "protocol": "frozen late-decay checkpoint feature_pca versus projection_pca audit",
        "seeds": list(options.seeds),
        "arms": list(ARMS),
        "training_performed": False,
        "single_changed_factor": "cluster_feature",
        "test_labels_used_only_for_retrospective_metrics": True,
        "results": results,
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
