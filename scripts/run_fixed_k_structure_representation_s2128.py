"""Diagnostic comparison of fixed K versus automatic K for two cluster features.

This script reuses the audited student checkpoint and discovery protocol from
the structure-representation pilot.  It runs two fresh discovery arms with
``cluster_k=40`` and changes only ``cluster_feature`` between the arms:

* ``feature_pca``: classification backbone feature;
* ``structure_pca``: supervised structure-head feature.

The fixed value 40 is the known number of held-out novel classes in this
semantic-hard 60/40 split.  It is used only as a diagnostic oracle and is not
treated as an deployable open-world setting.
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


SEED = 2128
BASE_TAG = "structure_representation_pilot_s2128"
SPLIT_PATH = ROOT / "splits/semantic_hard/seed2128/cifar100_60_40_semantic_hard.json"
STUDENT_CKPT = ROOT / "runs/structure_representation_pilot_s2128_student/student.pt"
SOURCE_CONFIG = ROOT / "runs/structure_representation_pilot_s2128_feature/discover_config.json"
RESULT_PATH = ROOT / "analysis/structure_representation_fixed_k_oracle40_s2128.json"
FIXED_K = 40


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "auroc",
        "unknown_reject_rate",
        "cluster_candidate_count",
        "cluster_candidate_true_unknown_count_before_purification",
        "cluster_candidate_false_reject_count_before_purification",
        "cluster_candidate_purity_before_purification",
        "cluster_candidate_unknown_recall",
        "cluster_unknown_only_acc",
        "cluster_unknown_only_nmi",
        "cluster_unknown_only_ari",
        "cluster_candidate_unknown_oracle_acc",
        "cluster_candidate_unknown_oracle_nmi",
        "cluster_candidate_unknown_oracle_ari",
        "cluster_k",
        "cluster_true_k",
        "cluster_k_abs_error",
        "cluster_feature",
    )
    return {key: report.get(key) for key in keys}


def load_args(work_dir: Path, cluster_feature: str):
    args = parse_args(["discover"])
    source = read_json(SOURCE_CONFIG)
    for key, value in source.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = SEED
    args.model_seed = None
    args.split_path = str(SPLIT_PATH)
    args.student_ckpt = str(STUDENT_CKPT)
    args.work_dir = str(work_dir)
    # These fields belong to the training configuration rather than the
    # ``discover`` subparser, but discovery must still know the structure-head
    # dimensionality when it reconstructs the student model.  The source
    # pilot explicitly supplied the same values before calling ``discover``.
    # Keep them explicit here so the fixed-K diagnostic uses the identical
    # checkpoint architecture instead of silently dropping the saved config.
    args.structure_feature_dim = 128
    args.alpha_structure_supcon = 0.1
    args.cluster_feature = cluster_feature
    # The discover interface supports only "auto" and "oracle". An integer
    # is truthy but is not treated as fixed K by run_discovery, so it silently
    # enters the automatic-selection branch. Oracle mode is diagnostic-only
    # here and fixes K through the configured 40 novel classes.
    args.cluster_k = "oracle"
    args.num_novel = FIXED_K
    args.skip_clustering = False
    args.score_mode = "feature_rejector"
    args.threshold_policy = "known_coverage"
    args.target_known_coverage = 0.95
    args.open_val_ratio = 0.0
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in (SPLIT_PATH, STUDENT_CKPT, SOURCE_CONFIG):
        assert_file(path)
    if RESULT_PATH.exists():
        raise FileExistsError(f"result already exists; refusing reuse: {RESULT_PATH}")

    split = read_json(SPLIT_PATH)
    if split.get("protocol") != "semantic_hard":
        raise RuntimeError("split protocol is not semantic_hard")
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise RuntimeError("expected semantic-hard CIFAR-100 60/40 split")

    arm_outputs: dict[str, Any] = {}
    for name, cluster_feature in (("feature", "feature_pca"), ("structure", "structure_pca")):
        work_dir = ROOT / "runs" / f"{BASE_TAG}_{name}_oraclek40"
        if work_dir.exists():
            raise FileExistsError(f"refusing to reuse existing output: {work_dir}")
        args = load_args(work_dir, cluster_feature)
        print(f"[discover] cluster_feature={cluster_feature} cluster_k={FIXED_K}", flush=True)
        discover(args)
        report_path = work_dir / "discovery_report.json"
        config_path = work_dir / "discover_config.json"
        report = read_json(report_path)
        config = read_json(config_path)
        protocol = report.get("rejector_protocol", {})
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"label-flow audit failed: {work_dir}")
        if abs(float(report.get("calibration_known_accept_rate", 0.0)) - 0.95) > 0.02:
            raise RuntimeError(f"known validation calibration outside tolerance: {work_dir}")
        if config.get("cluster_k") != "oracle":
            raise RuntimeError(f"oracle-K config audit failed: {config.get('cluster_k')}")
        if int(report.get("cluster_k", -1)) != FIXED_K:
            raise RuntimeError(
                f"actual cluster count is not fixed to {FIXED_K}: "
                f"{report.get('cluster_k')}"
            )
        arm_outputs[name] = {
            "config": config,
            "metrics": report,
            "summary": summary(report),
            "output": str(work_dir),
        }

    left = arm_outputs["feature"]["config"]
    right = arm_outputs["structure"]["config"]
    allowed = {"cluster_feature", "work_dir"}
    config_differences = {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    unexpected = {key: value for key, value in config_differences.items() if key not in allowed}
    if unexpected:
        raise RuntimeError(f"unexpected condition differences: {unexpected}")

    upstream_keys = (
        "auroc",
        "known_accept_rate",
        "unknown_reject_rate",
        "calibration_known_accept_rate",
        "cluster_candidate_count",
        "cluster_candidate_true_unknown_count_before_purification",
        "cluster_candidate_false_reject_count_before_purification",
        "cluster_candidate_purity_before_purification",
        "cluster_candidate_unknown_recall",
    )
    upstream_differences = {
        key: (
            arm_outputs["feature"]["metrics"].get(key),
            arm_outputs["structure"]["metrics"].get(key),
        )
        for key in upstream_keys
        if arm_outputs["feature"]["metrics"].get(key)
        != arm_outputs["structure"]["metrics"].get(key)
    }
    if upstream_differences:
        raise RuntimeError(f"fixed-K arms changed upstream metrics: {upstream_differences}")

    auto_reference = {}
    for name in ("feature", "structure"):
        reference = ROOT / "runs" / f"{BASE_TAG}_{name}" / "discovery_report.json"
        assert_file(reference)
        report = read_json(reference)
        auto_reference[name] = {
            "report": str(reference),
            "summary": summary(report),
        }

    result = {
        "protocol": "diagnostic oracle-K=40 comparison on the existing semantic-hard 60/40 split",
        "seed": SEED,
        "split": str(SPLIT_PATH),
        "student_checkpoint": str(STUDENT_CKPT),
        "fixed_cluster_k": FIXED_K,
        "cluster_k_mode": "oracle",
        "fixed_k_is_diagnostic_only": True,
        "test_labels_used_only_for_retrospective_metrics": True,
        "single_changed_factor_between_fixed_k_arms": "cluster_feature",
        "config_differences": config_differences,
        "upstream_metric_identity_audit": {
            "keys": list(upstream_keys),
            "differences": upstream_differences,
        },
        "automatic_k_reference": auto_reference,
        "arms": arm_outputs,
    }
    write_json(RESULT_PATH, result)
    print(json.dumps({"result": str(RESULT_PATH), "config_differences": config_differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
