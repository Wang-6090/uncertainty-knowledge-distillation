"""Compare risk-ranked and risk-diversity candidate selection on seed 2128.

The detector, calibration rule, student checkpoint, discovery pool, and
clustering configuration are frozen.  The only experimental factor is the
label-free selection rule applied after detector rejection:

* ``open_score`` keeps the highest-risk rejected samples;
* ``risk_diversity`` keeps high-risk samples while reducing redundancy in the
  student feature space.

The test labels are read only after both discovery runs, for retrospective
candidate-pool and clustering analysis.  They are never used for training,
threshold calibration, candidate selection, or choosing the method.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


SEED = 2128
KEEP_RATIO = 0.95
SPLIT_PATH = ROOT / "splits/semantic_hard/seed2128/cifar100_60_40_semantic_hard.json"
SOURCE_CONFIG = ROOT / "runs/structure_representation_pilot_s2128_feature/discover_config.json"
STUDENT_CKPT = ROOT / "runs/structure_representation_pilot_s2128_student/student.pt"
ANALYSIS_PATH = ROOT / "analysis/risk_diversity_pilot_s2128.json"
BASE_TAG = "risk_diversity_pilot_s2128"
MODES = ("open_score", "risk_diversity")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {path}")


def make_args(mode: str, work_dir: Path):
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

    # Freeze the protocol explicitly so a stale source config cannot silently
    # change the scientific comparison.
    args.score_mode = "feature_rejector"
    args.threshold_policy = "known_coverage"
    args.target_known_coverage = 0.95
    args.open_val_ratio = 0.0
    args.discovery_pool_mode = "mixed"
    args.matched_discovery_pool = True
    args.matched_pool_size = 5400
    args.matched_known_prior = 0.2
    args.cluster_feature = "feature_pca"
    args.cluster_method = "kmeans"
    args.cluster_selection = "silhouette"
    args.cluster_k = "auto"
    args.cluster_candidate_mode = "detector"
    args.candidate_purify = mode
    args.candidate_keep_ratio = KEEP_RATIO
    args.skip_clustering = False
    return args


def arrays_from_detail(path: Path) -> dict[str, np.ndarray]:
    detail = read_json(path)
    return {
        key: np.asarray(detail[key])
        for key in ("pred_known", "true_known", "raw_labels", "pred_cluster")
    }


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == 0):
        return 0.0
    ordered = np.sort(values)
    index = np.arange(1, len(ordered) + 1, dtype=float)
    return float((2.0 * np.sum(index * ordered) / (len(ordered) * np.sum(ordered))) - (len(ordered) + 1.0) / len(ordered))


def retrospective_candidate_analysis(work_dir: Path, novel_classes: list[int]) -> dict[str, Any]:
    arrays = arrays_from_detail(work_dir / "discovery_detail.json")
    pred_known = arrays["pred_known"].astype(bool)
    raw_labels = arrays["raw_labels"].astype(int)
    candidate_mask = ~pred_known
    novel_set = set(int(x) for x in novel_classes)
    unknown_mask = np.asarray([label in novel_set for label in raw_labels], dtype=bool)

    purification = read_json(work_dir / "discovery_report.json").get("candidate_purification", {})
    # The selected mask is reconstructible from the persisted candidate count
    # and the cluster prediction mask only for the no-purification case.  For
    # both experimental arms, train.py stores the selected cluster assignments
    # as non-negative labels and leaves all unselected samples at -1.
    pred_cluster = arrays["pred_cluster"].astype(int)
    selected_mask = pred_cluster >= 0
    if selected_mask.sum() == 0 and int(purification.get("after_count", 0)) > 0:
        raise RuntimeError(f"cannot reconstruct selected candidate mask: {work_dir}")

    # A selected cluster assignment is the subset used for clustering.  The
    # detector candidate mask remains the upstream, unchanged reference.
    selected_unknown = selected_mask & unknown_mask
    unknown_counts = np.asarray(
        [int(np.sum(selected_unknown & (raw_labels == cls))) for cls in novel_classes],
        dtype=int,
    )
    top_n = min(10, len(unknown_counts))
    total_selected_unknown = int(selected_unknown.sum())
    sorted_counts = np.sort(unknown_counts)[::-1]
    return {
        "candidate_count_before_purification": int(candidate_mask.sum()),
        "candidate_count_after_purification": int(selected_mask.sum()),
        "candidate_unknown_count_before_purification": int(np.sum(candidate_mask & unknown_mask)),
        "candidate_unknown_count_after_purification": int(selected_unknown.sum()),
        "candidate_purity_before_purification": float(np.sum(candidate_mask & unknown_mask) / max(int(candidate_mask.sum()), 1)),
        "candidate_purity_after_purification": float(np.sum(selected_unknown) / max(int(selected_mask.sum()), 1)),
        "candidate_unknown_recall_before_purification": float(np.sum(candidate_mask & unknown_mask) / max(int(unknown_mask.sum()), 1)),
        "candidate_unknown_recall_after_purification": float(np.sum(selected_unknown) / max(int(unknown_mask.sum()), 1)),
        "unknown_class_candidate_counts_after_purification": {
            str(cls): int(count) for cls, count in zip(novel_classes, unknown_counts)
        },
        "unknown_class_candidate_count_gini_after_purification": gini(unknown_counts),
        "top10_unknown_class_candidate_mass_after_purification": float(
            sorted_counts[:top_n].sum() / max(total_selected_unknown, 1)
        ),
        "unknown_classes_with_at_least_one_candidate_after_purification": int(np.sum(unknown_counts > 0)),
        "unknown_class_coverage_after_purification": float(np.mean(unknown_counts > 0)),
    }


def report_summary(work_dir: Path, novel_classes: list[int]) -> dict[str, Any]:
    report = read_json(work_dir / "discovery_report.json")
    return {
        "auroc": report.get("auroc"),
        "aupr": report.get("aupr"),
        "fpr95": report.get("fpr95"),
        "known_accept_rate": report.get("known_accept_rate"),
        "unknown_reject_rate": report.get("unknown_reject_rate"),
        "cluster_candidate_count": report.get("cluster_candidate_count"),
        "cluster_candidate_count_before_purification": report.get(
            "cluster_candidate_count_before_purification"
        ),
        "cluster_candidate_purity": report.get("cluster_candidate_purity"),
        "cluster_candidate_unknown_recall": report.get("cluster_candidate_unknown_recall"),
        "cluster_acc": report.get("cluster_acc"),
        "cluster_nmi": report.get("cluster_nmi"),
        "cluster_ari": report.get("cluster_ari"),
        "cluster_k": report.get("cluster_k"),
        "cluster_k_abs_error": report.get("cluster_k_abs_error"),
        "candidate_purification": report.get("candidate_purification"),
        "retrospective_candidate_analysis": retrospective_candidate_analysis(work_dir, novel_classes),
    }


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def main() -> None:
    for path in (SPLIT_PATH, SOURCE_CONFIG, STUDENT_CKPT):
        require_file(path)
    require_new(ANALYSIS_PATH)
    split = read_json(SPLIT_PATH)
    if split.get("protocol") != "semantic_hard":
        raise RuntimeError("expected semantic_hard split")
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise RuntimeError("expected CIFAR-100 semantic-hard 60/40 split")
    novel_classes = [int(x) for x in split["novel_classes"]]

    arms: dict[str, Any] = {}
    for mode in MODES:
        work_dir = ROOT / "runs" / f"{BASE_TAG}_{mode}_r095"
        require_new(work_dir)
        print(f"[discover] mode={mode} keep_ratio={KEEP_RATIO}", flush=True)
        discover(make_args(mode, work_dir))
        config = read_json(work_dir / "discover_config.json")
        report = read_json(work_dir / "discovery_report.json")
        calibration = read_json(work_dir / "calibration_report.json")
        if config.get("candidate_purify") != mode:
            raise RuntimeError(f"candidate purifier did not materialize: {work_dir}")
        if float(config.get("candidate_keep_ratio")) != KEEP_RATIO:
            raise RuntimeError(f"candidate keep ratio did not materialize: {work_dir}")
        protocol = report.get("rejector_protocol", {})
        if protocol.get("test_labels_used_for_training_or_calibration") is not False:
            raise RuntimeError(f"test-label flow audit failed: {work_dir}")
        if not math.isclose(float(report.get("calibration_known_accept_rate", 0.0)), 0.95, abs_tol=0.02):
            raise RuntimeError(f"known-coverage calibration audit failed: {work_dir}")
        arms[mode] = {
            "output": str(work_dir),
            "config": config,
            "metrics": report_summary(work_dir, novel_classes),
            "calibration": {
                "threshold": calibration.get("calibrated_threshold"),
                "known_accept_rate": calibration.get("threshold_policy", {}).get("known_accept_rate"),
                "test_labels_used_for_training_or_calibration": protocol.get("test_labels_used_for_training_or_calibration"),
            },
        }

    left = arms[MODES[0]]["config"]
    right = arms[MODES[1]]["config"]
    config_differences = {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if not values_equal(left.get(key), right.get(key))
    }
    allowed_differences = {"candidate_purify", "work_dir"}
    unexpected_config_differences = {
        key: value for key, value in config_differences.items() if key not in allowed_differences
    }
    if unexpected_config_differences:
        raise RuntimeError(f"unexpected condition differences: {unexpected_config_differences}")

    upstream_keys = (
        "auroc",
        "aupr",
        "fpr95",
        "known_accept_rate",
        "unknown_reject_rate",
        "cluster_candidate_count_before_purification",
        "cluster_candidate_purity_before_purification",
        "cluster_candidate_unknown_recall_before_purification",
    )
    upstream_differences = {
        key: (
            arms[MODES[0]]["metrics"].get(key),
            arms[MODES[1]]["metrics"].get(key),
        )
        for key in upstream_keys
        if not values_equal(
            arms[MODES[0]]["metrics"].get(key), arms[MODES[1]]["metrics"].get(key)
        )
    }
    if upstream_differences:
        raise RuntimeError(f"upstream detector metrics changed between arms: {upstream_differences}")

    result = {
        "protocol": "single-factor candidate selection comparison on semantic-hard CIFAR-100 60/40",
        "seed": SEED,
        "split": str(SPLIT_PATH),
        "student_checkpoint": str(STUDENT_CKPT),
        "candidate_keep_ratio": KEEP_RATIO,
        "test_labels_used_only_for_retrospective_metrics": True,
        "single_changed_factor": "candidate_purify",
        "fixed_conditions": {
            "score_mode": "feature_rejector",
            "threshold_policy": "known_coverage",
            "target_known_coverage": 0.95,
            "discovery_pool": "matched mixed pool, size 5400, known fraction 0.2",
            "cluster_feature": "feature_pca",
            "cluster_method": "kmeans",
            "cluster_selection": "silhouette",
            "cluster_k": "auto",
        },
        "config_differences": config_differences,
        "unexpected_config_differences": unexpected_config_differences,
        "upstream_metric_identity_audit": {
            "keys": list(upstream_keys),
            "differences": upstream_differences,
        },
        "arms": arms,
    }
    write_json(ANALYSIS_PATH, result)
    print(json.dumps({"result": str(ANALYSIS_PATH), "config_differences": config_differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
