"""Single-factor audit of class-conditional rejection thresholds.

The control run and the treatment run reuse the same split, student checkpoint,
rejector score, discovery pool, calibration protocol, candidate features and
clustering settings.  The only experimental factor is the threshold policy:

* control: ``known_coverage`` uses one threshold for all predicted classes;
* treatment: ``class_conditional`` calibrates one threshold per predicted class
  from known validation samples, with the existing small-class fallback.

Unknown labels are retained only for retrospective test metrics.  The script
refuses to reuse an existing treatment directory or result file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args  # noqa: E402


SEED = 2129
SOURCE_CONFIG = ROOT / "runs/structure_representation_pilot_s2129_feature/discover_config.json"
CONTROL_REPORT = ROOT / "runs/structure_representation_pilot_s2129_feature/discovery_report.json"
CONTROL_CALIBRATION = ROOT / "runs/structure_representation_pilot_s2129_feature/calibration_report.json"
SPLIT_PATH = ROOT / "splits/semantic_hard/seed2129/cifar100_60_40_semantic_hard.json"
STUDENT_CKPT = ROOT / "runs/structure_representation_pilot_s2129_student/student.pt"
TREATMENT_DIR = ROOT / "runs/class_conditional_threshold_pilot_s2129"
RESULT_PATH = ROOT / "analysis/class_conditional_threshold_pilot_s2129.json"


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


def load_args() -> Any:
    args = parse_args(["discover"])
    source = read_json(SOURCE_CONFIG)
    for key, value in source.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = SEED
    args.model_seed = None
    args.work_dir = str(TREATMENT_DIR)
    args.split_path = str(SPLIT_PATH)
    args.student_ckpt = str(STUDENT_CKPT)
    args.threshold_policy = "class_conditional"
    args.open_val_ratio = 0.0
    args.target_known_coverage = 0.95
    args.cluster_feature = "feature_pca"
    args.score_mode = "feature_rejector"
    args.cluster_k = "auto"
    args.skip_clustering = False
    return args


def metric_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "auroc",
        "aupr",
        "fpr95",
        "oscr",
        "known_accept_rate",
        "unknown_reject_rate",
        "known_class_accuracy_after_accept",
        "accepted_correct_fraction_of_all_known",
        "cluster_candidate_count",
        "cluster_candidate_true_unknown_count_before_purification",
        "cluster_candidate_false_reject_count_before_purification",
        "cluster_candidate_purity_before_purification",
        "cluster_candidate_unknown_recall",
        "cluster_unknown_only_acc",
        "cluster_unknown_only_nmi",
        "cluster_unknown_only_ari",
        "cluster_k",
        "cluster_true_k",
        "cluster_k_abs_error",
    )
    return {key: report.get(key) for key in keys}


def numeric_delta(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(control) | set(treatment)):
        before, after = control.get(key), treatment.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            result[key] = {
                "control": before,
                "class_conditional": after,
                "delta": float(after) - float(before),
            }
    return result


def classwise_known_rejection(
    report_dir: Path,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    detail = read_json(report_dir / "discovery_detail.json")
    pred_class = np.asarray(detail["pred_class"], dtype=np.int64)
    # Discovery detail uses ``true_known`` for the retrospective identity
    # field; it is only read here to summarize test outcomes after the run.
    is_known = np.asarray(detail["true_known"], dtype=bool)
    pred_known = np.asarray(detail["pred_known"], dtype=bool)
    result: dict[str, Any] = {}
    for class_id in sorted(np.unique(pred_class[is_known]).tolist()):
        mask = is_known & (pred_class == class_id)
        result[str(int(class_id))] = {
            "count": int(mask.sum()),
            "known_reject_rate": float(np.mean(~pred_known[mask])) if mask.any() else 0.0,
        }
    threshold = calibration.get("calibrated_threshold")
    threshold_values = np.asarray(threshold, dtype=float).reshape(-1)
    return {
        "classwise_known_test_rejection": result,
        "calibration_threshold_count": int(len(threshold_values)),
        "calibration_threshold_min": float(threshold_values.min()),
        "calibration_threshold_median": float(np.median(threshold_values)),
        "calibration_threshold_max": float(threshold_values.max()),
        "calibration_threshold_mean": float(threshold_values.mean()),
    }


def config_differences(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    differences = {
        key: {"control": control.get(key), "treatment": treatment.get(key)}
        for key in sorted(set(control) | set(treatment))
        if control.get(key) != treatment.get(key)
    }
    allowed = {"threshold_policy", "work_dir"}
    unexpected = {key: value for key, value in differences.items() if key not in allowed}
    if unexpected:
        raise RuntimeError(f"unexpected condition differences: {unexpected}")
    if treatment.get("threshold_policy") != "class_conditional":
        raise RuntimeError("treatment configuration did not persist class_conditional")
    return differences


def audit_protocol(report: dict[str, Any], calibration: dict[str, Any]) -> None:
    protocol = report.get("rejector_protocol", {})
    if protocol.get("test_labels_used_for_training_or_calibration") is not False:
        raise RuntimeError("test labels were used for training or calibration")
    # Older discovery reports do not persist these optional audit fields.  A
    # missing field is not evidence of label leakage; an explicit True is.
    if report.get("auto_score_selection_uses_open_labels") is True:
        raise RuntimeError("open labels affected score selection")
    if report.get("hierarchical_entropy", {}).get("unknown_labels_used_for_score_or_calibration") is True:
        raise RuntimeError("unknown labels affected score or calibration")
    validation = calibration.get("validation_protocol", {})
    if validation.get("uses_disjoint_calibration_subset") is not True:
        raise RuntimeError("checkpoint selection and calibration subsets are not disjoint")
    if float(validation.get("requested_calibration_ratio", 0.0)) != 0.25:
        raise RuntimeError("calibration ratio changed")
    report_threshold_policy = report.get("threshold_policy")
    report_threshold_type = (
        report_threshold_policy.get("type")
        if isinstance(report_threshold_policy, dict)
        else report_threshold_policy
    )
    if report_threshold_type != "known_val_class_conditional":
        raise RuntimeError("unexpected threshold policy report")
    if calibration.get("threshold_policy", {}).get("type") != "known_val_class_conditional":
        raise RuntimeError("calibration report does not record class conditional policy")
    if calibration.get("validation_protocol", {}).get("uses_disjoint_calibration_subset") is not True:
        raise RuntimeError("calibration report lost disjoint-subset audit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="audit and summarize an already completed treatment run without rerunning it",
    )
    options = parser.parse_args()
    if options.seed != SEED:
        raise ValueError(f"this locked pilot only supports seed {SEED}")
    for path in (SOURCE_CONFIG, CONTROL_REPORT, CONTROL_CALIBRATION, SPLIT_PATH, STUDENT_CKPT):
        require_file(path)
    if not options.finalize_existing and (TREATMENT_DIR.exists() or RESULT_PATH.exists()):
        raise FileExistsError("refusing to reuse class-conditional pilot outputs")

    if options.finalize_existing:
        for path in (
            TREATMENT_DIR / "discovery_report.json",
            TREATMENT_DIR / "calibration_report.json",
            TREATMENT_DIR / "discovery_detail.json",
        ):
            require_file(path)
        print(json.dumps({"mode": "finalize_existing", "work_dir": str(TREATMENT_DIR)}, indent=2), flush=True)
    else:
        args = load_args()
        print(
            json.dumps(
                {
                    "single_changed_factor": "threshold_policy",
                    "control": "known_coverage",
                    "treatment": "class_conditional",
                    "split": str(SPLIT_PATH),
                    "student_checkpoint": str(STUDENT_CKPT),
                    "work_dir": str(TREATMENT_DIR),
                },
                indent=2,
            ),
            flush=True,
        )
        discover(args)

    control_report = read_json(CONTROL_REPORT)
    treatment_report = read_json(TREATMENT_DIR / "discovery_report.json")
    control_config = read_json(SOURCE_CONFIG)
    treatment_config = read_json(TREATMENT_DIR / "discover_config.json")
    control_calibration = read_json(CONTROL_CALIBRATION)
    treatment_calibration = read_json(TREATMENT_DIR / "calibration_report.json")
    audit_protocol(treatment_report, treatment_calibration)
    differences = config_differences(control_config, treatment_config)
    control_classwise = classwise_known_rejection(
        CONTROL_REPORT.parent, control_calibration
    )
    treatment_classwise = classwise_known_rejection(TREATMENT_DIR, treatment_calibration)

    result = {
        "protocol": "single-factor class-conditional threshold pilot",
        "seed": SEED,
        "split": str(SPLIT_PATH),
        "student_checkpoint": str(STUDENT_CKPT),
        "unknown_labels_used_only_for_retrospective_metrics": True,
        "single_changed_factor": "threshold_policy",
        "control_threshold_policy": "known_coverage",
        "treatment_threshold_policy": "class_conditional",
        "config_differences": differences,
        "control": {
            "run_dir": str(CONTROL_REPORT.parent),
            "metrics": metric_snapshot(control_report),
            "threshold_policy": control_calibration.get("threshold_policy"),
            **control_classwise,
        },
        "treatment": {
            "run_dir": str(TREATMENT_DIR),
            "metrics": metric_snapshot(treatment_report),
            "threshold_policy": treatment_calibration.get("threshold_policy"),
            **treatment_classwise,
        },
        "delta_treatment_minus_control": numeric_delta(
            metric_snapshot(control_report), metric_snapshot(treatment_report)
        ),
    }
    write_json(RESULT_PATH, result)
    print(json.dumps({"result": str(RESULT_PATH), "config_differences": differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
