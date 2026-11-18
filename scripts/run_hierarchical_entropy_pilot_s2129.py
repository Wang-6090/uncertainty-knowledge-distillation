"""Run a controlled hierarchical-entropy discovery comparison on seed 2129.

The existing feature-rejector discovery run is the control.  This pilot reuses
the same student checkpoint, split, rejector protocol, candidate pool,
threshold policy, and clustering settings; the only scientific factor changed
is the final discovery score from ``feature_rejector`` to
``hierarchical_entropy``.  The CIFAR-100 coarse taxonomy comes from the split
metadata and is used without any test or novel labels during scoring or
calibration.
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


SEED = 2129
SOURCE_DIR = ROOT / "runs/structure_representation_pilot_s2129_feature"
SOURCE_CONFIG = SOURCE_DIR / "discover_config.json"
STUDENT_CKPT = ROOT / "runs/structure_representation_pilot_s2129_student/student.pt"
SPLIT_PATH = ROOT / "splits/semantic_hard/seed2129/cifar100_60_40_semantic_hard.json"
TREATMENT_DIR = ROOT / "runs/hierarchical_entropy_pilot_s2129"
RESULT_PATH = ROOT / "analysis/hierarchical_entropy_pilot_s2129.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def require_files() -> None:
    for path in (SOURCE_CONFIG, STUDENT_CKPT, SPLIT_PATH):
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
    args.split_path = str(SPLIT_PATH)
    args.student_ckpt = str(STUDENT_CKPT)
    args.work_dir = str(TREATMENT_DIR)
    args.score_mode = "hierarchical_entropy"
    # The hierarchical-entropy detector does not consume a feature-rejector
    # representation.  Keep the rejector settings inherited from the control
    # run, but use the neutral feature mode required by the configuration
    # validator when that rejector is inactive.
    args.rejector_feature_mode = "embedding"
    args.threshold_policy = "known_coverage"
    args.target_known_coverage = 0.95
    args.open_val_ratio = 0.0
    args.cluster_feature = "feature_pca"
    args.cluster_k = "auto"
    args.skip_clustering = False
    return args


def summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "auroc",
        "aupr",
        "fpr95",
        "known_accept_rate",
        "unknown_reject_rate",
        "calibration_known_accept_rate",
        "cluster_candidate_count",
        "cluster_candidate_purity_before_purification",
        "cluster_candidate_unknown_recall",
        "cluster_unknown_only_acc",
        "cluster_unknown_only_nmi",
        "cluster_unknown_only_ari",
        "cluster_k",
        "cluster_true_k",
        "cluster_k_abs_error",
        "cluster_feature",
    )
    return {key: report.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    require_files()
    if TREATMENT_DIR.exists() or RESULT_PATH.exists():
        raise FileExistsError(
            "refusing to reuse existing hierarchical-entropy outputs: "
            f"{TREATMENT_DIR} or {RESULT_PATH}"
        )

    split = read_json(SPLIT_PATH)
    if split.get("protocol") != "semantic_hard":
        raise RuntimeError("expected semantic_hard split")
    if len(split.get("known_classes", [])) != 60 or len(split.get("novel_classes", [])) != 40:
        raise RuntimeError("expected CIFAR-100 60/40 split")
    fine_to_coarse = {int(k): int(v) for k, v in split["fine_to_coarse"].items()}
    known_per_coarse = {}
    for raw_class in split["known_classes"]:
        coarse = fine_to_coarse[int(raw_class)]
        known_per_coarse[coarse] = known_per_coarse.get(coarse, 0) + 1
    if sorted(known_per_coarse.values()) != [3] * 20:
        raise RuntimeError(f"unexpected known coarse composition: {known_per_coarse}")

    args = load_args()
    print("[discover] score_mode=hierarchical_entropy", flush=True)
    discover(args)

    treatment_config = read_json(TREATMENT_DIR / "discover_config.json")
    treatment_report = read_json(TREATMENT_DIR / "discovery_report.json")
    calibration = read_json(TREATMENT_DIR / "calibration_report.json")
    control_config = read_json(SOURCE_CONFIG)
    control_report = read_json(SOURCE_DIR / "discovery_report.json")

    differences = {
        key: (control_config.get(key), treatment_config.get(key))
        for key in sorted(set(control_config) | set(treatment_config))
        if control_config.get(key) != treatment_config.get(key)
    }
    unexpected = {
        key: value
        for key, value in differences.items()
        if key not in {"score_mode", "work_dir", "rejector_feature_mode"}
    }
    if unexpected:
        raise RuntimeError(f"controlled comparison changed extra factors: {unexpected}")

    hierarchy_report = calibration.get("hierarchical_entropy", {})
    if calibration.get("score_mode") != "hierarchical_entropy":
        raise RuntimeError("treatment did not record hierarchical_entropy")
    if hierarchy_report.get("unknown_labels_used_for_score_or_calibration") is not False:
        raise RuntimeError("unknown-label audit failed")
    if calibration.get("threshold_policy", {}).get("type") != "known_val_known_coverage":
        raise RuntimeError("unexpected threshold policy")
    if calibration.get("validation_protocol", {}).get("open_val_size", 0) not in (0, None):
        raise RuntimeError("open validation data entered calibration")

    metric_keys = (
        "auroc",
        "aupr",
        "fpr95",
        "known_accept_rate",
        "unknown_reject_rate",
        "calibration_known_accept_rate",
        "cluster_candidate_count",
        "cluster_candidate_purity_before_purification",
        "cluster_candidate_unknown_recall",
        "cluster_unknown_only_acc",
        "cluster_unknown_only_nmi",
        "cluster_unknown_only_ari",
        "cluster_k",
        "cluster_k_abs_error",
    )
    deltas = {
        key: (
            treatment_report.get(key) - control_report.get(key)
            if isinstance(treatment_report.get(key), (int, float))
            and isinstance(control_report.get(key), (int, float))
            else None
        )
        for key in metric_keys
    }
    result = {
        "protocol": "controlled score-only comparison on semantic-hard CIFAR-100 60/40",
        "seed": SEED,
        "hypothesis": (
            "coarse-aware local entropy may expose ambiguity among semantically related "
            "known classes more clearly than the learned rejector score"
        ),
        "single_changed_factor": "score_mode: feature_rejector -> hierarchical_entropy",
        "label_audit": {
            "unknown_labels_used_for_score_or_calibration": False,
            "unknown_labels_used_only_for_final_metrics": True,
        },
        "config_differences": differences,
        "control": {"output": str(SOURCE_DIR), "metrics": summary(control_report)},
        "treatment": {"output": str(TREATMENT_DIR), "metrics": summary(treatment_report)},
        "deltas_treatment_minus_control": deltas,
        "taxonomy": {
            "source": "split_metadata.fine_to_coarse",
            "known_classes": 60,
            "novel_classes": 40,
            "coarse_classes": len(set(fine_to_coarse.values())),
            "known_per_coarse": known_per_coarse,
        },
    }
    write_json(RESULT_PATH, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
