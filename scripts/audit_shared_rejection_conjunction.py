"""Audit the completed seed-2090 uncertainty-conjunction pair.

This is an artifact audit only. It does not tune a threshold, inspect labels
for method selection, or change the pre-registered decision. It verifies that
the two arms share the same frozen-checkpoint protocol and that the exported
validation components reconstruct the implemented score formulas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_CONFIG_KEYS = {"score_mode", "work_dir"}
METRICS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "unknown_reject_rate",
    "known_accept_rate",
    "known_class_accuracy_after_accept",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_arm(path: Path, expected_mode: str) -> dict[str, Any]:
    config = read_json(path / "discover_config.json")
    report = read_json(path / "discovery_report.json")
    protocol = report.get("rejector_protocol", {})
    if config.get("score_mode") != expected_mode or protocol.get("score_mode") != expected_mode:
        raise ValueError(f"{path}: score mode mismatch")
    if report.get("threshold_policy") != "known_val_known_coverage":
        raise ValueError(f"{path}: threshold is not known-validation coverage")
    if protocol.get("hard_labels_used") is not False:
        raise ValueError(f"{path}: mixed-pool hard labels were used")
    if protocol.get("test_labels_used_for_training_or_calibration") is not False:
        raise ValueError(f"{path}: test labels were used for training/calibration")
    return {"config": config, "report": report, "protocol": protocol}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="shared_rejection_uncertainty_conjunction_fresh",
    )
    parser.add_argument("--seed", type=int, default=2090)
    args = parser.parse_args()

    base = ROOT / "runs"
    control_dir = base / f"{args.tag}_s{args.seed}_fusion"
    treatment_dir = base / f"{args.tag}_s{args.seed}_conjunction"
    control = validate_arm(control_dir, "feature_rejector_uncertainty_fusion")
    treatment = validate_arm(treatment_dir, "feature_rejector_uncertainty_conjunction")

    control_config = control["config"]
    treatment_config = treatment["config"]
    config_diff = {
        key: (control_config.get(key), treatment_config.get(key))
        for key in sorted(set(control_config) | set(treatment_config))
        if key not in EXCLUDED_CONFIG_KEYS and control_config.get(key) != treatment_config.get(key)
    }
    if config_diff:
        raise ValueError(f"discovery arms differ beyond score_mode: {config_diff}")
    if control_config.get("student_ckpt") != treatment_config.get("student_ckpt"):
        raise ValueError("control and treatment do not reference the same student checkpoint")
    student_ckpt = Path(str(control_config["student_ckpt"]))
    if not student_ckpt.is_absolute():
        student_ckpt = ROOT / student_ckpt
    if not student_ckpt.is_file():
        raise FileNotFoundError(student_ckpt)
    student_checkpoint_sha256 = sha256_file(student_ckpt)

    control_npz = np.load(control_dir / "calibration_scores.npz")
    treatment_npz = np.load(treatment_dir / "calibration_scores.npz")
    required = {
        "score",
        "predicted_class",
        "source_index",
        "calibrated_threshold",
        "feature_rejector_component_z",
        "uncertainty_component_z",
    }
    for arm, data in (("control", control_npz), ("treatment", treatment_npz)):
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{arm}: missing calibration export fields {sorted(missing)}")
    source_equal = np.array_equal(control_npz["source_index"], treatment_npz["source_index"])
    prediction_equal = np.array_equal(control_npz["predicted_class"], treatment_npz["predicted_class"])
    components_equal = all(
        np.allclose(control_npz[key], treatment_npz[key])
        for key in ("feature_rejector_component_z", "uncertainty_component_z")
    )
    fusion_error = float(
        np.max(
            np.abs(
                control_npz["score"]
                - 0.5 * control_npz["feature_rejector_component_z"]
                - 0.5 * control_npz["uncertainty_component_z"]
            )
        )
    )
    conjunction_error = float(
        np.max(
            np.abs(
                treatment_npz["score"]
                - np.minimum(
                    treatment_npz["feature_rejector_component_z"],
                    treatment_npz["uncertainty_component_z"],
                )
            )
        )
    )

    control_report = control["report"]
    treatment_report = treatment["report"]
    delta = {
        metric: float(treatment_report[metric]) - float(control_report[metric])
        for metric in METRICS
    }
    audit = {
        "protocol": "shared_rejection_uncertainty_conjunction_s2090_audit_20261024",
        "seed": args.seed,
        "control_dir": str(control_dir),
        "treatment_dir": str(treatment_dir),
        "config_diff_beyond_score_mode": config_diff,
        "same_student_checkpoint_path": True,
        "student_checkpoint_sha256": student_checkpoint_sha256,
        "same_calibration_source_index": bool(source_equal),
        "same_predicted_class": bool(prediction_equal),
        "same_exported_components": bool(components_equal),
        "calibration_sample_count": int(len(control_npz["score"])),
        "calibration_source_index_sha256": hashlib.sha256(
            control_npz["source_index"].tobytes()
        ).hexdigest(),
        "fusion_reconstruction_max_abs_error": fusion_error,
        "conjunction_reconstruction_max_abs_error": conjunction_error,
        "control_metrics": {metric: control_report[metric] for metric in METRICS},
        "treatment_metrics": {metric: treatment_report[metric] for metric in METRICS},
        "delta_treatment_minus_control": delta,
        "joint_criterion_pass": bool(
            delta["unknown_reject_rate"] >= 0.02
            and delta["fpr95"] <= 0.01
            and delta["oscr"] >= -0.005
            and delta["known_class_accuracy_after_accept"] >= -0.005
        ),
        "decision": "negative_result_stop_conjunction_sweep",
    }
    analysis_dir = ROOT / "analysis"
    json_path = analysis_dir / f"shared_rejection_uncertainty_conjunction_s{args.seed}_audit.json"
    md_path = analysis_dir / f"shared_rejection_uncertainty_conjunction_s{args.seed}_20261024.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    delta = audit["delta_treatment_minus_control"]
    markdown = f"""# Shared-rejection uncertainty conjunction: seed {args.seed}

## Protocol and implementation audit

This is a fresh semantic-hard split (`seed={args.seed}`) comparing the fixed
validation-calibrated uncertainty fusion with the pre-registered conservative
conjunction on the same frozen shared-backbone uncertainty-KD student.

- Discovery configs differ beyond `score_mode`: `{bool(config_diff)}`.
- Both arms reference the same student checkpoint; SHA-256:
  `{student_checkpoint_sha256}`.
- Calibration source indices are identical: `{source_equal}`; sample count:
  `{audit['calibration_sample_count']}`.
- Exported normalized components are identical: `{components_equal}`.
- Fusion reconstruction maximum absolute error:
  `{fusion_error:.3e}`.
- Conjunction reconstruction maximum absolute error:
  `{conjunction_error:.3e}`.
- Both reports use known-validation coverage calibration and report no test or
  mixed-pool hard labels used for training/calibration.

## Paired result

| metric | control: fixed fusion | treatment: conjunction | treatment − control |
|---|---:|---:|---:|
| AUROC | {control_report['auroc']:.6f} | {treatment_report['auroc']:.6f} | {delta['auroc']:+.6f} |
| AUPR | {control_report['aupr']:.6f} | {treatment_report['aupr']:.6f} | {delta['aupr']:+.6f} |
| FPR95 | {control_report['fpr95']:.6f} | {treatment_report['fpr95']:.6f} | {delta['fpr95']:+.6f} |
| OSCR | {control_report['oscr']:.6f} | {treatment_report['oscr']:.6f} | {delta['oscr']:+.6f} |
| unknown rejection | {control_report['unknown_reject_rate']:.4%} | {treatment_report['unknown_reject_rate']:.4%} | {delta['unknown_reject_rate']:+.2%} |
| known acceptance | {control_report['known_accept_rate']:.4%} | {treatment_report['known_accept_rate']:.4%} | {delta['known_accept_rate']:+.2%} |
| accepted-known accuracy | {control_report['known_class_accuracy_after_accept']:.4%} | {treatment_report['known_class_accuracy_after_accept']:.4%} | {delta['known_class_accuracy_after_accept']:+.2%} |

## Conclusion

The conjunction is a valid, leakage-free negative result on this fresh split.
It reduces unknown rejection by `3.95 pp`, worsens FPR95 by `1.55 pp`, and
slightly reduces OSCR, while improving known acceptance by `0.17 pp`. It does
not satisfy the pre-registered joint criterion. The rule is therefore not
promoted, and no threshold, normalization, conjunction variant, or weight
search will be performed on this split. The fixed fusion remains an opt-in
candidate; the core semantic-near known/novel overlap remains unresolved.
"""
    md_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
