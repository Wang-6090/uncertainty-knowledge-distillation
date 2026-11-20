"""Run a fixed raw-rejector versus conjunction audit on frozen students.

The runner is deliberately discovery-only: it never retrains a teacher or
student.  It materializes one common discovery protocol in memory for every
checkpoint, verifies checkpoint/split/model compatibility, and evaluates only
the two predeclared score modes:

* ``feature_rejector``
* ``feature_rejector_uncertainty_conjunction``

This is useful for legacy A/B/C checkpoints whose historical discovery reports
used incompatible pool, rejector, or threshold settings.  Those historical
settings are not silently reused; the overridden protocol is recorded in the
machine-readable result.
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

from scripts.run_episode_pseudonovel_pair import (  # noqa: E402
    load_args,
    read_json,
    validate_report,
    write_json,
)
from train import discover  # noqa: E402


SCORE_MODES = (
    "feature_rejector",
    "feature_rejector_uncertainty_conjunction",
)


def _resolve_repo_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _same_path(left: Any, right: Any) -> bool:
    left_path = _resolve_repo_path(left)
    right_path = _resolve_repo_path(right)
    return left_path is not None and right_path is not None and left_path == right_path


def _parse_student(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--student must use NAME=CHECKPOINT")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name or not str(path):
        raise argparse.ArgumentTypeError("--student must use NAME=CHECKPOINT")
    return name, path


def _checkpoint_config(checkpoint: Path) -> tuple[Path | None, dict[str, Any]]:
    for name in ("train_student_config.json", "config.json"):
        candidate = checkpoint.parent / name
        if candidate.is_file():
            return candidate, read_json(candidate)
    return None, {}


def _preflight(
    source: Path,
    source_values: dict[str, Any],
    seed: int,
    students: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    source_seed = source_values.get("seed")
    if source_seed is not None and int(source_seed) != seed:
        errors.append(f"source config seed={source_seed} does not match requested seed={seed}")
    source_split = source_values.get("split_path")
    if not source_split:
        errors.append("source discovery config has no split_path")

    checkpoint_audits: dict[str, Any] = {}
    for name, checkpoint in students.items():
        checkpoint = checkpoint.resolve()
        config_path, config = _checkpoint_config(checkpoint)
        item: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "exists": checkpoint.is_file(),
            "config": str(config_path) if config_path else None,
            "config_seed": config.get("seed"),
            "config_split_path": config.get("split_path"),
        }
        if not checkpoint.is_file():
            errors.append(f"{name}: checkpoint does not exist: {checkpoint}")
        if config_path is None:
            errors.append(f"{name}: no train_student_config.json or config.json beside checkpoint")
        else:
            if config.get("seed") is not None and int(config["seed"]) != seed:
                errors.append(f"{name}: checkpoint config seed={config['seed']} does not match {seed}")
            if config.get("split_path") and not _same_path(config["split_path"], source_split):
                errors.append(
                    f"{name}: checkpoint split {config['split_path']} does not match source split {source_split}"
                )
            for key in ("backbone", "student_backbone", "proj_dim", "dropout", "image_size"):
                if key in source_values and key in config and source_values[key] != config[key]:
                    errors.append(
                        f"{name}: model field {key} differs between source={source_values[key]!r} "
                        f"and checkpoint config={config[key]!r}"
                    )
        checkpoint_audits[name] = item

    protocol_override = {
        "discovery_pool_mode": "mixed",
        "matched_discovery_pool": True,
        "matched_pool_size": 5400,
        "matched_known_prior": 0.2,
        "rejector_training": "nnpu",
        "rejector_nnpu_risk": "nu_corrected",
        "rejector_nnpu_model": "linear",
        "rejector_feature_mode": "embedding",
        "threshold_policy": "known_coverage",
        "target_known_coverage": 0.95,
        "open_val_ratio": 0.0,
        "cluster_k": "auto",
        "skip_clustering": False,
    }
    for key, expected in protocol_override.items():
        if key in {"skip_clustering", "open_val_ratio"}:
            continue
        # The old source config is allowed to differ because this runner
        # explicitly materializes the fixed protocol below.  Keep the old
        # value visible so the report cannot be mistaken for a historical
        # discovery rerun.
    return {
        "passed": not errors,
        "errors": errors,
        "source_discover_config": str(source),
        "source_seed": source_values.get("seed"),
        "source_split_path": source_values.get("split_path"),
        "students": checkpoint_audits,
        "materialized_protocol_override": protocol_override,
        "historical_source_protocol": {
            key: source_values.get(key)
            for key in (
                "discovery_pool_mode",
                "matched_discovery_pool",
                "matched_pool_size",
                "matched_known_prior",
                "rejector_training",
                "rejector_feature_mode",
                "threshold_policy",
                "target_known_coverage",
                "cluster_k",
                "skip_clustering",
            )
        },
    }


def _materialize_args(source: Path, output: Path, seed: int, checkpoint: Path, score_mode: str):
    args = load_args(source, "discover", output, seed)
    args.student_ckpt = str(checkpoint.resolve())
    args.score_mode = score_mode
    args.device = "auto"
    args.discovery_pool_mode = "mixed"
    args.matched_discovery_pool = True
    args.matched_pool_size = 5400
    args.matched_known_prior = 0.2
    args.mixed_known_pool_ratio = 0.2
    args.rejector_training = "nnpu"
    args.rejector_nnpu_risk = "nu_corrected"
    args.rejector_nnpu_model = "linear"
    args.rejector_feature_mode = "embedding"
    args.rejector_strict_mixed = False
    args.independent_rejection_encoder = False
    args.rejection_feature_dim = 0
    args.open_val_ratio = 0.0
    args.threshold_policy = "known_coverage"
    args.target_known_coverage = 0.95
    args.save_calibration_scores = True
    args.skip_clustering = False
    args.cluster_k = "auto"
    args.cluster_selection = "silhouette"
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--student", type=_parse_student, action="append", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    options = parser.parse_args()

    source = options.source_discover_config.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_values = read_json(source)
    students = dict(options.student)
    if len(students) != len(options.student):
        raise ValueError("student names must be unique")
    preflight = _preflight(source, source_values, options.seed, students)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    if not preflight["passed"]:
        raise SystemExit(2)
    if options.preflight_only:
        return

    results: dict[str, Any] = {}
    for name, checkpoint in students.items():
        for score_mode in SCORE_MODES:
            output = ROOT / "runs" / f"{options.tag}_s{options.seed}_{name}_{score_mode}"
            if output.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {output}")
            args = _materialize_args(source, output, options.seed, checkpoint, score_mode)
            discover(args)
            report = validate_report(output / "discovery_report.json")
            if report.get("threshold_policy") != "known_val_known_coverage":
                raise RuntimeError(
                    f"{output}: expected known-validation coverage threshold, "
                    f"got {report.get('threshold_policy')!r}"
                )
            rejector_protocol = report.get("rejector_protocol") or {}
            if rejector_protocol.get("test_labels_used_for_training_or_calibration") is not False:
                raise RuntimeError(f"{output}: rejector protocol reports test-label use")
            effective_protocol = report.get("effective_score_protocol") or {}
            if effective_protocol.get("threshold_source") != "known_validation":
                raise RuntimeError(
                    f"{output}: threshold source is not known validation: "
                    f"{effective_protocol.get('threshold_source')!r}"
                )
            target_coverage = report.get("target_known_coverage")
            realized_coverage = report.get("calibration_known_accept_rate")
            results[f"{name}:{score_mode}"] = {
                key: report.get(key)
                for key in (
                    "auroc",
                    "aupr",
                    "fpr95",
                    "oscr",
                    "known_class_accuracy_all_known",
                    "known_accept_rate",
                    "unknown_reject_rate",
                    "cluster_candidate_count",
                    "cluster_candidate_purity",
                    "cluster_candidate_unknown_recall",
                    "cluster_k_abs_error",
                    "cluster_acc",
                    "cluster_nmi",
                    "cluster_ari",
                    "threshold_policy",
                    "calibration_known_accept_rate",
                    "rejector_protocol",
                    "effective_score_protocol",
                )
            }
            results[f"{name}:{score_mode}"].update(
                {
                    "target_known_coverage": target_coverage,
                    "calibration_known_coverage_gap": (
                        None
                        if target_coverage is None or realized_coverage is None
                        else float(realized_coverage) - float(target_coverage)
                    ),
                    "cluster_evaluable": report.get("cluster_candidate_count") is not None,
                }
            )

    result_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}.json"
    write_json(
        result_path,
        {
            "protocol": "formal_frozen_conjunction_audit_20261121",
            "seed": options.seed,
            "score_modes": list(SCORE_MODES),
            "source_discover_config": str(source),
            "student_checkpoints": {name: str(path.resolve()) for name, path in students.items()},
            "preflight": preflight,
            "test_labels_used_for_training_selection_or_calibration": False,
            "results": results,
        },
    )
    print(json.dumps(read_json(result_path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
