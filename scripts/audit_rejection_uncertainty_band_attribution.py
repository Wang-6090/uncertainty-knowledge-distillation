"""Attribute boundary-uncertainty gains to score re-ranking or threshold changes.

This audit reuses frozen baseline and uncertainty-band outputs.  It does not
train a model, select a parameter from test labels, or change any run output.
For each split it evaluates four cases:

1. baseline: baseline score + baseline threshold
2. threshold_only: baseline score + uncertainty-band threshold
3. residual_only_same_threshold: uncertainty-band score + baseline threshold
4. full: uncertainty-band score + uncertainty-band threshold

The purpose is to separate the effect of uncertainty-based score re-ranking
from the effect of recalibrating the operating threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (2121, 2122, 2123, 2124, 2125, 2127)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value: np.ndarray | float | int) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"expected scalar, got shape={array.shape}")
    return float(array[0])


def binary_metrics(scores: np.ndarray, known_mask: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    known_mask = np.asarray(known_mask, dtype=bool).reshape(-1)
    if scores.shape != known_mask.shape:
        raise ValueError(f"score/known-mask mismatch: {scores.shape} vs {known_mask.shape}")
    unknown_mask = ~known_mask
    pred_unknown = scores > threshold
    candidate_count = int(pred_unknown.sum())
    candidate_unknown_count = int((pred_unknown & unknown_mask).sum())
    return {
        "auroc": float(roc_auc_score(unknown_mask.astype(np.int8), scores)),
        "aupr": float(average_precision_score(unknown_mask.astype(np.int8), scores)),
        "threshold": float(threshold),
        "known_accept_rate": float((~pred_unknown[known_mask]).mean()),
        "unknown_reject_rate": float(pred_unknown[unknown_mask].mean()),
        "unknown_false_accept": int((~pred_unknown[unknown_mask]).sum()),
        "known_rejected": int(pred_unknown[known_mask].sum()),
        "candidate_count": candidate_count,
        "candidate_unknown_count": candidate_unknown_count,
        "candidate_coverage": float(candidate_unknown_count / unknown_mask.sum()),
        "candidate_purity": float(candidate_unknown_count / max(1, candidate_count)),
    }


def delta_metrics(case: dict, baseline: dict) -> dict:
    keys = (
        "auroc",
        "aupr",
        "known_accept_rate",
        "unknown_reject_rate",
        "candidate_count",
        "candidate_coverage",
        "candidate_purity",
        "unknown_false_accept",
        "known_rejected",
    )
    return {key: case[key] - baseline[key] for key in keys}


def evaluate_seed(seed: int, uncertainty_suffix: str = "") -> dict:
    baseline_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control_discover"
    uncertainty_dir = ROOT / "runs" / f"rejection_uncertainty_band_full_s{seed}{uncertainty_suffix}"
    baseline_export = (
        ROOT / "runs" / f"rejection_boundary_calibration_s{seed}" / "calibration_scores.npz"
    )
    baseline_detail_path = baseline_dir / "discovery_detail.json"
    baseline_calibration_path = baseline_dir / "calibration_report.json"
    uncertainty_detail_path = uncertainty_dir / "discovery_detail.json"
    uncertainty_calibration_path = uncertainty_dir / "calibration_report.json"
    required = (baseline_detail_path, uncertainty_detail_path, uncertainty_calibration_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen outputs:\n" + "\n".join(missing))

    baseline_detail = read_json(baseline_detail_path)
    baseline_calibration = read_json(baseline_calibration_path)
    uncertainty_detail = read_json(uncertainty_detail_path)
    uncertainty_calibration = read_json(uncertainty_calibration_path)
    if baseline_export.is_file():
        with np.load(baseline_export, allow_pickle=False) as payload:
            baseline_threshold = scalar(payload["calibrated_threshold"])
            calibration_sample_count = int(len(payload["score"]))
        baseline_threshold_source = str(baseline_export.relative_to(ROOT))
    else:
        baseline_threshold = scalar(baseline_calibration["calibrated_threshold"])
        calibration_sample_count = None
        baseline_threshold_source = str(baseline_calibration_path.relative_to(ROOT))

    baseline_score = np.asarray(baseline_detail["score"], dtype=np.float64).reshape(-1)
    uncertainty_score = np.asarray(uncertainty_detail["score"], dtype=np.float64).reshape(-1)
    uncertainty_raw_score = np.asarray(
        uncertainty_detail["raw_score"], dtype=np.float64
    ).reshape(-1)
    known_mask = np.asarray(baseline_detail["true_known"], dtype=bool).reshape(-1)
    if not (
        baseline_score.shape
        == uncertainty_score.shape
        == uncertainty_raw_score.shape
        == known_mask.shape
    ):
        raise ValueError(f"test arrays are not aligned for seed={seed}")
    raw_score_max_abs_diff = float(np.max(np.abs(baseline_score - uncertainty_raw_score)))
    if raw_score_max_abs_diff > 1e-8:
        raise ValueError(
            f"raw score changed for seed={seed}: max abs diff={raw_score_max_abs_diff}"
        )

    boundary_report = uncertainty_calibration["boundary_uncertainty"]
    modified_threshold = float(uncertainty_calibration["calibrated_threshold"])
    cases = {
        "baseline": binary_metrics(baseline_score, known_mask, baseline_threshold),
        "threshold_only": binary_metrics(
            baseline_score, known_mask, modified_threshold
        ),
        "residual_only_same_threshold": binary_metrics(
            uncertainty_score, known_mask, baseline_threshold
        ),
        "full": binary_metrics(uncertainty_score, known_mask, modified_threshold),
    }
    baseline_metrics = cases["baseline"]
    for name, metrics in cases.items():
        if name != "baseline":
            metrics["delta_from_baseline"] = delta_metrics(metrics, baseline_metrics)

    full_report = read_json(uncertainty_dir / "discovery_report.json")
    report_consistency = {
        "full_auroc_abs_diff": abs(cases["full"]["auroc"] - full_report["auroc"]),
        "full_aupr_abs_diff": abs(cases["full"]["aupr"] - full_report["aupr"]),
        "full_known_accept_rate_abs_diff": abs(
            cases["full"]["known_accept_rate"] - full_report["known_accept_rate"]
        ),
        "full_unknown_reject_rate_abs_diff": abs(
            cases["full"]["unknown_reject_rate"] - full_report["unknown_reject_rate"]
        ),
    }
    calibration_report = {
        "baseline_threshold": baseline_threshold,
        "modified_threshold": modified_threshold,
        "threshold_shift": modified_threshold - baseline_threshold,
        "validation_boundary_count": boundary_report.get("validation_boundary_count"),
        "test_boundary_count": boundary_report.get("test_boundary_count"),
        "band_multiplier": boundary_report.get("band_multiplier"),
        "alpha": boundary_report.get("alpha"),
        "score_std": boundary_report.get("score_std"),
        "uncertainty_mean": boundary_report.get("uncertainty_mean"),
        "uncertainty_std": boundary_report.get("uncertainty_std"),
    }
    baseline_decision = baseline_score > baseline_threshold
    modified_decision = uncertainty_score > modified_threshold
    threshold_only_decision = baseline_score > modified_threshold
    residual_only_decision = uncertainty_score > baseline_threshold
    return {
        "seed": seed,
        "uncertainty_suffix": uncertainty_suffix,
        "calibration": calibration_report,
        "cases": cases,
        "raw_score_max_abs_diff": raw_score_max_abs_diff,
        "decision_changes": {
            "threshold_only": int(np.count_nonzero(baseline_decision != threshold_only_decision)),
            "residual_only_same_threshold": int(
                np.count_nonzero(baseline_decision != residual_only_decision)
            ),
            "full": int(np.count_nonzero(baseline_decision != modified_decision)),
        },
        "decision_change_rates": {
            "threshold_only": float(np.mean(baseline_decision != threshold_only_decision)),
            "residual_only_same_threshold": float(
                np.mean(baseline_decision != residual_only_decision)
            ),
            "full": float(np.mean(baseline_decision != modified_decision)),
        },
        "report_consistency": report_consistency,
        "protocol": {
            "test_labels_used_for_selection": False,
            "selection_artifacts": "frozen baseline and uncertainty-band outputs",
            "baseline_score_source": str(baseline_detail_path.relative_to(ROOT)),
            "uncertainty_score_source": str(uncertainty_detail_path.relative_to(ROOT)),
            "baseline_threshold_source": baseline_threshold_source,
            "calibration_source": str(uncertainty_calibration_path.relative_to(ROOT)),
            "calibration_export_sample_count": calibration_sample_count,
        },
    }


def summarize(rows: list[dict]) -> dict:
    case_names = tuple(rows[0]["cases"])
    summary = {"seed_count": len(rows), "cases": {}}
    ddof = 1 if len(rows) > 1 else 0
    for case_name in case_names:
        metrics = rows[0]["cases"][case_name]
        case_summary = {}
        for key in metrics:
            if key == "delta_from_baseline":
                continue
            values = [row["cases"][case_name][key] for row in rows]
            if all(isinstance(value, (int, float)) for value in values):
                case_summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=ddof)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        if case_name != "baseline":
            case_summary["delta_from_baseline"] = {}
            for key in rows[0]["cases"][case_name]["delta_from_baseline"]:
                values = [
                    row["cases"][case_name]["delta_from_baseline"][key]
                    for row in rows
                ]
                case_summary["delta_from_baseline"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=ddof)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "positive_count": int(np.count_nonzero(np.asarray(values) > 0)),
                }
        summary["cases"][case_name] = case_summary
    for key in ("threshold_shift", "validation_boundary_count", "test_boundary_count"):
        values = [row["calibration"][key] for row in rows]
        summary.setdefault("calibration", {})[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=ddof)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    summary["raw_score_max_abs_diff"] = {
        "max": float(max(row["raw_score_max_abs_diff"] for row in rows))
    }
    summary["decision_changes"] = {}
    for key in rows[0]["decision_changes"]:
        values = [row["decision_changes"][key] for row in rows]
        summary["decision_changes"][key] = {
            "mean": float(np.mean(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--uncertainty-suffix",
        default="",
        help="suffix appended to frozen uncertainty-band run directories, e.g. _alpha1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis" / "rejection_uncertainty_band_attribution_6split.json",
    )
    args = parser.parse_args()
    rows = [evaluate_seed(seed, args.uncertainty_suffix) for seed in args.seeds]
    result = {
        "question": "Are boundary-uncertainty gains caused by score re-ranking, threshold recalibration, or both?",
        "protocol": "frozen outputs; four diagnostic cases; test labels used only for final metric reporting",
        "seeds": list(args.seeds),
        "uncertainty_suffix": args.uncertainty_suffix,
        "rows": rows,
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "seeds": args.seeds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
