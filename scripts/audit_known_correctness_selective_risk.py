"""Audit a known-correctness selective-risk model without touching test data.

The frozen rejection runs export several label-free validation signals.  This
script restores the known validation labels from the CIFAR-100 protocol only
to define whether the student's known-class prediction was correct.  It then
uses deterministic out-of-fold predictions to test whether a small
regularized risk model can rank classification errors better than the already
available single signals.

No unknown/test labels are used for fitting, calibration, model selection, or
threshold choice.  The audit does not modify a training run or promote a new
default configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, dataset_source_indices  # noqa: E402


SEEDS = (2121, 2124, 2125, 2127)
FEATURES = (
    "feature_rejector_score",
    "head_uncertainty",
    "entropy",
)
BASELINES = {
    "formal_score": "feature_rejector_score",
    "head_uncertainty": "head_uncertainty",
    "entropy": "entropy",
}


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, score))


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y).size < 2:
        return None
    return float(average_precision_score(y, score))


def _safe_log_loss(y: np.ndarray, probability: np.ndarray) -> float | None:
    if np.unique(y).size < 2:
        return None
    return float(log_loss(y, np.clip(probability, 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def _cohort_metrics(y: np.ndarray, score: np.ndarray, accepted: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "all_count": int(len(y)),
        "all_auroc": _safe_auc(y, score),
        "all_aupr": _safe_ap(y, score),
        "accepted_count": int(accepted.sum()),
        "accepted_error_rate": float(y[accepted].mean()) if accepted.any() else None,
        "accepted_auroc": _safe_auc(y[accepted], score[accepted]) if accepted.any() else None,
        "accepted_aupr": _safe_ap(y[accepted], score[accepted]) if accepted.any() else None,
    }
    return rows


def _coverage_error(y: np.ndarray, score: np.ndarray, coverage: float) -> float | None:
    if len(y) == 0:
        return None
    keep = max(1, min(len(y), int(np.ceil(len(y) * coverage))))
    order = np.argsort(score, kind="mergesort")
    return float(y[order[:keep]].mean())


def _restore_known_validation(seed: int, run_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    split_path = Path(cfg["split_path"])
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if not split_path.exists():
        split_path = ROOT / "splits" / "semantic_hard" / f"seed{seed}" / "cifar100_60_40_semantic_hard.json"
    data_root = Path(cfg.get("data_root", "data"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    bundle = build_data_bundle(
        "cifar100",
        str(data_root),
        int(cfg["num_known"]),
        int(cfg["seed"]),
        int(cfg["image_size"]),
        split_path=str(split_path),
        discovery_pool_mode=str(cfg.get("discovery_pool_mode", "mixed")),
        mixed_known_pool_ratio=float(cfg.get("mixed_known_pool_ratio", 0.2)),
        matched_discovery_pool=bool(cfg.get("matched_discovery_pool", True)),
        matched_pool_size=int(cfg.get("matched_pool_size", 5400)),
        matched_known_prior=float(cfg.get("matched_known_prior", 0.2)),
        calibration_ratio=float(cfg.get("calibration_ratio", 0.25)),
        transform_profile=str(cfg.get("transform_profile", "cifar")),
    )
    score_path = run_dir / "calibration_scores.npz"
    with np.load(score_path, allow_pickle=False) as payload:
        required = {"source_index", "predicted_class", "calibrated_threshold", *[f"validation_{x}" for x in FEATURES]}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{score_path} missing arrays: {sorted(missing)}")
        source_index = np.asarray(payload["source_index"], dtype=np.int64)
        predicted = np.asarray(payload["predicted_class"], dtype=np.int64)
        threshold = float(np.asarray(payload["calibrated_threshold"]).reshape(-1)[0])
        feature_values = {
            name: np.asarray(payload[f"validation_{name}"], dtype=np.float64).reshape(-1)
            for name in FEATURES
        }
    if bundle.calibration is None:
        raise RuntimeError(f"{run_dir} did not reconstruct a calibration subset")
    mapped_labels = np.asarray([int(bundle.calibration[i][1]) for i in range(len(bundle.calibration))], dtype=np.int64)
    restored_indices = np.asarray(dataset_source_indices(bundle.calibration), dtype=np.int64)
    if not np.array_equal(source_index, restored_indices):
        raise RuntimeError(f"calibration source_index mismatch for seed {seed}")
    if len(mapped_labels) != len(predicted):
        raise RuntimeError("restored label count does not match exported predictions")
    y_error = (predicted != mapped_labels).astype(np.int8)
    signals = {name: feature_values[name] for name in FEATURES}
    signals["accepted_margin"] = threshold - feature_values["feature_rejector_score"]
    accepted = feature_values["feature_rejector_score"] <= threshold
    return cfg, y_error, accepted, signals


def _cross_fit_risk_model(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(y).size < 2:
        raise RuntimeError("known validation labels contain only one correctness class")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1901)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for train_idx, valid_idx in splitter.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=seed + 1902,
            ),
        )
        model.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fitted risk predictions contain non-finite values")
    # Fit a separate full-data model only to expose coefficients for diagnosis;
    # all reported metrics use the out-of-fold probabilities above.
    full_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=seed + 1903),
    )
    full_model.fit(X, y)
    coefficients = full_model[-1].coef_[0].astype(float)
    return oof, coefficients


def audit_split(seed: int, run_dir: Path) -> dict[str, Any]:
    cfg, y, accepted, signals = _restore_known_validation(seed, run_dir)
    X = np.column_stack([signals[name] for name in FEATURES])
    risk, coefficients = _cross_fit_risk_model(X, y, seed)
    scores = {name: signals[signal_name] for name, signal_name in BASELINES.items()}
    scores["selective_risk_logistic"] = risk
    metrics = {
        name: _cohort_metrics(y, score, accepted)
        for name, score in scores.items()
    }
    coverage = {}
    for name, score in scores.items():
        coverage[name] = {
            str(int(level * 100)): _coverage_error(y, score, level)
            for level in (0.80, 0.90, 0.95)
        }
    return {
        "protocol": {
            "seed": seed,
            "source_run": str(run_dir),
            "known_validation_only": True,
            "test_or_unknown_labels_used_for_fit_or_selection": False,
            "risk_model": "5-fold out-of-fold standardization plus balanced logistic regression",
            "risk_model_features": list(FEATURES),
            "accepted_definition": "feature_rejector_score <= frozen known-validation threshold",
            "reported_model_metrics_are_out_of_fold": True,
        },
        "sample_counts": {
            "known_validation": int(len(y)),
            "known_validation_errors": int(y.sum()),
            "known_validation_correct": int((1 - y).sum()),
            "formal_accepted": int(accepted.sum()),
            "formal_accepted_errors": int(y[accepted].sum()),
        },
        "metrics": metrics,
        "error_rate_at_coverage": coverage,
        "full_data_diagnostic_coefficients": {
            name: float(value) for name, value in zip(FEATURES, coefficients)
        },
        "threshold": float(cfg.get("target_known_coverage", 0.95)),
    }


def _mean_metric(reports: dict[str, dict[str, Any]], metric: str, cohort: str) -> float:
    values = [row[metric] for report in reports.values() for row in [report["metrics"][cohort]] if row[metric] is not None]
    return float(np.mean(values)) if values else float("nan")


def render_markdown(payload: dict[str, Any]) -> str:
    reports = payload["reports"]
    lines = [
        "# Known-correctness selective-risk audit",
        "",
        "This is a frozen, validation-only audit on the four formal semantic-hard runs.",
        "The target is whether the student's known-class prediction is correct. A five-fold out-of-fold logistic model is trained only on known validation samples; unknown/test labels are not used for fitting, model selection, or threshold choice.",
        "",
        "## What was tested",
        "",
        "The formal rejector already accepts a large cohort containing both correctly accepted known samples and unknown false accepts. This audit asks a narrower question: among known validation samples, can a learned selective-risk score rank classification errors better than any one existing signal? A positive score means higher predicted classification-error risk.",
        "",
        "## Four-split mean metrics",
        "",
        "| risk score | all known AUROC | accepted-cohort AUROC | all known AUPR | accepted-cohort error rate |",
        "|---|---:|---:|---:|---:|",
    ]
    names = ("formal_score", "head_uncertainty", "entropy", "selective_risk_logistic")
    for name in names:
        lines.append(
            f"| {name} | {_mean_metric(reports, 'all_auroc', name):.4f} | {_mean_metric(reports, 'accepted_auroc', name):.4f} | {_mean_metric(reports, 'all_aupr', name):.4f} | {_mean_metric(reports, 'accepted_error_rate', name):.4f} |"
        )
    lines.extend([
        "",
        "## Per-split accepted-cohort AUROC",
        "",
        "| split | formal score | head uncertainty | entropy | logistic risk |",
        "|---|---:|---:|---:|---:|",
    ])
    for split, report in reports.items():
        lines.append(
            f"| {split} | {report['metrics']['formal_score']['accepted_auroc']:.4f} | {report['metrics']['head_uncertainty']['accepted_auroc']:.4f} | {report['metrics']['entropy']['accepted_auroc']:.4f} | {report['metrics']['selective_risk_logistic']['accepted_auroc']:.4f} |"
        )
    lines.extend([
        "",
        "## Error rate after retaining the lowest-risk known samples",
        "",
        "| risk score | 80% coverage | 90% coverage | 95% coverage |",
        "|---|---:|---:|---:|",
    ])
    for name in names:
        means = []
        for level in ("80", "90", "95"):
            values = [report["error_rate_at_coverage"][name][level] for report in reports.values()]
            means.append(float(np.mean(values)))
        lines.append(f"| {name} | {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} |")
    lines.extend([
        "",
        "## Decision",
        "",
        "This audit justifies a training experiment only if the logistic risk model has accepted-cohort AUROC above 0.5 in all four splits and improves on the strongest single-signal baseline in a consistent way. If that condition is not met, the direction is stopped here and no training code is changed.",
        "",
        "The audit does not claim that a known-class correctness model solves unknown detection: known validation correctness is a proxy for selective classification risk. A later training experiment would still need to evaluate unknown rejection, known acceptance, candidate purity/coverage, and clustering ACC/NMI/ARI on the unchanged formal protocol.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "analysis" / "known_correctness_selective_risk_4split.json"))
    parser.add_argument("--markdown", default=str(ROOT / "analysis" / "known_correctness_selective_risk_4split.md"))
    args = parser.parse_args()
    reports: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"rejection_boundary_calibration_s{seed}"
        print(f"[audit] s{seed}", flush=True)
        reports[f"s{seed}"] = audit_split(seed, run_dir)
    payload = {"reports": reports}
    output = Path(args.output)
    markdown = Path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(f"saved {output}")
    print(f"saved {markdown}")


if __name__ == "__main__":
    main()
