"""Audit a known-correctness risk gate on the frozen discovery pools.

The previous selective-risk audit only showed that known-validation entropy and
logistic risk can rank classification errors.  This script tests the missing
task-level link: can that risk score itself produce a better unknown candidate
pool than the formal feature-rejector threshold?

The risk model is fitted only from known validation images.  Its threshold is
also selected only from known-validation out-of-fold predictions.  Discovery
labels are read only after all scores and masks have been produced, for final
reporting.  No checkpoint, training run, or default configuration is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(y, score)) if np.unique(y).size > 1 else None


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(average_precision_score(y, score)) if np.unique(y).size > 1 else None


def _fit_model(X: np.ndarray, y: np.ndarray, seed: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed + 1903,
        ),
    )
    model.fit(X, y)
    return model


def _cross_fit_and_fit_full(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, Any, np.ndarray]:
    if np.unique(y).size < 2:
        raise RuntimeError("known validation labels contain only one correctness class")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1901)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for train_idx, valid_idx in splitter.split(X, y):
        model = _fit_model(X[train_idx], y[train_idx], seed)
        oof[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fitted risk predictions contain non-finite values")
    full = _fit_model(X, y, seed)
    coefficients = full[-1].coef_[0].astype(float)
    return oof, full, coefficients


def _load_known_validation(
    formal_run_dir: Path, calibration_run_dir: Path, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Recover known-validation labels and exported signals without test labels."""
    cfg = json.loads((formal_run_dir / "config.json").read_text(encoding="utf-8"))
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
    if bundle.calibration is None:
        raise RuntimeError(f"{formal_run_dir} did not reconstruct a calibration subset")
    with np.load(calibration_run_dir / "calibration_scores.npz", allow_pickle=False) as payload:
        source_index = np.asarray(payload["source_index"], dtype=np.int64)
        predicted = np.asarray(payload["predicted_class"], dtype=np.int64)
        threshold = float(np.asarray(payload["calibrated_threshold"]).reshape(-1)[0])
        signals = {
            name: np.asarray(payload[f"validation_{name}"], dtype=np.float64).reshape(-1)
            for name in FEATURES
        }
    labels = np.asarray([int(bundle.calibration[i][1]) for i in range(len(bundle.calibration))], dtype=np.int64)
    restored_indices = np.asarray(dataset_source_indices(bundle.calibration), dtype=np.int64)
    if not np.array_equal(source_index, restored_indices):
        raise RuntimeError(f"calibration source_index mismatch for seed {seed}")
    if len(labels) != len(predicted):
        raise RuntimeError("restored label count does not match calibration predictions")
    y_error = (predicted != labels).astype(np.int8)
    return y_error, signals, threshold


def _candidate_metrics(
    true_known: np.ndarray,
    pred_unknown: np.ndarray,
    pred_class: np.ndarray,
) -> dict[str, Any]:
    true_known = np.asarray(true_known, dtype=bool)
    pred_unknown = np.asarray(pred_unknown, dtype=bool)
    pred_class = np.asarray(pred_class, dtype=np.int64)
    unknown = ~true_known
    accepted = ~pred_unknown
    known_accepted = true_known & accepted
    return {
        "candidate_count": int(pred_unknown.sum()),
        "candidate_unknown_count": int((pred_unknown & unknown).sum()),
        "candidate_known_contamination_count": int((pred_unknown & true_known).sum()),
        "candidate_purity": float((pred_unknown & unknown).sum() / max(int(pred_unknown.sum()), 1)),
        "candidate_unknown_recall": float((pred_unknown & unknown).sum() / max(int(unknown.sum()), 1)),
        "known_acceptance": float(accepted[true_known].mean()),
        "unknown_rejection": float(pred_unknown[unknown].mean()),
        "unknown_false_accept_count": int((accepted & unknown).sum()),
        "known_rejected_count": int((pred_unknown & true_known).sum()),
    }


def _score_metrics(
    true_known: np.ndarray,
    score: np.ndarray,
    pred_unknown: np.ndarray,
    pred_class: np.ndarray,
) -> dict[str, Any]:
    unknown = (~np.asarray(true_known, dtype=bool)).astype(np.int8)
    result = {
        "auroc": _safe_auc(unknown, score),
        "aupr": _safe_ap(unknown, score),
    }
    result.update(_candidate_metrics(true_known, pred_unknown, pred_class))
    return result


def _known_acceptance_accuracy(
    true_known: np.ndarray,
    true_label: np.ndarray,
    pred_class: np.ndarray,
    pred_unknown: np.ndarray,
) -> float | None:
    true_known = np.asarray(true_known, dtype=bool)
    accepted_known = true_known & ~np.asarray(pred_unknown, dtype=bool)
    if not accepted_known.any():
        return None
    return float((pred_class[accepted_known] == true_label[accepted_known]).mean())


def audit_split(seed: int, formal_run_dir: Path) -> dict[str, Any]:
    calibration_run_dir = ROOT / "runs" / f"rejection_boundary_calibration_s{seed}"
    y_error, validation_signals, formal_threshold = _load_known_validation(
        formal_run_dir, calibration_run_dir, seed
    )
    X = np.column_stack([validation_signals[name] for name in FEATURES])
    oof_risk, full_model, coefficients = _cross_fit_and_fit_full(X, y_error, seed)
    risk_threshold = float(np.quantile(oof_risk, 0.95, method="linear"))

    detail = json.loads((formal_run_dir / "discovery_detail.json").read_text(encoding="utf-8"))
    true_known = np.asarray(detail["true_known"], dtype=bool)
    pred_known_formal = np.asarray(detail["pred_known"], dtype=bool)
    pred_class = np.asarray(detail["pred_class"], dtype=np.int64)
    true_label = np.asarray(detail["true_label"], dtype=np.int64)
    formal_score = np.asarray(detail["feature_rejector_score"], dtype=np.float64)
    discovery_X = np.column_stack([
        np.asarray(detail[name], dtype=np.float64) for name in FEATURES
    ])
    discovery_risk = full_model.predict_proba(discovery_X)[:, 1]

    baseline_unknown = ~pred_known_formal
    risk_unknown = discovery_risk > risk_threshold
    # This is the direct gate interpretation: retain the formal candidates and
    # add formally accepted samples whose known-class risk is high.
    gated_unknown = baseline_unknown | risk_unknown

    baseline = _score_metrics(true_known, formal_score, baseline_unknown, pred_class)
    risk_only = _score_metrics(true_known, discovery_risk, risk_unknown, pred_class)
    gated = _score_metrics(true_known, discovery_risk, gated_unknown, pred_class)
    baseline["accepted_known_accuracy"] = _known_acceptance_accuracy(
        true_known, true_label, pred_class, baseline_unknown
    )
    risk_only["accepted_known_accuracy"] = _known_acceptance_accuracy(
        true_known, true_label, pred_class, risk_unknown
    )
    gated["accepted_known_accuracy"] = _known_acceptance_accuracy(
        true_known, true_label, pred_class, gated_unknown
    )

    def delta(treatment: dict[str, Any]) -> dict[str, float | None]:
        keys = (
            "auroc",
            "aupr",
            "known_acceptance",
            "unknown_rejection",
            "candidate_purity",
            "candidate_unknown_recall",
            "accepted_known_accuracy",
        )
        return {
            key: (treatment[key] - baseline[key])
            if treatment[key] is not None and baseline[key] is not None
            else None
            for key in keys
        }

    return {
        "seed": seed,
        "formal_run": str(formal_run_dir),
        "calibration_run": str(calibration_run_dir),
        "protocol": {
            "known_validation_only_fit_and_threshold": True,
            "unknown_or_discovery_labels_used_only_for_final_reporting": True,
            "risk_features": list(FEATURES),
            "risk_model": "5-fold out-of-fold balanced logistic regression; full known-validation refit for discovery scoring",
            "risk_threshold": "95th percentile of known-validation out-of-fold predicted error risk",
            "gated_definition": "formal reject candidates OR risk above the known-validation risk threshold",
            "clustering_recomputed": False,
        },
        "validation": {
            "sample_count": int(len(y_error)),
            "error_count": int(y_error.sum()),
            "formal_threshold": formal_threshold,
            "risk_threshold": risk_threshold,
            "oof_error_auroc": _safe_auc(y_error, oof_risk),
            "oof_error_aupr": _safe_ap(y_error, oof_risk),
            "coefficients": {name: float(value) for name, value in zip(FEATURES, coefficients)},
        },
        "metrics": {
            "formal_baseline": baseline,
            "risk_only": risk_only,
            "risk_gate_on_formal": gated,
        },
        "deltas_vs_formal": {
            "risk_only": delta(risk_only),
            "risk_gate_on_formal": delta(gated),
        },
    }


def _mean(reports: dict[str, dict[str, Any]], method: str, metric: str) -> float:
    values = [r["metrics"][method][metric] for r in reports.values() if r["metrics"][method][metric] is not None]
    return float(np.mean(values)) if values else float("nan")


def render_markdown(payload: dict[str, Any]) -> str:
    reports = payload["reports"]
    lines = [
        "# Known-correctness risk gate audit",
        "",
        "This audit tests whether a risk model trained on known-validation classification correctness can improve the frozen discovery candidate pool.",
        "The model and its threshold use known validation only. Discovery labels are used only for final metrics; no clustering labels are reused because the candidate pool changes.",
        "",
        "## Four-split mean results",
        "",
        "| method | AUROC | AUPR | known acceptance | unknown rejection | candidate purity | unknown recall | accepted-known accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in (
        ("formal_baseline", "formal feature-rejector"),
        ("risk_only", "risk-only gate"),
        ("risk_gate_on_formal", "formal reject OR risk gate"),
    ):
        lines.append(
            f"| {label} | {_mean(reports, method, 'auroc'):.4f} | {_mean(reports, method, 'aupr'):.4f} | "
            f"{_mean(reports, method, 'known_acceptance'):.4f} | {_mean(reports, method, 'unknown_rejection'):.4f} | "
            f"{_mean(reports, method, 'candidate_purity'):.4f} | {_mean(reports, method, 'candidate_unknown_recall'):.4f} | "
            f"{_mean(reports, method, 'accepted_known_accuracy'):.4f} |"
        )
    lines.extend([
        "",
        "## Per-split unknown rejection and candidate purity",
        "",
        "| split | baseline rejection / purity | risk-only rejection / purity | gated rejection / purity |",
        "|---|---:|---:|---:|",
    ])
    for split, report in reports.items():
        rows = report["metrics"]
        lines.append(
            f"| {split} | {rows['formal_baseline']['unknown_rejection']:.4f} / {rows['formal_baseline']['candidate_purity']:.4f} | "
            f"{rows['risk_only']['unknown_rejection']:.4f} / {rows['risk_only']['candidate_purity']:.4f} | "
            f"{rows['risk_gate_on_formal']['unknown_rejection']:.4f} / {rows['risk_gate_on_formal']['candidate_purity']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The risk-only gate is the fair operating-point comparison because its threshold is calibrated to 95% known-validation acceptance. The formal-reject-or-risk gate shows the direct additive use of risk, but it may lower known acceptance because it adds rejections to the existing detector.",
        "",
        "Clustering is deliberately not reported in this audit: the treatment candidate pool differs from the baseline pool, while discovery_detail.json contains only baseline cluster assignments. A positive candidate-pool result must be followed by a fresh PCA plus KMeans pass on the treatment candidates before any end-to-end claim.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "known_correctness_gate_4split.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "analysis" / "known_correctness_gate_4split.md")
    args = parser.parse_args()
    reports: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"rejection_ce_decay_full_s{seed}_static_control_discover"
        print(f"[audit] s{seed}", flush=True)
        reports[f"s{seed}"] = audit_split(seed, run_dir)
    payload = {"reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(f"saved {args.output}")
    print(f"saved {args.markdown}")


if __name__ == "__main__":
    main()
