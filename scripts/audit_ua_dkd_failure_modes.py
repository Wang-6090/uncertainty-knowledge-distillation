"""Audit frozen UA-DKD failure modes without retraining or selecting a method.

This diagnostic compares the three UA-DKD arms on their existing discovery
artifacts. It uses labels only after the runs finish to describe classification
errors, candidate-pool contamination, unknown-class coverage, and clustering
outcomes. It never changes a checkpoint, threshold, candidate mask, or model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ARMS = ("standard", "decoupled", "uncertainty_decoupled")


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def gini(values: list[int]) -> float:
    xs = np.sort(np.asarray(values, dtype=np.float64))
    total = float(xs.sum())
    if len(xs) == 0 or total == 0.0:
        return 0.0
    n = len(xs)
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * xs) / (n * total))


def finite_auroc(target: np.ndarray, score: np.ndarray) -> float | None:
    valid = np.isfinite(score)
    if valid.sum() < 2 or len(np.unique(target[valid])) < 2:
        return None
    return float(roc_auc_score(target[valid], score[valid]))


def finite_aupr(target: np.ndarray, score: np.ndarray) -> float | None:
    valid = np.isfinite(score)
    if valid.sum() < 2 or len(np.unique(target[valid])) < 2:
        return None
    return float(average_precision_score(target[valid], score[valid]))


def normalized_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def representation_geometry(features: np.ndarray, raw_labels: np.ndarray, known: np.ndarray) -> dict[str, Any]:
    mask = known.astype(bool)
    x = normalized_features(np.asarray(features[mask], dtype=np.float64))
    labels = raw_labels[mask]
    classes = np.unique(labels)
    within: list[float] = []
    centroids: list[np.ndarray] = []
    for label in classes:
        current = x[labels == label]
        if len(current) > 1:
            similarity = current @ current.T
            upper = np.triu_indices(len(current), k=1)
            within.extend((1.0 - similarity[upper]).tolist())
        centroid = current.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    centers = np.asarray(centroids)
    similarity = centers @ centers.T
    upper = np.triu_indices(len(centers), k=1)
    between = (1.0 - similarity[upper]).tolist()
    return {
        "known_samples": int(mask.sum()),
        "class_count": int(len(classes)),
        "within_class_cosine_distance_mean": mean(within),
        "between_class_centroid_cosine_distance_mean": mean(between),
        "between_to_within_distance_ratio": (
            float(np.mean(between) / max(np.mean(within), 1e-12))
            if within and between
            else None
        ),
    }


def load_arm(root: Path, seed: int, arm: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.lib.npyio.NpzFile]:
    if seed == 2132:
        prefix = f"team_a_ua_dkd_s2132_s2132_{arm}"
    else:
        prefix = f"team_a_ua_dkd_confirm_s{seed}_{arm}"
    discover_dir = root / "runs" / f"{prefix}_discover"
    student_dir = root / "runs" / prefix
    detail = json.loads((discover_dir / "discovery_detail.json").read_text(encoding="utf-8"))
    report = json.loads((discover_dir / "discovery_report.json").read_text(encoding="utf-8"))
    config = json.loads((student_dir / "train_student_config.json").read_text(encoding="utf-8"))
    inputs = np.load(discover_dir / "feature_rejector_inputs.npz", allow_pickle=False)
    return detail, report, config, inputs


def audit_arm(root: Path, seed: int, arm: str) -> dict[str, Any]:
    detail, report, config, inputs = load_arm(root, seed, arm)
    known = np.asarray(detail["true_known"], dtype=bool)
    pred_known = np.asarray(detail["pred_known"], dtype=bool)
    raw_labels = np.asarray(detail["raw_labels"], dtype=int)
    pred_class = np.asarray(detail["pred_class"], dtype=int)
    true_label = np.asarray(detail["true_label"], dtype=int)
    score = np.asarray(detail["score"], dtype=np.float64)
    if not (len(known) == len(pred_known) == len(raw_labels) == len(pred_class) == len(true_label)):
        raise ValueError(f"unaligned discovery detail arrays for seed {seed}, arm {arm}")

    known_correct = known & (pred_class == true_label)
    candidates = ~pred_known
    unknown = ~known
    unknown_labels = np.unique(raw_labels[unknown])
    coverage = []
    false_rejects = []
    unknown_rejections = []
    for label in unknown_labels:
        class_mask = unknown & (raw_labels == label)
        coverage.append(float((candidates & class_mask).sum() / max(class_mask.sum(), 1)))
    known_labels = np.unique(raw_labels[known])
    known_error_rates = []
    for label in known_labels:
        class_mask = known & (raw_labels == label)
        known_error_rates.append(float((~known_correct & class_mask).sum() / max(class_mask.sum(), 1)))

    component_auroc: dict[str, float | None] = {}
    component_aupr: dict[str, float | None] = {}
    for key in (
        "score", "raw_score", "entropy", "epistemic", "aleatoric",
        "head_uncertainty", "feature_rejector_score", "known_support_score",
        "proto_dist", "mahalanobis", "knn_distance", "vim_residual",
    ):
        if key not in detail:
            continue
        values = np.asarray(detail[key], dtype=np.float64)
        component_auroc[key] = finite_auroc(unknown.astype(int), values)
        component_aupr[key] = finite_aupr(unknown.astype(int), values)

    features = np.asarray(inputs["test_features"], dtype=np.float64)
    if len(features) != len(known):
        raise ValueError(f"feature/detail length mismatch for seed {seed}, arm {arm}")

    metrics = report
    return {
        "seed": seed,
        "arm": arm,
        "best_epoch": config.get("best_epoch"),
        "best_val_known_acc": config.get("best_val_known_acc"),
        "test_known_accuracy": float(known_correct[known].mean()),
        "test_known_error_rate": float((~known_correct[known]).mean()),
        "known_false_reject_rate": float((known & candidates).sum() / max(known.sum(), 1)),
        "known_accept_rate": float(pred_known[known].mean()),
        "accepted_known_accuracy": float(known_correct[known & pred_known].mean())
        if (known & pred_known).any()
        else None,
        "unknown_rejection_rate": float((unknown & candidates).sum() / max(unknown.sum(), 1)),
        "candidate_count": int(candidates.sum()),
        "candidate_purity": float((unknown & candidates).sum() / max(candidates.sum(), 1)),
        "unknown_class_coverage": {
            "min": float(np.min(coverage)),
            "q25": quantile(coverage, 0.25),
            "median": quantile(coverage, 0.5),
            "q75": quantile(coverage, 0.75),
            "max": float(np.max(coverage)),
            "below_20pct": int(sum(value < 0.2 for value in coverage)),
            "gini": gini([int((candidates & unknown & (raw_labels == label)).sum()) for label in unknown_labels]),
        },
        "known_class_error_rate": {
            "min": float(np.min(known_error_rates)),
            "median": quantile(known_error_rates, 0.5),
            "max": float(np.max(known_error_rates)),
            "above_30pct": int(sum(value > 0.3 for value in known_error_rates)),
        },
        "reported": {
            "auroc": metrics.get("auroc"),
            "aupr": metrics.get("aupr"),
            "oscr": metrics.get("oscr"),
            "cluster_k": metrics.get("cluster_k"),
            "cluster_acc": metrics.get("cluster_acc"),
            "cluster_nmi": metrics.get("cluster_nmi"),
            "cluster_ari": metrics.get("cluster_ari"),
        },
        "component_unknown_auroc": component_auroc,
        "component_unknown_aupr": component_aupr,
        "representation_geometry_known_test": representation_geometry(features, raw_labels, known),
        "labels_used_only_for_posthoc_diagnostic": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2132, 2133, 2134])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/team_a_ua_dkd_failure_modes.json"),
    )
    args = parser.parse_args()
    results = [audit_arm(args.root, seed, arm) for seed in args.seeds for arm in ARMS]
    payload = {
        "purpose": "frozen UA-DKD failure-mode diagnostic; no training or method selection",
        "seeds": args.seeds,
        "arms": ARMS,
        "results": results,
    }
    output = args.output if args.output.is_absolute() else args.root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
