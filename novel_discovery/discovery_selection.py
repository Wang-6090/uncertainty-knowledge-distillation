"""Standalone utilities for open-set candidate selection and clustering.

The functions in this module are label-free and deliberately independent from
the training losses.  They are useful for controlled ablations of hard
filtering, soft weighting, neighborhood agreement, and automatic K selection.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def risk_score(
    entropy: Iterable[float],
    msp: Iterable[float] | None = None,
    energy: Iterable[float] | None = None,
    uncertainty: Iterable[float] | None = None,
    mode: str = "consensus",
) -> np.ndarray:
    """Build a higher-is-more-unknown score from detector signals."""
    signals = [np.asarray(entropy, dtype=np.float64)]
    if msp is not None:
        signals.append(np.asarray(msp, dtype=np.float64))
    if energy is not None:
        signals.append(np.asarray(energy, dtype=np.float64))
    if uncertainty is not None:
        signals.append(np.asarray(uncertainty, dtype=np.float64))
    if not signals or any(x.shape != signals[0].shape for x in signals):
        raise ValueError("detector signals must have matching shapes")
    if mode == "entropy":
        return _minmax(signals[0])
    if mode == "consensus":
        return np.mean([_minmax(x) for x in signals], axis=0)
    raise ValueError(f"unsupported score mode: {mode}")


def select_candidates(score: Iterable[float], ratio: float = 0.25, *, soft: bool = False,
                      weight_floor: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic hard mask and optional continuous weights."""
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    n = values.size
    mask = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float64)
    if n == 0 or ratio <= 0:
        return mask, weights
    count = min(n, max(1, int(np.ceil(n * min(float(ratio), 1.0)))))
    order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="stable")
    mask[order[:count]] = True
    if soft:
        normalized = _minmax(values)
        floor = float(np.clip(weight_floor, 0.0, 1.0))
        weights[mask] = floor + (1.0 - floor) * normalized[mask]
    else:
        weights[mask] = 1.0
    return mask, weights


def knn_agreement(mask: Iterable[bool], features: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute selected-neighbor fraction, safely clamping k to batch size."""
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != selected.size:
        raise ValueError("features must be a 2D array aligned with mask")
    n = selected.size
    agreement = np.zeros(n, dtype=np.float64)
    if n <= 1 or k <= 0:
        return agreement
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.maximum(norm, 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(-similarity, kth=min(int(k), n - 1) - 1, axis=1)[:, :min(int(k), n - 1)]
    return selected[neighbors].mean(axis=1)


def auto_k(features: np.ndarray, k_values: Iterable[int] | None = None,
           max_k: int = 20,
           random_state: int = 0, n_init: int = 10) -> tuple[int | None, list[dict]]:
    """Choose K by silhouette, handling tiny/degenerate candidate pools."""
    x = np.asarray(features, dtype=np.float64)
    n = x.shape[0] if x.ndim == 2 else 0
    if n < 3:
        return (1 if n else None), []
    candidates = list(k_values or range(2, min(max(int(max_k), 2), n - 1) + 1))
    diagnostics: list[dict] = []
    best_k, best_value = None, -np.inf
    for k in candidates:
        if k < 2 or k >= n:
            continue
        labels = KMeans(n_clusters=int(k), n_init=n_init, random_state=random_state).fit_predict(x)
        if np.unique(labels).size < 2:
            continue
        value = float(silhouette_score(x, labels))
        diagnostics.append({"k": int(k), "silhouette": value})
        if value > best_value:
            best_k, best_value = int(k), value
    return (best_k or 1), diagnostics


class EMASelector:
    """Mean-Teacher-style exponential moving average for detector signals."""

    def __init__(self, decay: float = 0.99):
        self.decay = float(np.clip(decay, 0.0, 0.999999))
        self.value: np.ndarray | None = None

    def update(self, value: Iterable[float]) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.value is None or self.value.shape != value.shape:
            self.value = value.copy()
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * value
        return self.value.copy()


def filter_by_neighbor_agreement(mask: Iterable[bool], features: np.ndarray,
                                 k: int = 5, min_agreement: float = 0.5) -> np.ndarray:
    """Apply a SCAN-style local consistency gate to a hard candidate mask."""
    mask = np.asarray(mask, dtype=bool)
    agreement = knn_agreement(mask, features, k=k)
    return mask & (agreement >= float(np.clip(min_agreement, 0.0, 1.0)))


def relation_affinity(features: np.ndarray, k: int = 5, temperature: float = 0.2) -> np.ndarray:
    """Build a sparse AutoNovel-style sample relation matrix.

    The matrix contains cosine-similarity weights only for each sample's k
    nearest neighbours, is row-normalized, and uses no labels.  It can be
    consumed by downstream relation/graph clustering ablations.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    n = x.shape[0]
    affinity = np.zeros((n, n), dtype=np.float64)
    if n <= 1 or k <= 0:
        return affinity
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    effective_k = min(int(k), n - 1)
    neighbours = np.argpartition(-similarity, kth=effective_k - 1, axis=1)[:, :effective_k]
    temperature = max(float(temperature), 1e-6)
    for row, indices in enumerate(neighbours):
        values = np.exp(np.clip(similarity[row, indices] / temperature, -60.0, 60.0))
        affinity[row, indices] = values / max(values.sum(), 1e-12)
    return affinity


def evaluate_selection(y_unknown: Iterable[bool], score: Iterable[float],
                       candidate_mask: Iterable[bool], *,
                       cluster_features: np.ndarray | None = None,
                       cluster_labels: Iterable[int] | None = None,
                       cluster_k: str | int = "auto",
                       random_state: int = 0) -> dict:
    """Evaluate detection and candidate/clustering quality in one report.

    Scores are higher for unknown samples. Clustering labels are optional and
    are only used for diagnostics; detection metrics never use unknown labels
    to choose candidates.
    """
    y = np.asarray(y_unknown, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if y.shape != score.shape or y.shape != mask.shape:
        raise ValueError("y_unknown, score, and candidate_mask must align")
    report = {
        "candidate_count": int(mask.sum()),
        "candidate_purity": float(y[mask].mean()) if mask.any() else 0.0,
        "unknown_reject_rate": float((mask & y).sum() / max(int(y.sum()), 1)),
        "auroc": float(roc_auc_score(y, score)) if np.unique(y).size > 1 else float("nan"),
        "aupr": float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan"),
    }
    order = np.argsort(score)[::-1]
    positives, negatives = y.sum(), (~y).sum()
    tp = fp = 0
    fpr95 = 1.0
    if positives and negatives:
        for index in order:
            if y[index]: tp += 1
            else: fp += 1
            if tp / positives >= 0.95:
                fpr95 = fp / negatives
                break
    report["fpr95"] = float(fpr95)
    if cluster_features is not None and cluster_labels is not None and mask.sum() >= 2:
        x = np.asarray(cluster_features, dtype=np.float64)[mask]
        labels = np.asarray(cluster_labels)[mask]
        if cluster_k == "auto":
            chosen, diagnostics = auto_k(x, random_state=random_state)
        else:
            chosen, diagnostics = int(cluster_k), []
        if chosen and chosen >= 2 and chosen < len(x):
            pred = KMeans(n_clusters=chosen, n_init=10, random_state=random_state).fit_predict(x)
            report.update({
                "cluster_k": int(chosen),
                "silhouette": float(silhouette_score(x, pred)),
                "nmi": float(normalized_mutual_info_score(labels, pred)),
                "ari": float(adjusted_rand_score(labels, pred)),
                "cluster_diagnostics": diagnostics,
            })
        else:
            report.update({"cluster_k": chosen, "silhouette": float("nan"),
                           "nmi": float("nan"), "ari": float("nan"),
                           "cluster_diagnostics": diagnostics})
    return report
