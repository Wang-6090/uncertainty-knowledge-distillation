from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metrics.roc_auc_score(y_true, scores))


def compute_aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metrics.average_precision_score(y_true, scores))


def compute_fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y_true = y_true[order]
    positives = (y_true == 1).sum()
    negatives = (y_true == 0).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    tp = 0
    fp = 0
    for label in y_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / positives
        if tpr >= 0.95:
            return float(fp / negatives)
    return 1.0


def compute_oscr(
    known_mask: np.ndarray,
    scores: np.ndarray,
    pred_class: np.ndarray,
    true_label: np.ndarray,
) -> float:
    """Open-set classification rate: CCR vs false-positive unknown rate.

    Unknown samples are ranked by increasing open score. At each operating
    point the closed-set correct classification rate is measured only on
    known samples that remain accepted.
    """
    known_mask = np.asarray(known_mask).astype(bool)
    scores = np.asarray(scores, dtype=float)
    pred_class = np.asarray(pred_class)
    true_label = np.asarray(true_label)
    unknown_scores = scores[~known_mask]
    known_scores = scores[known_mask]
    known_correct = (pred_class[known_mask] == true_label[known_mask]).astype(float)
    n_known = max(int(known_mask.sum()), 1)
    n_unknown = int((~known_mask).sum())
    if n_unknown == 0:
        return float(known_correct.mean()) if n_known else float("nan")

    order = np.argsort(unknown_scores)
    unknown_sorted = unknown_scores[order]
    ccr = []
    fpr = []
    for i, thr in enumerate(unknown_sorted):
        accepted = known_scores <= thr
        ccr.append(float(known_correct[accepted].sum() / n_known))
        fpr.append(float((i + 1) / n_unknown))
    if len(ccr) < 2:
        return float(ccr[0]) if ccr else float("nan")
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(area(ccr, fpr))


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    if confidences.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        ece += abs(correctness[mask].mean() - confidences[mask].mean()) * (mask.mean())
    return float(ece)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((labels_true.size, labels_pred.size), dtype=np.int64)
    for i, t in enumerate(labels_true):
        for j, p in enumerate(labels_pred):
            cost[i, j] = np.sum((y_true == t) & (y_pred == p))
    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    matched = cost[row_ind, col_ind].sum()
    return float(matched / len(y_true))


def clustering_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": clustering_accuracy(y_true, y_pred),
        "nmi": float(metrics.normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(metrics.adjusted_rand_score(y_true, y_pred)),
    }


def detection_cluster_split(
    known_mask: np.ndarray,
    pred_known: np.ndarray,
    novel_true: np.ndarray | None,
    novel_pred: np.ndarray | None,
) -> Dict[str, float]:
    """Separate unknown-detection errors from clustering errors.

    Detection error is the fraction of true unknowns that were never sent to
    clustering. Clustering error is 1 - cluster ACC on the filtered unknown
    set, or NaN when clustering was not run.
    """
    known_mask = np.asarray(known_mask).astype(bool)
    pred_known = np.asarray(pred_known).astype(bool)
    unknown_total = int((~known_mask).sum())
    unknown_missed = int((~known_mask & pred_known).sum())
    known_leaked = int((known_mask & ~pred_known).sum())
    cluster_acc = float("nan")
    if novel_true is not None and novel_pred is not None and len(novel_true) > 0:
        cluster_acc = clustering_accuracy(novel_true, novel_pred)
    return {
        "unknown_detection_miss_rate": float(unknown_missed / max(unknown_total, 1)),
        "known_leak_into_cluster": float(known_leaked / max(int(known_mask.sum()), 1)),
        "cluster_error_on_filtered": float(1.0 - cluster_acc) if cluster_acc == cluster_acc else float("nan"),
        "filtered_unknown_count": int((~pred_known).sum()),
    }


def open_set_confusion(y_true_known: np.ndarray, y_pred_known: np.ndarray):
    y_true_known = np.asarray(y_true_known).astype(bool)
    y_pred_known = np.asarray(y_pred_known).astype(bool)
    tp = int(np.sum(y_true_known & y_pred_known))
    tn = int(np.sum((~y_true_known) & (~y_pred_known)))
    fp = int(np.sum((~y_true_known) & y_pred_known))
    fn = int(np.sum(y_true_known & (~y_pred_known)))
    total = int(len(y_true_known))
    return {
        "known_correct": tp,
        "known_rejected": fn,
        "unknown_correct_reject": tn,
        "unknown_false_accept": fp,
        "total": total,
        "known_accept_rate": float(tp / max(int(y_true_known.sum()), 1)),
        "unknown_reject_rate": float(tn / max(int((~y_true_known).sum()), 1)),
    }
