from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metrics.roc_auc_score(y_true, scores))


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
            return fp / negatives
    return 1.0


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
