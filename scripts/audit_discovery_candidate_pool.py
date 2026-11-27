"""Audit mixed-pool candidate contamination on one frozen student checkpoint.

This is a diagnostic only.  It extracts the matched mixed discovery pool with
deterministic evaluation transforms, applies the candidate selectors used by
the training code, and reports known contamination and unknown recall.  Labels
are used only to write the audit report; they do not affect selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import ConcatDataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
from novel_discovery.models import build_model  # noqa: E402
from novel_discovery.pipeline import (  # noqa: E402
    build_loader,
    collect_classwise_knn_support,
    select_discovery_candidates,
    select_support_boundary_candidates,
    support_boundary_candidate_scores,
    select_support_distance_candidates,
    select_prototype_boundary_candidates,
    fit_known_support_rejector,
)
from train import load_checkpoint, parse_args, resolve_device  # noqa: E402


def clone_eval(dataset, image_size: int, profile: str):
    """Clone common dataset wrappers while replacing only the transform."""
    if isinstance(dataset, Subset):
        cloned = copy.copy(dataset)
        cloned.dataset = clone_eval(dataset.dataset, image_size, profile)
        return cloned
    if isinstance(dataset, ConcatDataset):
        return ConcatDataset(
            [clone_eval(item, image_size, profile) for item in dataset.datasets]
        )
    if hasattr(dataset, "base"):
        cloned = copy.copy(dataset)
        cloned.base = copy.copy(dataset.base)
        cloned.base.transform = build_transforms(
            image_size, train=False, profile=profile
        )
        return cloned
    raise TypeError(f"unsupported dataset wrapper: {type(dataset)!r}")


@torch.no_grad()
def extract_deterministic_audit_outputs(model, loader, device) -> dict[str, np.ndarray]:
    """Extract one deterministic eval-mode forward pass per image.

    The shared ``extract_outputs`` helper performs MC dropout even when
    ``mc_samples=1``. Candidate-mechanism audits need repeatable scores, so
    this helper deliberately uses only the ordinary eval-mode forward pass.
    """
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "logits", "head_uncertainty", "features", "rejection_features",
            "rejection_logits", "labels", "raw_labels", "is_known",
        )
    }
    for images, labels, raw_labels, is_known, _ in loader:
        output = model(images.to(device))
        chunks["logits"].append(output["logits"].cpu())
        chunks["head_uncertainty"].append(output["uncertainty"].cpu())
        chunks["features"].append(output["features"].cpu())
        if output.get("rejection_features") is not None:
            chunks["rejection_features"].append(output["rejection_features"].cpu())
        if output.get("rejection_logits") is not None:
            chunks["rejection_logits"].append(output["rejection_logits"].cpu())
        chunks["labels"].append(labels.cpu())
        chunks["raw_labels"].append(raw_labels.cpu())
        chunks["is_known"].append(is_known.cpu())
    return {
        key: torch.cat(values).numpy()
        for key, values in chunks.items()
        if values
    }


@torch.no_grad()
def extract_mc_audit_scores(
    model, loader, device, mc_samples: int, mc_seed: int
) -> dict[str, np.ndarray]:
    """Estimate predictive entropy and epistemic MI with stochastic dropout."""
    if mc_samples < 2:
        raise ValueError("mc_samples must be at least 2")
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("predictive_entropy", "epistemic", "head_uncertainty")
    }
    cuda_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(mc_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(mc_seed))
        for images, *_ in loader:
            scores = model.mc_predict(images.to(device), mc_samples=mc_samples)
            for key in chunks:
                chunks[key].append(scores[key].cpu())
    return {key: torch.cat(values).numpy() for key, values in chunks.items()}


def fixed_budget_top_mask(scores: np.ndarray, ratio: float) -> np.ndarray:
    """Select the largest scores with deterministic tie handling."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be nonempty and finite")
    budget = min(len(values), max(1, int(np.ceil(len(values) * ratio))))
    order = np.argsort(-values, kind="stable")
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:budget]] = True
    return mask


def fixed_budget_rank_fusion(
    first: np.ndarray, second: np.ndarray, ratio: float, first_weight: float = 0.5
) -> np.ndarray:
    """Fuse heterogeneous scores by fixed within-pool percentile ranks."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not len(left):
        raise ValueError("rank-fusion score arrays must be nonempty and aligned")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("rank-fusion scores must be finite")
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first_weight must be in [0, 1]")

    def percentile_rank(values):
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, len(values))
        return ranks

    fused = first_weight * percentile_rank(left) + (1.0 - first_weight) * percentile_rank(right)
    return fixed_budget_top_mask(fused, ratio)


def validation_uncertainty_error_diagnostics(outputs: dict[str, np.ndarray]) -> dict:
    """Measure whether the frozen uncertainty head predicts known-class errors.

    This uses only the known validation labels.  It is a calibration/ranking
    diagnostic, not a selector and not a source of training or test-time
    tuning.  A useful uncertainty signal should assign larger values to
    validation misclassifications than to correct classifications.
    """
    labels = np.asarray(outputs["labels"], dtype=int).reshape(-1)
    logits = np.asarray(outputs["logits"], dtype=float)
    uncertainty = np.asarray(outputs["head_uncertainty"], dtype=float).reshape(-1)
    valid = (labels >= 0) & np.isfinite(uncertainty)
    labels = labels[valid]
    logits = logits[valid]
    uncertainty = uncertainty[valid]
    errors = (logits.argmax(axis=1) != labels).astype(np.int8)
    result = {
        "sample_count": int(len(errors)),
        "error_rate": float(errors.mean()) if len(errors) else None,
        "uncertainty_mean": float(uncertainty.mean()) if len(uncertainty) else None,
        "uncertainty_std": float(uncertainty.std()) if len(uncertainty) else None,
        "labels_used_only_for_validation_diagnostic": True,
    }
    if len(np.unique(errors)) < 2 or len(errors) < 2:
        result.update({"error_auroc": None, "error_aupr": None, "error_correlation": None})
        return result
    result["error_auroc"] = float(roc_auc_score(errors, uncertainty))
    result["error_aupr"] = float(average_precision_score(errors, uncertainty))
    result["error_correlation"] = float(np.corrcoef(errors.astype(float), uncertainty)[0, 1])
    order = np.argsort(uncertainty, kind="stable")
    low = errors[order[: max(1, len(errors) // 5)]]
    high = errors[order[-max(1, len(errors) // 5):]]
    result["lowest_uncertainty_quintile_error_rate"] = float(low.mean())
    result["highest_uncertainty_quintile_error_rate"] = float(high.mean())
    result["high_minus_low_quintile_error_rate"] = float(high.mean() - low.mean())
    return result


def load_config(path: Path, student_ckpt: Path):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args(["discover"])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.student_ckpt = str(student_ckpt)
    return args


def summarize(mask: np.ndarray, is_known: np.ndarray, total_unknown: int) -> dict:
    selected = int(mask.sum())
    selected_known = int((mask & is_known).sum())
    selected_unknown = int((mask & ~is_known).sum())
    return {
        "selected": selected,
        "selected_ratio": float(selected / max(len(mask), 1)),
        "known_contamination": float(selected_known / max(selected, 1)),
        "unknown_purity": float(selected_unknown / max(selected, 1)),
        "unknown_recall": float(selected_unknown / max(total_unknown, 1)),
    }


def summarize_classwise(
    mask: np.ndarray, raw_labels: np.ndarray, is_known: np.ndarray
) -> dict:
    """Report selector coverage by class, for retrospective audit only."""
    if mask.ndim != 1 or raw_labels.ndim != 1 or is_known.ndim != 1:
        raise ValueError("mask, raw_labels and is_known must be one-dimensional")
    if not (len(mask) == len(raw_labels) == len(is_known)):
        raise ValueError("mask, raw_labels and is_known must have equal length")

    novel_recall = {}
    known_selection_rate = {}
    for raw_label in np.unique(raw_labels):
        class_mask = raw_labels == raw_label
        selected_count = int((mask & class_mask).sum())
        class_count = int(class_mask.sum())
        if bool(is_known[class_mask][0]):
            known_selection_rate[str(int(raw_label))] = {
                "selected": selected_count,
                "total": class_count,
                "selection_rate": float(selected_count / max(class_count, 1)),
            }
        else:
            novel_recall[str(int(raw_label))] = {
                "selected": selected_count,
                "total": class_count,
                "recall": float(selected_count / max(class_count, 1)),
            }
    recalls = [item["recall"] for item in novel_recall.values()]
    return {
        "novel_recall_by_raw_label": novel_recall,
        "known_selection_rate_by_raw_label": known_selection_rate,
        "novel_recall_summary": {
            "classes": len(recalls),
            "min": float(min(recalls)) if recalls else 0.0,
            "median": float(np.median(recalls)) if recalls else 0.0,
            "max": float(max(recalls)) if recalls else 0.0,
            "zero_recall_classes": int(sum(value == 0.0 for value in recalls)),
        },
    }


def select_class_balanced_candidates(
    logits: torch.Tensor,
    uncertainty: torch.Tensor | None,
    ratio: float,
) -> torch.Tensor:
    """Offline diagnostic: allocate a fixed risk budget across predicted classes."""
    batch_size, num_classes = logits.shape
    if uncertainty is not None and uncertainty.shape != (batch_size,):
        raise ValueError("uncertainty must have one value per logit row")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    with torch.no_grad():
        probabilities = logits.softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        risk = entropy if uncertainty is None else entropy + uncertainty
        predicted_class = logits.argmax(dim=-1)
        budget = min(batch_size, max(1, int(np.ceil(batch_size * ratio))))
        quota = budget // num_classes
        mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
        if quota:
            for class_id in range(num_classes):
                indices = torch.nonzero(predicted_class == class_id, as_tuple=False).flatten()
                take = min(quota, int(indices.numel()))
                if take:
                    chosen = indices[torch.topk(risk[indices], k=take).indices]
                    mask[chosen] = True
        remaining = budget - int(mask.sum().item())
        if remaining:
            fill_scores = risk.masked_fill(mask, float("-inf"))
            fill = torch.topk(fill_scores, k=remaining, largest=True).indices
            mask[fill] = True
    return mask


def select_knn_vote_entropy_candidates(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    pool_features: np.ndarray,
    ratio: float = 0.25,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank pool samples by entropy of labels among known-support cosine kNNs.

    This is a frozen-feature diagnostic only. The support labels define the
    known-class neighborhoods; no pool labels are accepted by this function.
    Stable ordering makes the fixed-budget selection repeatable under ties.
    """
    support = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels)
    pool = np.asarray(pool_features, dtype=np.float32)
    if support.ndim != 2 or pool.ndim != 2:
        raise ValueError("support_features and pool_features must be 2-D")
    if support.shape[1] != pool.shape[1] or support.shape[1] == 0:
        raise ValueError("support and pool feature dimensions must match and be nonempty")
    if len(support) == 0 or len(pool) == 0:
        raise ValueError("support and pool feature matrices must be nonempty")
    if labels.ndim != 1 or len(labels) != len(support):
        raise ValueError("support_labels must align with support_features")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("support_labels must be nonnegative integer class ids")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if k < 1:
        raise ValueError("k must be positive")
    num_classes = int(labels.max()) + 1
    support_norms = np.linalg.norm(support, axis=1, keepdims=True)
    pool_norms = np.linalg.norm(pool, axis=1, keepdims=True)
    if np.any(support_norms <= 1e-12) or np.any(pool_norms <= 1e-12):
        raise ValueError("features must have nonzero norm")
    support = support / support_norms
    pool = pool / pool_norms
    neighbor_count = min(int(k), len(support))
    entropy = np.empty(len(pool), dtype=np.float64)
    for start in range(0, len(pool), 512):
        stop = min(start + 512, len(pool))
        similarities = pool[start:stop] @ support.T
        neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbor_count]
        for row, indices in enumerate(neighbors):
            votes = np.bincount(labels[indices].astype(np.int64), minlength=num_classes)
            probabilities = votes[votes > 0] / neighbor_count
            entropy[start + row] = -np.sum(probabilities * np.log(probabilities))
    budget = min(len(pool), max(1, int(np.ceil(len(pool) * ratio))))
    order = np.argsort(-entropy, kind="stable")
    mask = np.zeros(len(pool), dtype=bool)
    mask[order[:budget]] = True
    return mask, entropy


def select_knn_classifier_disagreement_candidates(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    pool_features: np.ndarray,
    pool_logits: np.ndarray,
    ratio: float = 0.25,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank samples whose classifier class disagrees with known-support kNN."""
    support = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels)
    pool = np.asarray(pool_features, dtype=np.float32)
    logits = np.asarray(pool_logits)
    if support.ndim != 2 or pool.ndim != 2 or logits.ndim != 2:
        raise ValueError("support_features, pool_features and pool_logits must be 2-D")
    if support.shape[1] != pool.shape[1] or support.shape[1] == 0:
        raise ValueError("support and pool feature dimensions must match and be nonempty")
    if len(support) == 0 or len(pool) == 0 or len(logits) != len(pool):
        raise ValueError("support and pool arrays must be nonempty and align")
    if labels.ndim != 1 or len(labels) != len(support):
        raise ValueError("support_labels must align with support_features")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("support_labels must be nonnegative integer class ids")
    if int(labels.max()) >= logits.shape[1]:
        raise ValueError("support_labels must be within pool_logits class range")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if k < 1:
        raise ValueError("k must be positive")
    support_norms = np.linalg.norm(support, axis=1, keepdims=True)
    pool_norms = np.linalg.norm(pool, axis=1, keepdims=True)
    if np.any(support_norms <= 1e-12) or np.any(pool_norms <= 1e-12):
        raise ValueError("features must have nonzero norm")
    support = support / support_norms
    pool = pool / pool_norms
    neighbor_count = min(int(k), len(support))
    predicted = logits.argmax(axis=1)
    disagreement = np.empty(len(pool), dtype=np.float64)
    for start in range(0, len(pool), 512):
        stop = min(start + 512, len(pool))
        similarities = pool[start:stop] @ support.T
        neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbor_count]
        for row, indices in enumerate(neighbors):
            neighbor_match = labels[indices] == predicted[start + row]
            disagreement[start + row] = 1.0 - float(np.mean(neighbor_match))
    budget = min(len(pool), max(1, int(np.ceil(len(pool) * ratio))))
    order = np.argsort(-disagreement, kind="stable")
    mask = np.zeros(len(pool), dtype=bool)
    mask[order[:budget]] = True
    return mask, disagreement


def support_distance_scores(
    features: np.ndarray, support_bank: dict, k: int = 5
) -> np.ndarray:
    """Return minimum classwise kNN distance relative to known support radius."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or len(values) == 0:
        raise ValueError("features must be a nonempty 2-D matrix")
    banks = support_bank.get("features_by_class")
    radii = support_bank.get("radii")
    if not banks or radii is None or len(banks) != len(radii):
        raise ValueError("support_bank must contain class features and matching radii")
    if k < 1:
        raise ValueError("k must be positive")
    values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)
    class_scores = []
    for class_id, raw_bank in enumerate(banks):
        bank = (
            raw_bank.detach().cpu().numpy().astype(np.float32, copy=False)
            if isinstance(raw_bank, torch.Tensor)
            else np.asarray(raw_bank, dtype=np.float32)
        )
        if bank.size == 0:
            continue
        if bank.ndim != 2 or bank.shape[1] != values.shape[1]:
            raise ValueError("support bank feature dimension must match query features")
        bank = bank / np.clip(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8, None)
        neighbor_count = min(int(k), len(bank))
        distance = 1.0 - np.partition(values @ bank.T, -neighbor_count, axis=1)[:, -neighbor_count:].mean(axis=1)
        radius_value = radii[class_id]
        if isinstance(radius_value, torch.Tensor):
            radius_value = radius_value.detach().cpu().item()
        radius = max(float(radius_value), 1e-4)
        class_scores.append((distance - radius) / radius)
    if not class_scores:
        raise ValueError("support bank contains no non-empty class bank")
    return np.min(np.stack(class_scores, axis=1), axis=1)


def select_dual_representation_candidates(
    shared_scores: np.ndarray,
    rejection_scores: np.ndarray,
    ratio: float = 0.25,
    fusion_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Select by fixed rank fusion of shared and rejection support distances."""
    shared = np.asarray(shared_scores, dtype=np.float64)
    rejection = np.asarray(rejection_scores, dtype=np.float64)
    if shared.ndim != 1 or rejection.ndim != 1 or len(shared) != len(rejection):
        raise ValueError("shared and rejection scores must be aligned 1-D arrays")
    if not len(shared) or not np.isfinite(shared).all() or not np.isfinite(rejection).all():
        raise ValueError("scores must be nonempty and finite")
    if not 0.0 < ratio <= 1.0 or not 0.0 <= fusion_weight <= 1.0:
        raise ValueError("ratio must be in (0, 1] and fusion_weight in [0, 1]")

    def rank(scores):
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, len(scores))
        return ranks

    fused = float(fusion_weight) * rank(shared) + (1.0 - float(fusion_weight)) * rank(rejection)
    budget = min(len(fused), max(1, int(np.ceil(len(fused) * ratio))))
    selected = np.argsort(-fused, kind="stable")[:budget]
    mask = np.zeros(len(fused), dtype=bool)
    mask[selected] = True
    return mask, fused


def select_shared_support_veto_candidates(
    shared_scores: np.ndarray,
    rejection_scores: np.ndarray,
    ratio: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """Prioritize samples outside shared known support, then rank by rejection distance."""
    shared = np.asarray(shared_scores, dtype=np.float64)
    rejection = np.asarray(rejection_scores, dtype=np.float64)
    if shared.ndim != 1 or rejection.ndim != 1 or len(shared) != len(rejection):
        raise ValueError("shared and rejection scores must be aligned 1-D arrays")
    if not len(shared) or not np.isfinite(shared).all() or not np.isfinite(rejection).all():
        raise ValueError("scores must be nonempty and finite")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    budget = min(len(shared), max(1, int(np.ceil(len(shared) * ratio))))
    outside_support = shared > 0.0
    # Lexicographic ordering: shared-support eligibility is the hard primary
    # key; rejection-space distance ranks examples within each group.
    order = np.lexsort((-rejection, ~outside_support))
    selected = order[:budget]
    mask = np.zeros(len(shared), dtype=bool)
    mask[selected] = True
    return mask, {
        "shared_outside_support_pool_count": int(outside_support.sum()),
        "selected_outside_shared_support": int(outside_support[mask].sum()),
        "selected_count": int(mask.sum()),
    }


def select_validation_classwise_percentile_candidates(
    validation_scores: np.ndarray,
    validation_logits: np.ndarray,
    pool_scores: np.ndarray,
    pool_logits: np.ndarray,
    ratio: float = 0.25,
    min_class_samples: int = 20,
    validation_is_known: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrate support-distance scores using only known validation examples."""
    val_scores = np.asarray(validation_scores, dtype=np.float64).reshape(-1)
    val_logits = np.asarray(validation_logits)
    scores = np.asarray(pool_scores, dtype=np.float64).reshape(-1)
    logits = np.asarray(pool_logits)
    if val_logits.ndim != 2 or logits.ndim != 2:
        raise ValueError("validation_logits and pool_logits must be 2-D")
    if val_logits.shape[1] != logits.shape[1]:
        raise ValueError("validation and pool class dimensions must match")
    if len(val_scores) != len(val_logits) or len(scores) != len(logits):
        raise ValueError("score arrays must align with logits")
    if validation_is_known is None:
        known_mask = np.ones(len(val_scores), dtype=bool)
    else:
        known_mask = np.asarray(validation_is_known, dtype=bool).reshape(-1)
        if len(known_mask) != len(val_scores):
            raise ValueError("validation_is_known must align with validation scores")
    val_scores = val_scores[known_mask]
    val_logits = val_logits[known_mask]
    if not len(val_scores) or not len(scores):
        raise ValueError("known validation and pool scores must be nonempty")
    if not np.isfinite(val_scores).all() or not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not 0.0 < ratio <= 1.0 or min_class_samples < 1:
        raise ValueError("ratio must be in (0, 1] and min_class_samples positive")
    val_pred = val_logits.argmax(axis=1)
    pool_pred = logits.argmax(axis=1)
    global_reference = np.sort(val_scores)
    references = {}
    fallback_classes = []
    for class_id in range(val_logits.shape[1]):
        class_scores = val_scores[val_pred == class_id]
        if len(class_scores) >= min_class_samples:
            references[class_id] = np.sort(class_scores)
        else:
            references[class_id] = global_reference
            fallback_classes.append(class_id)
    percentiles = np.empty(len(scores), dtype=np.float64)
    for class_id, reference in references.items():
        rows = np.flatnonzero(pool_pred == class_id)
        if len(rows):
            percentiles[rows] = (
                np.searchsorted(reference, scores[rows], side="right") / len(reference)
            )
    budget = min(len(scores), max(1, int(np.ceil(len(scores) * ratio))))
    selected = np.argsort(-percentiles, kind="stable")[:budget]
    mask = np.zeros(len(scores), dtype=bool)
    mask[selected] = True
    return mask, percentiles, {
        "target_ratio": float(ratio),
        "selected_count": int(mask.sum()),
        "known_validation_count": int(len(val_scores)),
        "excluded_unknown_validation_count": int((~known_mask).sum()),
        "fallback_predicted_classes": fallback_classes,
        "validation_count_by_predicted_class": {
            str(class_id): int((val_pred == class_id).sum())
            for class_id in range(val_logits.shape[1])
        },
    }


def validation_class_threshold_candidates(
    validation_logits: np.ndarray,
    validation_uncertainty: np.ndarray,
    pool_logits: np.ndarray,
    pool_uncertainty: np.ndarray,
    coverage: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """Select pool items exceeding predicted-class risk quantiles from known val."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    val_logits = np.asarray(validation_logits)
    pool_logits = np.asarray(pool_logits)
    val_uncertainty = np.asarray(validation_uncertainty).reshape(-1)
    pool_uncertainty = np.asarray(pool_uncertainty).reshape(-1)
    if val_logits.ndim != 2 or pool_logits.ndim != 2:
        raise ValueError("logits must be 2-D arrays")
    if val_logits.shape[1] != pool_logits.shape[1]:
        raise ValueError("validation and pool class dimensions must match")
    if len(val_logits) != len(val_uncertainty) or len(pool_logits) != len(pool_uncertainty):
        raise ValueError("uncertainty arrays must align with logits")
    val_probs = torch.from_numpy(val_logits).softmax(dim=-1).numpy()
    pool_probs = torch.from_numpy(pool_logits).softmax(dim=-1).numpy()
    val_entropy = -(val_probs * np.log(np.clip(val_probs, 1e-8, None))).sum(axis=-1)
    pool_entropy = -(pool_probs * np.log(np.clip(pool_probs, 1e-8, None))).sum(axis=-1)
    val_risk = val_entropy + val_uncertainty
    pool_risk = pool_entropy + pool_uncertainty
    val_pred = val_logits.argmax(axis=-1)
    pool_pred = pool_logits.argmax(axis=-1)
    global_threshold = float(np.quantile(val_risk, coverage))
    thresholds = {}
    for class_id in range(val_logits.shape[1]):
        class_risk = val_risk[val_pred == class_id]
        thresholds[class_id] = (
            float(np.quantile(class_risk, coverage))
            if class_risk.size >= 20
            else global_threshold
        )
    mask = np.asarray(
        [pool_risk[i] > thresholds[int(pool_pred[i])] for i in range(len(pool_pred))],
        dtype=bool,
    )
    calibration = {
        "target_known_coverage": float(coverage),
        "global_fallback_threshold": global_threshold,
        "per_predicted_class_thresholds": {
            str(key): value for key, value in thresholds.items()
        },
        "fallback_classes": [
            int(class_id)
            for class_id in range(val_logits.shape[1])
            if int((val_pred == class_id).sum()) < 20
        ],
        "validation_sample_count": int(len(val_logits)),
        "validation_classwise_acceptance": {
            str(class_id): float(
                np.mean(val_risk[val_pred == class_id] <= thresholds[class_id])
            )
            for class_id in range(val_logits.shape[1])
            if np.any(val_pred == class_id)
        },
    }
    return mask, calibration


def validation_class_percentile_candidates(
    validation_logits: np.ndarray,
    validation_uncertainty: np.ndarray,
    pool_logits: np.ndarray,
    pool_uncertainty: np.ndarray,
    ratio: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """Rank pool risk by known-validation CDF within predicted class, then top-k."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    val_logits = np.asarray(validation_logits)
    pool_logits = np.asarray(pool_logits)
    val_uncertainty = np.asarray(validation_uncertainty).reshape(-1)
    pool_uncertainty = np.asarray(pool_uncertainty).reshape(-1)
    if val_logits.ndim != 2 or pool_logits.ndim != 2:
        raise ValueError("logits must be 2-D arrays")
    if val_logits.shape[1] != pool_logits.shape[1]:
        raise ValueError("validation and pool class dimensions must match")
    if len(val_logits) != len(val_uncertainty) or len(pool_logits) != len(pool_uncertainty):
        raise ValueError("uncertainty arrays must align with logits")

    def risk(logits, uncertainty):
        probs = torch.from_numpy(logits).softmax(dim=-1).numpy()
        entropy = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=-1)
        return entropy + uncertainty

    val_risk = risk(val_logits, val_uncertainty)
    pool_risk = risk(pool_logits, pool_uncertainty)
    val_pred = val_logits.argmax(axis=-1)
    pool_pred = pool_logits.argmax(axis=-1)
    global_reference = np.sort(val_risk)
    per_class_reference = {}
    fallback_classes = []
    for class_id in range(val_logits.shape[1]):
        scores = val_risk[val_pred == class_id]
        if scores.size >= 20:
            per_class_reference[class_id] = np.sort(scores)
        else:
            per_class_reference[class_id] = global_reference
            fallback_classes.append(class_id)
    percentile = np.empty(len(pool_risk), dtype=np.float64)
    for class_id, reference in per_class_reference.items():
        rows = np.flatnonzero(pool_pred == class_id)
        if rows.size:
            percentile[rows] = np.searchsorted(
                reference, pool_risk[rows], side="right"
            ) / len(reference)
    budget = min(len(percentile), max(1, int(np.ceil(len(percentile) * ratio))))
    selected = np.argpartition(percentile, len(percentile) - budget)[-budget:]
    mask = np.zeros(len(percentile), dtype=bool)
    mask[selected] = True
    report = {
        "target_selected_ratio": float(ratio),
        "selected_count": int(mask.sum()),
        "predicted_classes_with_global_fallback": fallback_classes,
        "known_validation_count_by_predicted_class": {
            str(class_id): int((val_pred == class_id).sum())
            for class_id in range(val_logits.shape[1])
        },
        "validation_empirical_percentile_range_by_predicted_class": {
            str(class_id): {
                "min": float(np.min(np.searchsorted(reference, reference, side="right") / len(reference))),
                "max": float(np.max(np.searchsorted(reference, reference, side="right") / len(reference))),
            }
            for class_id, reference in per_class_reference.items()
        },
    }
    return mask, report


def select_rank_fused_boundary_candidates(
    logits: torch.Tensor,
    uncertainty: torch.Tensor,
    features: torch.Tensor,
    support_bank: dict,
    ratio: float = 0.25,
    fusion_weight: float = 0.5,
) -> torch.Tensor:
    """Fixed-weight fusion of classifier risk and known-support boundary rank.

    This is an offline mechanism diagnostic. Both components are converted to
    within-pool ranks before averaging, so the fixed 0.5 weight is not a
    hidden scale/temperature sweep. It uses only known support features and
    the frozen model outputs; pool labels never enter selection.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if not 0.0 <= fusion_weight <= 1.0:
        raise ValueError("fusion_weight must be in [0, 1]")
    if logits.ndim != 2 or features.ndim != 2:
        raise ValueError("logits and features must be 2-D")
    if len(logits) != len(features) or uncertainty.shape != (len(logits),):
        raise ValueError("candidate tensors must align")
    probabilities = logits.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    risk = entropy + uncertainty
    boundary = support_boundary_candidate_scores(features, support_bank)

    def rank(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values, stable=True)
        result = torch.empty_like(values, dtype=torch.float32)
        result[order] = torch.linspace(0.0, 1.0, len(values), device=values.device)
        return result

    score = float(fusion_weight) * rank(risk) + (1.0 - float(fusion_weight)) * rank(boundary)
    budget = min(len(score), max(1, int(np.ceil(len(score) * ratio))))
    selected = torch.topk(score, k=budget, largest=True).indices
    mask = torch.zeros(len(score), dtype=torch.bool, device=score.device)
    mask[selected] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selector-ratio",
        type=float,
        default=None,
        help="override discovery_select_ratio for this retrospective audit only",
    )
    parser.add_argument(
        "--validation-coverage",
        type=float,
        default=0.95,
        help="known-validation coverage target for the diagnostic classwise threshold selector",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=20,
        help="stochastic forwards for retrospective MC-Dropout selector diagnostics",
    )
    parser.add_argument(
        "--mc-seed", type=int, default=0,
        help="random seed for the isolated MC-Dropout sampling stream",
    )
    parser.add_argument(
        "--selection-indices-output",
        type=Path,
        default=None,
        help=(
            "optional .npz path for deterministic candidate-selection masks; "
            "indices are audit artifacts and do not contain labels"
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.validation_coverage < 1.0:
        raise ValueError("--validation-coverage must be in (0, 1)")
    config = load_config(args.source_config, args.student_ckpt)
    if args.selector_ratio is not None:
        if not 0.0 < args.selector_ratio <= 1.0:
            raise ValueError("--selector-ratio must be in (0, 1]")
        config.discovery_select_ratio = float(args.selector_ratio)
    device = resolve_device(config.device)

    bundle = build_data_bundle(
        config.dataset,
        config.data_root,
        config.num_known,
        config.seed,
        config.image_size,
        download=config.download,
        split_path=config.split_path,
        limit_train=config.limit_train or None,
        limit_val=config.limit_val or None,
        limit_test=config.limit_test or None,
        limit_discovery=config.limit_discovery or None,
        discovery_pool_mode=config.discovery_pool_mode,
        open_val_ratio=config.open_val_ratio,
        calibration_ratio=config.calibration_ratio,
        calibration_overlap_control=config.calibration_overlap_control,
        known_split_mode=config.known_split_mode,
        mixed_known_pool_ratio=config.mixed_known_pool_ratio,
        matched_discovery_pool=config.matched_discovery_pool,
        matched_pool_size=config.matched_pool_size,
        matched_known_prior=config.matched_known_prior,
        transform_profile=config.transform_profile,
    )
    model = build_model(
        len(bundle.known_classes),
        backbone=config.student_backbone or config.backbone,
        proj_dim=config.proj_dim,
        dropout=config.dropout,
        pretrained=config.pretrained,
        cifar_stem=config.cifar_stem,
        rejection_feature_dim=config.rejection_feature_dim,
        independent_rejection_encoder=bool(
            getattr(config, "independent_rejection_encoder", False)
        ),
        independent_rejection_encoder_init=str(
            getattr(config, "independent_rejection_encoder_init", "scratch_cnn")
        ),
        independent_rejection_input_norm=str(
            getattr(config, "independent_rejection_input_norm", "none")
        ),
    ).to(device)
    load_checkpoint(
        model,
        args.student_ckpt,
        device,
        expected_known_classes=bundle.known_classes,
        expected_novel_classes=bundle.novel_classes,
    )
    model.eval()

    support_loader = build_loader(
        clone_eval(bundle.train, config.image_size, config.transform_profile),
        config.batch_size,
        False,
        config.num_workers,
    )
    pool_loader = build_loader(
        clone_eval(bundle.discovery_pool, config.image_size, config.transform_profile),
        config.batch_size,
        False,
        config.num_workers,
    )
    val_loader = build_loader(
        clone_eval(bundle.val, config.image_size, config.transform_profile),
        config.batch_size,
        False,
        config.num_workers,
    )
    support = collect_classwise_knn_support(
        model,
        support_loader,
        device,
        len(bundle.known_classes),
        k=int(getattr(config, "discovery_knn_k", 5)),
        quantile=float(getattr(config, "discovery_knn_quantile", 0.95)),
    )
    independent_rejection_encoder = bool(
        getattr(config, "independent_rejection_encoder", False)
    )
    rejection_support = None
    if independent_rejection_encoder:
        rejection_support = collect_classwise_knn_support(
            model,
            support_loader,
            device,
            len(bundle.known_classes),
            k=int(getattr(config, "discovery_knn_k", 5)),
            quantile=float(getattr(config, "discovery_knn_quantile", 0.95)),
            feature_key="rejection_features",
        )
    support_outputs = extract_deterministic_audit_outputs(model, support_loader, device)
    prototype_support = fit_known_support_rejector(
        support_outputs, quantile=float(config.support_quantile)
    )
    outputs = extract_deterministic_audit_outputs(model, pool_loader, device)
    mc_outputs = extract_mc_audit_scores(
        model, pool_loader, device, args.mc_samples, args.mc_seed
    )
    val_outputs = extract_deterministic_audit_outputs(model, val_loader, device)
    logits = torch.from_numpy(outputs["logits"])
    uncertainty = torch.from_numpy(outputs["head_uncertainty"])
    features = torch.from_numpy(outputs["features"])
    prototypes = model.classifier.weight.detach().cpu()
    is_known = np.asarray(outputs["is_known"], dtype=bool)
    raw_labels = np.asarray(outputs["raw_labels"], dtype=np.int64)
    total_unknown = int((~is_known).sum())
    ratio = float(getattr(config, "discovery_select_ratio", 0.25))
    mode = str(getattr(config, "discovery_select_mode", "entropy_uncertainty"))
    support_k = int(getattr(config, "discovery_knn_k", 5))
    support_quantile = float(getattr(config, "discovery_knn_quantile", 0.95))

    masks = {
        f"shared_{mode}": select_discovery_candidates(
            logits,
            uncertainty,
            ratio=ratio,
            mode=mode,
            features=features,
            prototypes=prototypes,
        ).numpy(),
        "shared_consensus": select_discovery_candidates(
            logits,
            uncertainty,
            ratio=ratio,
            mode="consensus",
            features=features,
            prototypes=prototypes,
        ).numpy(),
        "shared_prototype_distance": select_discovery_candidates(
            logits,
            uncertainty,
            ratio=ratio,
            mode="prototype_distance",
            features=features,
            prototypes=prototypes,
        ).numpy(),
        "shared_distance_consensus": select_discovery_candidates(
            logits,
            uncertainty,
            ratio=ratio,
            mode="distance_consensus",
            features=features,
            prototypes=prototypes,
        ).numpy(),
        "class_balanced_entropy_uncertainty": select_class_balanced_candidates(
            logits, uncertainty, ratio=ratio
        ).numpy(),
        "prototype_boundary": select_prototype_boundary_candidates(
            features,
            torch.from_numpy(prototype_support["centers"]),
            torch.from_numpy(prototype_support["radii"]),
            ratio=ratio,
        ).numpy(),
        "support_boundary": select_support_boundary_candidates(
            features,
            support,
            ratio=ratio,
            k=support_k,
        ).numpy(),
        "support_distance": select_support_distance_candidates(
            features,
            support,
            ratio=ratio,
            k=support_k,
        ).numpy(),
        "rank_fused_entropy_support_boundary": select_rank_fused_boundary_candidates(
            logits,
            uncertainty,
            features,
            support,
            ratio=ratio,
            fusion_weight=0.5,
        ).numpy(),
    }
    masks["mc_predictive_entropy"] = fixed_budget_top_mask(
        mc_outputs["predictive_entropy"], ratio
    )
    masks["mc_epistemic_mi"] = fixed_budget_top_mask(
        mc_outputs["epistemic"], ratio
    )
    masks["mc_epistemic_plus_head_uncertainty"] = fixed_budget_top_mask(
        mc_outputs["epistemic"] + mc_outputs["head_uncertainty"], ratio
    )
    masks["mc_predictive_entropy_plus_head_uncertainty"] = fixed_budget_top_mask(
        mc_outputs["predictive_entropy"] + mc_outputs["head_uncertainty"], ratio
    )
    masks["mc_epistemic_head_rank_fusion"] = fixed_budget_rank_fusion(
        mc_outputs["epistemic"], mc_outputs["head_uncertainty"], ratio
    )
    if rejection_support is not None:
        rejection_features = outputs["rejection_features"]
        masks["rejection_representation_support_distance"] = select_support_distance_candidates(
            torch.from_numpy(rejection_features), rejection_support,
            ratio=ratio, k=support_k,
        ).numpy()
        shared_distance_score = support_distance_scores(
            outputs["features"], support, k=support_k
        )
        rejection_distance_score = support_distance_scores(
            rejection_features, rejection_support, k=support_k
        )
        dual_mask, dual_score = select_dual_representation_candidates(
            shared_distance_score,
            rejection_distance_score,
            ratio=ratio,
            fusion_weight=0.5,
        )
        masks["dual_representation_support_rank_fusion"] = dual_mask
        veto_mask, veto_report = select_shared_support_veto_candidates(
            shared_distance_score,
            rejection_distance_score,
            ratio=ratio,
        )
        masks["shared_support_veto_rejection_rank"] = veto_mask
        shared_validation_score = support_distance_scores(
            val_outputs["features"], support, k=support_k
        )
        rejection_validation_score = support_distance_scores(
            val_outputs["rejection_features"], rejection_support, k=support_k
        )
        shared_calibrated_mask, shared_calibrated_percentile, shared_calibration = (
            select_validation_classwise_percentile_candidates(
                shared_validation_score,
                val_outputs["logits"],
                shared_distance_score,
                outputs["logits"],
                ratio=ratio,
                validation_is_known=val_outputs["is_known"],
            )
        )
        rejection_calibrated_mask, rejection_calibrated_percentile, rejection_calibration = (
            select_validation_classwise_percentile_candidates(
                rejection_validation_score,
                val_outputs["rejection_logits"],
                rejection_distance_score,
                outputs["rejection_logits"],
                ratio=ratio,
                validation_is_known=val_outputs["is_known"],
            )
        )
        masks["shared_support_distance_validation_cdf"] = shared_calibrated_mask
        masks["rejection_support_distance_validation_cdf"] = rejection_calibrated_mask
    knn_vote_mask, knn_vote_entropy = select_knn_vote_entropy_candidates(
        support_outputs["features"],
        support_outputs["labels"],
        outputs["features"],
        ratio=ratio,
        k=10,
    )
    masks["known_support_knn_vote_entropy"] = knn_vote_mask
    knn_disagreement_mask, knn_disagreement = select_knn_classifier_disagreement_candidates(
        support_outputs["features"],
        support_outputs["labels"],
        outputs["features"],
        outputs["logits"],
        ratio=ratio,
        k=10,
    )
    masks["known_support_knn_classifier_disagreement"] = knn_disagreement_mask
    validation_mask, validation_calibration = validation_class_threshold_candidates(
        val_outputs["logits"],
        val_outputs["head_uncertainty"],
        outputs["logits"],
        outputs["head_uncertainty"],
        coverage=args.validation_coverage,
    )
    masks["validation_predicted_class_q95"] = validation_mask
    percentile_mask, percentile_calibration = validation_class_percentile_candidates(
        val_outputs["logits"],
        val_outputs["head_uncertainty"],
        outputs["logits"],
        outputs["head_uncertainty"],
        ratio=ratio,
    )
    masks["validation_predicted_class_percentile_top25"] = percentile_mask
    report = {
        "protocol": {
            "seed": int(config.seed),
            "pool_size": int(len(is_known)),
            "known_fraction": float(is_known.mean()),
            "selector_ratio": ratio,
            "validation_coverage": float(args.validation_coverage),
            "mc_samples": int(args.mc_samples),
            "mc_seed": int(args.mc_seed),
            "mc_dropout_selection_is_retrospective_only": True,
            "mc_pool_labels_used_for_selection": False,
            "mc_scores": {
                key: {
                    "mean": float(np.mean(value)),
                    "std": float(np.std(value)),
                    "q95": float(np.quantile(value, 0.95)),
                }
                for key, value in mc_outputs.items()
            },
            "selector_mode_from_config": mode,
            "support_k": support_k,
            "support_quantile": support_quantile,
            "labels_used_only_for_audit": True,
            "validation_known_labels_used_only_for_calibration": True,
            "selector_count": len(masks),
            "independent_rejection_encoder": independent_rejection_encoder,
            "representation_comparison": (
                "shared_vs_rejection_support_distance_same_checkpoint"
                if independent_rejection_encoder
                else "shared_representation_only"
            ),
            "dual_representation_fusion": (
                {
                    "fusion_weight_shared": 0.5,
                    "fusion_weight_rejection": 0.5,
                    "shared_score_min": float(shared_distance_score.min()),
                    "rejection_score_min": float(rejection_distance_score.min()),
                    "fused_score_min": float(dual_score.min()),
                }
                if rejection_support is not None
                else None
            ),
            "shared_support_veto": veto_report if rejection_support is not None else None,
            "support_distance_validation_cdf": (
                {
                    "min_class_samples": 20,
                    "shared": shared_calibration,
                    "rejection": rejection_calibration,
                    "shared_percentile_median": float(np.median(shared_calibrated_percentile)),
                    "rejection_percentile_median": float(np.median(rejection_calibrated_percentile)),
                    "validation_labels_used_only_for_calibration": True,
                    "pool_labels_used_for_selection": False,
                }
                if rejection_support is not None
                else None
            ),
            "knn_vote_entropy": {
                "k": min(10, len(support_outputs["labels"])),
                "support_count": int(len(support_outputs["labels"])),
                "score_min": float(knn_vote_entropy.min()),
                "score_median": float(np.median(knn_vote_entropy)),
                "score_max": float(knn_vote_entropy.max()),
                "pool_labels_used_for_selection": False,
            },
            "knn_classifier_disagreement": {
                "k": min(10, len(support_outputs["labels"])),
                "score_min": float(knn_disagreement.min()),
                "score_median": float(np.median(knn_disagreement)),
                "score_max": float(knn_disagreement.max()),
                "pool_labels_used_for_selection": False,
            },
        },
        "validation_classwise_calibration": validation_calibration,
        "validation_uncertainty_error_diagnostic": validation_uncertainty_error_diagnostics(val_outputs),
        "validation_classwise_percentile_calibration": percentile_calibration,
        "candidate_audit": {
            name: summarize(mask, is_known, total_unknown)
            for name, mask in masks.items()
        },
        "classwise_candidate_audit": {
            name: summarize_classwise(mask, raw_labels, is_known)
            for name, mask in masks.items()
        },
    }
    if args.selection_indices_output is not None:
        args.selection_indices_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.selection_indices_output,
            **{name: np.asarray(mask, dtype=np.uint8) for name, mask in masks.items()},
        )
        report["protocol"]["selection_indices_output"] = str(
            args.selection_indices_output
        )
        report["protocol"]["selection_indices_are_label_free"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
