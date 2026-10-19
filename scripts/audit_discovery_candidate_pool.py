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
        key: [] for key in ("logits", "head_uncertainty", "features", "labels", "raw_labels", "is_known")
    }
    for images, labels, raw_labels, is_known, _ in loader:
        output = model(images.to(device))
        chunks["logits"].append(output["logits"].cpu())
        chunks["head_uncertainty"].append(output["uncertainty"].cpu())
        chunks["features"].append(output["features"].cpu())
        chunks["labels"].append(labels.cpu())
        chunks["raw_labels"].append(raw_labels.cpu())
        chunks["is_known"].append(is_known.cpu())
    return {
        key: torch.cat(values).numpy()
        for key, values in chunks.items()
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.source_config, args.student_ckpt)
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
    support_outputs = extract_deterministic_audit_outputs(model, support_loader, device)
    prototype_support = fit_known_support_rejector(
        support_outputs, quantile=float(config.support_quantile)
    )
    outputs = extract_deterministic_audit_outputs(model, pool_loader, device)
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
    }
    validation_mask, validation_calibration = validation_class_threshold_candidates(
        val_outputs["logits"],
        val_outputs["head_uncertainty"],
        outputs["logits"],
        outputs["head_uncertainty"],
        coverage=0.95,
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
            "selector_mode_from_config": mode,
            "support_k": support_k,
            "support_quantile": support_quantile,
            "labels_used_only_for_audit": True,
            "validation_known_labels_used_only_for_calibration": True,
            "selector_count": len(masks),
        },
        "validation_classwise_calibration": validation_calibration,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
