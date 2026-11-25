"""Audit whether teacher uncertainty aligns with known-label teacher error.

This is a read-only diagnostic. Known labels are used only after the frozen
teacher has run, to measure whether the detached uncertainty signal used by
uncertainty-KD is related to teacher correctness. No model, threshold, or
training weight is selected from this report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
from novel_discovery.models import build_model  # noqa: E402
from novel_discovery.pipeline import build_loader  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_teacher(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = read_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Unsupported checkpoint layout: {checkpoint_path}")
    model = build_model(
        num_classes=len(checkpoint["known_classes"]),
        backbone=config["backbone"],
        proj_dim=int(config.get("proj_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        pretrained=False,
        cifar_stem=bool(config.get("cifar_stem", False)),
        rejection_feature_dim=int(config.get("rejection_feature_dim", 0)),
        independent_rejection_encoder=bool(
            config.get("independent_rejection_encoder", False)
        ),
        independent_rejection_encoder_init=config.get(
            "independent_rejection_encoder_init", "shared_backbone"
        ),
        independent_rejection_input_norm=config.get(
            "independent_rejection_input_norm", "none"
        ),
        structure_feature_dim=int(config.get("structure_feature_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return config, checkpoint, model.to(device).eval()


def replace_with_eval_transform(dataset, transform) -> None:
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    if hasattr(current, "base"):
        current.base.transform = transform
        return
    if hasattr(current, "transform"):
        current.transform = transform
        return
    raise TypeError("Could not replace dataset transform for deterministic audit")


@torch.inference_mode()
def extract(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    uncertainty: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, mapped_labels, _raw_labels, is_known, _indices in loader:
        if not bool(is_known.all()):
            raise ValueError("Teacher uncertainty alignment audit requires known-only data")
        output = model(images.to(device, non_blocking=True))
        logits = output["logits"]
        predicted = logits.argmax(dim=-1)
        uncertainty.append(output["uncertainty"].reshape(-1).float().cpu().numpy())
        errors.append(predicted.ne(mapped_labels.to(device)).cpu().numpy().astype(bool))
        confidence.append(logits.softmax(dim=-1).amax(dim=-1).float().cpu().numpy())
        labels.append(mapped_labels.cpu().numpy())
    if not uncertainty:
        raise ValueError("Cannot audit an empty dataset")
    return {
        "uncertainty": np.concatenate(uncertainty),
        "error": np.concatenate(errors),
        "confidence": np.concatenate(confidence),
        "labels": np.concatenate(labels),
    }


def summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "q05": float(np.quantile(values, 0.05)) if values.size else None,
        "median": float(np.quantile(values, 0.50)) if values.size else None,
        "q95": float(np.quantile(values, 0.95)) if values.size else None,
    }


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / (len(weights) * np.square(weights).sum() + 1e-12))


def minibatch_weight_diagnostics(
    labels: np.ndarray,
    uncertainty: np.ndarray,
    batch_size: int,
    repeats: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate the current and class-balanced weights under shuffled batches."""
    labels = np.asarray(labels, dtype=np.int64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if labels.ndim != 1 or uncertainty.shape != labels.shape or labels.size == 0:
        raise ValueError("labels and uncertainty must be non-empty aligned vectors")
    if batch_size < 1 or repeats < 1:
        raise ValueError("batch_size and repeats must be positive")

    rng = np.random.default_rng(seed)
    epoch_stats: dict[str, list[dict[str, float]]] = {
        "global_uncertainty": [],
        "within_class_uncertainty": [],
    }
    singleton_exposures = 0
    total_exposures = 0
    for _ in range(repeats):
        order = rng.permutation(labels.size)
        for start in range(0, labels.size, batch_size):
            indices = order[start : start + batch_size]
            batch_labels = labels[indices]
            base = np.exp(np.clip(-uncertainty[indices], -60.0, 60.0))
            global_weights = base / max(float(base.mean()), 1e-12)
            class_weights = np.empty_like(base)
            for label in np.unique(batch_labels):
                mask = batch_labels == label
                class_weights[mask] = base[mask] / max(float(base[mask].mean()), 1e-12)
                if int(mask.sum()) == 1:
                    singleton_exposures += 1
            class_weights /= max(float(class_weights.mean()), 1e-12)
            total_exposures += len(indices)
            for name, weights in (
                ("global_uncertainty", global_weights),
                ("within_class_uncertainty", class_weights),
            ):
                epoch_stats[name].append(
                    {
                        "weight_std": float(weights.std()),
                        "ess_fraction": effective_sample_size(weights),
                    }
                )

    return {
        "batch_size": int(batch_size),
        "simulated_epochs": int(repeats),
        "seed": int(seed),
        "singleton_class_exposure_fraction": singleton_exposures / max(total_exposures, 1),
        "global_uncertainty": {
            "mean_batch_weight_std": float(
                np.mean([row["weight_std"] for row in epoch_stats["global_uncertainty"]])
            ),
            "mean_batch_ess_fraction": float(
                np.mean([row["ess_fraction"] for row in epoch_stats["global_uncertainty"]])
            ),
        },
        "within_class_uncertainty": {
            "mean_batch_weight_std": float(
                np.mean([row["weight_std"] for row in epoch_stats["within_class_uncertainty"]])
            ),
            "mean_batch_ess_fraction": float(
                np.mean([row["ess_fraction"] for row in epoch_stats["within_class_uncertainty"]])
            ),
        },
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def classwise_alignment(
    values: dict[str, np.ndarray], raw_class_ids: list[int] | None = None
) -> dict[str, Any]:
    labels = values["labels"].astype(np.int64)
    uncertainty = values["uncertainty"].astype(np.float64)
    confidence = values["confidence"].astype(np.float64)
    error = values["error"].astype(bool)
    class_rows = []
    for label_index in np.unique(labels):
        mask = labels == label_index
        class_error = error[mask]
        class_uncertainty = uncertainty[mask]
        class_confidence = confidence[mask]
        class_rows.append(
            {
                "label_index": int(label_index),
                "raw_class_id": int(raw_class_ids[label_index])
                if raw_class_ids is not None
                else int(label_index),
                "samples": int(mask.sum()),
                "teacher_error_rate": float(class_error.mean()),
                "mean_uncertainty": float(class_uncertainty.mean()),
                "mean_confidence": float(class_confidence.mean()),
                "error_uncertainty_auroc": (
                    float(roc_auc_score(class_error.astype(np.int64), class_uncertainty))
                    if len(np.unique(class_error)) == 2
                    else None
                ),
            }
        )
    valid_aucs = [
        row["error_uncertainty_auroc"]
        for row in class_rows
        if row["error_uncertainty_auroc"] is not None
    ]
    rates = np.asarray([row["teacher_error_rate"] for row in class_rows])
    mean_uncertainty = np.asarray([row["mean_uncertainty"] for row in class_rows])
    mean_confidence = np.asarray([row["mean_confidence"] for row in class_rows])
    return {
        "class_count": len(class_rows),
        "within_class_error_uncertainty_macro_auroc": float(np.mean(valid_aucs))
        if valid_aucs
        else None,
        "within_class_error_uncertainty_valid_class_count": len(valid_aucs),
        "spearman_class_error_rate_vs_mean_uncertainty": _spearman(
            rates, mean_uncertainty
        ),
        "spearman_class_error_rate_vs_mean_confidence": _spearman(
            rates, mean_confidence
        ),
        "classes": class_rows,
    }


def binned_error_rates(uncertainty: np.ndarray, error: np.ndarray, weights: np.ndarray):
    order = np.argsort(uncertainty, kind="stable")
    bins = []
    for index, rows in enumerate(np.array_split(order, 5)):
        if len(rows) == 0:
            continue
        bins.append(
            {
                "bin": int(index + 1),
                "count": int(len(rows)),
                "uncertainty_mean": float(uncertainty[rows].mean()),
                "teacher_error_rate": float(error[rows].mean()),
                "normalized_weight_mean": float(weights[rows].mean()),
            }
        )
    return bins


def audit_split(
    values: dict[str, np.ndarray],
    raw_class_ids: list[int] | None = None,
    simulation_batch_size: int = 64,
    simulation_seed: int = 0,
) -> dict[str, Any]:
    uncertainty = values["uncertainty"].astype(np.float64)
    error = values["error"].astype(bool)
    confidence = values["confidence"].astype(np.float64)
    weights = np.exp(np.clip(-uncertainty, -60.0, 60.0))
    normalized_weights = weights / max(float(weights.mean()), 1e-12)
    report: dict[str, Any] = {
        "samples": int(len(error)),
        "teacher_error_count": int(error.sum()),
        "teacher_error_rate": float(error.mean()),
        "uncertainty": summary(uncertainty),
        "confidence": summary(confidence),
        "error_uncertainty_auroc": (
            float(roc_auc_score(error.astype(np.int64), uncertainty))
            if len(np.unique(error)) == 2
            else None
        ),
        "error_uncertainty_mean_correct": float(uncertainty[~error].mean())
        if (~error).any()
        else None,
        "error_uncertainty_mean_error": float(uncertainty[error].mean())
        if error.any()
        else None,
        "error_confidence_auroc": (
            float(roc_auc_score(error.astype(np.int64), -confidence))
            if len(np.unique(error)) == 2
            else None
        ),
        "exp_minus_uncertainty": {
            "normalized_weight_mean": float(normalized_weights.mean()),
            "normalized_weight_std": float(normalized_weights.std()),
            "ess_fraction": effective_sample_size(weights),
            "error_weight_mass_fraction": float(
                normalized_weights[error].sum() / max(float(normalized_weights.sum()), 1e-12)
            )
            if error.any()
            else 0.0,
            "error_weight_mass_lift_over_error_rate": float(
                (normalized_weights[error].sum() / normalized_weights.sum())
                / max(float(error.mean()), 1e-12)
            )
            if error.any()
            else None,
            "mean_weight_correct": float(normalized_weights[~error].mean())
            if (~error).any()
            else None,
            "mean_weight_error": float(normalized_weights[error].mean())
            if error.any()
            else None,
        },
        "uncertainty_sorted_bins": binned_error_rates(
            uncertainty, error, normalized_weights
        ),
        "classwise_alignment": classwise_alignment(values, raw_class_ids),
        "minibatch_weight_simulation": minibatch_weight_diagnostics(
            values["labels"],
            uncertainty,
            batch_size=simulation_batch_size,
            repeats=4,
            seed=simulation_seed,
        ),
    }
    return report


def audit(args) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    config_path = Path(args.teacher_config).resolve()
    checkpoint_path = Path(args.teacher_checkpoint).resolve()
    config, checkpoint, model = load_teacher(config_path, checkpoint_path, device)
    if int(config.get("seed")) != int(args.seed):
        raise ValueError("teacher config seed does not match requested seed")
    split_path = Path(config["split_path"]).resolve()
    if split_path != Path(args.split_path).resolve():
        raise ValueError("teacher config and requested split differ")
    if config.get("transform_profile") != "cifar":
        raise ValueError("this audit requires the CIFAR transform profile")

    bundle = build_data_bundle(
        "cifar100",
        config["data_root"],
        int(config["num_known"]),
        int(args.seed),
        int(config["image_size"]),
        split_path=split_path,
        discovery_pool_mode="unknown",
        transform_profile="cifar",
    )
    eval_transform = build_transforms(
        int(config["image_size"]), train=False, profile="cifar"
    )
    replace_with_eval_transform(bundle.train, eval_transform)
    train_values = extract(model, build_loader(bundle.train, args.batch_size, False, 0), device)
    val_values = extract(model, build_loader(bundle.val, args.batch_size, False, 0), device)
    return {
        "purpose": "frozen teacher uncertainty alignment diagnostic",
        "protocol": {
            "seed": int(args.seed),
            "split_path": str(split_path),
            "teacher_config": str(config_path),
            "teacher_checkpoint": str(checkpoint_path),
            "teacher_backbone": config.get("backbone"),
            "transform_profile": config.get("transform_profile"),
            "known_labels_used_for": "post-hoc teacher error alignment only",
            "used_for_training_or_calibration": False,
            "checkpoint_known_classes_match": checkpoint.get("known_classes")
            == read_json(split_path).get("known_classes"),
        },
        "train": audit_split(
            train_values,
            checkpoint["known_classes"],
            simulation_batch_size=64,
            simulation_seed=int(args.seed),
        ),
        "validation": audit_split(
            val_values,
            checkpoint["known_classes"],
            simulation_batch_size=64,
            simulation_seed=int(args.seed) + 1,
        ),
        "cross_split_class_rank_stability": {
            "matched_class_count": int(
                len(
                    np.intersect1d(
                        np.unique(train_values["labels"]),
                        np.unique(val_values["labels"]),
                    )
                )
            ),
            "spearman_class_error_rate_train_vs_validation": _spearman(
                np.asarray(
                    [
                        train_values["error"][train_values["labels"] == index].mean()
                        for index in range(len(checkpoint["known_classes"]))
                    ]
                ),
                np.asarray(
                    [
                        val_values["error"][val_values["labels"] == index].mean()
                        for index in range(len(checkpoint["known_classes"]))
                    ]
                ),
            ),
            "spearman_class_mean_uncertainty_train_vs_validation": _spearman(
                np.asarray(
                    [
                        train_values["uncertainty"][train_values["labels"] == index].mean()
                        for index in range(len(checkpoint["known_classes"]))
                    ]
                ),
                np.asarray(
                    [
                        val_values["uncertainty"][val_values["labels"] == index].mean()
                        for index in range(len(checkpoint["known_classes"]))
                    ]
                ),
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
