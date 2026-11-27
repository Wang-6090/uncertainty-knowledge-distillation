"""Audit whether the rejection representation is actually independent.

This is a post-hoc mechanism audit.  Labels from the selected evaluation split
are used only for descriptive representation diagnostics; they are never used
to fit a project model, choose a checkpoint, or calibrate a threshold.  Test
known/unknown labels are only used for the explicitly marked final probe.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.data import build_data_bundle, build_transforms
from novel_discovery.models import build_model
from novel_discovery.pipeline import build_loader
from novel_discovery.utils import save_json


def _deterministic_outputs(model, loader, device):
    features = []
    rejection_raw_inputs = []
    rejection_inputs = []
    rejection = []
    logits = []
    entropy = []
    uncertainty = []
    known = []
    labels = []
    raw = []
    with torch.no_grad():
        model.eval()
        for images, _labels, raw_labels, is_known, _indices in loader:
            out = model(images.to(device), stochastic=False)
            probs = out["logits"].softmax(dim=-1)
            features.append(out["features"].cpu().numpy())
            rejection_raw_inputs.append(out["rejection_raw_input"].cpu().numpy())
            rejection_inputs.append(out["rejection_input"].cpu().numpy())
            rejection.append(out["rejection_features"].cpu().numpy())
            logits.append(out["logits"].cpu().numpy())
            entropy.append((-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).cpu().numpy())
            uncertainty.append(out["uncertainty"].cpu().numpy())
            labels.append(_labels.cpu().numpy().astype(int))
            known.append(is_known.cpu().numpy().astype(bool))
            raw.append(raw_labels.cpu().numpy())
    return {
        "features": np.concatenate(features).astype(np.float32),
        "rejection_raw_inputs": np.concatenate(rejection_raw_inputs).astype(np.float32),
        "rejection_inputs": np.concatenate(rejection_inputs).astype(np.float32),
        "rejection_features": np.concatenate(rejection).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
        "entropy": np.concatenate(entropy).astype(np.float32),
        "head_uncertainty": np.concatenate(uncertainty).astype(np.float32),
        "labels": np.concatenate(labels).astype(int),
        "is_known": np.concatenate(known),
        "raw_labels": np.concatenate(raw).astype(int),
    }


def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _effective_rank(x):
    s = np.linalg.svd(x - x.mean(axis=0, keepdims=True), compute_uv=False)
    energy = s * s
    if not np.any(energy > 0):
        return 0.0, 0.0
    p = energy / energy.sum()
    participation = float(1.0 / np.sum(p * p))
    cumulative = np.cumsum(p)
    rank95 = int(np.searchsorted(cumulative, 0.95) + 1)
    return participation, float(rank95)


def _neighbor_jaccard(shared, rejection, k=10):
    shared = _normalize(shared)
    rejection = _normalize(rejection)
    shared_sim = shared @ shared.T
    rejection_sim = rejection @ rejection.T
    np.fill_diagonal(shared_sim, -np.inf)
    np.fill_diagonal(rejection_sim, -np.inf)
    shared_nn = np.argpartition(shared_sim, -k, axis=1)[:, -k:]
    rejection_nn = np.argpartition(rejection_sim, -k, axis=1)[:, -k:]
    values = []
    for a, b in zip(shared_nn, rejection_nn):
        values.append(len(set(a.tolist()) & set(b.tolist())) / float(k))
    return float(np.mean(values))


def _binary_probe(x, y, seed):
    if len(np.unique(y)) < 2:
        return None
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(x, y))
    # Fit the scaler on the train portion only; this is a descriptive probe,
    # not a detector used by the project.
    scaler = StandardScaler().fit(x[train_idx])
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(scaler.transform(x[train_idx]), y[train_idx])
    score = model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
    return float(roc_auc_score(y[test_idx], score))


def _fit_multiclass_probe(x, y, seed):
    """Fit a descriptive known-class probe on a supplied feature matrix."""
    y = np.asarray(y, dtype=np.int64)
    if len(x) == 0 or len(np.unique(y)) < 2:
        return None
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )
    model.fit(scaler.transform(x), y)
    return scaler, model


def _multiclass_probe_accuracy(x, y, seed):
    """Within-split known-class accuracy; labels are never used by training."""
    y = np.asarray(y, dtype=np.int64)
    if len(x) == 0 or len(np.unique(y)) < 2:
        return None
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(x, y))
    fitted = _fit_multiclass_probe(x[train_idx], y[train_idx], seed)
    if fitted is None:
        return None
    scaler, model = fitted
    prediction = model.predict(scaler.transform(x[test_idx]))
    return float(np.mean(prediction == y[test_idx]))


def _cross_split_multiclass_probe_accuracy(source, target, seed):
    """Train a known-class probe on one split and evaluate it on another."""
    source_y = np.asarray(source["labels"], dtype=np.int64)
    target_y = np.asarray(target["labels"], dtype=np.int64)
    source_mask = source_y >= 0
    target_mask = target_y >= 0
    if source_mask.sum() == 0 or target_mask.sum() == 0:
        return None
    source_classes = set(np.unique(source_y[source_mask]).tolist())
    target_classes = set(np.unique(target_y[target_mask]).tolist())
    if len(source_classes) < 2 or not target_classes.issubset(source_classes):
        return None
    fitted = _fit_multiclass_probe(
        source["features"][source_mask], source_y[source_mask], seed
    )
    if fitted is None:
        return None
    scaler, model = fitted
    prediction = model.predict(scaler.transform(target["features"][target_mask]))
    return float(np.mean(prediction == target_y[target_mask]))


def _feature_statistics(x, labels):
    """Known-only scale and class-variance diagnostics for one representation."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mask = labels >= 0
    x = x[mask]
    labels = labels[mask]
    if len(x) == 0:
        return {"sample_count": 0}
    centered = x - x.mean(axis=0, keepdims=True)
    total = float(np.mean(np.sum(centered * centered, axis=1)))
    within_values = []
    for label in np.unique(labels):
        group = x[labels == label]
        if len(group):
            center = group.mean(axis=0, keepdims=True)
            within_values.append(np.sum((group - center) ** 2, axis=1))
    within = float(np.mean(np.concatenate(within_values))) if within_values else 0.0
    between = max(total - within, 0.0)
    norms = np.linalg.norm(x, axis=1)
    return {
        "sample_count": int(len(x)),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_q05": float(np.quantile(norms, 0.05)),
        "norm_q95": float(np.quantile(norms, 0.95)),
        "total_variance": total,
        "within_class_variance": within,
        "between_class_variance": between,
        "between_to_total_ratio": float(between / max(total, 1e-12)),
        "class_count": int(len(np.unique(labels))),
    }


def _clone_dataset_with_eval_transform(dataset, image_size, transform_profile):
    """Clone train/validation wrappers without changing their source indices."""
    cloned = copy.copy(dataset)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        cloned.dataset = _clone_dataset_with_eval_transform(
            dataset.dataset, image_size, transform_profile
        )
        return cloned
    if hasattr(dataset, "base"):
        cloned.base = copy.copy(dataset.base)
        cloned.base.transform = build_transforms(
            image_size, train=False, profile=transform_profile
        )
        return cloned
    raise TypeError("evaluation split does not expose a clonable base dataset")


def _split_dataset(bundle, split_name, image_size, transform_profile):
    if split_name not in {"train", "val", "test"}:
        raise ValueError("evaluation split must be one of: train, val, test")
    dataset = getattr(bundle, split_name)
    if split_name in {"train", "val"}:
        dataset = _clone_dataset_with_eval_transform(
            dataset, image_size, transform_profile
        )
    return dataset


def _cross_mapping_r2(source, target, seed):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    # Use a balanced pseudo stratification so the split is deterministic even
    # though this is only a representation diagnostic.
    strata = np.arange(len(source)) % 2
    train_idx, test_idx = next(splitter.split(source, strata))
    source_scaler = StandardScaler().fit(source[train_idx])
    target_mean = target[train_idx].mean(axis=0, keepdims=True)
    reg = Ridge(alpha=10.0).fit(
        source_scaler.transform(source[train_idx]), target[train_idx] - target_mean
    )
    prediction = reg.predict(source_scaler.transform(source[test_idx])) + target_mean
    residual = np.sum((target[test_idx] - prediction) ** 2)
    total = np.sum((target[test_idx] - target_mean) ** 2)
    return float(1.0 - residual / max(total, 1e-12))


def _score_correlations(detail_path, outputs):
    if not detail_path:
        return {}
    detail = json.loads(Path(detail_path).read_text(encoding="utf-8"))
    result = {}
    score = np.asarray(detail.get("score", []), dtype=float)
    if len(score) != len(outputs["features"]):
        return {"error": "saved detector detail length does not match deterministic extraction"}
    candidates = {
        "score": score,
        "feature_rejector_score": detail.get("feature_rejector_score"),
        "entropy": outputs["entropy"],
        "head_uncertainty": outputs["head_uncertainty"],
        "shared_logit_norm": np.linalg.norm(outputs["logits"], axis=1),
        "shared_feature_norm": np.linalg.norm(outputs["features"], axis=1),
        "rejection_feature_norm": np.linalg.norm(outputs["rejection_features"], axis=1),
    }
    for name, values in candidates.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if len(values) != len(score) or np.std(values) < 1e-12:
            continue
        result[name] = float(np.corrcoef(score, values)[0, 1])
    return result


def audit_one(
    args,
    device,
    split_path,
    checkpoint,
    detail_path,
    seed,
    independent_rejection_encoder=False,
    independent_rejection_encoder_init="shared_backbone",
    independent_rejection_input_norm="none",
    evaluation_split="test",
    comparison_split=None,
    open_val_ratio=0.0,
):
    bundle = build_data_bundle(
        "cifar100", args.data_root, args.num_known, seed, args.image_size,
        split_path=split_path, limit_train=0, limit_val=0, limit_test=0,
        limit_discovery=0, discovery_pool_mode="mixed",
        matched_discovery_pool=True, matched_pool_size=args.matched_pool_size,
        matched_known_prior=args.matched_known_prior,
        open_val_ratio=float(open_val_ratio),
        transform_profile=args.transform_profile,
    )
    eval_dataset = _split_dataset(
        bundle, evaluation_split, args.image_size, args.transform_profile
    )
    loader = build_loader(eval_dataset, args.batch_size, False, args.num_workers)
    comparison_loader = None
    if comparison_split is not None:
        comparison_dataset = _split_dataset(
            bundle, comparison_split, args.image_size, args.transform_profile
        )
        comparison_loader = build_loader(
            comparison_dataset, args.batch_size, False, args.num_workers
        )
    model = build_model(
        args.num_known, backbone=args.backbone, proj_dim=args.proj_dim,
        dropout=args.dropout, pretrained=False, cifar_stem=args.cifar_stem,
        rejection_feature_dim=args.rejection_feature_dim,
        independent_rejection_encoder=bool(independent_rejection_encoder),
        independent_rejection_encoder_init=independent_rejection_encoder_init,
        independent_rejection_input_norm=independent_rejection_input_norm,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    outputs = _deterministic_outputs(model, loader, device)
    comparison_outputs = (
        _deterministic_outputs(model, comparison_loader, device)
        if comparison_loader is not None
        else None
    )
    y = (~outputs["is_known"]).astype(np.int64)
    shared = outputs["features"]
    rejection = outputs["rejection_features"]
    rejection_raw_input = outputs["rejection_raw_inputs"]
    rejection_input = outputs["rejection_inputs"]
    known_mask = outputs["labels"] >= 0
    known_labels = outputs["labels"][known_mask]
    binary_probe = {}
    if len(np.unique(y)) >= 2:
        binary_probe = {
            "shared_features": _binary_probe(shared, y, seed),
            "rejection_features": _binary_probe(rejection, y, seed),
            "shared_logits": _binary_probe(outputs["logits"], y, seed),
        }
    cross_split_probe = {}
    if comparison_outputs is not None:
        cross_split_probe = {
            "shared_features": _cross_split_multiclass_probe_accuracy(
                outputs, comparison_outputs, seed
            ),
            "rejection_features": _cross_split_multiclass_probe_accuracy(
                {**outputs, "features": outputs["rejection_features"]},
                {
                    **comparison_outputs,
                    "features": comparison_outputs["rejection_features"],
                },
                seed,
            ),
            "rejection_input": _cross_split_multiclass_probe_accuracy(
                {**outputs, "features": outputs["rejection_raw_inputs"]},
                {
                    **comparison_outputs,
                    "features": comparison_outputs["rejection_raw_inputs"],
                },
                seed,
            ),
            "processed_rejection_input": _cross_split_multiclass_probe_accuracy(
                {**outputs, "features": outputs["rejection_inputs"]},
                {
                    **comparison_outputs,
                    "features": comparison_outputs["rejection_inputs"],
                },
                seed,
            ),
        }
    summary = {
        "protocol": {
            "seed": seed,
            "split_path": str(split_path),
            "checkpoint": str(checkpoint),
            "deterministic_eval": True,
            "evaluation_split": str(evaluation_split),
            "comparison_split": comparison_split,
            "open_val_ratio": float(open_val_ratio),
            "independent_rejection_encoder": bool(independent_rejection_encoder),
            "independent_rejection_encoder_init": str(independent_rejection_encoder_init),
            "independent_rejection_input_norm": str(independent_rejection_input_norm),
            "unknown_labels_used_only_for_posthoc_diagnostics": evaluation_split == "test",
            "known_class_labels_used_only_for_posthoc_diagnostics": True,
        },
        "shapes": {
            "shared": list(shared.shape),
            "rejection_raw_input": list(rejection_raw_input.shape),
            "rejection_input": list(rejection_input.shape),
            "rejection": list(rejection.shape),
        },
        "effective_rank": {
            "shared_participation": _effective_rank(shared)[0],
            "shared_rank95": _effective_rank(shared)[1],
            "rejection_participation": _effective_rank(rejection)[0],
            "rejection_rank95": _effective_rank(rejection)[1],
        },
        "geometry": {
            "neighbor_jaccard_at_10": _neighbor_jaccard(shared, rejection, k=10),
            "shared_to_rejection_cross_r2": _cross_mapping_r2(shared, rejection, seed),
            "rejection_to_shared_cross_r2": _cross_mapping_r2(rejection, shared, seed),
        },
        "known_only": {
            "shared_feature_statistics": _feature_statistics(
                shared, outputs["labels"]
            ),
            "rejection_feature_statistics": _feature_statistics(
                rejection, outputs["labels"]
            ),
            "rejection_input_statistics": _feature_statistics(
                rejection_raw_input, outputs["labels"]
            ),
            "processed_rejection_input_statistics": _feature_statistics(
                rejection_input, outputs["labels"]
            ),
            "within_split_multiclass_probe_accuracy": {
                "shared_features": _multiclass_probe_accuracy(
                    shared[known_mask], known_labels, seed
                ),
                "rejection_features": _multiclass_probe_accuracy(
                    rejection[known_mask], known_labels, seed
                ),
                "rejection_input": _multiclass_probe_accuracy(
                    rejection_raw_input[known_mask], known_labels, seed
                ),
                "processed_rejection_input": _multiclass_probe_accuracy(
                    rejection_input[known_mask], known_labels, seed
                ),
            },
            "cross_split_multiclass_probe_accuracy": cross_split_probe,
        },
        "posthoc_binary_probe_auc": binary_probe,
        "posthoc_binary_probe_note": (
            "not_applicable_on_known_only_split"
            if len(np.unique(y)) < 2
            else "descriptive known-vs-unknown probe; never used for training or selection"
        ),
        "score_correlations": _score_correlations(detail_path, outputs),
        "cohort_sizes": {
            "known": int(outputs["is_known"].sum()),
            "unknown": int((~outputs["is_known"]).sum()),
        },
        "comparison_cohort_sizes": (
            {
                "known": int(comparison_outputs["is_known"].sum()),
                "unknown": int((~comparison_outputs["is_known"]).sum()),
            }
            if comparison_outputs is not None
            else None
        ),
    }
    return summary


def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    reports = {}
    case_items = list(args.case)
    for case_path in args.case_file:
        payload = json.loads(Path(case_path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            case_items.extend(json.dumps(item) for item in payload)
        else:
            case_items.append(json.dumps(payload))
    for item in case_items:
        case = json.loads(item)
        reports[case["name"]] = audit_one(
            args,
            device,
            case["split_path"],
            case["checkpoint"],
            case.get("detail_path"),
            int(case["seed"]),
            case.get("independent_rejection_encoder", args.independent_rejection_encoder),
            case.get(
                "independent_rejection_encoder_init",
                args.independent_rejection_encoder_init,
            ),
            case.get(
                "independent_rejection_input_norm",
                args.independent_rejection_input_norm,
            ),
            case.get("evaluation_split", args.evaluation_split),
            case.get("comparison_split", args.comparison_split),
            case.get("open_val_ratio", args.open_val_ratio),
        )
        print(case["name"], json.dumps(reports[case["name"]], ensure_ascii=False))
    save_json(args.output, reports)
    print(f"saved representation audit to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[], help="JSON object with name, seed, split_path, checkpoint, optional detail_path")
    parser.add_argument("--case-file", action="append", default=[], help="JSON file containing one case object or a list of case objects")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--num-known", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--transform-profile", choices=["imagenet", "cifar"], default="cifar")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rejection-feature-dim", type=int, default=128)
    parser.add_argument("--independent-rejection-encoder", action="store_true")
    parser.add_argument(
        "--independent-rejection-encoder-init",
        choices=["scratch_cnn", "shared_backbone", "random_backbone"],
        default="shared_backbone",
    )
    parser.add_argument(
        "--independent-rejection-input-norm",
        choices=["none", "layernorm"],
        default="none",
    )
    parser.add_argument("--cifar-stem", action="store_true")
    parser.add_argument("--matched-pool-size", type=int, default=5400)
    parser.add_argument("--matched-known-prior", type=float, default=0.2)
    parser.add_argument(
        "--evaluation-split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument(
        "--comparison-split",
        choices=["train", "val", "test"],
        default=None,
    )
    parser.add_argument("--open-val-ratio", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.case and not args.case_file:
        parser.error("at least one --case or --case-file is required")
    return args


if __name__ == "__main__":
    main(parse_args())
