"""Post-hoc feature-overlap diagnostics for a fixed baseline/variant pair.

Test labels are used only to stratify descriptive evaluation statistics. They
are never used to fit models, select thresholds, or tune the method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Allow direct execution as ``python scripts/analyze_feature_overlap.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.data import build_data_bundle, build_transforms
from novel_discovery.models import build_model
from novel_discovery.pipeline import build_loader
from novel_discovery.utils import save_json


def _extract_features(model, loader, device):
    features, labels, known_flags = [], [], []
    model.eval()
    with torch.no_grad():
        for images, mapped_labels, _raw, is_known, _indices in loader:
            output = model(images.to(device))
            features.append(F.normalize(output["features"].float(), dim=-1).cpu().numpy())
            labels.append(mapped_labels.cpu().numpy())
            known_flags.append(is_known.cpu().numpy().astype(bool))
    if not features:
        raise ValueError("Cannot extract features from an empty dataset")
    return {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels),
        "is_known": np.concatenate(known_flags),
    }


def _class_centroids(features: np.ndarray, labels: np.ndarray, num_classes: int):
    centers = []
    present = []
    for cls in range(num_classes):
        values = features[labels == cls]
        if len(values):
            center = values.mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            centers.append(center)
            present.append(cls)
    if not centers:
        raise ValueError("No known training class has an observed feature")
    return np.asarray(centers, dtype=np.float32), present


def _distribution_summary(known_distance: np.ndarray, unknown_distance: np.ndarray):
    known_distance = np.asarray(known_distance, dtype=float)
    unknown_distance = np.asarray(unknown_distance, dtype=float)
    if not len(known_distance) or not len(unknown_distance):
        raise ValueError("Both known and unknown evaluation samples are required")
    lower = float(min(known_distance.min(), unknown_distance.min()))
    upper = float(max(known_distance.max(), unknown_distance.max()))
    if upper - lower < 1e-12:
        overlap = 1.0
    else:
        bins = np.linspace(lower, upper, 51)
        known_hist = np.histogram(known_distance, bins=bins)[0].astype(float)
        unknown_hist = np.histogram(unknown_distance, bins=bins)[0].astype(float)
        known_hist /= max(float(known_hist.sum()), 1.0)
        unknown_hist /= max(float(unknown_hist.sum()), 1.0)
        overlap = float(np.minimum(known_hist, unknown_hist).sum())

    def stats(values):
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q10": float(np.quantile(values, 0.10)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "q90": float(np.quantile(values, 0.90)),
        }

    labels = np.concatenate([np.zeros(len(known_distance)), np.ones(len(unknown_distance))])
    scores = np.concatenate([known_distance, unknown_distance])
    return {
        "known_distance": stats(known_distance),
        "unknown_distance": stats(unknown_distance),
        "unknownness_auroc": float(roc_auc_score(labels, scores)),
        "histogram_overlap_0_to_1": overlap,
    }


def _diagnose_one(model, train_data, evaluation_data, device):
    known_train = train_data["is_known"]
    train_features = train_data["features"][known_train]
    train_labels = train_data["labels"][known_train]
    if np.any(train_labels < 0):
        raise ValueError("Known training samples unexpectedly have negative labels")

    means, present_classes = _class_centroids(
        train_features, train_labels, model.classifier.weight.shape[0]
    )
    classifier_prototypes = F.normalize(
        model.classifier.weight.detach().float(), dim=-1
    ).cpu().numpy()
    evaluation_features = evaluation_data["features"]
    known_mask = evaluation_data["is_known"]
    unknown_mask = ~known_mask

    classifier_similarity = evaluation_features @ classifier_prototypes.T
    centroid_similarity = evaluation_features @ means.T
    nearest_train_similarity = evaluation_features @ train_features.T
    distance_sets = {
        "classifier_prototype": 1.0 - classifier_similarity.max(axis=1),
        "empirical_class_centroid": 1.0 - centroid_similarity.max(axis=1),
        "nearest_known_train_sample": 1.0 - nearest_train_similarity.max(axis=1),
    }
    return {
        "known_training_samples": int(len(train_features)),
        "known_training_classes_observed": int(len(present_classes)),
        "evaluation_known_count": int(known_mask.sum()),
        "evaluation_unknown_count": int(unknown_mask.sum()),
        "distances": {
            name: _distribution_summary(values[known_mask], values[unknown_mask])
            for name, values in distance_sets.items()
        },
    }


def analyze(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    bundle = build_data_bundle(
        "cifar100",
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        split_path=args.split_path,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        limit_discovery=args.limit_discovery,
        discovery_pool_mode="unknown",
    )
    # The training dataset normally has random augmentation. For a stable
    # support bank, use the same deterministic evaluation transform as test.
    train_subset = bundle.train
    while hasattr(train_subset, "dataset"):
        train_subset = train_subset.dataset
    if not hasattr(train_subset, "base"):
        raise TypeError("Expected a CIFAR-style wrapped dataset with a base dataset")
    train_subset.base.transform = build_transforms(args.image_size, train=False)

    train_features_loader = build_loader(
        bundle.train, args.batch_size, False, args.num_workers
    )
    evaluation_loader = build_loader(
        bundle.test, args.batch_size, False, args.num_workers
    )
    common = {
        "dataset": "cifar100",
        "split_path": str(args.split_path),
        "seed": int(args.seed),
        "num_known": int(args.num_known),
        "image_size": int(args.image_size),
        "limit_train": int(args.limit_train),
        "limit_val": int(args.limit_val),
        "limit_test": int(args.limit_test),
        "note": "test labels only stratify descriptive diagnostics; no fitting or threshold selection",
    }
    results = {"protocol": common, "runs": {}}
    for name, checkpoint_path in (
        ("baseline", args.baseline_ckpt),
        ("boundary_variant", args.variant_ckpt),
    ):
        model = build_model(
            len(bundle.known_classes),
            backbone=args.backbone,
            proj_dim=args.proj_dim,
            dropout=args.dropout,
            pretrained=False,
            cifar_stem=args.cifar_stem,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        train_data = _extract_features(model, train_features_loader, device)
        evaluation_data = _extract_features(model, evaluation_loader, device)
        results["runs"][name] = _diagnose_one(
            model, train_data, evaluation_data, device
        )
        print(f"{name}: {results['runs'][name]}")
    save_json(args.output, results)
    print(f"saved diagnostics to {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--num-known", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    # Full-data is the safe default for a post-hoc diagnostic. Use explicit
    # positive limits when running a smoke test.
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--limit-discovery", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cifar-stem", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--variant-ckpt", required=True)
    parser.add_argument(
        "--output",
        default="./analysis/hybrid_unknown_boundary_feature_diagnostics_s42.json",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    analyze(parse_args())
