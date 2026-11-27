"""Post-hoc kNN neighborhood audit for known errors and unknown false accepts.

Test labels are used only to form descriptive cohorts. They are not used to
fit a model, calibrate a threshold, or select a method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.data import build_data_bundle, build_transforms
from novel_discovery.models import build_model
from novel_discovery.pipeline import build_loader
from novel_discovery.utils import save_json


def _extract(model, loader, device):
    features, labels, raw_labels, known_flags = [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, mapped, raw, is_known, _indices in loader:
            output = model(images.to(device))
            features.append(F.normalize(output["features"].float(), dim=-1).cpu().numpy())
            labels.append(mapped.cpu().numpy())
            raw_labels.append(raw.cpu().numpy())
            known_flags.append(is_known.cpu().numpy().astype(bool))
    return {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels).astype(int),
        "raw_labels": np.concatenate(raw_labels).astype(int),
        "is_known": np.concatenate(known_flags),
    }


def _knn_neighbors(query, bank, bank_labels, k, batch_size):
    if k < 1 or k > len(bank):
        raise ValueError(f"k must be in [1, {len(bank)}], got {k}")
    neighbor_labels, neighbor_similarities = [], []
    for start in range(0, len(query), batch_size):
        similarities = query[start : start + batch_size] @ bank.T
        top = np.argpartition(similarities, -k, axis=1)[:, -k:]
        row = np.arange(len(top))[:, None]
        order = np.argsort(similarities[row, top], axis=1)[:, ::-1]
        top = top[row, order]
        neighbor_labels.append(bank_labels[top])
        neighbor_similarities.append(similarities[row, top])
    return np.concatenate(neighbor_labels), np.concatenate(neighbor_similarities)


def _summarize_group(mask, pred, neighbor_labels, neighbor_similarities, k):
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}
    labels = neighbor_labels[mask]
    pred_labels = pred[mask, None]
    counts = (labels == pred_labels).sum(axis=1)
    class_entropy = []
    for row in labels:
        _, row_counts = np.unique(row, return_counts=True)
        probs = row_counts / row_counts.sum()
        class_entropy.append(float(-(probs * np.log(probs + 1e-12)).sum()))
    return {
        "count": count,
        "mean_nearest_cosine_distance": float(np.mean(1.0 - neighbor_similarities[mask, 0])),
        "median_nearest_cosine_distance": float(np.median(1.0 - neighbor_similarities[mask, 0])),
        "top1_neighbor_matches_predicted_class": float(np.mean(labels[:, 0] == pred[mask])),
        "mean_topk_purity_for_predicted_class": float(np.mean(counts / k)),
        "mean_topk_neighbor_class_entropy": float(np.mean(class_entropy)),
        "mean_topk_neighbor_class_entropy_normalized": float(
            np.mean(class_entropy) / np.log(k) if k > 1 else 0.0
        ),
    }


def _finite_mean(values, mask):
    """Return a JSON-safe mean for an optional per-sample diagnostic vector."""
    values = np.asarray(values, dtype=float)
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    return float(selected.mean()) if len(selected) else None


def _score_component_summary(detail, mask):
    """Summarize detector components without assuming their score direction.

    The audit is descriptive: some components are distances (higher means more
    unknown), while others are model outputs whose direction depends on the
    fitted rejector.  We therefore report means only and do not combine or
    threshold them here.
    """
    names = (
        "score",
        "feature_rejector_score",
        "proto_dist",
        "knn_distance",
        "knn_predicted_class_distance",
        "knn_predicted_class_relative_distance",
        "knn_min_class_distance",
        "knn_min_class_relative_distance",
        "knn_predicted_class_support",
        "entropy",
        "feature_norm",
    )
    return {
        name: _finite_mean(detail.get(name), mask)
        for name in names
        if name in detail
    }


def _analyze_checkpoint(args, bundle, train_loader, test_loader, known_classes, name, ckpt_path, detail_dir, device):
    detail_path = Path(detail_dir) / "discovery_detail.json"
    report_path = Path(detail_dir) / "discovery_report.json"
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    model = build_model(
        args.num_known,
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        rejection_feature_dim=args.rejection_feature_dim,
        pretrained=False,
        cifar_stem=args.cifar_stem,
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_data = _extract(model, train_loader, device)
    test_data = _extract(model, test_loader, device)

    known_train = train_data["is_known"]
    bank = train_data["features"][known_train]
    bank_labels = train_data["labels"][known_train]
    if np.any(bank_labels < 0):
        raise ValueError("Known training feature bank contains negative labels")

    scores = np.asarray(detail["score"], dtype=float)
    pred = np.asarray(detail["pred_class"], dtype=int)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    true_label = np.asarray(detail["true_label"], dtype=int)
    threshold = float(report["score_distribution"]["threshold"])
    if not (len(scores) == len(test_data["features"]) == len(pred)):
        raise ValueError("Detector details and deterministic test loader have different lengths")
    if not np.array_equal(true_known, test_data["is_known"]):
        raise ValueError("Detector details and test loader known flags do not align")
    if not np.array_equal(np.asarray(detail["raw_labels"]), test_data["raw_labels"]):
        raise ValueError("Detector details and test loader sample order/raw labels do not align")

    k = min(args.k, len(bank))
    neighbor_mapped, neighbor_similarity = _knn_neighbors(
        test_data["features"], bank, bank_labels, k, args.query_batch_size
    )
    accepted = scores <= threshold
    known_correct = true_known & (pred == true_label)
    known_wrong = true_known & (pred != true_label)
    groups = {
        "known_correct_accepted": known_correct & accepted,
        "known_wrong_accepted": known_wrong & accepted,
        "known_correct_rejected": known_correct & ~accepted,
        "known_wrong_rejected": known_wrong & ~accepted,
        "unknown_false_accept": ~true_known & accepted,
        "unknown_rejected": ~true_known & ~accepted,
    }
    summaries = {
        group: _summarize_group(mask, pred, neighbor_mapped, neighbor_similarity, k)
        for group, mask in groups.items()
    }

    raw_known_classes = np.asarray(known_classes, dtype=int)
    neighbor_raw = raw_known_classes[neighbor_mapped]
    pred_raw = raw_known_classes[pred]
    unknown_mask = ~true_known
    unknown_class_rows = []
    for raw_class in np.unique(test_data["raw_labels"][unknown_mask]):
        cls_mask = unknown_mask & (test_data["raw_labels"] == raw_class)
        false_mask = cls_mask & accepted
        if not false_mask.any():
            continue
        nearest_counts = np.bincount(neighbor_raw[false_mask, 0], minlength=100)
        prediction_counts = np.bincount(pred_raw[false_mask], minlength=100)
        unknown_class_rows.append({
            "raw_unknown_class": int(raw_class),
            "count": int(cls_mask.sum()),
            "false_accept_count": int(false_mask.sum()),
            "false_accept_rate": float(false_mask.sum() / cls_mask.sum()),
            "accepted_unknown_score_components": _score_component_summary(
                detail, false_mask
            ),
            "rejected_unknown_score_components": _score_component_summary(
                detail, cls_mask & ~accepted
            ),
            "top_nearest_known_train_classes": [
                {"raw_class": int(c), "count": int(nearest_counts[c])}
                for c in np.argsort(nearest_counts)[::-1][:5]
                if nearest_counts[c] > 0
            ],
            "top_predicted_known_classes": [
                {"raw_class": int(c), "count": int(prediction_counts[c])}
                for c in np.argsort(prediction_counts)[::-1][:5]
                if prediction_counts[c] > 0
            ],
        })
    unknown_class_rows.sort(key=lambda row: row["false_accept_rate"], reverse=True)

    false_accept = groups["unknown_false_accept"]
    topk_prediction_support = (neighbor_mapped == pred[:, None]).mean(axis=1)
    false_accept_nearest_raw = neighbor_raw[false_accept, 0] if false_accept.any() else np.asarray([], dtype=int)
    false_accept_nearest_counts = np.bincount(false_accept_nearest_raw, minlength=100)
    return {
        "checkpoint": str(ckpt_path),
        "detector_detail": str(detail_path),
        "protocol": {
            "dataset": "cifar100",
            "split_path": str(args.split_path),
            "seed": args.seed,
            "known_train_count": int(known_train.sum()),
            "test_count": int(len(scores)),
            "k": k,
            "discovery_pool_mode": "mixed",
            "matched_pool_size": int(args.matched_pool_size),
            "matched_known_prior": float(args.matched_known_prior),
            "mixed_known_pool_ratio": float(args.mixed_known_pool_ratio),
            "threshold": threshold,
            "threshold_source": "existing validation-calibrated discovery_report; not refit",
            "note": "test labels are used only for post-hoc cohort summaries",
        },
        "cohorts": summaries,
        "unknown_false_accept_local_support": {
            "count": int(false_accept.sum()),
            "mean_topk_neighbor_support_for_predicted_class": float(topk_prediction_support[false_accept].mean()) if false_accept.any() else None,
            "score_components": _score_component_summary(detail, false_accept),
            "unknown_rejected_score_components": _score_component_summary(
                detail, groups["unknown_rejected"]
            ),
            "top_nearest_known_train_classes_over_all_false_accepts": [
                {"raw_class": int(c), "count": int(false_accept_nearest_counts[c])}
                for c in np.argsort(false_accept_nearest_counts)[::-1][:10]
                if false_accept_nearest_counts[c] > 0
            ],
            "unknown_class_breakdown": unknown_class_rows,
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
        "cifar100", args.data_root, args.num_known, args.seed, args.image_size,
        split_path=args.split_path, limit_train=0, limit_val=0, limit_test=0,
        limit_discovery=0, discovery_pool_mode="mixed",
        mixed_known_pool_ratio=args.mixed_known_pool_ratio,
        matched_discovery_pool=True,
        matched_pool_size=args.matched_pool_size,
        matched_known_prior=args.matched_known_prior,
    )
    subset = bundle.train
    while hasattr(subset, "dataset"):
        subset = subset.dataset
    if not hasattr(subset, "base"):
        raise TypeError("Expected a CIFAR-style wrapped dataset with a base dataset")
    subset.base.transform = build_transforms(args.image_size, train=False)
    train_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
    test_loader = build_loader(bundle.test, args.batch_size, False, args.num_workers)

    outputs = {}
    for name, ckpt, detail_dir in (
        (args.first_name, args.five_epoch_ckpt, args.five_epoch_detail_dir),
        (args.second_name, args.ten_epoch_ckpt, args.ten_epoch_detail_dir),
    ):
        outputs[name] = _analyze_checkpoint(
            args, bundle, train_loader, test_loader, bundle.known_classes,
            name, ckpt, detail_dir, device,
        )
        print(name, outputs[name]["cohorts"])
    save_json(args.output, outputs)
    print(f"saved neighborhood audit to {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--num-known", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--rejection-feature-dim",
        type=int,
        default=0,
        help="Must match the checkpoint's rejection branch dimension.",
    )
    parser.add_argument("--cifar-stem", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mixed-known-pool-ratio", type=float, default=0.2)
    parser.add_argument("--matched-pool-size", type=int, default=5400)
    parser.add_argument("--matched-known-prior", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--five-epoch-ckpt", required=True)
    parser.add_argument("--five-epoch-detail-dir", required=True)
    parser.add_argument("--ten-epoch-ckpt", required=True)
    parser.add_argument("--ten-epoch-detail-dir", required=True)
    parser.add_argument(
        "--first-name",
        default="five_epoch",
        help="Report key for the first checkpoint (legacy default: five_epoch).",
    )
    parser.add_argument(
        "--second-name",
        default="ten_epoch",
        help="Report key for the second checkpoint (legacy default: ten_epoch).",
    )
    parser.add_argument("--output", default="./analysis/error_neighborhoods_s123_20261011.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    analyze(parse_args())
