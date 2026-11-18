"""Audit class-conditional local density as a post-hoc unknown score.

The hypothesis is that a semantically hard unknown can receive a high known
class confidence while still being an outlier inside the predicted known
class.  All score normalization and thresholds are fitted from known train /
calibration data.  Test labels are read only to report retrospective metrics.

This script deliberately does not alter the training or discovery pipeline.
It is a screening experiment: only a consistent improvement across the four
semantic-hard splits justifies a later formal ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.data import build_data_bundle, build_transforms
from novel_discovery.models import build_model
from novel_discovery.pipeline import build_loader


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract(model, loader, device):
    features, labels, predicted, known = [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, mapped, _raw, is_known, _indices in loader:
            output = model(images.to(device))
            features.append(F.normalize(output["features"].float(), dim=-1).cpu().numpy())
            labels.append(mapped.cpu().numpy().astype(int))
            predicted.append(output["logits"].argmax(dim=1).cpu().numpy().astype(int))
            known.append(is_known.cpu().numpy().astype(bool))
    return {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels),
        "predicted": np.concatenate(predicted),
        "known": np.concatenate(known),
    }


def _knn(query, bank, bank_labels, k, batch_size=256):
    labels, similarities = [], []
    for start in range(0, len(query), batch_size):
        sim = query[start : start + batch_size] @ bank.T
        top = np.argpartition(sim, -k, axis=1)[:, -k:]
        rows = np.arange(len(top))[:, None]
        order = np.argsort(sim[rows, top], axis=1)[:, ::-1]
        top = top[rows, order]
        labels.append(bank_labels[top])
        similarities.append(sim[rows, top])
    return np.concatenate(labels), np.concatenate(similarities)


def _class_nearest_distance(query, bank, bank_labels, predicted, class_radii):
    distances = np.empty(len(query), dtype=np.float64)
    ratios = np.empty(len(query), dtype=np.float64)
    for index, (vector, cls) in enumerate(zip(query, predicted)):
        candidates = bank[bank_labels == cls]
        if len(candidates) == 0:
            distances[index] = np.inf
            ratios[index] = np.inf
            continue
        nearest = 1.0 - float(np.max(candidates @ vector))
        distances[index] = nearest
        ratios[index] = nearest / max(float(class_radii.get(int(cls), 1.0)), 1e-6)
    return distances, ratios


def _class_radii(bank, bank_labels, quantile=0.95):
    radii = {}
    for cls in np.unique(bank_labels):
        values = bank[bank_labels == cls]
        if len(values) < 2:
            radii[int(cls)] = 1.0
            continue
        sim = values @ values.T
        np.fill_diagonal(sim, -np.inf)
        nearest = 1.0 - np.max(sim, axis=1)
        radii[int(cls)] = float(np.quantile(nearest, quantile))
    return radii


def _reference_scores(reference, bank, bank_labels, k):
    predicted = reference["predicted"]
    distances, ratios = _class_nearest_distance(
        reference["features"], bank, bank_labels, predicted, _class_radii(bank, bank_labels)
    )
    neighbor_labels, _ = _knn(reference["features"], bank, bank_labels, k)
    support = (neighbor_labels == predicted[:, None]).mean(axis=1)
    entropy = np.zeros(len(reference["features"]), dtype=np.float64)
    for index, row in enumerate(neighbor_labels):
        counts = np.bincount(row, minlength=int(bank_labels.max()) + 1)
        probs = counts[counts > 0] / len(row)
        entropy[index] = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(k))
    return {
        "class_distance": distances,
        "class_relative_distance": ratios,
        "negative_predicted_class_support": 1.0 - support,
        "neighborhood_entropy": entropy,
    }


def _zscore(values, reference):
    mean = float(np.mean(reference))
    std = float(np.std(reference))
    return (np.asarray(values, dtype=np.float64) - mean) / max(std, 1e-8)


def _metrics(
    known,
    score,
    calibration_score,
    true_label,
    predicted,
    baseline_score=None,
    threshold_override=None,
):
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(np.quantile(calibration_score, 0.95, method="higher"))
    )
    accepted = score <= threshold
    unknown = ~known
    result = {
        "auroc": float(roc_auc_score(unknown, score)),
        "aupr": float(average_precision_score(unknown, score)),
        "threshold_from_known_calibration_95": threshold,
        "known_acceptance": float(accepted[known].mean()),
        "unknown_rejection_at_known_calibration_95": float((~accepted[unknown]).mean()),
        "accepted_known_classification_accuracy": float(
            (predicted[known & accepted] == true_label[known & accepted]).mean()
        ),
        "all_known_classification_accuracy": float((predicted[known] == true_label[known]).mean()),
    }
    if baseline_score is not None:
        result["delta_auroc_vs_baseline"] = result["auroc"] - float(roc_auc_score(unknown, baseline_score))
    return result


def _normalization_report(reference, candidate):
    return {
        key: {
            "reference_mean": float(np.mean(value)),
            "reference_std": float(np.std(value)),
            "reference_p95": float(np.quantile(value, 0.95)),
        }
        for key, value in reference.items()
    }


def analyze_run(run_dir: Path, data_root: Path, device: torch.device, k: int):
    discover_config = _read_json(run_dir / "discover_config.json")
    detail = _read_json(run_dir / "discovery_detail.json")
    train_config = _read_json(run_dir / "config.json")
    seed = int(discover_config["seed"])
    split_path = Path(discover_config["split_path"])
    if not split_path.exists():
        split_path = PROJECT_ROOT / "splits" / "semantic_hard" / f"seed{seed}" / split_path.name
    bundle = build_data_bundle(
        "cifar100",
        str(data_root),
        int(discover_config["num_known"]),
        seed,
        int(discover_config["image_size"]),
        split_path=str(split_path),
        limit_train=0,
        limit_val=0,
        limit_test=0,
        limit_discovery=0,
        discovery_pool_mode=discover_config.get("discovery_pool_mode", "mixed"),
        mixed_known_pool_ratio=float(discover_config.get("mixed_known_pool_ratio", 0.2)),
        matched_discovery_pool=bool(discover_config.get("matched_discovery_pool", True)),
        matched_pool_size=int(discover_config.get("matched_pool_size", 5400)),
        matched_known_prior=float(discover_config.get("matched_known_prior", 0.2)),
        calibration_ratio=float(discover_config.get("calibration_ratio", 0.0)),
        transform_profile=str(discover_config.get("transform_profile", "imagenet")),
    )
    # Reproduce the deterministic evaluation transform used by the stored run.
    for dataset in (bundle.train, bundle.calibration, bundle.test):
        subset = dataset
        while hasattr(subset, "dataset"):
            subset = subset.dataset
        if hasattr(subset, "base"):
            subset.base.transform = build_transforms(int(discover_config["image_size"]), train=False)
    loader_args = (int(discover_config.get("batch_size", 64)), False, int(discover_config.get("num_workers", 0)))
    train_loader = build_loader(bundle.train, *loader_args)
    calibration_loader = build_loader(bundle.calibration, *loader_args)
    test_loader = build_loader(bundle.test, *loader_args)

    model = build_model(
        int(discover_config["num_known"]),
        backbone=str(train_config.get("student_backbone", train_config.get("backbone", "resnet18"))),
        proj_dim=int(train_config.get("proj_dim", 128)),
        dropout=float(train_config.get("dropout", 0.2)),
        rejection_feature_dim=int(train_config.get("rejection_feature_dim", 0)),
        independent_rejection_encoder=bool(train_config.get("independent_rejection_encoder", False)),
        independent_rejection_encoder_init=str(train_config.get("independent_rejection_encoder_init", "shared_backbone")),
        independent_rejection_input_norm=str(train_config.get("independent_rejection_input_norm", "none")),
        pretrained=False,
        cifar_stem=bool(train_config.get("cifar_stem", False)),
    ).to(device)
    checkpoint = torch.load(run_dir.parent / run_dir.name.replace("_discover", "") / "student.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train = _extract(model, train_loader, device)
    calibration = _extract(model, calibration_loader, device)
    test = _extract(model, test_loader, device)

    known_train = train["known"]
    bank = train["features"][known_train]
    bank_labels = train["labels"][known_train]
    radii = _class_radii(bank, bank_labels)
    known_calibration = calibration["known"]
    test_pred = np.asarray(detail["pred_class"], dtype=int)
    test_known = np.asarray(detail["true_known"], dtype=bool)
    test_true_label = np.asarray(detail["true_label"], dtype=int)
    if not np.array_equal(test_known, test["known"]):
        raise ValueError(f"stored detail order does not match deterministic test loader: {run_dir}")

    calibration_features = calibration["features"][known_calibration]
    calibration_pred = calibration["predicted"][known_calibration]
    calibration_probe = {"features": calibration_features, "predicted": calibration_pred}
    test_probe = {"features": test["features"], "predicted": test_pred}
    calibration_signals = _reference_scores(calibration_probe, bank, bank_labels, k)
    test_signals = _reference_scores(test_probe, bank, bank_labels, k)
    # The baseline is the score already produced by the formal run. New
    # signals are standardized only with known calibration data.
    baseline_test = np.asarray(detail["score"], dtype=np.float64)
    calibration_report_path = run_dir / "calibration_report.json"
    calibration_report = _read_json(calibration_report_path)
    calibrated_threshold = float(calibration_report["calibrated_threshold"])
    # The formal run already calibrated its operating threshold on the known
    # validation subset. Reuse that saved value instead of deriving a test-set
    # threshold from labels or scores.
    component = {"class_relative_distance": 0.5, "negative_predicted_class_support": 0.25, "neighborhood_entropy": 0.25}
    fused_test = sum(weight * _zscore(test_signals[name], calibration_signals[name]) for name, weight in component.items())
    fused_calibration = sum(weight * _zscore(calibration_signals[name], calibration_signals[name]) for name, weight in component.items())

    results = {
        "baseline_formal_score": _metrics(
            test_known,
            baseline_test,
            np.asarray(detail["score"])[test_known],
            test_true_label,
            test_pred,
            threshold_override=calibrated_threshold,
        ),
        "class_distance": _metrics(
            test_known,
            test_signals["class_distance"],
            calibration_signals["class_distance"],
            test_true_label,
            test_pred,
            baseline_test,
        ),
        "class_relative_distance": _metrics(
            test_known,
            test_signals["class_relative_distance"],
            calibration_signals["class_relative_distance"],
            test_true_label,
            test_pred,
            baseline_test,
        ),
        "negative_predicted_class_support": _metrics(
            test_known,
            test_signals["negative_predicted_class_support"],
            calibration_signals["negative_predicted_class_support"],
            test_true_label,
            test_pred,
            baseline_test,
        ),
        "neighborhood_entropy": _metrics(
            test_known,
            test_signals["neighborhood_entropy"],
            calibration_signals["neighborhood_entropy"],
            test_true_label,
            test_pred,
            baseline_test,
        ),
        "fixed_fusion": _metrics(
            test_known,
            fused_test,
            fused_calibration,
            test_true_label,
            test_pred,
            baseline_test,
        ),
    }
    return {
        "run": str(run_dir),
        "seed": seed,
        "split_path": str(split_path),
        "k": int(k),
        "bank_count": int(len(bank)),
        "calibration_known_count": int(known_calibration.sum()),
        "test_count": int(len(test_known)),
        "class_radii": {str(key): value for key, value in radii.items()},
        "normalization": _normalization_report(calibration_signals, test_signals),
        "metrics": results,
    }


def summarize(rows):
    methods = list(rows[0]["metrics"])
    summary = {}
    for method in methods:
        keys = list(rows[0]["metrics"][method])
        summary[method] = {}
        for key in keys:
            values = [row["metrics"][method][key] for row in rows if isinstance(row["metrics"][method].get(key), (int, float))]
            if values:
                summary[method][key] = float(np.mean(values))
    baseline = np.asarray([row["metrics"]["baseline_formal_score"]["auroc"] for row in rows])
    for method in methods:
        if method == "baseline_formal_score":
            continue
        values = np.asarray([row["metrics"][method]["auroc"] for row in rows])
        summary[method]["direction_consistent_auroc_improvement"] = bool(np.all(values > baseline))
        summary[method]["mean_auroc_delta_vs_baseline"] = float(np.mean(values - baseline))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "analysis" / "classconditional_local_density_4split.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "analysis" / "classconditional_local_density_4split_20261112.md")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device != "cuda" else torch.device("cuda")
    rows = []
    for seed in (2121, 2124, 2125, 2127):
        run = args.runs_root / f"rejection_ce_decay_full_s{seed}_static_control_discover"
        print(f"analyzing {run}", flush=True)
        rows.append(analyze_run(run, args.data_root, device, args.k))
    payload = {"warning": "Retrospective diagnostic. Test labels are used only for metrics; no test labels are used to fit scores, weights, or thresholds.", "runs": rows, "summary": summarize(rows), "decision_rule": {"required": "consistent AUROC improvement in all four splits, mean delta >= 0.01, and no decrease in unknown rejection at known-calibration 95% coverage", "action": "Do not modify the formal pipeline unless the rule is satisfied."}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Class-conditional local-density diagnostic", "", payload["warning"], "", "## Per-split results", "", "| split | method | AUROC | AUPR | unknown rejection | delta AUROC vs baseline |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        for method, values in row["metrics"].items():
            lines.append(f"| {row['seed']} | {method} | {values['auroc']:.4f} | {values['aupr']:.4f} | {values['unknown_rejection_at_known_calibration_95']:.4f} | {values.get('delta_auroc_vs_baseline', 0.0):+.4f} |")
    lines.extend(["", "## Four-split summary", "", "| method | mean AUROC | mean AUPR | mean unknown rejection | mean AUROC delta | direction consistent |", "|---|---:|---:|---:|---:|---|"])
    for method, values in payload["summary"].items():
        lines.append(f"| {method} | {values.get('auroc', float('nan')):.4f} | {values.get('aupr', float('nan')):.4f} | {values.get('unknown_rejection_at_known_calibration_95', float('nan')):.4f} | {values.get('mean_auroc_delta_vs_baseline', 0.0):+.4f} | {values.get('direction_consistent_auroc_improvement', False)} |")
    lines.extend(["", "## Decision", "", "The fixed fusion is considered a candidate only if it improves AUROC in all four splits, has mean AUROC gain at least 0.01, and does not reduce unknown rejection at the calibration operating point. Otherwise this direction remains a diagnostic finding and is not promoted to the formal algorithm."])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {args.output}")
    print(f"saved {args.report}")


if __name__ == "__main__":
    main()
