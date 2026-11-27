from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel_discovery.data import build_data_bundle
from novel_discovery.joint_discovery import NovelPrototypeHead, combine_known_novel_logits
from novel_discovery.metrics import compute_fpr95, compute_oscr
from novel_discovery.models import build_model
from novel_discovery.pipeline import build_loader, calibrate_coverage_threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrospectively evaluate whether a joint known/novel head's "
            "novel-mass score actually detects held-out unknown classes."
        )
    )
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--known-coverage", type=float, default=0.95)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


@torch.no_grad()
def collect_scores(model, novel_head, loader, device):
    scores, is_known, mapped_labels, raw_labels, known_predictions = [], [], [], [], []
    model.eval()
    novel_head.eval()
    for images, mapped, raw, known_mask, _ in loader:
        output = model(images.to(device))
        novel_logits = novel_head(output["features"])
        joint_logits = combine_known_novel_logits(output["logits"], novel_logits)
        novel_mass = joint_logits.softmax(dim=-1)[:, output["logits"].size(-1) :].sum(dim=-1)
        scores.append(novel_mass.cpu())
        is_known.append(torch.as_tensor(known_mask).bool())
        mapped_labels.append(torch.as_tensor(mapped).long())
        raw_labels.append(torch.as_tensor(raw).long())
        known_predictions.append(output["logits"].argmax(dim=-1).cpu())
    return tuple(torch.cat(parts).numpy() for parts in (
        scores, is_known, mapped_labels, raw_labels, known_predictions
    ))


def summarize_scores(scores, is_known, mapped_labels, raw_labels, predictions, threshold, coverage_target):
    is_known = np.asarray(is_known, dtype=bool)
    mapped_labels = np.asarray(mapped_labels, dtype=np.int64)
    raw_labels = np.asarray(raw_labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    is_unknown = ~is_known
    predicted_unknown = scores > threshold
    retrospective_threshold, retrospective_calibration = calibrate_coverage_threshold(
        scores[is_known], coverage_target
    )
    retrospective_predicted_unknown = scores > retrospective_threshold
    fpr95 = compute_fpr95(is_unknown.astype(np.int8), scores)
    known_correct = predictions[is_known] == mapped_labels[is_known]
    oscr = compute_oscr(is_known, ~predicted_unknown, known_correct, scores)
    novel_classes = np.unique(raw_labels[is_unknown])
    per_novel_class = {}
    for cls in novel_classes:
        mask = is_unknown & (raw_labels == cls)
        per_novel_class[str(int(cls))] = {
            "count": int(mask.sum()),
            "false_accept_rate": float((~predicted_unknown[mask]).mean()),
            "mean_novel_mass": float(scores[mask].mean()),
        }
    sorted_classes = sorted(
        per_novel_class.items(), key=lambda item: item[1]["false_accept_rate"], reverse=True
    )
    return {
        "auroc_novel_mass": float(roc_auc_score(is_unknown.astype(np.int8), scores)),
        "fpr95": fpr95,
        "oscr": float(oscr),
        "threshold_from_known_validation": float(threshold),
        "target_known_coverage": float(coverage_target),
        "test_known_accept_rate": float((~predicted_unknown[is_known]).mean()),
        "test_unknown_reject_rate": float(predicted_unknown[is_unknown].mean()),
        "candidate_purity": float(is_unknown[predicted_unknown].mean()) if predicted_unknown.any() else None,
        "candidate_recall": float(predicted_unknown[is_unknown].mean()),
        "retrospective_test_known_coverage_diagnostic_only": {
            "threshold_from_test_known_labels_diagnostic_only": retrospective_threshold,
            "calibration_known_accept_rate": retrospective_calibration["known_accept_rate"],
            "known_accept_rate": float((~retrospective_predicted_unknown[is_known]).mean()),
            "unknown_reject_rate": float(retrospective_predicted_unknown[is_unknown].mean()),
            "candidate_purity": (
                float(is_unknown[retrospective_predicted_unknown].mean())
                if retrospective_predicted_unknown.any() else None
            ),
        },
        "known_accuracy_all": float(known_correct.mean()),
        "known_accuracy_after_accept": (
            float(known_correct[~predicted_unknown[is_known]].mean())
            if (~predicted_unknown[is_known]).any() else None
        ),
        "score_distribution": {
            name: {
                "mean": float(scores[mask].mean()),
                "median": float(np.median(scores[mask])),
                "q90": float(np.quantile(scores[mask], 0.90)),
            }
            for name, mask in (("known", is_known), ("unknown", is_unknown))
        },
        "most_false_accepted_novel_classes": sorted_classes[:10],
    }


def evaluate_run(run_dir: Path, device: torch.device, num_workers: int, coverage: float):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "student.pt", map_location=device, weights_only=False)
    bundle = build_data_bundle(
        config["dataset"],
        config["data_root"],
        int(config["num_known"]),
        int(config["seed"]),
        int(config["image_size"]),
        split_path=config.get("split_path"),
        limit_train=config.get("limit_train") or None,
        limit_val=config.get("limit_val") or None,
        limit_test=config.get("limit_test") or None,
        limit_discovery=config.get("limit_discovery") or None,
        discovery_pool_mode=config.get("discovery_pool_mode", "unknown"),
        mixed_known_pool_ratio=float(config.get("mixed_known_pool_ratio", 0.2)),
        known_split_mode=config.get("known_split_mode", "random"),
    )
    model = build_model(
        len(bundle.known_classes),
        backbone=config.get("student_backbone") or config.get("backbone", "resnet18"),
        proj_dim=int(config.get("proj_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    novel_head = NovelPrototypeHead(
        model.encoder.out_dim,
        int(config["joint_num_novel"]),
        temperature=float(config.get("joint_head_temperature", 0.2)),
    ).to(device)
    novel_state = checkpoint.get("joint_novel_head")
    if novel_state is None:
        raise ValueError(f"{run_dir} checkpoint has no joint novel head")
    novel_head.load_state_dict(novel_state)

    batch_size = int(config.get("batch_size", 128))
    val_loader = build_loader(bundle.val, batch_size, False, num_workers)
    test_loader = build_loader(bundle.test, batch_size, False, num_workers)
    val_scores, val_known, _, _, _ = collect_scores(model, novel_head, val_loader, device)
    test_scores, test_known, test_mapped, test_raw, test_predictions = collect_scores(
        model, novel_head, test_loader, device
    )
    if not val_known.all():
        raise ValueError("Validation loader must contain known samples only")
    threshold, validation_calibration = calibrate_coverage_threshold(
        val_scores[val_known], coverage
    )
    return {
        "run_dir": str(run_dir),
        "method_config": {
            "joint_novel_mass": config.get("joint_novel_mass", False),
            "joint_novel_neighbor_support": config.get("joint_novel_neighbor_support", False),
            "joint_novel_ema_weights": config.get("joint_novel_ema_weights", False),
            "joint_space": config.get("joint_space"),
            "alpha_joint_discovery": config.get("alpha_joint_discovery"),
            "joint_head_temperature": config.get("joint_head_temperature"),
            "device_recorded": config.get("device"),
            "seed": config.get("seed"),
            "train_val_test_limits": [
                config.get("limit_train"), config.get("limit_val"), config.get("limit_test")
            ],
        },
        "validation_known_samples": int(len(val_scores)),
        "validation_known_accept_rate_at_threshold": validation_calibration["known_accept_rate"],
        "test_samples": int(len(test_scores)),
        "test_known_samples": int(test_known.sum()),
        "test_unknown_samples": int((~test_known).sum()),
        "test": summarize_scores(
            test_scores, test_known, test_mapped, test_raw, test_predictions, threshold, coverage
        ),
    }


def main():
    args = parse_args()
    if not 0.0 < args.known_coverage < 1.0:
        raise ValueError("--known-coverage must be between 0 and 1")
    device = resolve_device(args.device)
    reports = [
        evaluate_run(Path(run_dir), device, args.num_workers, args.known_coverage)
        for run_dir in args.run_dirs
    ]
    report = {
        "device": str(device),
        "score": "softmax(concat(known_logits / known_temperature, novel_logits)).sum(novel classes)",
        "notes": [
            "Novel-mass is evaluated directly here; earlier detector comparisons used normalized_entropy_mahalanobis.",
            "Threshold calibration and FPR95 use the project's shared implementations.",
            "Test labels are used only for retrospective metrics; test-known coverage calibration is diagnostic-only, not deployable.",
            "The score's probability interpretation assumes comparable known/novel logit scales; this diagnostic does not calibrate them.",
        ],
        "runs": reports,
    }
    output = Path(args.output) if args.output else Path("analysis/novel_mass_direct_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
