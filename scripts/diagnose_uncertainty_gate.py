"""Audit whether a trained uncertainty score is a useful soft gate on a mixed pool.

Pool labels are used only for this post-hoc diagnostic. They are never passed
to model training, loss construction, or threshold fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle
from novel_discovery.models import build_model
from novel_discovery.pipeline import build_loader


def _summary(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "q10": float(np.quantile(values, 0.10)) if values.size else None,
        "median": float(np.quantile(values, 0.50)) if values.size else None,
        "q90": float(np.quantile(values, 0.90)) if values.size else None,
    }


def diagnose(args) -> dict:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    config_path = Path(args.checkpoint).parent / "train_student_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle = build_data_bundle(
        "cifar100", args.data_root, args.num_known, args.seed, args.image_size,
        split_path=args.split_path, limit_train=0, limit_val=0, limit_test=0,
        limit_discovery=0, discovery_pool_mode="mixed",
        mixed_known_pool_ratio=float(config.get("mixed_known_pool_ratio", 0.2)),
        known_split_mode=config.get("known_split_mode", "random"),
    )
    model = build_model(
        len(bundle.known_classes),
        backbone=config.get("student_backbone") or config.get("backbone", "resnet18"),
        proj_dim=int(config.get("proj_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        pretrained=False,
        cifar_stem=bool(config.get("cifar_stem", False)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = build_loader(bundle.discovery_pool, args.batch_size, False, args.num_workers)
    uncertainty, known_flags, max_probs = [], [], []
    with torch.no_grad():
        for images, _mapped, _raw, is_known, _index in loader:
            out = model(images.to(device))
            uncertainty.append(out["uncertainty"].float().cpu().numpy())
            known_flags.append(is_known.numpy().astype(bool))
            max_probs.append(out["logits"].softmax(dim=-1).max(dim=-1).values.float().cpu().numpy())

    u = np.concatenate(uncertainty)
    known = np.concatenate(known_flags)
    msp = np.concatenate(max_probs)
    unknown = ~known
    if known.sum() == 0 or unknown.sum() == 0:
        raise ValueError("Diagnostic requires both known and novel examples in mixed pool")

    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "seed": int(args.seed),
            "pool_size": int(len(u)),
            "known_count": int(known.sum()),
            "unknown_count": int(unknown.sum()),
            "known_prior": float(known.mean()),
            "labels_used_for": "post-hoc gate audit only; not training or calibration",
        },
        "uncertainty": {
            "unknownness_auroc": float(roc_auc_score(unknown.astype(int), u)),
            "known": _summary(u[known]),
            "unknown": _summary(u[unknown]),
            "mean_weight": float(u.mean()),
            "weight_ess_fraction": float(u.sum() ** 2 / (len(u) * np.square(u).sum() + 1e-12)),
        },
        "msp": {
            "unknownness_auroc": float(roc_auc_score(unknown.astype(int), 1.0 - msp)),
            "known_mean": float(msp[known].mean()),
            "unknown_mean": float(msp[unknown].mean()),
        },
        "uncertainty_weighted_margin_gate": {},
        "normalized_power_weight_audit": {},
    }
    for power in (1.0, 2.0, 4.0):
        weights = np.power(np.clip(u, 1e-8, 1.0), power)
        report["normalized_power_weight_audit"][str(power)] = {
            "ess_fraction": float(weights.sum() ** 2 / (len(weights) * np.square(weights).sum() + 1e-12)),
            "unknown_weight_mass_fraction": float(weights[unknown].sum() / weights.sum()),
            "known_weight_mass_fraction": float(weights[known].sum() / weights.sum()),
            "unknown_weight_mass_lift_over_pool_prior": float(
                (weights[unknown].sum() / weights.sum()) / unknown.mean()
            ),
        }
    for ratio in (0.10, 0.25, 0.50):
        cutoff = float(np.quantile(u, 1.0 - ratio))
        selected = u >= cutoff
        report["uncertainty_weighted_margin_gate"][str(ratio)] = {
            "selected_count": int(selected.sum()),
            "unknown_precision": float(unknown[selected].mean()) if selected.any() else None,
            "known_contamination": float(known[selected].mean()) if selected.any() else None,
            "unknown_recall": float(unknown[selected].sum() / unknown.sum()),
            "known_selected_fraction": float(known[selected].sum() / known.sum()),
            "mean_uncertainty_selected": float(u[selected].mean()) if selected.any() else None,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--num-known", type=int, default=60)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = diagnose(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
