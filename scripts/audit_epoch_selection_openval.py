from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, dataset_source_indices
from novel_discovery.metrics import compute_aupr, compute_auroc, compute_fpr95
from novel_discovery.models import build_model
from novel_discovery.pipeline import (
    attach_knn_distances,
    build_loader,
    collect_knn_feature_bank,
    compute_open_score,
    extract_outputs,
    fit_score_normalization,
)
from train import clone_dataset_with_eval_transform, load_checkpoint


SPLIT = ROOT / "splits_cifar100_protocols" / "cifar100_60_40_semantic_isolated.json"


def evaluate(model, loader, device, bank, normalization, prototypes):
    output = extract_outputs(model, loader, device, mc_samples=4)
    attach_knn_distances(output, bank, k=10, feature_key="features")
    score, _ = compute_open_score(
        output,
        prototypes=prototypes,
        score_mode="normalized_entropy_min_class_knn",
        normalization=normalization,
    )
    labels = (np.asarray(output["is_known"]) == 0).astype(np.int64)
    return {
        "auroc": compute_auroc(labels, score),
        "aupr": compute_aupr(labels, score),
        "fpr95": compute_fpr95(labels, score),
        "known_accuracy": float(
            (np.asarray(output["logits"])[output["is_known"] == 1].argmax(axis=1)
             == np.asarray(output["labels"])[output["is_known"] == 1]).mean()
        ),
        "known_score_mean": float(np.asarray(score)[output["is_known"] == 1].mean()),
        "unknown_score_mean": float(np.asarray(score)[output["is_known"] == 0].mean()),
        "score_ks": float(
            __import__("scipy").stats.ks_2samp(
                np.asarray(score)[output["is_known"] == 1],
                np.asarray(score)[output["is_known"] == 0],
            ).statistic
        ),
        "unknown_below_known_q95": float(
            (np.asarray(score)[output["is_known"] == 0]
             <= np.quantile(np.asarray(score)[output["is_known"] == 1], 0.95)).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or ROOT / "runs" / f"checkpoint_selection_openval_s{args.seed}"
    output_path = args.output or ROOT / "analysis" / f"checkpoint_selection_openval_s{args.seed}_20261012.json"
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_data_bundle(
        "cifar100",
        str(ROOT / "data"),
        num_known=60,
        seed=args.seed,
        image_size=64,
        split_path=str(SPLIT),
        discovery_pool_mode="mixed",
        open_val_ratio=0.1,
        calibration_ratio=0.25,
        matched_discovery_pool=True,
        matched_pool_size=5400,
        matched_known_prior=0.2,
        transform_profile="cifar",
    )
    train_eval = clone_dataset_with_eval_transform(bundle.train, 64, "cifar")
    train_loader = build_loader(train_eval, 64, False, 0)
    calibration_loader = build_loader(bundle.calibration, 64, False, 0)
    open_val_loader = build_loader(bundle.open_val, 64, False, 0)
    calibration_indices = set(dataset_source_indices(bundle.calibration))
    open_val_indices = set(dataset_source_indices(bundle.open_val))
    calibration_openval_overlap = len(calibration_indices & open_val_indices)
    if calibration_openval_overlap:
        raise RuntimeError(
            "Calibration and open-validation samples overlap; refusing selection audit."
        )
    results = {"protocol": {
        "seed": args.seed,
        "score_mode": "normalized_entropy_min_class_knn",
        "mc_samples": 4,
        "open_val_ratio": 0.1,
        "calibration_ratio": 0.25,
        "calibration_vs_open_val_index_overlap": calibration_openval_overlap,
        "test_used_for_selection": False,
        "device": str(device),
    }, "epochs": {}}

    for epoch in range(1, 6):
        checkpoint = run_dir / f"student_epoch_{epoch}.pt"
        model = build_model(
            60,
            backbone="resnet18",
            proj_dim=128,
            dropout=0.2,
            pretrained=False,
        ).to(device)
        ckpt = load_checkpoint(
            model,
            checkpoint,
            device,
            expected_known_classes=bundle.known_classes,
            expected_novel_classes=bundle.novel_classes,
        )
        prototypes = ckpt.get("prototypes")
        if torch.is_tensor(prototypes):
            prototypes = prototypes.detach().cpu().numpy()
        # The embedding changes at every epoch: never reuse a bank from a
        # different checkpoint when measuring kNN distances.
        bank = collect_knn_feature_bank(
            model,
            train_loader,
            device,
            feature_key="features",
            support_k=10,
            support_quantile=0.95,
        )
        calibration = extract_outputs(model, calibration_loader, device, mc_samples=4)
        attach_knn_distances(
            calibration, bank, k=10, feature_key="features"
        )
        normalization = fit_score_normalization(
            calibration, prototypes, include_mahalanobis=False
        )
        epoch_metrics = {
            "open_val": evaluate(
                model, open_val_loader, device, bank, normalization, prototypes
            ),
        }
        results["epochs"][str(epoch)] = epoch_metrics
        print(f"epoch={epoch} open_val={epoch_metrics['open_val']}")

    history = json.loads((run_dir / "train_history.json").read_text(encoding="utf-8"))
    known_acc_best_epoch = max(
        history,
        key=lambda row: float(row["validation"]["known_acc"]),
    )["epoch"]
    openval_auroc_best_epoch = max(
        results["epochs"],
        key=lambda epoch: results["epochs"][epoch]["open_val"]["auroc"],
    )
    results["selection"] = {
        "known_accuracy_best_epoch": int(known_acc_best_epoch),
        "open_val_auroc_best_epoch": int(openval_auroc_best_epoch),
        "known_accuracy_best_open_val": results["epochs"][str(known_acc_best_epoch)]["open_val"],
        "open_val_auroc_best_open_val": results["epochs"][str(openval_auroc_best_epoch)]["open_val"],
        "caveat": (
            "Open-validation metrics are used only for this checkpoint-selection audit. "
            "Official test summaries are read from the separately run discover reports, "
            "not used for selecting the epoch. This exploratory test inspection means "
            "the selected epoch needs confirmation on a fresh seed."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
