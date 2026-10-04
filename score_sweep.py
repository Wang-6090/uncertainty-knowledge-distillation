from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from novel_discovery.data import build_data_bundle
from novel_discovery.models import build_model
from novel_discovery.pipeline import (
    build_loader,
    calibrate_threshold,
    compute_open_score,
    extract_outputs,
    fit_score_normalization,
    run_discovery,
)
from novel_discovery.utils import ensure_dir, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Fair score comparison using one MC extraction")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--split-path", default="./splits_cifar100_60_40.json")
    p.add_argument("--num-known", type=int, default=60)
    p.add_argument("--num-novel", type=int, default=40)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--student-ckpt", required=True)
    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    p.add_argument("--work-dir", default="./analysis/score_sweep")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def load_checkpoint(model, path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=False,
        split_path=args.split_path,
    )
    test_loader = build_loader(bundle.test, args.batch_size, False, args.num_workers)
    val_loader = build_loader(bundle.val, args.batch_size, False, args.num_workers)
    model = build_model(
        len(bundle.known_classes),
        backbone=args.backbone,
        proj_dim=128,
        dropout=0.2,
        pretrained=True,
    ).to(device)
    checkpoint = load_checkpoint(model, Path(args.student_ckpt), device)

    # Extract validation and test outputs exactly once. This prevents MC
    # dropout randomness from becoming an unwanted variable between scores.
    outputs_test = extract_outputs(model, test_loader, device, mc_samples=args.mc_samples)
    outputs_val = extract_outputs(model, val_loader, device, mc_samples=args.mc_samples)

    prototypes = checkpoint.get("prototypes")
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    gaussian_stats = checkpoint.get("gaussian_stats")
    if gaussian_stats is not None:
        gaussian_stats = {
            key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            for key, value in gaussian_stats.items()
        }
    normalization = fit_score_normalization(outputs_val, prototypes, gaussian_stats)

    modes = [
        "max_softmax",
        "energy",
        "entropy_only",
        "proto_only",
        "entropy_proto",
        "full",
        "normalized_entropy_mahalanobis",
    ]
    results = []
    for mode in modes:
        val_score, _ = compute_open_score(
            outputs_val,
            prototypes=prototypes,
            score_mode=mode,
            normalization=normalization,
            gaussian_stats=gaussian_stats,
        )
        threshold = calibrate_threshold(val_score, percentile=args.threshold_percentile)
        report, _, _, _ = run_discovery(
            outputs_test,
            threshold,
            args.num_novel,
            prototypes=prototypes,
            score_mode=mode,
            normalization=normalization,
            gaussian_stats=gaussian_stats,
            cluster_k="oracle",
        )
        results.append({"score_mode": mode, "threshold": threshold, "metrics": report})
        print(mode, json.dumps(report, ensure_ascii=False))

    output = {
        "protocol": vars(args),
        "note": "All score modes reuse one validation/test MC extraction.",
        "results": results,
    }
    out_dir = ensure_dir(args.work_dir)
    save_json(out_dir / "score_sweep.json", output)
    print(f"saved to {out_dir / 'score_sweep.json'}")


if __name__ == "__main__":
    main()
